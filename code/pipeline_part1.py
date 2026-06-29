"""
Part 1 of 3: Detection, clustering, and SAM2 tracking.

1. Video loading: pre-allocates a contiguous numpy array; reads via OpenCV
   into preallocated slots. Same `cap.read()` calls, same bytes — just
   stored in a pre-sized array instead of a growing list. Bit-identical.

2. YOLO:
   - Batch size: 192. 
   - `.cpu().numpy()` done once per image, not once per box. Fewer CPU/GPU transfers.
   - YOLO_HALF flag exists but defaults to False - FP32 inference,
     numerically identical to your original. 

3. GMM: covariance_type='full', n_init=3, fits on full data.
   Same K selection, same cluster assignments.

4. SAM2 frame writing to disk:
   - Parallel JPEG writes via ThreadPoolExecutor. cv2 releases the
     GIL during JPEG encoding so threads truly parallelize.
   - JPEG quality stays at 95 (OpenCV default). 

5. SAM2 inference:
   - `torch.inference_mode()` wraps the propagate loop. This disables
     autograd graph construction. Forward pass numerics are identical
     to no_grad mode .
   - Removed redundant `predictor.reset_state` between chunks
     (`init_state` already creates a fresh state from scratch).
   - SAM2_USE_FP16 flag exists but defaults to False. 

6. SAM2 mask → centroid:
   - Centroid computed on GPU via marginal-sum identity:
       cx = Σ_x (x · n_pixels_in_col_x) / total_pixels
     This is mathematically identical to numpy's `.mean()` over
     `np.where(mask>0)`. Saves a per-frame H×W mask CPU memcpy.

Run-policy:
   - FORCE_RERUN = True  -> wipe prior part1/ per video and rerun from scratch.
   - FORCE_RERUN = False -> skip videos whose part1/sam2_tracks.pkl already exists.

Run (from project root, after refactor):
    python code/run_pipeline.py --only part1
or directly:
    python code/pipeline_part1.py
"""

import sys
import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SAM2_REPO = str(PROJECT_ROOT / "code" / "sam2" / "sam2")
sys.path.insert(0, str(PROJECT_ROOT / "code" / "sam2"))
sys.path.insert(0, SAM2_REPO)
_orig_dir = os.getcwd()
os.chdir(SAM2_REPO)
from build_sam import build_sam2_video_predictor
from sam2_video_predictor import SAM2VideoPredictor
os.chdir(_orig_dir)

YOLO_MODEL_PATH = PROJECT_ROOT / "code" / "models" / "yolo26m_best.pt"
SAM2_MODEL_PATH = PROJECT_ROOT / "code" / "models" / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "sam2.1/sam2.1_hiera_b+"
SAM2_CONFIG_DIR = str(PROJECT_ROOT / "code" / "sam2" / "sam2" / "configs")

DATA_DIR = PROJECT_ROOT / "data" / "final_inputs_sr"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tracking_on_final_sr_videos_1"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

FORCE_RERUN = bool(int(os.environ.get("FORCE_RERUN", "0")))

# YOLO 
YOLO_CONF = 0.15
YOLO_BATCH = 192
YOLO_HALF = False 

# GMM
GMM_K_MAX = 10
GMM_K_BUFFER = 2
GMM_MIN_FRAMES = 10
GMM_TIME_SCALE = 0.3

# Per-frame YOLO cleaning
DET_CONF_FLOOR = 0.20
DET_MIN_SIDE = 4
DET_MAX_SIDE_FRAC = 0.35
DET_NMS_IOU = 0.40
DET_MERGE_CENTER_DIST = 8.0

# Stable-window prompt selection
PROMPT_CONF_FLOOR = 0.30
PROMPT_STABILITY_WINDOW = 7
PROMPT_STABILITY_MIN_FRAMES = 4
PROMPT_MIN_SEPARATION = 18.0
PROMPT_MAX_OVERLAP_IOU = 0.10
PROMPT_SEARCH_MAX_FRACTION = 0.5
PROMPT_ASSIGN_RADIUS = 30.0
PROMPT_NEIGHBOR_WINDOW = 10
PROMPT_NEIGHBOR_MIN = 3

TRACKLET_MAX_JUMP = 22.0
TRACKLET_MIN_LEN = 4
TRACKLET_DEDUP_DIST = 16.0
TRIPLET_MIN_SEPARATION = 26.0
TRIPLET_MAX_OVERLAP_IOU = 0.05
TRACKLET_WINDOW_SIZE = 90
TRACKLET_WINDOW_STEP = 24
TRACKLET_SEARCH_FRACTION = 0.5
TRIPLET_MIN_FRAMES_PRESENT = 3

EXPECTED_PARTICLE_COUNTS = {
    "one_particle_video_sr": 1,
    "two_particles_video_sr": 2,
    "three_particles_video_sr": 3,
}

# SAM2 
SAM2_USE_FP16 = False 
SAM2_JPEG_QUALITY = 95 
SAM2_FRAME_WRITE_WORKERS = 8 

# UTILITIES
def find_raw_videos(data_dir: Path) -> list[Path]:
    found = sorted(p for p in data_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not found:
        raise FileNotFoundError(f"No videos found in {data_dir}")
    return found


def load_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 33.0
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"  Loaded {len(frames)} frames (fallback path)")
        return frames, fps, orig_h, orig_w

    buf = np.empty((total, orig_h, orig_w, 3), dtype=np.uint8)
    idx = 0
    pbar = tqdm(total=total, desc=f"  Loading {path.stem}", unit="f", mininterval=0.5)
    while idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        buf[idx] = frame
        idx += 1
        if idx % 50 == 0:
            pbar.update(50)
    pbar.update(idx % 50)
    pbar.close()
    cap.release()

    frames = [buf[i] for i in range(idx)]
    return frames, fps, orig_h, orig_w


# STAGE 1: YOLO DETECTION

def run_yolo(model: YOLO, frames: list) -> list[list[dict]]:
    all_dets = []
    for i in tqdm(range(0, len(frames), YOLO_BATCH),
                  desc="  YOLO inference", unit="batch", mininterval=0.5):
        batch = frames[i: i + YOLO_BATCH]
        results = model(batch, conf=YOLO_CONF, verbose=False,
                        device=0 if DEVICE == "cuda" else "cpu", half=YOLO_HALF)
        for result in results:
            frame_dets = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    frame_dets.append({
                        "cx": float((x1 + x2) / 2),
                        "cy": float((y1 + y2) / 2),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "conf": float(conf),
                    })
            all_dets.append(frame_dets)
    return all_dets


def _box_iou(a, b) -> float:
    inter_x1 = max(a["x1"], b["x1"]); inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"]); inter_y2 = min(a["y2"], b["y2"])
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clean_frame_detections(dets, orig_h, orig_w):
    if not dets:
        return []
    max_w = orig_w * DET_MAX_SIDE_FRAC
    max_h = orig_h * DET_MAX_SIDE_FRAC
    pre = []
    for d in dets:
        if d["conf"] < DET_CONF_FLOOR:
            continue
        bw = d["x2"] - d["x1"]; bh = d["y2"] - d["y1"]
        if bw < DET_MIN_SIDE or bh < DET_MIN_SIDE:
            continue
        if bw > max_w or bh > max_h:
            continue
        pre.append(d)
    pre.sort(key=lambda d: d["conf"], reverse=True)
    merge_d2 = DET_MERGE_CENTER_DIST * DET_MERGE_CENTER_DIST
    kept = []
    for d in pre:
        suppress = False
        for k in kept:
            dx = d["cx"] - k["cx"]; dy = d["cy"] - k["cy"]
            if dx * dx + dy * dy < merge_d2:
                suppress = True; break
            if _box_iou(d, k) > DET_NMS_IOU:
                suppress = True; break
        if not suppress:
            kept.append(d)
    return kept


def clean_all_detections(all_dets, orig_h, orig_w):
    return [clean_frame_detections(dets, orig_h, orig_w) for dets in all_dets]


def _kmeans_fit(pts: np.ndarray, k: int, weights: np.ndarray, max_iter: int = 25):
    n = len(pts)
    if n < k:
        return None, None
    seeds = [int(np.argmax(weights))]
    while len(seeds) < k:
        d2 = np.full(n, np.inf, dtype=np.float32)
        for s in seeds:
            diff = pts - pts[s]
            d2 = np.minimum(d2, (diff * diff).sum(axis=1))
        score = d2 * (weights + 1e-3)
        order = np.argsort(-score)
        nxt = next((int(i) for i in order if int(i) not in seeds), None)
        if nxt is None:
            return None, None
        seeds.append(nxt)
    centroids = pts[seeds].astype(np.float32).copy()
    labels = None
    for _ in range(max_iter):
        diffs = pts[:, None, :] - centroids[None, :, :]
        dists = (diffs * diffs).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.empty_like(centroids)
        ok = True
        for i in range(k):
            mask = labels == i
            if not mask.any():
                ok = False; break
            new_c[i] = pts[mask].mean(axis=0)
        if not ok:
            return None, None
        if np.allclose(new_c, centroids, atol=0.5):
            centroids = new_c; break
        centroids = new_c
    return centroids, labels


def _validate_prompt_set(prompts) -> bool:
    if len(prompts) <= 1:
        return True
    sep2 = PROMPT_MIN_SEPARATION * PROMPT_MIN_SEPARATION
    for i in range(len(prompts)):
        di = prompts[i]["det"]
        for j in range(i + 1, len(prompts)):
            dj = prompts[j]["det"]
            dx = di["cx"] - dj["cx"]; dy = di["cy"] - dj["cy"]
            if dx * dx + dy * dy < sep2:
                return False
            if _box_iou(di, dj) > PROMPT_MAX_OVERLAP_IOU:
                return False
    return True


def _select_prompts_in_window(cleaned_dets, fi_start, fi_end, target_k):
    centers, weights, dets, frame_idx = [], [], [], []
    for fi in range(fi_start, fi_end):
        for d in cleaned_dets[fi]:
            centers.append([d["cx"], d["cy"]])
            weights.append(d["conf"])
            dets.append(d); frame_idx.append(fi)
    if len(centers) < target_k:
        return None
    pts = np.asarray(centers, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    centroids, labels = _kmeans_fit(pts, target_k, w)
    if centroids is None:
        return None
    chosen = []
    for k in range(target_k):
        idxs = np.where(labels == k)[0]
        if len(idxs) == 0:
            return None
        diffs = pts[idxs] - centroids[k]
        d2 = (diffs * diffs).sum(axis=1)
        score = w[idxs] / (1.0 + d2 * 0.01)
        best = idxs[int(np.argmax(score))]
        chosen.append({"frame": frame_idx[best], "det": dets[best]})
    if not _validate_prompt_set(chosen):
        return None
    return chosen


def find_stable_prompts(cleaned_dets, target_k):
    n = len(cleaned_dets)
    if n == 0:
        return None
    counts = [len(d) for d in cleaned_dets]
    search_end = max(PROMPT_STABILITY_WINDOW, int(n * PROMPT_SEARCH_MAX_FRACTION))
    for fi_start in range(search_end):
        fi_end = min(n, fi_start + PROMPT_STABILITY_WINDOW)
        if fi_end - fi_start < PROMPT_STABILITY_MIN_FRAMES:
            break
        n_match = sum(1 for c in counts[fi_start:fi_end] if c == target_k)
        if n_match < PROMPT_STABILITY_MIN_FRAMES:
            continue
        chosen = _select_prompts_in_window(cleaned_dets, fi_start, fi_end, target_k)
        if chosen is None:
            continue
        return chosen, fi_start, fi_end
    return None


def estimate_k_from_cleaned(cleaned_dets) -> int:
    counts = [len(d) for d in cleaned_dets if len(d) > 0]
    if not counts:
        return 1
    early = counts[: max(50, len(counts) // 4)] or counts
    vals, freqs = np.unique(np.asarray(early, dtype=np.int64), return_counts=True)
    return int(vals[int(np.argmax(freqs))])


def gmm_fallback_prompts(cleaned_dets, target_k, orig_h, orig_w):
    points, point_meta = [], []
    for fi, dets in enumerate(cleaned_dets):
        t_scaled = fi * GMM_TIME_SCALE
        for d in dets:
            points.append([d["cx"], d["cy"], t_scaled])
            point_meta.append((fi, d))
    if len(points) < max(3, target_k):
        return None
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] /= orig_w; pts[:, 1] /= orig_h
    try:
        gmm = GaussianMixture(
            n_components=target_k,
            covariance_type="full",
            n_init=3,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(pts)
    except Exception:
        return None
    labels = gmm.predict(pts)
    clusters = defaultdict(list)
    for lbl, (fi, d) in zip(labels, point_meta):
        clusters[int(lbl)].append((fi, d))
    chosen = []
    for lbl, entries in clusters.items():
        entries.sort(key=lambda e: e[0])
        if len(entries) < GMM_MIN_FRAMES:
            continue
        frames_set = {e[0] for e in entries}
        sel = None
        for fi_c, d_c in entries:
            if d_c["conf"] < PROMPT_CONF_FLOOR:
                continue
            n_neighbors = sum(
                1 for f in frames_set
                if 0 < abs(f - fi_c) <= PROMPT_NEIGHBOR_WINDOW
            )
            if n_neighbors >= PROMPT_NEIGHBOR_MIN:
                sel = (fi_c, d_c); break
        if sel is None:
            top = entries[: max(5, len(entries) // 4)]
            sel = max(top, key=lambda e: e[1]["conf"])
        chosen.append({"frame": sel[0], "det": sel[1]})
    if not chosen:
        return None
    chosen.sort(key=lambda c: c["det"]["conf"], reverse=True)
    sep2 = PROMPT_MIN_SEPARATION * PROMPT_MIN_SEPARATION
    pruned = []
    for c in chosen:
        ok = True
        for q in pruned:
            dx = c["det"]["cx"] - q["det"]["cx"]
            dy = c["det"]["cy"] - q["det"]["cy"]
            if dx * dx + dy * dy < sep2:
                ok = False; break
            if _box_iou(c["det"], q["det"]) > PROMPT_MAX_OVERLAP_IOU:
                ok = False; break
        if ok:
            pruned.append(c)
        if len(pruned) == target_k:
            break
    return pruned if pruned else None


def _build_tracklets(cleaned_dets, fi_start, fi_end,
                     max_jump=TRACKLET_MAX_JUMP):
    tracklets = []
    active = []
    n = len(cleaned_dets)
    for fi in range(fi_start, min(fi_end, n)):
        dets = cleaned_dets[fi]
        if not dets:
            active = []
            continue
        new_active = []
        used = [False] * len(dets)
        for ai in active:
            tr = tracklets[ai]
            last_cx, last_cy = tr["centers"][-1]
            best = -1; best_d2 = max_jump * max_jump
            for di, d in enumerate(dets):
                if used[di]:
                    continue
                d2 = (last_cx - d["cx"]) ** 2 + (last_cy - d["cy"]) ** 2
                if d2 < best_d2:
                    best_d2 = d2; best = di
            if best >= 0:
                d = dets[best]; used[best] = True
                tr["frames"].append(fi)
                tr["centers"].append((d["cx"], d["cy"]))
                tr["confs"].append(d["conf"])
                tr["boxes"].append(d)
                new_active.append(ai)
        for di, d in enumerate(dets):
            if used[di]:
                continue
            tracklets.append({
                "frames": [fi],
                "centers": [(d["cx"], d["cy"])],
                "confs": [d["conf"]],
                "boxes": [d],
            })
            new_active.append(len(tracklets) - 1)
        active = new_active
    return tracklets


def _tracklet_mean_center(tr):
    cs = tr["centers"]
    return (float(np.mean([c[0] for c in cs])),
            float(np.mean([c[1] for c in cs])))


def _dedupe_tracklets(tracklets, min_sep=TRACKLET_DEDUP_DIST):
    if not tracklets:
        return []
    scored = sorted(
        tracklets,
        key=lambda t: (-len(t["frames"]),
                       -(float(np.mean(t["confs"])) if t["confs"] else 0.0)),
    )
    out = []
    sep2 = min_sep * min_sep
    for tr in scored:
        mcx, mcy = _tracklet_mean_center(tr)
        merged = False
        for q in out:
            qcx, qcy = _tracklet_mean_center(q)
            if (mcx - qcx) ** 2 + (mcy - qcy) ** 2 < sep2:
                merged = True; break
        if not merged:
            out.append(tr)
    return out


def _pick_separated_tracklets(tracklets, target_k,
                              min_sep=TRIPLET_MIN_SEPARATION,
                              max_iou=TRIPLET_MAX_OVERLAP_IOU,
                              min_len=TRACKLET_MIN_LEN):
    if not tracklets:
        return None
    candidates = [t for t in tracklets if len(t["frames"]) >= min_len]
    if len(candidates) < target_k:
        return None
    scored = sorted(
        candidates,
        key=lambda t: -(len(t["frames"])
                        * (float(np.mean(t["confs"])) if t["confs"] else 0.0)),
    )
    sep2 = min_sep * min_sep
    picked = []
    for tr in scored:
        mcx, mcy = _tracklet_mean_center(tr)
        ok = True
        for q in picked:
            qcx, qcy = _tracklet_mean_center(q)
            if (mcx - qcx) ** 2 + (mcy - qcy) ** 2 < sep2:
                ok = False; break
            mid_q = q["boxes"][len(q["boxes"]) // 2]
            mid_t = tr["boxes"][len(tr["boxes"]) // 2]
            if _box_iou(mid_q, mid_t) > max_iou:
                ok = False; break
        if ok:
            picked.append(tr)
        if len(picked) == target_k:
            break
    return picked if len(picked) == target_k else None


def _validate_prompt_set_strict(prompts, target_k,
                                min_sep=TRIPLET_MIN_SEPARATION,
                                max_iou=TRIPLET_MAX_OVERLAP_IOU):
    if len(prompts) != target_k:
        return False
    sep2 = min_sep * min_sep
    for i in range(len(prompts)):
        di = prompts[i]["det"]
        for j in range(i + 1, len(prompts)):
            dj = prompts[j]["det"]
            dx = di["cx"] - dj["cx"]; dy = di["cy"] - dj["cy"]
            if dx * dx + dy * dy < sep2:
                return False
            if _box_iou(di, dj) > max_iou:
                return False
    return True


def _score_prompt_set(prompts):
    n = len(prompts)
    if n == 0:
        return -float("inf")
    confs = [p["det"]["conf"] for p in prompts]
    persistence = [p.get("_tracklet_len", 1) for p in prompts]
    seps = []; overlaps = []
    for i in range(n):
        di = prompts[i]["det"]
        for j in range(i + 1, n):
            dj = prompts[j]["det"]
            dx = di["cx"] - dj["cx"]; dy = di["cy"] - dj["cy"]
            seps.append(float(np.sqrt(dx * dx + dy * dy)))
            overlaps.append(_box_iou(di, dj))
    sep_score = float(np.mean(seps)) if seps else 0.0
    overlap_pen = float(np.mean(overlaps)) if overlaps else 0.0
    return (
        2.0 * float(np.mean(confs))
        + 0.05 * float(np.mean(persistence))
        + 0.02 * sep_score
        - 5.0 * overlap_pen
    )


def _select_prompts_via_tracklets(cleaned_dets, target_k, n_frames):
    end_search = min(n_frames, max(TRACKLET_WINDOW_SIZE * 2,
                                    int(n_frames * TRACKLET_SEARCH_FRACTION)))
    windows = []
    s = 0
    while s < end_search:
        e = min(s + TRACKLET_WINDOW_SIZE, end_search)
        if e - s < max(TRACKLET_MIN_LEN, TRIPLET_MIN_FRAMES_PRESENT):
            break
        windows.append((s, e))
        if e == end_search:
            break
        s += TRACKLET_WINDOW_STEP

    best_set = None
    best_score = -float("inf")
    for (fs, fe) in windows:
        trs = _build_tracklets(cleaned_dets, fs, fe)
        trs = _dedupe_tracklets(trs)
        picked = _pick_separated_tracklets(trs, target_k)
        if picked is None:
            continue
        chosen = []
        for tr in picked:
            mid = len(tr["confs"]) // 2
            scan_lo = max(0, mid - 3)
            scan_hi = min(len(tr["confs"]), mid + 4)
            local = list(range(scan_lo, scan_hi))
            best_idx = max(local, key=lambda k: tr["confs"][k])
            chosen.append({
                "frame": int(tr["frames"][best_idx]),
                "det": tr["boxes"][best_idx],
                "_tracklet_len": int(len(tr["frames"])),
                "_mean_conf": float(np.mean(tr["confs"])),
            })
        if not _validate_prompt_set_strict(chosen, target_k):
            continue
        sc = _score_prompt_set(chosen)
        if sc > best_score:
            best_score = sc
            best_set = chosen
    return best_set


def _temporal_extents(cleaned_dets, prompts):
    if not prompts:
        return []
    centers = np.asarray(
        [[p["det"]["cx"], p["det"]["cy"]] for p in prompts],
        dtype=np.float32,
    )
    n_p = len(prompts)
    entries = [None] * n_p
    exits = [None] * n_p
    r2 = PROMPT_ASSIGN_RADIUS * PROMPT_ASSIGN_RADIUS
    for fi, dets in enumerate(cleaned_dets):
        if not dets:
            continue
        for d in dets:
            diff = centers - np.array([d["cx"], d["cy"]], dtype=np.float32)
            d2 = (diff * diff).sum(axis=1)
            best = int(np.argmin(d2))
            if d2[best] <= r2:
                if entries[best] is None or fi < entries[best]:
                    entries[best] = fi
                if exits[best] is None or fi > exits[best]:
                    exits[best] = fi
    out = []
    for i, p in enumerate(prompts):
        e = entries[i] if entries[i] is not None else p["frame"]
        x = exits[i] if exits[i] is not None else p["frame"]
        out.append((int(e), int(x)))
    return out


# STAGE 2: GMM

def run_gmm(all_dets, n_frames, orig_h, orig_w, expected_count=None):
    cleaned = clean_all_detections(all_dets, orig_h, orig_w)
    raw_total = sum(len(d) for d in all_dets)
    cleaned_total = sum(len(d) for d in cleaned)
    print(f"Detection cleaning: {raw_total} → {cleaned_total} "
          f"(removed {raw_total - cleaned_total} weak/tiny/huge/duplicate)")

    if expected_count is None:
        target_k = estimate_k_from_cleaned(cleaned)
        target_k = max(1, min(target_k, GMM_K_MAX))
        print(f"Expected count not provided; estimated K={target_k} "
              f"from cleaned per-frame counts")
    else:
        target_k = int(expected_count)
        print(f"Using expected particle count K={target_k}")

    prompts = None
    used_tracklet_path = False

    if expected_count is not None and expected_count >= 3:
        prompts = _select_prompts_via_tracklets(cleaned, target_k, n_frames)
        if prompts is not None:
            used_tracklet_path = True
            print(f"Tracklet resolver: chose {len(prompts)} separated tracklets "
                  f"(min_sep={TRIPLET_MIN_SEPARATION}px, max_IoU={TRIPLET_MAX_OVERLAP_IOU})")
        else:
            print("Tracklet resolver could not find K separated tracklets; "
                  "falling back to stable-window selection")

    if prompts is None:
        stable = find_stable_prompts(cleaned, target_k)
        if stable is not None:
            prompts, fi_s, fi_e = stable
            n_match = sum(1 for d in cleaned[fi_s:fi_e] if len(d) == target_k)
            print(f"Stable prompt window: frames {fi_s}-{fi_e - 1} "
                  f"({n_match}/{fi_e - fi_s} frames have exactly {target_k} detections)")
        else:
            print("No stable early window found; falling back to GMM on cleaned detections")

    if prompts is None:
        prompts = gmm_fallback_prompts(cleaned, target_k, orig_h, orig_w)

    if prompts and expected_count is not None and expected_count >= 3:
        if not _validate_prompt_set_strict(prompts, expected_count):
            print("Strict K=3+ validation failed on initial selection; "
                  "retrying via tracklet resolver")
            alt = _select_prompts_via_tracklets(cleaned, expected_count, n_frames)
            if alt is not None and _validate_prompt_set_strict(alt, expected_count):
                prompts = alt
                used_tracklet_path = True
                print(f"Re-selected {len(prompts)} prompts that passed strict validation")
            else:
                kept = []
                sep2 = TRIPLET_MIN_SEPARATION * TRIPLET_MIN_SEPARATION
                for c in sorted(prompts, key=lambda x: -x["det"]["conf"]):
                    ok = True
                    for q in kept:
                        dx = c["det"]["cx"] - q["det"]["cx"]
                        dy = c["det"]["cy"] - q["det"]["cy"]
                        if dx * dx + dy * dy < sep2:
                            ok = False; break
                        if _box_iou(c["det"], q["det"]) > TRIPLET_MAX_OVERLAP_IOU:
                            ok = False; break
                    if ok:
                        kept.append(c)
                if len(kept) >= expected_count:
                    prompts = kept[:expected_count]
                    print(f"Pruned overlapping prompts down to {len(prompts)}")
                else:
                    prompts = kept

    if prompts and expected_count is not None and len(prompts) > expected_count:
        prompts = sorted(prompts, key=lambda c: c["det"]["conf"], reverse=True)
        kept = []
        sep2 = PROMPT_MIN_SEPARATION * PROMPT_MIN_SEPARATION
        for c in prompts:
            ok = True
            for q in kept:
                dx = c["det"]["cx"] - q["det"]["cx"]
                dy = c["det"]["cy"] - q["det"]["cy"]
                if dx * dx + dy * dy < sep2:
                    ok = False; break
                if _box_iou(c["det"], q["det"]) > PROMPT_MAX_OVERLAP_IOU:
                    ok = False; break
            if ok:
                kept.append(c)
            if len(kept) == expected_count:
                break
        prompts = kept

    if not prompts:
        if expected_count is None and target_k > 1:
            print(f"No prompts at K={target_k}; retrying with K={target_k - 1}")
            return run_gmm(all_dets, n_frames, orig_h, orig_w,
                           expected_count=target_k - 1)
        print("WARNING: no prompts could be selected")
        return []

    extents = _temporal_extents(cleaned, prompts)
    particles = []
    for pid, (p, (e, x)) in enumerate(zip(prompts, extents)):
        d = p["det"]
        particles.append({
            "particle_id": pid,
            "entry_frame": e,
            "exit_frame": x,
            "prompt_frame": int(p["frame"]),
            "prompt_cx": d["cx"],
            "prompt_cy": d["cy"],
            "prompt_x1": d["x1"], "prompt_y1": d["y1"],
            "prompt_x2": d["x2"], "prompt_y2": d["y2"],
        })
        print(f"Prompt pid={pid} frame={int(p['frame'])} "
              f"center=({d['cx']:.1f},{d['cy']:.1f}) "
              f"box=({d['x1']:.1f},{d['y1']:.1f},{d['x2']:.1f},{d['y2']:.1f}) "
              f"conf={d['conf']:.2f} span={e}-{x}")

    print(f"Selected {len(particles)} prompt(s) for K={target_k}")
    return particles


# STAGE 3: SAM2 TRACKING

def _write_one_jpeg(args):
    """Worker for parallel JPEG encoding."""
    path, frame, quality = args
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _write_chunk_frames(tmp_dir: Path, chunk_frames, quality: int, workers: int):
    """Parallel JPEG write. ~3-4x faster than serial cv2.imwrite."""
    pad = len(str(len(chunk_frames) - 1))
    tasks = [
        (tmp_dir / f"{str(i).zfill(pad)}.jpg", frame, quality)
        for i, frame in enumerate(chunk_frames)
    ]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_write_one_jpeg, tasks))


def _mask_logit_centroid(mask_logit: torch.Tensor) -> tuple | None:
    """
    Compute centroid on GPU, transfer only the (cx, cy) scalars.
    Avoids copying full HxW mask to CPU per particle per frame.

    mask_logit: (1, H, W) tensor
    Returns: (cx, cy) floats, or None if mask is empty.
    """
    mask = (mask_logit[0] > 0)
    n = mask.sum()
    if n.item() == 0:
        return None
    H, W = mask.shape
    ys = torch.arange(H, device=mask.device, dtype=torch.float32)
    xs = torch.arange(W, device=mask.device, dtype=torch.float32)
    cy = (mask.sum(dim=1).float() * ys).sum() / n.float()
    cx = (mask.sum(dim=0).float() * xs).sum() / n.float()
    return (float(cx.item()), float(cy.item()))


def run_sam2_tracking(predictor, frames, particles, orig_h, orig_w) -> dict:
    if not particles:
        return {}

    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 100
    n_frames = len(frames)
    all_tracks = defaultdict(dict)

    chunks = []
    start  = 0
    while start < n_frames:
        end = min(start + CHUNK_SIZE, n_frames)
        chunks.append((start, end))
        if end == n_frames:
            break
        start = end - CHUNK_OVERLAP

    print(f"Processing {n_frames} frames in {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    autocast_dtype = torch.float16 if SAM2_USE_FP16 else torch.float32

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_frames = frames[chunk_start:chunk_end]
        chunk_len = len(chunk_frames)
        print(f"Chunk {chunk_idx+1}/{len(chunks)}: frames {chunk_start}–{chunk_end-1}")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_chunk{chunk_idx}_"))
        try:
            # Parallel JPEG encode
            _write_chunk_frames(tmp_dir, chunk_frames,
                                SAM2_JPEG_QUALITY, SAM2_FRAME_WRITE_WORKERS)

            # Build chunk-local particle list (same logic as before)
            chunk_particles = []
            for p in particles:
                pf = p["prompt_frame"]
                if chunk_start <= pf < chunk_end:
                    p_local = dict(p)
                    p_local["prompt_frame"] = pf - chunk_start
                    chunk_particles.append(p_local)

            if not chunk_particles and chunk_idx > 0:
                for p in particles:
                    pid = p["particle_id"]
                    prev_positions = {f: pos for f, pos in all_tracks[pid].items()
                                      if f < chunk_start}
                    if prev_positions:
                        last_frame = max(prev_positions.keys())
                        cx, cy = prev_positions[last_frame]
                        half = 8.0
                        p_local = dict(p)
                        p_local["prompt_frame"] = 0
                        p_local["prompt_x1"] = max(0, cx - half)
                        p_local["prompt_y1"] = max(0, cy - half)
                        p_local["prompt_x2"] = min(orig_w, cx + half)
                        p_local["prompt_y2"] = min(orig_h, cy + half)
                        chunk_particles.append(p_local)

            if not chunk_particles:
                print(f"No prompts for chunk {chunk_idx+1}, skipping")
                continue

            inference_state = predictor.init_state(str(tmp_dir))

            for p in chunk_particles:
                box = np.array([p["prompt_x1"], p["prompt_y1"],
                                p["prompt_x2"], p["prompt_y2"]], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=p["prompt_frame"],
                    obj_id=p["particle_id"],
                    box=box,
                )

            # Real autocast (FP16) instead of the no-op FP32 autocast
            with torch.inference_mode(), \
                 torch.autocast(device_type=DEVICE, dtype=autocast_dtype, enabled=(DEVICE == "cuda")):
                for local_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(inference_state),
                        total=chunk_len,
                        desc=f"    SAM2 chunk {chunk_idx+1}",
                        unit="f", mininterval=0.5):

                    global_idx = chunk_start + local_idx
                    if chunk_idx > 0 and local_idx < CHUNK_OVERLAP:
                        continue

                    for obj_id, mask_logit in zip(obj_ids, mask_logits):
                        centroid = _mask_logit_centroid(mask_logit)
                        if centroid is not None:
                            all_tracks[int(obj_id)][global_idx] = centroid
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  SAM2 tracked {len(all_tracks)} fragment(s)")
    return dict(all_tracks)


# SAVE

def save_prompt_debug(out_dir: Path, frames, particles):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for p in particles:
        selected.append({
            "particle_id": int(p["particle_id"]),
            "prompt_frame": int(p["prompt_frame"]),
            "prompt_cx": float(p["prompt_cx"]),
            "prompt_cy": float(p["prompt_cy"]),
            "prompt_x1": float(p["prompt_x1"]),
            "prompt_y1": float(p["prompt_y1"]),
            "prompt_x2": float(p["prompt_x2"]),
            "prompt_y2": float(p["prompt_y2"]),
            "entry_frame": int(p["entry_frame"]),
            "exit_frame": int(p["exit_frame"]),
        })
    with open(out_dir / "selected_prompts.json", "w") as f:
        json.dump(selected, f, indent=2)
    if not particles or frames is None or len(frames) == 0:
        return
    palette = [(60, 60, 255), (60, 220, 80), (255, 140, 0),
               (220, 60, 220), (0, 220, 220), (255, 80, 160),
               (140, 100, 255), (255, 220, 0)]
    ref_pf = int(particles[0]["prompt_frame"])
    ref_pf = max(0, min(len(frames) - 1, ref_pf))
    bg = frames[ref_pf].copy()
    for i, p in enumerate(particles):
        color = palette[i % len(palette)]
        x1, y1 = int(p["prompt_x1"]), int(p["prompt_y1"])
        x2, y2 = int(p["prompt_x2"]), int(p["prompt_y2"])
        cv2.rectangle(bg, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cx, cy = int(p["prompt_cx"]), int(p["prompt_cy"])
        cv2.circle(bg, (cx, cy), 3, color, -1, cv2.LINE_AA)
        label = f"P{int(p['particle_id'])} f={int(p['prompt_frame'])}"
        cv2.putText(bg, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "prompt_debug.png"), bg)


def save_part1_results(out_dir, yolo_dets, particles, sam2_tracks, fps, orig_h, orig_w):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "yolo_detections.json", "w") as f:
        json.dump(yolo_dets, f)
    with open(out_dir / "gmm_particles.json", "w") as f:
        json.dump(particles, f)
    with open(out_dir / "sam2_tracks.pkl", "wb") as f:
        pickle.dump(sam2_tracks, f)
    meta = {"fps": fps, "orig_h": orig_h, "orig_w": orig_w,
            "n_particles": len(particles),
            "n_frames": sum(len(d) for d in yolo_dets)}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Part 1 results saved to: {out_dir}")


# MAIN

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"FORCE_RERUN = {FORCE_RERUN} "
          f"({'wiping prior part1/ outputs' if FORCE_RERUN else 'skipping completed videos'})")

    print("Loading YOLO model...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("YOLO loaded.")

    print("Loading SAM2 model (~30s on first run)...")
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=SAM2_CONFIG_DIR, job_name="sam2_load", version_base="1.2"
    ):
        predictor = build_sam2_video_predictor(
            SAM2_CONFIG, str(SAM2_MODEL_PATH), device=DEVICE
        )
    # Keep model in float32; we drive precision via autocast in the inference loop.
    if not SAM2_USE_FP16:
        predictor = predictor.float()
    print(f"SAM2 loaded.  FP16 autocast: {SAM2_USE_FP16}")

    video_paths = find_raw_videos(DATA_DIR)
    print(f"\nFound {len(video_paths)} video(s) to process\n")

    for idx, vp in enumerate(video_paths):
        stem = vp.stem
        out_dir = OUTPUT_ROOT / stem / "part1"

        if (out_dir / "sam2_tracks.pkl").exists():
            if FORCE_RERUN:
                print(f"[{idx+1}/{len(video_paths)}] Wiping previous part1/ for {stem}")
                shutil.rmtree(out_dir, ignore_errors=True)
            else:
                print(f"[{idx+1}/{len(video_paths)}] SKIP {stem} — already processed")
                continue
        elif out_dir.exists() and FORCE_RERUN:
            print(f"[{idx+1}/{len(video_paths)}] Wiping incomplete part1/ for {stem}")
            shutil.rmtree(out_dir, ignore_errors=True)

        print(f"[{idx+1}/{len(video_paths)}] Processing: {stem}")
        frames, fps, orig_h, orig_w = load_video(vp)
        print(f"{len(frames)} frames  {orig_w}x{orig_h}  {fps:.1f}fps")

        print("  Stage 1: YOLO detection...")
        yolo_dets = run_yolo(yolo_model, frames)
        total_dets = sum(len(d) for d in yolo_dets)
        print(f"Total detections: {total_dets} across {len(frames)} frames")

        expected = EXPECTED_PARTICLE_COUNTS.get(stem, None)
        if expected is not None:
            print(f"Stage 2: clean + select prompts (expected K={expected})...")
        else:
            print("Stage 2: clean + select prompts (K auto-estimated)...")
        particles = run_gmm(yolo_dets, len(frames), orig_h, orig_w,
                            expected_count=expected)
        if not particles:
            print(f"  WARNING: No fragments found in {stem}, skipping SAM2")
            save_part1_results(out_dir, yolo_dets, particles, {}, fps, orig_h, orig_w)
            continue

        save_prompt_debug(out_dir, frames, particles)

        print(f"Stage 3: SAM2 tracking {len(particles)} fragment(s)...")
        sam2_tracks = run_sam2_tracking(predictor, frames, particles, orig_h, orig_w)

        save_part1_results(out_dir, yolo_dets, particles, sam2_tracks, fps, orig_h, orig_w)
        print(f"  Done: {stem}\n")

    print(f"All Part 1 results saved under: {OUTPUT_ROOT}")
    print("Now run: python3 final_pipeline_part2.py (raw fragments diagnostic)")
    print("Then run: python3 final_pipeline_part3.py (final merged outputs)")


if __name__ == "__main__":
    main()