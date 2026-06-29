from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Callable
import shutil
import math

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Helpers 

def _find_predicted_video(output_run_dir: Path) -> Path:
    if not output_run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {output_run_dir}")
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    candidates = [p for p in output_run_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in video_exts]
    if not candidates:
        raise FileNotFoundError(f"No predicted video found in {output_run_dir}")
    mp4s = [p for p in candidates if p.suffix.lower() == ".mp4"]
    return mp4s[0] if mp4s else candidates[0]


def _track_colour(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour per track ID via golden-angle hue spacing."""
    hue = int((track_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# Physical feature extraction from a YOLO bounding box 

def _safe_log_hu(hu: np.ndarray) -> np.ndarray:
    hu = np.asarray(hu, dtype=float).reshape(-1)
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


def _intensity_com_offset(patch: np.ndarray) -> tuple[float, float]:
    h, w = patch.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0
    pf    = patch.astype(float)
    total = pf.sum()
    if total < 1e-8:
        return 0.0, 0.0
    ys, xs = np.mgrid[0:h, 0:w]
    cx_w = (xs * pf).sum() / total
    cy_w = (ys * pf).sum() / total
    return float((cx_w - w / 2) / max(w, 1)), float((cy_w - h / 2) / max(h, 1))


def extract_patch_features(
    frame_gray: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> dict:

    H, W = frame_gray.shape[:2]
    x1 = max(0, min(int(bbox[0]), W - 1))
    y1 = max(0, min(int(bbox[1]), H - 1))
    x2 = max(0, min(int(bbox[2]), W))
    y2 = max(0, min(int(bbox[3]), H))

    _empty = {
        "area": 0.0, "circularity": 0.0, "orientation": 0.0,
        "hu": np.zeros(7, dtype=float), "mean_intensity": 0.0,
        "com_dx": 0.0, "com_dy": 0.0,
    }

    if x2 <= x1 or y2 <= y1:
        return _empty

    patch = frame_gray[y1:y2, x1:x2]
    mean_intensity = float(np.mean(patch))
    com_dx, com_dy = _intensity_com_offset(patch)

    blur = cv2.GaussianBlur(patch, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return {**_empty,
                "area": float((x2-x1) * (y2-y1)),
                "mean_intensity": mean_intensity,
                "com_dx": com_dx, "com_dy": com_dy}

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circ = (4.0 * math.pi * area / perim ** 2) if perim > 1e-8 else 0.0
    moments = cv2.moments(cnt)
    hu = _safe_log_hu(cv2.HuMoments(moments).flatten())
    orient = 0.0
    if len(cnt) >= 5:
        _, _, orient = cv2.fitEllipse(cnt)

    return {
        "area": area,
        "circularity": float(circ),
        "orientation": float(orient),
        "hu": hu,
        "mean_intensity": mean_intensity,
        "com_dx": com_dx,
        "com_dy": com_dy,
    }


# Neighbour geometry 

def _add_neighbor_features(detections: list[dict], k: int = 3) -> None:
    n = len(detections)
    if n == 0:
        return
    pts = np.array([[d["x"], d["y"]] for d in detections], dtype=float)
    for i, det in enumerate(detections):
        dx = pts[:, 0] - det["x"]
        dy = pts[:, 1] - det["y"]
        dist = np.sqrt(dx * dx + dy * dy)
        order = [j for j in np.argsort(dist) if j != i][:k]
        dists, angles = [], []
        for j in order:
            dists.append(float(dist[j]))
            angles.append(float(math.degrees(math.atan2(dy[j], dx[j])) % 360))
        while len(dists) < k:
            dists.append(1e6); angles.append(0.0)
        det["neighbor_dists"]  = np.array(dists,  dtype=float)
        det["neighbor_angles"] = np.array(angles, dtype=float)


# Track dataclass 

class YOLOTrack:

    def __init__(self, track_id: int, det: dict):
        self.track_id = track_id
        self.x = float(det["x"])
        self.y = float(det["y"])
        self.vx = 0.0
        self.vy = 0.0

        self.area:           float      = float(det.get("area", 0))
        self.circularity:    float      = float(det.get("circularity", 0))
        self.orientation:    float      = float(det.get("orientation", 0))
        self.hu:             np.ndarray = np.asarray(det.get("hu", np.zeros(7)), dtype=float)
        self.mean_intensity: float      = float(det.get("mean_intensity", 0))
        self.com_dx:         float      = float(det.get("com_dx", 0))
        self.com_dy:         float      = float(det.get("com_dy", 0))
        self.neighbor_dists: np.ndarray = np.asarray(
            det.get("neighbor_dists", np.zeros(3)), dtype=float)
        self.neighbor_angles: np.ndarray = np.asarray(
            det.get("neighbor_angles", np.zeros(3)), dtype=float)

        self.age:    int  = 1
        self.misses: int  = 0
        self.history: list[tuple[float, float]] = [(self.x, self.y)]

    def predict(self) -> tuple[float, float]:
        return self.x + self.vx, self.y + self.vy

    def update(self, det: dict) -> None:
        nx, ny = float(det["x"]), float(det["y"])
        self.vx, self.vy = nx - self.x, ny - self.y
        self.x, self.y   = nx, ny

        self.area = float(det.get("area", self.area))
        self.circularity = float(det.get("circularity", self.circularity))
        self.orientation = float(det.get("orientation", self.orientation))
        self.hu = np.asarray(det.get("hu", self.hu), dtype=float)
        self.mean_intensity = float(det.get("mean_intensity", self.mean_intensity))
        self.com_dx = float(det.get("com_dx", self.com_dx))
        self.com_dy = float(det.get("com_dy", self.com_dy))
        self.neighbor_dists = np.asarray(
            det.get("neighbor_dists", self.neighbor_dists), dtype=float)
        self.neighbor_angles = np.asarray(
            det.get("neighbor_angles", self.neighbor_angles), dtype=float)

        self.age += 1
        self.misses = 0
        self.history.append((self.x, self.y))

    def mark_missed(self) -> None:
        self.misses += 1
        self.age += 1
        self.x += self.vx
        self.y += self.vy
        self.history.append((self.x, self.y))


# YOLO particle tracker 

class YOLOParticleTracker:
    """
    Hungarian-algorithm tracker for YOLO detections.

    Cost matrix combines:
      - Motion distance (primary, hard-gated)
      - Area similarity
      - Circularity similarity
      - Hu moment distance
      - Mean intensity difference
      - Intensity COM offset distance  ← professor's physical fingerprint
      - Orientation similarity
      - Neighbour angle similarity     ← inter-particle geometry
    """
    from scipy.optimize import linear_sum_assignment as _lsa

    def __init__(
        self,
        max_distance:    float = 50.0,
        max_misses:      int   = 8,
        neighbor_k:      int   = 3,
        w_motion:        float = 1.0,
        w_area:          float = 0.15,
        w_circularity:   float = 0.10,
        w_hu:            float = 0.20,
        w_intensity:     float = 0.10,
        w_com:           float = 0.20,   
        w_orientation:   float = 0.10,
        w_neighbor_dist: float = 0.08,
        w_neighbor_angle: float = 0.07,
    ):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.neighbor_k = neighbor_k
        self.w_motion = w_motion
        self.w_area = w_area
        self.w_circularity = w_circularity
        self.w_hu = w_hu
        self.w_intensity = w_intensity
        self.w_com = w_com
        self.w_orientation = w_orientation
        self.w_neighbor_dist = w_neighbor_dist
        self.w_neighbor_angle = w_neighbor_angle

        self._next_id: int = 1
        self.tracks: list[YOLOTrack] = []

    def _cost_matrix(self, detections: list[dict]) -> np.ndarray:
        from scipy.optimize import linear_sum_assignment  
        cost = np.full((len(self.tracks), len(detections)), 1e9)
        for i, tr in enumerate(self.tracks):
            px, py = tr.predict()
            for j, det in enumerate(detections):
                dist = math.hypot(det["x"] - px, det["y"] - py)
                if dist > self.max_distance:
                    continue

                motion_c = dist / self.max_distance
                area_c = abs(det["area"] - tr.area) / max(det["area"], tr.area, 1.0)
                circ_c = abs(det.get("circularity", 0) - tr.circularity)
                hu_c = float(np.mean(np.abs(
                    np.asarray(det.get("hu", tr.hu), dtype=float) - tr.hu
                )))

                int_c = abs(det.get("mean_intensity", tr.mean_intensity)
                             - tr.mean_intensity) / 255.0

                # Intensity COM — stable internal fingerprint
                com_c = math.hypot(
                    det.get("com_dx", tr.com_dx) - tr.com_dx,
                    det.get("com_dy", tr.com_dy) - tr.com_dy,
                )

                ori_c = _angle_diff_deg(
                    det.get("orientation", tr.orientation), tr.orientation
                ) / 90.0

                nd = np.asarray(det.get("neighbor_dists",  tr.neighbor_dists), dtype=float)
                na = np.asarray(det.get("neighbor_angles", tr.neighbor_angles), dtype=float)
                nd_c = float(np.mean(
                    np.abs(nd - tr.neighbor_dists) / np.maximum(np.abs(tr.neighbor_dists), 1.0)
                ))
                na_c = float(np.mean([
                    _angle_diff_deg(a, b) / 180.0
                    for a, b in zip(na.tolist(), tr.neighbor_angles.tolist())
                ]))

                cost[i, j] = (
                    self.w_motion * motion_c
                    + self.w_area * area_c
                    + self.w_circularity * circ_c
                    + self.w_hu * hu_c
                    + self.w_intensity * int_c
                    + self.w_com * com_c
                    + self.w_orientation * ori_c
                    + self.w_neighbor_dist  * nd_c
                    + self.w_neighbor_angle * na_c
                )
        return cost

    def update(self, detections: list[dict]) -> list[dict]:
        from scipy.optimize import linear_sum_assignment

        _add_neighbor_features(detections, k=self.neighbor_k)

        if not self.tracks:
            for det in detections:
                self._spawn(det)
            return detections

        cost = self._cost_matrix(detections)
        matched_t, matched_d = set(), set()

        if cost.size > 0:
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e8:
                    self.tracks[r].update(detections[c])
                    detections[c]["track_id"] = self.tracks[r].track_id
                    matched_t.add(r); matched_d.add(c)

        for idx, tr in enumerate(self.tracks):
            if idx not in matched_t:
                tr.mark_missed()

        for j, det in enumerate(detections):
            if j not in matched_d:
                self._spawn(det)
                det["track_id"] = self.tracks[-1].track_id

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return detections

    def _spawn(self, det: dict) -> None:
        tr = YOLOTrack(self._next_id, det)
        det["track_id"] = tr.track_id
        self._next_id += 1
        self.tracks.append(tr)

    def reset(self) -> None:
        self.tracks    = []
        self._next_id  = 1


# Trajectory drawing 

def draw_yolo_trajectories(
    frame_bgr: np.ndarray,
    tracker: YOLOParticleTracker,
    detections: list[dict],
    trail_length: int = 60,
) -> np.ndarray:
    """
    Draw YOLO bounding boxes (colour-coded per track ID) and fading
    trajectory trails onto a copy of the frame.
    """
    out = frame_bgr.copy()

    # Draw bounding boxes colour-coded by track ID
    det_by_id = {d.get("track_id"): d for d in detections if "track_id" in d}
    for tid, det in det_by_id.items():
        colour = _track_colour(tid)
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        conf_label = f"#{tid} {det.get('conf', 0):.2f}"
        cv2.putText(out, conf_label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)

    # Draw fading trajectory trails
    for tr in tracker.tracks:
        colour  = _track_colour(tr.track_id)
        history = tr.history[-trail_length:]

        for k in range(1, len(history)):
            p1 = (int(history[k - 1][0]), int(history[k - 1][1]))
            p2 = (int(history[k][0]), int(history[k][1]))
            thickness = max(1, int((k / len(history)) * 2.5))
            cv2.line(out, p1, p2, colour, thickness, cv2.LINE_AA)

        cx, cy = int(tr.x), int(tr.y)
        cv2.circle(out, (cx, cy), 4, colour, -1, cv2.LINE_AA)

        # Label: ID + orientation angle extracted from patch features
        label = f"P{tr.track_id}  {tr.orientation:.0f}°"
        cv2.putText(out, label, (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, colour, 1, cv2.LINE_AA)

    return out


# CSV export 

def export_yolo_predictions_csv(
    video_path: Path,
    model: YOLO,
    csv_path: Path,
    imgsz: int = 1024,
    conf: float = 0.25,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Run YOLO frame-by-frame and export detections to CSV.
    One row = one bounding box in one frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=imgsz, conf=conf,
                                    save=False, verbose=False)
            if results:
                result = results[0]
                boxes  = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy    = boxes.xyxy.cpu().numpy()
                    confs   = boxes.conf.cpu().numpy() if boxes.conf  is not None else np.array([])
                    classes = boxes.cls.cpu().numpy()  if boxes.cls   is not None else np.array([])
                    for di, box in enumerate(xyxy):
                        x1, y1, x2, y2 = map(float, box)
                        rows.append({
                            "frame": int(frame_idx),
                            "detection_id_in_frame": int(di),
                            "class_id": int(classes[di]) if di < len(classes) else -1,
                            "confidence": float(confs[di]) if di < len(confs) else np.nan,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "x_center": (x1 + x2) / 2,
                            "y_center": (y1 + y2) / 2,
                            "width":  x2 - x1,
                            "height": y2 - y1,
                            "area": (x2 - x1) * (y2 - y1),
                        })

            frame_idx += 1
            if progress_callback and frame_count > 0:
                progress_callback(frame_idx, frame_count)
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df, csv_path


# Trajectory video with tracking pass

def make_yolo_trajectory_video(
    video_path: Path,
    model: YOLO,
    out_path: Path,
    imgsz: int = 1024,
    conf: float = 0.25,
    max_distance: float = 50.0,
    max_misses: int = 8,
    trail_length: int = 60,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Second pass over the video:
      1. Run YOLO inference per frame
      2. Extract physical features from each detection's patch
      3. Update YOLOParticleTracker (Hungarian + physical features)
      4. Draw colour-coded bounding boxes + fading trajectory trails
      5. Save to MP4

    Also returns a tracking DataFrame (one row per track per frame) which
    adds orientation, com_dx/com_dy columns not present in the raw CSV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    tracker  = YOLOParticleTracker(max_distance=max_distance, max_misses=max_misses)
    all_rows: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # YOLO inference
            results = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf,
                                    save=False, verbose=False)

            detections: list[dict] = []
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy  = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
                    for di, box in enumerate(xyxy):
                        x1, y1, x2, y2 = map(float, box)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        # Extract physical features from the patch
                        phys = extract_patch_features(frame_gray, (x1, y1, x2, y2))

                        detections.append({
                            "x": cx, "y": cy,
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(confs[di]) if di < len(confs) else 0.0,
                            **phys,
                        })

            # Update tracker
            tracked = tracker.update(detections)

            # Record tracking results
            for det in tracked:
                if "track_id" not in det:
                    continue
                all_rows.append({
                    "frame": frame_idx,
                    "track_id": det["track_id"],
                    "x": det["x"],
                    "y": det["y"],
                    "conf": det.get("conf", np.nan),
                    "area": det.get("area", np.nan),
                    "orientation": det.get("orientation", np.nan),
                    "circularity": det.get("circularity", np.nan),
                    "com_dx":  det.get("com_dx", np.nan),
                    "com_dy": det.get("com_dy", np.nan),
                })

            # Draw trajectories
            annotated = draw_yolo_trajectories(
                frame_bgr, tracker, tracked, trail_length=trail_length,
            )
            cv2.putText(annotated, f"frame={frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)

            frame_idx += 1
            if progress_callback and frame_count > 0:
                progress_callback(frame_idx, frame_count)

    finally:
        cap.release()
        writer.release()

    df_tracking = pd.DataFrame(all_rows)
    return out_path, df_tracking


# Static plots 

def create_detection_count_plot(df: pd.DataFrame, out_path: Path) -> Path | None:
    if df.empty or "frame" not in df.columns:
        return None
    counts = df.groupby("frame").size().reset_index(name="num_detections")
    if counts.empty:
        return None
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(counts["frame"], counts["num_detections"], linewidth=1)
    plt.xlabel("frame"); plt.ylabel("detections")
    plt.title("YOLO detections per frame")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    return out_path


def create_trajectory_plot(df_tracking: pd.DataFrame, out_path: Path) -> Path | None:
    """Plot 2D trajectories for each tracked particle (matches CV pipeline style)."""
    if df_tracking.empty or "track_id" not in df_tracking.columns:
        return None
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    for tid, grp in df_tracking.dropna(subset=["x","y"]).groupby("track_id"):
        grp_s = grp.sort_values("frame")
        hue = int((tid * 137.508) % 180)
        hsv = np.uint8([[[hue, 220, 200]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        c = (bgr[2]/255, bgr[1]/255, bgr[0]/255)
        plt.plot(grp_s["x"], grp_s["y"],
                 marker="o", markersize=2, linewidth=1, label=f"P{int(tid)}", color=c)
    plt.legend(fontsize=7, ncol=3, loc="best")
    plt.title("YOLO tracked trajectories")
    plt.xlabel("x (px)"); plt.ylabel("y (px)")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close()
    return out_path


def create_center_coordinate_plot(df: pd.DataFrame, out_path: Path) -> Path | None:
    if df.empty:
        return None
    required = {"frame", "x_center", "y_center", "confidence"}
    if not required.issubset(df.columns):
        return None
    rep = (df.sort_values(["frame", "confidence"], ascending=[True, False])
             .groupby("frame", as_index=False).first())
    if rep.empty:
        return None
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(rep["frame"], rep["x_center"], linewidth=1)
    axes[0].set_ylabel("x_center")
    axes[0].set_title("Representative YOLO x-center over time")
    axes[1].plot(rep["frame"], rep["y_center"], linewidth=1)
    axes[1].set_ylabel("y_center"); axes[1].set_xlabel("frame")
    axes[1].set_title("Representative YOLO y-center over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close(fig)
    return out_path


# Public entry point 

def run_yolo_pipeline(
    video_path:   str | Path,
    model_path:   str | Path,
    output_dir:   str | Path,
    imgsz:        int   = 1024,
    conf:         float = 0.25,
    max_distance: float = 50.0,
    max_misses:   int   = 8,
    trail_length: int   = 60,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict:
    """
    Full YOLO pipeline with physical-feature tracking and trajectory video.

    Stages:
      1. Load model
      2. Ultralytics annotated video (bounding boxes, confidence labels)
      3. Raw detections CSV
      4. Tracking pass → trajectory video with fading coloured trails
      5. Static plots
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load model
    if progress_callback:
        progress_callback("loading_model", 0.05)
    model = YOLO(str(model_path))

    # Stage 2: Ultralytics annotated video
    if progress_callback:
        progress_callback("creating_predicted_video", 0.10)
    run_name = f"{video_path.stem}_{model_path.stem}_prediction"
    model.predict(
        source=str(video_path), imgsz=imgsz, conf=conf,
        save=True, project=str(output_dir), name=run_name,
        exist_ok=True, verbose=False,
    )
    run_dir = output_dir / run_name
    predicted_video_src  = _find_predicted_video(run_dir)
    predicted_video_path = output_dir / f"{video_path.stem}_{model_path.stem}_yolo.mp4"
    if predicted_video_src.resolve() != predicted_video_path.resolve():
        shutil.copy2(predicted_video_src, predicted_video_path)

    # Stage 3: Raw detections CSV
    if progress_callback:
        progress_callback("exporting_csv", 0.30)
    csv_path = output_dir / f"{video_path.stem}_{model_path.stem}_predictions.csv"

    def csv_prog(done: int, total: int):
        if progress_callback and total > 0:
            progress_callback("exporting_csv", 0.30 + 0.20 * done / total)

    predictions_df, _ = export_yolo_predictions_csv(
        video_path, model, csv_path, imgsz=imgsz, conf=conf,
        progress_callback=csv_prog,
    )

    # Stage 4: Tracking pass → trajectory video
    if progress_callback:
        progress_callback("tracking_and_trajectory_video", 0.50)
    traj_video_path = output_dir / f"{video_path.stem}_{model_path.stem}_trajectory.mp4"
    tracking_csv    = output_dir / f"{video_path.stem}_{model_path.stem}_tracking.csv"

    def traj_prog(done: int, total: int):
        if progress_callback and total > 0:
            progress_callback("tracking_and_trajectory_video", 0.50 + 0.35 * done / total)

    traj_video_path, df_tracking = make_yolo_trajectory_video(
        video_path, model, traj_video_path,
        imgsz=imgsz, conf=conf,
        max_distance=max_distance, max_misses=max_misses,
        trail_length=trail_length,
        progress_callback=traj_prog,
    )
    df_tracking.to_csv(tracking_csv, index=False)

    # Stage 5: Static plots
    if progress_callback:
        progress_callback("creating_plots", 0.92)

    detection_count_plot = create_detection_count_plot(
        predictions_df,
        output_dir / f"{video_path.stem}_{model_path.stem}_detection_counts.png",
    )
    trajectory_plot = create_trajectory_plot(
        df_tracking,
        output_dir / f"{video_path.stem}_{model_path.stem}_trajectory_plot.png",
    )
    coordinate_plot = create_center_coordinate_plot(
        predictions_df,
        output_dir / f"{video_path.stem}_{model_path.stem}_coordinate_plot.png",
    )

    if progress_callback:
        progress_callback("done", 1.0)

    num_total_detections      = int(len(predictions_df))
    num_frames_with_detections = (
        int(predictions_df["frame"].nunique())
        if not predictions_df.empty and "frame" in predictions_df.columns else 0
    )
    num_particles = (
        int(df_tracking["track_id"].nunique())
        if not df_tracking.empty and "track_id" in df_tracking.columns else 0
    )

    return {
        "predicted_video": str(predicted_video_path),
        "trajectory_video": str(traj_video_path),
        "predictions_csv":  str(csv_path),
        "tracking_csv": str(tracking_csv),
        "predictions_df":  predictions_df,
        "tracking_df": df_tracking,
        "detection_count_plot": str(detection_count_plot) if detection_count_plot else None,
        "trajectory_plot": str(trajectory_plot) if trajectory_plot else None,
        "coordinate_plot": str(coordinate_plot)  if coordinate_plot else None,
        "run_dir": str(run_dir),
        "num_frames_with_detections": num_frames_with_detections,
        "num_total_detections": num_total_detections,
        "num_particles": num_particles,
        "params": {
            "imgsz":  imgsz,
            "conf": conf,
            "max_distance": max_distance,
            "max_misses": max_misses,
            "trail_length": trail_length,
            "model_name": model_path.name,
        },
    }


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent

    VIDEO_PATH = PROJECT_ROOT / "data" / "final_inputs" / "three_particles_video.mp4"
    MODEL_PATH = PROJECT_ROOT / "code" / "models" / "yolo26n_best.pt"
    OUTPUT_DIR = HERE / "demo_outputs" / "yolo_test"

    out = run_yolo_pipeline(
        video_path=VIDEO_PATH, model_path=MODEL_PATH, output_dir=OUTPUT_DIR,
        imgsz=1024, conf=0.25,
    )
    print("Predicted video: ", out["predicted_video"])
    print("Trajectory video: ", out["trajectory_video"])
    print("Tracking CSV: ", out["tracking_csv"])
    print("Particles tracked:", out["num_particles"])