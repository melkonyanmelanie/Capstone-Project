"""
Part 2 of 3: Diagnostic outputs of the RAW fragments before merging.

This part produces the per-fragment CSV/plot/video, useful for
inspecting what SAM2 actually returned. Final results come from Part 3.


1. Annotated video writer:
   - Per-track sorted-frame array and int-rounded points are precomputed
     ONCE and reused for every frame. 
   - np.searchsorted to find current frame's track index in O(log N)
     instead of building a fresh sorted list per frame.
   - Tail rendering UNCHANGED: still per-segment cv2.line calls with the
     same alpha-fade gradient (k/M) as your original. Output pixels
     identical.

2. Frame loader: same preallocated-buffer trick as Part 1.

3. CSV/plot/Kalman/SavGol code.

Run (from project root, after refactor):
    python code/run_pipeline.py --only part2
or directly:
    python code/pipeline_part2.py
"""

import json
import pickle
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tracking_on_final_sr_videos_1"
DATA_DIR = PROJECT_ROOT / "data" / "final_inputs_sr"

KALMAN_Q = 1e-3
KALMAN_R = 15.0
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3

TAIL_LENGTH = 40
PALETTE = [
    (255, 60, 60), (60, 255, 60), ( 60, 100, 255), (255, 220, 0),
    (0, 220, 255), (255, 60, 220), (180, 60, 255), (255, 140, 0),
    (0, 255, 180), (140, 255, 0), (255, 160, 100), (100, 200, 255),
    (200, 100, 255), (255, 100, 150), (150, 255, 100),
]


# KALMAN + SAVGOL  

def kalman_smooth(positions: dict) -> dict:
    if not positions:
        return {}
    sorted_frames = sorted(positions.keys())
    x = np.array([positions[sorted_frames[0]][0],
                  positions[sorted_frames[0]][1], 0.0, 0.0], dtype=np.float64)
    P = np.eye(4) * 500.0
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * KALMAN_R
    smoothed = {}
    prev_fi = None
    for fi in sorted_frames:
        if prev_fi is not None:
            gap = fi - prev_fi
            F_gap = np.array([[1, 0, gap, 0], [0, 1, 0, gap],
                              [0, 0, 1,   0], [0, 0, 0,   1]], dtype=np.float64)
            Q_gap = np.eye(4) * KALMAN_Q * gap
            x = F_gap @ x
            P = F_gap @ P @ F_gap.T + Q_gap
        else:
            x = np.array([positions[fi][0], positions[fi][1], 0.0, 0.0])
        z = np.array([positions[fi][0], positions[fi][1]], dtype=np.float64)
        y_k = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_k
        P = (np.eye(4) - K @ H) @ P
        smoothed[fi] = (float(x[0]), float(x[1]))
        prev_fi = fi
    return smoothed


def savgol_smooth(positions: dict) -> dict:
    if len(positions) < 4:
        return positions
    sorted_frames = sorted(positions.keys())
    xs = [positions[f][0] for f in sorted_frames]
    ys = [positions[f][1] for f in sorted_frames]
    n = len(sorted_frames)
    w = min(SMOOTH_WINDOW, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return positions
    poly = min(SMOOTH_POLY, w - 1)
    xs_s = savgol_filter(xs, w, poly)
    ys_s = savgol_filter(ys, w, poly)
    return {f: (float(xs_s[i]), float(ys_s[i])) for i, f in enumerate(sorted_frames)}


# CSV

def save_csv(smoothed_tracks, particles, fps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tracks.csv"
    pinfo = {p["particle_id"]: p for p in particles}
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "particle_id", "x", "y",
                         "displacement_px", "cumulative_displacement_px",
                         "entry_frame", "exit_frame"])
        for pid, positions in sorted(smoothed_tracks.items()):
            sorted_frames = sorted(positions.keys())
            cum_disp = 0.0
            prev_xy  = None
            p = pinfo.get(pid, {})
            for fi in sorted_frames:
                cx, cy = positions[fi]
                disp = np.hypot(cx - prev_xy[0], cy - prev_xy[1]) if prev_xy else 0.0
                cum_disp += disp
                prev_xy = (cx, cy)
                writer.writerow([fi, round(fi / fps, 5), pid,
                                 round(cx, 3), round(cy, 3),
                                 round(disp, 4), round(cum_disp, 4),
                                 p.get("entry_frame", ""),
                                 p.get("exit_frame", "")])
    print(f"CSV saved: {out}")


# PLOTS

def save_plots(smoothed_tracks, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not smoothed_tracks:
        return
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for pid, positions in sorted(smoothed_tracks.items()):
        sorted_frames = sorted(positions.keys())
        if not sorted_frames:
            continue
        c = colors[pid % len(colors)]
        xs = [positions[f][0] for f in sorted_frames]
        ys = [positions[f][1] for f in sorted_frames]
        axes[0].plot(sorted_frames, xs, color=c, label=f"P{pid}", linewidth=1.5)
        axes[1].plot(sorted_frames, ys, color=c, label=f"P{pid}", linewidth=1.5)
        axes[2].plot(xs, ys, color=c, label=f"P{pid}", linewidth=1.5)
        axes[2].plot(xs[0],  ys[0],  "o", color=c, markersize=6)
        axes[2].plot(xs[-1], ys[-1], "s", color=c, markersize=6)
    for ax, xl, yl, title in [
        (axes[0], "Frame", "X (px)", "X Position vs Frame (Kalman + SavGol smoothed)"),
        (axes[1], "Frame", "Y (px)", "Y Position vs Frame (Kalman + SavGol smoothed)"),
        (axes[2], "X (px)", "Y (px)", "XY Trajectories  (○=start  □=end)"),
    ]:
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    axes[2].invert_yaxis()
    plt.tight_layout()
    out = out_dir / "tracks_plot.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Plot saved: {out}")


# ANNOTATED VIDEO  

def save_annotated_video(frames, smoothed_tracks, particles, fps, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    pinfo = {p["particle_id"]: p for p in particles}

    track_data = {}
    for pid, positions in smoothed_tracks.items():
        if not positions:
            continue
        sorted_frames = sorted(positions.keys())
        sf_arr = np.array(sorted_frames, dtype=np.int64)
        pts = np.array([positions[f] for f in sorted_frames], dtype=np.float32)
        pts_int = np.round(pts).astype(np.int32)
        track_data[pid] = (sf_arr, pts_int)

    frame_pids = defaultdict(list)
    for pid, (sf_arr, _) in track_data.items():
        for f in sf_arr:
            frame_pids[int(f)].append(pid)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for fi, frame in enumerate(tqdm(frames, desc="  Writing video",
                                    unit="f", mininterval=0.5)):
        canvas = frame.copy()

        for pid in frame_pids.get(fi, []):
            sf_arr, pts_int = track_data[pid]
            i = np.searchsorted(sf_arr, fi)
            if i >= len(sf_arr) or sf_arr[i] != fi:
                continue
            color = PALETTE[pid % len(PALETTE)]
            cx, cy = int(pts_int[i, 0]), int(pts_int[i, 1])

            tail_start = max(0, i - TAIL_LENGTH + 1)
            tail_pts = pts_int[tail_start:i+1]
            M = len(tail_pts)
            if M >= 2:
                for k in range(1, M):
                    alpha = k / M
                    tc = tuple(int(c * alpha) for c in color)
                    p1 = (int(tail_pts[k-1, 0]), int(tail_pts[k-1, 1]))
                    p2 = (int(tail_pts[k,   0]), int(tail_pts[k,   1]))
                    cv2.line(canvas, p1, p2, tc, 1, cv2.LINE_AA)

            # Particle dot + ID
            cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{pid}", (cx + 6, cy - 6),
                        font, 0.4, color, 1, cv2.LINE_AA)

            # Entry/exit markers
            p = pinfo.get(pid, {})
            if fi == p.get("entry_frame"):
                cv2.circle(canvas, (cx, cy), 8, color, 2)
            if fi == p.get("exit_frame"):
                cv2.line(canvas, (cx-6, cy-6), (cx+6, cy+6), color, 2)
                cv2.line(canvas, (cx+6, cy-6), (cx-6, cy+6), color, 2)

        cv2.putText(canvas,
                    f"Frame {fi:05d}  particles: {len(frame_pids.get(fi, []))}",
                    (4, 12), font, 0.4, (255, 255, 255), 1)
        writer.write(canvas)

    writer.release()
    print(f"Annotated video saved: {out_path}")


# MAIN

def _load_frames_fast(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    buf = np.empty((n, h, w, 3), dtype=np.uint8)
    idx = 0
    while idx < n:
        ret, frame = cap.read()
        if not ret:
            break
        buf[idx] = frame
        idx += 1
    cap.release()
    return [buf[i] for i in range(idx)]


def main():
    part1_dirs = sorted(OUTPUT_ROOT.glob("*/part1/meta.json"))
    if not part1_dirs:
        raise FileNotFoundError(f"No Part 1 results under {OUTPUT_ROOT}.")
    print(f"Found {len(part1_dirs)} processed video(s)\n")

    for meta_path in part1_dirs:
        part1_dir = meta_path.parent
        stem = part1_dir.parent.name
        out_dir = part1_dir.parent / "part2"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"─── {stem}")
        with open(part1_dir / "meta.json") as f:
            meta = json.load(f)
        with open(part1_dir / "gmm_particles.json") as f:
            particles = json.load(f)
        with open(part1_dir / "sam2_tracks.pkl", "rb") as f:
            sam2_tracks = pickle.load(f)

        fps = meta["fps"]
        print(f"{meta['orig_w']}x{meta['orig_h']}  {fps:.1f}fps  "
              f"{len(particles)} fragments")

        if not sam2_tracks:
            print(f"WARNING: No SAM2 tracks, skipping output\n")
            continue

        print("Smoothing tracks...")
        smoothed = {pid: savgol_smooth(kalman_smooth(pos))
                    for pid, pos in sam2_tracks.items()}
        print(f"Tracks smoothed: {len(smoothed)}")

        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            cand = DATA_DIR / f"{stem}{ext}"
            if cand.exists():
                video_path = cand; break

        if video_path is None:
            print(f"  WARNING: no source video for {stem}; skipping annotated video")
            save_csv(smoothed, particles, fps, out_dir)
            save_plots(smoothed, out_dir)
            continue

        frames = _load_frames_fast(video_path)
        save_csv(smoothed, particles, fps, out_dir)
        save_plots(smoothed, out_dir)
        save_annotated_video(frames, smoothed, particles, fps,
                             out_dir / "tracked_video.mp4")
        print()

    print(f"All Part 2 outputs saved under: {OUTPUT_ROOT}")
    print("Run final_pipeline_part3.py for FINAL merged outputs.")


if __name__ == "__main__":
    main()