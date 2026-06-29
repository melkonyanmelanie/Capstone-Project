from pathlib import Path
import csv
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "final_inputs_sr" / "one_particle_video_sr.mp4"
CSV_PATH   = PROJECT_ROOT / "outputs" / "tracking_on_final_sr_videos_1" / "one_particle_video_sr" / "part3" / "tracks_merged.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "manual_validation"

FRAME_STRIDE = 100  
N_PARTICLES  = 1    

FPS = 33

ZOOM_FACTOR = 2.0   


_boxes_this_frame = []      # list of (x1, y1, x2, y2) in DISPLAY coords
_drag_start = None          # (x, y) start of an in-progress drag, or None
_drag_current = None        # (x, y) current mouse position while dragging


def _mouse_callback(event, x, y, flags, param):
    global _boxes_this_frame, _drag_start, _drag_current
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_start = (x, y)
        _drag_current = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if _drag_start is not None:
            _drag_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if _drag_start is not None:
            x1, y1 = _drag_start
            x2, y2 = x, y
            bx1, bx2 = sorted([x1, x2])
            by1, by2 = sorted([y1, y2])
            if (bx2 - bx1) >= 3 and (by2 - by1) >= 3:
                _boxes_this_frame.append((bx1, by1, bx2, by2))
            _drag_start = None
            _drag_current = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _boxes_this_frame:
            _boxes_this_frame.pop()


def _redraw(base_frame, boxes_display, drag_start, drag_current,
            frame_idx, expected_n):
    img = base_frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_display):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(img, f"P{i}", (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    if drag_start is not None and drag_current is not None:
        cv2.rectangle(img, drag_start, drag_current, (0, 255, 0), 1, cv2.LINE_AA)

    msg = (f"Frame {frame_idx}  boxes: {len(boxes_display)}/{expected_n}   "
           "DRAG=box  Rclick=undo  SPACE=next  S=skip  R=reset  Q=quit")
    cv2.putText(img, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _load_existing_boxes(out_csv: Path):
    if not out_csv.exists():
        return {}
    df = pd.read_csv(out_csv)
    out = {}
    for fi, sub in df.groupby("frame"):
        out[int(fi)] = [
            (float(r.x1_orig), float(r.y1_orig),
             float(r.x2_orig), float(r.y2_orig))
            for r in sub.itertuples()
        ]
    return out


def _append_boxes(out_csv: Path, frame_idx: int, boxes_original: list):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["frame", "box_index",
                        "x1_orig", "y1_orig", "x2_orig", "y2_orig",
                        "cx_orig", "cy_orig", "w_orig", "h_orig"])
        for i, (x1, y1, x2, y2) in enumerate(boxes_original):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w_orig, h_orig = (x2 - x1), (y2 - y1)
            w.writerow([frame_idx, i,
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                        round(cx, 2), round(cy, 2),
                        round(w_orig, 2), round(h_orig, 2)])


def annotate_video(video_path: Path, stride: int, expected_n: int,
                   out_csv: Path) -> dict:
    global _boxes_this_frame, _drag_start, _drag_current

    existing = _load_existing_boxes(out_csv)
    if existing:
        print(f"Resuming: {len(existing)} frames already annotated")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = [fi for fi in range(0, total, stride)]
    print(f"Will annotate {len(frame_indices)} frames "
          f"(every {stride} frames out of {total})")

    win = "Manual annotation - draw box around each particle"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_callback)

    annotations = dict(existing)
    quit_early = False
    for frame_idx in frame_indices:
        if frame_idx in annotations:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}; stopping")
            break

        disp = (cv2.resize(frame, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR,
                           interpolation=cv2.INTER_NEAREST)
                if ZOOM_FACTOR != 1.0 else frame.copy())

        _boxes_this_frame = []
        _drag_start = None
        _drag_current = None

        while True:
            shown = _redraw(disp, _boxes_this_frame,
                            _drag_start, _drag_current,
                            frame_idx, expected_n)
            cv2.imshow(win, shown)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(' '):
                orig = [(x1 / ZOOM_FACTOR, y1 / ZOOM_FACTOR,
                         x2 / ZOOM_FACTOR, y2 / ZOOM_FACTOR)
                        for (x1, y1, x2, y2) in _boxes_this_frame]
                annotations[frame_idx] = orig
                _append_boxes(out_csv, frame_idx, orig)
                break
            elif key == ord('s'):
                annotations[frame_idx] = []
                _append_boxes(out_csv, frame_idx, [])
                break
            elif key == ord('r'):
                _boxes_this_frame = []
                _drag_start = None
                _drag_current = None
            elif key == ord('q'):
                quit_early = True
                break

        if quit_early:
            print("Quitting early. Progress saved.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return annotations


def load_pipeline_tracks(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for pid, sub in df.groupby("particle_id"):
        out[int(pid)] = sub.sort_values("frame").reset_index(drop=True)
    return out


def boxes_to_centers(annotations: dict) -> dict:
    out = {}
    for fi, boxes in annotations.items():
        out[fi] = [((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                   for (x1, y1, x2, y2) in boxes]
    return out


def match_annotations_to_tracks(centers: dict, tracks: dict):
    rows = []
    for fi, ann_centers in sorted(centers.items()):
        if not ann_centers:
            continue
        tracker_positions = {}
        for pid, df in tracks.items():
            hit = df[df["frame"] == fi]
            if not hit.empty:
                tracker_positions[pid] = (float(hit["x"].iloc[0]),
                                          float(hit["y"].iloc[0]))
        if not tracker_positions:
            continue

        unmatched_ann = list(enumerate(ann_centers))
        unmatched_pid = list(tracker_positions.keys())
        while unmatched_ann and unmatched_pid:
            best, best_d = None, float("inf")
            for ai, (cx, cy) in unmatched_ann:
                for pid in unmatched_pid:
                    tx, ty = tracker_positions[pid]
                    d = np.hypot(cx - tx, cy - ty)
                    if d < best_d:
                        best_d = d
                        best = (ai, pid)
            if best is None:
                break
            ai, pid = best
            cx, cy = ann_centers[ai]
            tx, ty = tracker_positions[pid]
            rows.append({
                "frame": int(fi),
                "particle_id": int(pid),
                "manual_x": cx, "manual_y": cy,
                "tracker_x": tx, "tracker_y": ty,
                "error_px": float(best_d),
            })
            unmatched_ann = [(a_i, xy) for (a_i, xy) in unmatched_ann if a_i != ai]
            unmatched_pid = [p for p in unmatched_pid if p != pid]
    return rows


def write_overlay_plot(tracks: dict, centers: dict, out_path: Path,
                       fps=None):
    """
    Displacement-from-start vs. time plot.

      Line  - pipeline trajectory: ||(x(t), y(t)) - (x0, y0)||
              where (x0, y0) is the FIRST recorded pipeline position
              for that particle.
      Dots  - manual annotations: ||(cx, cy) - (x0, y0)|| at the
              corresponding frame, where (x0, y0) is taken from the
              SAME particle's pipeline start (so both share an origin).

    If fps is provided and > 0, the x-axis is time in seconds. Otherwise
    it is the frame index.
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    palette = plt.cm.tab10.colors
    use_time = (fps is not None) and (fps > 0)

    # 1. Pipeline displacement curves, one per particle
    particle_origins = {}  # pid -> (x0, y0)
    for pid, df in tracks.items():
        df = df.sort_values("frame").reset_index(drop=True)
        x0, y0 = float(df["x"].iloc[0]), float(df["y"].iloc[0])
        particle_origins[pid] = (x0, y0)
        disp = np.hypot(df["x"] - x0, df["y"] - y0)
        t = (df["frame"] / fps) if use_time else df["frame"]
        c = palette[pid % len(palette)]
        ax.plot(t, disp, color=c, linewidth=1.0,
                label=f"Pipeline P{pid}")

    # 2. Manual annotation displacements (matched per-frame to closest pipeline particle)
    red_t, red_d = [], []
    for fi, ann_centers in sorted(centers.items()):
        if not ann_centers:
            continue
        tracker_positions = {}
        for pid, df in tracks.items():
            hit = df[df["frame"] == fi]
            if not hit.empty:
                tracker_positions[pid] = (float(hit["x"].iloc[0]),
                                          float(hit["y"].iloc[0]))
        if not tracker_positions:
            continue

        unmatched_ann = list(ann_centers)
        unmatched_pid = list(tracker_positions.keys())
        while unmatched_ann and unmatched_pid:
            best, best_d = None, float("inf")
            for ai, (cx, cy) in enumerate(unmatched_ann):
                for pid in unmatched_pid:
                    tx, ty = tracker_positions[pid]
                    d = np.hypot(cx - tx, cy - ty)
                    if d < best_d:
                        best_d = d
                        best = (ai, pid)
            if best is None:
                break
            ai, pid = best
            cx, cy = unmatched_ann[ai]
            x0, y0 = particle_origins[pid]
            disp = float(np.hypot(cx - x0, cy - y0))
            red_t.append((fi / fps) if use_time else fi)
            red_d.append(disp)
            unmatched_ann.pop(ai)
            unmatched_pid.remove(pid)

    if red_t:
        ax.plot(red_t, red_d, "o", color="red", markersize=7,
                label=f"Human annotation (n={len(red_t)})", zorder=5)

    ax.set_xlabel("Time (s)" if use_time else "Frame")
    ax.set_ylabel("Displacement from start (px)")
    ax.set_title("Particle displacement — Pipeline vs Human")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Overlay plot written: {out_path}")


def write_error_summary(matches: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_frame_errors.csv"
    txt_path = out_dir / "annotation_error_summary.txt"

    pd.DataFrame(matches).to_csv(csv_path, index=False)

    lines = ["Manual-annotation vs pipeline error summary",
             "=" * 60, ""]
    if not matches:
        lines.append("No matched annotations.")
    else:
        errs = np.array([m["error_px"] for m in matches])
        lines.append(f"Annotated frames matched: {len(matches)}")
        lines.append(f"Mean error:    {errs.mean():.2f} px")
        lines.append(f"Median error:  {np.median(errs):.2f} px")
        lines.append(f"Std error:     {errs.std():.2f} px")
        lines.append(f"Max error:     {errs.max():.2f} px")
        lines.append(f"95th pct:      {np.percentile(errs, 95):.2f} px")
        df = pd.DataFrame(matches)
        for pid, sub in df.groupby("particle_id"):
            e = sub["error_px"].values
            lines.append("")
            lines.append(f"  Particle P{pid}: n={len(e)}, "
                         f"mean={e.mean():.2f} px, median={np.median(e):.2f} px")

    txt_path.write_text("\n".join(lines))
    print(f"Error summary written: {txt_path}")
    print("\n".join(lines))


def main():
    if not VIDEO_PATH.exists():
        sys.exit(f"Video not found: {VIDEO_PATH}")
    if not CSV_PATH.exists():
        sys.exit(f"Tracks CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boxes_csv = OUTPUT_DIR / "manual_boxes.csv"

    print("\n── Manual annotation tool (box-drawing mode) ──")
    print(f"Video : {VIDEO_PATH.name}")
    print(f"CSV   : {CSV_PATH.name}")
    print(f"Stride: every {FRAME_STRIDE} frames, expecting {N_PARTICLES} particle(s)\n")
    print("Controls:  LEFT-DRAG=box  RIGHT-click=undo  SPACE=next  "
          "S=skip  R=reset  Q=quit\n")

    annotations = annotate_video(VIDEO_PATH, FRAME_STRIDE,
                                 N_PARTICLES, boxes_csv)
    print(f"\nAnnotation finished: {len(annotations)} frames have data\n")

    centers = boxes_to_centers(annotations)
    tracks  = load_pipeline_tracks(CSV_PATH)
    print(f"Loaded {len(tracks)} pipeline particle(s) from CSV")

    matches = match_annotations_to_tracks(centers, tracks)

    write_overlay_plot(tracks, centers, OUTPUT_DIR / "overlay_plot.png", fps=FPS)
    write_error_summary(matches, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()