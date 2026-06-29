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

VIDEO_PATH = PROJECT_ROOT / "data" / "final_inputs_sr" / "two_particles_video_sr.mp4"
CSV_PATH   = PROJECT_ROOT / "outputs" / "tracking_on_final_sr_videos_1" / "two_particles_video_sr" / "part3" / "tracks_merged.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "manual_validation_two_particle"

FRAME_STRIDE = 200  
N_PARTICLES  = 2    

FPS = 33

ZOOM_FACTOR = 2.0   

ID_COLORS_BGR = [
    (255, 120, 60),    # P0 - blue/cyan
    (60, 60, 255),     # P1 - red
]
ID_COLORS_MPL = [
    "#3C78FF",         # P0
    "#FF3C3C",         # P1
]

_boxes_this_frame = []
_drag_start = None
_drag_current = None
_active_id = 0


def _mouse_callback(event, x, y, flags, param):
    global _boxes_this_frame, _drag_start, _drag_current, _active_id
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
                _boxes_this_frame.append((bx1, by1, bx2, by2, _active_id))
                # Auto-toggle to the other identity for the next box so the
                # common case (draw P0, draw P1) is a smooth two-stroke flow.
                _active_id = 1 - _active_id
            _drag_start = None
            _drag_current = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _boxes_this_frame:
            removed = _boxes_this_frame.pop()
            # When undoing, restore the active identity to that of the
            # removed box, so the next draw replaces what you just undid.
            _active_id = removed[4]


def _redraw(base_frame, boxes_display, drag_start, drag_current,
            frame_idx, expected_n, active_id):
    img = base_frame.copy()
    for (x1, y1, x2, y2, pid) in boxes_display:
        color = ID_COLORS_BGR[pid % len(ID_COLORS_BGR)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, f"P{pid}", (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    if drag_start is not None and drag_current is not None:
        preview_color = ID_COLORS_BGR[active_id % len(ID_COLORS_BGR)]
        cv2.rectangle(img, drag_start, drag_current, preview_color, 1, cv2.LINE_AA)

    msg1 = (f"Frame {frame_idx}  boxes: {len(boxes_display)}/{expected_n}   "
            f"ACTIVE ID = P{active_id}")
    msg2 = ("DRAG=box  '0'/'1'=set ID  Rclick=undo  SPACE=next  "
            "S=skip  R=reset  Q=quit")
    for y, text in [(22, msg1), (44, msg2)]:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Small color legend in the top-right
    h, w = img.shape[:2]
    legend_x = w - 130
    for pid, color in enumerate(ID_COLORS_BGR[:N_PARTICLES]):
        y = 20 + pid * 22
        cv2.rectangle(img, (legend_x, y - 12), (legend_x + 18, y + 4),
                      color, -1)
        cv2.putText(img, f"= P{pid}", (legend_x + 24, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img, f"= P{pid}", (legend_x + 24, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return img


def _load_existing_boxes(out_csv: Path):
    if not out_csv.exists():
        return {}
    df = pd.read_csv(out_csv)
    out = {}
    for fi, sub in df.groupby("frame"):
        out[int(fi)] = [
            (float(r.x1_orig), float(r.y1_orig),
             float(r.x2_orig), float(r.y2_orig),
             int(r.identity))
            for r in sub.itertuples()
        ]
    return out


def _append_boxes(out_csv: Path, frame_idx: int, boxes_original: list):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["frame", "identity", "box_index",
                        "x1_orig", "y1_orig", "x2_orig", "y2_orig",
                        "cx_orig", "cy_orig", "w_orig", "h_orig"])
        for i, (x1, y1, x2, y2, pid) in enumerate(boxes_original):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w_orig, h_orig = (x2 - x1), (y2 - y1)
            w.writerow([frame_idx, pid, i,
                        round(x1, 2), round(y1, 2),
                        round(x2, 2), round(y2, 2),
                        round(cx, 2), round(cy, 2),
                        round(w_orig, 2), round(h_orig, 2)])


def annotate_video(video_path: Path, stride: int, expected_n: int,
                   out_csv: Path) -> dict:
    global _boxes_this_frame, _drag_start, _drag_current, _active_id

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

    win = "Manual annotation - two particle (assign P0 / P1)"
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
        _active_id = 0  # each new frame starts ready to draw P0 first

        while True:
            shown = _redraw(disp, _boxes_this_frame,
                            _drag_start, _drag_current,
                            frame_idx, expected_n, _active_id)
            cv2.imshow(win, shown)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(' '):
                orig = [(x1 / ZOOM_FACTOR, y1 / ZOOM_FACTOR,
                         x2 / ZOOM_FACTOR, y2 / ZOOM_FACTOR, pid)
                        for (x1, y1, x2, y2, pid) in _boxes_this_frame]
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
                _active_id = 0
            elif key == ord('0'):
                _active_id = 0
            elif key == ord('1'):
                _active_id = 1
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
    """
    Returns: frame_idx -> list of (cx, cy, identity)
    """
    out = {}
    for fi, boxes in annotations.items():
        out[fi] = [((x1 + x2) / 2.0, (y1 + y2) / 2.0, pid)
                   for (x1, y1, x2, y2, pid) in boxes]
    return out


def _assign_identity_to_pipeline(centers: dict, tracks: dict):
    """
    Determine which pipeline particle_id corresponds to manual identity 0
    vs identity 1. This is done by looking at the FIRST annotated frame
    that has at least one manual box AND a pipeline detection: we map
    each manual identity to its nearest pipeline particle at that frame.

    Returns a dict: manual_identity -> pipeline_particle_id
    """
    for fi in sorted(centers.keys()):
        ann = centers[fi]
        if not ann:
            continue
        tracker_positions = {}
        for pid, df in tracks.items():
            hit = df[df["frame"] == fi]
            if not hit.empty:
                tracker_positions[pid] = (float(hit["x"].iloc[0]),
                                          float(hit["y"].iloc[0]))
        if not tracker_positions:
            continue

        # Greedy nearest-neighbour assignment between manual identities
        # and pipeline particle_ids at this single anchor frame.
        unmatched_ann = list(ann)  # (cx, cy, identity)
        unmatched_pid = list(tracker_positions.keys())
        mapping = {}
        while unmatched_ann and unmatched_pid:
            best, best_d = None, float("inf")
            for ai, (cx, cy, mid) in enumerate(unmatched_ann):
                for pid in unmatched_pid:
                    tx, ty = tracker_positions[pid]
                    d = np.hypot(cx - tx, cy - ty)
                    if d < best_d:
                        best_d = d
                        best = (ai, pid, mid)
            if best is None:
                break
            ai, pid, mid = best
            mapping[mid] = pid
            unmatched_ann.pop(ai)
            unmatched_pid.remove(pid)
        if mapping:
            print(f"Identity mapping (anchored at frame {fi}): "
                  f"{{manual_id: pipeline_pid}} = {mapping}")
            return mapping
    return {}


def match_annotations_to_tracks(centers: dict, tracks: dict, id_map: dict):
    """
    Match manual annotations to pipeline positions using the identity
    mapping (so we measure error and detect identity swaps).
    """
    rows = []
    for fi, ann_centers in sorted(centers.items()):
        if not ann_centers:
            continue
        for (cx, cy, mid) in ann_centers:
            pipeline_pid = id_map.get(mid)
            if pipeline_pid is None:
                continue
            df = tracks.get(pipeline_pid)
            if df is None:
                continue
            hit = df[df["frame"] == fi]
            if hit.empty:
                continue
            tx = float(hit["x"].iloc[0])
            ty = float(hit["y"].iloc[0])
            err = float(np.hypot(cx - tx, cy - ty))
            rows.append({
                "frame": int(fi),
                "manual_identity": int(mid),
                "pipeline_particle_id": int(pipeline_pid),
                "manual_x": cx, "manual_y": cy,
                "tracker_x": tx, "tracker_y": ty,
                "error_px": err,
            })
    return rows


def write_overlay_plot(tracks: dict, centers: dict, id_map: dict,
                       out_path: Path, fps=None):
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    use_time = (fps is not None) and (fps > 0)
    pid_to_color = {}
    for mid, pipeline_pid in id_map.items():
        pid_to_color[pipeline_pid] = ID_COLORS_MPL[mid % len(ID_COLORS_MPL)]

    particle_origins = {}  # pipeline_pid -> (x0, y0)
    for pid, df in tracks.items():
        df = df.sort_values("frame").reset_index(drop=True)
        x0, y0 = float(df["x"].iloc[0]), float(df["y"].iloc[0])
        particle_origins[pid] = (x0, y0)
        disp = np.hypot(df["x"] - x0, df["y"] - y0)
        t = (df["frame"] / fps) if use_time else df["frame"]
        color = pid_to_color.get(pid, "#888888")
        ax.plot(t, disp, color=color, linewidth=1.0,
                label=f"Pipeline P{pid}")


    by_id = {mid: ([], []) for mid in id_map.keys()}
    for fi, ann_centers in sorted(centers.items()):
        for (cx, cy, mid) in ann_centers:
            pipeline_pid = id_map.get(mid)
            if pipeline_pid is None or pipeline_pid not in particle_origins:
                continue
            x0, y0 = particle_origins[pipeline_pid]
            disp = float(np.hypot(cx - x0, cy - y0))
            tval = (fi / fps) if use_time else fi
            by_id[mid][0].append(tval)
            by_id[mid][1].append(disp)

    for mid, (ts, ds) in by_id.items():
        if not ts:
            continue
        color = ID_COLORS_MPL[mid % len(ID_COLORS_MPL)]
        ax.plot(ts, ds, "o", color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.6,
                label=f"Human P{mid} (n={len(ts)})", zorder=5)

    ax.set_xlabel("Time (s)" if use_time else "Frame")
    ax.set_ylabel("Displacement from start (px)")
    ax.set_title("Particle displacement — Pipeline vs Human (two particles)")
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

    lines = ["Manual-annotation vs pipeline error summary (two particles)",
             "=" * 60, ""]
    if not matches:
        lines.append("No matched annotations.")
    else:
        errs = np.array([m["error_px"] for m in matches])
        lines.append(f"Annotated detections matched: {len(matches)}")
        lines.append(f"Mean error:    {errs.mean():.2f} px")
        lines.append(f"Median error:  {np.median(errs):.2f} px")
        lines.append(f"Std error:     {errs.std():.2f} px")
        lines.append(f"Max error:     {errs.max():.2f} px")
        lines.append(f"95th pct:      {np.percentile(errs, 95):.2f} px")
        df = pd.DataFrame(matches)
        for mid, sub in df.groupby("manual_identity"):
            e = sub["error_px"].values
            pipeline_pids = sub["pipeline_particle_id"].unique().tolist()
            lines.append("")
            lines.append(f"  Manual P{mid} -> pipeline {pipeline_pids}: "
                         f"n={len(e)}, mean={e.mean():.2f} px, "
                         f"median={np.median(e):.2f} px, max={e.max():.2f} px")

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

    print("\n── Manual annotation tool (two-particle, identity-aware) ──")
    print(f"Video : {VIDEO_PATH.name}")
    print(f"CSV   : {CSV_PATH.name}")
    print(f"Stride: every {FRAME_STRIDE} frames, expecting {N_PARTICLES} particle(s)\n")
    print("Controls:  LEFT-DRAG=box  '0'/'1'=set identity  RIGHT-click=undo")
    print("           SPACE=next  S=skip  R=reset  Q=quit")
    print("Tip: identity auto-toggles after each box, so usually you can just")
    print("     draw P0, then draw P1, then press SPACE.\n")

    annotations = annotate_video(VIDEO_PATH, FRAME_STRIDE,
                                 N_PARTICLES, boxes_csv)
    print(f"\nAnnotation finished: {len(annotations)} frames have data\n")

    centers = boxes_to_centers(annotations)
    tracks  = load_pipeline_tracks(CSV_PATH)
    print(f"Loaded {len(tracks)} pipeline particle(s) from CSV")

    id_map = _assign_identity_to_pipeline(centers, tracks)
    if not id_map:
        print("WARNING: could not map manual identities to pipeline IDs "
              "(no frame had both manual + pipeline detections).")

    matches = match_annotations_to_tracks(centers, tracks, id_map)

    write_overlay_plot(tracks, centers, id_map,
                       OUTPUT_DIR / "overlay_plot.png", fps=FPS)
    write_error_summary(matches, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()