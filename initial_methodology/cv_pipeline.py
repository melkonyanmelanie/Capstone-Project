from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Callable
import os
import math

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
cv2.setNumThreads(1)


# Preprocessing

def preprocess_corrected_norm(
    gray_frame: np.ndarray,
    k_small: int = 5,
    K_big: int = 101,
) -> np.ndarray:
    """
    Two-stage Gaussian background correction.
    Dark particles become bright blobs after background subtraction.
    """
    if k_small % 2 == 0 or K_big % 2 == 0:
        raise ValueError("k_small and K_big must be odd integers.")
    gray_smooth = cv2.GaussianBlur(gray_frame, (k_small, k_small), 0)
    background  = cv2.GaussianBlur(gray_smooth, (K_big, K_big), 0)
    corrected   = cv2.subtract(background, gray_smooth)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)


# Thresholding 

def threshold_to_black_particles(
    corrected_norm: np.ndarray,
    method: str = "percentile",
    percentile_keep: float = 99.5,
) -> tuple[float, np.ndarray]:
    if method == "percentile":
        T_used = float(np.percentile(corrected_norm, percentile_keep))
    elif method == "otsu":
        T_used, _ = cv2.threshold(corrected_norm, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        T_used = float(T_used)
    else:
        raise ValueError("method must be 'percentile' or 'otsu'")
    _, binary_black = cv2.threshold(corrected_norm, T_used, 255, cv2.THRESH_BINARY_INV)
    return T_used, binary_black


# Morphological cleanup 

def clean_binary_black(binary_black: np.ndarray) -> np.ndarray:
    binary_white = cv2.bitwise_not(binary_black)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_white, cv2.MORPH_OPEN,  kernel, iterations=1)
    closed = cv2.morphologyEx(opened,       cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.bitwise_not(closed)


# Physical feature helpers

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


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference for orientation angles (180° symmetry)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def extract_physical_features(
    contour: np.ndarray,
    frame_gray: np.ndarray,
    offset_xy: tuple[int, int] = (0, 0),
) -> dict:
    """
    Extract full physical feature set from a contour.

    Returns: area, circularity, orientation (fitEllipse angle),
             hu (7 log-scaled Hu moments), mean_intensity,
             com_dx/com_dy (intensity COM offset — professor's idea).
    """
    ox, oy = offset_xy
    area      = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    moments   = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-8:
        cx_local, cy_local = 0.0, 0.0
    else:
        cx_local = moments["m10"] / moments["m00"]
        cy_local = moments["m01"] / moments["m00"]

    circularity = (4.0 * math.pi * area / perimeter ** 2) if perimeter > 1e-8 else 0.0
    hu = _safe_log_hu(cv2.HuMoments(moments).flatten())

    orientation = 0.0
    if len(contour) >= 5:
        _, _, orientation = cv2.fitEllipse(contour)

    # Mean intensity inside contour mask
    mask = np.zeros(frame_gray.shape[:2], dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] += ox
    shifted[:, 0, 1] += oy
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    mean_intensity = float(cv2.mean(frame_gray, mask=mask)[0])

    # Intensity COM offset from bounding-box patch
    x_, y_, w_, h_ = cv2.boundingRect(contour)
    patch = frame_gray[y_ + oy: y_ + oy + h_, x_ + ox: x_ + ox + w_]
    com_dx, com_dy = _intensity_com_offset(patch)

    return {
        "area":           area,
        "circularity":    float(circularity),
        "orientation":    float(orientation),
        "hu":             hu,
        "mean_intensity": mean_intensity,
        "com_dx":         com_dx,
        "com_dy":         com_dy,
    }


# Region extraction 

def extract_regions_by_labeling(
    binary_black_clean: np.ndarray,
    frame_idx: int,
    frame_gray: np.ndarray,
) -> list[dict]:
    """
    Connected-component labeling.  Now also extracts physical features
    (orientation, Hu moments, intensity COM) for every blob so the
    tracker can use them for disambiguation.
    """
    binary_white = cv2.bitwise_not(binary_black_clean)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_white, connectivity=8
    )

    rows = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        cx, cy = centroids[label_id]

        component_mask = np.uint8(labels == label_id) * 255
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        phys = {}
        if cnts:
            phys = extract_physical_features(cnts[0], frame_gray)
        else:
            patch = frame_gray[y: y + h, x: x + w]
            com_dx, com_dy = _intensity_com_offset(patch)
            phys = {
                "area": float(area), "circularity": 0.0, "orientation": 0.0,
                "hu": np.zeros(7), "mean_intensity": float(frame_gray[y:y+h, x:x+w].mean()),
                "com_dx": com_dx, "com_dy": com_dy,
            }

        rows.append({
            "frame": int(frame_idx),
            "blob_id": int(label_id),
            "x": float(cx),
            "y": float(cy),
            "area": float(area),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
            **phys,
        })

    return rows


def filter_regions(
    rows: list[dict],
    min_area: float = 10,
    max_area: float = 5000,
) -> list[dict]:
    return [r for r in rows if min_area <= r["area"] <= max_area]


# Kalman track 

class KalmanTrack:
    """
    Per-particle Kalman filter tracker.
    State: [x, y, vx, vy] — Measurement: [x, y]
    """
    _next_id: int = 1

    def __init__(self, x: float, y: float, area: float,
                 phys: dict | None = None,
                 track_id: int | None = None):
        if track_id is not None:
            self.track_id = track_id
        else:
            self.track_id = KalmanTrack._next_id
            KalmanTrack._next_id += 1

        self.area_history: deque = deque(maxlen=15)
        self.area_history.append(area)

        # Kalman filter — constant-velocity model
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 5e-3
        self.kf.processNoiseCov[3, 3] = 5e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 1.0
        self.kf.statePost           = np.array([[x], [y], [0.], [0.]], dtype=np.float32)

        # Lifecycle
        self.hits:        int  = 1
        self.missed:      int  = 0
        self.total_hits:  int  = 1
        self.confirmed:   bool = False
        self.last_x:      float = x
        self.last_y:      float = y

        # Physical appearance — updated each frame for cost matrix
        p = phys or {}
        self.circularity:    float      = float(p.get("circularity", 0.0))
        self.orientation:    float      = float(p.get("orientation", 0.0))
        self.hu:             np.ndarray = np.asarray(p.get("hu", np.zeros(7)), dtype=float)
        self.mean_intensity: float      = float(p.get("mean_intensity", 0.0))
        self.com_dx:         float      = float(p.get("com_dx", 0.0))
        self.com_dy:         float      = float(p.get("com_dy", 0.0))

        # Trajectory history for drawing
        self.history: list[tuple[float, float]] = [(x, y)]

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        self.last_x = float(pred[0])
        self.last_y = float(pred[1])
        return self.last_x, self.last_y

    def update(self, x: float, y: float, area: float, phys: dict | None = None) -> None:
        meas = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(meas)
        s = self.kf.statePost
        self.last_x = float(s[0])
        self.last_y = float(s[1])
        self.area_history.append(area)
        self.hits       += 1
        self.total_hits += 1
        self.missed      = 0
        self.history.append((self.last_x, self.last_y))

        # Update appearance features with latest detection
        if phys:
            self.circularity = float(phys.get("circularity", self.circularity))
            self.orientation = float(phys.get("orientation", self.orientation))
            self.hu = np.asarray(phys.get("hu", self.hu), dtype=float)
            self.mean_intensity = float(phys.get("mean_intensity", self.mean_intensity))
            self.com_dx = float(phys.get("com_dx", self.com_dx))
            self.com_dy = float(phys.get("com_dy", self.com_dy))

    def mark_missed(self) -> None:
        self.hits = 0
        self.missed += 1
        self.history.append((self.last_x, self.last_y))

    @property
    def expected_area(self) -> float:
        return float(np.median(self.area_history)) if self.area_history else 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.last_x, self.last_y


# Re-ID graveyard 

class DeadTrackRecord:
    """Snapshot kept after a track dies, used for Re-ID resurrection."""
    def __init__(self, track_id: int, x: float, y: float,
                 expected_area: float, died_at_frame: int):
        self.track_id      = track_id
        self.x             = x
        self.y             = y
        self.expected_area = expected_area
        self.died_at_frame = died_at_frame


# Multi-particle tracker 

class MultiParticleTracker:
    """
    Hungarian + Kalman + Re-ID graveyard tracker.

    Cost matrix now combines:
      - Euclidean distance (primary gate)
      - Area similarity
      - Circularity similarity
      - Hu moment distance
      - Mean intensity difference
      - Intensity COM offset distance  ← professor's physical fingerprint
      - Orientation similarity
    """

    def __init__(
        self,
        max_dist:       float = 50.0,
        reid_dist:      float = 80.0,
        max_missed:     int   = 10,
        min_hits:       int   = 3,
        area_weight:    float = 0.10,
        reid_area_tol:  float = 0.50,
        reid_ttl:       int   = 30,
        # appearance weights in cost matrix
        w_circularity:  float = 0.05,
        w_hu:           float = 0.10,
        w_intensity:    float = 0.05,
        w_com:          float = 0.15,   # intensity COM — key new discriminator
        w_orientation:  float = 0.05,
    ):
        self.max_dist = max_dist
        self.reid_dist = reid_dist
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.area_weight = area_weight
        self.reid_area_tol = reid_area_tol
        self.reid_ttl = reid_ttl
        self.w_circularity = w_circularity
        self.w_hu = w_hu
        self.w_intensity = w_intensity
        self.w_com = w_com
        self.w_orientation = w_orientation

        self.tracks: list[KalmanTrack]    = []
        self._graveyard: list[DeadTrackRecord] = []
        self._frame_idx: int                   = 0

    # cost matrices 

    def _build_cost_matrix(
        self,
        detections: list[dict],
        predicted_positions: list[tuple[float, float]],
    ) -> np.ndarray:
        INF = 1e9
        cost = np.full((len(predicted_positions), len(detections)), INF)

        for ti, (px, py) in enumerate(predicted_positions):
            tr = self.tracks[ti]
            for di, det in enumerate(detections):
                dx   = det["x"] - px
                dy   = det["y"] - py
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist > self.max_dist:
                    continue

                # Motion cost (primary)
                motion_c = dist

                # Area cost — FIX: normalise by max to avoid div-by-zero
                area_c = self.area_weight * abs(det["area"] - tr.expected_area) / max(
                    det["area"], tr.expected_area, 1.0
                )

                # Circularity
                circ_c = self.w_circularity * abs(
                    det.get("circularity", 0) - tr.circularity
                )

                # Hu moments (mean absolute difference of log-scaled values)
                hu_det = np.asarray(det.get("hu", tr.hu), dtype=float)
                hu_c   = self.w_hu * float(np.mean(np.abs(hu_det - tr.hu)))

                # Mean intensity
                int_c  = self.w_intensity * abs(
                    det.get("mean_intensity", tr.mean_intensity) - tr.mean_intensity
                ) / 255.0

                # Intensity COM offset — professor's physical fingerprint
                # Euclidean distance in the normalised COM space (range ~0..1)
                com_c  = self.w_com * math.hypot(
                    det.get("com_dx", tr.com_dx) - tr.com_dx,
                    det.get("com_dy", tr.com_dy) - tr.com_dy,
                )

                # Orientation (handles 180° symmetry)
                ori_c  = self.w_orientation * _angle_diff_deg(
                    det.get("orientation", tr.orientation), tr.orientation
                ) / 90.0

                cost[ti, di] = motion_c + area_c + circ_c + hu_c + int_c + com_c + ori_c

        return cost

    def _build_reid_cost_matrix(self, detections: list[dict]) -> np.ndarray:
        INF = 1e9
        cost = np.full((len(self._graveyard), len(detections)), INF)
        for gi, rec in enumerate(self._graveyard):
            for di, det in enumerate(detections):
                dist = float(math.hypot(det["x"] - rec.x, det["y"] - rec.y))
                if dist > self.reid_dist:
                    continue
                if rec.expected_area > 0:
                    ratio = abs(det["area"] - rec.expected_area) / rec.expected_area
                    if ratio > self.reid_area_tol:
                        continue
                cost[gi, di] = dist + self.area_weight * abs(det["area"] - rec.expected_area)
        return cost

    # main update 

    def update(self, detections: list[dict]) -> list[dict]:
        """
        One tracking step.  detections must contain x, y, area and optionally
        the physical feature fields (circularity, orientation, hu,
        mean_intensity, com_dx, com_dy).
        """
        self._frame_idx += 1
        predicted_positions = [t.predict() for t in self.tracks]

        matched_track_idx: set[int] = set()
        matched_det_idx:   set[int] = set()

        if self.tracks and detections:
            cost = self._build_cost_matrix(detections, predicted_positions)
            for ti, di in zip(*linear_sum_assignment(cost)):
                if cost[ti, di] < 1e8:
                    det = detections[di]
                    self.tracks[ti].update(
                        det["x"], det["y"], det["area"],
                        phys=det,   
                    )
                    matched_track_idx.add(ti)
                    matched_det_idx.add(di)

        # Mark missed / move to graveyard
        surviving: list[KalmanTrack] = []
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idx:
                track.mark_missed()
            if track.missed > self.max_missed:
                if track.confirmed:
                    self._graveyard.append(DeadTrackRecord(
                        track_id=track.track_id, x=track.last_x, y=track.last_y,
                        expected_area=track.expected_area,
                        died_at_frame=self._frame_idx,
                    ))
            else:
                surviving.append(track)
        self.tracks = surviving

        # Re-ID from graveyard
        unmatched = [(di, det) for di, det in enumerate(detections)
                     if di not in matched_det_idx]
        reid_matched_grave: set[int] = set()

        if unmatched and self._graveyard:
            reid_cost = self._build_reid_cost_matrix([d for _, d in unmatched])
            for gi, local_di in zip(*linear_sum_assignment(reid_cost)):
                if reid_cost[gi, local_di] >= 1e8:
                    continue
                rec               = self._graveyard[gi]
                original_di, det  = unmatched[local_di]
                resurrected = KalmanTrack(
                    det["x"], det["y"], det["area"],
                    phys=det, track_id=rec.track_id,
                )
                resurrected.confirmed  = True
                resurrected.total_hits = 1
                self.tracks.append(resurrected)
                matched_det_idx.add(original_di)
                reid_matched_grave.add(gi)

        for gi in sorted(reid_matched_grave, reverse=True):
            self._graveyard.pop(gi)

        # Brand-new tracks for remaining unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_idx:
                self.tracks.append(KalmanTrack(
                    det["x"], det["y"], det["area"], phys=det,
                ))

        # Confirm new tracks
        for track in self.tracks:
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True

        # Expire old graveyard records
        self._graveyard = [
            r for r in self._graveyard
            if (self._frame_idx - r.died_at_frame) <= self.reid_ttl
        ]

        # Return confirmed track states (includes physical features for CSV)
        return [
            {
                "track_id": t.track_id,
                "x": t.last_x,
                "y": t.last_y,
                "area": t.expected_area,
                "orientation": t.orientation,
                "circularity": t.circularity,
                "com_dx": t.com_dx,
                "com_dy": t.com_dy,
                "hits": t.hits,
                "missed": t.missed,
                "total_hits": t.total_hits,
            }
            for t in self.tracks if t.confirmed
        ]

    def reset(self) -> None:
        self.tracks = []
        self._graveyard = []
        self._frame_idx = 0
        KalmanTrack._next_id = 1


# Trajectory visualisation 

def _track_colour(track_id: int) -> tuple[int, int, int]:
    hue = int((track_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Keep legacy list as fallback for app colour-coding plots
_TRACK_COLORS = [_track_colour(i) for i in range(1, 21)]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return _track_colour(track_id)


def draw_trajectories_on_frame(
    frame_bgr: np.ndarray,
    active_tracks: list[KalmanTrack],
    detection_rows: list[dict] | None = None,
    trail_length: int = 60,
) -> np.ndarray:
    """
    Draw fading trajectory trails, current-position circles, and
    track-ID + orientation labels onto a copy of the frame.

    Args:
        frame_bgr:      BGR input frame (not modified in-place)
        active_tracks:  list of KalmanTrack from MultiParticleTracker
        detection_rows: optional raw detections for bounding-box overlay
        trail_length:   how many past positions to draw per track
    """
    out = frame_bgr.copy()

    # Raw detection bounding boxes (light cyan, thin)
    if detection_rows:
        for r in detection_rows:
            x, y, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)

    for tr in active_tracks:
        colour  = _track_colour(tr.track_id)
        history = tr.history[-trail_length:]

        # Fading trail: older segments are thinner
        for k in range(1, len(history)):
            p1 = (int(history[k - 1][0]), int(history[k - 1][1]))
            p2 = (int(history[k][0]),     int(history[k][1]))
            thickness = max(1, int((k / len(history)) * 2.5))
            cv2.line(out, p1, p2, colour, thickness, cv2.LINE_AA)

        cx, cy = int(tr.last_x), int(tr.last_y)

        # Current position: filled circle + ring
        cv2.circle(out, (cx, cy), 5, colour, -1,  cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 7, colour,  1,  cv2.LINE_AA)

        # Label: ID + orientation angle
        label = f"P{tr.track_id}  {tr.orientation:.0f}°"
        cv2.putText(out, label, (cx + 9, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

    return out


# Static plots 

def save_trajectory_plot(df_track: pd.DataFrame, out_path: Path) -> Path | None:
    df_plot = df_track.dropna(subset=["x", "y"]).copy()
    if df_plot.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    if "track_id" in df_plot.columns:
        for tid, grp in df_plot.groupby("track_id"):
            grp_s = grp.sort_values("frame")
            c = tuple(v / 255 for v in reversed(_track_colour(int(tid))))
            plt.plot(grp_s["x"], grp_s["y"],
                     marker="o", markersize=2, linewidth=1,
                     label=f"P{int(tid)}", color=c)
        plt.legend(fontsize=7, ncol=3, loc="best")
    else:
        plt.plot(df_plot["x"], df_plot["y"], marker="o", markersize=2, linewidth=1)
    plt.title("Trajectories: tracked particles over frames")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


def save_coordinate_plot(df_track: pd.DataFrame, out_path: Path) -> Path | None:
    df_plot = df_track.dropna(subset=["x", "y"]).copy()
    if df_plot.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    if "track_id" in df_plot.columns:
        for tid, grp in df_plot.groupby("track_id"):
            grp_s = grp.sort_values("frame")
            c = tuple(v / 255 for v in reversed(_track_colour(int(tid))))
            lbl = f"P{int(tid)}"
            axes[0].plot(grp_s["frame"], grp_s["x"], linewidth=1, label=lbl, color=c)
            axes[1].plot(grp_s["frame"], grp_s["y"], linewidth=1, label=lbl, color=c)
        axes[0].legend(fontsize=7, ncol=4, loc="best")
    else:
        axes[0].plot(df_plot["frame"], df_plot["x"], linewidth=1)
        axes[1].plot(df_plot["frame"], df_plot["y"], linewidth=1)
    axes[0].set_ylabel("x (px)")
    axes[0].set_title("x-coordinate over time (per particle)")
    axes[1].set_ylabel("y (px)")
    axes[1].set_xlabel("frame")
    axes[1].set_title("y-coordinate over time (per particle)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Core video processing 

def process_video_pipeline(
    video_path: Path,
    results_dir: Path,
    k_small: int = 5,
    K_big: int = 101,
    threshold_method: str = "percentile",
    percentile_keep: float = 99.5,
    min_area: float = 10,
    max_area: float = 5000,
    max_dist: float = 50.0,
    reid_dist: float = 80.0,
    max_missed: int = 10,
    min_hits: int = 3,
    area_weight: float = 0.1,
    reid_area_tol: float = 0.5,
    reid_ttl: int = 30,
    trail_length: int = 60,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Full multi-particle tracking pipeline.

    Returns:
        df          — one row per (frame, track_id)
        csv_path    — path to saved CSV
        video_path  — path to trajectory video with fading trails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video appears empty: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    traj_video_path = results_dir / "trajectory_video.mp4"
    results_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(traj_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    tracker = MultiParticleTracker(
        max_dist=max_dist, reid_dist=reid_dist, max_missed=max_missed,
        min_hits=min_hits, area_weight=area_weight,
        reid_area_tol=reid_area_tol, reid_ttl=reid_ttl,
    )

    all_rows: list[dict] = []

    try:
        for fid in range(frame_count):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            corrected_norm = preprocess_corrected_norm(gray, k_small=k_small, K_big=K_big)
            T_used, binary_black = threshold_to_black_particles(
                corrected_norm, method=threshold_method, percentile_keep=percentile_keep,
            )
            binary_clean = clean_binary_black(binary_black)

            # Pass frame_gray so physical features can be extracted per blob
            det_rows = extract_regions_by_labeling(binary_clean, fid, gray)
            det_rows = filter_regions(det_rows, min_area=min_area, max_area=max_area)

            track_results = tracker.update(det_rows)

            # Write annotated frame with fading trajectory trails
            annotated = draw_trajectories_on_frame(
                frame_bgr,
                tracker.tracks,          # all KalmanTrack objects (confirmed + tentative)
                detection_rows=det_rows,
                trail_length=trail_length,
            )
            cv2.putText(annotated, f"frame={fid}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)

            if track_results:
                for tr in track_results:
                    all_rows.append({
                        "frame": fid,
                        "track_id": tr["track_id"],
                        "x": tr["x"],
                        "y": tr["y"],
                        "area": tr["area"],
                        "orientation": tr["orientation"],
                        "circularity": tr["circularity"],
                        "com_dx": tr["com_dx"],
                        "com_dy": tr["com_dy"],
                        "hits": tr["hits"],
                        "missed": tr["missed"],
                        "thr": float(T_used),
                    })
            else:
                all_rows.append({
                    "frame": fid, "track_id": np.nan,
                    "x": np.nan, "y": np.nan, "area": np.nan,
                    "orientation": np.nan, "circularity": np.nan,
                    "com_dx": np.nan, "com_dy": np.nan,
                    "hits": 0, "missed": 0, "thr": float(T_used),
                })

            if progress_callback:
                progress_callback(fid + 1, frame_count)

    finally:
        cap.release()
        writer.release()

    df       = pd.DataFrame(all_rows)
    csv_path = results_dir / "tracking.csv"
    df.to_csv(csv_path, index=False)

    return df, csv_path, traj_video_path


# Public entry point 

def run_cv_pipeline(
    video_path: str | Path,
    output_dir: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
    max_dist: float = 50.0,
    reid_dist: float = 80.0,
    max_missed: int   = 10,
    min_hits: int   = 3,
    area_weight: float = 0.1,
    reid_area_tol: float = 0.5,
    reid_ttl: int   = 30,
    trail_length: int   = 60,
) -> dict:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    KalmanTrack._next_id = 1  # reset for fresh run

    df_track, csv_path, traj_video = process_video_pipeline(
        video_path=video_path,
        results_dir=results_dir,
        max_dist=max_dist, reid_dist=reid_dist, max_missed=max_missed,
        min_hits=min_hits, area_weight=area_weight,
        reid_area_tol=reid_area_tol, reid_ttl=reid_ttl,
        trail_length=trail_length,
        progress_callback=progress_callback,
    )

    trajectory_plot  = save_trajectory_plot(df_track,  results_dir / "trajectory_plot.png")
    coordinate_plot  = save_coordinate_plot(df_track,  results_dir / "coordinate_plots.png")

    num_frames    = int(df_track["frame"].nunique())
    unique_tracks = df_track["track_id"].dropna().unique()
    num_particles = int(len(unique_tracks))

    track_stats = []
    for tid in sorted(unique_tracks):
        grp = df_track[df_track["track_id"] == tid].dropna(subset=["x", "y"])
        track_stats.append({
            "track_id": int(tid),
            "num_frames_tracked": int(len(grp)),
            "first_frame": int(grp["frame"].min()) if len(grp) else -1,
            "last_frame": int(grp["frame"].max()) if len(grp) else -1,
        })

    frames_with_track = int(df_track.dropna(subset=["track_id"])["frame"].nunique())

    return {
        "tracking_df": df_track,
        "tracking_csv": str(csv_path),
        "trajectory_plot": str(trajectory_plot) if trajectory_plot else None,
        "coordinate_plot": str(coordinate_plot) if coordinate_plot else None,
        "trajectory_video": str(traj_video),
        "num_frames": num_frames,
        "num_particles": num_particles,
        "track_stats": track_stats,
        "tracking_success_ratio": frames_with_track / max(num_frames, 1),
        "params": {
            "k_small": 5, "K_big": 101,
            "threshold_method": "percentile", "percentile_keep": 99.5,
            "min_area": 10.0, "max_area": 5000.0,
            "max_dist": max_dist, "reid_dist": reid_dist,
            "max_missed": max_missed, "min_hits": min_hits,
            "area_weight": area_weight, "reid_area_tol": reid_area_tol,
            "reid_ttl": reid_ttl, "trail_length": trail_length,
        },
    }