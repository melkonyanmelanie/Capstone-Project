"""
Part 3 of 3: Trajectory-level fragment merging — FREEZE-IDENTITY approach.

Phase 1 — IDENTITY ANCHOR per fragment
    Each fragment gets a "signature" computed from its first
    SIGNATURE_FRAMES frames: average (x, y, vx, vy). This signature
    is what defines the fragment's identity.

  Phase 2 — CLUSTER fragments by signature
    Two fragments belong to the same particle if their starting
    positions are close AND starting velocities are close. Order-
    independent and stable.

  Phase 3 — FREEZE: assign one particle_id per cluster
    Each cluster becomes one final particle. ID is frozen here.

  Phase 4 — POPULATE trajectories with consistency filter


Run (from project root, after refactor):
    python code/run_pipeline.py --only part3
or directly:
    python code/pipeline_part3.py
"""

import json
import os
import pickle
import csv
import shutil
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

# CONFIGURATION
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tracking_on_final_sr_videos_1"
DATA_DIR = PROJECT_ROOT / "data" / "final_inputs_sr"

FORCE_RERUN = bool(int(os.environ.get("FORCE_RERUN", "0")))

EXPECTED_PARTICLE_COUNTS = {
    "one_particle_video_sr": 1,
    "two_particles_video_sr": 2,
    "three_particles_video_sr": 3,
}

# Phase 1: signature
SIGNATURE_FRAMES = 50
SIGNATURE_VEL_WINDOW = 10

# Phase 2: clustering
SIGNATURE_POS_THRESHOLD = 30.0    
SIGNATURE_VEL_THRESHOLD = 5.0     

# Phase 4: per-frame consistency filter
CONSISTENCY_MAX_DIST = 25.0       
CONSISTENCY_MAX_GAP = 60          

ANCHOR_MIN_SEPARATION = 22.0
ANCHOR_TRAJ_COLLAPSE_DIST = 8.0
ANCHOR_TRAJ_COLLAPSE_FRAC = 0.5
ANCHOR_CROSS_BIAS = 30.0

# Smoothing
KALMAN_Q = 1e-3
KALMAN_R = 15.0
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3

TAIL_LENGTH = 40
PALETTE = [
    (255, 60, 60), (60, 255, 60), (60, 100, 255), (255, 220, 0),
    (0, 220, 255), (255, 60, 220), (180, 60, 255), (255, 140, 0),
    (0, 255, 180), (140, 255, 0), (255, 160, 100), (100, 200, 255),
    (200, 100, 255), (255, 100, 150), (150, 255, 100),
]


# Phase 1: signature

def compute_signature(positions: dict) -> dict:
    """Compute identity signature from first SIGNATURE_FRAMES frames."""
    if not positions:
        return None
    sorted_frames = sorted(positions.keys())
    head = sorted_frames[:SIGNATURE_FRAMES]
    if len(head) < 2:
        f = head[0]
        x, y = positions[f]
        return {"x": x, "y": y, "vx": 0.0, "vy": 0.0,
                "first_frame": f, "n": len(head)}
    xs = [positions[f][0] for f in head]
    ys = [positions[f][1] for f in head]
    w = min(SIGNATURE_VEL_WINDOW, len(head) - 1)
    f_lo, f_hi = head[0], head[w]
    dt = f_hi - f_lo
    if dt > 0:
        vx = (positions[f_hi][0] - positions[f_lo][0]) / dt
        vy = (positions[f_hi][1] - positions[f_lo][1]) / dt
    else:
        vx, vy = 0.0, 0.0
    return {"x": float(np.mean(xs)), "y": float(np.mean(ys)),
            "vx": float(vx), "vy": float(vy),
            "first_frame": int(head[0]), "n": len(head)}


# Phase 2: cluster by signature

class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            if ri < rj:
                self.parent[rj] = ri
            else:
                self.parent[ri] = rj


def cluster_signatures(signatures: dict, expected_count: int = None) -> dict:
    """Cluster fragments by signature similarity. Returns {fid: cid}."""
    fids = sorted(signatures.keys())
    uf = UnionFind(fids)
    for i, a in enumerate(fids):
        sa = signatures[a]
        if sa is None:
            continue
        for b in fids[i+1:]:
            sb = signatures[b]
            if sb is None:
                continue
            dpos = np.hypot(sa["x"] - sb["x"], sa["y"] - sb["y"])
            dvel = np.hypot(sa["vx"] - sb["vx"], sa["vy"] - sb["vy"])
            if dpos <= SIGNATURE_POS_THRESHOLD and dvel <= SIGNATURE_VEL_THRESHOLD:
                uf.union(a, b)

    roots = sorted({uf.find(f) for f in fids})

    # Force expected count
    if expected_count is not None and len(roots) > expected_count:
        print(f"  Cluster count {len(roots)} > expected {expected_count}; "
              f"merging closest signatures")
        while len({uf.find(f) for f in fids}) > expected_count:
            cluster_sigs = {}
            for r in {uf.find(f) for f in fids}:
                members = [f for f in fids if uf.find(f) == r and signatures[f] is not None]
                if not members:
                    continue
                xs = [signatures[m]["x"] for m in members]
                ys = [signatures[m]["y"] for m in members]
                vxs = [signatures[m]["vx"] for m in members]
                vys = [signatures[m]["vy"] for m in members]
                cluster_sigs[r] = (np.mean(xs), np.mean(ys),
                                   np.mean(vxs), np.mean(vys))
            cluster_roots = sorted(cluster_sigs.keys())
            best = None
            best_score = np.inf
            for i, ra in enumerate(cluster_roots):
                xa, ya, vxa, vya = cluster_sigs[ra]
                for rb in cluster_roots[i+1:]:
                    xb, yb, vxb, vyb = cluster_sigs[rb]
                    dpos = np.hypot(xa-xb, ya-yb)
                    dvel = np.hypot(vxa-vxb, vya-vyb)
                    score = dpos + 5.0 * dvel
                    if score < best_score:
                        best_score = score
                        best = (ra, rb)
            if best is None:
                break
            uf.union(best[0], best[1])
            print(f"    forced-cluster-merge: {best[0]} ↔ {best[1]} "
                  f"(score={best_score:.2f})")
        roots = sorted({uf.find(f) for f in fids})

    if expected_count is not None and len(roots) < expected_count:
        print(f"  WARNING: cluster count {len(roots)} < expected "
              f"{expected_count}. Some particles were not detected as "
              f"distinct in the early frames.")

    root_to_cluster = {r: i for i, r in enumerate(roots)}
    return {f: root_to_cluster[uf.find(f)] for f in fids}


def cluster_signatures_anchored(signatures: dict, sam2_tracks: dict,
                                expected_count: int) -> dict:
    fids = sorted(signatures.keys())
    fids_with_sig = [f for f in fids if signatures[f] is not None
                     and sam2_tracks.get(f)]
    if not fids_with_sig:
        return {f: 0 for f in fids}

    if len(fids_with_sig) <= expected_count:
        anchors = list(fids_with_sig)
    else:
        scored = sorted(
            fids_with_sig,
            key=lambda f: -(len(sam2_tracks[f])
                            + (signatures[f]["n"] if signatures[f] else 0)),
        )
        anchors = []
        sep2 = ANCHOR_MIN_SEPARATION * ANCHOR_MIN_SEPARATION
        for f in scored:
            sa = signatures[f]
            ok = True
            for a in anchors:
                sb = signatures[a]
                if (sa["x"] - sb["x"]) ** 2 + (sa["y"] - sb["y"]) ** 2 < sep2:
                    ok = False; break
            if ok:
                anchors.append(f)
            if len(anchors) == expected_count:
                break
        if len(anchors) < expected_count:
            for f in scored:
                if f in anchors:
                    continue
                anchors.append(f)
                if len(anchors) == expected_count:
                    break

    anchors = anchors[:expected_count]
    anchor_to_cid = {a: i for i, a in enumerate(anchors)}

    assignment = {}
    for f in fids:
        if f in anchor_to_cid:
            assignment[f] = anchor_to_cid[f]
            continue
        sf = signatures[f]
        if sf is None:
            assignment[f] = 0
            continue
        best_a = anchors[0]; best_score = float("inf")
        for a in anchors:
            sa = signatures[a]
            if sa is None:
                continue
            dpos = np.hypot(sf["x"] - sa["x"], sf["y"] - sa["y"])
            dvel = np.hypot(sf["vx"] - sa["vx"], sf["vy"] - sa["vy"])
            score = dpos + 5.0 * dvel
            if score < best_score:
                best_score = score; best_a = a
        assignment[f] = anchor_to_cid[best_a]

    return assignment


def populate_trajectories_anchored(sam2_tracks: dict, assignment: dict,
                                   expected_count: int) -> dict:
    try:
        from scipy.optimize import linear_sum_assignment
        have_lap = True
    except Exception:
        have_lap = False

    clusters = defaultdict(list)
    for fid, cid in assignment.items():
        clusters[cid].append(fid)

    n_cl = expected_count
    seeds = {}
    for cid in range(n_cl):
        fids = clusters.get(cid, [])
        if not fids:
            continue
        anchor_fid = min(fids, key=lambda f: min(sam2_tracks[f].keys())
                         if sam2_tracks[f] else 10**9)
        if not sam2_tracks[anchor_fid]:
            continue
        f0 = min(sam2_tracks[anchor_fid].keys())
        x0, y0 = sam2_tracks[anchor_fid][f0]
        seeds[cid] = (f0, float(x0), float(y0))

    state = {}
    for cid, (f0, x0, y0) in seeds.items():
        state[cid] = {
            "last_frame": f0,
            "last_pos": (x0, y0),
            "velocity": (0.0, 0.0),
            "traj": {f0: (x0, y0)},
            "rejected": 0,
        }

    all_frames = set()
    for fid in assignment:
        all_frames.update(sam2_tracks[fid].keys())

    for f in sorted(all_frames):
        if not state:
            break
        cluster_obs = defaultdict(list)
        for fid, cid in assignment.items():
            if f in sam2_tracks[fid]:
                cluster_obs[cid].append((sam2_tracks[fid][f], fid))

        active_cids = sorted(state.keys())
        cands = []
        cand_native = []
        for cid in active_cids:
            for xy, fid in cluster_obs.get(cid, []):
                cands.append((xy, fid))
                cand_native.append(cid)

        if not cands:
            continue

        n_rows = len(active_cids)
        n_cols = len(cands)
        BIG = 1e9
        cost = np.full((n_rows, n_cols), BIG, dtype=np.float64)
        for i, cid in enumerate(active_cids):
            st = state[cid]
            gap = f - st["last_frame"]
            if gap <= 0 or gap > CONSISTENCY_MAX_GAP:
                continue
            px = st["last_pos"][0] + st["velocity"][0] * gap
            py = st["last_pos"][1] + st["velocity"][1] * gap
            for j, (xy, _fid) in enumerate(cands):
                bias = 0.0 if cand_native[j] == cid else ANCHOR_CROSS_BIAS
                d = float(np.hypot(xy[0] - px, xy[1] - py)) + bias
                if d <= CONSISTENCY_MAX_DIST + bias:
                    cost[i, j] = d

        if have_lap:
            row, col = linear_sum_assignment(cost)
        else:
            row = []; col = []; taken = set()
            order = np.argsort(cost.min(axis=1))
            for i in order:
                row_costs = cost[i]
                for j in np.argsort(row_costs):
                    if j in taken:
                        continue
                    if row_costs[j] >= BIG:
                        break
                    row.append(int(i)); col.append(int(j)); taken.add(int(j))
                    break

        chosen_pos = {}
        for i, j in zip(row, col):
            if cost[i, j] >= BIG:
                continue
            cid = active_cids[i]
            chosen_pos[cid] = cands[j][0]

        chosen_xy_set = []
        for cid, xy in list(chosen_pos.items()):
            ok = True
            for q_cid, q_xy in chosen_xy_set:
                if (xy[0] - q_xy[0]) ** 2 + (xy[1] - q_xy[1]) ** 2 < ANCHOR_TRAJ_COLLAPSE_DIST ** 2:
                    ok = False
                    break
            if ok:
                chosen_xy_set.append((cid, xy))
            else:
                state[cid]["rejected"] += 1
                del chosen_pos[cid]

        for cid in active_cids:
            if cid not in chosen_pos:
                continue
            xy = chosen_pos[cid]
            st = state[cid]
            gap = f - st["last_frame"]
            if gap <= 0:
                continue
            new_vx = (xy[0] - st["last_pos"][0]) / gap
            new_vy = (xy[1] - st["last_pos"][1]) / gap
            alpha = 0.3
            st["velocity"] = (alpha * new_vx + (1 - alpha) * st["velocity"][0],
                              alpha * new_vy + (1 - alpha) * st["velocity"][1])
            st["last_pos"] = xy
            st["last_frame"] = f
            st["traj"][f] = xy

    trajectories = {cid: state.get(cid, {"traj": {}})["traj"]
                    for cid in range(n_cl)}
    for cid in range(n_cl):
        if cid in state:
            print(f"    Anchor {cid}: {len(trajectories[cid])} accepted, "
                  f"{state[cid]['rejected']} rejected (anchor fragment="
                  f"{[f for f, c in assignment.items() if c == cid][0] if any(c == cid for c in assignment.values()) else 'n/a'})")
    return trajectories


def validate_identity_set(trajectories: dict, expected_count: int) -> dict:
    pids = sorted(trajectories.keys())
    coverages = {pid: len(trajectories[pid]) for pid in pids}
    pair_overlap_frac = {}
    pair_mean_dist = {}
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            pa = trajectories[a]; pb = trajectories[b]
            common = set(pa.keys()) & set(pb.keys())
            if not common:
                pair_overlap_frac[(a, b)] = 0.0
                pair_mean_dist[(a, b)] = float("inf")
                continue
            close = 0; dists = []
            for f in common:
                dx = pa[f][0] - pb[f][0]; dy = pa[f][1] - pb[f][1]
                d = float(np.hypot(dx, dy))
                dists.append(d)
                if d < ANCHOR_TRAJ_COLLAPSE_DIST:
                    close += 1
            pair_overlap_frac[(a, b)] = close / max(1, len(common))
            pair_mean_dist[(a, b)] = float(np.mean(dists)) if dists else float("inf")

    collapsed_pairs = [(a, b) for (a, b), frac in pair_overlap_frac.items()
                       if frac >= ANCHOR_TRAJ_COLLAPSE_FRAC]
    return {
        "expected_count": int(expected_count),
        "final_count": int(len(pids)),
        "coverage_per_particle": {str(p): coverages[p] for p in pids},
        "pair_overlap_fraction": {f"{a}-{b}": round(v, 4)
                                   for (a, b), v in pair_overlap_frac.items()},
        "pair_mean_distance_px": {f"{a}-{b}": (round(v, 3) if np.isfinite(v) else None)
                                   for (a, b), v in pair_mean_dist.items()},
        "collapsed_pairs": [f"{a}-{b}" for (a, b) in collapsed_pairs],
        "any_collapse": bool(collapsed_pairs),
    }


def repair_collapsed_trajectories(trajectories: dict,
                                  validation: dict) -> dict:
    if not validation.get("any_collapse"):
        return trajectories
    print("    Repair: detected collapsed trajectory pair(s); "
          "removing duplicate points by anchor priority")
    pids = sorted(trajectories.keys())
    repaired = {pid: dict(pos) for pid, pos in trajectories.items()}
    for label in validation.get("collapsed_pairs", []):
        try:
            a_str, b_str = label.split("-")
            a = int(a_str); b = int(b_str)
        except ValueError:
            continue
        if a not in repaired or b not in repaired:
            continue
        pa = repaired[a]; pb = repaired[b]
        common = set(pa.keys()) & set(pb.keys())
        keep_for_a = a < b
        for f in common:
            dx = pa[f][0] - pb[f][0]; dy = pa[f][1] - pb[f][1]
            if dx * dx + dy * dy < ANCHOR_TRAJ_COLLAPSE_DIST ** 2:
                if keep_for_a:
                    pb.pop(f, None)
                else:
                    pa.pop(f, None)
    return repaired


# Phase 4: populate trajectories with consistency filter

def populate_trajectories(sam2_tracks: dict, assignment: dict) -> dict:
    """
    For each cluster, walk through frames accepting only consistent positions.
    Returns {particle_id: {frame: (x, y)}}.
    """
    clusters = defaultdict(list)
    for fid, cid in assignment.items():
        clusters[cid].append(fid)

    trajectories = {}
    for cid, fids in sorted(clusters.items()):
        per_frame = defaultdict(list)
        for fid in fids:
            for fi, xy in sam2_tracks[fid].items():
                per_frame[fi].append((xy, fid))

        if not per_frame:
            trajectories[cid] = {}
            continue

        sorted_frames = sorted(per_frame.keys())
        first_frame = sorted_frames[0]
        # Seed: average all candidates at the first frame
        xs = [c[0][0] for c in per_frame[first_frame]]
        ys = [c[0][1] for c in per_frame[first_frame]]
        seed_pos = (float(np.mean(xs)), float(np.mean(ys)))

        traj = {first_frame: seed_pos}
        last_frame = first_frame
        last_pos = seed_pos
        velocity = (0.0, 0.0)
        n_rejected = 0

        for f in sorted_frames[1:]:
            gap = f - last_frame
            if gap > CONSISTENCY_MAX_GAP:
                break
            px = last_pos[0] + velocity[0] * gap
            py = last_pos[1] + velocity[1] * gap
            best = None
            best_d = np.inf
            for (xy, fid) in per_frame[f]:
                d = np.hypot(xy[0] - px, xy[1] - py)
                if d < best_d:
                    best_d = d
                    best = xy
            if best is None or best_d > CONSISTENCY_MAX_DIST:
                n_rejected += 1
                continue
            traj[f] = best
            new_vx = (best[0] - last_pos[0]) / gap
            new_vy = (best[1] - last_pos[1]) / gap
            alpha = 0.3
            velocity = (alpha * new_vx + (1 - alpha) * velocity[0],
                        alpha * new_vy + (1 - alpha) * velocity[1])
            last_pos = best
            last_frame = f

        trajectories[cid] = traj
        print(f"    Particle {cid}: {len(traj)} accepted, {n_rejected} rejected "
              f"(from fragments {sorted(fids)})")

    return trajectories


def freeze_identities(sam2_tracks: dict,
                      expected_count: int = None) -> tuple[dict, dict]:
    if not sam2_tracks:
        return {}, {}

    fragment_ids = sorted(sam2_tracks.keys())
    print(f"  Input: {len(fragment_ids)} fragment(s) from SAM2")

    print(f"  Phase 1: computing identity signatures (first "
          f"{SIGNATURE_FRAMES} frames per fragment)...")
    signatures = {fid: compute_signature(sam2_tracks[fid]) for fid in fragment_ids}
    for fid in fragment_ids:
        s = signatures[fid]
        if s is None:
            print(f"    Fragment {fid}: EMPTY")
        else:
            print(f"    Fragment {fid}: starts at frame {s['first_frame']}, "
                  f"pos=({s['x']:.1f}, {s['y']:.1f}), "
                  f"vel=({s['vx']:+.2f}, {s['vy']:+.2f})")

    use_anchored = (expected_count is not None
                    and expected_count >= 3
                    and len([f for f in fragment_ids if signatures[f] is not None]) >= 1)

    if use_anchored:
        print(f"  Phase 2: ANCHORED clustering (expected K={expected_count}); "
              f"identities will not be merged across anchors")
        assignment = cluster_signatures_anchored(signatures, sam2_tracks, expected_count)
    else:
        print("  Phase 2: clustering by signature...")
        assignment = cluster_signatures(signatures, expected_count)

    n_clusters = len(set(assignment.values()))
    print(f"    {n_clusters} cluster(s) formed:")
    cluster_to_frags = defaultdict(list)
    for fid, cid in assignment.items():
        cluster_to_frags[cid].append(fid)
    for cid in sorted(cluster_to_frags.keys()):
        print(f"      Cluster {cid}: fragments {sorted(cluster_to_frags[cid])}")

    if use_anchored:
        print(f"Phase 4: anchored exclusive per-frame assignment "
              f"(max_jump={CONSISTENCY_MAX_DIST}px, "
              f"cross-bias={ANCHOR_CROSS_BIAS}px, "
              f"collapse-dist={ANCHOR_TRAJ_COLLAPSE_DIST}px)...")
        trajectories = populate_trajectories_anchored(
            sam2_tracks, assignment, expected_count)
    else:
        print(f"Phase 4: populating trajectories with consistency filter "
              f"(max_jump={CONSISTENCY_MAX_DIST}px)...")
        trajectories = populate_trajectories(sam2_tracks, assignment)

    if expected_count is not None and expected_count >= 3:
        validation = validate_identity_set(trajectories, expected_count)
        print(f"validation: final={validation['final_count']}, "
              f"any_collapse={validation['any_collapse']}, "
              f"collapsed_pairs={validation['collapsed_pairs']}")
        if validation["any_collapse"]:
            trajectories = repair_collapsed_trajectories(trajectories, validation)
            validation = validate_identity_set(trajectories, expected_count)
            print(f"post-repair validation: final={validation['final_count']}, "
                  f"any_collapse={validation['any_collapse']}")
        log = {pid: sorted([fid for fid, cid in assignment.items() if cid == pid])
               for pid in trajectories.keys()}
        log["_identity_validation"] = validation
    else:
        log = {pid: sorted([fid for fid, cid in assignment.items() if cid == pid])
               for pid in trajectories.keys()}

    print(f"  Output: {len(trajectories)} final particle(s)")
    return trajectories, log


# Smoothing 

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


# CSV / Plots / Video

def save_csv(smoothed: dict, fps: float, out_dir: Path):
    out = out_dir / "tracks_merged.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "particle_id", "x", "y",
                         "displacement_px", "cumulative_displacement_px",
                         "entry_frame", "exit_frame"])
        for pid, positions in sorted(smoothed.items()):
            sorted_frames = sorted(positions.keys())
            if not sorted_frames:
                continue
            entry_f = sorted_frames[0]
            exit_f = sorted_frames[-1]
            cum_disp = 0.0
            prev_xy = None
            for fi in sorted_frames:
                cx, cy = positions[fi]
                disp = np.hypot(cx - prev_xy[0], cy - prev_xy[1]) if prev_xy else 0.0
                cum_disp += disp
                prev_xy = (cx, cy)
                writer.writerow([fi, round(fi / fps, 5), pid,
                                 round(cx, 3), round(cy, 3),
                                 round(disp, 4), round(cum_disp, 4),
                                 entry_f, exit_f])
    print(f"  CSV saved: {out}")


def save_plots(smoothed: dict, out_dir: Path):
    if not smoothed:
        return
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for pid, positions in sorted(smoothed.items()):
        sorted_frames = sorted(positions.keys())
        if not sorted_frames:
            continue
        c = colors[pid % len(colors)]
        xs = [positions[f][0] for f in sorted_frames]
        ys = [positions[f][1] for f in sorted_frames]
        axes[0].plot(sorted_frames, xs, color=c, label=f"P{pid}", linewidth=1.5)
        axes[1].plot(sorted_frames, ys, color=c, label=f"P{pid}", linewidth=1.5)
        axes[2].plot(xs, ys, color=c, label=f"P{pid}", linewidth=1.5)
        axes[2].plot(xs[0], ys[0], "o", color=c, markersize=6)
        axes[2].plot(xs[-1], ys[-1], "s", color=c, markersize=6)
    for ax, xl, yl, title in [
        (axes[0], "Frame", "X (px)", "X Position vs Frame (frozen IDs)"),
        (axes[1], "Frame", "Y (px)", "Y Position vs Frame (frozen IDs)"),
        (axes[2], "X (px)", "Y (px)", "XY Trajectories  (○=start  □=end)"),
    ]:
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    axes[2].invert_yaxis()
    plt.tight_layout()
    out = out_dir / "tracks_merged_plot.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"Plot saved: {out}")


def save_annotated_video(frames: list, smoothed: dict, fps: float, out_path: Path):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    frame_pids = defaultdict(list)
    for pid, positions in smoothed.items():
        for fi in positions:
            frame_pids[fi].append(pid)
    entry_frames = {pid: min(pos.keys()) for pid, pos in smoothed.items() if pos}
    exit_frames = {pid: max(pos.keys()) for pid, pos in smoothed.items() if pos}
    for fi, frame in enumerate(tqdm(frames, desc="  Writing video", unit="f")):
        canvas = frame.copy()
        for pid in frame_pids.get(fi, []):
            positions = smoothed[pid]
            if fi not in positions:
                continue
            color = PALETTE[pid % len(PALETTE)]
            cx, cy = positions[fi]
            icx, icy = int(round(cx)), int(round(cy))
            tail_frames = sorted(f for f in positions if f <= fi)[-TAIL_LENGTH:]
            for k in range(1, len(tail_frames)):
                alpha = k / len(tail_frames)
                tc = tuple(int(c * alpha) for c in color)
                p1 = (int(round(positions[tail_frames[k-1]][0])),
                      int(round(positions[tail_frames[k-1]][1])))
                p2 = (int(round(positions[tail_frames[k]][0])),
                      int(round(positions[tail_frames[k]][1])))
                cv2.line(canvas, p1, p2, tc, 1, cv2.LINE_AA)
            cv2.circle(canvas, (icx, icy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{pid}", (icx + 6, icy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            if fi == entry_frames.get(pid):
                cv2.circle(canvas, (icx, icy), 8, color, 2)
            if fi == exit_frames.get(pid):
                cv2.line(canvas, (icx-6, icy-6), (icx+6, icy+6), color, 2)
                cv2.line(canvas, (icx+6, icy-6), (icx-6, icy+6), color, 2)
        cv2.putText(canvas, f"Frame {fi:05d}  particles: {len(frame_pids.get(fi,[]))}",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(canvas)
    writer.release()
    print(f"  Annotated video saved: {out_path}")


# Main

def main():
    part1_dirs = sorted(OUTPUT_ROOT.glob("*/part1/meta.json"))
    if not part1_dirs:
        raise FileNotFoundError(
            f"No Part 1 results under {OUTPUT_ROOT}. Run pipeline_part1_track.py first."
        )

    print(f"Found {len(part1_dirs)} processed video(s)\n")

    for meta_path in part1_dirs:
        part1_dir = meta_path.parent
        stem = part1_dir.parent.name
        out_dir = part1_dir.parent / "part3"

        if FORCE_RERUN and out_dir.exists():
            print(f"  Wiping previous part3/ for {stem}")
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"─── {stem}")
        with open(part1_dir / "meta.json") as f:
            meta = json.load(f)
        with open(part1_dir / "sam2_tracks.pkl", "rb") as f:
            sam2_tracks = pickle.load(f)

        fps = meta["fps"]
        print(f"  {meta['orig_w']}x{meta['orig_h']}  {fps:.1f}fps  "
              f"{len(sam2_tracks)} fragments to evaluate")

        if not sam2_tracks:
            print("  No SAM2 tracks, skipping\n")
            continue

        expected = EXPECTED_PARTICLE_COUNTS.get(stem, None)
        if expected is not None:
            print(f"  Expected particle count: {expected}")

        print("  Freezing identities...")
        trajectories, log = freeze_identities(sam2_tracks, expected_count=expected)

        validation_payload = log.pop("_identity_validation", None)
        with open(out_dir / "merge_log.json", "w") as f:
            json.dump({str(k): v for k, v in log.items()}, f, indent=2)
        if validation_payload is not None:
            with open(out_dir / "identity_validation.json", "w") as f:
                json.dump(validation_payload, f, indent=2)

        print("  Smoothing trajectories (Kalman + Savitzky-Golay)...")
        smoothed = {pid: savgol_smooth(kalman_smooth(pos))
                    for pid, pos in trajectories.items()}

        save_csv(smoothed, fps, out_dir)
        save_plots(smoothed, out_dir)

        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            cand = DATA_DIR / f"{stem}{ext}"
            if cand.exists():
                video_path = cand
                break
        if video_path is not None:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            save_annotated_video(frames, smoothed, fps,
                                 out_dir / "tracked_video_merged.mp4")
        else:
            print(f"  WARNING: video not found for {stem}, skipping annotated video")
        print()

    print(f"All Part 3 outputs saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()



























































