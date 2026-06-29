from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import pandas as pd
import streamlit as st

from cv_pipeline import run_cv_pipeline

try:
    from yolo_pipeline import run_yolo_pipeline
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

UPLOAD_DIR = HERE / "demo_outputs" / "uploads"
OUTPUT_DIR = HERE / "demo_outputs" / "runs"

MODELS_DIR = PROJECT_ROOT / "code" / "models"

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="Particle Tracking Comparison", layout="wide")
st.title("Particle Tracking Comparison App")
st.write(
    "Upload an already-cropped video, run the CV pipeline (multi-particle) "
    "and one or more YOLO models, compare outputs, and download artifacts."
)


def init_session_state():
    defaults = {
        "uploaded_video_path":  None,
        "uploaded_video_name":  None,
        "video_hash":           None,
        "current_output_dir":   None,
        "cv_outputs":           None,
        "yolo_outputs":         {},
        "timings":              {},
        "summary_rows":         [],
        "has_results":          False,
        "last_run_config_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# Utility helpers

def compute_bytes_hash(file_bytes: bytes, length: int = 16) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:length]


def compute_config_hash(config: dict, length: int = 16) -> str:
    raw = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def save_uploaded_file(uploaded_file) -> tuple[Path, str]:
    file_bytes = uploaded_file.read()
    file_hash  = compute_bytes_hash(file_bytes)
    original   = Path(uploaded_file.name)
    safe_name  = f"{original.stem}_{file_hash}{original.suffix.lower()}"
    save_path  = UPLOAD_DIR / safe_name
    if not save_path.exists():
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    return save_path, file_hash


def safe_video_display(video_path: Path | str | None, title: str):
    st.subheader(title)
    if video_path and Path(video_path).exists():
        st.video(str(video_path))
    else:
        st.info(f"{title} is not available.")


def safe_image_display(image_path: Path | str | None, caption: str):
    if image_path and Path(image_path).exists():
        st.image(str(image_path), caption=caption, use_container_width=True)


def safe_download_button(file_path: Path | str | None, label: str, download_name: str):
    if not file_path:
        return
    file_path = Path(file_path)
    if not file_path.exists():
        return
    suffix = file_path.suffix.lower()
    mime = {"mp4": "video/mp4", "csv": "text/csv",
            "png": "image/png",  "jpg": "image/jpeg"}.get(suffix.lstrip("."),
                                                           "application/octet-stream")
    with open(file_path, "rb") as f:
        st.download_button(label=label, data=f, file_name=download_name, mime=mime)


# Cache loaders 

def load_cv_outputs(cv_output_dir: Path) -> dict | None:
    results_dir     = cv_output_dir / "results"
    csv_path        = results_dir / "tracking.csv"
    trajectory_video = results_dir / "trajectory_video.mp4"
    trajectory_plot  = results_dir / "trajectory_plot.png"
    coordinate_plot  = results_dir / "coordinate_plots.png"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    num_frames    = int(df["frame"].nunique()) if "frame" in df.columns else 0
    num_particles = int(df["track_id"].dropna().nunique()) if "track_id" in df.columns else 0
    frames_with   = int(df.dropna(subset=["track_id"])["frame"].nunique()) \
                    if "track_id" in df.columns else 0
    success_ratio = frames_with / num_frames if num_frames > 0 else 0.0

    track_stats: list[dict] = []
    if "track_id" in df.columns:
        for tid, grp in df.dropna(subset=["track_id"]).groupby("track_id"):
            grp_xy = grp.dropna(subset=["x", "y"])
            track_stats.append({
                "track_id":           int(tid),
                "num_frames_tracked": int(len(grp_xy)),
                "first_frame":        int(grp_xy["frame"].min()) if len(grp_xy) else -1,
                "last_frame":         int(grp_xy["frame"].max()) if len(grp_xy) else -1,
            })

    return {
        "tracking_df": df,
        "tracking_csv": str(csv_path),
        "trajectory_plot": str(trajectory_plot) if trajectory_plot.exists() else None,
        "coordinate_plot": str(coordinate_plot) if coordinate_plot.exists() else None,
        "trajectory_video": str(trajectory_video) if trajectory_video.exists() else None,
        "num_frames": num_frames,
        "num_particles": num_particles,
        "track_stats": track_stats,
        "tracking_success_ratio": success_ratio,
        "params": None,
        "loaded_from_cache": True,
    }


def load_yolo_outputs(
    yolo_model_output_dir: Path,
    uploaded_video_path: Path,
    model_name: str,
) -> dict | None:
    model_stem = Path(model_name).stem
    video_stem = uploaded_video_path.stem

    predicted_video = yolo_model_output_dir / f"{video_stem}_{model_stem}_yolo.mp4"
    trajectory_video = yolo_model_output_dir / f"{video_stem}_{model_stem}_trajectory.mp4"
    predictions_csv = yolo_model_output_dir / f"{video_stem}_{model_stem}_predictions.csv"
    tracking_csv = yolo_model_output_dir / f"{video_stem}_{model_stem}_tracking.csv"
    detection_count_plot = yolo_model_output_dir / f"{video_stem}_{model_stem}_detection_counts.png"
    trajectory_plot = yolo_model_output_dir / f"{video_stem}_{model_stem}_trajectory_plot.png"
    coordinate_plot = yolo_model_output_dir / f"{video_stem}_{model_stem}_coordinate_plot.png"

    if not predictions_csv.exists() and not predicted_video.exists():
        return None

    predictions_df = pd.read_csv(predictions_csv) if predictions_csv.exists() else pd.DataFrame()
    tracking_df = pd.read_csv(tracking_csv)    if tracking_csv.exists()    else pd.DataFrame()

    num_total_detections = int(len(predictions_df))
    num_frames_with_detections = (
        int(predictions_df["frame"].nunique())
        if not predictions_df.empty and "frame" in predictions_df.columns else 0
    )
    num_particles = (
        int(tracking_df["track_id"].nunique())
        if not tracking_df.empty and "track_id" in tracking_df.columns else 0
    )

    run_dir_candidates = list(yolo_model_output_dir.glob("*_prediction"))
    run_dir = str(run_dir_candidates[0]) if run_dir_candidates else str(yolo_model_output_dir)

    return {
        "predicted_video": str(predicted_video) if predicted_video.exists()  else None,
        "trajectory_video": str(trajectory_video) if trajectory_video.exists() else None,
        "predictions_csv": str(predictions_csv) if predictions_csv.exists()  else None,
        "tracking_csv": str(tracking_csv) if tracking_csv.exists()     else None,
        "predictions_df": predictions_df,
        "tracking_df": tracking_df,
        "detection_count_plot": str(detection_count_plot) if detection_count_plot.exists() else None,
        "trajectory_plot": str(trajectory_plot) if trajectory_plot.exists()  else None,
        "coordinate_plot": str(coordinate_plot) if coordinate_plot.exists()  else None,
        "run_dir": run_dir,
        "num_frames_with_detections": num_frames_with_detections,
        "num_total_detections": num_total_detections,
        "num_particles": num_particles,
        "params": None,
        "loaded_from_cache": True,
    }



def make_run_config(
    run_cv: bool,
    run_yolo: bool,
    selected_yolo_models: list[str],
    yolo_imgsz: int,
    yolo_conf: float,
) -> dict:
    return {
        "run_cv": run_cv,
        "run_yolo": run_yolo,
        "selected_yolo_models": selected_yolo_models,
        "cv_pipeline_mode": "multi_particle_hungarian_kalman_physical_features",
        "yolo_params": {"imgsz": yolo_imgsz, "conf": yolo_conf},
        "app_version": "v4",
    }


def build_summary_rows(cv_outputs: dict | None, yolo_outputs: dict[str, dict]) -> list[dict]:
    rows = []
    if cv_outputs is not None:
        rows.append({
            "method": "CV (multi-particle)",
            "frames": cv_outputs.get("num_frames"),
            "particles_tracked": cv_outputs.get("num_particles"),
            "frame_coverage": f"{cv_outputs.get('tracking_success_ratio', 0):.1%}",
            "cached": cv_outputs.get("loaded_from_cache", False),
        })
    for model_name, outputs in yolo_outputs.items():
        rows.append({
            "method": model_name,
            "frames": outputs.get("num_frames_with_detections"),
            "particles_tracked": outputs.get("num_particles", "—"),  # now populated
            "frame_coverage": "—",
            "cached": outputs.get("loaded_from_cache", False),
        })
    return rows


def persist_run_state(
    uploaded_video_path: Path, video_hash: str, current_output_dir: Path,
    cv_outputs: dict | None, yolo_outputs: dict[str, dict],
    timings: dict[str, float], summary_rows: list[dict], config_hash: str,
):
    st.session_state.update({
        "uploaded_video_path": str(uploaded_video_path),
        "uploaded_video_name": uploaded_video_path.name,
        "video_hash": video_hash,
        "current_output_dir": str(current_output_dir),
        "cv_outputs": cv_outputs,
        "yolo_outputs": yolo_outputs,
        "timings": timings,
        "summary_rows": summary_rows,
        "has_results": True,
        "last_run_config_hash": config_hash,
    })


def clear_run_state():
    st.session_state.update({
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "video_hash": None,
        "current_output_dir": None,
        "cv_outputs": None,
        "yolo_outputs": {},
        "timings": {},
        "summary_rows": [],
        "has_results": False,
        "last_run_config_hash": None,
    })


# Per-particle tracking preview (CV) 

def render_cv_tracking_preview(df: pd.DataFrame):
    st.write("CV tracking data preview")
    if "track_id" not in df.columns:
        st.dataframe(df.head(20), use_container_width=True)
        return
    track_ids = sorted(df["track_id"].dropna().unique().astype(int).tolist())
    if not track_ids:
        st.info("No confirmed tracks found.")
        return
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_id = st.selectbox(
            "View particle track", options=track_ids,
            format_func=lambda x: f"Particle {x}", key="cv_track_selector",
        )
    with col_info:
        grp = df[df["track_id"] == selected_id].dropna(subset=["x", "y"])
        st.caption(
            f"Particle {selected_id}: {len(grp)} frames tracked, "
            f"frames {int(grp['frame'].min())}–{int(grp['frame'].max())}"
            if len(grp) else f"Particle {selected_id}: no position data"
        )
    st.dataframe(df[df["track_id"] == selected_id].reset_index(drop=True),
                 use_container_width=True)


def render_yolo_tracking_preview(df_tracking: pd.DataFrame):
    """Per-particle selector for the YOLO tracking CSV (mirrors CV preview)."""
    st.write("YOLO tracking data preview")
    if df_tracking.empty or "track_id" not in df_tracking.columns:
        st.info("No tracking data available.")
        return
    track_ids = sorted(df_tracking["track_id"].dropna().unique().astype(int).tolist())
    if not track_ids:
        st.info("No confirmed tracks found.")
        return
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_id = st.selectbox(
            "View particle track", options=track_ids,
            format_func=lambda x: f"Particle {x}", key="yolo_track_selector",
        )
    with col_info:
        grp = df_tracking[df_tracking["track_id"] == selected_id].dropna(subset=["x", "y"])
        st.caption(
            f"Particle {selected_id}: {len(grp)} frames tracked, "
            f"frames {int(grp['frame'].min())}–{int(grp['frame'].max())}"
            if len(grp) else f"Particle {selected_id}: no position data"
        )
    st.dataframe(
        df_tracking[df_tracking["track_id"] == selected_id].reset_index(drop=True),
        use_container_width=True,
    )


# Results rendering 

def render_results_section(
    uploaded_video_path: Path,
    cv_outputs: dict | None,
    yolo_outputs: dict[str, dict],
    timings: dict[str, float],
    summary_rows: list[dict],
):
    st.markdown("---")
    st.header("Run summary")

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if timings:
        st.subheader("Timing")
        st.dataframe(
            pd.DataFrame([{"method": m, "elapsed_seconds": round(s, 2)}
                          for m, s in timings.items()]),
            use_container_width=True,
        )

    st.markdown("---")
    st.header("Results")

    # ── CV results ────────────────────────────────────────────────────────────
    if cv_outputs is not None:
        with st.expander("CV results (multi-particle)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                safe_video_display(cv_outputs.get("trajectory_video"), "CV trajectory video")
                safe_download_button(
                    cv_outputs.get("trajectory_video"),
                    "Download CV trajectory video",
                    f"{uploaded_video_path.stem}_cv_trajectory.mp4",
                )
                safe_download_button(
                    cv_outputs.get("tracking_csv"),
                    "Download CV tracking CSV",
                    f"{uploaded_video_path.stem}_cv_tracking.csv",
                )
            with col2:
                safe_image_display(cv_outputs.get("trajectory_plot"),  "CV trajectory plot")
                safe_image_display(cv_outputs.get("coordinate_plot"),  "CV coordinate plots")

            track_stats = cv_outputs.get("track_stats", [])
            if track_stats:
                st.write("Per-particle summary")
                st.dataframe(pd.DataFrame(track_stats), use_container_width=True)

            if "tracking_df" in cv_outputs and isinstance(cv_outputs["tracking_df"], pd.DataFrame):
                render_cv_tracking_preview(cv_outputs["tracking_df"])

            st.caption(
                f"Particles tracked: {cv_outputs.get('num_particles', 0)} | "
                f"Total frames: {cv_outputs.get('num_frames', 0)} | "
                f"Frame coverage: {cv_outputs.get('tracking_success_ratio', 0):.1%}"
            )
    else:
        st.info("CV pipeline was not run.")

    # YOLO results 
    if yolo_outputs:
        st.subheader("YOLO results")
        for model_name, outputs in yolo_outputs.items():
            with st.expander(f"YOLO: {model_name}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    # Two videos: Ultralytics annotated + trajectory
                    safe_video_display(
                        outputs.get("predicted_video"),
                        f"{model_name} — YOLO detections video",
                    )
                    safe_video_display(
                        outputs.get("trajectory_video"),
                        f"{model_name} — trajectory video",
                    )
                    safe_download_button(
                        outputs.get("predicted_video"),
                        f"Download {model_name} detections video",
                        f"{uploaded_video_path.stem}_{Path(model_name).stem}_yolo.mp4",
                    )
                    safe_download_button(
                        outputs.get("trajectory_video"),
                        f"Download {model_name} trajectory video",
                        f"{uploaded_video_path.stem}_{Path(model_name).stem}_trajectory.mp4",
                    )
                    safe_download_button(
                        outputs.get("predictions_csv"),
                        f"Download {model_name} detections CSV",
                        f"{uploaded_video_path.stem}_{Path(model_name).stem}_predictions.csv",
                    )
                    safe_download_button(
                        outputs.get("tracking_csv"),
                        f"Download {model_name} tracking CSV",
                        f"{uploaded_video_path.stem}_{Path(model_name).stem}_tracking.csv",
                    )
                with col2:
                    safe_image_display(
                        outputs.get("detection_count_plot"),
                        f"{model_name} detections per frame",
                    )
                    safe_image_display(
                        outputs.get("trajectory_plot"),
                        f"{model_name} trajectory plot",
                    )
                    safe_image_display(
                        outputs.get("coordinate_plot"),
                        f"{model_name} coordinate plot",
                    )

                # Per-particle tracking preview (mirrors CV)
                tracking_df = outputs.get("tracking_df")
                if tracking_df is not None and isinstance(tracking_df, pd.DataFrame) \
                        and not tracking_df.empty:
                    render_yolo_tracking_preview(tracking_df)
                elif "predictions_df" in outputs and isinstance(outputs["predictions_df"], pd.DataFrame):
                    st.write(f"{model_name} raw detections preview")
                    st.dataframe(outputs["predictions_df"].head(20), use_container_width=True)

                st.caption(
                    f"Particles tracked: {outputs.get('num_particles', '—')} | "
                    f"Frames with detections: {outputs.get('num_frames_with_detections', 0)} | "
                    f"Total detections: {outputs.get('num_total_detections', 0)}"
                )
    else:
        st.info("YOLO pipeline was not run.")


# Sidebar

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload already-cropped video", type=["mp4", "avi", "mov", "mkv"]
    )

    st.markdown("---")
    st.subheader("Pipelines")
    run_cv   = st.checkbox("Run CV pipeline",   value=True)
    run_yolo = st.checkbox("Run YOLO pipeline", value=True)
    use_cache = st.checkbox("Use cached results", value=True)

    st.markdown("---")
    st.subheader("YOLO settings")
    available_models = ["yolo26n_best.pt", "yolo26s_best.pt", "yolo26m_best.pt"]
    selected_yolo_models = st.multiselect(
        "YOLO models", options=available_models,
        default=["yolo26n_best.pt"] if run_yolo else [],
    )
    yolo_imgsz = st.number_input("Image size",  min_value=320, max_value=2048, value=1024, step=32)
    yolo_conf  = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)

    st.markdown("---")
    st.subheader("Tracker settings")
    max_distance  = st.slider("Max distance (px)",  10, 200, 50)
    max_misses    = st.slider("Max misses (frames)", 1,  30,   8)
    trail_length  = st.slider("Trail length (frames)", 10, 120, 60)

    st.markdown("---")
    run_button   = st.button("Run processing", type="primary")
    clear_button = st.button("Clear current results")


# Clear 

if clear_button:
    clear_run_state()
    st.rerun()


# Main 

if uploaded_file is None and not st.session_state["has_results"]:
    st.info("Upload a cropped video to begin.")
else:
    current_uploaded_video_path: Path | None = None
    current_video_hash:          str  | None = None

    if uploaded_file is not None:
        current_uploaded_video_path, current_video_hash = save_uploaded_file(uploaded_file)
    elif st.session_state["uploaded_video_path"] is not None:
        current_uploaded_video_path = Path(st.session_state["uploaded_video_path"])
        current_video_hash          = st.session_state["video_hash"]

    if current_uploaded_video_path and current_uploaded_video_path.exists():
        st.subheader("Original video")
        st.video(str(current_uploaded_video_path))

    if run_button:
        if current_uploaded_video_path is None:
            st.warning("Upload a video first."); st.stop()
        if not run_cv and not run_yolo:
            st.warning("Select at least one pipeline."); st.stop()
        if run_yolo and not selected_yolo_models:
            st.warning("YOLO enabled but no model selected."); st.stop()
        if run_yolo and not YOLO_AVAILABLE:
            st.error("yolo_pipeline.py or ultralytics not found."); st.stop()

        config      = make_run_config(run_cv, run_yolo, selected_yolo_models, yolo_imgsz, yolo_conf)
        config_hash = compute_config_hash(config)
        run_name    = f"{current_uploaded_video_path.stem}__{config_hash}"
        current_output_dir = OUTPUT_DIR / run_name
        current_output_dir.mkdir(parents=True, exist_ok=True)

        with open(current_output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        status_box   = st.empty()
        progress_bar = st.progress(0)
        timings: dict[str, float]     = {}
        cv_outputs:   dict | None     = None
        yolo_outputs: dict[str, dict] = {}

        total_steps     = int(run_cv) + (len(selected_yolo_models) if run_yolo else 0)
        completed_steps = 0

        # CV
        if run_cv:
            cv_output_dir = current_output_dir / "cv"
            cv_output_dir.mkdir(parents=True, exist_ok=True)
            cached_cv = load_cv_outputs(cv_output_dir) if use_cache else None

            if cached_cv is not None:
                status_box.info("Loaded CV results from cache.")
                cv_outputs = cached_cv
                timings["CV"] = 0.0
            else:
                status_box.info("Running CV pipeline (multi-particle + physical features)...")
                cv_prog_ph = st.empty()

                def cv_progress(done: int, total: int):
                    if total > 0:
                        cv_prog_ph.caption(f"CV: {done}/{total} frames ({done/total:.1%})")

                t0 = time.time()
                cv_outputs = run_cv_pipeline(
                    video_path=current_uploaded_video_path,
                    output_dir=cv_output_dir,
                    progress_callback=cv_progress,
                    max_dist=max_distance,
                    max_missed=max_misses,
                    trail_length=trail_length,
                )
                timings["CV"] = time.time() - t0
                cv_outputs["loaded_from_cache"] = False
                status_box.success(
                    f"CV done in {timings['CV']:.1f}s — "
                    f"{cv_outputs.get('num_particles', 0)} particle(s) tracked."
                )

            completed_steps += 1
            progress_bar.progress(int(100 * completed_steps / max(total_steps, 1)))

        # YOLO 
        if run_yolo:
            for model_name in selected_yolo_models:
                yolo_model_output_dir = current_output_dir / "yolo" / Path(model_name).stem
                yolo_model_output_dir.mkdir(parents=True, exist_ok=True)

                cached_yolo = load_yolo_outputs(
                    yolo_model_output_dir, current_uploaded_video_path, model_name,
                ) if use_cache else None

                if cached_yolo is not None:
                    status_box.info(f"Loaded YOLO ({model_name}) from cache.")
                    yolo_outputs[model_name] = cached_yolo
                    timings[model_name] = 0.0
                    completed_steps += 1
                    progress_bar.progress(int(100 * completed_steps / max(total_steps, 1)))
                    continue

                model_path = MODELS_DIR / model_name
                if not model_path.exists():
                    st.error(f"Model not found: {model_path}"); continue

                yolo_prog_ph = st.empty()

                def yolo_progress(stage: str, frac: float, _m=model_name):
                    yolo_prog_ph.caption(f"YOLO [{_m}] {stage} ({frac:.1%})")

                status_box.info(f"Running YOLO pipeline for {model_name}...")
                t0 = time.time()
                outputs = run_yolo_pipeline(
                    video_path=current_uploaded_video_path,
                    model_path=model_path,
                    output_dir=yolo_model_output_dir,
                    imgsz=yolo_imgsz, conf=yolo_conf,
                    max_distance=max_distance, max_misses=max_misses,
                    trail_length=trail_length,
                    progress_callback=yolo_progress,
                )
                timings[model_name] = time.time() - t0
                outputs["loaded_from_cache"] = False
                yolo_outputs[model_name] = outputs

                status_box.success(
                    f"YOLO ({model_name}) done in {timings[model_name]:.1f}s — "
                    f"{outputs.get('num_particles', 0)} particle(s) tracked."
                )
                completed_steps += 1
                progress_bar.progress(int(100 * completed_steps / max(total_steps, 1)))

        progress_bar.progress(100)
        status_box.success("Processing complete.")

        summary_rows = build_summary_rows(cv_outputs, yolo_outputs)
        persist_run_state(
            current_uploaded_video_path, current_video_hash or "",
            current_output_dir, cv_outputs, yolo_outputs,
            timings, summary_rows, config_hash,
        )

    # Render results
    if st.session_state["has_results"] and st.session_state["uploaded_video_path"]:
        render_results_section(
            uploaded_video_path=Path(st.session_state["uploaded_video_path"]),
            cv_outputs=st.session_state["cv_outputs"],
            yolo_outputs=st.session_state["yolo_outputs"],
            timings=st.session_state["timings"],
            summary_rows=st.session_state["summary_rows"],
        )