# Particle Tracking in Liquid-Crystal Microscopy Videos

## 1. Project objective

Understanding particle transport in complex anisotropic media is an important problem in soft matter physics, particularly in liquid-crystal systems where particle motion is influenced by orientational order, defect structures, and external fields. These conditions create complex optical backgrounds and dynamic particle behavior, making reliable detection and long-term trajectory reconstruction difficult. This work proposes a data-driven pipeline for particle detection and tracking in free-surface liquid crystal (FSLC) microscopy videos. Particles are first detected per-frame using a fine-tuned YOLO object detector, and then, a small set of high-confidence detections initializes tracking for each expected particle. Each initialization is propagated into a continuous trajectory using SAM 2 - a video segmentation model that maintains identity-preserving object masks across frames. Then, motion-based post-processing stage merges fragments with consistent spatial and velocity behavior to recover stable particle identities. Evaluated on videos containing one to three particles, the method produced stable trajectories in single-particle and well-separated multi-particle cases, establishing a viable approach for data-driven particle detection, tracking, and trajectory reconstruction in complex liquid-crystal environments.

---

## 2. Required Project Resources (Before Installation)

To keep this repository lightweight and within GitHub's file size limits, several large project resources are **not included** in the repository:

* `data/`
* `outputs/`
* `code/models/sam2.1_hiera_base_plus.pt`

These resources are available here:

**Google Drive:**
https://drive.google.com/drive/folders/1-u1dR_ctr4bqWY2gJhFu9pNJzS7W1NTS?usp=sharing

Download the contents of the Google Drive folder and place them into the project so that the directory structure matches the following (or for more detailed review of the structure, please check **Section 3: Repository Structure**):

```text
DS299 Final Capstone Project Melanie Melkonyan/
│
├── code/
│   ├── models/
│   │   ├── yolo26m_best.pt
│   │   ├── yolo26n_best.pt
│   │   ├── yolo26s_best.pt
│   │   ├── RealESRGAN_x2plus.pth
│   │   └── sam2.1_hiera_base_plus.pt   ← download from Google Drive
│   └── ...
│
├── data/                               ← download from Google Drive
│   ├── raw_data/
│   ├── final_inputs/
│   └── final_inputs_sr/
│
├── outputs/                            ← download from Google Drive
│   └── tracking_on_final_sr_videos_1/
│
├── paper/
├── initial_methodology/
├── README.md
└── requirements.txt
```

Once these resources have been placed in the correct locations, continue with the installation instructions in **Section 4: Quick Start — install and reproduce**.

---


## 3. Repository structure

The following structure represents the complete project after the required external resources from Section 2 have been added.
```
DS299 Final Capstone Project Melanie Melkonyan/
├── code/                                       # production source code
│   ├── run_pipeline.py                         # one-command orchestrator
│   ├── pipeline_part1.py                       # YOLO + prompt selection + SAM2
│   ├── pipeline_part2.py                       # diagnostic raw-fragment renders
│   ├── pipeline_part3.py                       # identity-freeze merge -> final trajectories
│   ├── super_resolution/
│   │   └── real_esrgan_infer.py                # standalone full-video Real-ESRGAN
│   ├── metrics/
│   │   ├── manual_annotation_overlay.py        
│   │   ├── manual_annotation_overlay_two_particle.py
│   │   └── manual_annotation_overlay_three_particle.py
│   ├── intermediate_experiments_analysis_and_yolo_setup/
│   │   ├── EDA_and_initial_preprocessing.ipynb
│   │   ├── Data_Processing.ipynb
│   │   ├── Data_Preprocessing_Pipeline_Final_Version.ipynb
│   │   ├── Moving_object_detection_with_OpenCV.ipynb
│   │   ├── YOLO_setup.ipynb                    # YOLO dataset prep + training setup
│   │   ├── YOLO.ipynb                          # YOLO training runs and results
│   │   └── Data_Vizualization_and_Metrics.ipynb
│   ├── models/                                 # model checkpoints (mirror on Drive — see Section 7)
│   │   ├── yolo26m_best.pt                     # USED by pipeline_part1.py
│   │   ├── yolo26n_best.pt                     # used by initial_methodology
│   │   ├── yolo26s_best.pt                     # alternative size
│   │   ├── sam2.1_hiera_base_plus.pt           # USED by pipeline_part1.py
│   │   └── RealESRGAN_x2plus.pth               # USED by real_esrgan_infer.py
│   └── sam2/                                   # vendored SAM 2 
│       ├── sam2/                               
│       ├── setup.py, pyproject.toml, LICENSE, README.md
├── data/
│   ├── final_inputs/                           # 3 cropped raw mp4  (pipeline input)
│   ├── final_inputs_sr/                        # 3 super-resolved mp4  (pipeline input if SR skipped)
│   └── raw_data/                               # 13 .avi originals before cropping (evidence)
├── outputs/                                    # preserved pipeline results (do not delete to keep paper figures)
│   ├── tracking_on_final_sr_videos_1/<stem>/{part1,part2,part3}/
│   └── manual_validation*/                     # human-annotated validation results
├── initial_methodology/                        # exploratory baselines (CV + YOLO + Streamlit app)
│   ├── cv_pipeline.py
│   ├── yolo_pipeline.py
│   └── app.py
├── requirements.txt
├── paper/                                       
│   ├── figures/
│   ├── Data-Driven Modeling of Particle Transport and Tracking Evolution in Complex Media.tex
│   ├── references.bib
│   ├── DS299 Capstone Paper Melkonyan Melanie.pdf                           
├── .gitignore
└── README.md                                   
```

---

## 4. Quick Start — install and reproduce

The two extra `pip install` steps for `numpy` and `basicsr` BEFORE
`requirements.txt` are intentional and necessary:

- `basicsr == 1.4.2` is incompatible with `numpy >= 2.0` (it uses numpy APIs
  that were removed in numpy 2.0). Install `numpy < 2` first so the right
  version is pinned before any other dependency drags in newer numpy.
- `basicsr` uses an older `setup.py` build pattern that needs
  `--no-build-isolation` to install cleanly against the upgraded
  setuptools/wheel.

### Windows (PowerShell)

```powershell
# 1. Open PowerShell at the project root.
cd "C:\path\to\DS299 Final Capstone Project Melanie Melkonyan"

# 2. Create and activate a Python 3.10 virtual environment.
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip / setuptools / wheel.
python -m pip install --upgrade pip setuptools wheel

# 4. Install numpy 1.x BEFORE anything else.
python -m pip install --no-cache-dir "numpy<2"

# 5. Install basicsr separately with --no-build-isolation.
python -m pip install --no-cache-dir basicsr==1.4.2 --no-build-isolation

# 6. Install everything else from requirements.txt.
python -m pip install --no-cache-dir -r requirements.txt

# 7. Run the full pipeline.
python code\run_pipeline.py --force
```

### Linux / macOS (bash)

```bash
# 1. Open a terminal at the project root.
cd "/path/to/DS299 Final Capstone Project Melanie Melkonyan"

# 2. Create and activate a Python 3.10 virtual environment.
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip / setuptools / wheel.
pip install --upgrade pip setuptools wheel

# 4. Install numpy 1.x BEFORE anything else.
pip install --no-cache-dir "numpy<2"

# 5. Install basicsr separately with --no-build-isolation.
pip install --no-cache-dir basicsr==1.4.2 --no-build-isolation

# 6. Install everything else from requirements.txt.
pip install --no-cache-dir -r requirements.txt

# 7. Run the full pipeline.
python code/run_pipeline.py --force
```

**That's it.** Every trajectory CSV, every plot, and every annotated video
appears under `outputs/tracking_on_final_sr_videos_1/<stem>/{part1,part2,part3}/`.

If you just want to *inspect* the already-computed results that produced the
paper, skip step 7 — the `outputs/` folder ships with the full result tree.
Open any `tracks_merged.csv` or `tracked_video_merged.mp4` directly.

> **GPU note.** The pipeline auto-detects CUDA: if torch was installed with
> CUDA support and an NVIDIA GPU is present, the pipeline uses the GPU
> automatically. Otherwise it falls back to CPU (much slower — see §4.6).
> For best speed, install a CUDA-enabled torch BEFORE step 6:
>
> ```bash
> # Pick the URL matching YOUR CUDA version:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
> ```


Each video stem (e.g. `three_particles_video_sr`) gets its own subdirectory
under `outputs/tracking_on_final_sr_videos_1/`, with three sub-folders:

> **Final results - the trajectories cited in the paper - live in
> `outputs/tracking_on_final_sr_videos_1/<stem>/part3/`.** Specifically
> `tracks_merged.csv` (final per-particle trajectories), `tracks_merged_plot.png`
> (2D plot), and `tracked_video_merged.mp4` (annotated source video). The
> `part1/` and `part2/` subfolders contain detection / SAM 2 intermediates
> and raw-fragment diagnostics respectively - useful for debugging, not the
> final results.


---

## 5. Detailed reproduction

### 5.1 Why `--force` matters

The pipeline defaults to **resume mode**: if `outputs/.../part1/sam2_tracks.pkl`
already exists for a video, Part 1 skips that video and reuses the saved
intermediates. This is desirable for everyday development (rerun Part 3 only,
without re-doing the expensive SAM 2 pass) but it means that `python
code/run_pipeline.py` on a fresh clone will see the bundled `outputs/` folder
and do nothing.

To actually verify reproduction:

```bash
# Wipe the existing outputs so the pipeline computes everything from scratch:
# (Windows) Remove-Item -Recurse -Force outputs\tracking_on_final_sr_videos_1
# (Linux)   rm -rf outputs/tracking_on_final_sr_videos_1

python code/run_pipeline.py --force
```

`--force` propagates `FORCE_RERUN=1` into Part 1 and Part 3, which wipe their
per-video output directories before recomputing.

### 5.2 What the orchestrator does

For each `*.mp4` in `data/final_inputs/`:

1. **Decide whether to run super-resolution.**
   - If the filename ends in `_sr` *or* `--skip-sr` is passed → skip SR.
   - Otherwise → call `code/super_resolution/real_esrgan_infer.py` to write
     `data/final_inputs_sr/<stem>_sr.mp4`.

2. **Run Part 1** (`code/pipeline_part1.py`).
   YOLO detection across all frames → prompt selection (stable-window /
   tracklet resolver / GMM fallback) → SAM 2 segmentation tracking of each
   prompt fragment. Writes `outputs/.../<stem>/part1/`.

3. **Run Part 2** (`code/pipeline_part2.py`).
   Loads Part 1's pickled SAM 2 tracks, smooths them, writes a diagnostic
   CSV, a per-fragment 2D plot, and an annotated video showing every raw
   fragment trail. Diagnostic only - Part 3 reads Part 1 directly, not
   Part 2.

4. **Run Part 3** (`code/pipeline_part3.py`).
   Loads Part 1's tracks, computes each fragment's identity signature from
   its first 50 frames, clusters fragments into K particles, freezes those
   identities, then populates per-particle trajectories with a consistency
   filter. Writes the final `tracks_merged.csv`, merged trajectory plot,
   merged annotated video, and a merge log.

### 5.3 Two input modes

| Filename | Behavior |
|---|---|
| `<stem>.mp4` (no `_sr`) | Real-ESRGAN runs, then Parts 1/2/3 |
| `<stem>_sr.mp4` | SR skipped, Parts 1/2/3 run directly |
| any, with `--skip-sr` | SR skipped for every input |

The `data/final_inputs_sr/*.mp4` already contains the SR videos that produced
the paper.

### 5.4 All command-line flags

```
python code/run_pipeline.py
    [--input PATH]      single video; default: every mp4/avi/mov/mkv in data/final_inputs/
    [--skip-sr]         skip Real-ESRGAN for every input
    [--only STAGE]      run only one stage: sr | part1 | part2 | part3
    [--force]           force re-run; wipes existing per-video output dirs
                        (passes FORCE_RERUN=1 to Part 1 and Part 3)
```

### 5.5 Running stages individually

```bash
# Super-resolution only, single video:
python code/super_resolution/real_esrgan_infer.py \
    --input  data/final_inputs/three_particles_video.mp4 \
    --output data/final_inputs_sr/three_particles_video_sr.mp4 \
    --scale 2

# Part 1 only (reads everything in data/final_inputs_sr/):
python code/pipeline_part1.py

# Part 2 only (assumes Part 1 outputs exist):
python code/pipeline_part2.py

# Part 3 only (assumes Part 1 outputs exist):
python code/pipeline_part3.py
```

To force a re-run of a single direct invocation (without going through the
orchestrator), set the environment variable:

```bash
# Windows:
$env:FORCE_RERUN = "1"; python code\pipeline_part1.py

# Linux / macOS:
FORCE_RERUN=1 python code/pipeline_part1.py
```

### 5.6 Expected runtime (reference)

Times depend heavily on GPU model, CUDA / cuDNN version, and
disk speed. This might take from 1.5 hours to 25+ hours to run.

---

## 6. Data

| Folder | Contents | Role |
|---|---|---|
| `data/raw_data/` | 13 original `.avi` microscopy captures (`Video_2025...avi`) | Provenance evidence; not consumed by the pipeline |
| `data/final_inputs/` | `one_particle_video.mp4`, `two_particles_video.mp4`, `three_particles_video.mp4` | Cropped pipeline inputs; consumed by `run_pipeline.py` |
| `data/final_inputs_sr/` | `<stem>_sr.mp4` for each of the three above | Pre-computed Real-ESRGAN outputs; consumed by Parts 1/2/3 when SR is skipped |

**Where the data came from.** The 13 `.avi` files in `data/raw_data/` were
recorded on the project's microscope rig and manually cropped down to the
the files in `data/final_inputs/`.
The cropping step depends on human judgement (selecting the spatial region
of interest with stability), so it is not an automated re-runnable step; the cropped
outputs are checked in directly.

---

## 7. Output structure

Each video stem (e.g. `three_particles_video_sr`) gets its own subdirectory
under `outputs/tracking_on_final_sr_videos_1/`, with three sub-folders:

### `part1/` - detection, prompts, SAM 2 tracks

| File | Description |
|---|---|
| `meta.json` | Video metadata: fps, dimensions, frame count, particle count |
| `yolo_detections.json` | Per-frame raw YOLO detections (every box) |
| `gmm_particles.json` | Particle prompts after cleaning and selection |
| `selected_prompts.json` | Final prompt set fed into SAM 2 |
| `prompt_debug.png` | Visual sanity check of the chosen prompts |
| `sam2_tracks.pkl` | Pickled SAM 2 per-frame centroid tracks (consumed by Parts 2 and 3) |

### `part2/` - diagnostic raw fragments

| File | Description |
|---|---|
| `tracks.csv` | Per-frame, per-fragment x/y coordinates |
| `tracks_plot.png` | 2D trajectory plot of raw fragments |
| `tracked_video.mp4` | Source video with fragment trails drawn on |

### `part3/` - final results 

| File | Description |
|---|---|
| `tracks_merged.csv` | **Final trajectories.** One row per (frame, particle_id). |
| `tracks_merged_plot.png` | 2D plot of merged trajectories |
| `tracked_video_merged.mp4` | Source video with final colored per-particle trails |
| `merge_log.json` | Which fragments were assigned to which particle ID |
| `identity_validation.json` | Strict K = 3 separation + overlap checks (three-particle case only) |

---

## 8. Models and checkpoints

### Drive mirror

All YOLO checkpoints trained during the project are mirrored on Google Drive:

**https://drive.google.com/drive/folders/1c0NPYVKhu6bp74acYz2VThGRQG2dFGXz?usp=sharing**

The `code/models/` folder in this repo contains exactly the subset the
pipeline and demo apps need. All models checkpoints are included **directly** in this repository, only one of them is distributed through the external resources described in **Section 2**.

### What is in `code/models/`

| File | Purpose | Consumer |
|---|---|---|
| `yolo26m_best.pt` | YOLO11-m (medium) - production detector | `code/pipeline_part1.py` |
| `yolo26n_best.pt` | YOLO11-n (nano) - lighter alternative | `initial_methodology/yolo_pipeline.py`, `initial_methodology/app.py` |
| `yolo26s_best.pt` | YOLO11-s (small) - middle option | `initial_methodology/app.py` (model dropdown) |
| `sam2.1_hiera_base_plus.pt` | SAM 2.1 hiera base+ weights | `code/pipeline_part1.py` |
| `RealESRGAN_x2plus.pth` | Real-ESRGAN ×2+ weights | `code/super_resolution/real_esrgan_infer.py` |

### YOLO training and setup

The YOLO models were custom-trained on the project's microscopy dataset
(not off-the-shelf weights). The full setup, training, and evaluation are
documented in two Jupyter notebooks:

- **`code/intermediate_experiments_analysis_and_yolo_setup/YOLO_setup.ipynb`**
  — dataset preparation, annotation conversion, train / val / test split, and
  Ultralytics training configuration.
- **`code/intermediate_experiments_analysis_and_yolo_setup/YOLO.ipynb`**
  — training runs for the nano, small, and medium variants; validation
  metrics (mAP, precision, recall); and final model selection rationale.

The resulting `_best.pt` checkpoints are mirrored in `code/models/` (used by
the pipeline) and on the Drive link above.

### SAM 2

The vendored SAM 2 sits in `code/sam2/`. Part 1 imports `build_sam` and
`sam2_video_predictor` from `code/sam2/sam2/` after a brief `sys.path`
mangling, and it `os.chdir`s into that folder so Hydra can locate the
config file at `code/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`.
All of this happens automatically when `pipeline_part1.py` runs - no manual
setup is required.

### Real-ESRGAN

The `code/super_resolution/real_esrgan_infer.py` script is a clean
standalone wrapper around the official `RRDBNet` + `realesrgan.RealESRGANer`
combination. It uses the same model architecture and the same checkpoint
(`RealESRGAN_x2plus.pth`) that was validated during model selection, so
SR outputs are bit-identical at the pixel level.

---

## 9. Environment and dependencies

### Python version

Python 3.10 - matches the original RunPod environment that produced the
paper. Other 3.10.x patch versions are fine; 3.11/3.12 should work but are
not tested.

### GPU

A CUDA-capable NVIDIA GPU is strongly recommended. The reference hardware
was an NVIDIA A100; the pipeline has been tested on consumer 12 GB cards
with `--tile 256` passed to the SR script to avoid VRAM exhaustion. CPU-only
execution is technically possible but impractically slow for Real-ESRGAN
and SAM 2.

### Install

The install procedure is documented step-by-step in §3 (Quick Start). The
key constraints to be aware of:

- `numpy` must be `< 2.0` (because `basicsr == 1.4.2` uses numpy APIs that
  were removed in numpy 2.0).
- `basicsr == 1.4.2` should be installed with `--no-build-isolation` so it
  uses the upgraded setuptools / wheel from your environment.
- After those two are in place, the rest installs cleanly from
  `requirements.txt`.

If your CUDA / cuDNN combination requires a specific `torch` wheel (e.g.
CUDA 11.8 vs 12.1), install `torch` first with the right index URL from
[pytorch.org](https://pytorch.org/get-started/locally/), then install the
rest of the requirements.

### Direct dependencies (also enumerated in `requirements.txt`)

- `torch`, `torchvision` - deep learning core
- `numpy`, `scipy`, `scikit-learn` - numerics, smoothing, clustering
- `opencv-python`, `pillow` - image and video I/O
- `ultralytics` - YOLO inference
- `hydra-core`, `iopath` - SAM 2 configuration
- `matplotlib` - trajectory plots
- `tqdm` - progress bars
- `basicsr`, `realesrgan` - Real-ESRGAN inference
- `streamlit` - interactive demo app in `initial_methodology/`

---

## 10. Figures - which command produces which file

Per the rubric requirement that figures must be generated programmatically:

| Paper figure | Generated by | Output file |
|---|---|---|
| Final 2D trajectory plots (one per video) | `pipeline_part3.py` (called by `run_pipeline.py`) | `outputs/.../<stem>/part3/tracks_merged_plot.png` |
| Annotated trajectory videos | `pipeline_part3.py` | `outputs/.../<stem>/part3/tracked_video_merged.mp4` |
| Per-particle x/y coordinate plots | `pipeline_part3.py` (via Part 2 helper) | `outputs/.../<stem>/part2/tracks_plot.png` |
| Manual-vs-pipeline overlay plots | `code/metrics/manual_annotation_overlay*.py` | `outputs/manual_validation*/overlay_plot.png` |
| Pipeline prompt debug image | `pipeline_part1.py` | `outputs/.../<stem>/part1/prompt_debug.png` |

Every image in the results section is generated by one of the scripts above.

---

## 11. `code/metrics/` - annotation overlay tool

Three interactive scripts that overlay manually-annotated particle positions
on top of the pipeline's trajectory output, and produce per-frame error
plots plus summary statistics. This is how the validation section of the
paper was produced.

| Script | Use for |
|---|---|
| `manual_annotation_overlay.py` | 1-particle videos |
| `manual_annotation_overlay_two_particle.py` | 2-particle videos |
| `manual_annotation_overlay_three_particle.py` | 3-particle videos |

To launch:

```bash
python code/metrics/manual_annotation_overlay_three_particle.py
```

An OpenCV window opens with sampled frames; the annotator draws a box around
each particle and the script saves the annotated boxes, per-frame errors,
and overlay plots into `outputs/manual_validation*/`.

| Key | Action |
|---|---|
| Left-drag | Draw a box around a particle |
| Right-click | Undo the last box in this frame |
| `0`, `1`, `2` | Set the active identity (2- and 3-particle tools only) |
| `Space` | Commit boxes and advance to next frame |
| `S` | Skip the current frame |
| `R` | Reset all boxes in this frame |
| `Q` | Quit (progress saved) |

The exact error numbers depend on how carefully the annotator preserves
particle identities across frames, especially in moments where particles
cross or briefly occlude. The annotation files that produced the paper's
numbers are bundled in `outputs/manual_validation*/manual_boxes.csv`; the
scripts auto-detect these on launch, so re-running with the bundled CSVs in
place reproduces the paper's exact overlay plots without re-annotating.

---

## 12. `initial_methodology/` - exploratory baselines

This folder contains the project's earlier attempts at the tracking problem,
preserved as methodology evidence and as a comparison baseline. The
production pipeline (`code/run_pipeline.py`) replaced these approaches
because they gave visibly worse results on the same videos.

| File | What it does |
|---|---|
| `cv_pipeline.py` | Pure-OpenCV multi-particle tracker. Background subtraction → contour extraction → Kalman filter + Hungarian assignment + re-ID graveyard, with physical-feature cost terms (Hu moments, intensity-centroid offset, orientation, neighbour geometry). |
| `yolo_pipeline.py` | YOLO-only tracker. Per-frame YOLO inference plus the same physical-feature cost matrix and Hungarian assignment as the CV tracker. |
| `app.py` | Streamlit web UI that imports both pipelines and lets the user upload a video, run both side-by-side, and compare outputs. |

To launch the comparison app:

```bash
streamlit run "initial_methodology/app.py"
```


This folder is **not part of the reproducible production pipeline.** It is
preserved at the project root (rather than under `code/`) precisely to keep
that distinction visible.

---

## 13. `code/intermediate_experiments_analysis_and_yolo_setup/` - exploratory notebooks

Seven Jupyter notebooks documenting the development history of the project:

| Notebook | What's inside |
|---|---|
| `EDA_and_initial_preprocessing.ipynb` | Initial dataset exploration; per-video statistics; quality assessment |
| `Data_Processing.ipynb` | First-pass data processing experiments |
| `Data_Preprocessing_Pipeline_Final_Version.ipynb` | The final preprocessing pipeline that produced the cropped `data/final_inputs/` mp4 files |
| `Moving_object_detection_with_OpenCV.ipynb` | Pure-OpenCV motion-detection baseline (later superseded by YOLO + SAM 2) |
| `YOLO_setup.ipynb` | YOLO dataset preparation, annotation conversion, train / val / test split, Ultralytics training configuration |
| `YOLO.ipynb` | YOLO training runs (nano / small / medium variants), validation metrics, model selection |
| `Data_Vizualization_and_Metrics.ipynb` | Visualization helpers and per-frame metric computation |

These notebooks are documentation of how the project was built - they are
not invoked by `run_pipeline.py` and are not required for reproduction.


---

## 14. Notes

- **Default resume behaviour.** `python code/run_pipeline.py` without
  `--force` skips work where outputs already exist. Always delete
  `outputs/tracking_on_final_sr_videos_1/` and use `--force` if you want to
  verify true reproduction.
---

## 15. Citations and licenses

External libraries and pretrained models used in this project:

- **SAM 2** - Meta AI. Vendored in `code/sam2/`. Apache License 2.0. See
  `code/sam2/LICENSE`.
- **Real-ESRGAN** - Xintao Wang et al. Used via the `realesrgan` PyPI
  package. BSD 3-Clause License.
- **Ultralytics YOLO** - Ultralytics. Used via the `ultralytics` PyPI
  package. AGPL-3.0.
- **scikit-learn, OpenCV, NumPy, SciPy, Matplotlib** - standard scientific
  Python stack, each under its own permissive license (BSD-3 / Apache-2 /
  MIT).
- **Streamlit** - Streamlit Inc. Apache License 2.0.

All YOLO checkpoints in `code/models/` were trained as part of this project
on the project's own microscopy dataset; training is documented in
`YOLO_setup.ipynb` and `YOLO.ipynb`.

---

## 16. Contact

**Melanie Melkonyan**: mmelkonyanmelanie@gmail.com
