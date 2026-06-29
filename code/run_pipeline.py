"""
run_pipeline.py
---------------
Unified one-command orchestrator for the capstone tracking pipeline.

Workflow per input video:

    [raw mp4]  --(Real-ESRGAN)-->  [<stem>_sr.mp4]
                                        |
                                        v
                                  pipeline_part1.py
                                        |
                                        v
                                  pipeline_part2.py  (diagnostics)
                                        |
                                        v
                                  pipeline_part3.py  (final trajectories)

Two input modes:

    (a) Raw video -> Real-ESRGAN -> Part 1 -> Part 2 -> Part 3
        triggered when filename does NOT end in "_sr".

    (b) Already-SR video -> skip Real-ESRGAN -> Part 1 -> Part 2 -> Part 3
        triggered when filename ends in "_sr" OR when --skip-sr is passed.

Usage:
    # Process every video in data/final_inputs/ (skip-SR auto-detected per file)
    python code/run_pipeline.py

    # Skip Real-ESRGAN for every input (treat all as already-SR)
    python code/run_pipeline.py --skip-sr

    # Single video
    python code/run_pipeline.py --input data/final_inputs/one_particle_video.mp4

    # Run only a single stage across all inputs
    python code/run_pipeline.py --only part2

    # Force re-run of part1/part3 (wipes existing per-video diagnostic dirs)
    python code/run_pipeline.py --force
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "final_inputs"
SR_OUTPUT_DIR = PROJECT_ROOT / "data" / "final_inputs_sr"

SR_SCRIPT = CODE_DIR / "super_resolution" / "real_esrgan_infer.py"
PART1_SCRIPT = CODE_DIR / "pipeline_part1.py"
PART2_SCRIPT = CODE_DIR / "pipeline_part2.py"
PART3_SCRIPT = CODE_DIR / "pipeline_part3.py"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
STAGE_ORDER = ("sr", "part1", "part2", "part3")


def _log(msg: str) -> None:
    print(f"[run_pipeline] {msg}", flush=True)


def _run_subprocess(cmd: list[str], env: dict[str, str] | None = None) -> None:
    _log("$ " + " ".join(str(c) for c in cmd))
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage failed with exit code {completed.returncode}: {' '.join(str(c) for c in cmd)}"
        )


def _is_sr_filename(path: Path) -> bool:
    return path.stem.endswith("_sr")


def _discover_inputs(input_arg: Path | None) -> list[Path]:
    if input_arg is not None:
        p = input_arg.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Input video not found: {p}")
        return [p]
    if not DEFAULT_INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Default input dir not found: {DEFAULT_INPUT_DIR}. "
            "Place raw videos there or pass --input."
        )
    found = sorted(
        p for p in DEFAULT_INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not found:
        raise FileNotFoundError(f"No videos found in {DEFAULT_INPUT_DIR}")
    return found


def _sr_output_for(input_video: Path) -> Path:
    SR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return SR_OUTPUT_DIR / f"{input_video.stem}_sr.mp4"


def _stage_sr(input_video: Path, force: bool) -> Path:
    """Returns the SR mp4 path."""
    if _is_sr_filename(input_video):
        _log(f"input already SR ({input_video.name}); skipping Real-ESRGAN")
        return input_video
    sr_path = _sr_output_for(input_video)
    cmd = [
        sys.executable, str(SR_SCRIPT),
        "--input", str(input_video),
        "--output", str(sr_path),
    ]
    if force:
        cmd.append("--force")
    _run_subprocess(cmd)
    return sr_path


def _child_env(force: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["FORCE_RERUN"] = "1" if force else "0"
    return env


def _stage_part1(force: bool) -> None:
    _run_subprocess([sys.executable, str(PART1_SCRIPT)], env=_child_env(force))


def _stage_part2(force: bool) -> None:
    _run_subprocess([sys.executable, str(PART2_SCRIPT)], env=_child_env(force))


def _stage_part3(force: bool) -> None:
    _run_subprocess([sys.executable, str(PART3_SCRIPT)], env=_child_env(force))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capstone pipeline orchestrator.")
    p.add_argument("--input", type=Path, default=None,
                   help="Single input video (default: every video in data/final_inputs/).")
    p.add_argument("--skip-sr", action="store_true",
                   help="Skip the Real-ESRGAN stage for every input (treat all as already-SR).")
    p.add_argument("--only", choices=STAGE_ORDER, default=None,
                   help="Run only one stage: sr | part1 | part2 | part3.")
    p.add_argument("--force", action="store_true",
                   help="Force re-run: passes FORCE_RERUN=1 to Part 1 / Part 3, "
                        "and --force to the SR script.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.only == "sr":
        inputs = _discover_inputs(args.input)
        for vid in inputs:
            _stage_sr(vid, force=args.force)
        return 0

    if args.only in ("part1", "part2", "part3"):
        if args.only == "part1":
            _stage_part1(args.force)
        elif args.only == "part2":
            _stage_part2(args.force)
        else:
            _stage_part3(args.force)
        return 0

    # Full pipeline.
    inputs = _discover_inputs(args.input)
    sr_paths: list[Path] = []
    for vid in inputs:
        if args.skip_sr or _is_sr_filename(vid):
            _log(f"SR skipped for {vid.name}")
            # If user passed a raw filename with --skip-sr, assume the matching
            # _sr file already exists in data/final_inputs_sr/.
            if not _is_sr_filename(vid):
                sr_paths.append(_sr_output_for(vid))
            else:
                sr_paths.append(vid)
            continue
        sr_paths.append(_stage_sr(vid, force=args.force))

    _log("super-resolution stage complete; SR videos:")
    for p in sr_paths:
        _log(f"  - {p}")


    _stage_part1(args.force)
    _stage_part2(args.force)
    _stage_part3(args.force)

    _log("pipeline complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  
        print(f"[run_pipeline] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
