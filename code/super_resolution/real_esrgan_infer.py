from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np



# basicsr <= 1.4.2 imports torchvision.transforms.functional_tensor, which
# torchvision >= 0.17 removed. Install a runtime shim BEFORE basicsr is
# imported, so `from basicsr.archs... import RRDBNet` does not blow up.

def _patch_basicsr_torchvision() -> None:
    try:
        import torchvision.transforms.functional_tensor  
        return
    except ModuleNotFoundError:
        pass
    import torchvision.transforms.functional as F
    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "code" / "models" / "RealESRGAN_x2plus.pth"


def _build_upsampler(weights_path: Path, scale: int, fp16: bool,
                     tile: int, tile_pad: int):
    _patch_basicsr_torchvision()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    import torch  

    if scale not in (2, 4):
        print(f"[real_esrgan_infer] scale={scale} unsupported, falling back to 2",
              file=sys.stderr)
        scale = 2

    net = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )

    # Auto-detect compute device: use CUDA if a GPU is available, else CPU.
    # On GPU machines, behavior is identical to the original gpu_id=0 setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half_effective = bool(fp16) and device.type == "cuda"  # half precision only on GPU
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights_path),
        model=net,
        tile=int(tile),
        tile_pad=int(tile_pad),
        pre_pad=0,
        half=half_effective,
        device=device,
    )
    return upsampler, scale


def _open_writer(out_path: Path, fps: float, frame_size: tuple[int, int]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    return writer


def super_resolve_video(
    input_path: Path,
    output_path: Path,
    weights_path: Path = DEFAULT_WEIGHTS,
    scale: int = 2,
    fp16: bool = False,
    tile: int = 0,
    tile_pad: int = 10,
    force: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    weights_path = Path(weights_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if output_path.exists() and not force:
        print(f"[real_esrgan_infer] output exists, skipping (use --force to overwrite): "
              f"{output_path}")
        return output_path

    upsampler, eff_scale = _build_upsampler(weights_path, scale, fp16, tile, tile_pad)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_size = (src_w * eff_scale, src_h * eff_scale)

    print(f"[real_esrgan_infer] {input_path.name}: {n_frames} frames @ {fps:.2f} fps, "
          f"{src_w}x{src_h} -> {dst_size[0]}x{dst_size[1]} (x{eff_scale})")

    writer = _open_writer(output_path, fps, dst_size)

    t0 = time.time()
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out_img, _ = upsampler.enhance(frame, outscale=eff_scale)
            if (out_img.shape[1], out_img.shape[0]) != dst_size:
                out_img = cv2.resize(out_img, dst_size, interpolation=cv2.INTER_LANCZOS4)
            writer.write(out_img)
            written += 1
            if written % 50 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed else 0.0
                print(f"  ... {written}/{n_frames or '?'} frames "
                      f"({rate:.2f} fps)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    rate = written / elapsed if elapsed else 0.0
    print(f"[real_esrgan_infer] done: wrote {written} frames in {elapsed:.1f}s "
          f"({rate:.2f} fps) -> {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone Real-ESRGAN video inference.")
    p.add_argument("--input", required=True, type=Path,
                   help="Input video file (mp4, avi, mov, mkv).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output video path (mp4).")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"Real-ESRGAN .pth weights (default: {DEFAULT_WEIGHTS}).")
    p.add_argument("--scale", type=int, default=2, choices=(2, 4),
                   help="SR scale factor (default: 2).")
    p.add_argument("--fp16", action="store_true",
                   help="Run model in half precision.")
    p.add_argument("--tile", type=int, default=0,
                   help="Tile size for low-VRAM machines (default 0 = no tiling).")
    p.add_argument("--tile-pad", type=int, default=10,
                   help="Tile padding (default 10).")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if --output already exists.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        super_resolve_video(
            input_path=args.input,
            output_path=args.output,
            weights_path=args.weights,
            scale=args.scale,
            fp16=args.fp16,
            tile=args.tile,
            tile_pad=args.tile_pad,
            force=args.force,
        )
    except Exception as exc:
        print(f"[real_esrgan_infer] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
