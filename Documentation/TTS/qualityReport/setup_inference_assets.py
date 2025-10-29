#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


# Download everything to the qualityReport directory
TTS_DIR = Path(__file__).resolve().parent


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fetch_voice_af_heart(dest: Path) -> Path:
    """Download af_heart voice pack to dest using huggingface_hub.

    dest should be qualityReport/voices/af_heart.pt
    """
    if dest.exists():
        print(f"[ok] Voice exists: {dest}")
        return dest
    ensure_dir(dest.parent)
    print("[dl] Downloading voice: hexgrad/Kokoro-82M: voices/af_heart.pt")
    src = Path(
        hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename="voices/af_heart.pt",
        )
    )
    shutil.copy2(src, dest)
    print(f"[ok] Wrote {dest}")
    return dest


def fetch_coreml_mlpackage(name: str, dest_dir: Path) -> Path:
    """Download a CoreML .mlpackage folder from HF into dest_dir.

    name: e.g. "kokoro_24_15s" (without .mlpackage)
    """
    pkg_dir = dest_dir / f"{name}.mlpackage"
    if pkg_dir.exists():
        print(f"[ok] CoreML exists: {pkg_dir}")
        return pkg_dir
    ensure_dir(dest_dir)
    pattern = f"{name}.mlpackage/**"
    print(f"[dl] Downloading {name}.mlpackage into {dest_dir}")
    # snapshot_download will materialize the folder structure under dest_dir
    snapshot_download(
        repo_id="FluidInference/kokoro-82m-coreml",
        allow_patterns=[pattern],
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
        max_workers=8,
    )
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Expected {pkg_dir} after download")
    print(f"[ok] Wrote {pkg_dir}")
    return pkg_dir


def fetch_onnx_model(onnx_model: Path) -> Optional[Path]:
    """Download ONNX model from official onnx-community HuggingFace repo.

    Downloads model.onnx (FP32, 326MB) from onnx-community/Kokoro-82M-v1.0-ONNX.
    This is the official ONNX export with correct input names (input_ids, style, speed).
    ONNX inference is optional - the quality report works fine without it.
    """
    if onnx_model.exists():
        print(f"[ok] ONNX model exists: {onnx_model}")
        return onnx_model

    print("[dl] Downloading official ONNX model from onnx-community...")
    try:
        src = Path(
            hf_hub_download(
                repo_id="onnx-community/Kokoro-82M-ONNX",
                filename="onnx/model.onnx",  # FP32 for best quality (326 MB)
            )
        )
        shutil.copy2(src, onnx_model)
        print(f"[ok] Wrote {onnx_model} ({onnx_model.stat().st_size / 1024 / 1024:.1f} MB)")
        return onnx_model
    except Exception as e:
        print(f"[skip] ONNX download failed: {e}")
        print("[info] ONNX inference is optional. CoreML and PyTorch comparisons will still work.")
        return None


def main() -> None:
    # Voice pack
    voice_pt = TTS_DIR / "voices" / "af_heart.pt"
    fetch_voice_af_heart(voice_pt)

    # CoreML packages
    fetch_coreml_mlpackage("kokoro_24_15s", TTS_DIR)
    fetch_coreml_mlpackage("kokoro_21_10s", TTS_DIR)
    # Optional v24 10s
    try:
        fetch_coreml_mlpackage("kokoro_24_10s", TTS_DIR)
    except Exception as e:
        print(f"[info] Optional kokoro_24_10s.mlpackage not fetched: {e}")

    # ONNX (optional - downloads from official onnx-community repo)
    fetch_onnx_model(TTS_DIR / "kokoro-v1.0.onnx")

    print("\n[ok] Assets ready. Try:")
    print("  cd Documentation/TTS/qualityReport && uv run python verify_all.py")


if __name__ == "__main__":
    main()

