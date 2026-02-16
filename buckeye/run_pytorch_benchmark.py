#!/usr/bin/env python3
"""Run PyTorch Qwen3-ForcedAligner on Buckeye segments and save timestamps.

Usage:
  uv run python run_pytorch_benchmark.py --num-files 1000 --output pytorch_buckeye_1000.json
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

BENCHMARK_DIR = Path(__file__).parent / "buckeye-benchmark"
HF_DATASET_BASE = "https://huggingface.co/datasets/alexwengg/buckeye/resolve/main"


def download_buckeye(benchmark_dir: Path) -> None:
    """Download Buckeye dataset from HuggingFace if not present."""
    manifest_path = benchmark_dir / "manifest.json"
    audio_dir = benchmark_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Download manifest
    if not manifest_path.exists():
        typer.echo("Downloading manifest.json from HuggingFace...")
        urllib.request.urlretrieve(f"{HF_DATASET_BASE}/manifest.json", manifest_path)

    with open(manifest_path) as f:
        data = json.load(f)

    # Download missing audio files
    samples = data["samples"]
    missing = [s for s in samples if not (benchmark_dir / s["audio"]).exists()]
    if not missing:
        return

    typer.echo(f"Downloading {len(missing)} audio files from HuggingFace...")
    for i, sample in enumerate(missing):
        audio_path = benchmark_dir / sample["audio"]
        url = f"{HF_DATASET_BASE}/{sample['audio']}"
        urllib.request.urlretrieve(url, audio_path)
        if (i + 1) % 100 == 0 or i + 1 == len(missing):
            typer.echo(f"  {i + 1}/{len(missing)} downloaded")

    typer.echo("Buckeye dataset ready.")


def load_pytorch_aligner(model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"):
    """Load the PyTorch ForcedAligner."""
    import torch
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
    )
    return aligner


@app.command()
def benchmark(
    num_files: int = typer.Option(1000, "--num-files", help="Number of segments to process"),
    output: Path = typer.Option("pytorch_buckeye_1000.json", "--output", help="Output JSON"),
    model_id: str = typer.Option(
        "Qwen/Qwen3-ForcedAligner-0.6B", "--model-id", help="HF model ID"
    ),
    auto_download: bool = typer.Option(True, "--auto-download/--no-auto-download", help="Auto-download dataset from HuggingFace"),
) -> None:
    """Run PyTorch forced alignment on Buckeye segments."""
    if auto_download:
        download_buckeye(BENCHMARK_DIR)

    manifest = BENCHMARK_DIR / "manifest.json"
    with open(manifest) as f:
        data = json.load(f)

    samples = data["samples"][:num_files]
    typer.echo(f"Loaded {len(samples)} samples from {manifest}")

    # Load model
    typer.echo(f"\nLoading PyTorch model: {model_id}")
    aligner = load_pytorch_aligner(model_id)
    typer.echo("Model loaded.\n")

    results = []
    total_latency = 0.0

    for i, sample in enumerate(samples):
        audio_path = BENCHMARK_DIR / sample["audio"]
        transcript = sample["transcript"]

        start = time.perf_counter()
        try:
            alignment = aligner.align(
                audio=str(audio_path), text=transcript, language="English"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            typer.echo(f"  [{i+1}/{len(samples)}] {sample['id']}: FAILED - {e}")
            continue

        items = []
        for item in alignment[0]:
            items.append({
                "text": item.text,
                "start_time_ms": round(item.start_time * 1000, 1),
                "end_time_ms": round(item.end_time * 1000, 1),
            })

        results.append({
            "id": sample["id"],
            "speaker": sample["speaker"],
            "audio": sample["audio"],
            "transcript": transcript,
            "duration_s": sample["duration_s"],
            "ground_truth": sample["words"],
            "pytorch_alignments": items,
            "pytorch_latency_ms": round(elapsed_ms, 1),
        })

        total_latency += elapsed_ms

        if (i + 1) % 10 == 0 or i == 0:
            typer.echo(
                f"  [{i+1}/{len(samples)}] {sample['id']}: "
                f"{len(items)} words, {elapsed_ms:.0f}ms"
            )

    # Save results
    output_data = {
        "model_id": model_id,
        "dataset": "Buckeye Corpus v2.0",
        "num_samples": len(results),
        "total_pytorch_latency_ms": round(total_latency, 1),
        "samples": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2))

    typer.echo(f"\n=== Summary ===")
    typer.echo(f"Processed: {len(results)}/{len(samples)} segments")
    typer.echo(f"Total PyTorch time: {total_latency/1000:.1f}s")
    typer.echo(f"Avg per segment: {total_latency/len(results):.0f}ms")
    typer.echo(f"Output: {output}")


if __name__ == "__main__":
    app()
