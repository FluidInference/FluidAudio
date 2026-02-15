#!/usr/bin/env python3
"""Parity benchmark: PyTorch Qwen3-ForcedAligner vs CoreML conversion.

Runs the same audio+text through both PyTorch and CoreML, compares:
  - Per-word timestamp differences (start_time, end_time)
  - AAS: mean absolute boundary error (ms)
  - Max error (ms)
  - % boundaries within 20ms / 50ms tolerance
  - Latency comparison

Usage:
  uv run python compare-models.py --audio-dir /path/to/test-clean --num-files 10
  uv run python compare-models.py --audio-file audio.wav --text "hello world" --language English
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class AlignmentResult:
    """One word's alignment from either PyTorch or CoreML."""
    text: str
    start_time_ms: float
    end_time_ms: float


@dataclass
class ParityMetrics:
    """Comparison metrics between PyTorch and CoreML alignments."""
    num_words: int
    aas_ms: float  # mean absolute boundary error
    max_error_ms: float
    pct_within_20ms: float
    pct_within_50ms: float
    pytorch_latency_ms: float
    coreml_latency_ms: float
    speedup: float


def load_pytorch_aligner(model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"):
    """Load the PyTorch ForcedAligner for reference inference."""
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
    )
    return aligner


def run_pytorch_alignment(
    aligner, audio_path: str, text: str, language: str
) -> Tuple[List[AlignmentResult], float]:
    """Run PyTorch forced alignment, return results and latency in ms."""
    start = time.perf_counter()
    results = aligner.align(audio=audio_path, text=text, language=language)
    elapsed_ms = (time.perf_counter() - start) * 1000

    items = []
    for item in results[0]:
        items.append(AlignmentResult(
            text=item.text,
            start_time_ms=item.start_time * 1000,
            end_time_ms=item.end_time * 1000,
        ))
    return items, elapsed_ms


def compute_parity(
    ref: List[AlignmentResult],
    hyp: List[AlignmentResult],
    ref_latency_ms: float,
    hyp_latency_ms: float,
) -> ParityMetrics:
    """Compute parity metrics between reference (PyTorch) and hypothesis (CoreML)."""
    if len(ref) != len(hyp):
        typer.echo(f"  WARNING: word count mismatch: ref={len(ref)}, hyp={len(hyp)}")

    n = min(len(ref), len(hyp))
    errors = []
    for i in range(n):
        errors.append(abs(ref[i].start_time_ms - hyp[i].start_time_ms))
        errors.append(abs(ref[i].end_time_ms - hyp[i].end_time_ms))

    errors = np.array(errors)
    return ParityMetrics(
        num_words=n,
        aas_ms=float(np.mean(errors)),
        max_error_ms=float(np.max(errors)),
        pct_within_20ms=float(np.mean(errors <= 20.0) * 100),
        pct_within_50ms=float(np.mean(errors <= 50.0) * 100),
        pytorch_latency_ms=ref_latency_ms,
        coreml_latency_ms=hyp_latency_ms,
        speedup=ref_latency_ms / hyp_latency_ms if hyp_latency_ms > 0 else 0,
    )


def load_librispeech_samples(
    test_clean_dir: Path, num_files: int
) -> List[Tuple[Path, str]]:
    """Load audio+transcript pairs from LibriSpeech test-clean."""
    samples = []
    trans_files = sorted(test_clean_dir.rglob("*.trans.txt"))

    for trans_file in trans_files:
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                audio_id, text = parts
                audio_path = trans_file.parent / f"{audio_id}.flac"
                if audio_path.exists():
                    samples.append((audio_path, text))
                    if len(samples) >= num_files:
                        return samples

    return samples


def print_parity_report(metrics: ParityMetrics, label: str = "") -> None:
    """Print a formatted parity report."""
    prefix = f"[{label}] " if label else ""
    typer.echo(f"\n{prefix}Parity Report ({metrics.num_words} words):")
    typer.echo(f"  AAS (mean boundary error):  {metrics.aas_ms:.1f} ms")
    typer.echo(f"  Max boundary error:         {metrics.max_error_ms:.1f} ms")
    typer.echo(f"  Within 20ms:                {metrics.pct_within_20ms:.1f}%")
    typer.echo(f"  Within 50ms:                {metrics.pct_within_50ms:.1f}%")
    typer.echo(f"  PyTorch latency:            {metrics.pytorch_latency_ms:.0f} ms")
    typer.echo(f"  CoreML latency:             {metrics.coreml_latency_ms:.0f} ms")
    typer.echo(f"  Speedup:                    {metrics.speedup:.2f}x")


@app.command()
def compare(
    audio_file: Optional[Path] = typer.Option(
        None, "--audio-file", help="Single audio file for comparison"
    ),
    text: Optional[str] = typer.Option(
        None, "--text", help="Transcript for single file mode"
    ),
    language: str = typer.Option(
        "English", "--language", help="Language for alignment"
    ),
    audio_dir: Optional[Path] = typer.Option(
        None, "--audio-dir",
        help="LibriSpeech test-clean directory for batch comparison"
    ),
    num_files: int = typer.Option(
        10, "--num-files", help="Number of files for batch mode"
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen3-ForcedAligner-0.6B", "--model-id",
        help="HuggingFace model ID for PyTorch reference"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Output JSON file for results"
    ),
) -> None:
    """Compare PyTorch and CoreML forced alignment outputs."""

    # Determine test samples
    if audio_file and text:
        samples = [(audio_file, text)]
    elif audio_dir:
        samples = load_librispeech_samples(audio_dir, num_files)
        typer.echo(f"Loaded {len(samples)} samples from {audio_dir}")
    else:
        # Default: use cached test-clean
        default_dir = Path.home() / "Library" / "Application Support" / "FluidAudio" / "Datasets" / "LibriSpeech" / "test-clean"
        if default_dir.exists():
            samples = load_librispeech_samples(default_dir, num_files)
            typer.echo(f"Loaded {len(samples)} samples from cached test-clean")
        else:
            typer.echo("ERROR: No audio source specified and test-clean not cached.")
            typer.echo("  Use --audio-file + --text, or --audio-dir, or run:")
            typer.echo("  swift run fluidaudio download --dataset librispeech-test-clean")
            raise typer.Exit(1)

    # Load PyTorch reference
    typer.echo(f"\nLoading PyTorch model: {model_id}")
    aligner = load_pytorch_aligner(model_id)

    # Run PyTorch on all samples
    typer.echo(f"\nRunning PyTorch alignment on {len(samples)} samples...")
    pytorch_results = []
    for audio_path, transcript in samples:
        items, latency = run_pytorch_alignment(
            aligner, str(audio_path), transcript, language
        )
        pytorch_results.append((items, latency, audio_path, transcript))
        typer.echo(f"  {audio_path.name}: {len(items)} words, {latency:.0f}ms")

    # TODO: Load CoreML model and run comparison
    # For now, just save PyTorch reference results as ground truth
    typer.echo("\n--- PyTorch reference results saved ---")
    typer.echo("CoreML comparison will be added after conversion is validated.")

    # Save reference results
    if output:
        results_data = {
            "model_id": model_id,
            "language": language,
            "num_samples": len(samples),
            "samples": [],
        }
        for items, latency, audio_path, transcript in pytorch_results:
            results_data["samples"].append({
                "audio": str(audio_path),
                "transcript": transcript,
                "latency_ms": latency,
                "alignments": [
                    {
                        "text": item.text,
                        "start_time_ms": item.start_time_ms,
                        "end_time_ms": item.end_time_ms,
                    }
                    for item in items
                ],
            })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results_data, indent=2))
        typer.echo(f"\nResults written to {output}")

    # Summary stats
    total_words = sum(len(items) for items, _, _, _ in pytorch_results)
    total_latency = sum(lat for _, lat, _, _ in pytorch_results)
    typer.echo(f"\nSummary:")
    typer.echo(f"  Total words aligned: {total_words}")
    typer.echo(f"  Total PyTorch time: {total_latency:.0f}ms")
    typer.echo(f"  Avg per-sample: {total_latency / len(samples):.0f}ms")


if __name__ == "__main__":
    app()
