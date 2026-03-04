#!/usr/bin/env python3
"""
FluidAudio Benchmark Suite

Runs ASR, VAD, and Diarization benchmarks and saves results to JSON.
Compare results against Documentation/Benchmarks.md baselines.

Usage:
    python run_benchmarks.py              # Run all benchmarks
    python run_benchmarks.py --quick      # Quick smoke test
    python run_benchmarks.py --asr-only   # ASR benchmark only
    python run_benchmarks.py --vad-only   # VAD benchmark only
    python run_benchmarks.py --diar-only  # Diarization only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Baseline values from Documentation/Benchmarks.md
BASELINES = {
    "asr": {
        "wer_percent": 5.8,
        "rtfx_min": 200,  # M4 Pro: ~210x
        "description": "LibriSpeech test-clean, Parakeet TDT 0.6B"
    },
    "vad": {
        "f1_percent": 85.0,
        "rtfx_min": 500,
        "description": "VOiCES dataset, Silero VAD"
    },
    "diarization": {
        "der_percent": 17.7,
        "rtfx_min": 1.0,
        "description": "AMI SDM, pyannote-based"
    }
}


def run_command(cmd: list[str], output_file: Path | None = None) -> tuple[int, str]:
    """Run a command and optionally save output."""
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    if output_file:
        output_file.write_text(output)

    return result.returncode, output


def build_release() -> bool:
    """Build the project in release mode."""
    print("\n" + "=" * 60)
    print("Building release...")
    print("=" * 60)

    returncode, _ = run_command(["swift", "build", "-c", "release"])

    if returncode != 0:
        print("ERROR: Build failed!")
        return False

    print("Build successful.")
    return True


def run_asr_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run ASR benchmark on LibriSpeech test-clean."""
    print("\n" + "=" * 60)
    print("ASR Benchmark (LibriSpeech test-clean)")
    print("=" * 60)

    max_files = "100" if quick else "all"
    output_json = output_dir / f"asr_results.json"

    cmd = [
        "swift", "run", "-c", "release", "fluidaudio", "asr-benchmark",
        "--subset", "test-clean",
        "--max-files", max_files,
        "--output", str(output_json)
    ]

    returncode, output = run_command(cmd, output_dir / "asr_log.txt")

    if returncode != 0:
        print(f"ERROR: ASR benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def run_vad_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run VAD benchmark."""
    print("\n" + "=" * 60)
    print("VAD Benchmark")
    print("=" * 60)

    dataset = "mini50" if quick else "voices-subset"
    output_json = output_dir / f"vad_results.json"

    cmd = [
        "swift", "run", "-c", "release", "fluidaudio", "vad-benchmark",
        "--dataset", dataset,
        "--all-files",
        "--threshold", "0.5",
        "--output", str(output_json)
    ]

    returncode, output = run_command(cmd, output_dir / "vad_log.txt")

    if returncode != 0:
        print(f"ERROR: VAD benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def run_diarization_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run diarization benchmark on AMI SDM."""
    print("\n" + "=" * 60)
    print("Diarization Benchmark (AMI SDM)")
    print("=" * 60)

    output_json = output_dir / f"diarization_results.json"

    cmd = [
        "swift", "run", "-c", "release", "fluidaudio", "diarization-benchmark",
        "--auto-download",
        "--output", str(output_json)
    ]

    if quick:
        cmd.extend(["--single-file", "ES2004a"])

    returncode, output = run_command(cmd, output_dir / "diarization_log.txt")

    if returncode != 0:
        print(f"ERROR: Diarization benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def compare_results(results: dict) -> None:
    """Compare results against baselines."""
    print("\n" + "=" * 60)
    print("Results vs Baselines (Documentation/Benchmarks.md)")
    print("=" * 60)

    if "asr" in results and results["asr"]:
        asr = results["asr"]
        baseline = BASELINES["asr"]
        wer = asr.get("wer", asr.get("average_wer", 0)) * 100
        rtfx = asr.get("rtfx", asr.get("median_rtfx", 0))

        wer_status = "✓" if wer <= baseline["wer_percent"] * 1.1 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] * 0.8 else "✗"

        print(f"\nASR ({baseline['description']}):")
        print(f"  WER:  {wer:.1f}% (baseline: {baseline['wer_percent']}%) {wer_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")

    if "vad" in results and results["vad"]:
        vad = results["vad"]
        baseline = BASELINES["vad"]
        f1 = vad.get("f1_score", 0)
        rtfx = vad.get("rtfx", 0)

        f1_status = "✓" if f1 >= baseline["f1_percent"] * 0.9 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] * 0.5 else "✗"

        print(f"\nVAD ({baseline['description']}):")
        print(f"  F1:   {f1:.1f}% (baseline: {baseline['f1_percent']}%+) {f1_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")

    if "diarization" in results and results["diarization"]:
        diar = results["diarization"]
        baseline = BASELINES["diarization"]
        der = diar.get("der", diar.get("average_der", 0)) * 100
        rtfx = diar.get("rtfx", diar.get("average_rtfx", 0))

        der_status = "✓" if der <= baseline["der_percent"] * 1.2 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] else "✗"

        print(f"\nDiarization ({baseline['description']}):")
        print(f"  DER:  {der:.1f}% (baseline: {baseline['der_percent']}%) {der_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")


def main():
    parser = argparse.ArgumentParser(description="FluidAudio Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with smaller datasets")
    parser.add_argument("--asr-only", action="store_true", help="Run ASR benchmark only")
    parser.add_argument("--vad-only", action="store_true", help="Run VAD benchmark only")
    parser.add_argument("--diar-only", action="store_true", help="Run diarization benchmark only")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    args = parser.parse_args()

    # Determine which benchmarks to run
    run_all = not (args.asr_only or args.vad_only or args.diar_only)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("benchmark-results") / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FluidAudio Benchmark Suite")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output: {output_dir}")
    print(f"Time: {timestamp}")

    # Build first
    if not build_release():
        sys.exit(1)

    results = {}

    # Run benchmarks
    if run_all or args.asr_only:
        results["asr"] = run_asr_benchmark(output_dir, args.quick)

    if run_all or args.vad_only:
        results["vad"] = run_vad_benchmark(output_dir, args.quick)

    if run_all or args.diar_only:
        results["diarization"] = run_diarization_benchmark(output_dir, args.quick)

    # Save combined results
    combined_output = output_dir / "benchmark_results.json"
    combined_output.write_text(json.dumps({
        "timestamp": timestamp,
        "mode": "quick" if args.quick else "full",
        "baselines": BASELINES,
        "results": results
    }, indent=2))

    # Compare against baselines
    compare_results(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print(f"Results saved to: {combined_output}")


if __name__ == "__main__":
    main()
