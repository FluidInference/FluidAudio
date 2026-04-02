#!/usr/bin/env python3
"""Test FluidAudio CTC zh-CN model using THCHS-30 from HuggingFace.

Usage:
    python Scripts/test_ctc_zh_cn_hf.py --dataset your-username/thchs30-test --samples 100
    python Scripts/test_ctc_zh_cn_hf.py --dataset your-username/thchs30-test  # Full test set
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese text for CER calculation."""
    # Remove Chinese punctuation
    text = re.sub(r'[，。！？、；：""''（）《》【】…—·]', '', text)
    # Remove English punctuation
    text = re.sub(r'[,.!?;:()\[\]{}<>"\'\\-]', '', text)
    # Convert Arabic digits to Chinese
    digit_map = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    for digit, chinese in digit_map.items():
        text = text.replace(digit, chinese)
    # Normalize whitespace and remove spaces
    text = ' '.join(text.split())
    text = text.replace(' ', '')
    return text


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate using Levenshtein distance."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    distance = dp[m][n]
    return distance / len(ref_chars) if ref_chars else (1.0 if hyp_chars else 0.0)


def transcribe(audio_path: str) -> tuple[str | None, float]:
    """Transcribe audio using FluidAudio CLI."""
    cmd = ["swift", "run", "-c", "release", "fluidaudiocli", "ctc-zh-cn-transcribe", str(audio_path)]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start_time

    # Extract transcription (last non-log line)
    for line in reversed(result.stdout.split("\n")):
        line = line.strip()
        if line and not line.startswith("["):
            return line, elapsed

    return None, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test FluidAudio CTC zh-CN on THCHS-30 from HuggingFace")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g., username/thchs30-test)")
    parser.add_argument("--samples", type=int, help="Number of samples to test (default: all)")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with: pip install datasets soundfile")
        sys.exit(1)

    print("=" * 100)
    print("FluidAudio CTC zh-CN Test - THCHS-30 (HuggingFace)")
    print("=" * 100)
    print(f"Dataset: {args.dataset}")
    print()

    # Load dataset
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(args.dataset, split=args.split)

    # Limit samples if specified
    if args.samples:
        dataset = dataset.select(range(min(args.samples, len(dataset))))

    print(f"Samples: {len(dataset)}")
    print()

    # Build release
    print("Building release...")
    subprocess.run(["swift", "build", "-c", "release"], capture_output=True)
    print("✓ Build complete\n")

    print("Running tests...\n")

    cers = []
    latencies = []
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, sample in enumerate(dataset):
            # Save audio to temp file
            audio_path = Path(tmpdir) / f"temp_{idx}.wav"

            # Write audio file
            import soundfile as sf
            sf.write(str(audio_path), sample['audio']['array'], sample['audio']['sampling_rate'])

            # Transcribe
            hypothesis, elapsed = transcribe(str(audio_path))

            if hypothesis is None:
                print(f"{idx + 1}/{len(dataset)} FAIL - transcription error")
                failed += 1
                continue

            # Calculate CER
            ref_norm = normalize_chinese_text(sample['text'])
            hyp_norm = normalize_chinese_text(hypothesis)
            cer = calculate_cer(ref_norm, hyp_norm)

            cers.append(cer)
            latencies.append(elapsed)

            if (idx + 1) % 50 == 0:
                mean_cer = sum(cers) / len(cers) * 100
                print(f"{idx + 1}/{len(dataset)} - CER: {cer*100:.2f}% (running avg: {mean_cer:.2f}%)")

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    if cers:
        mean_cer = sum(cers) / len(cers) * 100
        sorted_cers = sorted(cers)
        median_cer = sorted_cers[len(sorted_cers) // 2] * 100
        mean_latency = sum(latencies) / len(latencies) * 1000

        print(f"Samples:        {len(dataset) - failed} (failed: {failed})")
        print(f"Mean CER:       {mean_cer:.2f}%")
        print(f"Median CER:     {median_cer:.2f}%")
        print(f"Mean Latency:   {mean_latency:.1f} ms")

        # CER distribution
        below5 = sum(1 for c in cers if c < 0.05)
        below10 = sum(1 for c in cers if c < 0.10)
        below20 = sum(1 for c in cers if c < 0.20)

        print()
        print("CER Distribution:")
        print(f"  <5%:  {below5:3d} samples ({below5/len(cers)*100:.1f}%)")
        print(f"  <10%: {below10:3d} samples ({below10/len(cers)*100:.1f}%)")
        print(f"  <20%: {below20:3d} samples ({below20/len(cers)*100:.1f}%)")

        # Exit with error if CER is too high
        if mean_cer > 10.0:
            print()
            print(f"❌ FAILED: Mean CER {mean_cer:.2f}% exceeds threshold of 10.0%")
            sys.exit(1)
        else:
            print()
            print(f"✓ PASSED: Mean CER {mean_cer:.2f}% is within acceptable range")
    else:
        print("❌ No successful transcriptions")
        sys.exit(1)

    print("=" * 100)


if __name__ == "__main__":
    main()
