#!/usr/bin/env python3
"""Benchmark FluidAudio CTC zh-CN on FLEURS Mandarin Chinese."""
import json
import subprocess
import sys
import time
from pathlib import Path


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese text for CER calculation (matches mobius)."""
    import re

    # Remove Chinese punctuation
    text = re.sub(r'[，。！？、；：""''（）《》【】…—·]', '', text)

    # Remove English punctuation
    text = re.sub(r'[,.!?;:()\[\]{}<>"\'-]', '', text)

    # CRITICAL FIX: Remove English/Latin text (FLEURS has mixed English in references)
    # Keep only Chinese characters, digits, and spaces
    text = re.sub(r'[a-zA-Zğü]+', '', text)  # Remove English words and Turkish chars

    # Convert Arabic digits to Chinese characters
    digit_map = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    for digit, chinese in digit_map.items():
        text = text.replace(digit, chinese)

    # Normalize whitespace
    text = ' '.join(text.split())

    # Remove all spaces for character-level comparison
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


def transcribe(audio_path: str, use_fp32: bool = False) -> tuple[str | None, float]:
    """Transcribe audio using FluidAudio CLI."""
    cmd = ["swift", "run", "-c", "release", "fluidaudiocli", "ctc-zh-cn-transcribe", str(audio_path)]
    if use_fp32:
        cmd.append("--fp32")

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
    import sys
    use_fp32 = "--fp32" in sys.argv

    # Load benchmark data
    benchmark_file = Path("mobius/models/stt/parakeet-ctc-0.6b-zh-cn/coreml/benchmark_results_full_pipeline_100.json")
    with open(benchmark_file) as f:
        data = json.load(f)

    audio_dir = Path("mobius/models/stt/parakeet-ctc-0.6b-zh-cn/coreml/test_audio_100")
    samples = data['results']

    encoder_type = "fp32 (1.1GB)" if use_fp32 else "int8 (0.55GB)"

    print("=" * 100)
    print("FluidAudio CTC zh-CN Benchmark - FLEURS Mandarin Chinese")
    print("=" * 100)
    print(f"Encoder: {encoder_type}")
    print(f"Samples: {len(samples)}")
    print()

    # Build release
    print("Building release...")
    subprocess.run(["swift", "build", "-c", "release"], capture_output=True)
    print("✓ Build complete\n")

    print("Running benchmark...")
    print()

    cers = []
    latencies = []
    failed = 0

    for idx, sample in enumerate(samples):
        audio_file = audio_dir / f"fleurs_cmn_{idx:03d}.wav"

        if not audio_file.exists():
            print(f"{idx + 1}/{len(samples)} SKIP - audio not found")
            failed += 1
            continue

        hypothesis, elapsed = transcribe(str(audio_file), use_fp32=use_fp32)

        if hypothesis is None:
            print(f"{idx + 1}/{len(samples)} FAIL - transcription error")
            failed += 1
            continue

        ref_norm = normalize_chinese_text(sample['reference'])
        hyp_norm = normalize_chinese_text(hypothesis)
        cer = calculate_cer(ref_norm, hyp_norm)

        cers.append(cer)
        latencies.append(elapsed)

        if (idx + 1) % 10 == 0:
            mean_cer = sum(cers) / len(cers) * 100
            print(f"{idx + 1}/{len(samples)} - CER: {cer*100:.2f}% (running avg: {mean_cer:.2f}%)")

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    if cers:
        mean_cer = sum(cers) / len(cers) * 100
        sorted_cers = sorted(cers)
        median_cer = sorted_cers[len(sorted_cers) // 2] * 100
        mean_latency = sum(latencies) / len(latencies) * 1000

        print(f"Samples:        {len(samples) - failed} (failed: {failed})")
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
    else:
        print("❌ No successful transcriptions")

    print("=" * 100)


if __name__ == "__main__":
    main()
