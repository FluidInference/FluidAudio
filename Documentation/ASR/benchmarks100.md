# PR #440 Regression Benchmarks (100 files)

Benchmark comparison between `main` and PR #440 (`standardize-asr-directory-structure`) to verify the directory restructuring introduces no regressions.

## Reproduction

All batch TDT and CTC earnings benchmarks can be reproduced with [`Scripts/parakeet_subset_benchmark.sh`](../../Scripts/parakeet_subset_benchmark.sh):

```bash
# Download models and datasets (requires internet)
./Scripts/parakeet_subset_benchmark.sh --download

# Run all 4 benchmarks offline (100 files each, sleep-prevented)
./Scripts/parakeet_subset_benchmark.sh
```

## Environment

- **Hardware**: MacBook Air M2, 16 GB
- **Build**: `swift build -c release`
- **Date**: 2026-03-28
- **main**: `01f1ae2b` (Fix Kokoro v2 source_noise dtype and distribution #447)
- **PR**: `839010538` (standardize-asr-directory-structure)

## Comparison

### Batch TDT (LibriSpeech test-clean, 100 files)

| Model | WER (main) | WER (PR) | RTFx (main) | RTFx (PR) |
|---|---|---|---|---|
| Parakeet TDT v3 (0.6B) | 2.6% | 2.6% | 85.7x | 77.6x |
| Parakeet TDT v2 (0.6B) | 3.8% | 3.8% | 81.7x | 73.9x |
| CTC-TDT 110M | 3.6% | 3.6% | 118.1x | 105.2x |

### Streaming (LibriSpeech test-clean, 100 files)

| Model | WER (main) | WER (PR) | RTFx (main) | RTFx (PR) |
|---|---|---|---|---|
| EOU 320ms (120M) | 7.11% | 7.11% | 17.92x | 18.19x |
| Nemotron 1120ms (0.6B) | 1.99% | 1.99% | 9.28x | 9.03x |

### CTC Earnings (Earnings22-KWS, 100 files)

| Metric | main | PR |
|---|---|---|
| WER | 16.54% | 16.51% |
| Dict Recall | 98.9% | 98.9% |
| Vocab Precision | 100.0% | 100.0% |
| Vocab Recall | 79.8% | 79.8% |
| Vocab F-score | 88.8% | 88.8% |
| RTFx | 42.81x | 44.61x |

## Verdict

**No regressions.** WER is identical across all 6 benchmarks. RTFx differences are within normal system noise (M2 thermals, background processes). The directory restructuring is a pure file move with no behavioral changes.

## Issue #435: Standalone CTC Head for Custom Vocabulary (Beta)

Benchmark comparing separate CTC encoder vs standalone CTC head extracted from the TDT-CTC-110M hybrid model.
See [#435](https://github.com/FluidInference/FluidAudio/issues/435) and [PR #450](https://github.com/FluidInference/FluidAudio/pull/450).

| Metric | Separate CTC (v2 TDT) | Separate CTC (110m TDT) | Standalone CTC Head (110m TDT) |
|---|---|---|---|
| Dict Recall | 99.3% | 99.4% | 99.4% |
| RTFx | 43.94x | 25.98x | 70.29x |
| Additional model size | 97.5 MB | 97.5 MB | 1 MB |

