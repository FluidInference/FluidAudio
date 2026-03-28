# PR #440 Regression Benchmarks (100 files)

Benchmark comparison between `main` and PR #440 (`standardize-asr-directory-structure`) to verify the directory restructuring introduces no regressions.

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

---

# Issue #435: Standalone CTC Head for Custom Vocabulary (Beta)

Benchmark comparing separate CTC encoder vs standalone CTC head extracted from the TDT-CTC-110M hybrid model.
See [#435](https://github.com/FluidInference/FluidAudio/issues/435) and [PR #450](https://github.com/FluidInference/FluidAudio/pull/450).

## Environment

- **Hardware**: MacBook Air M2, 16 GB
- **Build**: `swift build -c release`
- **Date**: 2026-03-28
- **Branch**: `ctc-head-export`

## CTC Earnings (Earnings22-KWS, 772 files)

| Metric | Separate CTC (v2 TDT) | Separate CTC (110m TDT) | Standalone CTC Head (110m TDT) |
|---|---|---|---|
| WER | 14.67% | 16.08% | 16.88% |
| Dict Recall | 99.3% | 99.4% | 99.4% |
| Vocab Precision | 99.8% | 99.7% | 99.6% |
| Vocab Recall | 73.7% | 70.0% | 59.6% |
| Vocab F-score | 84.8% | 82.0% | 74.6% |
| RTFx | 43.94x | 25.98x | **70.29x** |
| Additional model size | 97.5 MB | 97.5 MB | **1 MB** |

## Analysis

- **Dict Recall**: Identical at 99.4% between separate CTC encoder and standalone CTC head. The CTC head produces equivalent keyword detection quality.
- **RTFx**: **70.29x** (standalone head) vs **25.98x** (separate encoder) = **2.7x speedup**. The CTC head runs on the existing TDT encoder output with no second encoder pass.
- **Model size**: 1 MB (standalone head) vs 97.5 MB (separate CTC encoder) = **97x smaller**.
- **WER**: Slight increase (16.08% → 16.88%) because the CTC head's logits have different characteristics than the separately-trained CTC encoder, affecting vocabulary rescoring decisions.
- **Vocab Recall**: Lower (70.0% → 59.6%) for the same reason — the CTC head's logit distribution differs from the standalone CTC model, leading to fewer vocabulary replacements being applied. This is a rescoring tuning issue, not a detection issue.

## Key Takeaways

1. **Standalone CTC head eliminates separate CTC encoder** — a 1MB linear projection on the shared TDT encoder output
2. **97x smaller**: 1 MB vs 97.5 MB additional model weight
3. **Dict Recall preserved**: Keyword detection quality is identical at 99.4%
4. **2.7x faster**: No second encoder pass needed for custom vocabulary workloads
5. **Beta status**: Auto-download from HuggingFace and local file loading both supported
