# Main Branch Baseline Benchmarks

Baseline benchmark results from `main` branch for regression testing PR #440 (ASR directory restructuring).

## Environment

- **Hardware**: MacBook Air M2, 16 GB
- **Branch**: `main` @ `01f1ae2b` (Fix Kokoro v2 source_noise dtype and distribution #447)
- **Build**: `swift build -c release`
- **Date**: 2026-03-28

## Results

### Batch TDT (LibriSpeech test-clean, 100 files)

| Model | WER | RTFx | Audio | Time |
|---|---|---|---|---|
| Parakeet TDT v3 (0.6B) | 2.6% | 85.7x | 901.1s | 10.5s |
| Parakeet TDT v2 (0.6B) | 3.8% | 81.7x | 901.1s | 11.0s |
| CTC-TDT 110M | 3.6% | 118.1x | 901.1s | 7.6s |

### Streaming (LibriSpeech test-clean, 100 files)

| Model | WER | RTFx | Audio | Time |
|---|---|---|---|---|
| EOU 320ms (120M) | 7.11% | 17.92x | 470.6s | 27.3s |
| Nemotron 1120ms (0.6B) | 1.99% | 9.28x | 901.1s | 97.1s |

### CTC Earnings (Earnings22-KWS, 100 files)

| Metric | Value |
|---|---|
| WER | 16.54% |
| Dict Recall | 98.9% (184/186) |
| Vocab Precision | 100.0% |
| Vocab Recall | 79.8% |
| Vocab F-score | 88.8% |
| RTFx | 42.81x |
| Audio | 1499.5s |
| Time | 35.0s |
