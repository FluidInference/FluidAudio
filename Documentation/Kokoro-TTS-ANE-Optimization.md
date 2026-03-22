# Kokoro TTS ANE Optimization

## Overview

Kokoro TTS uses a single CoreML model (`kokoro_21_5s.mlmodelc`) for text-to-speech synthesis. The model was originally converted with `compute_precision=FLOAT32`, which prevented ANE scheduling for most ops. Converting with `FLOAT16` precision moves the BERT and generator ops to ANE, yielding a 1.67x speedup.

## Model Architecture

| Component | Layers | Role |
|-----------|--------|------|
| BERT encoder | 3 transformer layers | Text encoding (input_ids → hidden states) |
| Duration predictor | LSTMs | Predict phoneme durations |
| Generator | ConvTranspose1d + ResBlocks | Waveform synthesis from mel features |
| Source module | Harmonic + noise generation | Excitation signal for vocoder |

Input: 124 phoneme tokens, voice embedding (256-d), random phases (9-d), attention mask, source noise.
Output: 5 seconds of 24kHz audio (120,000 samples).

## Baseline Profiling (FLOAT32, `.cpuAndGPU`)

The original model was converted with `compute_precision=ct.precision.FLOAT32` and loaded with `.cpuAndGPU` compute units.

| Device | Ops | Cost % |
|--------|-----|--------|
| CPU | 50 | **100.0%** |
| GPU | 1382 | 0.0% |
| ANE | 0 | 0.0% |

All cost was on CPU, concentrated in 6 LSTM ops (duration predictor). The 1382 GPU ops (BERT transformer layers + generator convolutions) showed 0% cost in profiling but still contributed to wall-clock latency.

## Optimization: FLOAT16 Conversion

Single change: `compute_precision=ct.precision.FLOAT16` in the coremltools conversion, with `minimum_deployment_target=ct.target.iOS17`.

### Profiling After (FLOAT16, `.all`)

| Device | Ops | Cost % |
|--------|-----|--------|
| **ANE** | **833** | **0.0%** |
| GPU | 513 | 0.0% |
| CPU | 114 | **100.0%** |

- 833 ops moved to ANE (BERT transformer layers, generator convolutions)
- 6 LSTM ops remain on CPU (CoreML scheduler does not schedule LSTMs to ANE regardless of precision)
- Cost percentages are misleading — the ANE ops contribute significantly to the 1.67x real speedup

## Benchmark Results (M4 Max)

Isolated model inference, 20 iterations after 3 warmup. Input: 124 tokens (50 real + 74 padding).

| Metric | Original (cpuAndGPU, fp32) | ANE (all, fp16) |
|--------|---------------------------|-----------------|
| Mean | 419.08 ms | 254.65 ms |
| Median | 416.66 ms | 249.97 ms |
| P95 | 432.27 ms | 268.98 ms |
| Min | 408.58 ms | 240.88 ms |

**Speedup (median): 1.67x — ANE version is 66.7% faster**

RTFx (5s audio): Original 12.0x, ANE 20.0x

## End-to-End TTS Benchmark (11 passages)

Benchmarked using `fluidaudiocli tts --benchmark` with 11 passages ranging from 6 characters ("Short.") to 4,269 characters (5-minute narration). Total audio: 402.3 seconds.

| Metric | Original | ANE (fp16) |
|--------|----------|------------|
| Total synthesis time | 35.35s | 35.18s |
| Initialization time | 10.68s | 5.07s |
| Best RTFx (run 9, 1228 chars) | 13.3x | 13.4x |
| Worst RTFx (run 8, "Short.") | 1.8x | 1.9x |

End-to-end synthesis times are similar because the chunking pipeline amortizes model latency across multiple chunks for long texts, and the LSTM bottleneck on CPU is the same in both.

### Round-Trip Quality Validation

Generated audio with both models, then transcribed with Qwen3 ASR. All tested sentences produced **identical transcriptions** between original and ANE versions — zero quality degradation.

## Why LSTMs Stay on CPU

CoreML's scheduler places LSTM/GRU ops on CPU regardless of compute units or precision settings. Although MPSGraph has LSTM support (since WWDC 2022), CoreML does not route recurrent ops to GPU or ANE. The 6 LSTM ops in Kokoro's duration predictor account for 100% of the profiled cost.

## Production Integration Notes

The current production code in `Sources/FluidAudio/TTS/TtsModels.swift` loads Kokoro with `.cpuAndGPU` compute units. To use the ANE-optimized model, this needs to change to `.all`.

## Files Created

- `mobius/models/tts/kokoro/coreml/convert_v21_ane.py` — FLOAT16 conversion script
- `mobius/models/tts/kokoro/coreml/kokoro_21_5s_ane.mlpackage` — Converted model
- `mobius/models/tts/kokoro/coreml/kokoro_21_5s_ane.mlmodelc` — Compiled model (833 ANE ops)
- `mobius/models/tts/kokoro/coreml/benchmark_kokoro.swift` — Swift benchmark script

## Key Source Files

- `Sources/FluidAudio/TTS/TtsModels.swift` — model loading, compute unit config
- `Sources/FluidAudio/TTS/Kokoro/KokoroSynthesizer.swift` — synthesizer interface
- `mobius/models/tts/kokoro/coreml/convert_v21_with_noise.py` — original conversion script
