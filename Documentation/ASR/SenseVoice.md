# SenseVoice

Non-autoregressive multilingual ASR using [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) (FunASR) converted to CoreML. A SANM encoder + single CTC head produces all output tokens in one forward pass — no autoregressive decode loop.

## Model

**CoreML Model**: [FluidInference/sensevoice-small-coreml](https://huggingface.co/FluidInference/sensevoice-small-coreml)

3-stage pipeline:

| Stage | File | Precision | Compute unit |
|-------|------|-----------|--------------|
| Front-end | `SenseVoicePreprocessor.mlmodelc` | FP32 | CPU |
| Encoder + CTC | `SenseVoiceSmall.mlmodelc` | FP16 | **ANE** (`CPU_AND_NE`) |
| Encoder fallback | `SenseVoiceSmall_fp32.mlmodelc` | FP32 | any (`--fp32`) |

> **Compute-unit requirement.** The FP16 encoder is numerically correct only on the Neural Engine; it produces NaN on the CPU/GPU FP16 path. The loader pins it to `.cpuAndNeuralEngine`. On hardware without an ANE, use `--fp32`.

## Architecture

```
waveform → [Preprocessor FP32/CPU] → 560-d LFR features
        → [SenseVoiceSmall FP16/ANE encoder+CTC] → logits [1, T+4, 25055]
        → host greedy-CTC decode → text
```

- **Front-end**: kaldi fbank-80 → LFR (m=7, n=6 → 560-d) → CMVN. A CoreML replica of FunASR's `WavFrontend` (matches to max\|Δ\|≈2e-5). FP32/CPU because the power spectrum and log exceed the FP16 range and the framing convolutions do not ANE-compile.
- **Encoder**: enumerated sequence buckets `[128, 256, 512, 1024, 1800]`; the host pads features up to the smallest bucket ≥ T. The 4 leading logit positions are the language / emotion / event / inverse-text-norm query tokens.
- **Decode**: greedy CTC (blank = 0) → collapse repeats → SentencePiece detokenize → strip the leading `<|...|>` tags.

## Supported Languages

SenseVoiceSmall covers 50+ languages (strongest on zh / yue / en / ja / ko). Language is auto-detected by default (`language = 0`).

## Usage

### CLI

```bash
# Transcribe a file (auto language)
swift run -c release fluidaudiocli sensevoice-transcribe audio.wav

# FP32 encoder (no Neural Engine)
swift run -c release fluidaudiocli sensevoice-transcribe audio.wav --fp32

# FLEURS WER/CER benchmark (English + Chinese)
swift run -c release fluidaudiocli sensevoice-benchmark --languages en_us,cmn_hans_cn --samples 100
```

### Swift

```swift
let manager = try await SenseVoiceManager.load()           // fp16/ANE (use useFp32Encoder: true for fallback)
let text = try await manager.transcribe(audioURL: url)
```

## Benchmarks

Measured on Apple M5 Pro with the CoreML FP16 encoder on the Neural Engine, over
the **full** canonical test sets — directly comparable to the published
[SenseVoice-Small results](https://github.com/FunAudioLLM/SenseVoice).

### English — LibriSpeech test-clean (full, 2,620 utts)

| Metric | CoreML (ANE) | Official SenseVoice-Small |
|--------|--------------|---------------------------|
| **WER** | **3.22%** | ~3.1% |
| Median RTFx | 299 | — |

### Chinese — AISHELL-1 test (full, ~7,176 utts)

| Metric | CoreML (ANE) | Official SenseVoice-Small |
|--------|--------------|---------------------------|
| **CER** | **3.09%** | ~2.9% |
| Median RTFx | 382 | — |

Both reproduce the published numbers, confirming the conversion (front-end +
encoder + decode) is faithful. Offline CoreML↔PyTorch parity was also verified on
FLEURS (en WER Δ +0.00 pp, zh CER Δ −0.03 pp).

### Reproduction

English (FLEURS, in-repo) via the CLI:

```bash
swift run -c release fluidaudiocli sensevoice-benchmark \
    --languages en_us,cmn_hans_cn --samples all
```

The canonical LibriSpeech / AISHELL-1 numbers above were measured with the same
CoreML(ANE) model via the conversion repo's Python harness (the FLEURS CLI
benchmark above is the in-repo, multilingual reproducible path).
