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

## Conversion notes & findings

The conversion surfaced several non-obvious issues. They're recorded here so the
design choices above are clear and the dead-ends aren't re-tried.

### Conversion path
- **PyTorch (FunASR) → `torch.jit.trace` → coremltools**, not ONNX — coremltools
  dropped direct ONNX ingestion in 7.0, so we trace the original module.
- **`model.encode()` is bypassed.** Its source uses `torch.rand(1) > 0.2`, Python
  list-comprehensions and dict lookups (`lid_int_dict` / `textnorm_int_dict`) —
  all trace-hostile. We replicate only its deterministic inference path: prepend
  the 4 query embeddings `[language, event1, event2, style]`, then encoder + CTC.
  The host maps the language/text-norm choice to embed indices.

### The FP16-NaN investigation (the big one)
1. **Synthetic full-length parity hid the bug.** A 1800-frame random input gave
   100% argmax agreement; the failure only appears on *real* short audio
   zero-padded to a large shape.
2. **The NaN is compute-unit dependent, not a wrapper bug.** PyTorch handles the
   padding correctly, and the *same* FP16 `.mlpackage` gives **0 NaN on
   `CPU_AND_NE` (ANE)** but **all-NaN on `CPU_ONLY`, `CPU_AND_GPU`, and `ALL`**.
   The CPU/GPU FP16 path overflows on the zero-padded positions; ANE does not.
   (A freshly-loaded `MLModel(path)` defaults to `ALL` → watch the compute unit.)
3. **Clamping the attention mask fill (`-inf` → `-1e4`) did *not* fix it** — kept
   as a correct FP16-safety measure, but it was not the cure.
4. **FP32 is exact on every unit** but slow — kept only as the `--fp32` fallback.
5. **What fixed it:** `ct.EnumeratedShapes` length buckets `[128,256,512,1024,1800]`
   (small padding) **+ running on ANE**. 0 NaN, 100% argmax, and RTFx scales with
   audio length (5.5 s clip: bucket 128 → ~524 RTFx, bucket 1800 → ~14).

> **Known limitation.** The FP16 encoder is NaN on CPU/GPU; non-ANE hardware must
> use the `--fp32` build. Hardening FP16 for the CPU/GPU path (isolating the
> overflowing op via selective precision) is open follow-up work.

### Front-end (preprocessor)
- A CoreML replica of FunASR's `WavFrontend`: kaldi fbank-80 + LFR(m=7,n=6) + CMVN.
  Framing and LFR use **conv1d identity kernels** and the DFT is a **matmul against
  a precomputed cos/sin basis** — coremltools has no FFT and rejects `unfold`/int64
  `gather`. Kaldi's window + mel matrix are baked from torchaudio, so it's
  bit-for-bit kaldi (torch parity max\|Δ\| ≈ 2.3e-5).
- It runs **FP32 on CPU**: the power spectrum + log exceed the FP16 range, and the
  large framing convs fail ANE compile. FP32 CoreML parity: max\|Δ\| ≈ 2.9e-6.

### Parity metric
- For ASR we gate on **argmax CTC-token agreement** (what governs WER), not raw
  logit drift — FP16 logit drift on a 234M encoder is expected and benign once the
  argmax matches.

### Environment gotchas
- `torchaudio` is a required `funasr` import dependency.
- `ct.TensorType(dtype=…)` wants **numpy** dtypes (not `ct.converters.mil.types`).
- Benchmark inputs must be *representative*: all-zero or N(0,1) random inputs are
  out-of-distribution and either mask or fabricate failures.
