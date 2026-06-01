# Paraformer

Non-autoregressive Mandarin (zh) ASR using FunASR's **Paraformer-large**, converted
to CoreML. A SANM encoder + a CIF predictor (one acoustic-embedding token per output
character) + a single-pass parallel decoder.

## Model

**CoreML Model**: [FluidInference/paraformer-large-zh-coreml](https://huggingface.co/FluidInference/paraformer-large-zh-coreml)

4 CoreML stages + a host-side CIF:

| File | Precision | Compute unit | Role |
|------|-----------|--------------|------|
| `ParaformerPreprocessor.mlmodelc` | FP32 | CPU | waveform → 560-d LFR features |
| `ParaformerEncoder.mlmodelc` / `_int8` | FP16 / INT8 | **ANE** | SANM encoder (enumerated buckets) |
| `ParaformerCifAlphas.mlmodelc` | FP16 | **ANE** | enc_out → per-frame alphas |
| `ParaformerDecoder.mlmodelc` / `_int8` | FP16 / INT8 | **ANE** | parallel decoder → token logits |
| `vocab.json` | — | — | 8404 CharTokenizer tokens |

The **CIF predictor** emits a *dynamic* token count, so it can't be a fixed-shape
graph: its conv1d+linear+sigmoid (→ alphas) is the `CifAlphas` model, and the host
does only the integrate-and-fire loop (`ParaformerCif`, ported bit-exact from FunASR).

## Architecture

```
waveform → [Preprocessor fp32/CPU] → features [1,T,560]
        → [Encoder fp16/ANE] → enc_out [1,T,512]
        → [CifAlphas fp16/ANE] → alphas → [host integrate-and-fire] → acoustic_embeds, L
        → [Decoder fp16/ANE] → logits [1,L,8404]
        → argmax per token → drop sos(1)/eos(2)/blank(0) → CharTokenizer
```

Run encoder/CifAlphas/decoder on `.cpuAndNeuralEngine`; the preprocessor is FP32/CPU
(power-spectrum + log exceed the FP16 range).

## Audio length

Offline / whole-utterance (not streaming). ~60 ms per feature frame (fbank 10 ms ×
LFR n=6). The encoder enumerates `[128,256,512,1024,1800]` frames, but the **decoder
is fixed at 512 frames / 128 tokens**, so the effective ceiling is **~30 s of audio /
~128 characters** per call (`ParaformerManager` truncates longer audio). Segment with
VAD for long-form input.

## Usage

### CLI

```bash
# Transcribe (fp16, default)
swift run -c release fluidaudiocli paraformer-transcribe audio.wav

# int8 encoder/decoder (~half size, accuracy-neutral)
swift run -c release fluidaudiocli paraformer-transcribe audio.wav --int8
```

### Swift

```swift
let manager = try await ParaformerManager.load()            // .fp16 default; .int8 for half size
let text = try await manager.transcribe(audioURL: url)
```

## Benchmarks

Full canonical set on Apple M5 Pro (CoreML on ANE) — see
[Benchmarks.md#paraformer](../Benchmarks.md#paraformer) for the full table.

| Precision | size (enc+dec) | AISHELL-1 CER (7,176) | RTFx | peak RAM |
|-----------|----------------|------------------------|------|----------|
| fp16 | 411 MB | 2.12% | 85× | 0.38 GB |
| int8 | 207 MB | 2.12% | 84× | 0.24 GB |

Official Paraformer-large AISHELL-1 ≈ 1.95% (the ~0.17 pp gap is fp16 + the
fixed-shape decoder padding). int8 is accuracy-neutral.

## Conversion notes

Two SANM-specific fixes were required for fp16/ANE under bucket padding (shared with
SenseVoice): a fp16-safe attention mask fill (`-inf` → `-1e4`), and building the
encoder/decoder pad-masks from the **input tensor's seq dim** (so `EnumeratedShapes`
generalize on ANE) rather than `lengths.max()`. The decoder is currently fixed-shape
(enc 512 / tokens 128); an enumerated-shape decoder would raise RTFx on short clips
and lift the ~30 s ceiling.
