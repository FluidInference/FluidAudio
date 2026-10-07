# Paradee (beta)

[Paradee-8M v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0) (Sahil Mahendrakar, Apache-2.0) is
Kokoro-82M distilled into 8.07M parameters: Kokoro's own modules at smaller widths, one voice (`af_heart`),
American English, 24 kHz. Paper: [arXiv 2610.06817](https://arxiv.org/abs/2610.06817).

Core ML models: [FluidInference/paradee-8m-coreml](https://huggingface.co/FluidInference/paradee-8m-coreml)
(`int8/` 12 MB default, `fp32/` 34 MB).

> **Beta:** API, model artifacts and accuracy may change.

## Usage

```swift
let manager = ParadeeManager()                 // .int8, .cpuOnly
try await manager.initialize()                 // downloads models + English G2P assets
let samples = try await manager.synthesize(text: "Hello from FluidAudio.", speed: 1.0, noiseSeed: 0)
let phonemes = try await manager.phonemes(for: "Hello from FluidAudio.")
let fromPhonemes = try await manager.synthesize(phonemes: phonemes)
```

```bash
swift run fluidaudiocli tts "Hello from FluidAudio." --backend paradee --output out.wav
swift run fluidaudiocli tts "Hello." --backend paradee --variant fp32 --speed 1.2 --seed 7 --output out.wav
swift run fluidaudiocli tts-benchmark --backend paradee --corpus minimax-english
```

## Pipeline

```
text -> sentences -> NeMo TN -> Misaki lexicon / BART G2P -> ɾ→T, ʔ→t (misaki output form) -> ids
ParadeeText      ids [1,T]                      -> duration [1,T], d [1,224,T], asr_tok [1,512,T]
host             n = max(1, round(duration / speed)); repeat columns; N(0,1) source noise [1,1,600F]
ParadeeAcoustic  en [1,224,F], asr [1,512,F], noise -> audio [1,600F]
```

- Text is synthesized one sentence at a time, as in the upstream package; sentences over 510 phonemes are split.
- The English frontend is KokoroAne's. Its lexicon stores misaki's raw flap `ɾ` and glottal stop `ʔ`; misaki
  rewrites them to `T` / `t` before Kokoro v1.0 sees them, and Paradee was trained only on that output, so the
  Paradee text path applies the same rewrite. Without it "kittens", "satellite" and "patterns" come out garbled.
- `noiseSeed` seeds the harmonic source noise; the same seed gives the same audio.
- Compute units: `.cpuOnly` (default) or `.cpuAndNeuralEngine`. `.all` / `.cpuAndGPU` are rejected because the
  LSTMs abort in MPSGraph (`GPURNNOps … JIT not supported`).

## Benchmark

`tts-benchmark --backend paradee --corpus minimax-english`, 100 phrases, Parakeet TDT round trip,
M5 Pro, macOS 27, `.cpuOnly`:

| Variant | WER | CER | RTFx (audio / synth) | synth p50 / p95 | peak RSS |
|---|---:|---:|---:|---:|---:|
| int8 | 1.20 % | 0.14 % | 99.8× | 77 / 90 ms | 434 MB |
| fp32 | 1.20 % | 0.14 % | 100.3× | 77 / 90 ms | 433 MB |

Synth time covers the text frontend and both models. The models alone run ~150× real time; the
rest is the shared English G2P. The remaining WER is mostly text normalization ("will - power",
"five thousand" → "5,000", "spellbook").

Against upstream (Python, 10 sentences incl. a 28 s paragraph): 0 duration mismatches and identical lengths vs
PyTorch with matched noise; log-mel distance to the upstream ONNX equals the ONNX's own run-to-run noise; Whisper
transcripts identical to the ONNX's.
