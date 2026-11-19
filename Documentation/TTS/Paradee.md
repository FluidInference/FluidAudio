# Paradee (beta)

[Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) (Sahil Mahendrakar, Apache-2.0) is Kokoro-82M
distilled into ~8M parameters: Kokoro's own modules at smaller widths, English, 24 kHz.
Paper: [arXiv 2610.06817](https://arxiv.org/abs/2610.06817) (v1.0).

| `ParadeeVersion` | Voices | Decoder |
|---|---|---|
| `.v12` (default) | `af_heart`, `bf_isabella`, `am_puck`, `bm_fable` | v1.1's frame-rate source-filter head |
| `.v10` | `af_heart` | Kokoro's iSTFTNet generator + phase-lock filter |

`bf_isabella` and `bm_fable` are British and are read with British phonemes (below).

Core ML models: [FluidInference/paradee-8m-coreml](https://huggingface.co/FluidInference/paradee-8m-coreml)
(v1.2 in `v1.2/int8/` 11 MB default and `v1.2/fp32/` 34 MB; v1.0 in `int8/` and `fp32/`). Conversion:
[mobius `models/tts/paradee/coreml`](https://github.com/FluidInference/mobius/tree/main/models/tts/paradee/coreml).

> **Beta:** API, model artifacts and accuracy may change.

## Usage

```swift
let manager = ParadeeManager()                 // v1.2, .int8, .cpuOnly
try await manager.initialize()                 // downloads models + English G2P assets
let samples = try await manager.synthesize(text: "Hello from FluidAudio.", voice: .bmFable, speed: 1.0, noiseSeed: 0)
let phonemes = try await manager.phonemes(for: "Hello from FluidAudio.", voice: .bmFable)
let fromPhonemes = try await manager.synthesize(phonemes: phonemes, voice: .bmFable)

let v10 = ParadeeManager(version: .v10)       // the previous release, af_heart only
```

```bash
swift run fluidaudiocli tts "Hello from FluidAudio." --backend paradee --voice bf_isabella --output out.wav
swift run fluidaudiocli tts "Hello." --backend paradee --variant fp32 --speed 1.2 --seed 7 --output out.wav
swift run fluidaudiocli tts "Hello." --backend paradee --paradee-version 1.0 --output out.wav
swift run fluidaudiocli tts-benchmark --backend paradee --corpus minimax-english --voice bm_fable
```

## Pipeline

```
text -> sentences -> NeMo TN -> Misaki lexicon / BART G2P -> ɾ→T, ʔ→t (misaki output form) -> ids
ParadeeText      ids [1,T], voice [1] (v1.2)    -> duration [1,T], d [1,224,T], asr_tok [1,512,T]
host             n = max(1, round(duration / speed)); repeat columns; N(0,1) source noise [1,1,600F]
ParadeeAcoustic  en [1,224,F], asr [1,512,F], noise, voice [1] (v1.2) -> audio [1,600F]
```

- `voice` picks a row of each half's voice table; everything else is shared between the voices.
- Text is synthesized one sentence at a time, as in the upstream package; sentences over 510 phonemes are split.
- The English frontend is KokoroAne's. Its lexicon stores misaki's raw flap `ɾ` and glottal stop `ʔ`; misaki
  rewrites them to `T` / `t` before Kokoro v1.0 sees them, and Paradee was trained only on that output, so the
  Paradee text path applies the same rewrite. Without it "kittens", "satellite" and "patterns" come out garbled.
- `noiseSeed` seeds the source noise; the same seed gives the same audio.
- Compute units: `.cpuOnly` (default) or `.cpuAndNeuralEngine`. `.all` / `.cpuAndGPU` are rejected because the
  LSTMs abort in MPSGraph (`GPURNNOps … JIT not supported`).

## British voices

Upstream reads `bf_isabella` and `bm_fable` with misaki's British G2P, which is what Kokoro used when it made their
training data ("bath" `bˈɑːθ`, "go" `ɡˈQ`, "schedule" `ʃˈɛdjuːl`). FluidAudio does the same:

- Lexicon: `gb_lexicon_cache.json` (misaki `gb_gold` + `gb_silver`, in the same format as `us_lexicon_cache.json`),
  from the Kokoro repo. It is downloaded the first time a British voice is used.
- The `-'s` clitic after a sibilant is `ɪz` (`boss's` → `bˈɒsɪz`), as in misaki's British `_s`.
- Words missing from the lexicon go through the BART G2P, which only knows American phonemes. Its output is
  rewritten into misaki's British inventory (`MisakiBritishFallback`): `O`→`Q`, `ᵻ`→`ɪ`, flaps → `t`, non-rhotic
  `ɹ` before a non-vowel (`ɑɹ`→`ɑː`, `ɪɹ`→`ɪə`, `əɹ`→`ə` …), `æ`→`a`, `ɑ`→`ɒ`. On the 153,873 words in both misaki
  lexicons this halves the edit distance to the British entry (1.59 → 0.78 phonemes). Misaki itself falls back to
  eSpeak `en-gb` here.

## Benchmark

`tts-benchmark --backend paradee --corpus minimax-english`, 100 phrases, Parakeet TDT round trip,
M5 Pro, macOS 27, `.cpuOnly`:

| Release | Voice | Variant | WER | CER | RTFx (audio / synth) | synth p50 / p95 |
|---|---|---|---:|---:|---:|---:|
| v1.2 | `af_heart` | int8 | 0.93 % | 0.12 % | 108× | 70 / 85 ms |
| v1.2 | `af_heart` | fp32 | 0.93 % | 0.12 % | 102× | 75 / 99 ms |
| v1.2 | `bf_isabella` | int8 | 0.71 % | 0.18 % | 106× | 69 / 86 ms |
| v1.2 | `am_puck` | int8 | 1.00 % | 0.16 % | 110× | 68 / 83 ms |
| v1.2 | `bm_fable` | int8 | 0.84 % | 0.23 % | 97× | 74 / 104 ms |
| v1.0 | `af_heart` | int8 | 1.20 % | 0.14 % | 97× | 78 / 96 ms |

Synth time covers the text frontend and both models. The models alone run ~200× real time (v1.2) and ~150×
(v1.0); the rest is the shared English G2P. Most of the remaining WER is the corpus's own text ("will - power",
"date.It's", "five thousand" → "5,000") and homophones ("knights" → "nights"); the British voices add a few ASR
slips ("the" → "their", "transported" → "transporting").

Against upstream (Python, 10 inputs incl. a 28 s paragraph, each v1.2 voice): 0 duration mismatches and identical
lengths vs PyTorch with matched noise; log-mel distance to the released ONNX ≤ 0.012; Whisper WER equal to the
upstream ONNX for fp32 (3.39 %) and int8 (3.28 %).
