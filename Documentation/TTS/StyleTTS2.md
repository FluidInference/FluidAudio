# StyleTTS2 (LibriTTS, iteration_3) Swift Inference

Reference-audio–driven zero-shot TTS via an 8-stage CoreML pipeline.
24 kHz mono Float32 output, English only, ADPM2 diffusion sampler with
α/β style blending against a speaker reference clip.

## Overview

StyleTTS2 conditions every utterance on a short speaker reference: the
text is encoded against a 256-dim style vector (128 dims AdaIN reference
+ 128 dims prosody) split between a 5-step Karras-schedule diffusion
sampler and a reference encoder run over the speaker's mel. Adjusting
α (`defaultAlpha = 0.3`) and β (`defaultBeta = 0.7`) re-weights how
much of each slot comes from the diffusion-sampled style versus the
reference encoder. Output is a HiFi-GAN waveform at 24 kHz.

The Swift port targets the `iteration_3/compiled/` flavor on HuggingFace
— a re-trace of the upstream LibriTTS checkpoint with several stages
fused (`f0n_har_source`, `diffusion_sampler`) and the bucketed
`bert` / `fused_diffusion_sampler` axes split into T = 64 / 128 / 256
variants so callers don't need a single dynamic-shape model.

## Quick Start

### CLI

```bash
swift run fluidaudiocli tts "Hello from StyleTTS2." \
    --backend styletts2 \
    --reference path/to/speaker.wav \
    --output ~/Desktop/styletts2-demo.wav
```

`--reference` is required and must be readable by `AVAudioFile` (WAV /
AIFF / CAF / m4a — any sample rate / channel layout, internally
resampled to 24 kHz mono). On first invocation the CLI downloads the
8 default-path `.mlmodelc` bundles plus the shared lexicon cache + BART
G2P assets; subsequent calls reuse the disk cache.

Optional flags:

| Flag | Default | Notes |
|---|---|---|
| `--styletts2-alpha <f>` | `0.3` | Reference-side blend weight (0.0 = pure diffusion, 1.0 = pure reference) |
| `--styletts2-beta <f>` | `0.7` | Prosody-side blend weight (0.0 = pure diffusion, 1.0 = pure reference) |
| `--styletts2-ipa <s>` | — | Skip the lexicon + G2P pipeline; feed an IPA string directly (espeak parity escape hatch) |
| `--seed <u64>` | `0` | Diffusion sampler RNG seed; same seed → bit-identical audio |

### Swift

```swift
import FluidAudio

let manager = try await StyleTTS2Manager.downloadAndCreate(
    computeUnits: .cpuAndNeuralEngine
)

let samples = try await manager.synthesize(
    text: "Hello from StyleTTS2.",
    referenceAudioURL: refURL
)
// `samples`: Float32 PCM, 24 kHz mono.
```

`StyleTTS2Manager.synthesize(...)` is overloaded three ways:

```swift
// Text path (default — Misaki lexicon + BART G2P → tokens)
public func synthesize(text:referenceAudioURL:alpha:beta:noiseSeed:) async throws -> [Float]

// IPA path (skip phonemizer; useful when you have espeak server-side)
public func synthesize(ipa:referenceAudioURL:alpha:beta:noiseSeed:) async throws -> [Float]

// Pre-computed reference mel (cache once, reuse across many prompts)
public func synthesize(tokenIds:referenceMel:referenceMelFrames:alpha:beta:noiseSeed:) async throws -> [Float]
```

Use `manager.referenceMel(from:)` to compute the reference mel once and
hand it to the third overload — saves the FFT + mel work on subsequent
calls against the same speaker.

## Pipeline

```
text              reference audio (24 kHz mono)
  |                       |
  v                       v
StyleTTS2Phonemizer   StyleTTS2MelExtractor
  |  (Misaki lexicon       |  (HTK 80-bin log-mel,
  |   + BART G2P fallback) |   16 kHz filterbank quirk*)
  |                       |
  | tokenIds              | mel [1, 1, 80, F]
  v                       v
StyleTTS2Synthesizer  (8 CoreML stages)
  ├─ text_encoder            tokens → text hidden states
  ├─ bert (T=57|64|128|256)  contextual embeddings (bucketed)
  ├─ ref_encoder             reference mel → s_ref [1, 256]
  ├─ fused_diffusion_sampler 5-step ADPM2 → s_diff [1, 256]
  │     (T=57|64|128|256 bucketed; needs 4 aux noises)
  ├─ duration_predictor      per-token duration logits → frames
  ├─ fused_f0n_har_source    F0 + N + harmonic source generator
  ├─ decoder_pre             alignment + style mixing
  └─ decoder_upsample        HiFi-GAN generator → 24 kHz audio
```

*The mel filterbank uses `melFilterSampleRate = 16_000` even though the
audio is loaded at 24 kHz — the upstream `make_preprocess()` doesn't
override `sample_rate` on `torchaudio.transforms.MelSpectrogram`, so the
Swift extractor replicates that quirk to keep the mel bins aligned
with what the model saw during training.

## Files

| File | Role |
|---|---|
| `StyleTTS2Manager.swift` | Public actor — `initialize()`, `downloadAndCreate()`, three `synthesize()` overloads, `referenceMel(from:)`, `cleanup()` |
| `StyleTTS2Constants.swift` | All trace-time constants (sample rate, hop, mel filterbank, diffusion schedule, α/β defaults, style dim) |
| `StyleTTS2Error.swift` | Per-module `Error, LocalizedError` enum |
| `Assets/StyleTTS2ModelStore.swift` | Actor — loads the 8 default models with per-stage compute-units, lazy-loads bucket variants |
| `Assets/StyleTTS2ResourceDownloader.swift` | HuggingFace pull for `FluidInference/StyleTTS-2-coreml/iteration_3/compiled/` + ensures the shared lexicon cache + G2P assets |
| `Pipeline/Tokenizer/StyleTTS2Phonemizer.swift` | Misaki lexicon-cache lookup → BART G2P fallback → Misaki diphthong shorthand expansion |
| `Pipeline/Tokenizer/StyleTTS2TextCleaner.swift` | Symbol vocab + `encode(_:)` (leading pad token + per-symbol IDs) |
| `Pipeline/Preprocess/StyleTTS2MelExtractor.swift` | HTK 80-bin log-mel of the reference audio |
| `Pipeline/Synthesize/StyleTTS2Synthesizer.swift` | 8-stage CoreML driver — text_encoder → bert → ref_encoder → sampler → duration → f0n → decoder_pre → decoder_upsample |
| `Pipeline/Synthesize/StyleTTS2GlueOps.swift` | α/β style blend, alignment matrix construction, frame-count derivation |
| `Pipeline/Synthesize/StyleTTS2MultiArray.swift` | `MLMultiArray` ergonomics shared by the synthesizer stages |
| `Pipeline/Sampler/StyleTTS2DiffusionSchedule.swift` | Karras σ schedule + ADPM2 step coefficients for the fused sampler |

The English BART G2P model (`G2PModel`) lives under
`TTS/G2P/G2PModel.swift` — shared with KokoroAne.

## Phonemizer

StyleTTS2 was trained on espeak transcripts with stress markers
(`with_stress=True`). FluidAudio cannot ship the espeak C library, so
the default text path mirrors KokoroAne's tokenizer:

1. **Case-sensitive original spelling** in the Misaki lexicon
   (proper nouns, abbreviations like `AI`, `iPhone`).
2. **Case-sensitive normalized form** (`normalizeKey` lowercases +
   strips non-letter/digit/apostrophe).
3. **Lower-case lexicon** hit.
4. **BART G2P CoreML model** (`G2PEncoder.mlmodelc` / `G2PDecoder.mlmodelc`)
   — last resort for OOV English words.

After resolution, **Misaki's 5-char ASCII diphthong shorthand** is
expanded to the two-char espeak IPA the model was trained on:

| Misaki | Espeak IPA |
|---|---|
| `A` | `eɪ` |
| `O` | `oʊ` |
| `I` | `aɪ` |
| `Y` | `ɔɪ` |
| `W` | `aʊ` |

Without this expansion the encoder treats e.g. `O` as the Latin
uppercase letter (token id 30) instead of /oʊ/ — every word
containing /eɪ/, /oʊ/, /aɪ/, /ɔɪ/, /aʊ/ ends up as gibberish in the
synthesized audio. Confirmed by ASR round-trip.

**Lookup parity is good but not perfect.** Output is intelligible but
may not always reproduce the exact stress markers espeak would emit.
Callers with a higher-quality phonemizer (server-side espeak, custom
IPA frontend) can bypass the entire stack via
`StyleTTS2Manager.synthesize(ipa:referenceAudioURL:...)`.

## Bucket variants

Two stages — `bert` and `fused_diffusion_sampler` — can't accept
`RangeDim` on the token axis at trace time, so they ship as four
separate `.mlmodelc` bundles each:

| Bucket | Token capacity | When loaded |
|---|---|---|
| Default | T = 57 | Always loaded by `loadIfNeeded()` |
| T64 | ≤ 64 | First prompt > 57 tokens |
| T128 | ≤ 128 | First prompt > 64 tokens |
| T256 | ≤ 256 | First prompt > 128 tokens |

`StyleTTS2ModelStore.bertModel(forTokenCount:)` and
`samplerModel(forTokenCount:)` pick the smallest bucket that fits and
download + load it on demand. Prompts > 256 tokens throw
`StyleTTS2Error.noBucketAvailable(tokenCount:)` — chunk longer text
upstream.

## Per-stage compute units

`StyleTTS2ModelStore.loadIfNeeded()` assigns each stage a placement
matching the post-Trials 4 + 6 + 8b decisions documented in
`mobius/models/tts/styletts2/coreml/inference.py`:

| Stage | Compute units |
|---|---|
| `text_encoder` | `.cpuOnly` |
| `bert` (+ buckets) | `.all` |
| `ref_encoder` | `.cpuAndGPU` |
| `fused_diffusion_sampler` (+ buckets) | `.all` |
| `duration_predictor` | `.cpuOnly` |
| `fused_f0n_har_source` | `.cpuOnly` |
| `decoder_pre` | `.cpuAndNeuralEngine` |
| `decoder_upsample` | `.cpuOnly` |

Pass `computeUnits: .cpuOnly` on `StyleTTS2Manager.init` to force
every stage to CPU regardless (debugging only).

## Reference audio

- Any sample rate / channel layout — internally resampled to 24 kHz mono
  via `AudioConverter`.
- The mel extractor is reflect-padded by `nFFT/2 = 1024` samples on
  each side; `frames = 1 + audio.count / hop`. The `ref_encoder` graph
  expects whatever frame count the audio yields — **no truncation /
  padding to a fixed target**. A ~3 second clip works well in practice.
- Cache the mel for repeat speakers via `manager.referenceMel(from:)`
  → pass to `synthesize(tokenIds:referenceMel:referenceMelFrames:...)`
  to skip the FFT + mel work on each call.

## Diffusion sampler

5-step Karras-σ ADPM2 schedule baked into
`fused_diffusion_sampler_fp16.mlmodelc`:

| Constant | Value |
|---|---|
| `diffusionSteps` | 5 |
| `sigmaMin` | 0.0001 |
| `sigmaMax` | 3.0 |
| `rhoSchedule` | 9.0 |

The fused graph requires exactly `noiseSteps - 1 = 4` auxiliary noise
vectors. `noiseSeed` drives a deterministic RNG over those four
vectors so the same seed + same text + same reference reproduces audio
bit-for-bit.

## Known issues

- **Phonemizer parity gap.** Misaki + BART G2P approximates the espeak
  output StyleTTS2 was trained on. Common words match well after the
  diphthong shorthand expansion; rare / technical / loanword
  pronunciations may drift. Pass IPA directly via
  `synthesize(ipa:...)` when accuracy matters.
- **English only.** The bundled lexicon and G2P model are en-US.
  iteration_3 was trained on LibriTTS — non-English text routes
  through the BART fallback and produces nonsense.
- **Reference clip dependence.** Synthesis quality is bounded by the
  reference clip's recording quality. Noisy / heavily-compressed
  references bleed into the output. Prefer clean studio-style
  recordings ≥ 2 s long.

## Models

All CoreML packages live under
[`FluidInference/StyleTTS-2-coreml/iteration_3/compiled/`](https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/iteration_3/compiled).
The flat root-level `styletts2_decoder_*.mlpackage` /
`styletts2_text_predictor_*.mlpackage` artifacts are a legacy layout
and are **not** consumed by the Swift port.

| Stage | mlmodelc | Precision |
|---|---|---|
| Text encoder | `text_encoder_fp16.mlmodelc` | fp16 |
| BERT (default + 3 buckets) | `bert_fp16.mlmodelc`, `bert_fp16_t{64,128,256}.mlmodelc` | fp16 |
| Reference encoder | `ref_encoder_fp16.mlmodelc` | fp16 |
| Diffusion sampler (default + 3 buckets) | `fused_diffusion_sampler_fp16.mlmodelc`, `fused_diffusion_sampler_fp16_t{64,128,256}.mlmodelc` | fp16 |
| Duration predictor | `duration_predictor_fp16.mlmodelc` | fp16 |
| F0 + N + harmonic source | `fused_f0n_har_source.mlmodelc` | fp32 |
| Decoder (pre) | `decoder_pre_fp16.mlmodelc` | fp16 |
| Decoder (upsample / HiFi-GAN) | `decoder_upsample_fp16.mlmodelc` | fp16 |

The shared phonemizer assets (`G2PEncoder.mlmodelc`,
`G2PDecoder.mlmodelc`, `g2p_vocab.json`, `us_lexicon_cache.json`) are
fetched from
[`FluidInference/kokoro-82m-coreml`](https://huggingface.co/FluidInference/kokoro-82m-coreml)
and cached under `~/.cache/fluidaudio/Models/kokoro/`, the same
location KokoroAne reuses — a single download serves both backends.

## License

- **StyleTTS2 model weights:** MIT, inherited from
  [`yl4579/StyleTTS2`](https://github.com/yl4579/StyleTTS2) upstream
  (LibriTTS checkpoint).
- **FluidAudio SDK:** Apache 2.0.
