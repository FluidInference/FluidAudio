# Kokoro Text-to-Speech

## Overview

Kokoro is a TTS backend that generates the entire audio representation in one pass (all frames at once) using flow matching over mel spectrograms, then converts to audio with the Vocos vocoder.

## Quick Start

### CLI

```bash
swift run fluidaudiocli tts "Welcome to FluidAudio text to speech" \
  --output ~/Desktop/demo.wav \
  --voice af_heart
```

The first invocation downloads Kokoro models, phoneme dictionaries, and voice embeddings; later runs reuse the cached assets.

### Swift

```swift
import FluidAudio

let manager = KokoroTtsManager()
try await manager.initialize()

let audioData = try await manager.synthesize(text: "Hello from FluidAudio!")

let outputURL = URL(fileURLWithPath: "/tmp/demo.wav")
try audioData.write(to: outputURL)
```

Swap in `manager.initialize(models:)` when you want to preload only the long-form `.fifteenSecond` variant.

## Inspecting Chunk Metadata

```swift
let manager = KokoroTtsManager()
try await manager.initialize()

let detailed = try await manager.synthesizeDetailed(
    text: "FluidAudio can report chunk splits for you.",
    variantPreference: .fifteenSecond
)

for chunk in detailed.chunks {
    print("Chunk #\(chunk.index) -> variant: \(chunk.variant), tokens: \(chunk.tokenCount)")
    print("  text: \(chunk.text)")
}
```

`KokoroSynthesizer.SynthesisResult` also exposes `diagnostics` for per-run variant and audio footprint totals.

## Pipeline

```
text → G2P model → IPA phonemes → Kokoro model → audio
         ↑                ↑
   custom lexicon    SSML <phoneme>
   overrides here    overrides here
```

The G2P (grapheme-to-phoneme) step runs **outside** the model as a preprocessing step using a CoreML BART encoder-decoder. Words found in the built-in lexicon use dictionary lookup; out-of-vocabulary words fall back to the G2P model. You can intercept and edit phonemes before they reach the neural network. This is what enables all pronunciation control features.

## Pronunciation Control

Kokoro supports three ways to override pronunciation:

1. **SSML tags** — `<phoneme>`, `<sub>`, `<say-as>` (cardinal, ordinal, digits, date, time, telephone, fraction, characters). See [SSML.md](SSML.md).
2. **Custom lexicon** — word → IPA mapping files loaded via `setCustomLexicon()`. Entries matched case-sensitive first, then case-insensitive, then normalized. See [CustomPronunciation.md](../ASR/CustomPronunciation.md).
3. **Markdown syntax** — inline `[word](/ipa/)` overrides in the input text. Example: `[Kokoro](/kəˈkɔɹo/)`.

Precedence: custom lexicon > built-in dictionaries > morphological stemming > G2P model.

## Text Preprocessing

Kokoro includes comprehensive text normalization (numbers, currencies, times, decimal numbers, units, abbreviations, dates). SSML processing runs first, then markdown-style overrides, then normalization.

## How It Differs From PocketTTS

| | Kokoro | PocketTTS |
|---|---|---|
| Pipeline | text → CoreML G2P → IPA → model | text → SentencePiece → model |
| Voice conditioning | Style embedding vector | 125 audio prompt tokens |
| Generation | All frames at once | Frame-by-frame autoregressive |
| Flow matching target | Mel spectrogram | 32-dim latent per frame |
| Audio synthesis | Vocos vocoder | Mimi streaming codec |
| Latency to first audio | Must wait for full generation | ~80ms after prefill |
| SSML support | Yes (`<phoneme>`, `<sub>`, `<say-as>`) | No |
| Custom lexicon | Yes (word → IPA) | No |
| Markdown pronunciation | Yes (`[word](/ipa/)`) | No |
| Text preprocessing | Full (numbers, dates, currencies) | Minimal (whitespace, punctuation) |

Kokoro parallelizes across time (fast total, but must wait for everything). PocketTTS is sequential across time (slower total, but audio starts immediately).

PocketTTS cannot support phoneme-level features because it has no phoneme stage — the model was trained on text tokens, not IPA. See [PocketTTS.md](PocketTTS.md) for details on what can and cannot be added.

## V2 Models (ANE-Optimized)

The v2 models (`kokoro_21_5s_v2`, `kokoro_21_15s_v2`) are converted with `compute_precision=FLOAT16`, which moves 833 ops (BERT transformer layers + generator convolutions) to the Apple Neural Engine.

| Metric | V1 (fp32, cpuAndGPU) | V2 (fp16, .all) |
|--------|---------------------|-----------------|
| Median latency (5s) | 417 ms | 250 ms |
| RTFx (5s audio) | 12.0x | 20.0x |
| Speedup | — | **1.67x** |
| Quality | baseline | identical (round-trip TTS→ASR) |

The 6 LSTM ops (duration predictor) remain on CPU — CoreML does not schedule recurrent ops to ANE regardless of precision. The original v1 models are still available on HuggingFace for backward compatibility.

## Known Issues

- **Sibilance in high-pitched voices**: Some female `af_*` voices (e.g. `af_heart`, `af_bella`) produce harsh sibilant sounds (s, sh, z). This is baked into the model output and cannot be fixed with post-processing EQ. Lower-pitched voices (male `am_*` variants and some female voices) are unaffected. See [mobius#23](https://github.com/FluidInference/mobius/issues/23).

- **G2P phoneme mismatch limitation**: FluidAudio currently uses `graphemes_to_phonemes_en_us` (from HuggingFace: [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us)) for grapheme-to-phoneme conversion. The original Kokoro and KittenTTS models were trained using espeak for phoneme generation. This G2P mismatch can cause pronunciation issues in some words (e.g., "hello" and "day" in KittenTTS). We cannot use espeak directly due to licensing constraints. **Need**: An espeak-compatible alternative with a permissive license that produces matching phoneme outputs. This affects any TTS model in FluidAudio that relies on the shared Kokoro G2P pipeline. See [PR #409](https://github.com/FluidInference/FluidAudio/pull/409#issuecomment-2632827330) for examples.

## Enable TTS in Your Project

Kokoro TTS is included in the `FluidAudio` product — no separate product needed.

**Package.swift:**
```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.12.4"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "FluidAudio", package: "FluidAudio")
        ]
    )
]
```

**Import in your code:**
```swift
import FluidAudio
```

### CLI

```bash
swift run fluidaudiocli tts "Welcome to FluidAudio" --output ~/Desktop/demo.wav
```
