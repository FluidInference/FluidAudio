# Kokoro: High-Quality Text-to-Speech

## Overview

Kokoro is a high-quality, English-only TTS backend. It generates the entire audio representation in one pass (all frames at once) using flow matching over mel spectrograms, then converts to audio with the Vocos vocoder.

## Quick Start

### CLI

```bash
swift run fluidaudio tts "Welcome to FluidAudio text to speech" \
  --output ~/Desktop/demo.wav \
  --voice af_heart
```

The first invocation downloads Kokoro models, phoneme dictionaries, and voice embeddings; later runs reuse the cached assets.

### Swift

```swift
import FluidAudioTTS

let manager = TtSManager()
try await manager.initialize()

let audioData = try await manager.synthesize(text: "Hello from FluidAudio!")

let outputURL = URL(fileURLWithPath: "/tmp/demo.wav")
try audioData.write(to: outputURL)
```

Swap in `manager.initialize(models:)` when you want to preload only the long-form `.fifteenSecond` variant.

## Inspecting Chunk Metadata

```swift
let manager = TtSManager()
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
text → espeak G2P → IPA phonemes → Kokoro model → audio
         ↑                ↑
   custom lexicon    SSML <phoneme>
   overrides here    overrides here
```

Because espeak runs **outside** the model as a preprocessing step, you can intercept and edit phonemes before they reach the neural network. This is what enables all pronunciation control features.

## Pronunciation Control

Kokoro supports three ways to override pronunciation:

1. **SSML tags** — `<phoneme>`, `<sub>`, `<say-as>` (cardinal, ordinal, digits, date, time, telephone, fraction, characters). See [SSML.md](SSML.md).
2. **Custom lexicon** — word → IPA mapping files loaded via `setCustomLexicon()`. Entries matched case-sensitive first, then case-insensitive, then normalized. See [CustomPronunciation.md](../ASR/CustomPronunciation.md).
3. **Markdown syntax** — inline `[word](/ipa/)` overrides in the input text. Example: `[Kokoro](/kəˈkɔɹo/)`.

Precedence: custom lexicon > built-in dictionaries > grapheme-to-phoneme conversion.

## Text Preprocessing

Kokoro includes comprehensive text normalization (numbers, currencies, times, decimal numbers, units, abbreviations, dates). SSML processing runs first, then markdown-style overrides, then normalization.

## How It Differs From PocketTTS

| | Kokoro | PocketTTS |
|---|---|---|
| Pipeline | text → espeak G2P → IPA → model | text → SentencePiece → model |
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

## Enable TTS in Your Project

### App/Library Development (Xcode & SwiftPM)

When adding FluidAudio to your Xcode project or Package.swift, select the **`FluidAudioTTS`** product:

**Xcode:**
1. File > Add Package Dependencies
2. Enter FluidAudio repository URL
3. Choose **`FluidAudioTTS`**
4. Add it to your app target

**Package.swift:**
```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.7.7"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "FluidAudioTTS", package: "FluidAudio")
        ]
    )
]
```

**Import in your code:**
```swift
import FluidAudio       // Core functionality (ASR, diarization, VAD)
import FluidAudioTTS    // TTS features
```

### CLI Development

TTS support is enabled by default in the CLI:

```bash
swift run fluidaudio tts "Welcome to FluidAudio" --output ~/Desktop/demo.wav
```
