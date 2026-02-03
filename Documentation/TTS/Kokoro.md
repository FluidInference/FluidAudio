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

## SSML Support

Kokoro supports a subset of SSML tags for controlling pronunciation. See [SSML.md](SSML.md) for details.

## How It Differs From PocketTTS

| | Kokoro | PocketTTS |
|---|---|---|
| Text input | Phonemes (IPA via espeak) | Raw text (SentencePiece) |
| Voice conditioning | Style embedding vector | 125 audio prompt tokens |
| Generation | All frames at once | Frame-by-frame autoregressive |
| Flow matching target | Mel spectrogram | 32-dim latent per frame |
| Audio synthesis | Vocos vocoder | Mimi streaming codec |
| Latency to first audio | Must wait for full generation | ~80ms after prefill |

Kokoro parallelizes across time (fast total, but must wait for everything). PocketTTS is sequential across time (slower total, but audio starts immediately).

## Enable TTS in Your Project

### App/Library Development (Xcode & SwiftPM)

When adding FluidAudio to your Xcode project or Package.swift, select the **`FluidAudioWithTTS`** product:

**Xcode:**
1. File > Add Package Dependencies
2. Enter FluidAudio repository URL
3. Choose **`FluidAudioWithTTS`**
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
            .product(name: "FluidAudioWithTTS", package: "FluidAudio")
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
