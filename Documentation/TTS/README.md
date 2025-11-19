# Text-To-Speech (TTS) Code Examples

> **⚠️ Beta:** The TTS system is currently in beta and only supports American English. Additional language support is planned for future releases.

Quick recipes for running the Kokoro synthesis stack.

## Enable TTS in Your Project

### For App/Library Development (Xcode & SwiftPM)

When adding FluidAudio to your Xcode project or Package.swift, select the **`FluidAudioWithTTS`** product to include text-to-speech capabilities:

**Xcode:**
1. File → Add Package Dependencies
2. Enter FluidAudio repository URL
3. In the package product selection dialog, choose **`FluidAudioWithTTS`**
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

### For CLI Development

When developing or running the FluidAudio CLI, TTS support is enabled by default.

**Terminal:**
```bash
swift run fluidaudio tts "Welcome to FluidAudio" --output ~/Desktop/demo.wav

# Or explicitly build/test the CLI with TTS
swift build
swift test
```

## CLI quick start

```bash
swift run fluidaudio tts "Welcome to FluidAudio text to speech" \
  --output ~/Desktop/demo.wav \
  --voice af_heart
```

The first invocation downloads Kokoro models, phoneme dictionaries, and voice embeddings; later runs reuse the
cached assets.

## Swift async usage

```swift
import FluidAudio
import Foundation

@main
struct DemoTTS {
    static func main() async {
        let manager = TtSManager()

        do {
            try await manager.initialize()
            let audioData = try await manager.synthesize(text: "Hello from FluidAudio!")

            let outputURL = URL(fileURLWithPath: "/tmp/fluidaudio-demo.wav")
            try audioData.write(to: outputURL)
            print("Saved synthesized audio to: \(outputURL.path)")
        } catch {
            print("Synthesis failed: \(error)")
        }
    }
}
```

Swap in `manager.initialize(models:)` when you want to preload only the long-form `.fifteenSecond` variant.

## Inspecting chunk metadata

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
