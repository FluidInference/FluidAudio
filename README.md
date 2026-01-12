![banner.png](banner.png)

# FluidAudio - Transcription, Text-to-speech, VAD, Speaker diarization with CoreML Models

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face%20Models-100k%2B%20downloads-brightgreen?logo=huggingface)](https://huggingface.co/FluidInference)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/FluidAudio)

FluidAudio is a Swift SDK for fully local, low-latency audio AI on Apple devices, with inference offloaded to the Apple Neural Engine (ANE), resulting in less memory and generally faster inference.

The SDK includes state-of-the-art speaker diarization, transcription, and voice activity detection via open-source models (MIT/Apache 2.0) that can be integrated with just a few lines of code. Models are optimized for background processing, ambient computing and always on workloads by running inference on the ANE, minimizing CPU usage and avoiding GPU/MPS entirely.

For custom use cases, feedback, additional model support, or platform requests, join our [Discord](https://discord.gg/WNsvaCtmDe). We’re also bringing visual, language, and TTS models to device and will share updates there.

Below are some featured local AI apps using Fluid Audio models on macOS and iOS:

<p align="left">
  <a href="https://github.com/Beingpax/VoiceInk/"><img src="Documentation/assets/voiceink.png" height="40" alt="Voice Ink"></a>
  <a href="https://spokenly.app/"><img src="Documentation/assets/spokenly.png" height="40" alt="Spokenly"></a>
  <a href="https://slipbox.ai/"><img src="Documentation/assets/slipbox.png" height="40" alt="Slipbox"></a>
  
  <!-- Add your app: submit logo via PR -->
</p>

Want to convert your own model? Check [möbius](https://github.com/FluidInference/mobius)

## Highlights

- **Automatic Speech Recognition (ASR)**: Parakeet TDT v3 (0.6b) for transcription; supports all 25 European languages
- **Speaker Diarization (Online + Offline)**: Speaker separation and identification across audio streams. Streaming pipeline for real-time processing and offline batch pipeline with advanced clustering.
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **Voice Activity Detection (VAD)**: Voice activity detection with Silero models
- **Real-time Processing**: Designed for near real-time workloads but also works for offline processing
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption
- **Open-Source Models**: All models are publicly available on HuggingFace — converted and optimized by our team; permissive licenses

## Video Demos

| Link | Description |
| --- | --- |
| **[Spokenly Real-time ASR](https://www.youtube.com/watch?v=9fXKKkyL8JE)** | Video demonstration of FluidAudio's transcription accuracy and speed |
| **[Senko Integration](https://x.com/hamza_q_/status/1970228971657928995)** | Python Speaker diarization on Mac using FluidAudio's segmentation model |
| **[Mobile TTS](https://x.com/sach1n/status/1977817056507793521)** | Voice on mobile video using FluidAudio's Kokoro and Silero models |

## Showcase

Make a PR if you want to add your app, please keep it in chronological order.

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Senko](https://github.com/narcotic-sh/senko)** | A very fast and accurate speaker diarization pipeline. A [good example](https://github.com/narcotic-sh/senko/commit/51dbd8bde764c3c6648dbbae57d6aff66c5ca15c) for how to integrate FluidAudio into a Python app |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |
| **[Altic/Fluid Voice](https://github.com/altic-dev/Fluid-oss)** | Lightweight Fully free and Open Source Voice to Text dictation for macOS built using FluidAudio. Never pay for dictation apps |
| **[Paraspeech](https://paraspeech.com)** | AI powered voice to text. Fully offline. No subscriptions. |
| **[mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)** | Comparison of different local ASR, including one of the first verions of FluidAudio's ASR models |
| **[Starling](https://github.com/Ryandonofrio3/Starling)** | Open Source, fully local voice-to-text transcription with auto-paste at your cursor. |
| **[BoltAI](https://boltai.com/)** | Write content 10x faster using parakeet models |
| **[Voxeoflow](https://www.voxeoflow.app)** | Mac dictation app with real-time translation. Lightning-fast transcription in over 100 languages, instantly translated to your target language. |
| **[Speakmac](https://speakmac.app)** | Mac app that lets you type anywhere on your Mac using your voice. Fully local, private dictation built on FluidAudio. |
| **[SamScribe](https://github.com/Steven-Weng/SamScribe)** | An open-source macOS app that captures and transcribes audio from your microphone and meeting applications (Zoom, Teams, Chrome) in real-time, with cross-session speaker recognition. |
| **[WhisKey](https://whiskey.asktobuild.app/)** | Privacy-first voice dictation keyboard for iOS and macOS. On-device transcription with 12+ languages, AI meeting summaries, and mindmap generation. Great for daily use and vibe-coding. Uses speaker diarization. |

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.7.9"),
],
```

### Choosing a Product

FluidAudio provides two library products:

- **`FluidAudio`** (default) - Core functionality: ASR, diarization, VAD
  - Lightweight, no GPL dependencies
  - Recommended for most apps

- **`FluidAudioTTS`** - Text-to-Speech (Kokoro)
  - Separate optional product
  - Includes ESpeakNG framework (GPL-3.0)
  - Only bundled if you explicitly add it as a dependency

**In Xcode:**
1. Add the FluidAudio package to your project
2. In the "Add Package" dialog, select your desired product(s):
   - `FluidAudio` for core features (ASR, diarization, VAD)
   - `FluidAudioTTS` if you need text-to-speech
3. Add the selected product(s) to your app target

**In Package.swift:**
```swift
// Core features only (no GPL dependencies):
.product(name: "FluidAudio", package: "FluidAudio")

// Add TTS support (includes GPL ESpeakNG):
.product(name: "FluidAudioTTS", package: "FluidAudio")
```

**In Xcode:**
1. Add the FluidAudio package to your project
2. In the "Add Package" dialog, select your desired product:
   - `FluidAudio` for core features only
   - `FluidAudioWithTTS` if you need text-to-speech
3. Add the selected product to your app target

**CocoaPods:** We recommend using [cocoapods-spm](https://github.com/trinhngocthuyen/cocoapods-spm) for better SPM integration, but if needed, you can also use our podspec: `pod 'FluidAudio', '~> 0.7.8'`

> **Note:** The Kokoro TTS tooling currently ships arm64-only dependencies. See the [arm64 build requirements](Documentation/TTS/README.md#arm64-only-builds) guide if you hit linker errors targeting x86_64.

## Configuration

### Quick Reference

Both solve the same problem: **"I can't reach HuggingFace directly."** They're alternative approaches - pick whichever matches your setup:

| Scenario | Solution | Configuration |
|----------|----------|---|
| You have a **local mirror or internal model server** | Use Registry URL override | `REGISTRY_URL=https://your-mirror.com` |
| You're **behind a corporate firewall** with a proxy that can reach HuggingFace | Use Proxy configuration | `https_proxy=http://proxy.company.com:8080` |

**How they work:**
- **Registry URL** - App requests from `your-mirror.com` instead of `huggingface.co`
- **Proxy** - App still requests `huggingface.co`, but traffic routes through proxy to reach it

In most cases, you only need one. (You'd use both only if your mirror is behind the proxy and unreachable without it.)

<details>
<summary><b>Model Registry URL</b> - Change download destination</summary>

By default, FluidAudio downloads models from HuggingFace. You can override this to use a mirror, local server, or air-gapped environment.

**Programmatic override (recommended for apps):**
```swift
import FluidAudio

// Set custom registry before using any managers
ModelRegistry.baseURL = "https://your-mirror.example.com"

// Models will now download from the custom registry
let diarizer = DiarizerManager()
```

**Environment Variables (recommended for CLI/testing):**
```bash
# Use custom registry
export REGISTRY_URL=https://your-mirror.example.com
swift run fluidaudio transcribe audio.wav

# Or use the MODEL_REGISTRY_URL alias
export MODEL_REGISTRY_URL=https://models.internal.corp
swift run fluidaudio diarization-benchmark --auto-download
```

**Xcode Scheme Configuration:**
1. Edit Scheme → Run → Arguments
2. Go to **Environment Variables** tab
3. Click `+` and add: `REGISTRY_URL` = `https://your-mirror.example.com`
4. The custom registry will apply to all debug runs

</details>

<details>
<summary><b>Proxy Configuration</b> - Route downloads through a proxy server</summary>

If you're behind a corporate firewall and cannot reach HuggingFace (or your registry) directly, configure a proxy to forward requests:

Set the `https_proxy` environment variable:

```bash
export https_proxy=http://proxy.company.com:8080
# or for authenticated proxies:
export https_proxy=http://user:password@proxy.company.com:8080

swift run fluidaudio transcribe audio.wav
```

**Xcode Scheme Configuration for Proxy:**
1. Edit Scheme → Run → Arguments
2. Go to **Environment Variables** tab
3. Click `+` and add: `https_proxy` = `http://proxy.company.com:8080`
4. FluidAudio will automatically route downloads through the proxy

</details>

## Documentation

**[DeepWiki](https://deepwiki.com/FluidInference/FluidAudio)** for auto-generated docs for this repo.

### Documentation Index

- Guides
  - [Audio Conversion for Inference](Documentation/Guides/AudioConversion.md)
  - Manual model download & loading options: [ASR](Documentation/ASR/ManualModelLoading.md), [Diarizer](Documentation/SpeakerDiarization.md#manual-model-loading), [VAD](Documentation/VAD/GettingStarted.md#manual-model-loading)
  - Routing Hugging Face (or compatible) requests through a proxy? Set `https_proxy` before running the download helpers (see [Documentation/API.md](Documentation/API.md:9)).
  - [Kokoro TTS arm64 build requirements](Documentation/TTS/README.md#arm64-only-builds)
- Models
  - Automatic Speech Recognition/Transcription
    - [Getting Started](Documentation/ASR/GettingStarted.md)
    - [Last Chunk Handling](Documentation/ASR/LastChunkHandling.md)
  - Speaker Diarization
    - [Speaker Diarization Guide](Documentation/SpeakerDiarization.md)
  - VAD: [Getting Started](Documentation/VAD/GettingStarted.md)
    - [Segmentation](Documentation/VAD/Segmentation.md)
    - [Model Conversion Code](https://github.com/FluidInference/mobius)
- [Benchmarks]([Documentation/Benchmarks.md])
- [API Reference](Documentation/API.md)
- [Command Line Guide](Documentation/CLI.md)

### MCP Server

The repo is indexed by DeepWiki MCP server, so your coding tool can access the docs:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

For claude code:

```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```

## Automatic Speech Recognition (ASR) / Transcription

- **Models**:
  - `FluidInference/parakeet-tdt-0.6b-v3-coreml` (multilingual, 25 European languages)
  - `FluidInference/parakeet-tdt-0.6b-v2-coreml` (English-only, highest recall)
- **Processing Mode**: Batch transcription for complete audio files
- **Real-time Factor**: ~190x on M4 Pro (processes 1 hour of audio in ~19 seconds)
- **Streaming Support**: Coming soon — batch processing is recommended for production use
- **Backend**: Same Parakeet TDT v3 model powers our backend ASR

### ASR Quick Start

```swift
import FluidAudio

// Batch transcription from an audio file
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad(version: .v3)  // Switch to .v2 for English-only work
    let asrManager = AsrManager(config: .default)
    try await asrManager.initialize(models: models)

    // 3) Transcribe the audio 16hz, already converted
    let result = try await asrManager.transcribe(samples)

    // 3) Transcribe a file
    // let url = URL(fileURLWithPath: sample.audioPath)

    // 3) Transcribe AVAudioPCMBuffer
    // let result = try await asrManager.transcribe(audioBuffer)
    print("Transcription: \(result.text)")
}
```

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# English-only run with higher recall
swift run fluidaudio transcribe audio.wav --model-version v2
```

## Speaker Diarization

### Offline Speaker Diarization Pipeline

Pyannote Community-1 pipeline (powerset segmentation + WeSpeaker + VBx) for offline speaker diarization. Use this for most use cases, see Benchmarkds.md for benchmarks.

```swift
import FluidAudio

let config = OfflineDiarizerConfig()
let manager = OfflineDiarizerManager(config: config)
try await manager.prepareModels()  // Downloads + compiles Core ML bundles if they are missing

let samples = try AudioConverter().resampleAudioFile(path: "meeting.wav")
let result = try await manager.process(audio: samples)

for segment in result.segments {
    print("\(segment.speakerId) \(segment.startTimeSeconds)s → \(segment.endTimeSeconds)s")
}
```

For processing audio files, use the file-based API which automatically uses memory-mapped streaming for efficiency:

```swift
let url = URL(fileURLWithPath: "meeting.wav")
let result = try await manager.process(url)

for segment in result.segments {
    print("\(segment.speakerId) \(segment.startTimeSeconds)s → \(segment.endTimeSeconds)s")
}
```

```bash
# Process a meeting with full VBx clustering
swift run fluidaudio process ~/FluidAudioDatasets/ami_official/sdm/ES2004a.Mix-Headset.wav \
  --mode offline --threshold 0.6 --output es2004a_offline.json

# Run the AMI single-file benchmark with automatic downloads
swift run fluidaudio diarization-benchmark --mode offline --auto-download \
  --single-file ES2004a --threshold 0.6 --output offline_results.json
```

`offline_results.json` contains DER/JER/RTFx along with timing breakdowns for segmentation, embedding extraction, and VBx clustering. CI now runs this workflow on every PR to ensure the offline models stay healthy and the Hugging Face assets remain accessible.

### Streaming/Online Speaker Diarization

Use this if you need to show speaker labels while the transcription is happening, in most use cases, offline should be more than enough.

```swift
import FluidAudio

// Diarize an audio file
Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager() 
    diarizer.initialize(models: models)

    // Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/meeting.wav")

    // Run diarization
    let result = try diarizer.performCompleteDiarization(samples)
    for segment in result.segments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

For diarization streaming see [Documentation/SpeakerDiarization.md](Documentation/SpeakerDiarization.md)

```bash
swift run fluidaudio diarization-benchmark --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2
```

### CLI

```bash
# Process an individual file and save JSON
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6
```

## Voice Activity Detection (VAD)

Silero VAD powers our on-device detector. The latest release surfaces the same
timestamp extraction and streaming heuristics as the upstream PyTorch
implementation. Ping us on Discord if you need help tuning it for your
environment.

### VAD Quick Start (Offline Segmentation)

Simple call to return chunk-level probabilities every 256 ms hop:

```swift
let results = try await manager.process(samples)
for (index, chunk) in results.enumerated() {
    print(
        String(
            format: "Chunk %02d: prob=%.3f, inference=%.4fs",
            index,
            chunk.probability,
            chunk.processingTime
        )
    )
}
```

The following are higher level APIs better suited to integrate with other systems

```swift
import FluidAudio

Task {
    let manager = try await VadManager(
        config: VadConfig(defaultThreshold: 0.75)
    )

    let audioURL = URL(fileURLWithPath: "path/to/audio.wav")
    let samples = try AudioConverter().resampleAudioFile(audioURL)

    var segmentation = VadSegmentationConfig.default
    segmentation.minSpeechDuration = 0.25
    segmentation.minSilenceDuration = 0.4

    let segments = try await manager.segmentSpeech(samples, config: segmentation)
    for segment in segments {
        print(
            String(format: "Speech %.2f–%.2fs", segment.startTime, segment.endTime)
        )
    }
}
```

### Streaming

```swift
import FluidAudio

Task {
    let manager = try await VadManager()
    var state = await manager.makeStreamState()

    for chunk in microphoneChunks {
        let result = try await manager.processStreamingChunk(
            chunk,
            state: state,
            config: .default,
            returnSeconds: true,
            timeResolution: 2
        )

        state = result.state

        // Access raw probability (0.0-1.0) for custom logic
        print(String(format: "Probability: %.3f", result.probability))

        if let event = result.event {
            let label = event.kind == .speechStart ? "Start" : "End"
            print("\(label) @ \(event.time ?? 0)s")
        }
    }
}
```

### CLI

Start with the general-purpose `process` command, which runs the diarization
pipeline (and therefore VAD) end-to-end on a single file:

```bash
swift run fluidaudio process path/to/audio.wav
```

Once you need to experiment with VAD-specific knobs directly, reach for:

```bash
# Inspect offline segments (default mode)
swift run fluidaudio vad-analyze path/to/audio.wav

# Streaming simulation only (timestamps printed in seconds by default)
swift run fluidaudio vad-analyze path/to/audio.wav --streaming

# Benchmark accuracy/precision trade-offs
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3
```

`swift run fluidaudio vad-analyze --help` lists every tuning option, including
negative-threshold overrides, max-speech splitting, padding, and chunk size.
Offline mode also reports RTFx using the model's per-chunk processing time.

## Text‑To‑Speech (TTS)

> **⚠️ Beta:** The TTS system is currently in beta and only supports American English. Additional language support is planned for future releases.

- Model: Kokoro (CoreML unified model)
- Language: American English (beta)
- G2P: Dictionary first, then eSpeak NG (CEspeakNG) for OOV words
- Output: 24 kHz mono WAV

Requirements (macOS)
Ensure eSpeak NG headers/libs are available via pkg-config (`espeak-ng`).
<https://github.com/espeak-ng/espeak-ng/tree/master>

**For CLI usage:**
- The `fluidaudio` CLI is built with TTS support enabled by default:
  - `swift run fluidaudio tts "Hello" --output out.wav`
  - `swift build` (builds CLI with TTS)
  - `swift test` (runs tests including TTS coverage)

### Quick Start (CLI)

```bash
# First run will download the Kokoro model and vocab
swift run fluidaudio tts "Hello from FluidAudio." --auto-download --output out.wav

# Another example with punctuation and OOV handling
swift run fluidaudio tts "Edge-cases: URLs like https://example.com and e-mail test@example.com." --output out2.wav
```

Notes

- The TTS pipeline uses a word→phoneme dictionary first; unknown words are phonemized with eSpeak NG (C API) and mapped to the model’s token set.
- OOV words are printed with their IPA and mapped tokens for visibility during synthesis.
- We do not prepend any “language token” to avoid leading vowel artifacts.

### Quick Start (Code)

```swift
import FluidAudio

Task {
  do {
    let data = try await KokoroModel.synthesize(text: "Hello from FluidAudio.")
    try data.write(to: URL(fileURLWithPath: "out.wav"))
  } catch {
    print("TTS error: \(error)")
  }
}
```

Troubleshooting
Build requires eSpeak NG headers/libs for the C API discoverable via pkg-config (`espeak-ng`).

- If SwiftPM cannot find headers, build with explicit paths:
  - `swift build -Xcc -I/opt/homebrew/include -Xlinker -L/opt/homebrew/lib`
- Dictionary and model assets are cached under `~/.cache/fluidaudio/Models/kokoro`.

## Continuous Integration

- `tests.yml`: Default build matrix covering SwiftPM tests and an iOS archive smoke test.
- `diarizer-benchmark.yml`: Runs the streaming diarization benchmark on ES2004a for regression tracking.
- `offline-pipeline.yml`: Executes the VBx offline pipeline end-to-end (`fluidaudio diarization-benchmark --mode offline`) and fails if DER/JER drift beyond guardrails or if models fail to download. Use this workflow as a reference for provisioning model caches in your own CI.

## Everything Else

### FAQs

- CLI is available on macOS only. For iOS, use the library programmatically.
- Models auto-download on first use. If your network restricts Hugging Face access, set an HTTPS proxy: `export https_proxy=http://127.0.0.1:7890`.
- Windows alternative in development: [fluid-server](https://github.com/FluidInference/fluid-server)
- If you're looking to get the system audio on a Mac, take a look at this repo for reference [AudioCap](https://github.com/insidegui/AudioCap/tree/main)

### License

Apache 2.0 — see `LICENSE` for details.

### Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques.

Pyannote: <https://github.com/pyannote/pyannote-audio>

Wewpeaker: <https://github.com/wenet-e2e/wespeaker>

Parakeet-mlx: <https://github.com/senstella/parakeet-mlx>

silero-vad: <https://github.com/snakers4/silero-vad>

Kokoro-82M: <https://huggingface.co/hexgrad/Kokoro-82M>

### Citation

If you use FluidAudio in your work, please cite:

FluidInference Team. (2024). FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms (Version 0.7.0) [Computer software]. GitHub. <https://github.com/FluidInference/FluidAudio>

```bibtex
@software{FluidInferenceTeam_FluidAudio_2024,
  author = {{FluidInference Team}},
  title = {{FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms}},
  year = {2024},
  month = {12},
  version = {0.7.0},
  url = {https://github.com/FluidInference/FluidAudio},
  note = {Computer software}
}
```

---

## Show Your Support

Help the Fluid Inference community grow by adding a "Powered by Fluid Inference" badge to your project!

<p align="center">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</p>

Copy and paste this prompt to your coding agent where you host your homepage:

```text
Add a centered 'Powered by Fluid Inference' badge to the footer linking to fluidinference.com. Image: https://assets.inference.plus/fi-badge.png. Whitelist the image hostname in your framework's config.
```

Or use one of these code snippets:

<details>
<summary>React/Next.js</summary>

```jsx
<div className="flex justify-center py-8">
  <a href="https://fluidinference.com">
    <img
      src="https://assets.inference.plus/fi-badge.png"
      alt="Powered by Fluid Inference"
      height={80}
    />
  </a>
</div>
```

</details>

<details>
<summary>HTML</summary>

```html
<div style="text-align: center; padding: 20px;">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</div>
```

</details>

<details>
<summary>Markdown</summary>

```markdown
<p align="center">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</p>
```

</details>
