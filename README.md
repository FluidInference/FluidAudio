![banner.png](banner.png)

# FluidAudio - Speaker diarization, voice-activity-detection and transcription with CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![All Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/FluidAudio)

Fluid Audio is a Swift SDK for fully local, low-latency audio AI on Apple devices, with inference offloaded to the Apple Neural Engine (ANE), resulting in less memory and generally faster inference.

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
- **Speaker Diarization**: Speaker separation with speaker clustering via Pyannote models
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **Voice Activity Detection (VAD)**: Voice activity detection with Silero models
- **Real-time Processing**: Designed for near real-time workloads but also works for offline processing
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption
- **Open-Source Models**: All models are publicly available on HuggingFace — converted and optimized by our team; permissive licenses

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.6.1"),
],
```

**CocoaPods:** We recommend using [cocoapods-spm](https://github.com/trinhngocthuyen/cocoapods-spm) for better SPM integration, but if needed, you can also use our podspec: `pod 'FluidAudio', '~> 0.6.1'`

Important: When adding FluidAudio as a package dependency, only add the library to your target (not the executable). Select `FluidAudio` library in the package products dialog and add it to your app target.

## Documentation

**[DeepWiki](https://deepwiki.com/FluidInference/FluidAudio)** for auto-generated docs for this repo.

### Documentation Index

- Guides
  - [Audio Conversion for Inference](Documentation/Guides/AudioConversion.md)
  - Manual model download & loading options: [ASR](Documentation/ASR/ManualModelLoading.md), [Diarizer](Documentation/SpeakerDiarization.md#manual-model-loading), [VAD](Documentation/VAD/GettingStarted.md#manual-model-loading)
  - Routing Hugging Face (or compatible) requests through a proxy? Set `https_proxy` before running the download helpers (see [Documentation/API.md](Documentation/API.md:9)).
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

**AMI Benchmark Results** (Single Distant Microphone) using a subset of the files:

- **DER: 17.7%** — Competitive with Powerset BCE 2023 (18.5%)
- **JER: 28.0%** — Outperforms EEND 2019 (25.3%) and x-vector clustering (28.7%)
- **RTF: 0.02x** — Real-time processing with 50x speedup

### Speaker Diarization Quick Start

```swift
import FluidAudio

// Diarize an audio file
Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager()  // Uses optimal defaults (0.7 threshold = 17.7% DER)
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
        config: VadConfig(threshold: 0.75)
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

## Showcase

Make a PR if you want to add your app!

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Senko](https://github.com/narcotic-sh/senko)** | A very fast and accurate speaker diarization pipeline. A [good example](https://github.com/narcotic-sh/senko/commit/51dbd8bde764c3c6648dbbae57d6aff66c5ca15c) for how to integrate FluidAudio into a Python app |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |
| **[Altic/Fluid](https://github.com/altic-dev/Fluid-oss)** | Voice-to-text dictation app for macOS with AI enhancement. |
| **[Paraspeech](https://paraspeech.com)** | AI powered voice to text. Fully offline. No subscriptions. |
| **[mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)** | Comparison of different local ASR, including one of the first verions of FluidAudio's ASR models |

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

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx

silero-vad: https://github.com/snakers4/silero-vad

### Citation

If you use FluidAudio in your work, please cite:

FluidInference Team. (2024). FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms (Version 0.5.1) [Computer software]. GitHub. https://github.com/FluidInference/FluidAudio

```bibtex
@software{FluidInferenceTeam_FluidAudio_2024,
  author = {{FluidInference Team}},
  title = {{FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms}},
  year = {2024},
  month = {12},
  version = {0.5.1},
  url = {https://github.com/FluidInference/FluidAudio},
  note = {Computer software}
}
```
