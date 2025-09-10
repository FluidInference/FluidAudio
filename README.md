![banner.png](banner.png)

# FluidAudio - Speaker Diarization, VAD and Transcription with CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)
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

## Highlights

- **Automatic Speech Recognition (ASR)**: Parakeet TDT v3 (0.6b) for transcription; supports all 25 European languages
- **Speaker Diarization**: Speaker separation with speaker clustering via Pyannote models
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **Voice Activity Detection (VAD)**: Voice activity detection with Silero models
- **CoreML Models**: Native Apple CoreML backend with custom-converted models optimized for Apple Silicon
- **Open-Source Models**: All models are publicly available on HuggingFace — converted and optimized by our team; permissive licenses
- **Real-time Processing**: Designed for near real-time workloads but also works for offline processing
- **Cross-platform**: Support for macOS 14.0+ and iOS 17.0+ and Apple Silicon devices
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.4.1"),
],
```

Important: When adding FluidAudio as a package dependency, only add the library to your target (not the executable). Select `FluidAudio` library in the package products dialog and add it to your app target.

## Documentation

- **DeepWiki**: Auto-generated docs for this repo — https://deepwiki.com/FluidInference/FluidAudio
- **Modules & Guides**: See the links in the Table of Contents above.

### MCP

The repo is indexed by the DeepWiki MCP server, so your coding tool can access the docs.

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

Claude Code (CLI):

```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```

## Automatic Speech Recognition (ASR) / Transcription

- **Model**: `FluidInference/parakeet-tdt-0.6b-v3-coreml` - NVIDIA's SOTA 
- **Languages**: 25 European languages (see HF model card)
- **Mode**: Batch transcription for complete audio files (streaming coming soon)
- **RTF**: ~120x on M4 Pro (1 min ≈ 0.5s)

See full guide with code samples and CLI usage: [ASR Getting Started](Documentation/ASR/GettingStarted.md).

### Quick Start (Code)

```swift
import FluidAudio

// Batch transcription from an audio file
Task {
    let models = try await AsrModels.downloadAndLoad()
    let asrManager = AsrManager(config: .default)
    try await asrManager.initialize(models: models)

    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")
    let result = try await asrManager.transcribe(samples, source: .system)
    print(result.text)
}
```

## Speaker Diarization

**AMI Benchmark Results** (Single Distant Microphone) using a subset of the files:

- **DER: 17.7%** — Competitive with Powerset BCE 2023 (18.5%)
- **JER: 28.0%** — Outperforms EEND 2019 (25.3%) and x-vector clustering (28.7%)
- **RTF: 0.02x** — Real-time processing with 50x speedup

```text
RTF = Processing Time / Audio Duration

With RTF = 0.02x:
- 1 minute of audio takes 0.02 × 60 = 1.2 seconds to process
- 10 minutes of audio takes 0.02 × 600 = 12 seconds to process

For real-time speech-to-text:
- Latency: ~1.2 seconds per minute of audio
- Throughput: Can process 50x faster than real-time
- Pipeline impact: Minimal — diarization won't be the bottleneck
```

See full documentation, streaming examples, and CLI usage: [Speaker Diarization Guide](Documentation/SpeakerDiarization.md).

### Quick Start (Code)

```swift
import FluidAudio

Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager()
    diarizer.initialize(models: models)

    let samples = try await loadSamples16kMono(path: "path/to/meeting.wav")
    let result = try diarizer.performCompleteDiarization(samples)
    for seg in result.segments {
        print("Speaker \(seg.speakerId): \(seg.startTimeSeconds)-\(seg.endTimeSeconds)")
    }
}
```

## Voice Activity Detection (VAD)

The VAD APIs are tunable for different environments and use cases.

- **Model**: Silero VAD (CoreML)
- **Frames**: 512-sample processing with adaptive thresholding

See full guide with code samples and CLI usage: [VAD Getting Started](Documentation/VAD/GettingStarted.md).

### Quick Start (Code)

```swift
import FluidAudio

// Programmatic VAD over an audio file
Task {
    // 1) Initialize VAD (async load of Silero model)
    let vad = try await VadManager(config: VadConfig(threshold: 0.3))

    // 2) Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")

    // 3) Run VAD and print speech segments (512-sample frames)
    let results = try await vad.processAudioFile(samples)
    let sr = 16000.0
    let frame = 512.0

    var start: Int? = nil
    for (i, r) in results.enumerated() {
        if r.isVoiceActive {
            if start == nil { start = i }
        } else if let s = start {
            let startSec = (Double(s) * frame) / sr
            let endSec = (Double(i + 1) * frame) / sr
            print(String(format: "Speech: %.2f–%.2fs", startSec, endSec))
            start = nil
        }
    }
}
```

## Documentation

- [Automatic Speech Recognition](Documentation/ASR/GettingStarted.md)
- [Speaker Diarization](Documentation/SpeakerDiarization.md)
- [Voice Activity Detection](Documentation/VAD/GettingStarted.md)
- [Audio Conversion](Documentation/Guides/AudioConversion.md)
- [Troubleshooting](Documentation/Guides/Platform.md)
- [API Reference](Documentation/API.md)

## Showcase

Make a PR if you want to add your app!

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |


## API Reference

For a consolidated list of public types and methods across modules, see `Documentation/API.md`.
  
## Everything Else

### Platform & Networking Notes

See [Platform & Networking Notes](Documentation/Guides/Platform.md).

### License

Apache 2.0 — see `LICENSE` for details.

### Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx

silero-vad: https://github.com/snakers4/silero-vad
