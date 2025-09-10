![banner.png](banner.png)

# FluidAudio - Speaker Diarization, VAD and Transcription with CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/FluidAudio)

Fluid Audio is a Swift SDK for fully local, low-latency audio AI on Apple devices, with inference offloaded to the Apple Neural Engine (ANE), resulting in less memory and generally faster inference.

The SDK includes state-of-the-art speaker diarization, transcription, and voice activity detection via open-source models (MIT/Apache 2.0) that can be integrated with just a few lines of code. Models are optimized for background processing, ambient computing and always on workloads by running inference on the ANE, minimizing CPU usage and avoiding GPU/MPS entirely. 

For custom use cases, feedback, additional model support, or platform requests, join our [Discord]. We’re also bringing visual, language, and TTS models to device and will share updates there.

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

- Docs Index: Documentation/README.md
- ASR: Documentation/ASR/GettingStarted.md
- Diarization: Documentation/SpeakerDiarization.md (Streaming: Documentation/SpeakerDiarization.md#streamingreal-time-processing)
- VAD: Documentation/VAD/GettingStarted.md
- Audio Conversion: Documentation/Guides/AudioConversion.md
- CLI Commands: Documentation/CLI.md
- API Reference: Documentation/API.md
- Platform & Networking: Documentation/Guides/Platform.md
- MCP Guide: Documentation/Guides/MCP.md

## Automatic Speech Recognition (ASR) / Transcription

- **Model**: `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- **Languages**: All European languages (25) - see Huggingface models for exact list
- **Processing Mode**: Batch transcription for complete audio files
- **Real-time Factor**: ~120x on M4 Pro (processes 1 minute of audio in ~0.5 seconds)
- **Streaming Support**: Coming soon — batch processing is recommended for production use
- **Backend**: Same Parakeet TDT v3 model powers our backend ASR

### Quick Start (Code)

```swift
import FluidAudio

// Batch transcription from an audio file
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad()
    let asrManager = AsrManager(config: .default)
    try await asrManager.initialize(models: models)

    // 2) Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")

    // 3) Transcribe the audio
    let result = try await asrManager.transcribe(samples, source: .system)
    print("Transcription: \(result.text)")
    print("Confidence: \(result.confidence)")
}
```

### CLI
See Documentation/CLI.md for complete commands and benchmarks.

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

### Quick Start (Code)

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

See Documentation/SpeakerDiarization.md for the full guide and streaming example. CLI commands are listed in Documentation/CLI.md.

See Documentation/CLI.md for complete diarization commands.

## Voice Activity Detection (VAD)

The current VAD APIs require careful tuning for your specific use case. If you need help integrating VAD, reach out in our Discord channel.

Our goal is to provide a streamlined API similar to Apple's upcoming SpeechDetector in [OS26](https://developer.apple.com/documentation/speech/speechdetector).

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
    let sampleRate = 16000.0
    let frame = 512.0

    var startIndex: Int? = nil
    for (i, r) in results.enumerated() {
        if r.isVoiceActive {
            if startIndex == nil { startIndex = i }
        } else if let s = startIndex {
            let startSec = (Double(s) * frame) / sampleRate
            let endSec = (Double(i + 1) * frame) / sampleRate
            print(String(format: "Speech: %.2f–%.2fs", startSec, endSec))
            startIndex = nil
        }
    }
}
```

See Documentation/CLI.md for VAD commands.

## Showcase 

Make a PR if you want to add your app!

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |


## API Reference

See Documentation/API.md for the full API overview across modules.
  
## Everything Else

See Documentation/Guides/Platform.md for platform and networking notes.

### License

Apache 2.0 — see `LICENSE` for details.


### Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx

silero-vad: https://github.com/snakers4/silero-vad
