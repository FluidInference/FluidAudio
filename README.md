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

- **DeepWiki**: Auto-generated docs for this repo — https://deepwiki.com/FluidInference/FluidAudio

### MCP

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

### Audio Conversion (16 kHz mono)

Most features expect 16 kHz mono Float32 samples. Use `AudioConverter` to load and convert from any `AVAudioFile` format:

```swift
import AVFoundation
import FluidAudio

func loadSamples16kMono(path: String) async throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)
    let capacity = AVAudioFrameCount(file.length)
    guard let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: capacity) else {
        return []
    }
    try file.read(into: buf)
    let converter = AudioConverter()
    return try await converter.convertToAsrFormat(buf)
}
```

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

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# Transcribe multiple files in parallel
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudio asr-benchmark --subset test-clean --num-files 50

# Multilingual ASR (FLEURS) benchmark
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

# Download LibriSpeech test sets
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
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

### Streaming Diarization

Stream meeting audio in chunks while maintaining consistent speaker IDs across the session. Keep a single `DiarizerManager` alive, process fixed-size chunks, and rebase segment timestamps by the chunk’s start offset. Overlap helps reduce boundary errors and enables overlap speech handling.

```swift
import FluidAudio

Task {
    // 1) Initialize diarizer once (models are reused across chunks)
    let models = try await DiarizerModels.downloadIfNeeded()
    let config = DiarizerConfig(
        clusteringThreshold: 0.7,
        minSpeechDuration: 1.0,
        minSilenceGap: 0.5,
        minActiveFramesCount: 10.0,
        chunkDuration: 10.0,   // model window; also used for chunk sizing below
        chunkOverlap: 5.0,     // optional overlap
        debugMode: false
    )
    let diarizer = DiarizerManager(config: config)
    diarizer.initialize(models: models)

    // Optional: tune streaming behavior (assignment/update thresholds)
    diarizer.speakerManager.speakerThreshold = 0.84   // assign to existing speakers
    diarizer.speakerManager.embeddingThreshold = 0.56  // update embeddings over time

    // 2) Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/meeting.wav")

    // 3) Chunked streaming loop with timestamp rebasing
    let sr = 16000.0
    let chunkSeconds = 10.0
    let overlapSeconds = 5.0
    let chunkSize = Int(chunkSeconds * sr)
    let hop = Int(max(1.0, chunkSeconds - overlapSeconds) * sr)

    var position = 0
    var segments: [TimedSpeakerSegment] = []

    while position < samples.count {
        let end = min(position + chunkSize, samples.count)
        var chunk = Array(samples[position..<end])
        if chunk.count < chunkSize {
            // Pad final chunk to model’s expected window
            chunk += [Float](repeating: 0, count: chunkSize - chunk.count)
        }

        // Run diarization on this chunk
        let result = try diarizer.performCompleteDiarization(chunk)

        // Rebase chunk-relative times to stream time
        let offsetSec = Float(position) / Float(sr)
        for seg in result.segments {
            segments.append(
                TimedSpeakerSegment(
                    speakerId: seg.speakerId,
                    embedding: seg.embedding,
                    startTimeSeconds: seg.startTimeSeconds + offsetSec,
                    endTimeSeconds: seg.endTimeSeconds + offsetSec,
                    qualityScore: seg.qualityScore
                )
            )
        }

        position += hop
    }

    // Optional: merge or deduplicate segments across overlaps/boundaries.
    // diarizer.speakerManager preserves consistent IDs across all chunks.
}
```

CLI equivalents:
```bash
# Real-time-ish streaming benchmark (~3s chunks with 2s overlap)
swift run fluidaudio diarization-benchmark --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2

# Balanced throughput/quality (~10s chunks with 5s overlap)
swift run fluidaudio diarization-benchmark --dataset ami-sdm \
  --chunk-seconds 10 --overlap-seconds 5
```

Notes:
- Keep one `DiarizerManager` instance per stream so `SpeakerManager` maintains ID consistency.
- Always rebase per-chunk timestamps by `(chunkStartSample / sampleRate)`.
- Provide 16 kHz mono Float32 samples; pad final chunk to the model window.
- Tune `speakerThreshold` and `embeddingThreshold` to trade off ID stability vs. sensitivity.

**Speaker Enrollment:** The `Speaker` class includes a `name` field for enrollment workflows. When users introduce themselves ("My name is Alice"), update the speaker's name from the default (e.g. "Speaker_1") to enable personalized identification.

### CLI

```bash
# Run AMI benchmark (auto-download dataset)
swift run fluidaudio diarization-benchmark --auto-download

# Tune threshold and save results
swift run fluidaudio diarization-benchmark --threshold 0.7 --output results.json

# Quick test on a single AMI file
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.8

# Process an individual file and save JSON
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6

# Download AMI dataset
swift run fluidaudio download --dataset ami-sdm
```

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

### CLI

```bash
# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3

# Save results and enable debug output
swift run fluidaudio vad-benchmark --all-files --output vad_results.json --debug

# Download VAD dataset if needed
swift run fluidaudio download --dataset vad
```

## Showcase 

Make a PR if you want to add your app!

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |


## API Reference

**Diarization:**

- `DiarizerManager`: Main diarization class
- `performCompleteDiarization(_:sampleRate:)`: Process audio and return speaker segments
  - Accepts any `RandomAccessCollection<Float>` (Array, ArraySlice, ContiguousArray, etc.)
- `compareSpeakers(audio1:audio2:)`: Compare similarity between two audio samples
- `validateAudio(_:)`: Validate audio quality and characteristics

**Voice Activity Detection:**

- `VadManager`: Voice activity detection with CoreML models
- `VadConfig`: Configuration for VAD processing with adaptive thresholding
- `processChunk(_:)`: Process a single audio chunk and detect voice activity
- `processAudioFile(_:)`: Process complete audio file in chunks
- `VadAudioProcessor`: Advanced audio processing with SNR filtering

**Automatic Speech Recognition:**

- `AsrManager`: Main ASR class with TDT decoding for batch processing
- `AsrModels`: Model loading and management with automatic downloads
- `ASRConfig`: Configuration for ASR processing
- `transcribe(_:source:)`: Process complete audio and return transcription results
- `AudioProcessor.loadAudioFile(path:)`: Load and convert audio files to required format
- `AudioSource`: Enum for microphone vs system audio separation
  
## Everything Else

### Platform & Networking Notes

- CLI is available on macOS only. For iOS, use the library programmatically.
- Models auto-download on first use. If your network restricts Hugging Face access, set an HTTPS proxy: `export https_proxy=http://127.0.0.1:7890`.
- Windows alternative in development: [fluid-server](https://github.com/FluidInference/fluid-server)

If you're looking to get the system audio on a Mac, take a look at this repo for reference [AudioCap](https://github.com/insidegui/AudioCap/tree/main)

### License

Apache 2.0 — see `LICENSE` for details.

### Contributing

This project uses `swift-format` to maintain consistent code style. All pull requests are automatically checked for formatting compliance.

**Local Development:**

```bash
# Format all code (requires Swift 6+ for contributors only)
# Users of the library don't need Swift 6
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/ Examples/

# Check formatting without modifying
swift format lint --recursive --configuration .swift-format Sources/ Tests/ Examples/

# For Swift <6, install swift-format separately:
# git clone https://github.com/apple/swift-format
# cd swift-format && swift build -c release
# cp .build/release/swift-format /usr/local/bin/
```

**Automatic Checks:**

- PRs will fail if code is not properly formatted
- GitHub Actions runs formatting checks on all Swift file changes
- See `.swift-format` for style configuration

### Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx

silero-vad: https://github.com/snakers4/silero-vad
