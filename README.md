![banner.png](banner.png)

# FluidAudio - Swift SDK for Speaker Diarization and ASR with CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)

Fluid Audio is a Swift framework for fully local, low-latency audio processing on Apple devices. It provides state-of-the-art speaker diarization, ASR, and voice activity detection through open-source models (MIT/Apache 2.0 licensed) that we've converted to Core ML.

Our models are optimized for background processing on CPU, avoiding GPU/MPS/Shaders to ensure reliable performance. While we've tested CPU/GPU-based alternatives, they proved too slow or resource-intensive for our near real-time requirements.

For custom use cases, feedback, more model support, and other platform requests, join our Discord. We’re also working on porting video, language, and TTS models to run on device, and will share updates there.

## Features

- **Automatic Speech Recognition (ASR)**: Parakeet TDT v3 (0.6b) with Token Duration Transducer; supports 25 European languages
- **Speaker Diarization**: Speaker separation with speaker clustering via Pyannote models
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **Voice Activity Detection (VAD)**: Voice activity detection with Silero models
- **CoreML Models**: Native Apple CoreML backend with custom-converted models optimized for Apple Silicon
- **Open-Source Models**: All models are [publicly available on HuggingFace](https://huggingface.co/FluidInference) - converted and optimized by our team. Permissive licenses.
- **Real-time Processing**: Designed for near real-time workloads but also works for offline processing
- **Cross-platform**: Support for macOS 14.0+ and iOS 17.0+ and Apple Sillicon device
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.4.0"),
],
```

**Important**: When adding FluidAudio as a package dependency, **only add the library to your target** (not the executable). Select "FluidAudio" library in the package products dialog and add it to your app target.

## Documentation

- **DeepWiki**: [https://deepwiki.com/FluidInference/FluidAudio](https://deepwiki.com/FluidInference/FluidAudio) - Primary documentation
- **Local Docs**: [Documentation/](Documentation/) - Additional guides and API references

## MCP

The repo is indexed by [DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki-mcp) - the MCP server gives your coding tool access to the docs already.

For most clients:

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

## Speaker Diarization

**AMI Benchmark Results** (Single Distant Microphone) using a subset of the files:

- **DER: 17.7%** - Competitive with Powerset BCE 2023 (18.5%)
- **JER: 28.0%** - Outperforms EEND 2019 (25.3%) and x-vector clustering (28.7%)
- **RTF: 0.02x** - Real-time processing with 50x speedup

```text
  RTF = Processing Time / Audio Duration

  With RTF = 0.02x:
  - 1 minute of audio takes 0.02 × 60 = 1.2 seconds to process
  - 10 minutes of audio takes 0.02 × 600 = 12 seconds to process

  For real-time speech-to-text:
  - Latency: ~1.2 seconds per minute of audio
  - Throughput: Can process 50x faster than real-time
  - Pipeline impact: Minimal - diarization won't be the bottleneck
```

## Voice Activity Detection (VAD)

The current VAD APIs are more complicated than they should be and require careful tuning for your specific use case. If you need help integrating VAD, reach out in our Discord channel.

Our goal is to eventually provide a streamlined API similar to Apple's upcoming SpeechDetector in OS26: https://developer.apple.com/documentation/speech/speechdetector

## Automatic Speech Recognition (ASR)

- **Model**: [`FluidInference/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Languages**: All European languages (25)
- **Processing Mode**: Batch transcription for complete audio files
- **Real-time Factor**: ~110x on M4 Pro (processes 1 minute of audio in ~0.5 seconds)
- **Streaming Support**: Coming soon - batch processing is recommended for production use
- **Backend**: Same Parakeet TDT v3 model powers our backend ASR

### CLI Transcription

```bash
# Transcribe an audio file using batch processing
swift run fluidaudio transcribe audio.wav

# Show help and usage options
swift run fluidaudio transcribe --help
```

### Benchmark Performance

```bash
swift run fluidaudio asr-benchmark --subset test-clean --max-files 25
```

## Showcase

FluidAudio powers local AI apps like:

- **[Slipbox](https://slipbox.ai/)**: Privacy-first meeting assistant for real-time conversation intelligence. Uses FluidAudio Parakeet for iOS transcription and speaker diarization across all platforms.
- **[Whisper Mate](https://whisper.marksdo.com)**: Transcribes movies and audio to text locally. Records and transcribes in real time from speakers or system apps. Uses FluidAudio for speaker diarization.
- **[Voice Ink](https://tryvoiceink.com/)**: Uses local AI models to instantly transcribe speech with near-perfect accuracy and complete privacy. Utilizes FluidAudio for Parakeet ASR.
- **[Spokenly](https://spokenly.app/)**: Mac dictation app that provides fast, accurate voice-to-text conversion anywhere on your system with Parakeet ASR powered by FluidAudio. Supports real-time dictation, file transcription, and speaker diarization.

Make a PR if you want to add your app!

## Contributing

### Code Style

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

## Batch ASR Usage

### CLI Command (Recommended)

```bash
# Simple transcription
swift run fluidaudio transcribe audio.wav

# This will output:
# - Audio format information (sample rate, channels, duration)
# - Final transcription text
# - Performance metrics (processing time, RTFx, confidence)
```

### Programmatic API

```swift
import AVFoundation
import FluidAudio

// Batch transcription from an audio source
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad()
    let asrManager = AsrManager(config: .default)
    try await asrManager.initialize(models: models)

    // 2) Load and convert audio to 16kHz mono Float32 samples
    let samples = try await AudioProcessor.loadAudioFile(path: "path/to/audio.wav")

    // 3) Transcribe the audio
    let result = try await asrManager.transcribe(samples, source: .system)
    print("Transcription: \(result.text)")
    print("Confidence: \(result.confidence)")
}
```

### Speaker Diarization

```swift
import FluidAudio

// Initialize and process audio
Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager()  // Uses optimal defaults (0.7 threshold = 17.7% DER)
    diarizer.initialize(models: models)

    let audioSamples: audioSamples[1000..<5000]  // your 16kHz audio data, No memory copy!
    let result = try diarizer.performCompleteDiarization(audioSamples)

    for segment in result.segments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

**Speaker Enrollment (NEW)**: The `Speaker` class now includes a `name` field for enrollment workflows. When users introduce themselves ("My name is Alice"), you can update the speaker's name from the default "Speaker_1" to their actual name, enabling personalized speaker identification throughout the session.


## CLI Usage

FluidAudio includes a powerful command-line interface for benchmarking and audio processing:

**Note**: The CLI is available on macOS only. For iOS applications, use the FluidAudio library programmatically as shown in the usage examples above.
**Note**: FluidAudio automatically downloads required models during audio processing. If you encounter network restrictions when accessing Hugging Face, you can configure an HTTPS proxy by setting the environment variable. For example: `export https_proxy=http://127.0.0.1:7890`

### Diarization Benchmark

```bash
# Run AMI benchmark with automatic dataset download
swift run fluidaudio diarization-benchmark --auto-download

# Test with specific parameters
swift run fluidaudio diarization-benchmark --threshold 0.7 --output results.json

# Test a single file for quick parameter tuning
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.8
```

### ASR Commands

```bash
# Transcribe an audio file (batch processing)
swift run fluidaudio transcribe audio.wav

# Run LibriSpeech ASR benchmark
swift run fluidaudio asr-benchmark --subset test-clean --num-files 50

# Benchmark with specific configuration  
swift run fluidaudio asr-benchmark --subset test-other --output asr_results.json

# Test with automatic download
swift run fluidaudio asr-benchmark --auto-download --subset test-clean
```

### Process Individual Files

```bash
# Process a single audio file for diarization
swift run fluidaudio process meeting.wav

# Save results to JSON
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6
```

### Download Datasets

```bash
# Download AMI dataset for diarization benchmarking
swift run fluidaudio download --dataset ami-sdm

# Download LibriSpeech for ASR benchmarking
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
```

## API Reference

**Diarization:**

- **`DiarizerManager`**: Main diarization class
- **`performCompleteDiarization(_:sampleRate:)`**: Process audio and return speaker segments
  - Accepts any `RandomAccessCollection<Float>` (Array, ArraySlice, ContiguousArray, etc.)
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

**Voice Activity Detection:**

- **`VadManager`**: Voice activity detection with CoreML models
- **`VadConfig`**: Configuration for VAD processing with adaptive thresholding
- **`processChunk(_:)`**: Process a single audio chunk and detect voice activity
- **`processAudioFile(_:)`**: Process complete audio file in chunks
- **`VadAudioProcessor`**: Advanced audio processing with SNR filtering

**Automatic Speech Recognition:**

- **`AsrManager`**: Main ASR class with TDT decoding for batch processing
- **`AsrModels`**: Model loading and management with automatic downloads
- **`ASRConfig`**: Configuration for ASR processing
- **`transcribe(_:source:)`**: Process complete audio and return transcription results
- **`AudioProcessor.loadAudioFile(path:)`**: Load and convert audio files to required format
- **`AudioSource`**: Enum for microphone vs system audio separation

We're also working on a Windows alternative here :
https://github.com/FluidInference/fluid-server

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx

silero-vad: https://github.com/snakers4/silero-vad
