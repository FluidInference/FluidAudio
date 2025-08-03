![banner.png](banner.png)

# FluidAudio - Swift Speaker Diarization & ASR on CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/vz7YYZkkJg)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)

Fluid Audio is a Swift framework for fully local, low-latency audio processing on Apple devices. It provides state-of-the-art speaker diarization, ASR, and voice activity detection through open-source models (MIT/Apache 2.0 licensed) that we've converted to Core ML.

Our models are optimized for background processing on CPU, avoiding GPU/MPS/Shaders to ensure reliable performance. While we've tested CPU/GPU-based alternatives, they proved too slow or resource-intensive for our near real-time requirements.

For custom use cases and feedback, reach out on Discord.

## Features

- **Automatic Speech Recognition (ASR)**: Parakeet TDT-0.6b model with Token Duration Transducer support for streaming transcription
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
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.0.3"),
],
```

**Important**: When adding FluidAudio as a package dependency, **only add the library to your target** (not the executable). Select "FluidAudio" library in the package products dialog and add it to your app target.

## Documentation

See the public DeepWiki docs: [https://deepwiki.com/FluidInference/FluidAudio](https://deepwiki.com/FluidInference/FluidAudio)

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


## 🚀 Roadmap

**Coming Soon:**
- **System Audio Access**: Tap into system audio via CoreAudio for MacOS, don't need to use ScreenCaptureKit or Blackhole

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

## Voice Activity Detection (VAD) (beta)

The APIs here are too complicated for production usage; please use with caution and tune them as needed. To be transparent, VAD is the lowest priority in terms of maintenance for us at this point. If you need support here, please file an issue or contribute back!

Our goal is to offer a similar API to what Apple will introudce in OS26: https://developer.apple.com/documentation/speech/speechdetector

## Automatic Speech Recognition (ASR)

- **Model**: [`FluidInference/parakeet-tdt-0.6b-v2-coreml`](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml)
- **Real-time Factor**: Optimized for near-real-time transcription with chunking support
- **Streaming Support**: Follows the same API as OS 26

`RTFx - ~110x on a M4 Pro`

## Real-World Usage

FluidAudio powers production applications including:

- **[Slipbox](https://slipbox.ai/)**: Privacy-first meeting assistant for real-time conversation intelligence
- **[Whisper Mate](https://whisper.marksdo.com)**: Transcribe movie/audio to text locally. Realtime record & transcribe from speaker or system apps. 🔒 All process in local mac Whisper AI Model.

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

## Quick Start

### Streaming ASR (Reccomended)

```swift
import AVFoundation
import FluidAudio

// Simple streaming ASR - handles everything automatically
let streamingAsr = StreamingAsrManager()
try await streamingAsr.start()

// Set up audio capture (any format - auto-converts to 16kHz mono)
let audioEngine = AVAudioEngine()
let inputNode = audioEngine.inputNode
inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputNode.outputFormat(forBus: 0)) { buffer, _ in
    streamingAsr.streamAudio(buffer)  // No conversion needed!
}

// Listen for transcription updates
Task {
    for await update in streamingAsr.transcriptionUpdates {
        if update.isConfirmed {
            print("✓ \(update.text)")  // High confidence
        } else {
            print("~ \(update.text)")  // Low confidence (show in purple)
        }
    }
}

try audioEngine.start()
// ... recording ...
let finalText = try await streamingAsr.finish()
```

## Manual ASR

```swift
import FluidAudio

// Initialize ASR with configuration
let asrConfig = ASRConfig(
    maxSymbolsPerFrame: 3,
    realtimeMode: true,
    chunkSizeMs: 1500,          // Process in 1.5 second chunks
    tdtConfig: TdtConfig(
        durations: [0, 1, 2, 3, 4],
        maxSymbolsPerStep: 3
    )
)

// Transcribe audio
Task {
    let asrManager = AsrManager(config: asrConfig)

    // Load models (automatic download if needed)
    let models = try await AsrModels.downloadAndLoad()
    try await asrManager.initialize(models: models)

    let audioSamples: [Float] = // your 16kHz audio data
    let result = try await asrManager.transcribe(audioSamples)

    print("Transcription: \(result.text)")
    print("Processing time: \(result.processingTime)s")

    // For streaming/chunked transcription
    let chunkResult = try await asrManager.transcribeChunk(
        audioChunk,
        source: .microphone  // or .system for system audio
    )
}
```

### Manual Speaker Diarization

```swift
import FluidAudio

// Initialize and process audio
Task {
    let diarizer = DiarizerManager()
    diarizer.initialize(models: try await .downloadIfNeeded())

    let audioSamples: [Float] = // your 16kHz audio data
    let result = try diarizer.performCompleteDiarization(audioSamples, sampleRate: 16000)

    for segment in result.segments {
        print("\(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

## Voice Activity Detection Usage

**VAD Library API**:

```swift
import FluidAudio

// Use the default configuration (already optimized for best results)
let vadConfig = VadConfig()  // Threshold: 0.445, optimized settings

// Or customize the configuration
let customVadConfig = VadConfig(
    threshold: 0.445,            // Recommended threshold (98% accuracy)
    chunkSize: 512,              // 32ms at 16kHz
    sampleRate: 16000,
    adaptiveThreshold: true,     // Adapts to noise levels
    minThreshold: 0.1,
    maxThreshold: 0.7,
    enableSNRFiltering: true,    // Enhanced noise robustness
    minSNRThreshold: 6.0,        // Aggressive noise filtering
    computeUnits: .cpuAndNeuralEngine  // Use Neural Engine on Apple Silicon
)

// Process audio for voice activity detection
Task {
    let vadManager = VadManager(config: vadConfig)
    try await vadManager.initialize()

    // Process a single audio chunk (512 samples = 32ms at 16kHz)
    let audioChunk: [Float] = // your 16kHz audio chunk
    let vadResult = try await vadManager.processChunk(audioChunk)

    print("Speech probability: \(vadResult.probability)")
    print("Voice active: \(vadResult.isVoiceActive)")
    print("Processing time: \(vadResult.processingTime)s")

    // Or process an entire audio file
    let audioData: [Float] = // your complete 16kHz audio data
    let results = try await vadManager.processAudioFile(audioData)

    // Find segments with voice activity
    let voiceSegments = results.enumerated().compactMap { index, result in
        result.isVoiceActive ? index : nil
    }
    print("Voice detected in \(voiceSegments.count) chunks")
}
```

## CLI Usage

FluidAudio includes a powerful command-line interface for benchmarking and audio processing:

**Note**: The CLI is available on macOS only. For iOS applications, use the FluidAudio library programmatically as shown in the usage examples above.

### Diarization Benchmark

```bash
# Run AMI benchmark with automatic dataset download
swift run fluidaudio benchmark --auto-download

# Test with specific parameters
swift run fluidaudio benchmark --threshold 0.7 --min-duration-on 1.0 --output results.json

# Test a single file for quick parameter tuning
swift run fluidaudio benchmark --single-file ES2004a --threshold 0.8
```

### ASR Benchmark

```bash
# Run LibriSpeech ASR benchmark
swift run fluidaudio asr-benchmark --subset test-clean --num-files 50

# Benchmark with specific configuration
swift run fluidaudio asr-benchmark --subset test-other --chunk-size 2000 --output asr_results.json

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
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

**Voice Activity Detection:**

- **`VadManager`**: Voice activity detection with CoreML models
- **`VadConfig`**: Configuration for VAD processing with adaptive thresholding
- **`processChunk(_:)`**: Process a single audio chunk and detect voice activity
- **`processAudioFile(_:)`**: Process complete audio file in chunks
- **`VadAudioProcessor`**: Advanced audio processing with SNR filtering

**Automatic Speech Recognition:**

- **`AsrManager`**: Main ASR class with TDT decoding
- **`AsrModels`**: Model loading and management
- **`ASRConfig`**: Configuration for ASR processing
- **`transcribe(_:)`**: Process complete audio and return transcription
- **`transcribeChunk(_:source:)`**: Process audio chunks for streaming
- **`AudioSource`**: Enum for microphone vs system audio separation

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker
