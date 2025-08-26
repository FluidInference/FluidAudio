![banner.png](banner.png)

# FluidAudio | Transcription, Speaker Diarization, VAD via CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/FluidAudio)


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
- **Cross-platform**: Support for macOS 14.0+ and iOS 17.0+ and Apple Silicon device
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.3.0"),
],
```

**Important**: When adding FluidAudio as a package dependency, **only add the library to your target** (not the executable). Select "FluidAudio" library in the package products dialog and add it to your app target.

## Documentation

- **DeepWiki**: [https://deepwiki.com/FluidInference/FluidAudio](https://deepwiki.com/FluidInference/FluidAudio) - Primary documentation
- **Local Docs**: [Documentation/](Documentation/) - Additional guides and API references

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

## Showcase

FluidAudio powers local AI apps like:

- **[Slipbox](https://slipbox.ai/)**: Privacy-first meeting assistant for real-time conversation intelligence. Uses FluidAudio Parakeet for iOS transcription and speaker diarization across all platforms.
- **[Whisper Mate](https://whisper.marksdo.com)**: Transcribes movies and audio to text locally. Records and transcribes in real time from speakers or system apps. Uses FluidAudio for speaker diarization.
- **[Voice Ink](https://tryvoiceink.com/)**: Uses local AI models to instantly transcribe speech with near-perfect accuracy and complete privacy. Utilizes FluidAudio for Parakeet ASR.
- **[Spokenly](https://spokenly.app/)**: Mac dictation app that provides fast, accurate voice-to-text conversion anywhere on your system with Parakeet ASR powered by FluidAudio. Supports real-time dictation, file transcription, and speaker diarization.

Make a PR if you want to add your app!


## Automatic Speech Recognition (ASR)

- **Model**: [`FluidInference/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Languages**: All European languages (25)
- **Processing Modes**: Batch (files) and Streaming (live/continuous)
- **Real-time Factor (batch)**: ~110x on M4 Pro (1 minute in ~0.5 seconds)
- **Streaming Support**: Available via `StreamingAsrManager` and `StreamingAsrSession`
- **Backend**: Same Parakeet TDT v3 model powers our backend ASR

### CLI Transcription

```bash
# Transcribe an audio file (batch mode by default)
swift run fluidaudio transcribe audio.wav

# Transcribe with streaming simulation (incremental updates)
swift run fluidaudio transcribe audio.wav --streaming

# Run two parallel streams with shared models (macOS)
swift run fluidaudio multi-stream mic.wav system.wav

# Show help and usage options
swift run fluidaudio transcribe --help
```

### Benchmark Performance

```bash
swift run fluidaudio asr-benchmark --subset test-clean --max-files 25
```

### Batch Transcription API

This will take your audio, convert it to 16Hz if needed, break it into processable chunks and return the entire text all at once. Best for processing files or short audio chunks.

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
}
```

### Streaming Transcription API

This is best for long-running use cases like meeting transcription or live recordings. Streaming provides:

- **Volatile updates**: fast, speculative partial text for UI responsiveness.
- **Final updates**: corrected text once sufficient right-context is available.
- **Stateful decoding**: decoder state is preserved across chunks per `AudioSource`.

Basic streaming with a single source:

```swift
import AVFoundation
import FluidAudio

@available(iOS 16.0, macOS 13.0, *)
func runStreaming() async throws {
    // 1) Pick a streaming preset
    // 2) Create manager and start (downloads models if needed)
    let streaming = StreamingAsrManager()
    try await streaming.start(source: .microphone)

    // 3) Subscribe to incremental updates
    Task {
        for await snapshot in await streaming.snapshots {
            let final = String(snapshot.finalized.characters)
            let volatile = snapshot.volatile.map { String($0.characters) } ?? ""
            // Update your UI with final + volatile
        }
    }

    // 4) Feed audio from AVAudioEngine (any format is OK; conversion is handled)
    let engine = AVAudioEngine()
    let input = engine.inputNode
    let format = input.inputFormat(forBus: 0)
    input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
        Task { await streaming.streamAudio(buffer) }
    }
    try engine.start()

    // ... later, when done recording
    let finalText = try await streaming.finish()
    print(finalText)
}
```

Shared-models multi-stream (e.g., microphone + system audio):

```swift
@available(iOS 16.0, macOS 13.0, *)
func runMultiStream() async throws {
    let session = StreamingAsrSession()
    try await session.initialize() // load once; shared across streams

    let mic = try await session.createStream(source: .microphone, config: .default)
    let sys = try await session.createStream(source: .system, config: .default)

    Task { for await s in await mic.snapshots { /* update UI */ } }
    Task { for await s in await sys.snapshots { /* update UI */ } }

    // Feed your buffers: await mic.streamAudio(buffer) / await sys.streamAudio(buffer)

    let micFinal = try await mic.finish()
    let sysFinal = try await sys.finish()
}
```

Configuration differences and impact:

- `ASRConfig` (batch):
  - Keys: `sampleRate` (default 16k), `tdtConfig`, `enableDebug`.
  - Usage: pass into `AsrManager(config:)`. You provide 16kHz mono `Float` audio (`AudioProcessor.loadAudioFile(...)`).
  - Behavior: each `transcribe` resets decoder state between calls.

- `StreamingAsrConfig` (streaming):
  - Keys that shape latency vs accuracy at runtime:
    - `chunkSeconds` (default 11.0): core segment length. Larger → more stability, higher latency.
    - `leftContextSeconds` (default 2.0): past audio appended for stability; too large increases compute.
    - `rightContextSeconds` (default 2.0): lookahead before finalizing; smaller lowers latency, can reduce accuracy.
    - `volatileRightContextSeconds` (default 0.5): lookahead for partial (volatile) updates; smaller → snappier UI.
    - `volatileStepSeconds` (default 0.5): cadence of volatile updates; smaller → more frequent updates.
    - `minContextForFinalization` (default 10.0): guard so finals occur with sufficient context.
    - `finalizationThreshold` (default 0.85): confidence gate for finalizing segments.
  - Usage: pass into `StreamingAsrManager(config:)` or `StreamingAsrSession.createStream(config:)`.
  - Behavior: maintains per-`AudioSource` decoder state; call `reset()` to start a fresh session, or `finish()`/`cancel()`.

Notes:

- Audio conversion is automatic in streaming via an internal `AudioConverter` actor (any `AVAudioPCMBuffer` format is OK).
- For iOS/macOS apps, prefer streaming and consume `snapshots` for simple UI, or `results` for granular segment/timing.
- For file processing and benchmarks, prefer batch (`AsrManager`), which maximizes throughput and simplicity.




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

### Speaker Diarization Usage

```swift
import FluidAudio

// Initialize and process audio
Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager()  // Uses optimal defaults (0.7 threshold = 17.7% DER)
    diarizer.initialize(models: models)

    let audioSamples: [Float] = // your 16kHz audio data
    let result = try diarizer.performCompleteDiarization(audioSamples)

    for segment in result.segments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

**Speaker Enrollment (NEW)**: The `Speaker` class now includes a `name` field for enrollment workflows. When users introduce themselves ("My name is Alice"), you can update the speaker's name from the default "Speaker_1" to their actual name, enabling personalized speaker identification throughout the session.

## Voice Activity Detection (VAD) (beta)

The APIs here are too complicated for production usage; please use with caution and tune them as needed. To be transparent, VAD is the lowest priority in terms of maintenance for us at this point. If you need support here, please file an issue or contribute back!

Our goal is to offer a similar API to what Apple will introudce in OS26: https://developer.apple.com/documentation/speech/speechdetector

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
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# Transcribe with streaming simulation (incremental updates)
swift run fluidaudio transcribe audio.wav --streaming

# Two parallel streams with shared models (macOS)
swift run fluidaudio multi-stream mic.wav system.wav

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

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker

Parakeet-mlx: https://github.com/senstella/parakeet-mlx
