![banner.png](banner.png)

# FluidAudio - Swift Speaker Diarization on CoreML

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/vz7YYZkkJg)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/FluidInference/coreml-models-6873d9e310e638c66d22fba9)

FluidAudio is a Swift framework for on-device speaker diarization and audio processing, designed to maximize performance per watt by leveraging CoreML models exclusively. Optimized for Apple's Neural Engine, it delivers faster and more efficient processing than CPU or GPU alternatives.

Built to address the need for an open-source solution capable of real-time workloads on iOS and older macOS devices, FluidAudio fills a gap where existing solutions either rely on CPU-only models or remain closed-source behind paid licenses. Since speaker diarization and identification are among the most popular features for voice AI applications, we believe these capabilities should be freely available.

## Features

Our testing demonstrates that CoreML versions deliver significantly more efficient inference compared to their ONNX counterparts, making them truly suitable for real-time transcription use cases.

- **State-of-the-Art Diarization**: Research-competitive speaker separation with optimal speaker mapping
- **Voice Activity Detection (VAD)**: Production-ready VAD with 98% accuracy using CoreML models and adaptive thresholding
- **Apple Neural Engine Optimized**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **CoreML Models**: Native Apple CoreML backend with custom-converted models optimized for Apple Silicon
- **Open-Source Models**: All models are [publicly available on HuggingFace](https://huggingface.co/FluidInference/speaker-diarization-coreml) - converted and optimized by our team. Permissive licenses.
- **Real-time Processing**: Designed for real-time workloads but also works for offline processing
- **Cross-platform**: Full support for macOS 13.0+ and iOS 16.0+ and any Apple Sillicon device

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

**Completed:**
- ✅ **Voice Activity Detection (VAD)**: 98% accuracy CoreML-based VAD with adaptive thresholding and noise robustness

**Coming Soon:**
- **VAD CLI Integration**: Expose VAD commands through command-line interface
- **ASR Models**: Support for open-source ASR models
- **System Audio Access**: Tap into system audio via CoreAudio

## 🎯 Performance

**AMI Benchmark Results** (Single Distant Microphone) using a subset of the files:

- **DER: 17.7%** - Competitive with Powerset BCE 2023 (18.5%)
- **JER: 28.0%** - Outperforms EEND 2019 (25.3%) and x-vector clustering (28.7%)
- **RTF: 0.02x** - Real-time processing with 50x speedup

- **Efficient Computing**: Runs on Apple Neural Engine with zero performance trade-offs

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

## 🎙️ Voice Activity Detection (VAD)

**Production-Ready VAD with Research-Grade Performance:**

- **98% Accuracy** on MUSAN dataset at optimal threshold (0.445)
- **CoreML Pipeline**: STFT → Encoder → RNN → Enhanced Fallback architecture
- **GPU Acceleration**: Metal Performance Shaders for efficient processing
- **Adaptive Thresholding**: Dynamic threshold adjustment (0.1-0.7 range)
- **Noise Robustness**: SNR filtering (6.0 dB threshold), spectral analysis, temporal smoothing

**Model Sources & Datasets:**
- **CoreML Models**: [`alexwengg/coreml_silero_vad`](https://huggingface.co/FluidInference/silero-vad-coreml)
- **Training Data**: MUSAN dataset (curated subsets)
  - [`alexwengg/musan_mini50`](https://huggingface.co/datasets/alexwengg/musan_mini50) (50 test files)
  - [`alexwengg/musan_mini100`](https://huggingface.co/datasets/alexwengg/musan_mini100) (100 test files)

**Technical Achievements:**
- **Model Conversion**: Solved PyTorch → CoreML limitations with custom fallback algorithm
- **Performance**: Real-time processing with minimal latency overhead
- **Integration**: Ready for embedding into diarization pipeline

## 🏢 Real-World Usage

FluidAudio powers production applications including:

- **[Slipbox](https://slipbox.ai/)**: Privacy-first meeting assistant for real-time conversation intelligence
- **[Whisper Mate](https://whisper.marksdo.com)**: Transcribe movie/audio to text locally. Realtime record & transcribe from speaker or system apps. 🔒 All process in local mac Whisper AI Model.

Make a PR if you want to add your app!

## Quick Start

```swift
import FluidAudio

// Initialize and process audio
Task {
    let diarizer = DiarizerManager()
    try await diarizer.initialize()

    let audioSamples: [Float] = // your 16kHz audio data
    let result = try await diarizer.performCompleteDiarization(audioSamples, sampleRate: 16000)

    for segment in result.segments {
        print("\(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

## Voice Activity Detection Usage

**VAD Library API** (CLI commands coming soon):

```swift
import FluidAudio

// Initialize VAD with optimal configuration
let vadConfig = VADConfig(
    threshold: 0.445,             // Optimized for 98% accuracy
    chunkSize: 512,              // Audio chunk size for processing
    sampleRate: 16000,           // 16kHz audio processing
    adaptiveThreshold: true,     // Enable dynamic thresholding
    minThreshold: 0.1,           // Minimum threshold value
    maxThreshold: 0.7,           // Maximum threshold value
    enableSNRFiltering: true,    // SNR-based noise rejection
    minSNRThreshold: 6.0,        // Aggressive noise filtering
    useGPU: true                 // Metal Performance Shaders
)

// Process audio for voice activity detection
Task {
    let vadManager = VadManager(config: vadConfig)
    try await vadManager.initialize()
    
    let audioSamples: [Float] = // your 16kHz audio data
    let vadResult = try await vadManager.detectVoiceActivity(audioSamples)
    
    print("Voice activity detected: \(vadResult.hasVoice)")
    print("Confidence score: \(vadResult.confidence)")
}
```

## Configuration

Customize behavior with `DiarizerConfig`:

```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,      // Speaker similarity (0.0-1.0, higher = stricter)
    minActivityThreshold: 10.0,    // Minimum activity frames for speaker detection
    minDurationOn: 1.0,           // Minimum speech duration (seconds)
    minDurationOff: 0.5,          // Minimum silence between speakers (seconds)
    numClusters: -1,              // Number of speakers (-1 = auto-detect)
    debugMode: false
)
```

## CLI Usage

FluidAudio includes a powerful command-line interface for benchmarking and audio processing:

**Note**: The CLI is available on macOS only. For iOS applications, use the FluidAudio library programmatically as shown in the usage examples above.

### Benchmark with Beautiful Output

```bash
# Run AMI benchmark with automatic dataset download
swift run fluidaudio benchmark --auto-download

# Test with specific parameters
swift run fluidaudio benchmark --threshold 0.7 --min-duration-on 1.0 --output results.json

# Test a single file for quick parameter tuning  
swift run fluidaudio benchmark --single-file ES2004a --threshold 0.8
```

### Process Individual Files

```bash
# Process a single audio file
swift run fluidaudio process meeting.wav

# Save results to JSON
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6
```

### Download Datasets

```bash
# Download AMI dataset for benchmarking
swift run fluidaudio download --dataset ami-sdm
```

## API Reference

**Diarization:**
- **`DiarizerManager`**: Main diarization class
- **`performCompleteDiarization(_:sampleRate:)`**: Process audio and return speaker segments
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

**Voice Activity Detection:**
- **`VadManager`**: Voice activity detection with CoreML models
- **`VADConfig`**: Configuration for VAD processing with adaptive thresholding
- **`detectVoiceActivity(_:)`**: Process audio and detect voice activity
- **`VADAudioProcessor`**: Advanced audio processing with SNR filtering

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing. 

Pyannote: https://github.com/pyannote/pyannote-audio

Wewpeaker: https://github.com/wenet-e2e/wespeaker
