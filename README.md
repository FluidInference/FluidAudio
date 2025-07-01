# FluidAudio - Efficient Speaker Diarization for Apple devices in Swift

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow.svg)](https://huggingface.co/bweng/speaker-diarization-coreml)

FluidAudio is a high-performance Swift framework for on-device speaker diarization and audio processing that is competitive with research standards. The goal is to deliver maximum performance per watt by using only CoreML models. We are also working on adding ASR (Automatic Speech Recognition) support.

## Features

- **State-of-the-Art Diarization**: Research-competitive speaker separation with optimal speaker mapping
- **Apple Neural Engine Optimized**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering
- **CoreML Integration**: Native Apple CoreML backend with custom-converted models optimized for Apple Silicon and iOS
- **Open-Source Models**: All models are [publicly available on HuggingFace](https://huggingface.co/bweng/speaker-diarization-coreml) - converted and optimized by our team
- **Real-time Processing**: Support for streaming audio processing with minimal latency
- **Cross-platform**: Full support for macOS 13.0+ and iOS 16.0+

- **Comprehensive CLI**: Professional benchmarking tools with beautiful tabular output

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.0.2"),
],
```

## 🚀 Roadmap

**Coming Soon:**

- **Voice Activity Detection (VAD)**: Voice activity detection capabilities
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

## ⚡ Apple Neural Engine Optimization

FluidAudio leverages Apple's Neural Engine for maximum efficiency:

- **Zero Performance Trade-offs**: Full ANE utilization without sacrificing accuracy
- **Maximum Performance per Watt**: Dedicated to delivering the most efficient computing possible on Apple devices
- **Power Efficient**: Minimal battery impact on mobile devices
- **Custom Model Conversion**: Our team has converted and optimized all models specifically for CoreML
- **Ongoing CoreML Support**: We're committed to converting and supporting the latest models through CoreML for optimal Apple ecosystem performance
- **Open Source Models**: All converted models are [available on HuggingFace](https://huggingface.co/bweng/speaker-diarization-coreml) for transparency and community contribution

## 🏢 Real-World Usage

FluidAudio powers production applications including:

- **[Slipbox](https://slipbox.ai/)**: Privacy-first meeting assistant for real-time conversation intelligence

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

- **`DiarizerManager`**: Main diarization class
- **`performCompleteDiarization(_:sampleRate:)`**: Process audio and return speaker segments
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing. 

## Community

[Join our discord](https://discord.gg/8FbwRaDFJR)