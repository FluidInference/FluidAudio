# FluidAudioSwift

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FluidAudioSwift is a Swift framework for on-device speaker diarization and audio processing.

## Features

- **Speaker Diarization**: Automatically identify and separate different speakers in audio recordings
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering
- **CoreML Integration**: Native Apple CoreML backend for optimal performance on Apple Silicon and iOS support
- **Real-time Processing**: Support for streaming audio processing with minimal latency
- **Cross-platform**: Full support for macOS 13.0+ and iOS 16.0+

## Installation

Add FluidAudioSwift to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudioSwift.git", from: "1.0.0"),
],
```

## Quick Start

```swift
import FluidAudioSwift

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

## API Reference

- **`DiarizerManager`**: Main diarization class
- **`performCompleteDiarization(_:sampleRate:)`**: Process audio and return speaker segments
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.


