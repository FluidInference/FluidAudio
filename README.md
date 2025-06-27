# FluidAudioSwift

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FluidAudioSwift is a Swift framework for on-device speaker diarization and audio processing, providing advanced capabilities for identifying and separating different speakers in audio recordings with high accuracy and performance.

## Features

- **Speaker Diarization**: Automatically identify and separate different speakers in audio recordings
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering
- **CoreML Integration**: Native Apple CoreML backend for optimal performance on Apple Silicon
- **Real-time Processing**: Support for streaming audio processing with minimal latency
- **Cross-platform**: Full support for macOS 13.0+ and iOS 16.0+
- **Easy Integration**: Simple Swift API designed for seamless integration into your apps

## Table of Contents

* [Installation](#installation)
  * [Swift Package Manager](#swift-package-manager)
  * [Prerequisites](#prerequisites)
  * [Xcode Steps](#xcode-steps)
  * [Package.swift](#packageswift)
* [Getting Started](#getting-started)
  * [Quick Example](#quick-example)
  * [Configuration](#configuration)
  * [Model Management](#model-management)
* [API Reference](#api-reference)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Installation

### Swift Package Manager

FluidAudioSwift can be integrated into your Swift project using the Swift Package Manager.

### Prerequisites

* macOS 13.0 or later
* iOS 16.0 or later
* Xcode 15.0 or later

### Xcode Steps

1. Open your Swift project in Xcode
2. Navigate to `File` → `Add Package Dependencies...`
3. Enter the package repository URL: `https://github.com/[your-org]/FluidAudioSwift`
4. Choose the version range or specific version
5. Click `Finish` to add FluidAudioSwift to your project

### Package.swift

If you're using FluidAudioSwift as part of a Swift package, you can include it in your Package.swift dependencies as follows:

```swift
dependencies: [
    .package(url: "https://github.com/[your-org]/FluidAudioSwift.git", from: "1.0.0"),
],
```

Then add `FluidAudioSwift` as a dependency for your target:

```swift
.target(
    name: "YourApp",
    dependencies: ["FluidAudioSwift"]
),
```

## Getting Started

To get started with FluidAudioSwift, you need to initialize the diarization system and process your audio data.

### Quick Example

This example demonstrates how to perform speaker diarization on an audio file:

```swift
import FluidAudioSwift

// Initialize the diarization system
Task {
    let config = DiarizerConfig(
        clusteringThreshold: 0.7,
        minDurationOn: 1.0,
        minDurationOff: 0.5,
        numClusters: -1  // Auto-detect number of speakers
    )

    let diarizer = DiarizerManager(config: config)

    // Initialize (downloads models if needed)
    try await diarizer.initialize()

    // Process audio samples (16kHz, Float32)
    let audioSamples: [Float] = // your audio data
    let result = try await diarizer.performCompleteDiarization(audioSamples, sampleRate: 16000)

    // Process results
    for segment in result.segments {
        print("\(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

### Configuration

Customize the diarization behavior with `DiarizerConfig`:

```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,      // Speaker similarity threshold (0.0-1.0, higher = stricter)
    minDurationOn: 1.0,           // Minimum speech duration (seconds)
    minDurationOff: 0.5,          // Minimum silence duration (seconds)
    numClusters: -1,              // Number of speakers (-1 for auto-detection)
    debugMode: false,             // Enable debug logging
    modelCacheDirectory: nil      // Custom model storage location
)
```

### Model Management

FluidAudioSwift automatically downloads and manages the required CoreML models:

```swift
// Models are automatically downloaded on first initialization
try await diarizer.initialize()

// Check if system is ready
if diarizer.isAvailable {
    // Ready to process audio
}

// Clean up resources when done
await diarizer.cleanup()
```

## API Reference

### Core Classes

- **`DiarizerManager`**: Main class for speaker diarization operations
- **`DiarizerConfig`**: Configuration options for diarization behavior
- **`DiarizationResult`**: Complete diarization results with speaker segments
- **`TimedSpeakerSegment`**: Individual speaker segment with timing and embedding

### Key Methods

- `performCompleteDiarization(_:sampleRate:)`: Process audio and return diarization results
- `compareSpeakers(audio1:audio2:)`: Compare similarity between two audio samples
- `validateAudio(_:)`: Validate audio quality and characteristics
- `validateEmbedding(_:)`: Validate speaker embedding quality

## Contributing

We welcome contributions to FluidAudioSwift! Please refer to our contribution guidelines for submitting issues, pull requests, and coding standards.

## License

FluidAudioSwift is released under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.

---

**FluidAudioSwift** - On-device Speaker Diarization for Apple Platforms
