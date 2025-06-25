# FluidAudioSwift

A Swift package for fluid audio processing with speaker diarization and embedding extraction capabilities.

## Features

- **Speaker Diarization**: Identify and separate different speakers in audio
- **Speaker Embedding**: Extract speaker embeddings for comparison and clustering
- **CoreML Backend**: Native Apple CoreML integration for optimal performance on Apple devices
- **Cross-platform**: Support for macOS and iOS
- **Easy Integration**: Simple Swift API for audio processing tasks

## Installation

### Swift Package Manager

Add this package to your project using Xcode:

1. Open your project in Xcode
2. Go to File → Add Package Dependencies
3. Enter the repository URL
4. Select the version you want to use

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/[your-org]/FluidAudioSwift", from: "1.0.0")
]
```

## Quick Start

### Basic Speaker Diarization

```swift
import FluidAudioSwift

// Create a CoreML diarization manager
let config = DiarizerConfig(
    backend: .coreML,
    clusteringThreshold: 0.7,
    minDurationOn: 1.0,
    minDurationOff: 0.5
)

let diarizer = DiarizerFactory.createManager(config: config)

// Initialize the system (downloads models if needed)
try await diarizer.initialize()

// Process audio samples (16kHz, Float32)
let audioSamples: [Float] = // your audio data
let segments = try await diarizer.performSegmentation(audioSamples, sampleRate: 16000)

// Process results
for segment in segments {
    print("Speaker \(segment.speakerClusterId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
}
```

## Configuration

### DiarizerConfig

```swift
let config = DiarizerConfig(
    backend: .coreML,              // CoreML backend
    clusteringThreshold: 0.7,      // Speaker similarity threshold (0.0-1.0)
    minDurationOn: 1.0,           // Minimum speech duration (seconds)
    minDurationOff: 0.5,          // Minimum silence duration (seconds)
    numClusters: -1,              // Number of speakers (-1 for auto-detection)
    debugMode: false,             // Enable debug logging
    modelCacheDirectory: nil      // Custom model storage location
)
```

## Acknowledgments

This package leverages Apple's CoreML framework for high-performance audio processing on Apple devices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# macOS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
