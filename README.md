# SeamlessAudioSwift

A Swift package for seamless audio processing, speech recognition, and speaker diarization using SherpaOnnx.

## Features

- 🎤 **Speech Recognition**: Real-time and offline speech-to-text
- 👥 **Speaker Diarization**: Identify and separate different speakers in audio
- 🔊 **Speaker Embedding**: Extract speaker embeddings for identification
- 🎯 **Voice Activity Detection**: Detect speech segments in audio
- 📱 **Cross-Platform**: Works on macOS and iOS
- ⚡ **High Performance**: Optimized with native C++ libraries

## Installation

### Swift Package Manager

Add SeamlessAudioSwift to your project using Xcode:

1. In Xcode, go to **File → Add Package Dependencies**
2. Enter the repository URL: `https://github.com/SeamlessCompute/SeamlessAudioSwift.git`
3. Choose the version and add to your target

Or add it to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/SeamlessCompute/SeamlessAudioSwift.git", from: "1.0.0")
]
```

## Quick Start

### Basic Speech Recognition

```swift
import SeamlessAudioSwift

// Initialize the speech recognition manager
let manager = SpeakerDiarizationManager()

// Process audio samples
let audioSamples: [Float] = // your audio data
let result = try await manager.extractEmbedding(from: audioSamples)
```

### Speaker Diarization

```swift
import SeamlessAudioSwift

// Initialize with models
let manager = SpeakerDiarizationManager()
try await manager.initialize()

// Process audio for speaker separation
let segments = try await manager.performDiarization(on: audioSamples)

for segment in segments {
    print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
}
```

## Requirements

- **iOS**: 16.0+
- **macOS**: 13.0+
- **Xcode**: 16.0+
- **Swift**: 6.0+

## Models and Attribution

This package uses models and libraries from the excellent [**SherpaOnnx**](https://github.com/k2-fsa/sherpa-onnx) project by the K2-FSA team.

### SherpaOnnx Models

SherpaOnnx provides state-of-the-art speech recognition and audio processing models. You can find pre-trained models at:

- **Main Repository**: [https://github.com/k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **Pre-trained Models**: [https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html)
- **Documentation**: [https://k2-fsa.github.io/sherpa/onnx/](https://k2-fsa.github.io/sherpa/onnx/)

### Supported Model Types

- **Speech Recognition**: Transducer, Paraformer, Whisper, CTC models
- **Speaker Diarization**: Pyannote-based segmentation models
- **Speaker Embedding**: Speaker verification and identification models
- **Voice Activity Detection**: Silero VAD models

## Architecture

SeamlessAudioSwift is built on top of:

- **SherpaOnnx C++ Libraries**: High-performance audio processing
- **ONNX Runtime**: Optimized neural network inference
- **Swift Package Manager**: Modern dependency management
- **Git LFS**: Efficient handling of large model files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[SherpaOnnx](https://github.com/k2-fsa/sherpa-onnx)** by the K2-FSA team for the underlying speech processing libraries
- **[ONNX Runtime](https://onnxruntime.ai/)** for neural network inference
- **[Pyannote](https://github.com/pyannote/pyannote-audio)** for speaker diarization models
- **[Silero](https://github.com/snakers4/silero-vad)** for voice activity detection models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support:
- 📧 Contact: [SeamlessCompute](https://github.com/SeamlessCompute)
- 🐛 Issues: [GitHub Issues](https://github.com/SeamlessCompute/SeamlessAudioSwift/issues)

---

**Note**: This package includes pre-compiled libraries and models. The first build may take longer due to the size of the dependencies.

# macOS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
