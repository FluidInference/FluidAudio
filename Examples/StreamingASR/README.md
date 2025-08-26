# Streaming ASR Examples

This directory contains complete examples showing how to integrate FluidAudio's streaming ASR capabilities into your applications.

## Examples Included

### 1. UIKit Example (`UIKitExample.swift`)
- Complete UIKit implementation
- Shows microphone integration with AVAudioEngine
- Demonstrates proper UI updates with volatile/final text
- Includes error handling and lifecycle management

### 2. SwiftUI Example (`SwiftUIExample.swift`) 
- Modern SwiftUI with MVVM architecture
- Reactive patterns with `@StateObject` and `@Published`
- Clean separation of concerns
- Platform-agnostic (iOS + macOS)

## Quick Start

### Configuration Options

Choose the configuration that best fits your use case:

```swift
// Maximum responsiveness - text appears very quickly
let config = StreamingAsrConfig.realtime

// Balanced quality and latency
let config = StreamingAsrConfig.default

// Custom configuration
let config = StreamingAsrConfig(
    volatileRightContextSeconds: 0.125,  // How early volatile text appears
    volatileStepSeconds: 0.267,          // How often text updates
    rightContextSeconds: 1.5             // How long before text is finalized
)
```

### Basic Usage Pattern

```swift
import FluidAudio

// 1. Create streaming manager
let streamingAsr = StreamingAsrManager(config: .realtime)

// 2. Start the ASR engine
try await streamingAsr.start()

// 3. Listen for transcription updates
Task {
    for await snapshot in streamingAsr.snapshots {
        // snapshot.finalized: Stable, won't change
        // snapshot.volatile: In-progress, may change
        updateUI(finalized: snapshot.finalized, volatile: snapshot.volatile)
    }
}

// 4. Stream audio data
streamingAsr.streamAudio(audioBuffer)

// 5. Get final result
let finalTranscript = try await streamingAsr.finish()
```

## Key Features Demonstrated

### Real-time Updates
- **Volatile text**: Appears quickly as speech is processed
- **Final text**: Stable, corrected transcription that won't change
- **Smooth progression**: Text builds incrementally instead of sudden jumps

### Audio Integration
- Microphone capture with proper permissions
- 16kHz mono format conversion (handled automatically by FluidAudio)
- Buffer streaming from AVAudioEngine

### UI Best Practices
- Visual distinction between volatile and final text
- Proper threading with `@MainActor`
- Responsive button states and status messages
- Error handling and recovery

### Performance Optimizations
- Actor-based thread safety
- Efficient audio pipeline
- Memory management for continuous streaming

## Advanced Usage

### Detailed Segment Events
For more granular control, you can also listen to individual segment results:

```swift
Task {
    for await result in streamingAsr.results {
        if result.isFinal {
            print("Finalized: \(result.attributedText)")
            // This text won't change anymore
        } else {
            print("Volatile: \(result.attributedText)")
            // This may be updated in subsequent results
        }
    }
}
```

### Custom Audio Sources
The examples show microphone input, but you can stream any audio:

```swift
// From file
let audioFile = try AVAudioFile(forReading: fileURL)
// ... read buffers and stream them

// From network stream
// ... receive audio data and convert to AVAudioPCMBuffer

// From any audio source
streamingAsr.streamAudio(yourAudioBuffer)
```

## Requirements

- iOS 16.0+ / macOS 13.0+
- Microphone permissions for live transcription
- AVFoundation for audio capture
- FluidAudio framework

## Running the Examples

1. Copy the example code into your Xcode project
2. Add FluidAudio as a dependency
3. Set microphone permissions in Info.plist:
   ```xml
   <key>NSMicrophoneUsageDescription</key>
   <string>This app uses the microphone for speech transcription</string>
   ```
4. Build and run!

## Notes

- The first run will download ASR models (~100MB) automatically
- Models are cached locally for faster subsequent startups
- Use `.realtime` configuration for best user experience
- Always handle errors gracefully (network, model, permissions)