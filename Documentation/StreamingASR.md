# StreamingAsrManager Documentation

## Overview

`StreamingAsrManager` provides a high-level, easy-to-use API for real-time speech recognition in FluidAudio. It's designed to be similar to Apple's Speech Analyzer API, handling audio conversion, buffering, and confidence-based transcription confirmation automatically.

## Key Features

- **Automatic Audio Conversion**: Accepts any audio format and converts to 16kHz mono
- **AsyncStream-based API**: Natural Swift concurrency patterns
- **Two-tier Transcription**: Volatile (unconfirmed) and confirmed text states
- **Confidence-based Confirmation**: Automatic text confirmation based on confidence thresholds
- **No Manual Buffering**: Handles all audio buffering internally
- **Automatic Final Processing**: Processes remaining audio when stream ends

## Quick Start

```swift
import AVFoundation
import FluidAudio

// Initialize
let streamingAsr = StreamingAsrManager()
try await streamingAsr.start()

// Set up audio capture
let audioEngine = AVAudioEngine()
let inputNode = audioEngine.inputNode
inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputNode.outputFormat(forBus: 0)) { buffer, _ in
    // No conversion needed - just pass the buffer
    streamingAsr.streamAudio(buffer)
}

// Listen for updates
Task {
    for await update in streamingAsr.transcriptionUpdates {
        if update.isConfirmed {
            print("Confirmed: \(update.text)")
        } else {
            print("Volatile: \(update.text)") // Show in purple
        }
    }
}

// Start recording
try audioEngine.start()

// When done
let finalTranscript = try await streamingAsr.finish()
```

## Configuration Options

### Default Configuration
```swift
let streamingAsr = StreamingAsrManager(config: .default)
// Chunk duration: 10.0s (optimized for TDT decoder)
// Confirmation threshold: 0.85
// Balanced accuracy and latency
```

### Low Latency Configuration
```swift
let streamingAsr = StreamingAsrManager(config: .lowLatency)
// Chunk duration: 5.0s (minimum viable for TDT decoder)
// Confirmation threshold: 0.75
// Faster updates with slightly lower accuracy
```

### High Accuracy Configuration
```swift
let streamingAsr = StreamingAsrManager(config: .highAccuracy)
// Chunk duration: 10.0s (optimal for TDT decoder accuracy)
// Confirmation threshold: 0.90
// More accurate with slightly higher latency
```

### Custom Configuration
```swift
let config = StreamingAsrConfig(
    confirmationThreshold: 0.8,
    chunkDuration: 2.2,
    enableDebug: true
)
let streamingAsr = StreamingAsrManager(config: config)
```

## Two-Tier Transcription System

StreamingAsrManager uses a two-tier system similar to Apple's Speech API:

1. **Volatile Text**: Low-confidence transcriptions that may change
   - Displayed in purple in UIs (like Apple's implementation)
   - Stored in `volatileTranscript` property
   - May be updated or replaced by subsequent chunks

2. **Confirmed Text**: High-confidence transcriptions that won't change
   - Displayed in normal color
   - Stored in `confirmedTranscript` property
   - Text moves from volatile to confirmed when confidence exceeds threshold

## SwiftUI Integration

```swift
@MainActor
class TranscriptionViewModel: ObservableObject {
    @Published var confirmedText = ""
    @Published var volatileText = ""
    
    private var streamingAsr: StreamingAsrManager?
    
    func startTranscribing() async throws {
        streamingAsr = StreamingAsrManager()
        try await streamingAsr?.start()
        
        // Set up audio capture...
        
        // Listen for updates
        Task {
            guard let updates = streamingAsr?.transcriptionUpdates else { return }
            
            for await update in updates {
                await MainActor.run {
                    confirmedText = streamingAsr?.confirmedTranscript ?? ""
                    volatileText = streamingAsr?.volatileTranscript ?? ""
                }
            }
        }
    }
}

struct TranscriptionView: View {
    @StateObject private var viewModel = TranscriptionViewModel()
    
    var body: some View {
        VStack(alignment: .leading) {
            // Confirmed text
            Text(viewModel.confirmedText)
                .foregroundColor(.primary)
            
            // Volatile text (purple)
            Text(viewModel.volatileText)
                .foregroundColor(.purple.opacity(0.8))
        }
    }
}
```

## Audio Format Support

StreamingAsrManager accepts audio in any format and automatically converts to the required format (16kHz, mono, Float32). This includes:

- Different sample rates (8kHz, 44.1kHz, 48kHz, etc.)
- Multi-channel audio (stereo, 5.1, etc.)
- Different bit depths

No manual conversion is needed - just pass your AVAudioPCMBuffer directly.

## Error Handling

```swift
do {
    let streamingAsr = StreamingAsrManager()
    try await streamingAsr.start()
    
    // ... use the manager ...
    
    let finalText = try await streamingAsr.finish()
} catch {
    print("ASR Error: \(error)")
    // Handle specific errors if needed
}
```

## Comparison with RealtimeAsrManager

| Feature | StreamingAsrManager | RealtimeAsrManager |
|---------|-------------------|-------------------|
| Audio Conversion | Automatic | Manual |
| API Style | AsyncStream | Manual chunks |
| Buffering | Automatic | Manual |
| Confidence Logic | Built-in | Manual |
| Code Complexity | Simple | Complex |
| Use Case | Most apps | Advanced control |

## Migration from RealtimeAsrManager

If you're currently using RealtimeAsrManager, here's how to migrate:

### Before (RealtimeAsrManager)
```swift
let manager = RealtimeAsrManager(models: models)
let stream = try await manager.createStream(config: config)

// Manually convert audio to 16kHz mono
let samples = convertToMono16kHz(buffer)

// Process chunks manually
if let update = try await manager.processAudio(streamId: stream.id, samples: samples) {
    // Handle update
}

// Get final transcription
let final = try await manager.getFinalTranscription(streamId: stream.id)
```

### After (StreamingAsrManager)
```swift
let streamingAsr = StreamingAsrManager()
try await streamingAsr.start()

// No conversion needed
streamingAsr.streamAudio(buffer)

// Automatic updates via AsyncStream
for await update in streamingAsr.transcriptionUpdates {
    // Handle update
}

// Get final transcription
let final = try await streamingAsr.finish()
```

## Best Practices

1. **Start Early**: Call `start()` during app initialization to download models
2. **Handle Interruptions**: Stop/restart on audio session interruptions
3. **Memory Management**: Call `finish()` or `cancel()` when done
4. **UI Updates**: Use `@MainActor` for UI updates from transcription stream
5. **Error Recovery**: Implement retry logic for network/model errors

## Example Projects

See the `Examples/` directory for complete examples:
- `StreamingAsrExample.swift` - Basic command-line example
- `StreamingAsrSwiftUI.swift` - SwiftUI integration example
- `StreamingAsrAdvanced.swift` - Advanced features and error handling