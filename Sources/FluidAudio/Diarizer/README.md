# Speaker Diarization

Real-time speaker diarization for iOS and macOS, answering "who spoke when" in audio streams.

## Overview

The FluidAudio Diarizer provides production-ready speaker diarization with:
- **17.7% DER** on AMI corpus (competitive with state-of-the-art)
- **150+ RTFx** real-time performance on Apple Silicon
- **On-device processing** using Core ML
- **Streaming support** for real-time applications

## Quick Start

```swift
import FluidAudio

// 1. Download models (one-time setup)
let models = try await DiarizerModels.downloadIfNeeded()

// 2. Configure and initialize
let config = DiarizerConfig(
    clusteringThreshold: 0.7,  // Optimal for best accuracy
    minSpeechDuration: 1.0,     // Minimum speech segment
    minSilenceGap: 0.5          // Minimum gap between speakers
)

let diarizer = DiarizerManager(config: config)
diarizer.initialize(models: models)

// 3. Process audio
let audioSamples: [Float] = loadAudioFile() // 16kHz mono
let result = try diarizer.performCompleteDiarization(audioSamples)

// 4. Get results
for segment in result.segments {
    print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
}
```

## Core Components

### DiarizerManager
Main entry point for diarization pipeline:
```swift
let diarizer = DiarizerManager(config: config)
diarizer.initialize(models: models)
let result = try diarizer.performCompleteDiarization(audio)
```

### SpeakerManager
Tracks speaker identities across audio chunks:
```swift
let speakerManager = diarizer.speakerManager

// Get speaker information
print("Active speakers: \(speakerManager.speakerCount)")
for speakerId in speakerManager.speakerIds {
    if let info = speakerManager.getSpeakerInfo(for: speakerId) {
        print("\(speakerId): \(info.totalDuration)s total")
    }
}
```

### DiarizerConfig
Configuration parameters:
```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,      // Speaker separation threshold (0.0-1.0)
    minSpeechDuration: 1.0,         // Minimum speech duration in seconds
    minSilenceGap: 0.5,             // Minimum silence between speakers
    minActiveFramesCount: 10.0,     // Minimum active frames for valid segment
    debugMode: false                // Enable debug logging
)
```

## Streaming/Real-time Processing

Process audio in chunks for real-time applications:

```swift
// Configure for streaming
let diarizer = DiarizerManager(
    config: config,
    chunkDuration: 10.0,    // Process in 10-second chunks
    chunkOverlap: 0.0        // No overlap for lowest latency
)
diarizer.initialize(models: models)

// Process stream
for audioChunk in audioStream {
    let result = try diarizer.performCompleteDiarization(audioChunk)
    
    // Real-time results
    for segment in result.segments {
        handleSpeakerSegment(segment)
    }
}
```

## Known Speaker Recognition

Pre-load known speaker profiles:

```swift
// Create embeddings for known speakers
let aliceAudio = loadAudioFile("alice_sample.wav")
let aliceEmbedding = try diarizer.extractEmbedding(aliceAudio)

// Initialize with known speakers
let knownSpeakers = [
    "Alice": aliceEmbedding,
    "Bob": bobEmbedding
]
speakerManager.initializeKnownSpeakers(knownSpeakers)

// Process - will use "Alice" instead of "Speaker_1" when matched
let result = try diarizer.performCompleteDiarization(audioSamples)
```

## SwiftUI Integration

```swift
import SwiftUI
import FluidAudio

struct DiarizationView: View {
    @StateObject private var processor = DiarizationProcessor()
    
    var body: some View {
        VStack {
            Text("Speakers: \(processor.speakerCount)")
            
            List(processor.activeSpeakers) { speaker in
                HStack {
                    Circle()
                        .fill(speaker.isSpeaking ? Color.green : Color.gray)
                        .frame(width: 10, height: 10)
                    Text(speaker.name)
                    Spacer()
                    Text("\(speaker.duration, specifier: "%.1f")s")
                }
            }
            
            Button(processor.isProcessing ? "Stop" : "Start") {
                processor.toggleProcessing()
            }
        }
    }
}

@MainActor
class DiarizationProcessor: ObservableObject {
    @Published var speakerCount = 0
    @Published var activeSpeakers: [SpeakerDisplay] = []
    @Published var isProcessing = false
    
    private var diarizer: DiarizerManager?
    
    func toggleProcessing() {
        if isProcessing {
            stopProcessing()
        } else {
            startProcessing()
        }
    }
    
    private func startProcessing() {
        Task {
            let models = try await DiarizerModels.downloadIfNeeded()
            let config = DiarizerConfig(clusteringThreshold: 0.7)
            
            diarizer = DiarizerManager(config: config)
            diarizer?.initialize(models: models)
            isProcessing = true
            
            // Start audio capture and process chunks
            AudioCapture.start { [weak self] chunk in
                self?.processChunk(chunk)
            }
        }
    }
    
    private func processChunk(_ audio: [Float]) {
        Task { @MainActor in
            guard let diarizer = diarizer else { return }
            
            let result = try diarizer.performCompleteDiarization(audio)
            speakerCount = diarizer.speakerManager.speakerCount
            
            // Update UI with current speakers
            activeSpeakers = diarizer.speakerManager.speakerIds.compactMap { id in
                guard let info = diarizer.speakerManager.getSpeakerInfo(for: id) else { 
                    return nil 
                }
                return SpeakerDisplay(
                    id: id,
                    name: info.id,
                    duration: info.totalDuration,
                    isSpeaking: result.segments.contains { $0.speakerId == id }
                )
            }
        }
    }
}
```

## Performance Optimization

### Optimal Parameters
```swift
// Best accuracy (17.7% DER)
let config = DiarizerConfig(
    clusteringThreshold: 0.7,
    minSpeechDuration: 1.0,
    minSilenceGap: 0.5
)

// Lower latency for real-time
let config = DiarizerConfig(
    clusteringThreshold: 0.7,
    minSpeechDuration: 0.5,    // Faster response
    minSilenceGap: 0.3         // Quicker speaker switches
)
```

### Memory Management
```swift
// Reset between sessions to free memory
diarizer.speakerManager.reset()

// Or cleanup completely
diarizer.cleanup()
```

## Benchmarking

Evaluate performance on your audio:

```bash
# Command-line benchmark
swift run fluidaudio diarization-benchmark --single-file ES2004a

# Results:
# DER: 17.7% (Miss: 10.3%, FA: 1.6%, Speaker Error: 5.8%)
# RTFx: 141.2x (real-time factor)
```

## API Reference

### DiarizerManager

| Method | Description |
|--------|-------------|
| `initialize(models:)` | Initialize with Core ML models |
| `performCompleteDiarization(_:sampleRate:)` | Process audio and return segments |
| `cleanup()` | Release resources |

### SpeakerManager

| Method | Description |
|--------|-------------|
| `assignSpeaker(_:speechDuration:)` | Assign embedding to speaker |
| `initializeKnownSpeakers(_:)` | Load known speaker profiles |
| `getSpeakerInfo(for:)` | Get speaker details |
| `reset()` | Clear all speakers |

### DiarizerResult

| Property | Type | Description |
|----------|------|-------------|
| `segments` | `[TimedSpeakerSegment]` | Speaker segments with timing |
| `speakerCount` | `Int` | Number of unique speakers |
| `totalDuration` | `Float` | Total audio duration |

## Requirements

- iOS 16.0+ / macOS 13.0+
- Swift 5.9+
- ~100MB for Core ML models (downloaded on first use)

## Performance

| Device | RTFx | Notes |
|--------|------|-------|
| M2 MacBook Air | 150x | Apple Neural Engine |
| M1 iPad Pro | 120x | Neural Engine |
| iPhone 14 Pro | 80x | Neural Engine |
| GitHub Actions | 140x | CPU only |

## Troubleshooting

### High DER on certain audio
- Check if audio has overlapping speech (not yet supported)
- Ensure 16kHz sampling rate
- Verify audio isn't too noisy

### Memory issues
- Call `reset()` between sessions
- Process shorter chunks for streaming
- Reduce `minActiveFramesCount` if needed

### Model download fails
- Check internet connection
- Verify ~100MB free space
- Models cached after first download

## License

See main repository LICENSE file.