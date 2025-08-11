# SpeakerManager API Guide

The `SpeakerManager` class provides advanced speaker tracking and management capabilities for FluidAudio. It maintains speaker consistency across audio chunks and sessions.

## Core Concepts

### Speaker Tracking
The SpeakerManager maintains an in-memory database of speakers, tracking:
- **Speaker embeddings**: 256-dimensional voice fingerprints
- **Speaker IDs**: Consistent identifiers (e.g., "Speaker_1", "Speaker_2")
- **Speaker metadata**: Duration spoken, last seen time, update count

### Embedding Similarity
Speakers are identified by comparing embeddings using cosine distance:
- Distance < 0.7: Same speaker (high confidence)
- Distance 0.7-0.9: Possibly same speaker
- Distance > 0.9: Different speakers

## Basic Usage

### Initialize Manager
```swift
let speakerManager = SpeakerManager(
    speakerThreshold: 0.65,      // Max distance for speaker match
    embeddingThreshold: 0.45,     // Distance for embedding updates
    minSpeechDuration: 1.0        // Min seconds to create new speaker
)
```

### Assign Speakers
```swift
// Process new audio segment
let speakerId = speakerManager.assignSpeaker(
    embedding,                    // 256-dim embedding from model
    speechDuration: 2.5,          // Segment duration in seconds
    confidence: 0.95              // Optional confidence score
)
// Returns: "Speaker_1" or nil if too short
```

## Advanced Features

### Known Speaker Enrollment
Pre-load known speakers for meetings or sessions:
```swift
let knownSpeakers = [
    "Alice": aliceEmbedding,
    "Bob": bobEmbedding
]
speakerManager.initializeKnownSpeakers(knownSpeakers)
```

### Speaker Verification
Check if two audio samples are from the same person:
```swift
let (isSame, confidence) = speakerManager.verifySameSpeaker(
    embedding1: sample1,
    embedding2: sample2,
    threshold: 0.7
)
// Returns: (isSame: true, confidence: 0.85)
```

### Speaker Search
Find a specific speaker in recorded segments:
```swift
let matches = speakerManager.findSpeaker(
    targetEmbedding: targetVoice,
    in: segments,
    threshold: 0.65
)
// Returns array of Speaker objects with timestamps
```

### Similar Speaker Discovery
Find the most similar speakers to a target:
```swift
let similar = speakerManager.findSimilarSpeakers(
    to: unknownEmbedding,
    limit: 5
)
// Returns: [(speaker, distance)] sorted by similarity
```

## Session Management

### Export/Import Speakers
Save and restore speaker profiles between sessions:
```swift
// Export to JSON
let jsonData = try speakerManager.exportToJSON()
try jsonData.write(to: profilesURL)

// Import from JSON
let savedData = try Data(contentsOf: profilesURL)
try speakerManager.importFromJSON(savedData)
```

### Export as Speaker Objects
Get structured speaker data:
```swift
let speakers = speakerManager.exportAsSpeakers()
for speaker in speakers {
    print("\(speaker.id): \(speaker.totalDuration)s")
}
```

### Prune Inactive Speakers
Clean up long-running sessions:
```swift
// Remove speakers not seen in last 5 minutes
speakerManager.pruneInactiveSpeakers(olderThan: 300)
```

## Model Inference Methods

FluidAudio uses two main CoreML models for diarization:

### Segmentation Model
```swift
// SegmentationProcessor.getSegments()
// Detects speech activity and separates overlapping speakers
// Input: 10-second audio chunk (16kHz)
// Output: Speaker activity masks
```

### Embedding Model
```swift
// EmbeddingExtractor.getEmbeddings()
// Converts audio+masks into speaker embeddings
// Input: Audio + speaker masks
// Output: 256-dimensional embeddings
```

## Integration Example

Complete diarization with speaker tracking:
```swift
// 1. Initialize
let config = DiarizerConfig(clusteringThreshold: 0.7)
let manager = DiarizerManager(config: config)

// 2. Load known speakers (optional)
manager.speakerManager.initializeKnownSpeakers(knownProfiles)

// 3. Process audio
let result = try manager.performCompleteDiarization(audioSamples)

// 4. Access speaker info
let speakerInfo = manager.speakerManager.getAllSpeakerInfo()
for (id, info) in speakerInfo {
    print("\(id): \(info.totalDuration)s, updates: \(info.updateCount)")
}

// 5. Export for next session
let profiles = manager.speakerManager.exportAsSpeakers()
```

## Performance Tips

1. **Thresholds**: Lower thresholds = more speakers detected
2. **Min Duration**: Set to 1.0s to avoid noise being labeled as speakers
3. **Embedding Updates**: Only update embeddings for high-confidence matches
4. **Memory**: Prune inactive speakers in long sessions
5. **Known Speakers**: Pre-load for better accuracy in meetings

## API Reference

### SpeakerManager Methods
- `assignSpeaker(_:speechDuration:confidence:)` - Assign or create speaker
- `initializeKnownSpeakers(_:)` - Load known speaker profiles
- `verifySameSpeaker(embedding1:embedding2:threshold:)` - Compare speakers
- `findSpeaker(targetEmbedding:in:threshold:)` - Search for speaker
- `findSimilarSpeakers(to:limit:)` - Find similar speakers
- `exportToJSON()` / `importFromJSON(_:)` - Persistence
- `exportAsSpeakers()` / `importFromSpeakers(_:)` - Structured data
- `pruneInactiveSpeakers(olderThan:)` - Memory management
- `reset()` - Clear all speakers

### SpeakerInfo Properties
- `id: String` - Unique speaker identifier
- `embedding: [Float]` - 256-dim voice fingerprint
- `totalDuration: Float` - Total seconds spoken
- `lastSeen: Date` - Last activity timestamp
- `updateCount: Int` - Number of embedding updates