# SpeakerManager API

Tracks and manages speaker identities across audio chunks.

## Configuration

```swift
let speakerManager = SpeakerManager(
    speakerThreshold: 0.65,      // Max cosine distance for match
    embeddingThreshold: 0.45,     // Threshold for embedding updates
    minSpeechDuration: 1.0        // Min seconds to create speaker
)
```

## Core Methods

### assignSpeaker
```swift
let speakerId = speakerManager.assignSpeaker(
    embedding,                    // 256-dim array
    speechDuration: 2.5,          // seconds
    confidence: 0.95              // optional
)
// Returns: "Speaker_1" or nil
```

### initializeKnownSpeakers
```swift
let knownSpeakers = ["Alice": embedding1, "Bob": embedding2]
speakerManager.initializeKnownSpeakers(knownSpeakers)
```

## Speaker Operations (SpeakerOperations.swift)

### verifySameSpeaker
```swift
let (isSame, confidence) = speakerManager.verifySameSpeaker(
    embedding1: sample1,
    embedding2: sample2,
    threshold: 0.7
)
```

### findSpeaker
```swift
let matches = speakerManager.findSpeaker(
    targetEmbedding: targetVoice,
    in: segments,
    threshold: 0.65
)
```

### findSimilarSpeakers
```swift
let similar = speakerManager.findSimilarSpeakers(
    to: unknownEmbedding,
    limit: 5
)
// Returns: [(speaker, distance)]
```

## Persistence

### JSON Export/Import
```swift
let jsonData = try speakerManager.exportToJSON()

try speakerManager.importFromJSON(jsonData)
```

### Speaker Objects
```swift
let speakers = speakerManager.exportAsSpeakers()

speakerManager.importFromSpeakers(speakers)
```

## Memory Management

```swift
// Remove inactive speakers since seconds
speakerManager.pruneInactiveSpeakers(olderThan: 300)

// Clear database
speakerManager.reset()
```

## Integration with DiarizerManager

```swift
let diarizer = DiarizerManager()
diarizer.initialize(models: models)

let speakerManager = diarizer.speakerManager

let result = try diarizer.performCompleteDiarization(audio)

let info = speakerManager.getAllSpeakerInfo()
```

## Model Pipeline

1. **Segmentation** (`SegmentationProcessor.getSegments()`)
   - Input: 10s audio (16kHz)
   - Output: 3 speaker activity masks

2. **Embedding** (`EmbeddingExtractor.getEmbeddings()`)
   - Input: Audio + masks
   - Output: 3x256-dim embeddings

3. **Speaker Assignment** (`SpeakerManager.assignSpeaker()`)
   - Input: Embeddings
   - Output: Speaker IDs

## API Reference

### Methods
| Method | Returns | Description |
|--------|---------|-------------|
| `assignSpeaker(_:speechDuration:confidence:)` | `String?` | Assign/create speaker |
| `initializeKnownSpeakers(_:)` | `Void` | Load profiles |
| `getSpeakerInfo(for:)` | `SpeakerInfo?` | Get speaker data |
| `getAllSpeakerInfo()` | `[String: SpeakerInfo]` | All speakers |
| `verifySameSpeaker(embedding1:embedding2:threshold:)` | `(Bool, Float)` | Compare embeddings |
| `findSpeaker(targetEmbedding:in:threshold:)` | `[Speaker]` | Search segments |
| `findSimilarSpeakers(to:limit:)` | `[(Speaker, Float)]` | Ranked matches |
| `exportToJSON()` | `Data` | JSON export |
| `importFromJSON(_:)` | `Void` | JSON import |
| `exportAsSpeakers()` | `[Speaker]` | Speaker objects |
| `importFromSpeakers(_:)` | `Void` | Load speakers |
| `pruneInactiveSpeakers(olderThan:)` | `Void` | Remove old |
| `reset()` | `Void` | Clear all |

### SpeakerInfo
| Property | Type | Description |
|----------|------|-------------|
| `id` | `String` | Speaker ID |
| `embedding` | `[Float]` | 256-dim vector |
| `totalDuration` | `Float` | Total seconds |
| `lastSeen` | `Date` | Last update |
| `updateCount` | `Int` | Updates made |

### Thresholds
- `< 0.5`: Same speaker (high confidence)
- `0.5-0.7`: Same speaker (medium confidence)
- `0.7-0.9`: Different speakers (medium confidence)
- `> 0.9`: Different speakers (high confidence)
