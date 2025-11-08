
# SpeakerManager API

Tracks and manages speaker identities across audio chunks for streaming diarization.

## Overview

`SpeakerManager` maintains an in-memory database of speakers, tracks their voice embeddings, and assigns consistent IDs across audio chunks. It's used by `DiarizerManager` for streaming/real-time speaker diarization.

> **Note:** `SpeakerManager` is only compatible with `DiarizerManager` (streaming pipeline). It is not currently supported with `OfflineDiarizerManager`, which uses VBx clustering for speaker assignment.

## Configuration

```swift
let speakerManager = SpeakerManager(
    speakerThreshold: 0.65,           // Max cosine distance for speaker match
    embeddingThreshold: 0.45,         // Max distance for embedding updates
    minSpeechDuration: 1.0,           // Min seconds to create new speaker
    minEmbeddingUpdateDuration: 2.0   // Min seconds to update embeddings
)
```

### Configuration Parameters

- **`speakerThreshold`** (Float, default: 0.65)
  - Maximum cosine distance for assigning an embedding to an existing speaker
  - Lower values = stricter matching = more speakers created
  - Range: 0.0-1.0 (typical: 0.5-0.8)

- **`embeddingThreshold`** (Float, default: 0.45)
  - Maximum distance threshold for updating a speaker's embedding
  - Only high-quality matches below this threshold update the speaker profile
  - Range: 0.0-1.0 (typically lower than `speakerThreshold`)

- **`minSpeechDuration`** (Float, default: 1.0 seconds)
  - Minimum duration of speech required to create a new speaker
  - Prevents creating speakers from brief audio fragments

- **`minEmbeddingUpdateDuration`** (Float, default: 2.0 seconds)
  - Minimum speech duration required to update an existing speaker's embedding
  - Ensures only substantial speech segments update the profile

## Core Methods

### Speaker Assignment

#### assignSpeaker
Assign an embedding to an existing speaker or create a new one.

```swift
let speaker = speakerManager.assignSpeaker(
    embedding,                    // [Float] - 256-dim embedding array
    speechDuration: 2.5,          // seconds of speech
    confidence: 0.95              // optional confidence score
)
// Returns: Speaker? - the assigned or created speaker, or nil if duration too short
```

**Behavior:**
1. Finds the closest matching speaker using cosine distance
2. If distance < `speakerThreshold`: assigns to existing speaker
3. If no match and duration ≥ `minSpeechDuration`: creates new speaker
4. If distance < `embeddingThreshold` and duration ≥ `minEmbeddingUpdateDuration`: updates the speaker's embedding
5. Returns `nil` if speech too short to create a new speaker

### Known Speaker Management

#### initializeKnownSpeakers
Pre-load known speaker profiles for recognition.

```swift
let alice = Speaker(id: "alice", name: "Alice", currentEmbedding: aliceEmbedding)
let bob = Speaker(id: "bob", name: "Bob", currentEmbedding: bobEmbedding)
speakerManager.initializeKnownSpeakers([alice, bob])
```

Sometimes, there are already speakers in the database that may have the same ID.
```swift
let alice = Speaker(id: "alice", name: "Alice", currentEmbedding: aliceEmbedding)
let bob = Speaker(id: "bob", name: "Bob", currentEmbedding: bobEmbedding)
speakerManager.initializeKnownSpeakers([alice, bob], mode: .overwrite, preserveIfPermanent: false) // replace any speakers with ID "alice" or "bob" with the new speakers, even if the old ones were marked as permanent. 
```

> The `mode` argument dictates how to handle redundant speakers. It is of type `SpeakerInitializationMode`, and can take on one of four values:
> - `.reset`: reset the speaker database and add the new speakers
> - `.merge`: merge new speakers whose IDs match with existing ones
> - `.overwrite`: overwrite existing speakers with the same IDs as the new ones 
> - `.skip`: skip adding speakers whose IDs match existing ones
> 
> The `preserveIfPermanent` argument determines whether existing speakers marked as permanent should be preserved (i.e., not overwritten or merged). It is `true` by default.

**Use case:** When you have pre-recorded voice samples of known speakers and want to recognize them by name instead of numeric IDs.

#### upsertSpeaker
Update an existing speaker or insert a new one (insert-or-update).

```swift
// From Speaker object
speakerManager.upsertSpeaker(speaker)

// From individual parameters
speakerManager.upsertSpeaker(
    id: "speaker_1",
    currentEmbedding: embedding,
    duration: 15.3,
    rawEmbeddings: [], // optional
    updateCount: 5,    // optional
    createdAt: Date(), // optional
    updatedAt: Date()  // optional
    isPermanent: false // optional
)
```

**Behavior:**
- If speaker ID exists: updates the existing speaker's data
- If speaker ID is new: inserts as a new speaker
- Maintains ID uniqueness and tracks numeric IDs for auto-increment
- If `isPermanent` is true, then the new speaker or the existing speaker will become permanent. This means that the speaker will not be merged or removed without an override.

#### mergeSpeaker
```swift
// merge speaker 1 into "alice"
speakerManager.mergeSpeaker("1", into: "alice") 

// merge speaker 2 into speaker 3 under the name "bob", regardless of whether speaker 2 is permanent.
speakerManager.mergeSpeaker("2", into: "3", mergedName: "Bob", stopIfPermanent: false)
```

**Behavior:**
- Unless `stopIfPermanent` is `false`, the merge will be stopped if the first speaker is permanent.
- Otherwise: Merges the first speaker into the destination speaker and removes the first speaker from the known speaker database.
- If `mergedName` is provided, the destination speaker will be renamed. Otherwise, its name will be preserved.

> Note: the `mergedName` argument is optional.
> Note: `stopIfPermanent` is `true` by default. 

#### removeSpeaker
Remove a speaker from the database.

```swift
// remove speaker 1
speakerManager.removeSpeaker("1") 

// remove "alice" from the known speaker database, even if they are marked as permanent
speakerManager.removeSpeaker("alice", keepIfPermanent: false) 
```
> Note: `keepIfPermanent` is `true` by default.

#### removeSpeakersInactive
Remove speakers that have been inactive since a certain date or for a certain duration.

```swift
// remove speakers that have been inactive since `date`
speakerManager.removeSpeakersInactive(since: date) 

// remove speakers that have been inactive for 10 seconds, even if they were marked as permanent
speakerManager.removeSpeakersInactive(for: 10.0, keepIfPermanent: false)
```

> Note: Both versions of the method have an optional `keepIfPermanent` argument that defaults to `true`.

#### removeAllSpeakers
Remove all speakers that match a given predicate.

```swift
// remove all speakers with less than 5 seconds of speaking time
speakerManager.removeSpeakers(
    where: { $0.duration < 5.0 },
    keepIfPermanent: false // also remove permanent speakers (optional)
)

// Alternate syntax (does NOT remove permanent speakers)
speakerManager.removeSpeakers {
    $0.duration < 5.0
}
```

> Note: the predicate should take in a `Speaker` object and return a `Bool`.

#### makeSpeakerPermanent
Make the speaker permanent.

```swift
speakerManager.makeSpeakerPermanent("alice") // mark "alice" as permanent
```

#### revokePermanence
Make the speaker not permanent.

```swift
speakerManager.revokePermanence(from: "alice") // mark "alice" as not permanent
```

#### resetPermanentFlags
Mark all speakers as not permanent.

```swift
speakerManager.resetPermanentFlags()
```

### Speaker Retrieval

#### findSpeaker
Find the best matching speaker to an embedding vector and the cosine distance to them, unless no match is found.

```swift
let (id, distance) = speakerManager.findSpeaker(with: embedding)
```
> Note: there is an optional `speakerThreshold` argument to use a threshold other than the default.

#### findMatchingSpeakers
Find all speakers within the maximum `speakerThreshold` to an embedding vector.

```swift
for speaker in speakerManager.findMatchingSpeakers(with: embedding) {
    print("ID: \(speaker.id), Distance: \(speaker.distance)")
}
```

> Note: there is an optional `speakerThreshold` argument to use a threshold other than the default.

#### findSpeakers
Find all speakers that meet a certain predicate.
```swift
// two ways to find all speakers with > 5.0s of speaking time.
speakerManager.findSpeakers(where: { $0.duration > 5.0 })
speakerManager.findSpeakers{ $0.duration > 5.0 }
// Returns an array of IDs corresponding to speakers that meet the predicate.
```

> Note: the predicate should take in a `Speaker` object and return a `Bool`.

#### findMergeablePairs
Find all pairs of speakers that might be the same person. Specifically, find the pairs of speakers such that the cosine distance between them is less than the `speakerThreshold`. 

Returns a list of pairs of speaker IDs.

```swift
let pairs = speakerManager.findMergeablePairs(
    speakerThreshold: 0.6, // optional
    excludeIfBothPermanent: true // optional
)

for pair in pairs {
    print("Merge Speaker \(pair.speakerToMerge) into Speaker \(pair.destination)")
}
```

#### getSpeaker
Get a specific speaker by ID.

```swift
if let speaker = speakerManager.getSpeaker(for: "speaker_1") {
    print("\(speaker.name): \(speaker.duration)s total")
}
```

#### getAllSpeakers
Get all speakers in the database (for testing/debugging).

```swift
let allSpeakers = speakerManager.getAllSpeakers()
// Returns: [String: Speaker] - dictionary keyed by speaker ID
```

#### getSpeakerList
Get all speakers in the database as an array of speakers (for testing/debugging)
```swift
let allSpeakers = speakerManager.getSpeakerList()
// Returns: [Speaker] - Array of speakers
```

#### hasSpeaker
Check if the speaker database has a speaker with a given ID.

```swift
if speakerManager.hasSpeaker("alice") {
    print("Alice was found in the database")
}
```

#### speakerCount
Get the total number of tracked speakers.

```swift
print("Active speakers: \(speakerManager.speakerCount)")
```

#### speakerIds
Get all speaker IDs as a sorted array.

```swift
let ids = speakerManager.speakerIds
// Returns: [String] - sorted array of speaker IDs
```

### Database Management

#### reset
Clear all speakers from the database.

```swift
speakerManager.reset()
speakerManager.reset(keepIfPermanent: true) // remove all non-permanent speakers from the database 
```

Useful for:
- Starting a new session
- Freeing memory between recordings
- Resetting speaker tracking



## Speaker Enrollment

The `Speaker` class includes a `name` field for speaker enrollment workflows:

```swift
// User introduces themselves: "My name is Alice"
let speaker = speakerManager.assignSpeaker(embedding, speechDuration: 3.0)
speaker?.name = "Alice"  // Update from default "Speaker 1" to "Alice"

// Future segments will continue using "Alice" as the speaker ID/name
```

This enables applications where speakers introduce themselves at the start of a session and are automatically identified throughout.

## Extension Operations

Additional operations provided via `SpeakerOperations.swift` extension:

### reassignSegment
Move a segment embedding from one speaker to another.

```swift
let success = speakerManager.reassignSegment(
    segmentId: uuid,
    from: "speaker_1",
    to: "speaker_2"
)
```

Useful for correcting misclassified segments in post-processing.

### getCurrentSpeakerNames
Get a sorted list of all speaker IDs.

```swift
let names = speakerManager.getCurrentSpeakerNames()
// Returns: [String] - sorted speaker IDs
```

### getGlobalSpeakerStats
Get aggregate statistics across all speakers.

```swift
let stats = speakerManager.getGlobalSpeakerStats()
print("Total speakers: \(stats.totalSpeakers)")
print("Total duration: \(stats.totalDuration)s")
print("Average confidence: \(stats.averageConfidence)")
print("Speakers with history: \(stats.speakersWithHistory)")
```

Returns:
- `totalSpeakers`: Number of tracked speakers
- `totalDuration`: Combined speech duration across all speakers
- `averageConfidence`: Normalized confidence score (0.0-1.0)
- `speakersWithHistory`: Speakers with raw embedding history

## Integration with DiarizerManager

```swift
let diarizer = DiarizerManager()
let models = try await DiarizerModels.downloadIfNeeded()
diarizer.initialize(models: models)

// Access the speaker manager
let speakerManager = diarizer.speakerManager

// Process audio
let result = try diarizer.performCompleteDiarization(audio)

// Access speaker information
let speakers = speakerManager.getAllSpeakers()
for (id, speaker) in speakers {
    print("\(speaker.name): \(speaker.duration)s")
}
```

## Speaker Data Model

The `Speaker` class represents a speaker's profile:

```swift
public final class Speaker: Identifiable, Codable {
    public let id: String                    // Unique identifier
    public var name: String                  // Display name (defaults to ID)
    public var currentEmbedding: [Float]     // 256-dim L2-normalized embedding
    public var duration: Float               // Total speech duration (seconds)
    public var createdAt: Date               // Creation timestamp
    public var updatedAt: Date               // Last update timestamp
    public var updateCount: Int              // Number of updates
    public var rawEmbeddings: [RawEmbedding] // Historical embeddings (max 50)
    public var isPermanent: Bool             // Permanence flag
}
```

### Speaker Methods

#### updateMainEmbedding
Update the speaker's main embedding using exponential moving average.

```swift
speaker.updateMainEmbedding(
    duration: 3.5,
    embedding: newEmbedding,
    segmentId: UUID(),
    alpha: 0.9  // EMA weight (higher = more weight to current embedding)
)
```

#### addRawEmbedding
Add a historical embedding to the speaker's profile.

```swift
let rawEmbedding = RawEmbedding(
    segmentId: UUID(),
    embedding: embedding,
    timestamp: Date()
)
speaker.addRawEmbedding(rawEmbedding)
```

Maintains a FIFO queue with max 50 embeddings.

#### removeRawEmbedding
Remove a specific historical embedding by segment ID.

```swift
speaker.removeRawEmbedding(segmentId: uuid)
```

#### recalculateMainEmbedding
Recalculate the main embedding as the average of all raw embeddings.

```swift
speaker.recalculateMainEmbedding()
```

#### mergeWith
Merge another speaker into this one.

```swift
speaker1.mergeWith(speaker2, keepName: "Alice")
```

Combines raw embeddings, durations, and recalculates the main embedding.

## Utility Functions (SpeakerUtilities)

Static utility functions for speaker operations without requiring a `SpeakerManager` instance:

### cosineDistance
Calculate cosine distance between embeddings.

```swift
let distance = SpeakerUtilities.cosineDistance(embedding1, embedding2)
// Returns: Float (0.0 = identical, 2.0 = opposite)
```

### validateEmbedding
Check if an embedding is valid.

```swift
let isValid = SpeakerUtilities.validateEmbedding(embedding)
// Returns: Bool
```

### findClosestSpeaker
Find the closest matching speaker from candidates.

```swift
let (speaker, distance) = SpeakerUtilities.findClosestSpeaker(
    embedding: targetEmbedding,
    candidates: knownSpeakers
)
```

### averageEmbeddings
Calculate the average of multiple embeddings.

```swift
if let avgEmbedding = SpeakerUtilities.averageEmbeddings(embeddingArray) {
    // Use averaged embedding
}
```

## Threading and Concurrency

`SpeakerManager` is thread-safe:
- Uses internal `DispatchQueue` with concurrent reads and barrier writes
- All public methods can be safely called from any thread
- No `@unchecked Sendable` usage (proper Swift concurrency)

## Cosine Distance Interpretation

Understanding distance values for speaker matching:

| Distance | Interpretation | Action |
|----------|----------------|--------|
| < 0.3 | Same speaker (very high confidence) | Assign & update |
| 0.3-0.5 | Same speaker (high confidence) | Assign & update |
| 0.5-0.7 | Same speaker (medium confidence) | Assign, maybe update |
| 0.7-0.9 | Different speakers (medium confidence) | Create new |
| > 0.9 | Different speakers (high confidence) | Create new |

## Best Practices

1. **Keep one SpeakerManager per stream** - Don't share across independent audio streams
2. **Reset between sessions** - Call `reset()` when starting a new recording
3. **Pre-load known speakers** - Use `initializeKnownSpeakers()` for recognition scenarios
4. **Tune thresholds** - Adjust `speakerThreshold` based on your audio quality:
   - Clean audio: 0.6-0.7
   - Noisy audio: 0.7-0.8
5. **Monitor speaker count** - Excessive speakers may indicate threshold issues
6. **Enable enrollment** - Update speaker names for better user experience
7. **Use VAD preprocessing** - Filter audio with Voice Activity Detection before diarization (see below)

## Improving Diarization Quality with VAD

Voice Activity Detection (VAD) can significantly improve speaker diarization by:
- Filtering out silence and non-speech audio
- Reducing false speaker detections from noise
- Improving embedding quality by focusing on actual speech
- Lowering computational cost by skipping silent regions

### Basic VAD + Diarization Pipeline

```swift
import FluidAudio

class VADEnhancedDiarizer {
    let vadManager: VadManager
    let diarizer: DiarizerManager

    init() async throws {
        // Initialize VAD
        vadManager = try await VadManager(
            config: VadConfig(defaultThreshold: 0.5)
        )

        // Initialize diarizer
        let models = try await DiarizerModels.downloadIfNeeded()
        diarizer = DiarizerManager()
        diarizer.initialize(models: models)
    }

    func processWithVAD(_ audio: [Float]) async throws -> DiarizationResult {
        // Step 1: Run VAD to detect speech regions
        let vadResults = try await vadManager.process(audio)

        // Step 2: Extract speech-only segments
        let speechSegments = extractSpeechSegments(audio: audio, vadResults: vadResults)

        // Step 3: Run diarization only on speech segments
        var allSegments: [TimedSpeakerSegment] = []

        for (speechAudio, offset) in speechSegments {
            let result = try diarizer.performCompleteDiarization(speechAudio)

            // Adjust timestamps to account for VAD filtering
            let adjustedSegments = result.segments.map { segment in
                TimedSpeakerSegment(
                    speakerId: segment.speakerId,
                    startTimeSeconds: offset + segment.startTimeSeconds,
                    endTimeSeconds: offset + segment.endTimeSeconds
                )
            }
            allSegments.append(contentsOf: adjustedSegments)
        }

        return DiarizationResult(segments: allSegments)
    }

    private func extractSpeechSegments(
        audio: [Float],
        vadResults: [VadResult]
    ) -> [(audio: [Float], timeOffset: Float)] {
        var segments: [(audio: [Float], timeOffset: Float)] = []
        let chunkSize = VadManager.chunkSize
        let sampleRate = Float(VadManager.sampleRate)

        var currentSegment: [Float] = []
        var segmentStart: Int = 0
        var inSpeech = false

        for (index, result) in vadResults.enumerated() {
            let chunkStart = index * chunkSize
            let chunkEnd = min(chunkStart + chunkSize, audio.count)

            if result.isVoiceActive {
                if !inSpeech {
                    // Start new speech segment
                    segmentStart = chunkStart
                    inSpeech = true
                }
                currentSegment.append(contentsOf: audio[chunkStart..<chunkEnd])
            } else if inSpeech {
                // End of speech segment
                if currentSegment.count > sampleRate * 1.0 { // Min 1 second
                    let timeOffset = Float(segmentStart) / sampleRate
                    segments.append((currentSegment, timeOffset))
                }
                currentSegment = []
                inSpeech = false
            }
        }

        // Handle final segment
        if !currentSegment.isEmpty && currentSegment.count > Int(sampleRate * 1.0) {
            let timeOffset = Float(segmentStart) / sampleRate
            segments.append((currentSegment, timeOffset))
        }

        return segments
    }
}
```

### VAD Configuration for Diarization

Different thresholds work better for different scenarios:

```swift
// Noisy environments - more aggressive filtering
let config = VadConfig(defaultThreshold: 0.7)

// Clean audio - capture more speech
let config = VadConfig(defaultThreshold: 0.3)

// Balanced (recommended starting point)
let config = VadConfig(defaultThreshold: 0.5)
```

### When to Use VAD Preprocessing

**Highly Recommended:**
- Noisy recordings (background music, street noise)
- Long silence periods (meetings with pauses)
- Multi-speaker environments with frequent gaps
- Phone call recordings
- Podcast/interview recordings

**Less Critical:**
- Already clean, pre-processed audio
- Continuous speech without pauses
- When latency is critical (VAD adds processing time)

**Performance Impact:**
- VAD processing: ~0.5-2ms per 256ms chunk (very fast)
- Diarization on reduced audio: 30-50% faster (skips silence)
- Overall quality improvement: Reduces false speakers by 20-40%

## Example: Real-time Diarization with Speaker Names

```swift
class RealtimeDiarizer {
    let diarizer: DiarizerManager
    let speakerManager: SpeakerManager

    init() async throws {
        let models = try await DiarizerModels.downloadIfNeeded()
        diarizer = DiarizerManager()
        diarizer.initialize(models: models)
        speakerManager = diarizer.speakerManager
    }

    func processChunk(_ audio: [Float]) throws {
        let result = try diarizer.performCompleteDiarization(audio)

        for segment in result.segments {
            // Get speaker with updated name
            if let speaker = speakerManager.getSpeaker(for: segment.speakerId) {
                print("\(speaker.name): '\(segment.text ?? "")' at \(segment.startTimeSeconds)s")
            }
        }
    }

    func enrollSpeaker(name: String, audio: [Float]) throws {
        let result = try diarizer.performCompleteDiarization(audio)

        if let firstSegment = result.segments.first,
           let speaker = speakerManager.getSpeaker(for: firstSegment.speakerId) {
            speaker.name = name
            print("Enrolled \(name) as \(speaker.id)")
        }
    }
}
```

## Performance Characteristics

- **Memory:** ~50 KB per speaker (50 historical embeddings × 256 dims × 4 bytes + metadata)
- **Speaker lookup:** O(n) where n = number of speakers (typically < 10 in conversations)
- **Thread-safe:** Concurrent reads, barrier writes
- **Embedding update:** O(1) using exponential moving average

## API Reference

### SpeakerManager Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `assignSpeaker(_:speechDuration:confidence:)` | `Speaker?` | Assign/create speaker from embedding |
| `initializeKnownSpeakers(_:mode:preserveIfPermanent:)` | `Void` | Pre-load known speaker profiles |
| `findSpeaker(with:speakerThreshold:)` | `(id: String?, distance: Float)` | Find speaker that matches an embedding |
| `findMatchingSpeakers(with:speakerThreshold:)` | `[(id: String, distance: Float)]` | Find all speakers that match an embedding |
| `findSpeakers(where:)` | `[String]` | Find all speakers that meet a certain predicate
| `findMergeablePairs(speakerThreshold:excludeIfBothPermanent:)` | `[(speakerToMerge: String, destination: String)]` | Find all pairs of very similar speakers |
| `removeSpeaker(_:keepIfPermanent:)` | `Void` | Remove a speaker from the database |
| `removeSpeakersInactive(since:keepIfPermanent:)` | `Void` | Remove speakers inactive since a given date |
| `removeSpeakersInactive(for:keepIfPermanent:)` | `Void` | Remove speakers inactive for a given duration |
| `removeSpeakers(where:)` | `Void` | Remove speakers that satisfy a given predicate |
| `removeSpeakers(where:keepIfPermanent:)` | `Void` | Remove speakers that satisfy a given predicate |
| `mergeSpeaker(_:into:mergedName:stopIfPermanent:)` | `Void` | Merge a speaker into another one |
| `upsertSpeaker(_:)` | `Void` | Update or insert speaker (from object) |
| `upsertSpeaker(id:currentEmbedding:duration:...)` | `Void` | Update or insert speaker (from params) |
| `getSpeaker(for:)` | `Speaker?` | Get speaker by ID |
| `getAllSpeakers()` | `[String: Speaker]` | Get all speakers (debugging) |
| `getSpeakerList()` | `[Speaker]` | Get array of all speakers (debugging) |
| `hasSpeaker(_:)` | `Bool` | Check if database has a speaker with a given ID |
| `reset(keepIfPermanent:)` | `Void` | Clear speaker database |
| `resetPermanentFlags()` | `Void` | Mark all speakers as not permanent |
| `getCurrentSpeakerNames()` | `[String]` | Get sorted speaker IDs |
| `getGlobalSpeakerStats()` | `(Int, Float, Float, Int)` | Aggregate statistics |

### SpeakerManager Properties

| Property | Type | Description |
|----------|------|-------------|
| `speakerThreshold` | `Float` | Max distance for speaker assignment |
| `embeddingThreshold` | `Float` | Max distance for embedding updates |
| `minSpeechDuration` | `Float` | Min duration to create speaker (seconds) |
| `minEmbeddingUpdateDuration` | `Float` | Min duration to update embeddings (seconds) |
| `speakerCount` | `Int` | Number of tracked speakers |
| `speakerIds` | `[String]` | Sorted array of speaker IDs |
| `permanentSpeakerIds` | `[String]` | Sorted array of speaker IDs of permanent speakers |

### Speaker Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `String` | Unique speaker identifier |
| `name` | `String` | Display name (defaults to ID) |
| `currentEmbedding` | `[Float]` | 256-dim L2-normalized embedding |
| `duration` | `Float` | Total speech duration (seconds) |
| `createdAt` | `Date` | Creation timestamp |
| `updatedAt` | `Date` | Last update timestamp |
| `updateCount` | `Int` | Number of embedding updates |
| `rawEmbeddings` | `[RawEmbedding]` | Historical embeddings (max 50) |
| `isPermanent` | `Bool` | Permanence flag |

### Speaker Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `updateMainEmbedding(duration:embedding:segmentId:alpha:)` | `Void` | Update using EMA |
| `addRawEmbedding(_:)` | `Void` | Add historical embedding |
| `removeRawEmbedding(segmentId:)` | `RawEmbedding?` | Remove by segment ID |
| `recalculateMainEmbedding()` | `Void` | Recalculate from raw embeddings |
| `mergeWith(_:keepName:)` | `Void` | Merge with another speaker |
| `toSendable()` | `SendableSpeaker` | Convert to sendable format |

### SpeakerUtilities Static Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `cosineDistance(_:_:)` | `Float` | Distance between embeddings |
| `validateEmbedding(_:minMagnitude:)` | `Bool` | Check embedding validity |
| `findClosestSpeaker(embedding:candidates:)` | `(Speaker?, Float)` | Find closest match |
| `averageEmbeddings(_:)` | `[Float]?` | Average multiple embeddings |
| `createSpeaker(id:name:duration:embedding:config:)` | `Speaker?` | Create validated speaker |
| `updateEmbedding(current:new:alpha:)` | `[Float]?` | EMA update (pure function) |
| `reassignSegment(segmentId:from:to:)` | `Bool` | Move segment between speakers |

## See Also

- [SpeakerDiarization.md](SpeakerDiarization.md) - Full diarization pipeline documentation
- [API.md](API.md) - Complete API reference
- `DiarizerManager` - Main diarization orchestrator
- `Speaker` class - Speaker data model
- `SpeakerUtilities` - Utility functions for speaker operations
