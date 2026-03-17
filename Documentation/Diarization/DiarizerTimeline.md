# DiarizerTimeline

The unified timeline API for streaming and offline diarization across all model variants (Sortformer and LS-EEND). 

For a condensed overview of the API properties and methods, see the [API Reference](../../API.md#diarizertimeline).

## Overview

A `DiarizerTimeline` maintains the accumulated state of a speaker diarization session. It stores both the raw frame-by-frame probabilities output by the active model and the post-processed speech segments derived from those probabilities. 

Because streaming inference models rely on lookahead (right context), the timeline distinguishes between:
1. **Finalized data**: Predictions for frames that have fully passed through the model's lookahead window. These are immutable.
2. **Tentative data**: Predictions for frames still within the lookahead window. These are speculative and may be revised as more audio is buffered.

## Components

The timeline exposes several key data structures representing different views of the diarization state.

### DiarizerTimeline

The root object returned by `Diarizer.timeline` or offline inference methods.

**Key Responsibilities:**
- Serves as the single source of truth for all speaker tracks (`DiarizerSpeaker`).
- Manages the arrays of raw probabilities (`finalizedPredictions` and `tentativePredictions`).
- Automatically routes `DiarizerChunkResult` updates to specific speaker tracks during streaming.

```swift
let timeline = diarizer.timeline

// How many frames have been locked in?
print("Finalized Frames: \(timeline.numFinalizedFrames)")

// How many speculative frames exist?
print("Tentative Frames: \(timeline.numTentativeFrames)")

// Check if any speech has been detected at all
print("Speech Detected: \(timeline.hasSegments)")

// Access the individual speaker tracks
let speakers = timeline.speakers
```

### DiarizerSpeaker

Represents a single speaker slot (e.g., "Speaker 0", "Speaker 1"). 

**Key Responsibilities:**
- Maintains the speaker's display name, which can be overridden by enrollment.
- Separates speech timestamps into `finalizedSegments` and `tentativeSegments`.
- Computes aggregate statistics like `speechDuration` across all of its segments.

When streaming, `DiarizerSpeaker.tentativeSegments` can be used to update a live UI immediately without waiting for the model's full lookahead delay.

```swift
for (_, speaker) in timeline.speakers {
    print("\(speaker.name ?? "Speaker \(speaker.index)") spoke for \(speaker.speechDuration)s")
    
    // Iterate over finalized segments
    for segment in speaker.finalizedSegments {
        print("Confirmed: [\(segment.startTime) -> \(segment.endTime)]")
    }
}
```

### DiarizerSegment

A simple struct representing a continuous region of speech for a speaker.

- `startTime` / `endTime`: Real-world timestamps in seconds.
- `startFrame` / `endFrame`: Model indices representing the segment size.
- `isFinalized`: Whether the segment falls within the guaranteed immutable region.

### DiarizerTimelineUpdate

During streaming operations (`Diarizer.process()`), the model emits incremental timeline updates rather than a full timeline.

An update contains:
- `startFrame`: The global timeline index tracking the start of this chunk.
- `finalizedPredictions`: The raw probabilities just confirmed.
- `tentativePredictions`: The raw speculative probabilities for recent frames.

## Configuration

The logic merging raw probabilities into discrete `DiarizerSegment` ranges is governed by `DiarizerTimelineConfig`. While variants like Sortformer and LS-EEND use different default window sizes or model shapes, they share the same post-processing logic.

| Configuration Field | Description | Example / Best Practice |
|---|---|---|
| `onsetThreshold` | Probability to begin a speech segment | `0.5` is standard; increase to reduce false positives |
| `offsetThreshold` | Probability to end a speech segment | Usually matches `onsetThreshold` |
| `minFramesOn` | Drops segments shorter than this length | E.g. `minDurationOn: 0.2` seconds |
| `minFramesOff` | Merges gaps between segments shorter than this | E.g. `minDurationOff: 0.5` seconds |
| `onsetPadFrames` | Number of frames to prepend to any speech onset | Useful to prevent cutting off the start of words |
| `offsetPadFrames` | Number of frames to append to any speech offset | Useful to prevent cutting off trailing syllables |

**Constructing a Config:**

```swift
// Example: Stricter onset threshold with 100ms padding and 250ms gap closure
let config = DiarizerTimelineConfig(
    numSpeakers: 4,               // For Sortformer
    frameDurationSeconds: 0.08,   // For Sortformer
    onsetThreshold: 0.6,
    offsetThreshold: 0.5,
    onsetPadSeconds: 0.1,
    offsetPadSeconds: 0.1,
    minDurationOn: 0.15,
    minDurationOff: 0.25
)
```

## Internal Architecture

When `timeline.addChunk(_:)` is called internally by the diarizer:
1. The timeline appends the `finalizedPredictions` to its internal storage.
2. It completely overwrites its previous internal `tentativePredictions` array with the new one.
3. It iterates over all `DiarizerSpeaker` tracks, evaluating the boundaries (using `onsetThreshold` and `offsetThreshold`) to grow existing segments or spawn new ones.
4. Tentative segments are cleared and rebuilt from the trailing `tentativePredictions` array during every streaming tick. 

When the stream naturally finishes, the `Diarizer` automatically invokes `timeline.finalize()`, which flushes any remaining tentative segments up to finalized status and applies the `minFramesOn` deletion rules.
