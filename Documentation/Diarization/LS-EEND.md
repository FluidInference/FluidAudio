# LS-EEND Streaming Speaker Diarization

## Overview

LS-EEND (Long-Form Streaming End-to-End Neural Diarization) answers "who spoke when" in real-time. A causal Conformer encoder with a retention mechanism feeds an online attractor decoder that tracks speaker identities frame by frame, without separate VAD, segmentation, or clustering.

**Key specs:**
- 4–10 simultaneous speakers depending on variant (see below)
- ~100ms frame resolution (10 Hz output)
- Handles recordings up to one hour
- 8000 Hz input sample rate (automatic resampling via `processComplete(audioFileURL:)`)
- Frame-in-frame-out streaming with speculative preview frames
- CoreML-optimized for Apple Silicon

**Limitations:**
- 8000 Hz sample rate — lower audio fidelity than 16 kHz models
- Speaker identity is local to the recording; persistent speaker enrollment may be unreliable
- Variants are domain-specialized: using the wrong variant for a domain hurts accuracy

---

## Variant Selection

Each variant is a separate CoreML model trained on a specific corpus. Choose the one that best matches your audio.

### `.ami` — In-person meetings
Multi-speaker conference room recordings with close-talk and distant microphones.
Best for: boardroom meetings, panel discussions, speakers in a shared physical space.
- **DER (AMI test set):** 20.76%
- **Max speakers:** 4

### `.callhome` — Phone calls
Telephone conversations with codec noise and narrow bandwidth.
Best for: call center recordings, customer service calls, telephony audio.
- **DER (CALLHOME test set):** 12.11%
- **Max speakers:** 7

### `.dihard2` — Difficult mixed conditions
Dinner parties, clinical interviews, conference rooms, multi-channel arrays, child speech.
Best for: challenging acoustics, heavy overlap, non-standard recording setups.
- **DER (DIHARD II test set):** 27.58%
- **Max speakers:** 10

### `.dihard3` — In-the-wild conversations *(default)*
Podcasts, audiobooks, broadcast media, YouTube, field recordings — deliberately broad.
Best for: unknown or mixed recording conditions; the safest general-purpose choice.
- **DER (DIHARD III test set):** 19.61%
- **Max speakers:** 10

## Call Flow

### Complete Audio

```
LSEENDDiarizer.processComplete(_:sourceSampleRate:)
  |
  |-- normalizeSamplesLocked()            resample to 8 kHz when needed
  |-- processCompleteInternal()
  |     |
  |     |-- engine.createSession()        fresh streaming session
  |     |-- session.pushAudio()           run full buffer through LS-EEND
  |     |-- session.finalize()            flush remaining committed frames
  |     |-- DiarizerTimeline.addChunk()
  |     |-- DiarizerTimeline.finalize()
  |
  |-- return DiarizerTimeline
```

Under the hood, offline engine inference follows this path:

```
LSEENDInferenceHelper.infer(samples:sampleRate:)
  |
  |-- resampleIfNeeded()
  |-- offlineFeatureExtractor.extractFeatures(audio:)
  |     |
  |     |-- computeFlatTransposed()       STFT -> mel spectrogram
  |     |-- applyLogMelCumMeanNormalization()
  |     |-- spliceAndSubsample()
  |
  |-- createSession(inputSampleRate:)
  |-- session.ingestFeatures(features)
  |     |
  |     |-- FOR EACH MODEL FRAME:
  |     |     |
  |     |     |-- predictStep(frame:state:ingest:decode:)
  |     |     |     |
  |     |     |     |-- write frame into preallocated MLMultiArray
  |     |     |     |-- pass 6 recurrent state tensors:
  |     |     |     |     encRetKv, encRetScale, encConvCache,
  |     |     |     |     decRetKv, decRetScale, topBuffer
  |     |     |     |-- CoreML model.prediction()
  |     |     |     |-- read logits + next recurrent state tensors
  |     |     |
  |     |     |-- append committed full-output logits
  |
  |-- flushTail(from:pendingFrames:)      decode remaining tail frames
  |-- cropRealTracks()                    drop 2 boundary tracks
  |-- applyingSigmoid()                   logits -> probabilities
  |-- snapshot()
  |-- return LSEENDInferenceResult
```

### Streaming

```
LSEENDDiarizer.addAudio(_:sourceSampleRate:)
  |
  |-- normalizeSamplesLocked()            resample to 8 kHz when needed
  |-- append to pendingAudio
  |
  v
LSEENDDiarizer.process()
  |
  |-- engine.createSession()              first call only
  |-- session.pushAudio(pendingAudio)
  |     |
  |     |-- featureExtractor.pushAudio()
  |     |     |
  |     |     |-- append raw audio to buffer
  |     |     |-- appendSTFTFrames()
  |     |     |-- applyLogMelCumMeanNormalization()
  |     |     |-- emitModelFrames()
  |     |
  |     |-- ingestFeatures()
  |     |     |
  |     |     |-- predictStep(... ingest=1, decode=0) for stable frames
  |     |     |-- update 6 recurrent state tensors each step
  |     |     |-- append committed full-output logits
  |     |
  |     |-- copy current state
  |     |-- flushTail(from:pendingFrames:) on copied state
  |     |     -> speculative preview logits
  |     |
  |     |-- cropRealTracks()              remove boundary tracks
  |     |-- applyingSigmoid()             logits -> probabilities
  |     |-- return committed + preview update
  |
  |-- DiarizerTimeline.addChunk()
  |-- return DiarizerTimelineUpdate
```

---

## File Structure

```
Sources/FluidAudio/Diarizer/LS-EEND/
├── LSEENDDiarizer.swift           # High-level Diarizer protocol implementation
├── LSEENDInference.swift          # LSEENDInferenceHelper, LSEENDStreamingSession
├── LSEENDFeatureExtraction.swift  # Internal offline + streaming feature extraction
├── LSEENDSupport.swift            # Supporting data types, metadata, result structs, errors
└── LSEENDEvaluation.swift         # DER computation, RTTM parsing/writing, collar masking,
                                   #   optimal speaker assignment
```

---

## LSEENDDiarizer

The primary entry point. Implements the `Diarizer` protocol — the same API as `SortformerDiarizer`.

### Initialization

```swift
// Simple init (all parameters optional)
let diarizer = LSEENDDiarizer(
    computeUnits: .cpuOnly,       // Default: .cpuOnly (fastest for this model)
    onsetThreshold: 0.5,          // Probability to start a speech segment
    offsetThreshold: 0.5,         // Probability to end a speech segment
    onsetPadFrames: 0,            // Frames prepended to each segment
    offsetPadFrames: 0,           // Frames appended to each segment
    minFramesOn: 0,               // Discard segments shorter than this
    minFramesOff: 0,              // Close gaps shorter than this
    maxStoredFrames: nil          // Cap on retained finalized frames (nil = unlimited)
)

// Or pass a DiarizerTimelineConfig directly
let config = DiarizerTimelineConfig(onsetThreshold: 0.4, onsetPadFrames: 1)
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly, timelineConfig: config)
```

### Loading Models

```swift
// Download from HuggingFace (cached after first call)
try await diarizer.initialize(variant: .dihard3)   // default variant

// From a pre-built descriptor
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: .ami)
try diarizer.initialize(descriptor: descriptor)

// From a pre-loaded engine
let engine = try LSEENDInferenceHelper(descriptor: descriptor)
diarizer.initialize(engine: engine)
```

### Offline Processing

```swift
// From a file URL (resamples to 8kHz automatically)
let timeline = try diarizer.processComplete(audioFileURL: audioURL)

// From raw samples (specify sample rate if it's not 8kHz already)
let timeline = try diarizer.processComplete(
    samples,
    sourceSampleRate: 8000, 
    finalizeOnCompletion: true,
    progressCallback: { processed, total, chunks in
        print("\(processed)/\(total) samples")
    }
)
```

### Streaming
The `sourceSampleRate` argument is only needed if the audio samples are not already at 8kHz. 

```swift
// Push audio in chunks
diarizer.addAudio(audioChunk, sourceSampleRate: 8000)    // [Float] or any Collection<Float>
if let update = try diarizer.process() {
    for segment in update.finalizedSegments { ... }
    for tentative in update.tentativeSegments { ... } 
}

// Convenience: add + process in one call
if let update = try diarizer.process(samples: audioChunk) { ... }

// Flush remaining frames at the end of a stream
try diarizer.finalizeSession()
let finalTimeline = diarizer.timeline
```

Notes:
- `finalizeSession()` flushes the remaining audio by padding the end with silence.

### Speaker Enrollment

Use speaker enrollment to warm LS-EEND with a known speaker before the live stream starts. Enrollment keeps the active streaming session, resets the visible timeline back to frame 0, and preserves the speaker name inside the `DiarizerTimeline`.

```swift
let speaker = try diarizer.enrollSpeaker(
    withSamples: enrollmentAudio,
    sourceSampleRate: 16_000,
    named: "Alice",
    overwritingAssignedSpeakerName: false
)

// Later complete-buffer runs can keep the enrolled session state.
let timeline = try diarizer.processComplete(
    meetingAudio,
    sourceSampleRate: 16_000,
    keepingEnrolledSpeakers: true
)
```

Notes:
- Enrollment is per diarizer instance. Recreate or `reset()` the diarizer to start a fresh session.
- Enrollment can help with live identity continuity, but it is still less reliable than the WeSpeaker/Pyannote speaker database.
- Speaker slots are still chronological. Use `overwritingAssignedSpeakerName: false` if you want enrollment to fail instead of replacing the name on an already-named slot.

### Enrollment Limitations (Integration Feedback)

Real-world integration testing with 4-speaker audio reveals specific enrollment weaknesses compared to Sortformer:

**Score range:** LS-EEND scores are bounded between `sigmoid(-1)` and `sigmoid(1)`, roughly **0.2 to 0.8**. Internally the model applies sigmoid to cosine similarity scores, so raw outputs will never reach the 0.9+ confidence levels that external post-processing might suggest.

**Close-voice slot collision:** When enrolling speakers one at a time (strict enrollment path), LS-EEND's internal collision logic can reject a speaker whose voice is too similar to an already-enrolled slot. In a 4-speaker test, 3 speakers enrolled with strong mapping (~0.9 post-normalized confidence), but the 4th failed due to "too close to existing speaker." Sortformer auto-mapped all 4 with high confidence in the same test.

**Score-extraction fallback is weaker:** An alternative integration strategy — extracting per-slot scores over a sample, building a full score matrix, then running global assignment (e.g. Hungarian algorithm + threshold) — avoids hard enrollment rejection but produces weaker results. Non-dominant speakers can drop to ~0.2 confidence and one speaker can dominate multiple slot assignments.

**Root cause:** LS-EEND is an end-to-end model, making it difficult to force speaker registration into a specific slot. There is currently no API for per-slot similarity outputs or explicit slot-lock assignment. Suppressing existing attractors may be a path forward, but this has not been validated.

**Training data gap:** Sortformer was trained on a large volume of real-world data, giving it stronger generalization for speaker identity. LS-EEND was trained primarily on simulated data and then fine-tuned on real data — the base model without fine-tuning performs poorly.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `timeline` | `DiarizerTimeline` | Accumulated finalized results |
| `isAvailable` | `Bool` | Whether the model is loaded |
| `numFramesProcessed` | `Int` | Total committed frames processed |
| `targetSampleRate` | `Int?` | Expected input sample rate (8000) |
| `modelFrameHz` | `Double?` | Output frame rate (~10.0 Hz) |
| `numSpeakers` | `Int?` | Real speaker track count (`realOutputDim`) |
| `streamingLatencySeconds` | `Double?` | Minimum latency before first frame |
| `decodeMaxSpeakers` | `Int?` | Total model output slots (including boundary tracks) |
| `computeUnits` | `MLComputeUnits` | CoreML compute units |
| `timelineConfig` | `DiarizerTimelineConfig` | Current post-processing config |

### Lifecycle

```swift
try diarizer.finalizeSession() // Flush trailing context before reading final output
diarizer.reset()     // Reset streaming state for a new audio stream (keeps model loaded)
diarizer.cleanup()   // Release all resources including the loaded model
```

---

## LSEENDInferenceHelper

Lower-level engine for direct CoreML inference. Use this when you need access to raw logits, want to manage sessions manually, or are building tooling around the model.

### Creating an Engine

```swift
// Synchronous — model loading happens here
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: .dihard3)
let engine = try LSEENDInferenceHelper(
    descriptor: descriptor,
    computeUnits: .cpuOnly   // default
)
```

### Offline Inference

```swift
// From raw samples + sample rate (resamples internally if needed)
let result: LSEENDInferenceResult = try engine.infer(samples: audio, sampleRate: 16000)

// From a file (reads and resamples to targetSampleRate)
let result: LSEENDInferenceResult = try engine.infer(audioFileURL: url)
```

### Streaming Inference

```swift
// Create a session (inputSampleRate must equal engine.targetSampleRate)
let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

// Or with a caller-owned mel spectrogram (for thread isolation)
let mel = AudioMelSpectrogram(...)
let session = try engine.createSession(inputSampleRate: engine.targetSampleRate, melSpectrogram: mel)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `descriptor` | `LSEENDModelDescriptor` | Model variant and file paths |
| `computeUnits` | `MLComputeUnits` | CoreML compute units |
| `metadata` | `LSEENDModelMetadata` | Decoded model configuration |
| `model` | `MLModel` | Loaded CoreML model |
| `targetSampleRate` | `Int` | Expected input sample rate |
| `modelFrameHz` | `Double` | Output frame rate |
| `streamingLatencySeconds` | `Double` | Minimum latency before first output |
| `decodeMaxSpeakers` | `Int` | Total output slots including boundary tracks |

---

## LSEENDStreamingSession

A stateful streaming session created by `LSEENDInferenceHelper.createSession(inputSampleRate:)`. Maintains all six recurrent state tensors across calls.

> **Not thread-safe.** All calls must be serialized.

```swift
let session = try engine.createSession(inputSampleRate: 8000)

// Feed audio incrementally
while let chunk = audioSource.next() {
    if let update = try session.pushAudio(chunk) {
        // update.probabilities — committed, final frames
        // update.previewProbabilities — speculative frames, will be refined
    }
}

// Flush remaining frames and close the session
if let final = try session.finalize() {
    // Process any remaining frames
}

// Get the complete assembled result at any point
let result: LSEENDInferenceResult = session.snapshot()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `pushAudio(_ chunk: [Float])` | `LSEENDStreamingUpdate?` | Feed audio; returns committed + preview frames, or `nil` if no frames ready |
| `finalize()` | `LSEENDStreamingUpdate?` | Flush remaining frames and seal the session |
| `snapshot()` | `LSEENDInferenceResult` | Assemble full result from all frames emitted so far |

| Property | Type | Description |
|----------|------|-------------|
| `inputSampleRate` | `Int` | Sample rate this session was created with |

---

## Data Types

### LSEENDMatrix

A row-major 2D `Float` matrix used throughout LS-EEND. Rows are time frames; columns are speakers or feature dimensions.

### LSEENDInferenceResult

Output from `LSEENDInferenceHelper.infer(...)` or `LSEENDStreamingSession.snapshot()`.

| Property | Type | Description |
|----------|------|-------------|
| `logits` | `LSEENDMatrix` | Speaker logits, boundary tracks removed. Shape: `[frames, realOutputDim]` |
| `probabilities` | `LSEENDMatrix` | Sigmoid of `logits`. Shape: `[frames, realOutputDim]` |
| `fullLogits` | `LSEENDMatrix` | Raw logits including boundary tracks. Shape: `[frames, fullOutputDim]` |
| `fullProbabilities` | `LSEENDMatrix` | Sigmoid of `fullLogits` |
| `frameHz` | `Double` | Output frame rate in Hz |
| `durationSeconds` | `Double` | Duration of input audio processed |

### LSEENDStreamingUpdate

Returned by `LSEENDStreamingSession.pushAudio(_:)` and `finalize()`. Contains two regions:

| Property | Type | Description |
|----------|------|-------------|
| `startFrame` | `Int` | Frame index where committed region begins |
| `logits` | `LSEENDMatrix` | Committed speaker logits (boundary tracks removed) |
| `probabilities` | `LSEENDMatrix` | Committed speaker probabilities |
| `previewStartFrame` | `Int` | Frame index where preview region begins |
| `previewLogits` | `LSEENDMatrix` | Speculative logits for buffered-but-unconfirmed frames |
| `previewProbabilities` | `LSEENDMatrix` | Speculative probabilities (will be refined by future audio) |
| `frameHz` | `Double` | Output frame rate |
| `durationSeconds` | `Double` | Cumulative audio duration fed so far |
| `totalEmittedFrames` | `Int` | Running total of committed frames across all updates |

**Committed vs preview:** Committed frames have passed through the full causal encoder and are final. Preview frames are decoded by zero-padding the pending encoder state — they are a speculative "look ahead" that will be updated by the next `pushAudio` call.

---

## Feature Extraction

Feature extraction is internal. `LSEENDDiarizer` and `LSEENDInferenceHelper` handle it automatically.

---

## Model Loading

### LSEENDVariant

```swift
public typealias LSEENDVariant = ModelNames.LSEEND.Variant

// Cases
LSEENDVariant.ami        // rawValue: "AMI"
LSEENDVariant.callhome   // rawValue: "CALLHOME"
LSEENDVariant.dihard2    // rawValue: "DIHARD II"
LSEENDVariant.dihard3    // rawValue: "DIHARD III"
```

| Property | Type | Description |
|----------|------|-------------|
| `rawValue` | `String` | Dataset name string (e.g. `"DIHARD III"`) |
| `description` | `String` | Same as `rawValue` (`CustomStringConvertible`) |
| `id` | `String` | Same as `rawValue` (`Identifiable`) |
| `name` | `String` | Internal checkpoint name (e.g. `"ls_eend_dih3_step"`) |
| `stem` | `String` | `"<rawValue>/<name>"` — path prefix within the repo |
| `modelFile` | `String` | Relative path to the `.mlmodelc` file |
| `configFile` | `String` | Relative path to the `.json` metadata file |
| `fileNames` | `[String]` | `[modelFile, configFile]` |

### LSEENDModelDescriptor

Locates the CoreML model and metadata JSON for a variant.

```swift
// Download from HuggingFace (cached after first call)
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(
    variant: .dihard3,               // default
    cacheDirectory: customURL,       // optional; defaults to ~/Library/Application Support/FluidAudio/Models
    computeUnits: .cpuOnly,          // optional
    progressHandler: { progress in } // optional
)

// From explicit local paths
let descriptor = LSEENDModelDescriptor(
    variant: .dihard3,
    modelURL: URL(fileURLWithPath: "/path/to/model.mlmodelc"),
    metadataURL: URL(fileURLWithPath: "/path/to/metadata.json")
)
```

| Property | Type | Description |
|----------|------|-------------|
| `variant` | `LSEENDVariant` | Model variant |
| `modelURL` | `URL` | Path to `.mlmodelc` or `.mlpackage` |
| `metadataURL` | `URL` | Path to JSON metadata file |

### LSEENDModelMetadata

Decoded from the JSON file at `descriptor.metadataURL`. Read via `engine.metadata`.

| Property | Type | Description |
|----------|------|-------------|
| `realOutputDim` | `Int` | Usable speaker tracks (`fullOutputDim - 2`) |
| `frameHz` | `Double` | Output frame rate (frames per second) |
| `targetSampleRate` | `Int` | Required audio sample rate |
| `streamingLatencySeconds` | `Double` | Minimum startup latency before the first stable output |

---

## Evaluation API

`LSEENDEvaluation` provides RTTM parsing/writing and DER computation for benchmarks. If you need detailed scoring workflows, read the source or move that material into a separate evaluation-specific doc.

---

## Error Handling

All LS-EEND errors conform to `LocalizedError` and are thrown as `LSEENDError`.

| Case | Thrown when |
|------|-------------|
| `.invalidMetadata(String)` | Metadata JSON is malformed or contains invalid values |
| `.invalidMatrixShape(String)` | Matrix dimensions are mismatched or negative |
| `.unsupportedAudio(String)` | Wrong sample rate, empty buffer, or finalized session |
| `.modelPredictionFailed(String)` | CoreML forward pass failed, or model not initialized |
| `.missingFeature(String)` | Required output tensor absent from CoreML prediction |
| `.invalidPath(String)` | File path cannot be resolved |
| `.modelLoadFailed(String)` | CoreML model could not be loaded or compiled |

```swift
do {
    let timeline = try diarizer.processComplete(audioFileURL: url)
} catch let error as LSEENDError {
    switch error {
    case .unsupportedAudio(let message): print("Audio problem: \(message)")
    case .modelLoadFailed(let message): print("Model problem: \(message)")
    default: print(error.localizedDescription)
    }
}
```

---

## Usage Examples

### Offline File Processing

```swift
let diarizer = LSEENDDiarizer()
try await diarizer.initialize(variant: .ami)

let timeline = try diarizer.processComplete(audioFileURL: URL(fileURLWithPath: "meeting.wav"))
for segment in timeline.allSegments {
    print("Speaker \(segment.speakerIndex): \(segment.startTime)s–\(segment.endTime)s")
}
```

### Streaming from Microphone

```swift
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
try await diarizer.initialize(variant: .dihard3)

// Feed 8kHz mono chunks from AVAudioEngine
audioEngine.installTap(onBus: 0, bufferSize: 1600, format: format) { buffer, _ in
    let samples = Array(UnsafeBufferPointer(
        start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    diarizer.addAudio(samples)
    if let update = try? diarizer.process() {
        DispatchQueue.main.async { updateUI(diarizer.timeline) }
    }
}
```

### Low-Level Engine + Session

```swift
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: .callhome)
let engine = try LSEENDInferenceHelper(descriptor: descriptor)
let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

for chunk in chunkedAudio(samples, chunkSize: 800) {
    guard let update = try session.pushAudio(chunk) else { continue }
    // Committed frames: update.probabilities [newFrames × speakers]
    // Preview frames:   update.previewProbabilities [previewFrames × speakers]
}

let final = try session.finalize()
let result = session.snapshot()   // LSEENDInferenceResult
```

---

## CLI

```bash
# Diarize a single file (default variant: dihard3)
swift run fluidaudiocli lseend audio.wav
swift run fluidaudiocli lseend audio.wav --variant callhome
swift run fluidaudiocli lseend audio.wav --threshold 0.4 --median-width 5 --output result.json

# Benchmark on AMI (downloads dataset automatically)
swift run fluidaudiocli lseend-benchmark --auto-download --variant ami
swift run fluidaudiocli lseend-benchmark --variant callhome --threshold 0.35 --collar 0.25
swift run fluidaudiocli lseend-benchmark --variant dihard3 --output results.json --max-files 10
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `dihard3` | `ami` \| `callhome` \| `dihard2` \| `dihard3` |
| `--threshold` | `0.5` | Speaker activity binarization threshold |
| `--median-width` | `1` | Median filter width in frames (1 = disabled) |
| `--collar` | `0.0` | Collar in seconds around transitions (benchmark only) |
| `--onset` | — | Override onset threshold separately from `--threshold` |
| `--offset` | — | Override offset threshold separately from `--threshold` |
| `--pad-onset` | `0` | Frames prepended to each segment |
| `--pad-offset` | `0` | Frames appended to each segment |
| `--min-duration-on` | `0.0` | Minimum segment duration in seconds |
| `--min-duration-off` | `0.0` | Minimum gap duration in seconds |
| `--output` | — | Path to save JSON results |
| `--auto-download` | — | Auto-download AMI dataset if missing (benchmark only) |
| `--max-files` | — | Limit number of files processed (benchmark only) |
| `--verbose` | — | Print per-meeting debug output (benchmark only) |

---

## Model Files on HuggingFace

Hosted at [FluidInference/lseend-coreml](https://huggingface.co/FluidInference/lseend-coreml). Downloaded automatically on first use and cached at `~/Library/Application Support/FluidAudio/Models/`.

| Variant | Model file | Config file |
|---------|-----------|-------------|
| `.ami` | `AMI/ls_eend_ami_step.mlmodelc` | `AMI/ls_eend_ami_step.json` |
| `.callhome` | `CALLHOME/ls_eend_callhome_step.mlmodelc` | `CALLHOME/ls_eend_callhome_step.json` |
| `.dihard2` | `DIHARD II/ls_eend_dih2_step.mlmodelc` | `DIHARD II/ls_eend_dih2_step.json` |
| `.dihard3` | `DIHARD III/ls_eend_dih3_step.mlmodelc` | `DIHARD III/ls_eend_dih3_step.json` |

Pre-fetch before running:

```bash
swift run fluidaudiocli download --repo lseend
```

---

## References

- [LS-EEND Paper (arXiv 2410.06670)](https://arxiv.org/abs/2410.06670) — Di Liang, Xiaofei Li. *LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction.* IEEE TASLP.
- [LS-EEND GitHub Repository](https://github.com/Audio-WestlakeU/FS-EEND)
- [HuggingFace Models](https://huggingface.co/FluidInference/lseend-coreml)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [CALLHOME Corpus](https://catalog.ldc.upenn.edu/LDC97S42)
- [DIHARD Challenge](https://dihardchallenge.github.io/)
