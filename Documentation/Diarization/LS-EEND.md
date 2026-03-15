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
- Speaker identity is local to the recording; no persistent speaker embeddings
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

---

## LS-EEND vs Sortformer

| Feature | LS-EEND | Sortformer |
|---------|:-------:|:----------:|
| Max speakers | 4–10 (variant-dependent) | 4 |
| Sample rate | 8000 Hz | 16000 Hz |
| Phone/telephony audio | Best (.callhome, up to 7) | Poor |
| In-person meetings | Good (.ami, up to 4) | Good |
| Wild/unconstrained audio | Good (.dihard3, up to 10) | Good |
| Background noise robustness | Good | Best |
| Speaker count > 4 | Yes (.callhome, .dihard2, .dihard3) | No |
| Domain-specialized variants | Yes (4) | No |

---

## Architecture

```
Audio (8kHz) → Mel Spectrogram → Splice & Subsample → Causal Encoder → Attractor Decoder → Probabilities
                [T, 128]             [T', inputDim]      Retention          Cross-attn      [T', speakers]
```

1. **Mel spectrogram** (`NeMoMelSpectrogram`): 128 mel bins, 25ms window (200 samples), 10ms hop (80 samples) at 8 kHz.
2. **Splice-and-subsample**: Adjacent mel frames are concatenated with a `contextRecp` context window, then subsampled to ~10 Hz model frames. Input dimension = `nMels × (2 × contextRecp + 1)`.
3. **Causal encoder**: Conformer transformer with a **retention mechanism** instead of full self-attention, giving $O(n)$ memory over long recordings. Carries six recurrent state buffers between steps.
4. **Online attractor decoder**: Cross-attention between encoder and a `topBuffer` that tracks speaker identities. Produces per-frame logits for `maxNspks` slots.
5. **Boundary tracks**: The raw model output has two extra boundary tracks that are stripped before returning results to callers. `realOutputDim` = `fullOutputDim - 2`.
6. **Post-processing** (`DiarizerTimeline`): Sigmoid → optional median filter → onset/offset thresholding → minimum duration filtering → discrete segments.

### Streaming State

Six recurrent tensors are carried between inference steps (shapes come from the metadata JSON):

| Tensor | Description |
|--------|-------------|
| `encRetKv` | Encoder retention key-value cache |
| `encRetScale` | Encoder retention normalization scale |
| `encConvCache` | Encoder convolutional cache |
| `decRetKv` | Decoder retention key-value cache |
| `decRetScale` | Decoder retention normalization scale |
| `topBuffer` | Cross-attention buffer between encoder and decoder |

### Startup Latency

The model has a `convDelay` warmup period before the decoder can produce output. Minimum latency before the first frame:

```
streamingLatencySeconds = (nFFT/2 + contextRecp×hopLength + convDelay×subsampling×hopLength) / sampleRate
```

---

## File Structure

```
Sources/FluidAudio/Diarizer/LS-EEND/
├── LSEENDDiarizer.swift           # High-level Diarizer protocol implementation
├── LSEENDInference.swift          # LSEENDInferenceEngine, LSEENDStreamingSession
├── LSEENDFeatureExtraction.swift  # LSEENDFeatureConfig, offline + streaming feature extractors
├── LSEENDSupport.swift            # Data types: LSEENDMatrix, LSEENDModelDescriptor, LSEENDVariant,
│                                  #   LSEENDModelMetadata, LSEENDStateShapes, result structs, errors
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
let engine = try LSEENDInferenceEngine(descriptor: descriptor)
diarizer.initialize(engine: engine)
```

### Offline Processing

```swift
// From a file URL (resamples to 8kHz automatically)
let timeline = try diarizer.processComplete(audioFileURL: audioURL)

// From raw samples (must already be at targetSampleRate)
let timeline = try diarizer.processComplete(
    samples,
    finalizeOnCompletion: true,
    progressCallback: { processed, total, chunks in
        print("\(processed)/\(total) samples")
    }
)
```

### Streaming

Audio must be at the model's target sample rate (8000 Hz). Resample before calling `addAudio`.

```swift
// Push audio in chunks
diarizer.addAudio(audioChunk)                            // [Float] or any Collection<Float>
if let update = try diarizer.process() {
    for segment in update.newSegments { ... }
    for tentative in update.tentativeSegments { ... }    // Speculative, may change
}

// Convenience: add + process in one call
if let update = try diarizer.process(samples: audioChunk) { ... }

// Flush remaining frames at end of stream
try diarizer.finalizeSession()
let finalTimeline = diarizer.timeline
```

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
diarizer.reset()     // Reset streaming state for a new audio stream (keeps model loaded)
diarizer.cleanup()   // Release all resources including the loaded model
```

---

## LSEENDInferenceEngine

Lower-level engine for direct CoreML inference. Use this when you need access to raw logits, want to manage sessions manually, or are building tooling around the model.

### Creating an Engine

```swift
// Synchronous — model loading happens here
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: .dihard3)
let engine = try LSEENDInferenceEngine(
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
let mel = NeMoMelSpectrogram(...)
let session = try engine.createSession(inputSampleRate: engine.targetSampleRate, melSpectrogram: mel)
```

### Streaming Simulation

Replays a file through the streaming pipeline in fixed-size chunks. Useful for benchmarking or comparing streaming vs offline output.

```swift
let simulation: LSEENDStreamingSimulationResult = try engine.simulateStreaming(
    audioFileURL: url,
    chunkSeconds: 1.0
)
print("Final DER input frames: \(simulation.result.probabilities.rows)")
for update in simulation.updates {
    print("Chunk \(update.chunkIndex): \(update.numFramesEmitted) frames emitted")
}
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `descriptor` | `LSEENDModelDescriptor` | Model variant and file paths |
| `computeUnits` | `MLComputeUnits` | CoreML compute units |
| `metadata` | `LSEENDModelMetadata` | Decoded model configuration |
| `featureConfig` | `LSEENDFeatureConfig` | Resolved audio feature parameters |
| `model` | `MLModel` | Loaded CoreML model |
| `targetSampleRate` | `Int` | Expected input sample rate |
| `modelFrameHz` | `Double` | Output frame rate |
| `streamingLatencySeconds` | `Double` | Minimum latency before first output |
| `decodeMaxSpeakers` | `Int` | Total output slots including boundary tracks |

---

## LSEENDStreamingSession

A stateful streaming session created by `LSEENDInferenceEngine.createSession(inputSampleRate:)`. Maintains all six recurrent state tensors across calls.

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

A row-major 2D `Float` matrix used throughout the pipeline. Rows are time frames; columns are speakers or feature dimensions.

```swift
// Creation
let matrix = try LSEENDMatrix(rows: 100, columns: 4, values: floats)   // validated
let matrix = LSEENDMatrix(validatingRows: 100, columns: 4, values: floats)  // unvalidated
let zeros  = LSEENDMatrix.zeros(rows: 100, columns: 4)
let empty  = LSEENDMatrix.empty(columns: 4)      // 0 rows

// Access
let value = matrix[row, col]
let rowSlice: ArraySlice<Float> = matrix.row(3)

// Transforms (all return new matrices)
matrix.appendingRows(other)          // Vertical concatenation
matrix.droppingFirstRows(n)          // Remove first n rows
matrix.slicingRows(start: 10, end: 50)
matrix.prefixingColumns(n)           // Keep first n columns
matrix.applyingSigmoid()             // Element-wise σ(x)
matrix.rowMajorRows()                // [[Float]] per row

// Properties
matrix.rows      // Int
matrix.columns   // Int
matrix.values    // [Float] flat row-major
matrix.isEmpty   // Bool
```

### LSEENDInferenceResult

Output from `LSEENDInferenceEngine.infer(...)` or `LSEENDStreamingSession.snapshot()`.

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

### LSEENDStreamingProgress

Per-chunk entry in a streaming simulation, from `LSEENDStreamingSimulationResult.updates`.

| Property | Type | Description |
|----------|------|-------------|
| `chunkIndex` | `Int` | One-based chunk index |
| `bufferSeconds` | `Double` | Cumulative audio duration fed, in seconds |
| `numFramesEmitted` | `Int` | New committed frames emitted by this chunk |
| `totalFramesEmitted` | `Int` | Running total of committed frames |
| `flush` | `Bool` | `true` for the final finalization entry |

### LSEENDStreamingSimulationResult

| Property | Type | Description |
|----------|------|-------------|
| `result` | `LSEENDInferenceResult` | Complete assembled inference result after all chunks |
| `updates` | `[LSEENDStreamingProgress]` | Per-chunk progress log |

---

## Feature Extraction

### LSEENDFeatureConfig

Resolved audio feature parameters derived from `LSEENDModelMetadata`. Constructed automatically by the engine and diarizer — you only need this if you're building custom pipelines.

| Property | Type | Description |
|----------|------|-------------|
| `sampleRate` | `Int` | Audio sample rate (e.g. 8000) |
| `winLength` | `Int` | STFT window length in samples |
| `hopLength` | `Int` | STFT hop length in samples |
| `nFFT` | `Int` | FFT size |
| `nMels` | `Int` | Mel filterbank channels |
| `contextRecp` | `Int` | Splice context half-width |
| `subsampling` | `Int` | STFT frames per model frame |
| `inputDim` | `Int` | `nMels × (2 × contextRecp + 1)` |
| `stableBlockSize` | `Int` | Minimum audio chunk for whole-frame output (`hopLength × subsampling`) |

```swift
let config = LSEENDFeatureConfig(metadata: engine.metadata)
print("Stable chunk: \(config.stableBlockSize) samples")
```

### LSEENDOfflineFeatureExtractor

Converts a complete audio buffer into model input features in one pass: STFT → log-mel with cumulative mean normalization → splice-and-subsample.

```swift
let extractor = LSEENDOfflineFeatureExtractor(metadata: engine.metadata)
let features: LSEENDMatrix = try extractor.extractFeatures(audio: samples)
// features.shape: [frames, inputDim]
```

Use `LSEENDStreamingFeatureExtractor` for incremental processing.

### LSEENDStreamingFeatureExtractor

Incremental version. Maintains internal buffers and emits model frames as audio arrives.

> **Not thread-safe.**

```swift
let extractor = LSEENDStreamingFeatureExtractor(metadata: engine.metadata)

// Feed audio incrementally
let frames: LSEENDMatrix = try extractor.pushAudio(audioChunk)

// Flush remaining frames at end of stream
let remaining: LSEENDMatrix = try extractor.finalize()
```

Both methods return an `LSEENDMatrix` with shape `[newFrames, inputDim]`, or an empty matrix if no new frames are available.

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

Decoded from the JSON file at `descriptor.metadataURL`. Describes the model's architecture and audio parameters. Read via `engine.metadata`.

| Property | Type | Description |
|----------|------|-------------|
| `inputDim` | `Int` | Feature dimension per frame |
| `fullOutputDim` | `Int` | Total output tracks (including 2 boundary tracks) |
| `realOutputDim` | `Int` | Usable speaker tracks (`fullOutputDim - 2`) |
| `encoderLayers` | `Int` | Encoder transformer layer count |
| `decoderLayers` | `Int` | Decoder transformer layer count |
| `encoderDim` | `Int` | Encoder hidden dimension |
| `numHeads` | `Int` | Attention head count |
| `keyDim` | `Int` | Key dimension per head |
| `headDim` | `Int` | Value dimension per head |
| `encoderConvCacheLen` | `Int` | Convolutional cache length in frames |
| `topBufferLen` | `Int` | Cross-attention buffer length |
| `convDelay` | `Int` | Warmup frames before decoder starts decoding |
| `maxNspks` | `Int` | Max speaker slots in model output |
| `frameHz` | `Double` | Output frame rate (frames per second) |
| `targetSampleRate` | `Int` | Required audio sample rate |
| `stateShapes` | `LSEENDStateShapes` | Shapes for the six recurrent state tensors |
| `streamingLatencySeconds` | `Double` | Computed minimum startup latency |

Computed properties (`resolvedSampleRate`, `resolvedWinLength`, `resolvedHopLength`, `resolvedFFTSize`, `resolvedMelCount`, `resolvedContextRecp`, `resolvedSubsampling`) resolve optional fields to their defaults.

### LSEENDStateShapes

Tensor dimension arrays for the six recurrent state buffers. Read from metadata; used to allocate zero-initialized tensors at session start.

| Property | Type |
|----------|------|
| `encRetKv` | `[Int]` |
| `encRetScale` | `[Int]` |
| `encConvCache` | `[Int]` |
| `decRetKv` | `[Int]` |
| `decRetScale` | `[Int]` |
| `topBuffer` | `[Int]` |

---

## Evaluation API

These types support offline DER computation against RTTM ground truth. They are used by `LSEENDBenchmark` and can be used directly for custom evaluation pipelines.

### LSEENDRTTMEntry

A single speaker turn entry from an RTTM file.

```swift
let entry = LSEENDRTTMEntry(
    recordingID: "meeting_001",
    start: 12.5,       // seconds
    duration: 3.2,     // seconds
    speaker: "spk0"
)
```

| Property | Type |
|----------|------|
| `recordingID` | `String` |
| `start` | `Double` |
| `duration` | `Double` |
| `speaker` | `String` |

### LSEENDEvaluationSettings

Parameters for a DER evaluation run.

```swift
let settings = LSEENDEvaluationSettings(
    threshold: 0.5,       // Binarization threshold
    medianWidth: 1,       // Median filter kernel width (1 = disabled)
    collarSeconds: 0.25,  // Collar around speaker transitions to exclude from scoring
    frameRate: 10.0       // Frame rate in Hz
)
```

### LSEENDEvaluationResult

Detailed DER result including error breakdown and speaker mapping.

| Property | Type | Description |
|----------|------|-------------|
| `der` | `Double` | `(miss + falseAlarm + speakerError) / speakerScored` |
| `speakerScored` | `Double` | Reference-active frames scored (after collar exclusion) |
| `speakerMiss` | `Double` | Reference-active frames with no matching prediction |
| `speakerFalseAlarm` | `Double` | Predicted-active frames with no matching reference |
| `speakerError` | `Double` | Frames where both are active but mapped to different speakers |
| `threshold` | `Float` | Threshold used for binarization |
| `medianWidth` | `Int` | Median filter width applied |
| `collarSeconds` | `Double` | Collar used during scoring |
| `mappedBinary` | `LSEENDMatrix` | Binary predictions remapped to reference speaker order |
| `mappedProbabilities` | `LSEENDMatrix` | Continuous probabilities remapped to reference speaker order |
| `validMask` | `[Bool]` | Per-frame mask: `true` = included in scoring |
| `assignment` | `[Int: Int]` | Optimal speaker mapping `[refIndex: predIndex]` |
| `unmatchedPredictionIndices` | `[Int]` | Prediction columns with no reference match |

### LSEENDEvaluation

Static utility namespace for DER computation and RTTM I/O.

#### RTTM Parsing

```swift
// Parse an RTTM file
let (entries, speakers) = try LSEENDEvaluation.parseRTTM(url: rttmURL)
// entries: [LSEENDRTTMEntry]
// speakers: ordered [String] of unique speaker labels

// Convert to frame-level binary matrix
let referenceBinary: LSEENDMatrix = LSEENDEvaluation.rttmToFrameMatrix(
    entries: entries,
    speakers: speakers,
    numFrames: timeline.numFinalizedFrames,
    frameRate: 10.0
)
// Shape: [numFrames, speakers.count] — 1.0 where speaker is active
```

#### RTTM Writing

```swift
// Write binary prediction matrix to RTTM file
try LSEENDEvaluation.writeRTTM(
    recordingID: "meeting_001",
    binaryPrediction: binaryMatrix,    // [frames, speakers]
    outputURL: outputURL,
    frameRate: 10.0,
    speakerLabels: ["Alice", "Bob"]    // optional; defaults to "spk0", "spk1", ...
)
```

#### DER Computation

```swift
let result: LSEENDEvaluationResult = LSEENDEvaluation.computeDER(
    probabilities: probMatrix,         // [frames, predSpeakers] — continuous
    referenceBinary: referenceBinary,  // [frames, refSpeakers] — binary
    settings: settings
)
print("DER: \(result.der * 100)%")
print("Miss: \(result.speakerMiss / result.speakerScored * 100)%")
```

`computeDER` applies thresholding, median filtering, collar masking, and optimal Hungarian-style speaker assignment internally.

#### Lower-Level Primitives

```swift
// Binarize a probability matrix
let binary: LSEENDMatrix = LSEENDEvaluation.threshold(
    probabilities: probMatrix,
    value: 0.5    // strictly-greater-than
)

// Apply median filter along the time axis (1 or 0 = no-op)
let filtered: LSEENDMatrix = LSEENDEvaluation.medianFilter(binary: binary, width: 5)

// Compute collar validity mask
let mask: [Bool] = LSEENDEvaluation.collarMask(
    reference: referenceBinary,
    collarFrames: 3    // 0 = all frames valid
)
```

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
let engine = try LSEENDInferenceEngine(descriptor: descriptor)
let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

for chunk in chunkedAudio(samples, chunkSize: 800) {
    guard let update = try session.pushAudio(chunk) else { continue }
    // Committed frames: update.probabilities [newFrames × speakers]
    // Preview frames:   update.previewProbabilities [previewFrames × speakers]
}

let final = try session.finalize()
let result = session.snapshot()   // LSEENDInferenceResult
```

### Custom DER Evaluation

```swift
let engine = try LSEENDInferenceEngine(descriptor: descriptor)
let result = try engine.infer(audioFileURL: audioURL)

let (entries, speakers) = try LSEENDEvaluation.parseRTTM(url: rttmURL)
let reference = LSEENDEvaluation.rttmToFrameMatrix(
    entries: entries,
    speakers: speakers,
    numFrames: result.probabilities.rows,
    frameRate: result.frameHz
)

let evaluation = LSEENDEvaluation.computeDER(
    probabilities: result.probabilities,
    referenceBinary: reference,
    settings: LSEENDEvaluationSettings(
        threshold: 0.5,
        medianWidth: 1,
        collarSeconds: 0.25,
        frameRate: result.frameHz
    )
)
print("DER: \(String(format: "%.1f", evaluation.der * 100))%")
```

### Save Predictions as RTTM

```swift
let binaryPred = LSEENDEvaluation.threshold(
    probabilities: result.probabilities, value: 0.5)
try LSEENDEvaluation.writeRTTM(
    recordingID: "recording_001",
    binaryPrediction: binaryPred,
    outputURL: URL(fileURLWithPath: "output.rttm"),
    frameRate: result.frameHz
)
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
- [HuggingFace Models](https://huggingface.co/FluidInference/lseend-coreml)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [CALLHOME Corpus](https://catalog.ldc.upenn.edu/LDC97S42)
- [DIHARD Challenge](https://dihardchallenge.github.io/)
