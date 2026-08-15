# LS-EEND Streaming Speaker Diarization

## Overview

LS-EEND (Long-Form Streaming End-to-End Neural Diarization) answers "who spoke when" in real-time. A causal Conformer encoder with a retention mechanism feeds an online attractor decoder that tracks speaker identities frame by frame, without separate VAD, segmentation, or clustering.

**Key specs:**
- 4–10 simultaneous speakers depending on variant (see below)
- ~100ms frame resolution (10 Hz output) at the default step size
- Handles recordings up to one hour
- 8000 Hz input sample rate (automatic resampling)
- Chunk-in-chunk-out streaming; each CoreML call emits one step of committed frames
- CoreML-optimized for Apple Silicon (CPU-only is fastest for this model)

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

---

## Step Size

`LSEENDStepSize` selects how many output frames the model commits per CoreML call. Smaller steps reduce latency at the cost of more CoreML invocations; larger steps amortize per-call overhead and raise throughput. Each step size is a separately-converted CoreML bundle on HuggingFace, organized under `100ms/`, `200ms/`, … `500ms/` folders.

| Case | Frames per call | Latency floor |
|------|-----------------|---------------|
| `.step100ms` *(default)* | 1 | ~100 ms |
| `.step200ms` | 2 | ~200 ms |
| `.step300ms` | 3 | ~300 ms |
| `.step400ms` | 4 | ~400 ms |
| `.step500ms` | 5 | ~500 ms |

The `LSEENDStepSize` enum is `Int`-backed (raw values 1–5) and `CaseIterable`.

---

## Call Flow

### Complete Audio

```
LSEENDDiarizer.processComplete(samples, sourceSampleRate:)
  |
  |-- resetStreamingState()                  zero framesFedToModel, reset feeder, clear finalized flag
  |-- timeline.reset(keepingSpeakers:)
  |-- addAudio(samples, sourceSampleRate:)
  |     |-- session.enqueueAudio(samples, withSampleRate:)   resample if needed, push into raw audio queue,
  |                                                          eagerly run STFT -> log10-mel -> CMN
  |
  |-- flush(finalizeOnCompletion: true, progressCallback:)
  |     |
  |     |-- session.drainRightContextWithSilence()    pad tail with silence so the last chunk completes
  |     |-- flush(recordFrames: true, progressCallback:)
  |     |     |
  |     |     |-- FOR EACH ready chunk:
  |     |     |     |-- session.emitNextChunk() -> LSEENDInput
  |     |     |     |     (loads mel features + decoderMask into the persistent input,
  |     |     |     |      derives warmupFrames from decoderMask zeros)
  |     |     |     |-- input.warmupFrames = max(min(convDelay - framesFedToModel, chunkSize), 0)
  |     |     |     |-- model.predict(from: input)
  |     |     |     |     |-- CoreML model.prediction(from: input)
  |     |     |     |     |-- read probs + 6 next-state tensors
  |     |     |     |     |     (enc_kv_new, enc_scale_new, enc_conv_cache_new,
  |     |     |     |     |      cnn_window_new, dec_kv_new, dec_scale_new)
  |     |     |     |     |-- swap state tensors into input.state in-place
  |     |     |     |     |-- strip warmup rows, return flat [Float] (frames * maxSpeakers)
  |     |     |     |-- append to newPreds; framesFedToModel += chunkSize; progressCallback
  |     |     |
  |     |     |-- timeline.addPredictions(finalizedPredictions: newPreds, tentativePredictions: [])
  |     |
  |     |-- timeline.finalize()              only when finalizeOnCompletion
  |
  |-- return DiarizerTimeline
```

`processComplete(audioFileURL:)` is identical, except the first step is `session.enqueueAudioFile(at: url)` instead of `enqueueAudio(samples, withSampleRate:)`.

### Streaming

```
LSEENDDiarizer.addAudio(samples, sourceSampleRate:)
  |
  |-- session.enqueueAudio(samples, withSampleRate:)   resample if needed, push into raw audio queue,
                                                       eagerly run STFT -> log10-mel -> CMN

LSEENDDiarizer.process()
  |
  |-- flush(recordFrames: true, progressCallback: nil)
  |     |
  |     |-- FOR EACH ready chunk in session.melQueue:
  |     |     |-- session.emitNextChunk() -> LSEENDInput
  |     |     |-- compute warmupFrames from convDelay - framesFedToModel
  |     |     |-- model.predict(from: input)
  |     |     |     (CoreML forward + in-place state swap, returns probs for chunkSize - warmup frames)
  |     |     |-- append to newPreds; framesFedToModel += chunkSize
  |     |
  |     |-- timeline.addPredictions(finalizedPredictions: newPreds, tentativePredictions: [])
  |
  |-- return DiarizerTimelineUpdate?
```

```
LSEENDDiarizer.finalizeSession()
  |
  |-- session.drainRightContextWithSilence()   pad tail with silence so the final chunk completes
  |-- process()                                 one last flush pass
  |-- timeline.finalize()
  |-- return DiarizerTimelineUpdate?
```

Notes on the current streaming pipeline:
- One CoreML call commits `chunkSize` model frames (minus warmup). There is no separate per-frame ingest/decode split.
- Recurrent state lives on `LSEENDInput.state` and is swapped in place by `LSEENDModel.predict`. There is no external state-copy / speculative-preview branch — `tentativePredictions` is always passed empty, so `DiarizerTimelineUpdate.tentativeSegments` is currently unused by LS-EEND.
- The model already emits sigmoided probabilities (`probs`), and there are no boundary tracks to crop — output speaker count equals `maxSpeakers`.

---

## File Structure

```
Sources/FluidAudio/Diarizer/LS-EEND/
├── LSEENDDiarizer.swift       # LSEENDDiarizer (Diarizer protocol impl)
├── LSEENDInference.swift      # LSEENDModel, LSEENDInput
├── LSEENDPreprocessor.swift   # LSEENDFeatureProvider, StreamingChunkQueue
└── LSEENDTypes.swift          # LSEENDMetadata, LSEENDState, LSEENDError, variant typealiases
```

DER computation lives in the shared `Sources/FluidAudio/Diarizer/DiarizationDER.swift` (not LS-EEND-specific). RTTM parsing for benchmarks lives in `Sources/FluidAudioCLI/Utils/RTTMParser.swift`.

---

## LSEENDDiarizer

The primary entry point. Implements the `Diarizer` protocol — the same API as `SortformerDiarizer`.

### Initialization

Three constructors, picked based on whether you want the model downloaded inline:

```swift
// Async convenience: downloads (or reuses cached) model and initializes in one step.
let diarizer = try await LSEENDDiarizer(
    variant: .dihard3,
    stepSize: .step100ms,
    timelineConfig: nil                 // optional DiarizerTimelineConfig
)

// Lazy: build empty, then `initialize(...)` later.
let diarizer = LSEENDDiarizer(timelineConfig: nil)
try await diarizer.initialize(variant: .ami)

// Direct: caller already owns an LSEENDModel.
let model = try await LSEENDModel.loadFromHuggingFace(variant: .callhome)
let diarizer = try LSEENDDiarizer(model: model)
```

Post-processing knobs (onset/offset thresholds, padding, minimum-on/off durations) now flow through `DiarizerTimelineConfig`, not initializer parameters:

```swift
let config = DiarizerTimelineConfig(onsetThreshold: 0.4, onsetPadFrames: 1)
let diarizer = try await LSEENDDiarizer(variant: .dihard3, timelineConfig: config)
```

### Loading Models

```swift
// Async download (or cache hit) + initialize in place.
try await diarizer.initialize(
    variant: .callhome,
    stepSize: .step100ms,
    cacheDirectory: nil,                // optional; defaults to ~/Library/Application Support/FluidAudio/Models
    computeUnits: .cpuOnly,
    progressHandler: { progress in /* ... */ }
)

// Or load the model yourself and hand it in.
let model = try await LSEENDModel.loadFromHuggingFace(variant: .callhome)
try diarizer.initialize(model: model)
```

### Offline Processing

```swift
// From a file URL (resamples to the model's target sample rate automatically).
let timeline = try diarizer.processComplete(audioFileURL: audioURL)

// From raw samples (specify the source sample rate if it is not 8 kHz).
let timeline = try diarizer.processComplete(
    samples,
    sourceSampleRate: 16_000,
    keepingEnrolledSpeakers: nil,        // Bool? — nil means "keep if no segments yet"
    finalizeOnCompletion: true,
    progressCallback: { processed, total, chunks in
        print("\(processed)/\(total) samples (\(chunks) chunks)")
    }
)
```

### Streaming

`sourceSampleRate` is optional — only required when the audio is not already at the model's target rate.

```swift
// Push audio incrementally.
try diarizer.addAudio(audioChunk, sourceSampleRate: 16_000)
if let update = try diarizer.process() {
    for segment in update.finalizedSegments { /* ... */ }
}

// Convenience: add + process in one call.
if let update = try diarizer.process(samples: audioChunk, sourceSampleRate: 16_000) {
    /* ... */
}

// Flush remaining frames at end of stream.
try diarizer.finalizeSession()
let finalTimeline = diarizer.timeline
```

Notes:
- `addAudio` and `process` both `throw`. `process()` returns `nil` when there are not yet enough samples buffered to emit a chunk.
- `finalizeSession()` pads the remaining audio with silence so the final partial chunk can complete, then runs one last `process()` pass and finalizes the timeline. It is `@discardableResult`.
- The LS-EEND streaming path does not currently populate `update.tentativeSegments` — all emitted frames are committed.

### Speaker Enrollment

Enrollment warms LS-EEND with a known speaker before the live stream starts. It keeps the active streaming session, resets the visible timeline back to frame 0, and stores the speaker name on the chosen slot inside the `DiarizerTimeline`.

```swift
let speaker = try diarizer.enrollSpeaker(
    withAudio: enrollmentAudio,
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

Real-world integration testing with 4-speaker audio reveals specific enrollment weaknesses compared to Sortformer. The current `enrollSpeaker` implementation works by playing the enrollment audio through the live session, then picking the slot that accumulated the most speech activity during that audio — preferring slots that did not exist before the enrollment audio was played:

**Bounded probabilities:** LS-EEND emits sigmoid-of-cosine outputs, so raw per-slot probabilities saturate roughly between `sigmoid(-1)` and `sigmoid(1)` (~0.2–0.8). They will not reach the 0.9+ confidence levels external post-processing might suggest. The in-library enrollment path does not surface those scores directly; it works off cumulative speech activity per slot during the enrollment audio.

**Close-voice slot rejection:** When the enrollment audio is close enough to an already-enrolled speaker that no new slot opens during enrollment, the most-active slot will be an existing one. With `overwritingAssignedSpeakerName: false`, that causes a hard failure (snapshot rollback, `nil` return). With the default `true`, the existing slot gets renamed instead — usable, but not the "fresh registration" the caller likely wanted. In a 4-speaker test, 3 speakers enrolled cleanly; the 4th failed because no new slot emerged.

**Root cause:** LS-EEND is an end-to-end model, so there is no API for per-slot similarity outputs or explicit slot-lock assignment. If you need a custom enrollment scheme (per-slot scoring, Hungarian assignment, etc.) you have to build it directly on `LSEENDModel.predict` + `LSEENDFeatureProvider`. Suppressing existing attractors may be a path forward, but this has not been validated.

**Training data gap:** Sortformer was trained on a large volume of real-world data, giving it stronger generalization for speaker identity. LS-EEND was trained primarily on simulated data and then fine-tuned on real data — the base model without fine-tuning performs poorly.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `timeline` | `DiarizerTimeline` | Accumulated finalized results |
| `isAvailable` | `Bool` | Whether the model is loaded |
| `numFramesProcessed` | `Int` | Total committed frames processed |
| `targetSampleRate` | `Int?` | Expected input sample rate (8000) |
| `modelFrameHz` | `Double?` | Output frame rate (~10.0 Hz) |
| `numSpeakers` | `Int?` | Model output slot count (`maxSpeakers`) |

### Lifecycle

```swift
try diarizer.finalizeSession()   // Flush trailing context before reading final output
diarizer.reset()                 // Reset streaming state for a new audio stream (keeps model loaded)
diarizer.cleanup()               // Release all resources including the loaded model
```

---

## LSEENDModel

Lower-level model wrapper. Use this when you need direct access to per-chunk predictions, want to manage the feature pipeline yourself, or are building tooling around the model.

### Creating the Model

```swift
// From HuggingFace (cached after first call).
let model = try await LSEENDModel.loadFromHuggingFace(
    variant: .dihard3,                 // default
    stepSize: .step100ms,              // default
    cacheDirectory: nil,               // optional; defaults to ~/Library/Application Support/FluidAudio/Models
    computeUnits: .cpuOnly,            // .cpuOnly is fastest for this model
    progressHandler: { progress in /* ... */ }
)

// Or load directly from a local mlmodelc.
let model = try LSEENDModel(modelURL: localURL, computeUnits: .cpuOnly)
```

`LSEENDMetadata` is decoded from the `config` value in the CoreML model's `creatorDefinedKey` user metadata — there is no separate JSON metadata file.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `metadata` | `LSEENDMetadata` | Decoded model configuration |

### Inference

```swift
let probs = try model.predict(from: input)
// probs is flat [Float], row-major, shape (chunkSize - input.warmupFrames) * metadata.maxSpeakers,
// with sigmoid already applied.
```

`predict` is internally serialized (an `NSLock` guards the CoreML call) and updates `input.state` in place via output backings.

---

## LSEENDInput

`MLFeatureProvider` that carries the per-chunk inputs and recurrent state for one `LSEENDModel.predict` call.

```swift
let input = try LSEENDInput(from: model.metadata)        // fresh state
let input = try LSEENDInput(from: model.metadata, state: existingState)  // resume from snapshot
```

| Property | Type | Description |
|----------|------|-------------|
| `state` | `LSEENDState` | Six recurrent-state tensors (`~Copyable`). Updated in place by `predict`. |
| `melFeatures` | `MLMultiArray` | `[1, melFrames, nMels]`, fed to the model's `features` input |
| `decoderMask` | `MLMultiArray` | `[chunkSize]`, fed to the model's `valid_mask` input |
| `warmupFrames` | `Int` | Number of leading output frames to strip (warmup region inside `decoderMask`) |
| `featureNames` | `Set<String>` | The MLFeatureProvider feature set this input exposes |

```swift
// Reload features into the same input (no allocation on the hot path).
try input.loadInputs(
    melFeatures: newMelBuffer,           // any AccelerateBuffer of Float
    decoderMask: newDecoderMaskBuffer,
    warmupFrames: nil                    // nil => count zeros in newDecoderMaskBuffer
)

input.resetState()                       // zero all recurrent state tensors
```

---

## LSEENDFeatureProvider

Streaming feature pipeline that fronts `LSEENDModel`. Handles resampling, STFT, log-mel, cumulative mean normalization, context splicing, and chunk emission. This is the public surface that `LSEENDDiarizer.processComplete` and `process()` consume internally — pull it out only if you need a custom integration.

> **Not thread-safe.** The internal lock guards `readyChunks` only; all mutation paths assume a single caller.

```swift
let feeder = try LSEENDFeatureProvider(from: model.metadata)

// Push raw audio. eagerPreprocessing: true (default) runs STFT/log-mel/CMN immediately.
try feeder.enqueueAudio(samples, withSampleRate: 16_000)

// Or push from a file (returns sample count read).
let count = try feeder.enqueueAudioFile(at: audioURL)

// Pad the tail before the final predict pass.
try feeder.drainRightContextWithSilence()

// Pull ready chunks until none remain.
while let input = try feeder.emitNextChunk() {
    let probs = try model.predict(from: input)
    // probs has shape (chunkSize - input.warmupFrames) * metadata.maxSpeakers
}
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `enqueueAudio(_:withSampleRate:eagerPreprocessing:)` | `Void` | Push raw audio, optionally trigger feature extraction immediately |
| `enqueueAudioFile(at:)` | `Int` (`@discardableResult`) | Push audio from a file; returns sample count read |
| `drainRightContextWithSilence(flush:)` | `Void` | Pad the tail with silence so the final chunk completes |
| `emitNextChunk()` | `LSEENDInput?` | Load the next ready chunk's mel + mask into the persistent input, or `nil` |
| `takeSnapshot()` | `Snapshot` (`~Copyable`) | Snapshot for rollback (e.g. during enrollment) |
| `rollback(to:keepingState:)` | `Void` | Restore from a snapshot |
| `reset()` | `Void` | Clear queues and reset state |

| Property | Type | Description |
|----------|------|-------------|
| `readyChunks` | `Int` | Number of chunks currently ready for `emitNextChunk` |

---

## Data Types

### LSEENDMetadata (`Codable`)

Decoded from the CoreML model's `config` user metadata. Read via `model.metadata`.

| Property | Type | Description |
|----------|------|-------------|
| `chunkSize` | `Int` | Output frames committed per CoreML call (matches `LSEENDStepSize.rawValue`) |
| `frameDurationSeconds` | `Float` | Seconds per output frame |
| `maxSpeakers` | `Int` | Output speaker slot count |
| `sampleRate` | `Int` | Required audio sample rate |
| `maxNspks` | `Int` | Number of attractors (not output speakers) |
| `hopLength` | `Int` | Mel hop length |
| `winLength` | `Int` | Mel window length |
| `nMels` | `Int` | Number of mel bands |
| `contextSize` | `Int` | Mel context frames per output frame |
| `subsampling` | `Int` | Mel → prediction subsampling factor |
| `convDelay` | `Int` | Right-context output frames |
| `nUnits`, `nHeads`, `encNLayers`, `decNLayers`, `convKernelSize` | `Int` | Model architecture parameters |
| `headDim` *(computed)* | `Int` | `nUnits / nHeads` |
| `melFrames` *(computed)* | `Int` | `(chunkSize - 1) * subsampling + 2 * contextSize + 1` |
| `nFFT` *(computed)* | `Int` | Smallest power of two `>= winLength` |

### LSEENDState (`~Copyable`)

Holds the six recurrent-state tensors threaded through every CoreML call:

| Field | Shape (read from metadata) |
|-------|----------------------------|
| `encRetKv` | `[encNLayers, 1, nHeads, headDim, headDim]` |
| `encRetScale` | `[encNLayers, 1]` |
| `encConvCache` | `[encNLayers, 1, convKernelSize, nUnits]` |
| `cnnWindow` | `[1, nUnits, 2 * convDelay]` |
| `decRetKv` | `[decNLayers, maxNspks, nHeads, headDim, headDim]` |
| `decRetScale` | `[decNLayers, 1]` |

Constructors: `init(encRetKv:encRetScale:encConvCache:cnnWindow:decRetKv:decRetScale:)` for explicit tensors, or `init(from metadata:)` to allocate ANE-aligned zero buffers. Methods: `copy() throws -> LSEENDState`, `copy(to dst:)`, `reset()`.

### LSEENDFeatureProvider.Snapshot (`~Copyable`)

Opaque snapshot of feature-provider state for `takeSnapshot()` / `rollback(to:keepingState:)`.

### StreamingChunkQueue

Ring-buffer used internally by `LSEENDFeatureProvider` for both the raw audio queue and the mel queue. Public so advanced callers can compose their own pipelines; most users will not need it.

---

## Feature Extraction

Feature extraction is handled by `LSEENDFeatureProvider`. `LSEENDDiarizer` consumes it transparently; the type is public so callers can wire their own STFT/CMN/queue plumbing on top of `LSEENDModel`.

---

## Model Loading

### LSEENDVariant

```swift
public typealias LSEENDVariant = ModelNames.LSEEND.Variant

LSEENDVariant.ami        // "ls_eend_ami"
LSEENDVariant.callhome   // "ls_eend_ch"
LSEENDVariant.dihard2    // "ls_eend_dih2"
LSEENDVariant.dihard3    // "ls_eend_dih3"
```

| Member | Description |
|--------|-------------|
| `repo` | `ModelNames.Repo` the variant lives in |
| `name` | Internal model name (e.g. `"ls_eend_dih3"`) |
| `description` | Same as `name` (`CustomStringConvertible`) |
| `name(forStep:)` | `"<name>_<step>"`, e.g. `"ls_eend_dih3_100ms"` |
| `fileName(forStep:)` | `"<step>/<name>_<step>.mlmodelc"` |

### LSEENDStepSize

```swift
public typealias LSEENDStepSize = ModelNames.LSEEND.StepSize

LSEENDStepSize.step100ms   // rawValue 1, "100ms"
LSEENDStepSize.step200ms   // rawValue 2, "200ms"
LSEENDStepSize.step300ms   // rawValue 3, "300ms"
LSEENDStepSize.step400ms   // rawValue 4, "400ms"
LSEENDStepSize.step500ms   // rawValue 5, "500ms"
```

### LSEENDModel.loadFromHuggingFace

```swift
public static func loadFromHuggingFace(
    variant: LSEENDVariant = .dihard3,
    stepSize: LSEENDStepSize = .step100ms,
    cacheDirectory: URL? = nil,
    computeUnits: MLComputeUnits = .cpuOnly,
    progressHandler: ProgressHandler? = nil
) async throws -> LSEENDModel
```

Downloads only the mlmodelc for the requested `(variant, stepSize)` pair. Cached under `cacheDirectory` (defaults to `~/Library/Application Support/FluidAudio/Models`).

---

## Evaluation

`LS-EEND` has no dedicated evaluation module. DER computation lives in the shared `Sources/FluidAudio/Diarizer/DiarizationDER.swift`:

```swift
let result = DiarizationDER.compute(
    ref: referenceSegments,            // [DERSpeakerSegment]
    hyp: hypothesisSegments,           // [DERSpeakerSegment]
    frameStep: 0.01,
    collar: 0.0
)
// result.der, result.confusion, result.falseAlarm, result.miss, result.mapping
```

RTTM parsing for benchmark pipelines lives in `Sources/FluidAudioCLI/Utils/RTTMParser.swift`.

---

## Error Handling

All LS-EEND errors are thrown as `LSEENDError` and conform to `LocalizedError`.

| Case | Thrown when |
|------|-------------|
| `.initializationFailed(String)` | Model load failed, or decoding `config` from CoreML metadata failed |
| `.inferenceFailed(String)` | CoreML forward pass failed, or an expected output tensor was missing |
| `.invalidInputSize(String)` | `LSEENDInput.loadInputs` rejected a mismatched buffer size |
| `.notInitialized` | Diarizer method called before a model is loaded |

```swift
do {
    let timeline = try diarizer.processComplete(audioFileURL: url)
} catch let error as LSEENDError {
    switch error {
    case .notInitialized: print("Call initialize(variant:) before processComplete")
    case .initializationFailed(let message): print("Model problem: \(message)")
    case .inferenceFailed(let message): print("Inference problem: \(message)")
    case .invalidInputSize(let message): print("Input problem: \(message)")
    }
}
```

---

## Usage Examples

### Offline File Processing

```swift
let diarizer = try await LSEENDDiarizer(variant: .ami)

let timeline = try diarizer.processComplete(audioFileURL: URL(fileURLWithPath: "meeting.wav"))
for speaker in timeline.speakers.values {
    for segment in speaker.finalizedSegments {
        print("\(segment.speakerLabel): \(segment.startTime)s – \(segment.endTime)s")
    }
}
```

### Streaming from Microphone

```swift
let diarizer = try await LSEENDDiarizer(variant: .dihard3)

audioEngine.installTap(onBus: 0, bufferSize: 1600, format: format) { buffer, _ in
    let samples = Array(UnsafeBufferPointer(
        start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    do {
        try diarizer.addAudio(samples)
        if let update = try diarizer.process() {
            DispatchQueue.main.async { updateUI(diarizer.timeline) }
        }
    } catch {
        print("LS-EEND streaming error: \(error)")
    }
}
```

### Low-Level Model + Feature Provider

```swift
let model = try await LSEENDModel.loadFromHuggingFace(variant: .callhome)
let feeder = try LSEENDFeatureProvider(from: model.metadata)

for chunk in chunkedAudio(samples, chunkSize: 800) {
    try feeder.enqueueAudio(chunk, withSampleRate: 8_000)
    while let input = try feeder.emitNextChunk() {
        let probs = try model.predict(from: input)
        // probs: [Float] row-major, frames * model.metadata.maxSpeakers, sigmoid applied
    }
}

// Flush the tail.
try feeder.drainRightContextWithSilence()
while let input = try feeder.emitNextChunk() {
    let probs = try model.predict(from: input)
    _ = probs
}
```

---

## CLI

```bash
# Diarize a single file (default variant: ami, default step size: 500ms)
swift run fluidaudiocli lseend audio.wav
swift run fluidaudiocli lseend audio.wav --variant callhome
swift run fluidaudiocli lseend audio.wav --variant dihard3 --step-size 100ms --output result.json

# Benchmark on AMI (downloads dataset automatically)
swift run fluidaudiocli lseend-benchmark --auto-download --variant ami
swift run fluidaudiocli lseend-benchmark --variant callhome --threshold 0.35 --collar 0.25
swift run fluidaudiocli lseend-benchmark --variant dihard3 --output results.json --max-files 10
```

### `lseend` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `ami` | `ami` \| `callhome` \| `dihard2` \| `dihard3` |
| `--step-size` | `500ms` | `100ms` \| `200ms` \| `300ms` \| `400ms` \| `500ms` |
| `--threshold` | `0.5` | Seeds onset + offset thresholds |
| `--onset` | `0.5` | Overrides `--threshold` for onset only |
| `--offset` | `0.5` | Overrides `--threshold` for offset only |
| `--pad-onset` | — | Padding before speech segments (seconds) |
| `--pad-offset` | — | Padding after speech segments (seconds) |
| `--min-duration-on` | — | Minimum speech segment duration (seconds) |
| `--min-duration-off` | — | Minimum silence duration (seconds) |
| `--output` | — | Path to save JSON results |

### `lseend-benchmark` flags

Accepts all of the `lseend` flags above, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `ami` | `ami` \| `voxconverse` \| `callhome` |
| `--ami-split` | `test` | `dev` \| `test` \| `train` (only with `--dataset ami`) |
| `--single-file` | — | Process a specific meeting (e.g. `ES2004a`) |
| `--max-files` | — | Limit number of files processed |
| `--median-width` | `1` | Median filter width in frames (1 = disabled) |
| `--collar` | `0.0` (AMI) / `0.25` (other) | Collar around transitions (seconds) |
| `--progress` | `.lseend_progress.json` | Resume state path |
| `--resume` | off | Resume from previous progress file |
| `--verbose` | off | Print per-meeting debug output |
| `--auto-download` | off | Auto-download AMI dataset if missing |

---

## Model Files on HuggingFace

Hosted at [FluidInference/lseend-coreml](https://huggingface.co/FluidInference/lseend-coreml). Each `(variant, stepSize)` pair is a separate mlmodelc; metadata is embedded in the CoreML model under the `config` user-metadata key (no separate JSON file). Files are downloaded automatically on first use and cached at `~/Library/Application Support/FluidAudio/Models/`.

The on-disk layout is `<step>/ls_eend_<short>_<step>.mlmodelc`, where `<short>` is `ami`, `ch`, `dih2`, or `dih3`. Example for step `100ms`:

| Variant | File |
|---------|------|
| `.ami` | `100ms/ls_eend_ami_100ms.mlmodelc` |
| `.callhome` | `100ms/ls_eend_ch_100ms.mlmodelc` |
| `.dihard2` | `100ms/ls_eend_dih2_100ms.mlmodelc` |
| `.dihard3` | `100ms/ls_eend_dih3_100ms.mlmodelc` |

For other step sizes, replace `100ms` in both the folder and filename with the desired step (`200ms`, `300ms`, `400ms`, or `500ms`). The CLI and `LSEENDModel.loadFromHuggingFace` will download the requested file on first use.

---

## References

- [LS-EEND Paper (arXiv 2410.06670)](https://arxiv.org/abs/2410.06670) — Di Liang, Xiaofei Li. *LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction.* IEEE TASLP.
- [LS-EEND GitHub Repository](https://github.com/Audio-WestlakeU/FS-EEND)
- [HuggingFace Models](https://huggingface.co/FluidInference/lseend-coreml)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [CALLHOME Corpus](https://catalog.ldc.upenn.edu/LDC97S42)
- [DIHARD Challenge](https://dihardchallenge.github.io/)
