# API Reference

This page summarizes the primary public APIs across modules. See inline doc comments and module-specific documentation for complete details.

## Common Patterns

**Audio Format:** All modules expect 16kHz mono Float32 audio samples. Use `FluidAudio.AudioConverter` to convert `AVAudioPCMBuffer` or files to 16kHz mono for both CLI and library paths.

**Model Registry:** Models auto-download from HuggingFace by default. Customize the registry URL using:
- `ModelRegistry.baseURL` (programmatic) - recommended for apps
- `REGISTRY_URL` or `MODEL_REGISTRY_URL` environment variables - recommended for CLI/testing
- Priority order: programmatic override → env vars → default (HuggingFace)

**Proxy Configuration:** If behind a corporate firewall, set the `https_proxy` (or `http_proxy`) environment variable. Both registry URL and proxy configuration are centralized in `ModelRegistry`.

**Error Handling:** All async methods throw descriptive errors. Use proper error handling in production code.

**Thread Safety:** All managers are thread-safe and can be used concurrently across different queues.

## Diarization

### DiarizerManager
Main class for speaker diarization and "who spoke when" analysis.

**Key Methods:**
- `performCompleteDiarization(_:sampleRate:) throws -> DiarizationResult`
  - Process complete audio file and return speaker segments
  - Parameters: `RandomAccessCollection<Float>` audio samples, sample rate (default: 16000)
  - Returns: `DiarizerResult` with speaker segments and timing
- `validateAudio(_:) throws -> AudioValidationResult`
  - Validate audio quality, length, and format requirements

**Configuration:**
- `DiarizerConfig`: Clustering threshold, minimum durations, activity thresholds
- Optimal threshold: 0.7 (17.7% DER on AMI dataset)

### OfflineDiarizerManager
Full batch pipeline that mirrors the pyannote/Core ML exporter (powerset segmentation + VBx clustering).

> Requires macOS 14 / iOS 17 or later because the manager relies on Swift Concurrency features and C++ clustering shims that are unavailable on older OS releases.

**Key Methods:**
- `init(config: OfflineDiarizerConfig = .default)`
  - Creates manager with configuration
- `prepareModels(directory:configuration:forceRedownload:) async throws`
  - Downloads / compiles the Core ML bundles as needed and records timing metadata. Call once before processing when you don't already have `OfflineDiarizerModels`.
- `initialize(models: OfflineDiarizerModels)`
  - Initializes with models containing segmentation, embedding, and PLDA components (useful when you hydrate the bundles yourself).
- `process(audio: [Float]) async throws -> DiarizationResult`
  - Runs the full 10 s window pipeline: segmentation → soft mask interpolation → embedding → VBx → timeline reconstruction.
- `process(audioSource: StreamingAudioSampleSource, audioLoadingSeconds: TimeInterval) async throws -> DiarizationResult`
  - Streams audio from disk-backed sources without materializing the entire buffer in memory. Pair with `StreamingAudioSourceFactory` for large meetings.

**Supporting Types:**
- `OfflineDiarizerConfig`
  - Mirrors pyannote `config.yaml` (`clusteringThreshold`, `Fa`, `Fb`, `maxVBxIterations`, `minDurationOn/off`, batch sizes, logging flags).
- `SegmentationRunner`
  - Batches 160 k-sample chunks through the segmentation model (589 frames per chunk).
- `Binarization`
  - Converts log probabilities to soft VAD weights while retaining binary masks for diagnostics.
- `WeightInterpolation`
  - Reimplements `scipy.ndimage.zoom` (half-pixel offsets) so 589-frame weights align with the embedding model’s pooling stride.
- `EmbeddingRunner`
  - Runs the FBANK frontend + embedding backend, resamples masks to 589 frames, and emits 256-d L2-normalized embeddings.
- `PLDAScoring` / `VBxClustering`
  - Apply the exported PLDA transforms and iterative VBx refinement to group embeddings into speakers.
- `TimelineReconstruction`
  - Derives timestamps directly from the segmentation frame count and `OfflineDiarizerConfig.windowDuration`, then enforces minimum gap/duration constraints.
- `StreamingAudioSourceFactory`
  - Creates disk-backed or in-memory `StreamingAudioSampleSource` instances so large meetings never require fully materialized `[Float]` buffers.

Use `OfflineDiarizerManager` when you need offline DER parity or want to run the new CLI offline mode (`fluidaudio process --mode offline`, `fluidaudio diarization-benchmark --mode offline`).

---

### Diarizer Protocol

`SortformerDiarizer` and `LSEENDDiarizer` both conform to the `Diarizer` protocol, which provides a unified streaming and offline API.

**Protocol Properties:**
- `isAvailable: Bool` — Whether the model is loaded and ready
- `numFramesProcessed: Int` — Confirmed frames processed so far
- `targetSampleRate: Int?` — Model's expected audio sample rate in Hz
- `modelFrameHz: Double?` — Output frame rate in Hz (frames per second)
- `numSpeakers: Int?` — Number of real speaker output tracks
- `timeline: DiarizerTimeline` — Accumulated diarization results

**Streaming:**
- `addAudio<C: Collection>(_ samples: C, sourceSampleRate: Double?) throws` — Buffer audio for processing; pass a non-nil `sourceSampleRate` to resample on the fly
- `process() throws -> DiarizerTimelineUpdate?` — Run inference on buffered audio; returns `nil` if not enough audio has accumulated
- `process<C: Collection>(samples: C, sourceSampleRate: Double?) throws -> DiarizerTimelineUpdate?` — Convenience combining `addAudio` + `process` in one call

**Offline:**
- `processComplete<C: Collection>(_ samples: C, sourceSampleRate:, keepingEnrolledSpeakers:, finalizeOnCompletion:, progressCallback:) throws -> DiarizerTimeline` — Process a complete audio buffer in one call
- `processComplete(audioFileURL: URL, keepingEnrolledSpeakers:, finalizeOnCompletion:, progressCallback:) throws -> DiarizerTimeline` — Read, resample, and process an audio file end-to-end

**Speaker Enrollment:**
- `enrollSpeaker<C: Collection>(withAudio samples: C, sourceSampleRate:, named:, overwritingAssignedSpeakerName:) throws -> DiarizerSpeaker?` — Feed audio of a known speaker before streaming begins; warms model state and labels that speaker's slot for subsequent `process()` calls

**Lifecycle:**
- `reset()` — Clear all streaming state (session, buffers, timeline) while keeping the model loaded
- `cleanup()` — Release all resources including the loaded model

---

### DiarizerTimeline

Holds accumulated streaming predictions and derived speaker segments. Returned by `Diarizer.timeline` and `processComplete(...)`.

**Key Properties:**
- `config: DiarizerTimelineConfig` — Post-processing configuration used to build segments
- `speakers: [Int: DiarizerSpeaker]` — Speaker slots keyed by output track index
- `finalizedPredictions: [Float]` — Flat `[frames × numSpeakers]` array of finalized per-frame probabilities
- `tentativePredictions: [Float]` — Same layout; frames still within the right-context window that may be revised
- `numFinalizedFrames: Int` — Count of finalized frames
- `numTentativeFrames: Int` — Count of tentative frames
- `finalizedDuration: Float` — Duration in seconds of finalized audio
- `hasSegments: Bool` — Whether any speaker has at least one segment

**Mutation:**
- `addChunk(_ chunk: DiarizerChunkResult) throws -> DiarizerTimelineUpdate` — Append new predictions and rebuild segments; called internally by the diarizer
- `rebuild(finalizedPredictions:tentativePredictions:keepingSpeakers:isComplete:) throws` — Replace all predictions from scratch (used by offline processing)
- `reset(keepingSpeakers:)` / `reset(keepingSpeakersWhere:)` — Clear segments and optionally preserve named speakers or speaker metadata
- `finalize()` — Promote all tentative segments to finalized

**`DiarizerTimelineConfig`** — Shared configuration used by both diarizers:
| Parameter | Default | Description |
|---|---|---|
| `numSpeakers` | model-specific | Number of speaker output tracks |
| `frameDurationSeconds` | model-specific | Duration of one output frame |
| `onsetThreshold` | 0.5 | Probability threshold to begin a speech segment |
| `offsetThreshold` | 0.5 | Probability threshold to end a speech segment |
| `onsetPadFrames` | 0 | Frames prepended to each segment onset |
| `offsetPadFrames` | 0 | Frames appended to each segment offset |
| `minFramesOn` | 0 | Minimum segment length; shorter segments are dropped |
| `minFramesOff` | 0 | Minimum gap; shorter silences are closed |
| `maxStoredFrames` | nil | Rolling window cap on finalized frames (nil = unlimited) |

---

### DiarizerSpeaker

Represents a single speaker track within a `DiarizerTimeline`.

**Key Properties:**
- `id: UUID` — Stable identity across resets
- `index: Int` — Slot index in the diarizer output (0-based)
- `name: String?` — Optional display name (set via enrollment or manually)
- `finalizedSegments: [DiarizerSegment]` — Confirmed speech segments
- `tentativeSegments: [DiarizerSegment]` — Speculative segments within the right-context window
- `hasSegments: Bool` — Whether any finalized or tentative segments exist
- `numSpeechFrames: Int` — Total frames spanned by all segments (finalized + tentative)
- `speechDuration: Float` — Total speech duration in seconds

**`DiarizerSegment`** — A single time-range for one speaker:
- `startFrame / endFrame: Int` — Frame indices (convert using `frameDurationSeconds`)
- `startTime / endTime: Float` — Seconds
- `duration: Float` — Segment length in seconds
- `isFinalized: Bool` — Whether the segment has been confirmed

---

### SortformerDiarizer

End-to-end streaming diarization using NVIDIA's Sortformer model. Tracks **4 fixed speaker slots**.

- **Sample rate:** 16 kHz
- **Frame duration:** 80 ms (12.5 Hz output)
- **Streaming latency:** ~0.64 s (`default` config) or ~1.04 s (`nvidiaLowLatency` configs)
- **Accuracy:** ~11% DER on DIHARD III (streaming), ~20.6% DER on AMI SDM (`nvidiaLowLatencyV2_1`)

**Initialization:**
```swift
// Preferred: download and compile model automatically
let diarizer = SortformerDiarizer(config: .default, timelineConfig: .sortformerDefault)
try await diarizer.initialize(mainModelPath: modelURL)

// Or with pre-loaded models
diarizer.initialize(models: sortformerModels)
```

**`SortformerConfig` Presets:**

| Preset | Latency | Notes |
|---|---|---|
| `.default` / `.fastestV2_1` | ~0.64 s | Gradient Descent conversion, fastest |
| `.nvidiaLowLatencyV2_1` | ~1.04 s | 20.6% DER on AMI SDM |
| `.nvidiaHighLatencyV2_1` | ~30.4 s | Highest accuracy, offline-style |

All streaming methods are defined by the `Diarizer` protocol above. Additionally:
- `state: SortformerStreamingState` — Live speaker cache and FIFO queue state (for diagnostics)
- `config: SortformerConfig` — The configuration this instance was created with

---

### LSEENDDiarizer

End-to-end streaming diarization using LS-EEND (Linear Streaming End-to-End Neural Diarization). Supports a **variable number of speaker slots** depending on the model variant.

- **Sample rate:** 8 kHz
- **Frame duration:** 100 ms (10 Hz output)
- **Variants:** `LSEENDVariant` (`LSEENDModelDescriptor.LSEENDVariant`)

**Initialization:**
```swift
// Auto-download from HuggingFace
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
try await diarizer.initialize(variant: .dihard3)

// Or with an explicit descriptor
let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: .dihard3)
try diarizer.initialize(descriptor: descriptor)
```

**LS-EEND–Specific Properties:**
- `computeUnits: MLComputeUnits` — CoreML compute target (`.cpuOnly` is typically fastest)
- `streamingLatencySeconds: Double?` — Minimum audio required before first output frame
- `decodeMaxSpeakers: Int?` — Total output slots including internal boundary tracks
- `timelineConfig: DiarizerTimelineConfig` — Active post-processing configuration

**Additional Method:**
- `finalizeSession() throws -> DiarizerChunkResult?` — Flush pending audio and finalize the timeline; call at end of a stream before reading the final timeline

**`LSEENDModelDescriptor`:**
- `LSEENDModelDescriptor.loadFromHuggingFace(variant:cacheDirectory:computeUnits:) async throws -> LSEENDModelDescriptor` — Download and cache all model files; returns a descriptor ready for `initialize(descriptor:)`
- `init(variant:modelURL:metadataURL:)` — Construct from local paths if already cached

All streaming and offline methods are defined by the `Diarizer` protocol above.

## Voice Activity Detection

### VadManager
Voice activity detection using the Silero VAD Core ML model with 256 ms unified inference and ANE optimizations.

**Key Methods:**
- `process(_ url: URL) async throws -> [VadResult]`
  - Process an audio file end-to-end. Automatically converts to 16kHz mono Float32 and processes in 4096-sample frames (256 ms).
- `process(_ buffer: AVAudioPCMBuffer) async throws -> [VadResult]`
  - Convert and process an in-memory buffer. Supports any input format; resampled to 16kHz mono internally.
- `process(_ samples: [Float]) async throws -> [VadResult]`
  - Process pre-converted 16kHz mono samples.
- `processChunk(_:inputState:) async throws -> VadResult`
  - Process a single 4096-sample frame (256 ms at 16 kHz) with optional recurrent state.

**Constants:**
- `VadManager.chunkSize = 4096`  // samples per frame (256 ms @ 16 kHz, plus 64-sample context managed internally)
- `VadManager.sampleRate = 16000`

**Configuration (`VadConfig`):**
- `defaultThreshold: Float` — Baseline decision threshold (0.0–1.0) used when segmentation does not override. Default: `0.85`.
- `debugMode: Bool` — Extra logging for benchmarking and troubleshooting. Default: `false`.
- `computeUnits: MLComputeUnits` — Core ML compute target. Default: `.cpuAndNeuralEngine`.

Recommended `defaultThreshold` ranges depend on your acoustic conditions:
- Clean speech: 0.7–0.9
- Noisy/mixed content: 0.3–0.6 (higher recall, more false positives)

**Performance:**
- Optimized for Apple Neural Engine (ANE) with aligned `MLMultiArray` buffers, silent-frame short-circuiting, and recurrent state reuse (hidden/cell/context) for sequential inference.
- Significantly improved throughput by processing 8×32 ms audio windows in a single Core ML call.

## Automatic Speech Recognition

### AsrManager
Automatic speech recognition using Parakeet TDT models (v2 English-only, v3 multilingual).

**Key Methods:**
- `transcribe(_:source:) async throws -> ASRResult`
  - Accepts `[Float]` samples already converted to 16 kHz mono; returns transcription text, confidence, and token timings.
- `transcribe(_ url: URL, source:) async throws -> ASRResult`
  - Loads the file directly and performs format conversion internally (`AudioConverter`).
- `transcribe(_ buffer: AVAudioPCMBuffer, source:) async throws -> ASRResult`
  - Convenience overload for capture pipelines that already produce PCM buffers.
- `initialize(models:) async throws`
  - Load and initialize ASR models (automatic download if needed)

**Model Management:**
- `AsrModels.downloadAndLoad(version: AsrModelVersion = .v3) async throws -> AsrModels`
  - Download models from HuggingFace and compile for CoreML
  - Pass `.v2` to load the English-only bundle when you do not need multilingual coverage
  - Models cached locally after first download
- `ASRConfig`: Beam size, temperature, language model weights

- **Audio Processing:**
- `AudioConverter.resampleAudioFile(path:) throws -> [Float]`
  - Load and convert audio files to 16kHz mono Float32 (WAV, M4A, MP3, FLAC)
- `AudioConverter.resampleBuffer(_ buffer: AVAudioPCMBuffer) throws -> [Float]`
  - Convert a buffer to 16kHz mono (stateless conversion)
- `AudioSource`: `.microphone` or `.system` for different processing paths

> **Warning:** Avoid hand-decoding audio payloads (e.g., truncating WAV headers or treating bytes as raw `Int16` samples).
> The Core ML models require correctly resampled 16 kHz mono Float32 tensors; manual parsing will silently corrupt input when
> formats carry metadata chunks, different bit depths, stereo channels, or compression. Always route files and live buffers
> through `AudioConverter` before calling `AsrManager.transcribe`.

**Performance:**
- Real-time factor: ~120x on M4 Pro (processes 1min audio in 0.5s)
- Languages: 25 European languages supported

### StreamingEouAsrManager
Real-time streaming ASR with End-of-Utterance detection using Parakeet EOU models.

**Key Methods:**
- `init(configuration:chunkSize:eouDebounceMs:debugFeatures:)`
  - Create manager with MLModel configuration and chunk size
  - `chunkSize`: `.ms160` (default), `.ms320`, or `.ms1600`
  - `eouDebounceMs`: Minimum silence duration before EOU triggers (default: 1280)
- `loadModels(modelDir:) async throws`
  - Load CoreML models from directory (encoder, decoder, joint, vocab)
- `process(audioBuffer:) async throws -> String`
  - Process audio incrementally, returns empty string (use `finish()` for transcript)
- `finish() async throws -> String`
  - Finalize processing and return accumulated transcript
- `reset() async`
  - Reset all state for next utterance
- `setEouCallback(_:)`
  - Set callback invoked when End-of-Utterance is detected
- `appendAudio(_:) throws`
  - Append audio to buffer without processing (for VAD integration)

**Properties:**
- `eouDetected: Bool` — Whether EOU was detected in the last chunk
- `eouDebounceMs: Int` — Minimum silence duration before EOU triggers
- `chunkSize: StreamingChunkSize` — Current chunk size configuration

**StreamingChunkSize:**
- `.ms160` — 160ms chunks, lowest latency, ~8% WER
- `.ms320` — 320ms chunks, balanced, ~5% WER
- `.ms1600` — 1600ms chunks, highest throughput

**Usage:**
```swift
let manager = StreamingEouAsrManager(chunkSize: .ms160, eouDebounceMs: 1280)
try await manager.loadModels(modelDir: modelsURL)

// Process audio incrementally
_ = try await manager.process(audioBuffer: buffer1)
_ = try await manager.process(audioBuffer: buffer2)

// Get final transcript
let transcript = try await manager.finish()

// Reset for next utterance
await manager.reset()
```

**Performance:**
- Real-time factor: ~5x RTF (160ms), ~12x RTF (320ms) on Apple Silicon
- WER: ~8% (160ms), ~5% (320ms) on LibriSpeech test-clean
