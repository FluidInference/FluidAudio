# API Reference

Primary public APIs for FluidAudio components. See inline doc comments for complete details.

**Components:**
- [Common Patterns](#common-patterns)
- [Diarization](#diarization)
- [Voice Activity Detection](#voice-activity-detection)
- [Automatic Speech Recognition](#automatic-speech-recognition)
- [Text-to-Speech](#text-to-speech)

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

`SortformerDiarizer` and `LSEENDDiarizer` both conform to the `Diarizer` protocol, providing a unified streaming and offline API.

**Streaming:** `addAudio(_:sourceSampleRate:)` → `process()` → read `timeline`. Convenience `process(samples:sourceSampleRate:)` combines both steps. Returns `DiarizerTimelineUpdate?` (`nil` when not enough audio has accumulated).

**Offline:** `processComplete(_:sourceSampleRate:...)` or `processComplete(audioFileURL:...)` to process a full recording in one call.

**Speaker Enrollment:** `enrollSpeaker(withAudio:sourceSampleRate:named:...)` feeds known-speaker audio before streaming to label a slot.

**Lifecycle:** `finalizeSession()` flushes trailing context so the last true frame becomes finalized. `reset()` clears streaming state but keeps the model loaded. `cleanup()` releases everything.

---

### DiarizerTimeline & DiarizerSpeaker

`DiarizerTimeline` accumulates per-frame speaker probabilities and derives `DiarizerSpeaker` segments. Each speaker has `finalizedSegments` (confirmed) and `tentativeSegments` (may be revised). Segments expose `startTime`, `endTime`, `duration`, and `isFinalized`.

**`DiarizerTimelineConfig`** controls post-processing (onset/offset thresholds default to 0.5, min segment/gap duration, optional rolling window cap). Both diarizers accept this at init.

**Speaker Management:**
- `upsertSpeaker(named:atIndex:) -> DiarizerSpeaker?`
  - Add a speaker to a slot, or update the existing speaker's name if that slot is already occupied
  - If `atIndex` is `nil`, the first unused diarizer slot is chosen
- `upsertSpeaker(_:atIndex:transferCurrentSegment:) -> DiarizerSpeaker?`
  - Insert an existing `DiarizerSpeaker` into a slot, replacing any speaker already assigned there
  - If `atIndex` is `nil`, the first unused diarizer slot is chosen
  - `transferCurrentSegment` moves the in-progress segment (if one exists) to the new speaker before continuing
- `removeSpeaker(atIndex:clearCurrentSegment:) -> DiarizerSpeaker?`
  - Remove the speaker assigned to a diarizer output slot and return the removed speaker if present
  - `clearCurrentSegment` resets the in-progress speaking state for that slot before continuing
- `speakers: [Int: DiarizerSpeaker]`
  - Read or replace the full slot-to-speaker mapping directly when needed

---

### SortformerDiarizer

Streaming diarization using NVIDIA's Sortformer. 4 fixed speaker slots, 16 kHz input, 80 ms frame duration.

```swift
let diarizer = SortformerDiarizer(config: .default, timelineConfig: .sortformerDefault)
try await diarizer.initialize(mainModelPath: modelURL)
```

**Config presets:** `.default` / `.fastV2_1` (1.04 s latency), `.balancedV2_1` (1.04 s, 20.6% DER on AMI SDM), `.highContextV2_1` (30.4 s latency). v2 variants also available.

---

### LSEENDDiarizer

Streaming diarization using LS-EEND. Variable speaker slots, 8 kHz input, 100 ms frame duration, 20.7% DER on AMI SDM.

```swift
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
try await diarizer.initialize(variant: .dihard3)
```

**Variants:** ami, callhome, dihard2, dihard3 (via `LSEENDModelDescriptor.loadFromHuggingFace(variant:)`).

Call `finalizeSession()` at end-of-stream to flush pending audio before reading the final timeline.

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

### SlidingWindowAsrManager
Real-time sliding window ASR with overlap and cancellation support.

**Key Methods:**
- `init(models:config:) async throws`
  - Initialize with ASR models and configuration
- `transcribeChunk(_:isLastChunk:) async throws -> ASRResult`
  - Process audio chunk with sliding window overlap
  - Returns accumulated transcript with proper handling of chunk boundaries
- `reset()`
  - Reset internal state for new session

**Configuration:**
- Default chunk size: ~14.96 seconds
- Default overlap: 2.0 seconds
- Supports cancellation via Task cancellation

**Usage:**
```swift
let manager = try await SlidingWindowAsrManager()

for audioChunk in audioStream {
    let result = try await manager.transcribeChunk(
        audioChunk,
        isLastChunk: false
    )
    print("Partial: \(result.text)")
}

// Process final chunk
let final = try await manager.transcribeChunk(lastChunk, isLastChunk: true)
print("Final: \(final.text)")
```

### StreamingNemotronAsrManager
NVIDIA Nemotron streaming ASR with encoder cache for low-latency processing.

**Key Methods:**
- `init(chunkSize:configuration:) async throws`
  - Initialize with chunk size (160ms, 320ms, or 1600ms)
- `loadModels(modelDir:) async throws`
  - Load CoreML models from directory
- `transcribe(_:) async throws -> String`
  - Process audio and return transcript
- `reset() async`
  - Reset encoder cache and decoder state

**Chunk Sizes:**
- `.ms160` — 160ms chunks, lowest latency
- `.ms320` — 320ms chunks, balanced
- `.ms1600` — 1600ms chunks, highest throughput

**Performance:**
- Real-time factor: ~0.2x on Apple Silicon
- Maintains encoder cache across chunks for efficiency

### Qwen3AsrManager
Qwen3-based speech recognition with Whisper mel spectrogram frontend.

**Key Methods:**
- `init(modelDir:configuration:) async throws`
  - Initialize with model directory and CoreML configuration
- `transcribe(_:) async throws -> String`
  - Transcribe audio samples (16kHz mono Float32)
- `transcribe(_:) async throws -> String`
  - Transcribe from audio file URL

**Features:**
- Whisper-style mel spectrogram processing
- Multi-language support
- Experimental high-accuracy model

## Text-to-Speech (TTS)

### KokoroTtsManager
Text-to-speech synthesis using Kokoro CoreML models.

**Key Methods:**
- `init(defaultVoice:defaultSpeakerId:directory:computeUnits:customLexicon:)`
  - Create TTS manager with optional configuration
  - `computeUnits`: Use `.cpuAndGPU` on iOS 26+ to avoid ANE issues
- `initialize(preloadVoices:) async throws`
  - Download and initialize TTS models
  - Optionally preload specific voices
- `synthesize(text:voice:speakerId:speed:pitch:) async throws -> [Float]`
  - Synthesize speech from text
  - Returns audio samples at 24kHz
- `synthesizeDetailed(text:voice:speakerId:speed:pitch:) async throws -> (audio: [Float], alignments: [AlignmentInfo])`
  - Synthesize with phoneme-level timing information
- `synthesizeToFile(text:outputURL:voice:speakerId:speed:pitch:) async throws`
  - Synthesize directly to WAV file
- `setDefaultVoice(_:speakerId:) async throws`
  - Change default voice for subsequent synthesis
- `setCustomLexicon(_:)`
  - Set custom pronunciation dictionary
- `cleanup()`
  - Release models and free memory

**Available Voices:**
- `af` — American Female
- `af_bella`, `af_nicole`, `af_sarah` — American Female variants
- `am` — American Male
- `am_adam`, `am_michael` — American Male variants
- `bf` — British Female
- `bm` — British Male

**Configuration:**
- `defaultVoice`: Voice identifier (default: `"af"`)
- `defaultSpeakerId`: Speaker ID for multi-speaker voices (default: 0)
- `speed`: Speech rate multiplier (0.5–2.0, default: 1.0)
- `pitch`: Pitch shift in semitones (-12 to +12, default: 0)
- `customLexicon`: Custom pronunciation dictionary

**Usage:**
```swift
let manager = KokoroTtsManager(defaultVoice: "af")
try await manager.initialize()

let audio = try await manager.synthesize(
    text: "Hello from FluidAudio!",
    speed: 1.0,
    pitch: 0
)

// Save to file
try await manager.synthesizeToFile(
    text: "Hello world",
    outputURL: URL(fileURLWithPath: "output.wav")
)
```

**Performance:**
- Real-time factor: ~5-10x on Apple Silicon
- Output sample rate: 24kHz
- Supports SSML for prosody control

### PocketTtsManager
Lightweight streaming TTS with voice cloning support.

**Key Methods:**
- `init(directory:computeUnits:) async throws`
  - Initialize with optional directory and compute units
- `synthesize(text:voice:speakerId:) async throws -> [Float]`
  - Synthesize speech from text
  - Returns audio samples at 24kHz
- `synthesizeStreaming(text:voice:speakerId:) -> AsyncThrowingStream<[Float], Error>`
  - Stream audio chunks as they are generated
- `cloneVoice(from:name:) async throws -> String`
  - Clone voice from reference audio
  - Returns voice ID for later use
- `cleanup()`
  - Release models and free memory

**Features:**
- Streaming synthesis for long text
- Voice cloning from short audio samples
- Lower memory footprint than Kokoro
- Faster synthesis for real-time applications

**Usage:**
```swift
let manager = try await PocketTtsManager()

// Basic synthesis
let audio = try await manager.synthesize(text: "Hello world")

// Streaming synthesis
for try await chunk in manager.synthesizeStreaming(text: longText) {
    // Play audio chunk immediately
    playAudio(chunk)
}
```
