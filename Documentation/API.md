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
- `compareSpeakers(audio1:audio2:) throws -> Float`
  - Compare speaker similarity between two audio samples
  - Returns: Similarity score (0.0-1.0, higher = more similar)
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
