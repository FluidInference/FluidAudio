# API Reference

This page summarizes the primary public APIs across modules. See inline doc comments and module-specific documentation for complete details.

## Common Patterns

**Audio Format:** All modules expect 16kHz mono Float32 audio samples. Use `FluidAudio.AudioConverter` to convert `AVAudioPCMBuffer` or files to 16kHz mono for both CLI and library paths.

**Model Loading:** Models auto-download from HuggingFace on first use. Set `https_proxy` environment variable if behind corporate firewall.

**Error Handling:** All async methods throw descriptive errors. Use proper error handling in production code.

**Thread Safety:** All managers are thread-safe and can be used concurrently across different queues.

## Diarization

### DiarizerManager
Main class for speaker diarization and "who spoke when" analysis.

**Key Methods:**
- `performCompleteDiarization(_:sampleRate:) throws -> DiarizerResult`
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
- `threshold: Float` — Decision threshold (0.0–1.0). Default: `0.85`.
- `debugMode: Bool` — Extra logging for benchmarking and troubleshooting. Default: `false`.
- `computeUnits: MLComputeUnits` — Core ML compute target. Default: `.cpuAndNeuralEngine`.

Recommended threshold ranges depend on your acoustic conditions:
- Clean speech: 0.7–0.9
- Noisy/mixed content: 0.3–0.6 (higher recall, more false positives)

**Performance:**
- Optimized for Apple Neural Engine (ANE) with aligned `MLMultiArray` buffers, silent-frame short-circuiting, and recurrent state reuse (hidden/cell/context) for sequential inference.
- Significantly improved throughput by processing 8×32 ms audio windows in a single Core ML call.

## Automatic Speech Recognition

### AsrManager
Automatic speech recognition using Parakeet TDT models (v2 English-only, v3 multilingual).

**Key Methods:**
- `transcribe(_:source:) throws -> AsrTranscription`
  - Process complete audio and return transcription
  - Parameters: `RandomAccessCollection<Float>` samples, `AudioSource` (microphone/system)
  - Returns: `AsrTranscription` with text, confidence, and timing
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

**Performance:**
- Real-time factor: ~120x on M4 Pro (processes 1min audio in 0.5s)
- Languages: 25 European languages supported
- Streaming: Available via `StreamingAsrManager` (beta)
