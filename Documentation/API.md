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
Voice activity detection using Silero VAD models.

**Key Methods:**
- `processChunk(_:) throws -> VadResult`
  - Process single 512-sample chunk (32ms at 16kHz)
  - Returns: Voice activity probability and boolean decision
- `processAudioFile(_:) throws -> [VadResult]`
  - Process complete audio file in 512-sample chunks
  - Returns: Array of VAD results for each frame

**Configuration:**
- `VadConfig`: Threshold (0.0-1.0), window size, post-processing
- `VadAudioProcessor`: SNR filtering, noise reduction, adaptive thresholding
- Recommended threshold: 0.3-0.5 depending on noise conditions

## Automatic Speech Recognition

### AsrManager
Automatic speech recognition using Parakeet TDT v3 models.

**Key Methods:**
- `transcribe(_:source:) throws -> AsrTranscription`
  - Process complete audio and return transcription
  - Parameters: `RandomAccessCollection<Float>` samples, `AudioSource` (microphone/system)
  - Returns: `AsrTranscription` with text, confidence, and timing
- `initialize(models:) async throws`
  - Load and initialize ASR models (automatic download if needed)

**Model Management:**
- `AsrModels.downloadAndLoad() async throws -> AsrModels`
  - Download models from HuggingFace and compile for CoreML
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
