# API Reference

This page summarizes the primary public APIs across modules. See inline doc comments for full details.

## Diarization

- `DiarizerManager`: Main diarization class
- `performCompleteDiarization(_:sampleRate:)`: Process audio and return speaker segments
  - Accepts any `RandomAccessCollection<Float>` (Array, ArraySlice, ContiguousArray, etc.)
- `compareSpeakers(audio1:audio2:)`: Compare similarity between two audio samples
- `validateAudio(_:)`: Validate audio quality and characteristics

## Voice Activity Detection

- `VadManager`: Voice activity detection with CoreML models
- `VadConfig`: Configuration for VAD processing with adaptive thresholding
- `processChunk(_:)`: Process a single audio chunk and detect voice activity
- `processAudioFile(_:)`: Process complete audio file in chunks
- `VadAudioProcessor`: Advanced audio processing with SNR filtering

## Automatic Speech Recognition

- `AsrManager`: Main ASR class with TDT decoding for batch processing
- `AsrModels`: Model loading and management with automatic downloads
- `ASRConfig`: Configuration for ASR processing
- `transcribe(_:source:)`: Process complete audio and return transcription results
- `AudioProcessor.loadAudioFile(path:)`: Load and convert audio files to required format
- `AudioSource`: Enum for microphone vs system audio separation

