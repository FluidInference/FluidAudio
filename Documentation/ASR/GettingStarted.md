# Automatic Speech Recognition (ASR) / Transcription

- Model: `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- Languages: 25 European languages (see model card)
- Processing Modes:
  - Batch (direct): fastest end-to-end for full files
  - Batch (VAD-based): uses VAD segmentation to skip silence and reduce compute
  - Streaming API: real-time via `StreamingAsrManager` (CLI provides simulated streaming)
- Real-time Factor: ~120x on M4 Pro (1 minute ≈ 0.5 seconds)

## Quick Start (Code)

```swift
import FluidAudio

// Batch transcription from an audio file (direct)
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad()
    let asrManager = AsrManager(config: .default)
    try await asrManager.initialize(models: models)

    // 2) Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")

    // 3) Transcribe the audio
    let result = try await asrManager.transcribe(samples, source: .system)
    print("Transcription: \(result.text)")
    print("Confidence: \(result.confidence)")
}
```

### Batch with VAD-Based Chunking

Use when your file has significant silence. Speech is first segmented by VAD, short segments are merged (up to `minSegmentDuration`), and long ones are split to fit the model window.

```swift
import FluidAudio

Task {
    let models = try await AsrModels.downloadAndLoad()

    // Enable VAD-presegmented ASR
    let asrConfig = ASRConfig(
        useVadBasedChunking: true,
        vadSegmentationConfig: .default,  // or customize
        minSegmentDuration: 5.0           // merge shorts up to 5s
    )

    let asr = AsrManager(config: asrConfig)
    try await asr.initialize(models: models)

    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")
    let result = try await asr.transcribe(samples, source: .system)
    print(result.text)
}
```

Notes:
- VAD-based chunking is batch-only (not used by the streaming API).
- Segments > ~15s are split with small overlaps to respect model limits.

## CLI

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# Transcribe with VAD-based chunking (batch)
swift run fluidaudio transcribe audio.wav --vad-chunking

# Transcribe multiple files in parallel
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudio asr-benchmark --subset test-clean --num-files 50

# Multilingual ASR (FLEURS) benchmark
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

# Simulated streaming (shows incremental updates)
swift run fluidaudio transcribe audio.wav --streaming --metadata

# Download LibriSpeech test sets
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
```
