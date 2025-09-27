# Automatic Speech Recognition (ASR) / Transcription

- Model (multilingual): `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- Model (English-only): `FluidInference/parakeet-tdt-0.6b-v2-coreml`
- Languages: v3 spans 25 European languages; v2 focuses on English accuracy
- Processing Mode: Batch transcription for complete audio files
- Real-time Factor: ~120x on M4 Pro (1 minute ≈ 0.5 seconds)
- Streaming Support: Coming soon — batch processing recommended for production use

## Choosing a model version

- Prefer **v2** when you only need English. It reuses the fused TDT decoder from v3 but ships with a tighter vocabulary, delivering better recall on long-form English audio.
- Use **v3** for multilingual coverage (25 languages). English accuracy is still strong, but the broader vocab slightly trails v2 on rare words.
- Both versions share the same API surface—set `AsrModelVersion` in code or pass `--model-version` in the CLI.

```swift
// Download the English-only bundle when you only need English transcripts
let models = try await AsrModels.downloadAndLoad(version: .v2)
```

## Quick Start (Code)

```swift
import FluidAudio

// Batch transcription from an audio file
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad(version: .v3)  // Switch to .v2 for English-only
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

## Manual model loading

Working offline? Follow the [Manual Model Loading guide](ManualModelLoading.md) to stage the CoreML bundles and call `AsrModels.load` without triggering HuggingFace downloads.

## CLI

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# English-only run (better recall)
swift run fluidaudio transcribe audio.wav --model-version v2

# Transcribe multiple files in parallel
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudio asr-benchmark --subset test-clean --max-files 50

# Run the English-only benchmark
swift run fluidaudio asr-benchmark --subset test-clean --max-files 50 --model-version v2

# Multilingual ASR (FLEURS) benchmark
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

# Download LibriSpeech test sets
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
```
