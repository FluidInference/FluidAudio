# Automatic Speech Recognition (ASR) / Transcription

## Batch Transcription

- Model (multilingual): `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- Model (English-only): `FluidInference/parakeet-tdt-0.6b-v2-coreml`
- Languages: v3 spans 25 European languages; v2 focuses on English accuracy
- Processing Mode: Batch transcription for complete audio files
- Real-time Factor: ~120x on M4 Pro (1 minute ≈ 0.5 seconds)

## Streaming ASR (Parakeet EOU)

- Model: `FluidInference/parakeet-eou-1.1b-coreml`
- Chunk Sizes: 160ms (lowest latency), 320ms, 1600ms (highest throughput)
- End-of-Utterance Detection: Built-in silence detection with configurable debounce

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

> **Important:** Do not parse WAV/PCM bytes by hand (e.g., slicing headers or assuming 16-bit samples).
> Always convert with `AudioConverter` so differing bit depths, channel layouts, metadata chunks,
> or compressed formats (MP3/M4A/FLAC) get normalized to the 16 kHz mono Float32 tensors that Parakeet expects.
> Manually decoded buffers frequently contain garbage values, which shows up as empty transcripts even though the models load successfully.

### Transcribing directly from a file URL

If you already have an audio file on disk you can skip manual sample loading—`AsrManager.transcribe(_ url:source:)`
handles format conversion internally via `AudioConverter`.

```swift
let models = try await AsrModels.downloadAndLoad(version: .v3)
let asrManager = AsrManager()
try await asrManager.initialize(models: models)

let audioURL = URL(fileURLWithPath: "/path/to/audio.wav")
let result = try await asrManager.transcribe(audioURL, source: .system)
print(result.text)
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

## Streaming CLI (Parakeet EOU)

```bash
# Transcribe a file (--use-cache auto-downloads models)
swift run fluidaudio parakeet-eou --input audio.wav --use-cache

# Run benchmark on LibriSpeech test-clean
swift run fluidaudio parakeet-eou --benchmark --chunk-size 160 --max-files 100 --use-cache
```

**Options:** `--input <path>`, `--benchmark`, `--max-files <n>`, `--chunk-size <160|320|1600>`, `--eou-debounce <ms>`, `--use-cache`, `--models <path>`, `--output <path>`, `--verbose`
