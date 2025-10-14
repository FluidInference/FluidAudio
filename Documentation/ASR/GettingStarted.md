# Automatic Speech Recognition (ASR) / Transcription

- Model (multilingual): `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- Model (English-only): `FluidInference/parakeet-tdt-0.6b-v2-coreml`
- Languages: v3 spans 25 European languages; v2 focuses on English accuracy
- Processing Mode: Batch transcription for complete audio files
- Real-time Factor: ~120x on M4 Pro (1 minute ≈ 0.5 seconds)
- Streaming Support: Stabilized streaming with VAD gating (see [Stabilized Streaming](StabilizedStreaming.md))

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

### Streaming (stabilized + VAD-gated)

```swift
import AVFoundation
import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
func startStreaming() async throws {
    // 1) Download and load the streaming-capable ASR models
    let models = try await AsrModels.downloadAndLoad()
    // 2) Use the preset tuned for balanced latency and stability
    let streamingAsr = StreamingAsrManager(config: .streaming)
    try await streamingAsr.start(models: models, source: .microphone)

    // 3) Capture audio and push buffers into the streaming pipeline
    let engine = AVAudioEngine()
    let inputNode = engine.inputNode
    let format = inputNode.outputFormat(forBus: 0)
    let bufferSize: AVAudioFrameCount = 4096

    inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: format) { buffer, _ in
        Task(priority: .userInitiated) {
            await streamingAsr.streamAudio(buffer)
        }
    }

    engine.prepare()
    try engine.start()

    // 4) Handle volatile + confirmed updates in real time
    Task.detached {
        for await update in await streamingAsr.transcriptionUpdates {
            if update.isConfirmed {
                print("✅ \(update.text)")
            } else {
                print("… \(update.text)")
            }
        }
    }
}
```

The `.streaming` preset enables the built-in Silero VAD so silence is trimmed before decoding. Read [Stabilized Streaming](StabilizedStreaming.md) for configuration knobs, stabilization internals, and debugging tips.

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

# Streaming transcription with stabilized output and VAD gating
swift run fluidaudio transcribe audio.wav --streaming --stabilize-profile balanced
```
