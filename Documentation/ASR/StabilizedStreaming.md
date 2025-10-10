# Stabilized Streaming ASR

FluidAudio's streaming automatic speech recognition (ASR) stack delivers low-latency transcripts without the noisy rewinds that typically happen when partial hypotheses are revised. This guide walks through the moving pieces so you can tune or extend the pipeline confidently.

## Why stabilization matters

Large decoder windows provide accurate transcripts, but force the UI to wait for several seconds of context. Small windows respond faster but rewrite words as new context arrives. The stabilized streaming pipeline maintains two tiers:

- **Volatile updates** surface immediately so the UI can show draft text as the decoder produces new tokens.
- **Confirmed updates** commit once enough agreement accumulates and the stabilizer decides the text is stable.

Subscribers receive both tracks via `StreamingTranscriptionUpdate`, so product UIs can show real-time drafts and promote them to confirmed text without jarring rewinds.
The stabilizer is always active inside `StreamingAsrManager`; stream consumers tune its behavior rather than disabling it.

## Pipeline overview

1. **Audio ingestion** – `StreamingAsrManager` accepts `AVAudioPCMBuffer` frames from any source (microphone, file, custom stream) and resamples everything to 16 kHz mono.
2. **VAD gating** – `StreamingVadPipeline` runs the Core ML Silero VAD model (downloaded automatically) to drop silence and pre-speech padding before decoding.
3. **Window assembly** – `StreamingWindowProcessor` stitches left / main / right contexts so the decoder sees a consistent timeline.
4. **Decoder step** – `AsrManager.transcribeStreamingChunk` produces token IDs plus token-level timings for each window.
5. **Stabilization layer** – `StreamingStabilizerSink` compares successive hypotheses, promotes stable prefixes, and maintains the two-tier (`volatile` + `confirmed`) transcripts.

The components are all cooperative actors, so work stays thread-safe without `@unchecked Sendable`.

## Working with transcription updates

`StreamingAsrManager` exposes an async sequence that yields `StreamingTranscriptionUpdate` values. Each update includes:

- `text`: current transcript chunk.
- `isConfirmed`: `true` when the stabilizer has promoted this text to the confirmed tier.
- `confidence`: normalized decoder confidence (0.0 – 1.0).
- `tokenTimings`: arrays with per-token timing, useful for karaoke-style highlighting.

Volatile text should be rendered with "draft" styling; confirmed text can be appended to the final transcript. You can always retrieve the most recent strings via `streamingAsr.volatileTranscript` and `streamingAsr.confirmedTranscript`.

## Voice activity detection integration

Turning on VAD gating is as simple as using the `.streaming` preset. Under the hood, FluidAudio:

- Boots a `VadManager` instance with the default segmentation policy.
- Buffers a short pre-speech window so word beginnings are not clipped.
- Emits post-speech padding (`speechPadding`) to avoid truncating trailing phonemes.

If you already have a `VadManager` instance (shared across sessions), inject it via `StreamingAsrManager(config: customConfig, vadManager: existingManager)` so downloads and warm-up happen once.

### Supplying your own VAD manager

When you want to load the Core ML bundle from a pre-downloaded location (e.g. shipped inside your app bundle), create the `VadManager` yourself and pass it into the streaming manager:

```swift
let modelsURL = Bundle.main.resourceURL!.appendingPathComponent("Models/VAD", isDirectory: true)
let vadManager = try await VadManager(
    config: .default,
    modelDirectory: modelsURL  // Avoids network downloads; uses your staged model files
)

let streamingAsr = StreamingAsrManager(
    config: .streaming,
    vadManager: vadManager
)
try await streamingAsr.start(models: asrModels, source: .microphone)
```

The injected manager is reused across sessions and `StreamingAsrManager` will skip any automatic downloads. This is the recommended approach when you manage model assets yourself or need to share a single VAD instance between multiple concurrent streams.

## Configuration knobs

`StreamingAsrConfig` bundles the main tuning options:

- `chunkSeconds` – primary streaming window size used for decoding and stabilization.
- `leftContextSeconds` & `rightContextSeconds` – overlap passed to the decoder.
- `stabilizer` – a `StreamingStabilizerConfig` (or preset) that sets the agreement window, word-boundary policy, and max wait budget.
- `vad` – a `StreamingVadConfig` to enable/disable gating and tweak segmentation thresholds.

```swift
var config = StreamingAsrConfig.streaming
config = config.withStabilizer(
    StreamingStabilizerConfig
        .preset(.highStability)
        .withMaxWaitMilliseconds(900)
        .withDebugDumpEnabled(true)  // Writes per-window JSONL traces
)
config = config.withVad(
    StreamingVadConfig(
        isEnabled: true,
        vadConfig: .init(threshold: 0.30),
        segmentationConfig: VadSegmentationConfig(
            minSpeechDuration: 0.4,
            speechPadding: 0.25
        )
    )
)
let streamingAsr = StreamingAsrManager(config: config)
```

When adjusting latencies, remember that the stabilizer waits for enough consensus before committing tokens. Tweaking `chunkSeconds`, `leftContextSeconds`, and the stabilizer window lets you balance responsiveness against stability.

## CLI usage and further reading

- The `swift run fluidaudio transcribe --streaming` command uses the same stabilized pipeline and VAD gating. Use `--stabilize-profile` to switch between `balanced`, `low-latency`, and `high-stability` presets. See the [CLI guide](../CLI.md#asr) for invocation examples.
- The README's [ASR Quick Start](../../README.md#asr-quick-start) includes end-to-end Swift sample code for microphone streaming.
- Dive into the implementation in `Sources/FluidAudio/ASR/Streaming/` (notably `StreamingAsrManager.swift`, `StreamingVadPipeline.swift`, and `StreamingStabilizerSink.swift`) to explore internals or contribute enhancements.
