# Voice Activity Detection (VAD)

Fluid Audio ships the Silero VAD converted for Core ML together with Silero-style
timestamp extraction and streaming hysteresis. If you need help tuning the
parameters for your use case, reach out on Discord.

## Quick Start

Need chunk-level probabilities or state for custom pipelines? Call `process(_:)`
to inspect every 256 ms hop:

```swift
let results = try await manager.process(samples)
for (index, chunk) in results.enumerated() {
    print(
        String(
            format: "Chunk %02d: prob=%.3f, inference=%.4fs",
            index,
            chunk.probability,
            chunk.processingTime
        )
    )
}
```

## Manual Model Loading

Stage the Core ML bundle yourself when the runtime cannot reach HuggingFace.

### Required asset

- `silero-vad-unified-256ms-v6.0.0.mlmodelc`

The bundle lives in the `FluidInference/silero-vad-coreml` repo. Keep the folder name intact so `coremldata.bin` remains discoverable.

### Folder layout

```
/opt/models
└── silero-vad-coreml
    └── silero-vad-unified-256ms-v6.0.0.mlmodelc
        ├── coremldata.bin
        └── ...
```

Clone with Git LFS, download the archive from the HuggingFace UI, or copy from a machine that already initialized `VadManager()` (cache path: `~/Library/Application Support/FluidAudio/Models/silero-vad-coreml`).

### Loading without downloads

Supply the staged bundle to the `VadManager(config:vadModel:)` initializer:

```swift
import FluidAudio
import CoreML

Task {
    do {
        let modelURL = URL(fileURLWithPath: "/opt/models/silero-vad-coreml/silero-vad-unified-256ms-v6.0.0.mlmodelc", isDirectory: true)

        var configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let vadModel = try MLModel(contentsOf: modelURL, configuration: configuration)

        let manager = VadManager(config: .default, vadModel: vadModel)

        // Ready for segmenting or streaming without network downloads
    } catch {
        print("Failed to load VAD model: \(error)")
    }
}
```

Use `FileManager` to confirm the `.mlmodelc` directory exists before constructing the manager. When the bundle is present, no fallback download attempts occur.

## Offline Segmentation (Code)

`VadManager` can now emit ready-to-use speech intervals directly from PCM
samples. The segmentation logic mirrors the Silero reference implementation,
including minimum speech duration, silence padding, and max-duration splitting.

```swift
import FluidAudio

Task {
    let manager = try await VadManager(
        config: VadConfig(defaultThreshold: 0.75)
    )

    // Convert any supported file to 16 kHz mono Float32
    let audioURL = URL(fileURLWithPath: "path/to/audio.wav")
    let samples = try AudioConverter().resampleAudioFile(audioURL)

    // Tune segmentation behavior with VadSegmentationConfig
    var segmentation = VadSegmentationConfig.default
    segmentation.minSpeechDuration = 0.25
    segmentation.minSilenceDuration = 0.4
    segmentation.speechPadding = 0.12

    let segments = try await manager.segmentSpeech(samples, config: segmentation)
    for (index, segment) in segments.enumerated() {
        print(String(
            format: "Segment %02d: %.2f–%.2fs",
            index + 1,
            segment.startTime,
            segment.endTime
        ))
    }

    // Need audio chunks instead of timestamps?
    let clips = try await manager.segmentSpeechAudio(samples, config: segmentation)
    print("Extracted \(clips.count) buffered segments ready for ASR")
}
```

Need chunk-level probabilities for each 256 ms hop? Use `process(_:)` and inspect
`VadResult` directly:

```swift
let results = try await manager.process(samples)
for (index, chunk) in results.enumerated() {
    print(
        String(
            format: "Chunk %02d: prob=%.3f, inference=%.4fs",
            index,
            chunk.probability,
            chunk.processingTime
        )
    )
}
```

Key knobs in `VadSegmentationConfig`:
- `minSpeechDuration`: discard very short bursts.
- `minSilenceDuration`: silence length required to close a segment.
- `maxSpeechDuration`: automatically split long spans using the last detected silence (default 14 s).
- `speechPadding`: context added on both sides of each returned segment.
- `negativeThreshold`/`negativeThresholdOffset`: control hysteresis the same way as Silero's `threshold`/`neg_threshold`.

### Measuring Offline RTF

If you prefer to keep the per-chunk `VadResult` output, you can measure the
real-time factor (RTFx) of non-streaming runs by comparing total inference time
with the audio duration:

```swift
let results = try await manager.process(samples)
let totalInference = results.reduce(0.0) { $0 + $1.processingTime }
let audioSeconds = Double(samples.count) / Double(VadManager.sampleRate)
let rtf = audioSeconds / totalInference
print(String(format: "VAD RTFx: %.1f", rtf))
```

`VadResult.processingTime` is reported per 4096-sample chunk, so summing across
the array yields the full pass latency.

## Streaming API

For streaming workloads you control the chunk size and maintain a
`VadStreamState`. Each call emits at most one `VadStreamEvent` describing a
speech start or end boundary, along with the raw probability for the chunk.

```swift
import FluidAudio

Task {
    let manager = try await VadManager()
    var state = await manager.makeStreamState()

    for chunk in microphoneChunks { // chunk length ~256 ms at 16 kHz
        let result = try await manager.processStreamingChunk(
            chunk,
            state: state,
            config: .default,
            returnSeconds: true,
            timeResolution: 2
        )

        state = result.state

        // Access the raw VAD probability (0.0-1.0) for this chunk
        print(String(format: "Probability: %.3f", result.probability))

        if let event = result.event {
            switch event.kind {
            case .speechStart:
                print("Speech began at \(event.time ?? 0) s")
            case .speechEnd:
                print("Speech ended at \(event.time ?? 0) s")
            }
        }
    }
}
```

The `VadStreamResult` contains:
- `state`: Updated state to pass to the next chunk
- `event`: Optional speech start/end event (only emitted at boundaries)
- `probability`: Raw VAD probability (0.0-1.0) for the current chunk

Notes:
- Stream chunks do not need to be exactly 4096 samples; choose what matches your input cadence.
- Call `makeStreamState()` whenever you reset your audio stream (equivalent to Silero's `reset_states`).
- When requesting seconds (`returnSeconds: true`), timestamps are rounded using `timeResolution` decimal places.
- Use `probability` for custom thresholding logic or confidence tracking alongside the built-in hysteresis.

## CLI

Start with the general-purpose `process` command, which runs the diarization
pipeline (and therefore VAD) end-to-end on a single file:

```bash
swift run fluidaudio process path/to/audio.wav
```

Once you need to experiment with the VAD-specific heuristics directly, use the
CLI commands below:

```bash
# Inspect offline segments (default mode is offline only)
swift run fluidaudio vad-analyze path/to/audio.wav

# Streaming only, 128 ms chunks, tighter silence rules (timestamps are emitted in seconds)
swift run fluidaudio vad-analyze path/to/audio.wav --streaming --min-silence-ms 300

# Run both offline + streaming in one pass
swift run fluidaudio vad-analyze path/to/audio.wav --mode both

# Classic benchmark tooling remains available
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3
```

`swift run fluidaudio vad-analyze --help` prints the full list of tuning
options, including negative-threshold overrides and max-duration splitting.
Offline runs emit an RTFx summary calculated from per-chunk inference time. Use
`--mode both` if you also want to see streaming start/end events in the same run.

Datasets for benchmarking can be fetched with `swift run fluidaudio download --dataset vad`.
