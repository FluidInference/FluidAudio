# Speaker Diarization

Real-time speaker diarization for iOS and macOS, answering "who spoke when" in audio streams.

## Model Choice

Pick the diarizer based on the workflow:

**LS-EEND** handles noisy environments and overlapping speakers well and supports up to 10 speakers. It is also much more lightweight than Sortformer, able to run entirely on an M4 MAX CPU at a comparable speed to Sortformer on ANE. However, it is more prone to false alarms and usually less stable than Sortformer, unless many speakers are talking simultaneously. Streaming updates occur every 100ms with about 900ms of tentative predictions. 

**Sortformer** handles noisy environments and overlapping speakers well, but is limited to 4 speakers. Speaker identities are extremely stable, but it struggles when many loud voices are present. It also misses quiet speech as it's trained to ignore background conversations. The most common source of error is missed speech. Streaming updates occur every 480ms, with 560ms of tentative predictions.

**DiarizerManager** Legacy online diarizer. Struggles with background noise, background conversations, overlapping speech with more than 2 speakers, short utterances, and similar-sounding speakers. The most common source of error is incorrect labeling. Performs poorly for low-latency streaming. Requires external timestamp alignment between chunks if operating on a sliding window. This is also the most computationally heavy online diarizer.

**Offline VBx pipeline** is the best offline-quality option when you want a full batch pipeline with segmentation, embedding extraction, and clustering over a complete file.

| Environment | Sortformer | DiarizerManager | LS-EEND | 
|-------------|:----------:|:---------------:|:-------:|
| Clean/silent room | Best | Good | Best |
| Background noise | Best | Poor | Good |
| Speech from another room | Poor | Good | Good |
| Whispers | Poor | Good | Best |
| High overlap | Good | Poor | Best |
| Max speakers | 4 | No max | 10 |
| Benchmarks | Good | Poor | Best |
| Remembering speakers across meetings | Great | Best | Good |
| Pre-enrolled speaker mapping | Best | Good | Weak |

### Speaker Enrollment: Sortformer vs LS-EEND

For workflows that pre-enroll known speakers before live audio, Sortformer is the stronger choice:

- **Sortformer** auto-maps all speakers with high confidence, even when two voices are similar. It benefits from training on a large volume of real-world data and uses past context effectively through its speaker cache.
- **LS-EEND** can fail enrollment when voices are too similar ("too close to existing speaker" collision). Its scores are bounded to roughly 0.2–0.8 due to the internal sigmoid-over-cosine architecture. An external score-extraction + global assignment fallback avoids hard rejection but produces weaker mappings.
- **LS-EEND** is an end-to-end model, which makes it difficult to force speaker registration into a specific slot. There is no API for per-slot similarity outputs or explicit slot-lock assignment.

LS-EEND was trained primarily on simulated data (fine-tuned on real data), while Sortformer was trained on predominantly real-world data. This training data difference is the main reason for the enrollment accuracy gap.

See [LS-EEND.md](LS-EEND.md#enrollment-limitations-integration-feedback) and [Sortformer.md](Sortformer.md#enrollment-strengths-integration-feedback) for details.

## Quick Start

```swift
import FluidAudio

// 1. Download models (one-time setup)
let models = try await DiarizerModels.downloadIfNeeded()

// 2. Initialize with default config
let diarizer = DiarizerManager()
diarizer.initialize(models: models)

// 3. Normalize any audio file to 16kHz mono Float32 using AudioConverter
let converter = AudioConverter()
let url = URL(fileURLWithPath: "path/to/audio.wav")
let audioSamples = try converter.resampleAudioFile(url)

// 4. Run diarization (accepts any RandomAccessCollection<Float>)
let result = try diarizer.performCompleteDiarization(audioSamples)

// Alternative: Use ArraySlice for zero-copy processing
let audioSlice = audioSamples[1000..<5000]  // No memory copy!
let sliceResult = try diarizer.performCompleteDiarization(audioSlice)

// 5. Get results
for segment in result.segments {
    print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
}
```

## Source Layout

The diarizer module mirrors the three-stage pipeline and its offline counterpart. Files live under `Sources/FluidAudio/Diarizer/`:

```
Core/
├── DiarizerManager.swift        # Real-time orchestrator and chunk scheduler
├── DiarizerTypes.swift          # Public models/configs shared across stages
└── DiarizerModels.swift         # Core ML bundle management

Segmentation/
├── SegmentationProcessor.swift  # VAD + powerset segmentation inference
├── SlidingWindow.swift          # Frame windowing helpers
└── AudioValidation.swift        # Streaming quality/embedding validation

Extraction/
└── EmbeddingExtractor.swift     # WeSpeaker embedding inference

Clustering/
├── SpeakerManager.swift         # Active speaker tracking and assignment
├── SpeakerTypes.swift           # Speaker/raw embedding representations
└── SpeakerOperations.swift      # Distance/scoring utilities

Offline/
├── Core/
│   ├── OfflineDiarizerManager.swift
│   ├── OfflineDiarizerTypes.swift
│   └── OfflineDiarizerModels.swift
├── Segmentation/
│   └── OfflineSegmentationProcessor.swift
├── Extraction/
│   ├── OfflineEmbeddingExtractor.swift
│   ├── PLDATransform.swift
│   └── WeightInterpolation.swift
├── Clustering/
│   ├── AHCClustering.swift
│   └── VBxClustering.swift
└── Utils/
    ├── OfflineReconstruction.swift
    └── VDSPOperations.swift
```

Use this layout as the reference when adding new diarization capabilities so orchestration, segmentation, embedding extraction, and clustering stay isolated.

## Manual Model Loading

If you deploy in an offline environment, stage the Core ML bundles manually and skip the automatic HuggingFace downloader.

### Required assets

Download the two `.mlmodelc` folders from `FluidInference/speaker-diarization-coreml`:

- `pyannote_segmentation.mlmodelc`
- `wespeaker_v2.mlmodelc`

Keep the folder names unchanged so the loader can find `coremldata.bin` inside each bundle.

### Folder layout

Place the staged repo in persistent storage (replace `/opt/models` with your path):

```
/opt/models
└── speaker-diarization-coreml
    ├── pyannote_segmentation.mlmodelc
    │   ├── coremldata.bin
    │   └── ...
    └── wespeaker_v2.mlmodelc
        ├── coremldata.bin
        └── ...
```

You can clone with Git LFS, download the `.tar` archives from HuggingFace, or copy the directory from a machine that already ran `DiarizerModels.downloadIfNeeded()` (macOS cache: `~/Library/Application Support/FluidAudio/Models/speaker-diarization-coreml`).

### Loading without downloads

Point `DiarizerModels.load` at the staged bundles and initialize the manager with the returned models:

```swift
import FluidAudio

@main
struct OfflineDiarizer {
    static func main() async {
        do {
            let basePath = "/opt/models/speaker-diarization-coreml"
            let segmentation = URL(fileURLWithPath: basePath).appendingPathComponent("pyannote_segmentation.mlmodelc")
            let embedding = URL(fileURLWithPath: basePath).appendingPathComponent("wespeaker_v2.mlmodelc")

            let models = try await DiarizerModels.load(
                localSegmentationModel: segmentation,
                localEmbeddingModel: embedding
            )

            let diarizer = DiarizerManager()
            diarizer.initialize(models: models)

            // ... run diarization as usual
        } catch {
            print("Failed to load diarizer models: \(error)")
        }
    }
}
```

Use `FileManager` to verify the two `.mlmodelc` folders exist before loading. When paths are correct, the loader never contacts the network and no auto-download occurs.

### Custom Configuration (Optional)

For fine-tuning, you can customize the configuration:
```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,  // Speaker separation sensitivity (0.5-0.9)
    minSpeechDuration: 1.0,     // Minimum speech segment (seconds)
    minSilenceGap: 0.5          // Minimum gap between speakers (seconds)
)
let diarizer = DiarizerManager(config: config)
```

### Offline VBx Pipeline (Batch Diarization)

> Requires macOS 14 / iOS 17 or later. The offline stack uses native C++ clustering and AsyncStream coordination that are unavailable on older OS releases.

When you need full parity with the pyannote/Core ML exporter (powerset segmentation + VBx clustering), use `OfflineDiarizerManager`. It orchestrates segmentation, soft mask interpolation, WeSpeaker embedding extraction, PLDA/VBx clustering, and timeline reconstruction in one place:

```swift
import FluidAudio

let config = OfflineDiarizerConfig()
let manager = OfflineDiarizerManager(config: config)
try await manager.prepareModels()  // Downloads + compiles Core ML bundles when missing

let samples = try AudioConverter().resampleAudioFile(path: "meeting.wav")
let result = try await manager.process(audio: samples)

for segment in result.segments {
    print("\(segment.speakerId) → \(segment.startTimeSeconds)s – \(segment.endTimeSeconds)s")
}
```

For file-based processing, use the memory-mapped streaming API which automatically handles large audio files efficiently:

```swift
let url = URL(fileURLWithPath: "meeting.wav")
let result = try await manager.process(url)
```

The file-based API internally uses memory-mapped streaming to avoid materializing the entire buffer in memory.

The offline controller mirrors the reference pipeline:

- **Segmentation:** `SegmentationRunner` feeds 10 s/160 k sample chunks through the Core ML segmentation model. Each chunk yields 589 frame-level log probabilities over the 7 local powerset classes.
- **Binarization:** `Binarization.logProbsToWeights` converts log probabilities to soft VAD weights; binary masks are still available for diagnostics.
- **Weight interpolation:** `WeightInterpolation` applies the same half-pixel mapping as `scipy.ndimage.zoom`, preserving the Core ML exporter’s alignment when resampling 589-frame masks to the embedding model’s pooling rate.
- **Embedding extraction:** `EmbeddingRunner` batches audio + resampled weights and returns L2-normalized 256-d embeddings.
- **VBx clustering:** `VBxClustering` (with `AHCClustering` warm start and `PLDAScoring`) runs the full VBx refinement loop using the JSON parameters exported with the model bundle.
- **Timeline reconstruction:** `TimelineReconstruction` now derives frame duration from the actual segmentation output and `OfflineDiarizerConfig.windowDuration`, ensuring timestamps stay correct if you swap in models with different hop sizes.

`OfflineDiarizerConfig` groups knobs by pipeline stage:

- `segmentation`: Window length (default 10 s), step ratio, min on/off durations, and sample rate. These must align with the exported Core ML segmentation model.
- `embedding`: Batch size and overlap handling. Keep `excludeOverlap` enabled for community-1 style powerset outputs.
- `clustering`: The VBx warm-start threshold plus pyannote's Fa/Fb priors.
- `vbx`: Max iterations and convergence tolerance for the refinement loop.
- `postProcessing`: Minimum gap duration when stitching segments back together.
- `export`: Optional `embeddingsPath` for dumping per-speaker vectors to JSON.

`prepareModels` captures Core ML compilation timings (and download durations when a fresh fetch is needed), so `DiarizationResult.timings` reflects audio loading, segmentation, embedding, clustering, and post-processing costs in one place. Per-speaker embeddings are exposed in `speakerDatabase` for downstream analytics without toggling debug flags.

#### CLI shortcut

The CLI exposes the same controller via `fluidaudio process` and the diarization benchmark tooling:

```bash
swift run fluidaudiocli process meeting.wav --mode offline --threshold 0.6 --debug --chunk-seconds 10.0 --overlap-seconds 0.0
swift run fluidaudiocli diarization-benchmark --mode offline --dataset ami-sdm --threshold 0.6 --auto-download
```

Add `--rttm path/to/meeting.rttm` when you have ground-truth annotations to emit DER/JER directly on the console, or `--export-embeddings embeddings.json` to inspect cluster assignments. The GitHub Actions workflow [`offline-pipeline.yml`](../.github/workflows/offline-pipeline.yml) executes the single-file AMI benchmark on every PR, keeping downloads, PLDA transforms, and VBx clustering guard-railed.

Both commands reuse the shared model cache (`OfflineDiarizerModels.defaultModelsDirectory()`) and emit JSON payloads compatible with the streaming pipeline.

#### Advanced: Manual Audio Source Control

For use cases requiring fine-grained control over memory management or audio loading, you can manually construct the audio source using `StreamingAudioSourceFactory`:

```swift
import FluidAudio

let config = OfflineDiarizerConfig()
let manager = OfflineDiarizerManager(config: config)
try await manager.prepareModels()

let factory = StreamingAudioSourceFactory()
let (source, loadDuration) = try factory.makeDiskBackedSource(
    from: URL(fileURLWithPath: "meeting.wav"),
    targetSampleRate: config.segmentation.sampleRate
)
defer { source.cleanup() }

let result = try await manager.process(
    audioSource: source,
    audioLoadingSeconds: loadDuration
)
```

This approach is useful when you need to:

- Process the same file multiple times without reloading
- Measure audio loading time separately from diarization time
- Implement custom cleanup or caching logic

For most use cases, the simpler `manager.process(url)` API is recommended.

## Streaming/Real-time Processing

For online diarization, prefer LS-EEND or Sortformer first, and use WeSpeaker/Pyannote only when you specifically need its speaker-database behavior.

### LS-EEND Streaming

Use `LSEENDDiarizer` when you want live diarizer with low latency and up to 10 speakers. It is very eager to detect speech, and performs well even when many simultaneous speakers are present.

```swift
import FluidAudio

let diarizer = LSEENDDiarizer()
try await diarizer.initialize(variant: .dihard3)

try diarizer.addAudio(audioChunk, sourceSampleRate: 16_000)
if let update = try diarizer.process() {
    for segment in update.finalizedSegments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

Notes:

- Supports streaming and complete-buffer inference with the same API.
- Returns `DiarizerTimelineUpdate` values with finalized and tentative segments.
- Use `enrollSpeaker(withSamples:named:overwritingAssignedSpeakerName:)` to pre-enroll speakers before streaming. This keeps the warmed LS-EEND session state while resetting the visible timeline.
- See [LS-EEND.md](LS-EEND.md) for the full API and call flow.

### Sortformer Streaming

Use `SortformerDiarizer` when you want robust low-latency streaming with stronger speaker identity stability, and the 4-speaker limit is acceptable. It attempts to ignore background speakers, so it may miss quiet speech. It also sometimes gets overwhelmed when too many speakers are talking simultaneously.

```swift
import FluidAudio

let diarizer = SortformerDiarizer()
let models = try await SortformerModels.loadFromHuggingFace(config: .default)
diarizer.initialize(models: models)

try diarizer.addAudio(audioChunk, sourceSampleRate: 16_000)
if let update = try diarizer.process() {
    for segment in update.finalizedSegments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

Notes:

- Best when speaker ordering and identity stability matter more than maximum speaker count.
- Limited to 4 speakers and can miss quiet speech more often than LS-EEND.
- Use `enrollSpeaker(withAudio:named:overwritingAssignedSpeakerName:)` to pre-enroll speakers before streaming. This preserves the speaker cache/FIFO state and clears the visible timeline.
- Also returns `DiarizerTimelineUpdate` / `DiarizerTimeline` so the integration shape matches LS-EEND.
- See [Sortformer.md](Sortformer.md) for model loading, tuning, and behavior details.

### WeSpeaker/Pyannote Streaming

Use `DiarizerManager` when you need the classic segmentation + embedding + speaker-database pipeline. This is the slowest streaming option and works best with larger chunks.

Process audio in chunks for real-time applications:

```swift
// Configure for streaming
let diarizer = DiarizerManager()  // Default config works well
diarizer.initialize(models: models)

var stream = AudioStream(
    chunkDuration: 5.0, // 10.0 for best accuracy. 3.0 for lowest
    chunkSkip: 2.0, // Duration between successive chunk starts
    streamStartTime: 0.0, // Audio stream start time
    chunkingStrategy: .useMostRecent // Most recent
)

stream.bind { chunk, time in
    let results = try diarizer.performCompleteDiarization(chunk, atTime: time)
    for segment in results.segments {
        handleSpeakerSegment(segment)
    }
}

for audioSamples in audioStream {
    try stream.write(from: audioSamples)
}
```

Notes:

- Keep one `DiarizerManager` instance per stream so `SpeakerManager` maintains ID consistency.
- Always rebase per-chunk timestamps by `(chunkStartSample / sampleRate)`.
- Provide 16 kHz mono Float32 samples; pad final chunk to the model window.
- Tune `speakerThreshold` and `embeddingThreshold` to trade off ID stability vs. sensitivity.
- Prefer chunk sizes of at least 5 seconds; 10 seconds is usually better.

**Speaker Enrollment:** The `Speaker` class includes a `name` field for enrollment workflows. When users introduce themselves ("My name is Alice"), update the speaker's name from the default (e.g. "Speaker_1") to enable personalized identification.

### Chunk Size Considerations

The `performCompleteDiarization` function accepts audio of any length, but accuracy varies:

- **< 3 seconds**: May fail or produce unreliable results
- **3-5 seconds**: Minimum viable chunk, reduced accuracy
- **10 seconds**: Optimal balance of accuracy and latency (recommended)
- **> 10 seconds**: Good accuracy but higher latency
- **Maximum**: Limited only by memory

You can adjust chunk size based on your needs:
- **Low latency**: Use 3-5 second chunks (accept lower accuracy)
- **High accuracy**: Use 10+ second chunks (accept higher latency)

The diarizer doesn't automatically chunk audio - you need to:
1. Accumulate incoming audio samples to your desired chunk size
2. Process chunks with `performCompleteDiarization`
3. Maintain speaker IDs across chunks using `SpeakerManager`

### Real-time Audio Capture Example

```swift
import AVFoundation

class RealTimeDiarizer {
    private let audioEngine = AVAudioEngine()
    private let diarizer: DiarizerManager
    private var audioStream: AudioStream
    
    init() async throws {
        let models = try await DiarizerModels.downloadIfNeeded()
        diarizer = DiarizerManager()  // Default config
        diarizer.initialize(models: models)
        audioStream = AudioStream(
            chunkDuration: 5.0, // 5 second chunks work well
            chunkSkip: 3.0, // 3.0 second delay between chunks works well
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip // ensure chunks are evenly spaced
        )
        audioStream.bind { chunk, _ in
            Task {
                do {
                    let result = try diarizer.performCompleteDiarization(chunk)
                    await handleResults(result)
                } catch {
                    print("Diarization error: \(error)")
                }
            }
        }
    }

    func startCapture() throws {
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        // Install tap to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            try? self.audioStream.write(from: buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    @MainActor
    private func handleResults(_ result: DiarizationResult) {
        for segment in result.segments {
            print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
        }
    }
}
```

## Core Components

### DiarizerManager
Main entry point for diarization pipeline:
```swift
let diarizer = DiarizerManager()  // Default config (recommended)
diarizer.initialize(models: models)

// Normalize with AudioConverter
let samples = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: "path/to/audio.wav"))
let result = try diarizer.performCompleteDiarization(samples)
```

### SpeakerManager
Tracks speaker identities across audio chunks:
```swift
let speakerManager = diarizer.speakerManager

// Get speaker information
print("Active speakers: \(speakerManager.speakerCount)")
for speakerId in speakerManager.speakerIds {
    if let speaker = speakerManager.getSpeaker(for: speakerId) {
        print("\(speaker.name): \(speaker.duration)s total")
    }
}
```

### DiarizerConfig
Configuration parameters:
```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,      // Speaker separation threshold (0.0-1.0)
    minSpeechDuration: 1.0,         // Minimum speech duration in seconds
    minSilenceGap: 0.5,             // Minimum silence between speakers
    minActiveFramesCount: 10.0,     // Minimum active frames for valid segment
    debugMode: false                // Enable debug logging
)
```

## Known Speaker Recognition

Pre-load known speaker profiles:

```swift
// Create embeddings for known speakers
let aliceAudio = loadAudioFile("alice_sample.wav")
let aliceEmbedding = try diarizer.extractEmbedding(aliceAudio)

// Initialize with known speakers
let alice = Speaker(id: "Alice", name: "Alice", currentEmbedding: aliceEmbedding)
let bob = Speaker(id: "Bob", name: "Bob", currentEmbedding: bobEmbedding)
speakerManager.initializeKnownSpeakers([alice, bob])

// Process - will use "Alice" instead of "Speaker_1" when matched
let result = try diarizer.performCompleteDiarization(audioSamples)
```

## SwiftUI Integration

```swift
import SwiftUI
import FluidAudio

struct DiarizationView: View {
    @StateObject private var processor = DiarizationProcessor()

    var body: some View {
        VStack {
            Text("Speakers: \(processor.speakerCount)")

            List(processor.activeSpeakers) { speaker in
                HStack {
                    Circle()
                        .fill(speaker.isSpeaking ? Color.green : Color.gray)
                        .frame(width: 10, height: 10)
                    Text(speaker.name)
                    Spacer()
                    Text("\(speaker.duration, specifier: "%.1f")s")
                }
            }

            Button(processor.isProcessing ? "Stop" : "Start") {
                processor.toggleProcessing()
            }
        }
    }
}

@MainActor
class DiarizationProcessor: ObservableObject {
    @Published var speakerCount = 0
    @Published var activeSpeakers: [SpeakerDisplay] = []
    @Published var isProcessing = false

    private var diarizer: DiarizerManager?

    func toggleProcessing() {
        if isProcessing {
            stopProcessing()
        } else {
            startProcessing()
        }
    }

    private func startProcessing() {
        Task {
            let models = try await DiarizerModels.downloadIfNeeded()
            diarizer = DiarizerManager()  // Default config
            diarizer?.initialize(models: models)
            isProcessing = true

            // Start audio capture and process chunks
            AudioCapture.start { [weak self] chunk in
                self?.processChunk(chunk)
            }
        }
    }

    private func processChunk(_ audio: [Float]) {
        Task { @MainActor in
            guard let diarizer = diarizer else { return }

            let result = try diarizer.performCompleteDiarization(audio)
            speakerCount = diarizer.speakerManager.speakerCount

            // Update UI with current speakers
            activeSpeakers = diarizer.speakerManager.speakerIds.compactMap { id in
                guard let speaker = diarizer.speakerManager.getSpeaker(for: id) else {
                    return nil
                }
                return SpeakerDisplay(
                    id: id,
                    name: speaker.name,
                    duration: speaker.duration,
                    isSpeaking: result.segments.contains { $0.speakerId == id }
                )
            }
        }
    }
}
```

## Performance Optimization

```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,
    minSpeechDuration: 1.0,
    minSilenceGap: 0.5
)

// Lower latency for real-time
let config = DiarizerConfig(
    clusteringThreshold: 0.7,
    minSpeechDuration: 0.5,    // Faster response
    minSilenceGap: 0.3         // Quicker speaker switches
)
```

### Memory Management
```swift
// Reset between sessions to free memory
diarizer.speakerManager.reset()

// Or cleanup completely
diarizer.cleanup()
```

## Benchmarking

Evaluate performance on your audio:

```bash
# Command-line benchmark
swift run fluidaudiocli diarization-benchmark --single-file ES2004a

# Results:
# DER: 17.7% (Miss: 10.3%, FA: 1.6%, Speaker Error: 5.8%)
# RTFx: 141.2x (real-time factor) M1 2022
```

## API Reference

### DiarizerManager

| Method | Description |
|--------|-------------|
| `initialize(models:)` | Initialize with Core ML models |
| `performCompleteDiarization(_:sampleRate:)` | Process audio and return segments |
| `cleanup()` | Release resources |

### SpeakerManager

| Method | Description |
|--------|-------------|
| `assignSpeaker(_:speechDuration:)` | Assign embedding to speaker |
| `initializeKnownSpeakers(_:)` | Load known speaker profiles |
| `getSpeaker(for:)` | Get speaker details |
| `reset()` | Clear all speakers |

### DiarizationResult

| Property | Type | Description |
|----------|------|-------------|
| `segments` | `[TimedSpeakerSegment]` | Speaker segments with timing |
| `speakerDatabase` | `[String: [Float]]?` | Speaker embeddings keyed by speaker ID |
| `timings` | `PipelineTimings?` | Processing timings for the diarization pass |

## Requirements

- iOS 17.0+ / macOS 14.0+
- Swift 5.9+
- ~100MB for Core ML models (downloaded on first use)

## Performance

| Device | RTFx | Notes |
|--------|------|-------|
| M2 MacBook Air | 150x | Apple Neural Engine |
| M1 iPad Pro | 120x | Neural Engine |
| iPhone 14 Pro | 80x | Neural Engine |
| GitHub Actions | 140x | CPU only |

## Troubleshooting

### High DER on certain audio
- Check if audio has overlapping speech (not yet supported)
- Ensure 16kHz sampling rate
- Verify audio isn't too noisy

### Memory issues
- Call `reset()` between sessions
- Process shorter chunks for streaming
- Reduce `minActiveFramesCount` if needed

### Model download fails
- Check internet connection
- Verify ~100MB free space
- Models cached after first download

## License

See main repository LICENSE file.
