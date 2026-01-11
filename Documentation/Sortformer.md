# Sortformer Streaming Speaker Diarization

## Overview

Sortformer is an end-to-end neural speaker diarization model that answers "who spoke when" in real-time. Unlike traditional diarization pipelines that require separate VAD, segmentation, and clustering stages, Sortformer directly outputs frame-level speaker probabilities for 4 fixed speaker slots.

**Key Features:**
- Real-time streaming inference with configurable latency
- 4 fixed speaker slots (no clustering required)
- ~80ms frame resolution (8x subsampling of 10ms mel frames)
- CoreML-optimized for Apple Silicon

## Sortformer vs DiarizerManager (Pyannote-based)

**Sortformer** handles noisy environments and overlapping speakers well, but is limited to 4 speakers (for now). It also misses quiet speech as it's trained to ignore background conversations. The most common source of error is missed speech.

**DiarizerManager** works better when you have more than 4 speakers, but struggles with background noise, background conversations, and similar-sounding speakers. The most common source of error is incorrect labeling.

| Environment | Sortformer | DiarizerManager |
|-------------|:----------:|:---------------:|
| Clean/silent room | Best | Good |
| Background noise | Best | Poor |
| Speech from another room | Poor | Good |
| High overlap | Best | Good |
| More than 4 speakers | No | Yes |
| Remembering speakers across meetings | No | Yes |

## Architecture

### Processing Pipeline

```
Audio (16kHz) → Mel Spectrogram → CoreML Model → Speaker Probabilities
                    ↓                   ↓
              [T, 128] features   [T', 4] probabilities
```

The pipeline consists of:

1. **Mel Spectrogram** (`NeMoMelSpectrogram`): Converts raw audio to 128-bin mel features
2. **CoreML Model** (`DiarizerInference`): Combined encoder + attention + head
3. **Streaming State** (`SortformerStreamingState`): Maintains speaker cache and FIFO queue
4. **Post-processing** (`SortformerTimeline`): Converts probabilities to speaker segments

### Streaming State Management

Sortformer maintains two key buffers for streaming:

```
┌─────────────────┐  ┌──────────────┐  ┌─────────────┐
│  Speaker Cache  │  │  FIFO Queue  │  │  New Chunk  │
│   (historical)  │  │   (recent)   │  │  (current)  │
└─────────────────┘  └──────────────┘  └─────────────┘
     188 frames      40 or 188 frames  6 or 340 frames
```

- **Speaker Cache** (`spkcache`): Compressed historical embeddings representing long-term speaker context
- **FIFO Queue** (`fifo`): Recent embeddings for short-term context
- **Chunk**: Current audio chunk being processed

### Key Streaming Parameters

#### FIFO Length (`fifoLen`)

The FIFO (First-In-First-Out) queue stores recent embeddings that haven't been compressed into the speaker cache yet. This provides immediate context for the current chunk.

```
FIFO Queue Role:
┌────────────────────────────────────────────────┐
│  Recent frames waiting to be processed         │
│  ────────────────────────────────────────────  │
│  [frame_n-40] [frame_n-39] ... [frame_n-1]     │
│       ↓           ↓              ↓             │
│  Oldest ──────────────────────► Newest         │
└────────────────────────────────────────────────┘
```

| Config | `fifoLen` | Effect |
|--------|-----------|--------|
| Default | 40 | Smaller memory, faster compression cycles |
| NVIDIA Low | 188 | Larger context before compression |
| NVIDIA High | 40 | Same as default |

When `fifoLen + newChunkFrames > fifoLen` capacity, frames are popped from FIFO and either:
1. Added to speaker cache (if speaker was active)
2. Used to update the silence profile (if silence detected)

#### Right Context (`chunkRightContext`)

Right context determines how many future frames the model can "look ahead" before making predictions. This is the primary factor affecting **output latency**.

```
Chunk with Context:
┌─────────┬──────────────┬──────────────┐
│   LC    │     CORE     │      RC      │
│ (past)  │  (current)   │   (future)   │
└─────────┴──────────────┴──────────────┘
  1 frame    6 frames      7 frames
     ↓          ↓             ↓
  8 mel     48 mel        56 mel frames
  frames    frames
```

| Config | `rightContext` | Look-ahead | Latency Impact |
|--------|----------------|------------|----------------|
| Default | 7 | 7 × 80ms = 560ms | Low latency |
| NVIDIA Low | 7 | 7 × 80ms = 560ms | Low latency |
| NVIDIA High | 40 | 40 × 80ms = 3.2s | High latency, better quality |

**Why Right Context Matters:**

More right context = more future information = better predictions, but higher latency. You can get the predictions from the right context as tentative predictions.

```
Without right context (RC=0):
  Speaker A: "Hello, I am—"
  Model sees: "Hello, I am—" → Must predict NOW (may miss speaker change)

With the right context (RC=7):
  Speaker A: "Hello, I am—" [Speaker B: "Hi!"]
  Model sees: "Hello, I am— Hi!" → Can predict speaker change accurately
```

#### Left Context (`chunkLeftContext`)

Left context provides past frames for continuity between chunks. Unlike the right context, it doesn't add latency since these frames were already processed.

| Config | `leftContext` | Purpose |
|--------|---------------|---------|
| All | 1 | Minimal overlap for chunk boundary smoothing |

### Latency Calculation

Total output latency is calculated as:

```
latency = (chunkLen + rightContext) × subsamplingFactor × melStride / sampleRate

Default config:
  = (6 + 7) × 8 × 160 / 16000
  = 13 × 8 × 0.01
  = 1.04 seconds

NVIDIA High Latency config:
  = (340 + 40) × 8 × 160 / 16000
  = 380 × 8 × 0.01
  = 30.4 seconds
```

### Why These Parameters Are Baked Into Models

CoreML models have **static input shapes**. The tensor dimensions for FIFO and chunk inputs are fixed at conversion time:

```python
# During model conversion (Python)
fifo_len = 40        # Fixed in model
spkcache_len = 188   # Fixed in model
chunk_mel_frames = (chunk_len + lc + rc) * 8  # Fixed in model
```

This means you **cannot** change `fifoLen`, `spkcacheLen`, or context values at runtime. You must use a model that was converted with matching parameters.

## File Structure

```
Sources/FluidAudio/Diarizer/Sortformer/
├── SortformerConfig.swift      # Streaming parameters and model shape configuration
├── Pipeline.swift    # Main entry point, audio buffering, inference orchestration
├── DiarizerInference.swift      # CoreML model container and HuggingFace loading
├── StateUpdater.swift     # Speaker cache compression, FIFO queue, state updates
└── SortformerTypes.swift       # StreamingState, FeatureLoader, ChunkResult, Timeline, Segment
```

### SortformerConfig.swift

Defines streaming parameters that must match the CoreML model's static shapes:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunkLen` | 6 | Output frames per chunk |
| `chunkLeftContext` | 1 | Left context frames |
| `chunkRightContext` | 7 | Right context frames |
| `fifoLen` | 40 | FIFO queue capacity |
| `spkcacheLen` | 188 | Speaker cache capacity |
| `subsamplingFactor` | 8 | Encoder downsampling rate |

**Pre-defined Configurations:**

```swift
// Default (~1.04s latency, lowest latency)
SortformerConfig.default

// NVIDIA High Latency (30.4s latency, best quality)
SortformerConfig.nvidiaHighLatency

// NVIDIA Low Latency (1.04s latency)
SortformerConfig.nvidiaLowLatency
```

### Pipeline.swift

Main entry point for diarization:

```swift
let diarizer = Pipeline()

// Initialize with HuggingFace models
try await diarizer.initialize(mainModelPath: modelURL)

// Streaming mode
for audioChunk in audioStream {
    if let result = try diarizer.processSamples(audioChunk) {
        // Handle speaker probabilities
        for frame in 0..<result.frameCount {
            for speaker in 0..<4 {
                let prob = result.getSpeakerPrediction(speaker: speaker, frame: frame)
            }
        }
    }
}

// Or process complete file
let timeline = try diarizer.processComplete(audioSamples)
```

**Key Methods:**
- `addAudio(_:)` - Buffer audio samples
- `process()` - Run inference on buffered audio
- `processSamples(_:)` - Convenience method combining add + process
- `processComplete(_:)` - Batch process entire audio file

### DiarizerInference.swift

Handles CoreML model loading and inference:

```swift
// Load from HuggingFace
let models = try await DiarizerInference.loadFromHuggingFace(
    config: .default,
    computeUnits: .all
)

// Or load from local path
let models = try await DiarizerInference.load(
    config: .default,
    mainModelPath: localModelURL
)
```

**Model Inputs:**
- `chunk`: Mel features [1, T, 128]
- `chunk_lengths`: Actual chunk length [1]
- `spkcache`: Speaker cache embeddings [1, 188, 512]
- `spkcache_lengths`: Actual cache length [1]
- `fifo`: FIFO queue embeddings [1, 40, 512]
- `fifo_lengths`: Actual FIFO length [1]

**Model Outputs:**
- `speaker_preds`: Probabilities [T', 4] (sigmoid applied internally)
- `chunk_pre_encoder_embs`: Embeddings for state update
- `chunk_pre_encoder_lengths`: Actual embedding count

### StateUpdater.swift

Core streaming logic ported from NeMo:

```swift
let modules = StateUpdater(config: config)

let result = try modules.streamingUpdate(
    state: &state,
    chunk: chunkEmbeddings,
    preds: predictions,
    leftContext: leftContext,
    rightContext: rightContext
)

// result.confirmed - Final predictions for this chunk
// result.tentative - Predictions that may change with more context
```

**Key Functions:**
- `streamingUpdate()` - Main state update logic
- `compressSpkcache()` - Compress speaker cache when full
- `getTopKIndices()` - Select important frames for cache
- `updateSilenceProfile()` - Track silence embeddings

### SortformerTypes.swift

**SortformerStreamingState** - Mutable state for streaming:
```swift
struct SortformerStreamingState {
    var spkcache: [Float]           // Historical embeddings
    var spkcacheLength: Int
    var fifo: [Float]               // Recent embeddings
    var fifoLength: Int
    var meanSilenceEmbedding: [Float]  // Running silence mean
}
```

**SortformerChunkResult** - Output from each chunk:
```swift
struct SortformerChunkResult {
    let speakerPredictions: [Float]  // [frameCount, 4] flattened
    let frameCount: Int // Number of frames with predictions
    let startFrame: Int
    let tentativePredictions: [Float]  // Volatile predictions from frames in the right context.
    let tentativeFrameCount: Int // Number of tentative frames
    private(set) var tentativeStartFrame: Int // Frame index of first tentative frame
}
```

**SortformerTimeline** - Accumulated results with segments:
```swift
struct SortformerTimeline {
    let config: SortformerPostProcessingConfig                  // Post-processing configuration
    private(set) var framePredictions: [Float]                  // Finalized frame-wise speaker predictions [numFrames, numSpeakers]
    private(set) var tentativePredictions: [Float]              // Tentative predictions [numTentative, numSpeakers]
    private(set) var numFrames: Int                             // Total number of finalized median-filtered frames
    private(set) numTentative: Int                              // Number of tentative frames (including right context frames from chunk)
    private(set) var segments: [[SortformerSegment]]            // Finalized segments (completely before the median filter boundary)
    private(set) var tentativeSegments: [[SortformerSegment]]   // Tentative segments (may change as more predictions arrive)
    private(set) duration: Float                                // Get total duration of finalized predictions in seconds
    private(set) var tentativeDuration: Float                   // Get total duration including tentative predictions in seconds
}
```

**SortformerSegment** - Timeline Segment
```swift
public struct SortformerSegment {
    let id: UUID /// Segment ID
    var speakerIndex: Int // Speaker index in Sortformer output
    var startFrame: Int // Index of segment start frame
    var endFrame: Int // Index of segment end frame
    var isFinalized: Bool // Whether this segment is finalized
    private(set) var length: Int // Length of the segment in frames
    private(set) var startTime: Float // Start time in seconds
    private(set) var endTime: Float // End time in seconds
    private(set) var duration: Float // Duration in seconds
    private(set) var speakerLabel: String // Speaker label (e.g., "Speaker 0")
}
```

## Streaming Flow

```
┌────────────────────────────────────────────────────────────────┐
│                     Pipeline                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. addAudio(samples)                                          │
│     └─→ audioBuffer.append(samples)                            │
│     └─→ preprocessAudioToFeatures()                            │
│         └─→ melSpectrogram.computeFlatTransposed()             │
│         └─→ featureBuffer.append(mel)                          │
│                                                                │
│  2. process()                                                  │
│     └─→ while getNextChunkFeatures() != nil:                   │
│         │                                                      │
│         ├─→ models.runMainModel(chunk, state)                  │
│         │   └─→ CoreML inference                               │
│         │   └─→ returns: predictions, embeddings               │
│         │                                                      │
│         ├─→ modules.streamingUpdate(state, embeddings, preds)  │
│         │   └─→ Update FIFO queue                              │
│         │   └─→ Compress speaker cache if needed               │
│         │   └─→ returns: confirmed, tentative predictions      │
│         │                                                      │
│         └─→ timeline.addChunk(result)                          │
│             └─→ Update segments per speaker                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Latency vs Quality Trade-offs

| Config | Chunk Size | Latency | Quality |
|--------|------------|---------|---------|
| `default` | 6 frames | ~1.04s | Good |
| `nvidiaLowLatency` | 6 frames | ~1.04s | Better |
| `nvidiaHighLatency` | 340 frames | ~30.4s | Best |

Latency is determined by:
- `chunkLen * subsamplingFactor * melStride / sampleRate`
- Plus `rightContext` frames for look-ahead

## Speaker Cache Compression

When the speaker cache overflows, Sortformer compresses it by:

1. Computing log-probability scores for each frame
2. Boosting scores for recent and high-confidence frames
3. Selecting top-k frames per speaker using score ranking
4. Replacing silent frames with mean silence embedding

This preserves the most informative historical context while bounding memory usage.

## Post-Processing

`SortformerPostProcessingConfig` controls segment extraction:

```swift
let config = SortformerPostProcessingConfig(
    onsetThreshold: 0.5,    // Probability to start speech
    offsetThreshold: 0.5,   // Probability to end speech
    minDurationOn: 0.25,    // Min speech segment (seconds)
    minDurationOff: 0.1     // Min gap between segments
)
```

## Model Variants

Three CoreML models are available on HuggingFace:

| Variant | File | Config |
|---------|------|--------|
| Default | `Sortformer.mlmodelc` | `SortformerConfig.default` |
| NVIDIA Low | `SortformerNvidiaLow.mlmodelc` | `SortformerConfig.nvidiaLowLatency` |
| NVIDIA High | `SortformerNvidiaHigh.mlmodelc` | `SortformerConfig.nvidiaHighLatency` |

**Important:** Each model has baked-in static shapes. You must use the matching configuration.

## Usage Examples

### Real-time Streaming

```swift
let diarizer = SortformerDiarizer(config: .default)
let models = try await SortformerModels.loadFromHuggingFace(config: .default)
try await diarizer.initialize(models: models)

// Process audio in chunks (e.g., from microphone)
audioEngine.installTap { buffer in
    let samples = buffer.floatChannelData![0]
    if let result = try? diarizer.processSamples(Array(samples)) {
        // Update UI with speaker probabilities
        updateSpeakerDisplay(result)

        // OR update UI with updated timeline
        updateSpeakerDisplay(diarizer.timeline)
    }
}
```

### Batch Processing

```swift
let diarizer = SortformerDiarizer(config: .nvidiaHighLatency)
let models = try await SortformerModels.loadFromHuggingFace(config: .default)
try await diarizer.initialize(models: models)

let timeline = try diarizer.processComplete(audioSamples)

// Get segments per speaker
for (speakerIndex, segments) in timeline.segments.enumerated() {
    for segment in segments {
        print("Speaker \(speakerIndex): \(segment.startTime)s - \(segment.endTime)s")
    }
}
```

## References

- [NVIDIA Sortformer Paper](https://arxiv.org/abs/2409.06656)
- [NeMo Sortformer Implementation](https://github.com/NVIDIA/NeMo)
- [HuggingFace Models](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml)
