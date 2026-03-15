# LS-EEND Streaming Speaker Diarization

## Overview

LS-EEND (Long-Form Streaming End-to-End Neural Diarization) is a streaming diarization model that answers "who spoke when" in real-time. It uses a causal encoder and online attractor decoder to generate per-frame speaker probabilities without needing separate VAD, segmentation, or clustering.

**Key Features:**
- Real-time streaming inference with frame-in-frame-out output
- Up to 8 simultaneous speakers (no fixed speaker slots)
- ~100ms frame resolution (10 Hz output)
- Handles recordings up to one hour
- CoreML-optimized for Apple Silicon
- 8000 Hz input sample rate (automatic resampling via `processComplete(audioFileURL:)`)

**Limitations:**
- 8000 Hz sample rate means reduced audio fidelity compared to 16 kHz models
- Speaker identity is local to the recording — no persistent speaker embeddings across sessions
- Variants are specialized: using the wrong variant for a domain can significantly hurt accuracy

## LS-EEND vs Sortformer

LS-EEND and Sortformer share the same `Diarizer` protocol API. The main differences are their training domains, speaker capacity, and sample rate.

| Feature | LS-EEND | Sortformer |
|---------|:-------:|:----------:|
| Max speakers | 8 | 4 |
| Sample rate | 8000 Hz | 16000 Hz |
| Phone/telephony audio | Best | Poor |
| In-person meetings | Good (.ami) | Good |
| Wild/unconstrained audio | Good (.dihard3) | Good |
| Background noise robustness | Good | Best |
| Speaker count > 4 | Yes | No |
| Multiple domain variants | Yes (4) | No |

Use LS-EEND when you have phone calls, more than 4 speakers, or domain-specific recordings that match one of its training conditions. Use Sortformer for noisy environments, meetings up to 4 speakers, or when 16 kHz audio quality matters.

## Variant Selection

Each variant is a separate CoreML model trained on a specific corpus. Choose the variant that best matches your audio source.

### `.ami` — In-person meetings
Trained on the AMI meeting corpus: multi-speaker conference room recordings with close-talk and distant microphones. Best for boardroom meetings, panel discussions, or any scenario with 3–5 speakers in a shared physical space.

- **DER (AMI test set):** 20.76%
- **Typical speaker count:** 3–5

### `.callhome` — Phone calls
Trained on the CALLHOME corpus: two-party telephone conversations with the codec noise and narrow bandwidth typical of phone audio. Best for call center recordings, customer service calls, or any 2-speaker telephony audio.

- **DER (CALLHOME test set):** 12.11%
- **Typical speaker count:** 2

### `.dihard2` — Difficult mixed conditions
Trained on DIHARD II, which includes dinner parties, clinical interviews, conference rooms, multi-channel microphone arrays, and child speech. Best for situations with challenging acoustics, heavy overlap, or non-standard recording setups.

- **DER (DIHARD II test set):** 27.58%
- **Typical speaker count:** 2–6

### `.dihard3` — In-the-wild conversations *(default)*
Trained on DIHARD III, a deliberately broad and heterogeneous collection of "found audio" — podcasts, audiobooks, broadcast media, YouTube, field recordings, and other uncontrolled sources. Best general-purpose choice when the recording conditions are unknown.

- **DER (DIHARD III test set):** 19.61%
- **Typical speaker count:** 2–6

## Architecture

### Processing Pipeline

```
Audio (8kHz) → Mel Spectrogram → Causal Encoder → Online Attractor Decoder → Speaker Probabilities
                   ↓                   ↓                    ↓
             [T, 128] features   Retention state      [T', speakers] probs
```

1. **Mel Spectrogram** (`NeMoMelSpectrogram`): Converts raw 8 kHz audio to 128-channel log-mel features. Window: 200 samples (25ms), hop: 80 samples (10ms).
2. **Splice-and-subsample**: Groups adjacent mel frames with a context window, then subsamples, reducing the sequence to ~10 Hz.
3. **Causal Encoder**: A Conformer-based transformer encoder with a **retention mechanism** that replaces full self-attention. Carries recurrent state across chunks for $O(n)$ memory usage over long recordings.
4. **Online Attractor Decoder**: Learns to track speaker identities over time by maintaining a cross-attention buffer (`topBuffer`) between encoder and decoder. Produces per-frame speaker logits for up to `maxNspks` slots.
5. **Post-processing** (`DiarizerTimeline`): Applies sigmoid, optional median filtering, onset/offset thresholding, and minimum duration filters to convert probabilities to discrete speaker segments.

The model includes two "boundary tracks" in its raw output that are used internally for sequence boundary handling. These are stripped before the probabilities are returned to the caller — `realOutputDim` reflects the usable speaker tracks.

### Streaming State

Unlike Sortformer's speaker cache/FIFO approach, LS-EEND carries six recurrent state tensors between inference steps:

| Buffer | Description |
|--------|-------------|
| `encRetKv` | Encoder retention key-value cache |
| `encRetScale` | Encoder retention normalization scale |
| `encConvCache` | Encoder convolutional cache |
| `decRetKv` | Decoder retention key-value cache |
| `decRetScale` | Decoder retention normalization scale |
| `topBuffer` | Cross-attention buffer between encoder and decoder |

These tensors are allocated once at session creation (zero-initialized) and updated in-place each step. The shapes are fixed by the model and read from the metadata JSON.

### Startup Latency

The model has a `convDelay` parameter — an initial number of frames the convolutional encoder must consume before the decoder can produce output. Combined with the FFT center-padding and context receptive field, this determines the minimum latency before the first output frame appears:

```
streamingLatencySeconds = (nFFT/2 + contextRecp×hopLength + convDelay×subsampling×hopLength) / sampleRate
```

For the default models this is typically under 1 second.

## File Structure

```
Sources/FluidAudio/Diarizer/LS-EEND/
├── LSEENDDiarizer.swift          # Main entry point — implements Diarizer protocol
├── LSEENDSupport.swift           # LSEENDMatrix, LSEENDModelDescriptor, LSEENDModelMetadata,
│                                 #   LSEENDVariant, LSEENDStateShapes, streaming result types
├── LSEENDEvaluation.swift        # DER computation, RTTM parsing/writing, collar masking,
│                                 #   optimal speaker assignment (Hungarian algorithm)
├── LSEENDInferenceEngine.swift   # CoreML model wrapper, session management
└── LSEENDRuntimeProbeSupport.swift  # CLI probe harness for integration testing
```

## Usage

### Quick Start

```swift
// Initialize with default variant (.dihard3)
let diarizer = LSEENDDiarizer()
try await diarizer.initialize()

// Process a file (handles resampling to 8kHz automatically)
let timeline = try diarizer.processComplete(audioFileURL: audioURL)

for segment in timeline.allSegments {
    print("Speaker \(segment.speakerIndex): \(segment.startTime)s – \(segment.endTime)s")
}
```

### Selecting a Variant

```swift
// Phone call
let diarizer = LSEENDDiarizer()
try await diarizer.initialize(variant: .callhome)

// In-person meeting
try await diarizer.initialize(variant: .ami)

// Unknown/mixed conditions
try await diarizer.initialize(variant: .dihard3)
```

### Real-time Streaming

Audio must be fed at the model's target sample rate (8000 Hz). Resample before calling `addAudio`.

```swift
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
try await diarizer.initialize(variant: .dihard3)

// Push audio in chunks (e.g. from microphone, resampled to 8kHz)
diarizer.addAudio(audioChunk)
if let update = try diarizer.process() {
    // New segments are available
    for segment in update.newSegments {
        print("Speaker \(segment.speakerIndex) started at \(segment.startTime)s")
    }
}

// When the stream ends, flush remaining frames
try diarizer.finalizeSession()
let finalTimeline = diarizer.timeline
```

### Streaming with Tentative Predictions

LS-EEND emits both committed and preview (tentative) probabilities each step. Committed frames are final; preview frames are speculative and will be refined by future audio.

```swift
if let update = try diarizer.process() {
    // Finalized predictions — safe to display
    let committed = update.newSegments

    // Tentative predictions — useful for a live "preview" display
    let tentative = update.tentativeSegments
}
```

### Batch Processing (Raw Samples)

If you have already-resampled 8 kHz audio:

```swift
let diarizer = LSEENDDiarizer()
try await diarizer.initialize(variant: .ami)

// samples must already be at 8000 Hz, mono Float32
let timeline = try diarizer.processComplete(samples)
```

### Custom Timeline Configuration

`DiarizerTimelineConfig` controls post-processing. These settings apply to all variants.

```swift
let diarizer = LSEENDDiarizer(
    computeUnits: .cpuOnly,
    onsetThreshold: 0.5,     // Probability above which speech starts
    offsetThreshold: 0.5,    // Probability below which speech ends
    onsetPadFrames: 1,       // Frames added before each segment
    offsetPadFrames: 1,      // Frames added after each segment
    minFramesOn: 3,          // Discard segments shorter than this
    minFramesOff: 2          // Close gaps shorter than this
)
try await diarizer.initialize(variant: .dihard3)
```

Or pass a config struct directly:

```swift
let config = DiarizerTimelineConfig(onsetThreshold: 0.4, onsetPadFrames: 0)
let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly, timelineConfig: config)
```

### Loading Models from Local Paths

```swift
let descriptor = LSEENDModelDescriptor(
    variant: .dihard3,
    modelURL: URL(fileURLWithPath: "/path/to/lseend_dihard3.mlmodelc"),
    metadataURL: URL(fileURLWithPath: "/path/to/lseend_dihard3_metadata.json")
)
let diarizer = LSEENDDiarizer()
try diarizer.initialize(descriptor: descriptor)
```

## Model Variants on HuggingFace

All four variants are hosted at [FluidInference/lseend-coreml](https://huggingface.co/FluidInference/lseend-coreml) and are downloaded automatically on first use. Files are cached in the app support directory (`~/Library/Application Support/FluidAudio/Models/`).

| Variant | Model file | Config file |
|---------|-----------|-------------|
| `.ami` | `lseend_ami.mlmodelc` | `lseend_ami_metadata.json` |
| `.callhome` | `lseend_callhome.mlmodelc` | `lseend_callhome_metadata.json` |
| `.dihard2` | `lseend_dihard2.mlmodelc` | `lseend_dihard2_metadata.json` |
| `.dihard3` | `lseend_dihard3.mlmodelc` | `lseend_dihard3_metadata.json` |

Pre-fetch models ahead of time:

```bash
swift run fluidaudiocli download --repo lseend
```

## CLI

```bash
# Diarize a single file
swift run fluidaudiocli lseend audio.wav

# Use a specific variant
swift run fluidaudiocli lseend audio.wav --variant callhome

# Adjust thresholds
swift run fluidaudiocli lseend audio.wav --threshold 0.4 --median-width 5

# Save output to JSON
swift run fluidaudiocli lseend audio.wav --output result.json

# Benchmark on AMI
swift run fluidaudiocli lseend-benchmark --auto-download --variant ami

# Benchmark on CALLHOME with custom settings
swift run fluidaudiocli lseend-benchmark --variant callhome --threshold 0.35 --collar 0.25
```

### Available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `dihard3` | Model variant: `ami`, `callhome`, `dihard2`, `dihard3` |
| `--threshold` | `0.5` | Speaker activity threshold |
| `--median-width` | `1` | Median filter width in frames |
| `--collar` | `0.0` | Collar in seconds to ignore around speaker boundaries (benchmark only) |
| `--onset` | — | Override onset threshold |
| `--offset` | — | Override offset threshold |
| `--pad-onset` | `0` | Frames to pad before each segment |
| `--pad-offset` | `0` | Frames to pad after each segment |
| `--min-duration-on` | `0.0` | Minimum segment duration in seconds |
| `--min-duration-off` | `0.0` | Minimum gap duration in seconds |
| `--output` | — | Path to save JSON results |

## References

- [LS-EEND Paper (arXiv 2410.06670)](https://arxiv.org/abs/2410.06670) — Di Liang, Xiaofei Li. *LS-EEND: Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction.* IEEE TASLP.
- [HuggingFace Models](https://huggingface.co/FluidInference/lseend-coreml)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [CALLHOME Corpus](https://catalog.ldc.upenn.edu/LDC97S42)
- [DIHARD Challenge](https://dihardchallenge.github.io/)
