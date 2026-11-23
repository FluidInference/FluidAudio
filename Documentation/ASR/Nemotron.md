# Nemotron Speech Streaming 0.6B

FluidAudio supports NVIDIA's Nemotron Speech Streaming model for real-time streaming ASR on Apple devices.

## Overview

Nemotron Speech Streaming 0.6B is a FastConformer RNNT model optimized for streaming speech recognition. The CoreML conversion provides:

- **Multiple chunk sizes** for latency/accuracy trade-offs
- **Int8 quantized encoder** (~564MB, 4x smaller than float32)
- **Streaming inference** with encoder cache for continuous audio

## Benchmark Results

Three tiers ship (Apple M5 Pro, LibriSpeech test-clean, 100 files, CPU+NE), all from one
conversion with **B1 fusion** (`decoder_joint.mlmodelc` — decoder+joint merged into one
CoreML call per RNN-T step, loaded automatically; argmax stays in Swift):

| Tier | WER | RTFx | Δ vs 1120ms |
|------|-----|------|-------------|
| 560ms | 2.28% | 42.1 | −35% |
| 1120ms | 2.28% | 65.0 | — |
| **2240ms (default)** | **2.46%** | **93.6** | **+44%** |

`.ms2240` is the default: doubling the streaming chunk (224 mel frames = 2× the trained
14-encoder-frame chunk, so the chunked-attention mask still tiles cleanly) halves per-chunk
fixed overhead for throughput, at ~1.1 s extra latency and no accuracy cost. Drop to `.ms1120`
or `.ms560` for lower latency. B1 fusion contributes ~+15% on top of any tier.

> Weights are a faithful conversion of the public `nvidia/nemotron-speech-streaming-en-0.6b`
> checkpoint (decoder & joint match PyTorch at cos=1.0). WER parity against NVIDIA's internal
> tuning of the same model is a tracked follow-up; the ladder above is internally consistent
> (one conversion for all tiers) and reports the relative gains.

## Quick Start

### Basic Usage

```swift
import FluidAudio

// Create manager
let manager = StreamingNemotronAsrManager()

// Load models (defaults to the 2240ms tier + B1-fused decode)
let modelDir = URL(fileURLWithPath: "~/.cache/fluidaudio/models/nemotron-streaming/2240ms")
try await manager.loadModels(modelDir: modelDir)

// Process audio buffer
let partialResult = try await manager.process(audioBuffer: buffer)
print("Partial: \(partialResult)")

// Finalize and get complete transcript
let finalTranscript = try await manager.finish()
print("Final: \(finalTranscript)")

// Reset for next utterance
await manager.reset()
```

### Selecting Chunk Size

Use `NemotronChunkSize` to select latency/accuracy trade-off:

```swift
// Available chunk sizes
let chunkSize: NemotronChunkSize = .ms560  // Recommended balance

// Get the corresponding HuggingFace repo
let repo = chunkSize.repo  // .nemotronStreaming560

// Download models
try await ModelHub.download(repo, to: modelsBaseDir)

// Models will be at: modelsBaseDir/nemotron-streaming/560ms/
```

### Automatic Model Download

FluidAudio can automatically download models from HuggingFace:

```swift
let manager = StreamingNemotronAsrManager(requestedChunkSize: .ms560)
let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".cache/fluidaudio/models")

// Downloads the tier and its encoder_int8_v2 (see below) if not cached, then loads
try await manager.loadModels(to: modelsBaseDir)
```

`ModelHub.download(chunkSize.repo, to:)` fetches only the tier folder, so a
directory filled that way has `encoder/encoder_int8.mlmodelc` and no v2 encoder.

### Encoder v2 (M2 Neural Engine)

On an M2 Pro (macOS 27.2) the Neural Engine compiler rejects the original
1120 ms encoder, and Core ML silently runs the whole encoder on the CPU: 0 of
1,667 ops on the ANE and a 5-minute first load (#1170). The cause is the one
behind the multilingual model's #1141, and all three tiers have it: in each
Conformer layer the conv-cache update is a slice of a concat
(`concat(cache, x)[..., T:T+8]`). `encoder_int8_v2.mlmodelc` is the same
encoder with each of those slices rewritten as slices of the concat's inputs,
the rewrite that made the multilingual encoders compile on the M2 Pro.
Weights, inputs and outputs are unchanged, and on an M5 Pro its outputs are
byte-identical to the original's, on the CPU and on the ANE. The 560 ms tier
costs about 0.5 ms more per step there (one extra concat per layer); 1120 and
2240 ms are unchanged.

`loadModels(to:)` downloads it in place of the tier folder's
`encoder_int8.mlmodelc`, and `loadModels(from:)` loads it whenever
`encoder/encoder_int8_v2.mlmodelc` exists. It lives in `encoder-v2/<tier>ms/`
on Hugging Face, outside the tier folders that releases up to 0.15.5 download
whole. A cache from an earlier release downloads it once (~565 MB); while
offline, the cached `encoder_int8.mlmodelc` keeps loading. For a directory you
manage yourself, copy `encoder-v2/<tier>ms/encoder_int8_v2.mlmodelc` into its
`encoder/` folder.

## Architecture

### Streaming Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMING RNNT PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

1. PREPROCESSOR (per audio chunk)
   audio [1, samples] → mel [1, 128, chunk_mel_frames]

2. ENCODER (with cache)
   mel [1, 128, total_mel_frames] + cache → encoded [1, 1024, T] + new_cache
   (total_mel_frames = pre_encode_cache + chunk_mel_frames)

3. DECODER + JOINT (greedy decoding per encoder frame)
   For each encoder frame:
     token → DECODER → decoder_out
     encoder_step + decoder_out → JOINT → logits
     argmax → predicted token
     if token == BLANK: next encoder frame
     else: emit token, update decoder state
```

### Chunk Configuration

| Chunk Size | mel_frames | pre_encode_cache | total_frames | samples |
|------------|------------|------------------|--------------|---------|
| 2240ms (default) | 224 | 9 | 233 | 35,840 |
| 1120ms | 112 | 9 | 121 | 17,920 |
| 560ms | 56 | 9 | 65 | 8,960 |

**Formula:** `chunk_ms = mel_frames × 10ms` (10ms per mel frame)

### Encoder Cache

The encoder maintains three cache tensors for streaming continuity:

| Cache | Shape | Description |
|-------|-------|-------------|
| cache_channel | [1, 24, 70, 1024] | Attention context |
| cache_time | [1, 24, 1024, 8] | Convolution state |
| cache_len | [1] | Fill level |

## Model Files

Each chunk-size variant contains:

```
nemotron-streaming/{chunk_size}/
├── encoder/
│   └── encoder_int8_v2.mlmodelc # ~564MB (int8 quantized; encoder_int8.mlmodelc
│                                #   in caches from earlier releases)
├── preprocessor.mlmodelc        # ~1MB
├── decoder.mlmodelc             # ~28MB
├── joint.mlmodelc               # ~7MB
├── metadata.json                # Configuration
└── tokenizer.json               # 1024 tokens
```

**Total size per variant:** ~600MB

## CLI Benchmark

Run benchmarks using the FluidAudio CLI:

```bash
# Build release
swift build -c release

# Benchmark with default 1120ms chunks
swift run -c release fluidaudiocli nemotron-benchmark --max-files 100

# Benchmark with 560ms chunks
swift run -c release fluidaudiocli nemotron-benchmark --chunk 560 --max-files 100

# Benchmark on test-other subset
swift run -c release fluidaudiocli nemotron-benchmark --subset test-other --max-files 50
```

