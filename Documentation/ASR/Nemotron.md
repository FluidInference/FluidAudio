# Nemotron Speech Streaming 0.6B

FluidAudio supports NVIDIA's Nemotron Speech Streaming model for real-time streaming ASR on Apple devices.

## Overview

Nemotron Speech Streaming 0.6B is a FastConformer RNNT model optimized for streaming speech recognition. The CoreML conversion provides:

- **Multiple chunk sizes** for latency/accuracy trade-offs
- **Int8 quantized encoder** (~564MB, 4x smaller than float32)
- **Streaming inference** with encoder cache for continuous audio

## Benchmark Results

Tested on Apple M2 with LibriSpeech test-clean:

| Chunk Size | WER | RTFx | Files |
|------------|-----|------|-------|
| 1120ms | 1.99% | 9.6x | 100 |
| 560ms | 2.12% | 8.5x | 100 |
| 160ms | ~10% | 3.5x | 20 |
| 80ms | ~60% | 1.9x | 20 |

160ms and 80ms were only tested on 20 files.

## Quick Start

### Basic Usage

```swift
import FluidAudio

// Create manager
let manager = NemotronStreamingAsrManager()

// Load models (defaults to 1120ms chunk size)
let modelDir = URL(fileURLWithPath: "~/.cache/fluidaudio/models/nemotron-streaming/1120ms")
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
try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir)

// Models will be at: modelsBaseDir/nemotron-streaming/560ms/
```

### Automatic Model Download

FluidAudio can automatically download models from HuggingFace:

```swift
let chunkSize: NemotronChunkSize = .ms560
let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".cache/fluidaudio/models")

// Download if not cached
try await DownloadUtils.downloadRepo(chunkSize.repo, to: modelsBaseDir)

// Load from cache
let modelDir = modelsBaseDir.appendingPathComponent(chunkSize.repo.folderName)
try await manager.loadModels(modelDir: modelDir)
```

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
| 1120ms | 112 | 9 | 121 | 17,920 |
| 560ms | 56 | 9 | 65 | 8,960 |
| 160ms | 16 | 9 | 25 | 2,560 |
| 80ms | 8 | 9 | 17 | 1,280 |

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
│   └── encoder_int8.mlmodelc    # ~564MB (int8 quantized)
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

