# Parakeet TDT-CTC-110M

FluidAudio supports NVIDIA's Parakeet TDT-CTC-110M hybrid model for fast, accurate batch transcription on Apple devices.

## Overview

Parakeet TDT-CTC-110M is a hybrid Token-and-Duration Transducer (TDT) model with CTC-constrained decoding. The CoreML conversion provides:

- **Fused preprocessor+encoder** for reduced memory footprint and faster loading
- **96.5x real-time factor** on Apple Silicon (M2)
- **3.01% WER** on LibriSpeech test-clean
- **iOS compatible** with full ANE optimization
- **Stateless processing** - no encoder state carryover needed

## Benchmark Results

Tested on Apple M2 with LibriSpeech test-clean (full dataset):

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| **Average WER** | **3.01%** |
| **Median WER** | **0.0%** |
| Average CER | 1.09% |
| **Overall RTFx** | **96.5x** |
| **Median RTFx** | **86.4x** |
| Processing time | 201.5s (~3.4 minutes) |
| Audio duration | 19,452.5s (~5.4 hours) |

**Performance:** 1 hour of audio transcribes in **37 seconds** on M2 Mac.

## Quick Start

### Basic Usage

```swift
import FluidAudio

// Create manager
let manager = AsrManager()

// Load models (auto-downloads from HuggingFace if needed)
let models = try await AsrModels.downloadAndLoad(version: .tdtCtc110m)
try await manager.initialize(models: models)

// Transcribe audio file
let url = URL(fileURLWithPath: "audio.wav")
let result = try await manager.transcribe(url)
print("Transcript: \(result.text)")

// Or transcribe audio samples directly
let samples: [Float] = ... // 16kHz mono audio
let result = try await manager.transcribe(samples)
print("Transcript: \(result.text)")
```

### Streaming Processing

```swift
import FluidAudio

let manager = AsrManager()
let models = try await AsrModels.downloadAndLoad(version: .tdtCtc110m)
try await manager.initialize(models: models)

// Process live microphone audio
for audioChunk in microphoneStream {
    let result = try await manager.transcribe(audioChunk, source: .microphone)
    print("Partial: \(result.text)")
}

// Reset state between utterances
manager.resetState()
```

### Manual Model Loading

```swift
// Specify custom cache directory
let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("MyAppModels")

let models = try await AsrModels.downloadAndLoad(
    to: cacheDir,
    version: .tdtCtc110m
)
try await manager.initialize(models: models)
```

## Architecture

### Model Overview

TDT-CTC-110M uses a hybrid architecture combining:
- **TDT (Token-and-Duration Transducer)** for accurate token prediction
- **CTC (Connectionist Temporal Classification)** for beam search constraints
- **Fused preprocessor+encoder** for efficiency

**Key differences from v2/v3:**
- **1 decoder LSTM layer** (vs 2 in v2/v3)
- **110M parameters** (vs 600M in v2/v3)
- **Fused preprocessor+encoder** (single CoreML model)
- **Faster loading** (19.9s cold start vs 30s+ for v3)

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDT-CTC-110M PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

1. AUDIO CHUNKING
   Full audio → overlapping chunks (~14.96s chunk, 2.0s overlap)

2. FUSED PREPROCESSOR+ENCODER (per chunk)
   audio [239,360 samples] → encoded [1, 931, 512]
   - Preprocessor: audio → mel spectrogram (80 bins)
   - Encoder: mel → acoustic features (512-dim)
   - Both fused in single CoreML model for efficiency

3. DECODER (prediction network, 1 LSTM layer)
   previous_token + hidden_state → decoder_out [1, 1, 512]
   - Maintains LSTM state: hidden [1, 1, 512], cell [1, 1, 512]
   - Initial token: blank (1023)
   - State resets per chunk (stateless processing)

4. JOINT NETWORK
   encoder_step [512] + decoder_out [512] → logits [1024]
   - Combines acoustic and linguistic features
   - Outputs token probabilities

5. TDT DECODER (beam search with CTC)
   logits → tokens with durations
   - Beam size: 10
   - CTC-constrained beam search
   - Outputs: tokens, durations, scores

6. DETOKENIZATION
   tokens → text
   - Uses parakeet_vocab.json (1024 tokens)
   - Handles BPE subword units
```

### Chunk Processing Strategy

**Stateless per-chunk decoding:**
- Each chunk processed independently
- Decoder state resets at chunk boundaries
- No encoder state carryover needed
- Simpler than streaming models (Nemotron, Parakeet EOU)

**Chunking parameters:**
```swift
let chunkSamples = 239_360        // ~14.96s at 16kHz
let overlapSamples = 32_000       // 2.0s overlap
let samplesPerWindow = 16         // 1ms per window
```

**Overlap handling:**
- 2s overlap between chunks reduces boundary errors
- Overlapping regions discarded during final merge
- Ensures smooth transcription across chunk boundaries

## Code Workflow

### 1. Model Loading (`AsrModels.downloadAndLoad`)

```swift
// Sources/FluidAudio/ASR/Parakeet/AsrModels.swift
public static func downloadAndLoad(version: AsrModelVersion) async throws -> AsrModels

Flow:
1. Check cache directory for models
2. Download from HuggingFace if missing:
   - Preprocessor.mlmodelc (fused with encoder for tdtCtc110m)
   - Decoder.mlmodelc
   - JointDecision.mlmodelc
   - parakeet_vocab.json
3. Compile .mlpackage → .mlmodelc if needed
4. Load CoreML models into memory
5. Return AsrModels struct
```

### 2. Manager Initialization (`AsrManager.initialize`)

```swift
// Sources/FluidAudio/ASR/Parakeet/AsrManager.swift
public func initialize(models: AsrModels) async throws

Flow:
1. Store models reference
2. Load CoreML models:
   - preprocessorModel (fused preprocessor+encoder)
   - decoderModel (prediction network, 1 layer)
   - jointModel (joiner network)
3. Initialize decoder states:
   - microphoneDecoderState (1 layer for tdtCtc110m)
   - systemDecoderState (1 layer for tdtCtc110m)
4. Load vocabulary from parakeet_vocab.json
5. Initialize TDT decoder with beam_size=10
```

### 3. Transcription (`AsrManager.transcribe`)

```swift
// Sources/FluidAudio/ASR/Parakeet/AsrManager.swift
public func transcribe(_ samples: [Float], source: AudioSource = .file) async throws -> ASRResult

Flow:
1. Select decoder state based on source:
   - .microphone → microphoneDecoderState
   - .systemAudio → systemDecoderState
   - .file → fresh state per call

2. Process via ChunkProcessor:
   → ChunkProcessor.processAudioChunks()
```

### 4. Chunk Processing (`ChunkProcessor.processAudioChunks`)

```swift
// Sources/FluidAudio/ASR/Parakeet/ChunkProcessor.swift
static func processAudioChunks() async throws -> ASRResult

Flow for each chunk:
1. Extract chunk samples with overlap
2. Run fused preprocessor+encoder:
   samples → encoded frames [1, 931, 512]
3. Initialize chunk decoder state (1 layer)
4. Run TDT beam search:
   - For each encoder frame:
     a. Get decoder prediction
     b. Run joint network
     c. Compute logits
   - Beam search with CTC constraint
   - Output: tokens, durations, scores
5. Store TokenWindow results
6. Move to next chunk

After all chunks:
7. Merge overlapping chunks (discard overlap regions)
8. Detokenize merged tokens → text
9. Return ASRResult
```

### 5. TDT Beam Search (`TdtDecoder.decode`)

```swift
// Sources/FluidAudio/ASR/Parakeet/Decoder/TdtDecoderV3.swift
func decode(encodedAudio: MLMultiArray, decoderState: inout TdtDecoderState) throws -> [TokenWindow]

Flow:
1. Initialize beam with blank token (1023)
2. For each encoder frame (931 frames):
   a. Expand beam:
      - Run decoder LSTM for each hypothesis
      - Run joint network: encoder + decoder → logits
   b. Get top-k tokens per hypothesis
   c. Score new hypotheses
   d. Prune beam to size 10
3. Select best hypothesis
4. Extract tokens with durations
5. Return TokenWindow array
```

### 6. Detokenization (`Detokenizer.detokenize`)

```swift
// Sources/FluidAudio/ASR/Detokenizer.swift
static func detokenize(tokens: [Int], vocabulary: [String]) -> String

Flow:
1. Map token IDs → vocabulary strings
2. Concatenate subword units
3. Handle BPE merge rules
4. Return final text
```

## Model Files

### Directory Structure

```
~/.cache/huggingface/hub/models--FluidInference--parakeet-tdt-ctc-110m-coreml/
└── snapshots/{commit_hash}/
    ├── Preprocessor.mlmodelc/       # Fused preprocessor+encoder (~390MB)
    ├── Decoder.mlmodelc/            # Prediction network, 1 layer (~12MB)
    ├── JointDecision.mlmodelc/      # Joiner network (~5MB)
    └── parakeet_vocab.json          # 1024 BPE tokens
```

**Total size:** ~407MB (vs ~700MB for v3)

### Model Inputs/Outputs

**Preprocessor (fused with encoder):**
```
Input:  samples [239,360] (14.96s @ 16kHz)
Output: encoded [1, 931, 512] (acoustic features)
```

**Decoder:**
```
Inputs:
  - tokens [1, 1] (previous token)
  - hidden_state [1, 1, 512]
  - cell_state [1, 1, 512]
Outputs:
  - decoder_out [1, 1, 512]
  - hidden_state_out [1, 1, 512]
  - cell_state_out [1, 1, 512]
```

**Joint:**
```
Inputs:
  - encoder_frame [1, 1, 512]
  - decoder_out [1, 1, 512]
Output:
  - logits [1, 1, 1024]
```

## Configuration

### Decoder Layer Count

TDT-CTC-110M uses **1 decoder LSTM layer** (vs 2 in v2/v3):

```swift
// Sources/FluidAudio/ASR/Parakeet/AsrModels.swift
public var decoderLayers: Int {
    switch self {
    case .tdtCtc110m: return 1
    default: return 2  // v2, v3
    }
}
```

This reduces model size and improves inference speed while maintaining competitive accuracy.

### TDT Decoder Settings

```swift
// Sources/FluidAudio/ASR/Parakeet/Decoder/TdtDecoderV3.swift
let beamSize = 10                    // Beam search width
let blankId = 1023                   // Blank token ID
let encoderHiddenSize = 512          // Encoder output dim
let decoderHiddenSize = 512          // Decoder hidden dim
```

## CLI Benchmark

Run benchmarks using the FluidAudio CLI:

```bash
# Build release
swift build -c release

# Full test-clean benchmark (2,620 files)
swift run -c release fluidaudiocli asr-benchmark \
    --subset test-clean \
    --model-version tdt-ctc-110m

# Benchmark with limited files
swift run -c release fluidaudiocli asr-benchmark \
    --subset test-clean \
    --model-version tdt-ctc-110m \
    --max-files 100

# Benchmark on test-other subset
swift run -c release fluidaudiocli asr-benchmark \
    --subset test-other \
    --model-version tdt-ctc-110m \
    --max-files 50

# Single file test
swift run -c release fluidaudiocli asr-benchmark \
    --single-file 1089-134686-0000 \
    --model-version tdt-ctc-110m

# Output to custom JSON file
swift run -c release fluidaudiocli asr-benchmark \
    --subset test-clean \
    --model-version tdt-ctc-110m \
    --output my_results.json
```

Results saved to `asr_benchmark_results.json` with detailed per-file metrics.

## iOS Integration

### iOS Test App

See `TdtCtc110mTestApp/` for a complete iOS example:

```swift
import SwiftUI
import FluidAudio

struct ContentView: View {
    @State private var transcript: String = ""
    @State private var isTesting: Bool = false

    func runTest() async {
        // Auto-download models on device
        let models = try await AsrModels.downloadAndLoad(
            to: nil,  // Uses default cache
            version: .tdtCtc110m
        )

        // Initialize manager
        let manager = AsrManager()
        try await manager.initialize(models: models)

        // Load test audio
        let audioSamples: [Float] = ... // Load from bundle or record

        // Transcribe
        let result = try await manager.transcribe(audioSamples)
        transcript = result.text
    }
}
```

### Model Loading on iOS

Models auto-download to:
```
~/Library/Caches/huggingface/hub/models--FluidInference--parakeet-tdt-ctc-110m-coreml/
```

**First load:** ~20 seconds (model download + ANE compilation)
**Subsequent loads:** ~1 second (ANE cache hit)

### iOS Performance

Tested on iPhone (iOS 17+):
- **Cold start:** 19.9s (ANE compilation)
- **Warm start:** 764ms (ANE cache hit)
- **Inference:** Similar RTFx to Mac (70-100x on modern devices)
- **Memory:** ~400MB model + ~50MB runtime

## Comparison: TDT-CTC-110M vs v3

| Feature | TDT-CTC-110M | Parakeet TDT v3 |
|---------|--------------|-----------------|
| Parameters | 110M | 600M |
| Model size | ~407MB | ~700MB |
| Decoder layers | 1 | 2 |
| Architecture | Fused preprocessor+encoder | Separate models |
| Cold start | 19.9s | 30s+ |
| WER (test-clean) | 3.01% | ~2-3% |
| RTFx (M2) | 96.5x | ~80x |
| Languages | English | 25 European |
| iOS compatible | ✅ Yes | ✅ Yes |

**When to use TDT-CTC-110M:**
- English-only applications
- Memory-constrained devices
- Faster model loading preferred
- Competitive accuracy sufficient (3% WER)

**When to use v3:**
- Multilingual support needed
- Highest accuracy required
- Extra model size acceptable

## Standalone CTC Head for Custom Vocabulary (Beta)

The TDT-CTC-110M hybrid model shares one FastConformer encoder between its TDT and CTC decoder heads. FluidAudio exploits this by exporting the CTC decoder head as a standalone 1MB CoreML model (`CtcHead.mlmodelc`) that runs on the existing TDT encoder output, enabling custom vocabulary keyword spotting without a second encoder pass.

### How It Works

```
TDT Preprocessor (fused encoder)
        │
        ▼
encoder output [1, 512, T]
        │
   ┌────┴────┐
   │         │
   ▼         ▼
TDT Decoder  CtcHead (1MB, beta)
   │         │
   ▼         ▼
transcript   ctc_logits [1, T, 1025]
                  │
                  ▼
         Keyword Spotter / VocabularyRescorer
```

The CTC head is a single linear projection (512 → 1025) that maps the 512-dimensional encoder features to log-probabilities over 1024 BPE tokens + 1 blank token.

### Performance

Benchmarked on 772 earnings call files (Earnings22-KWS):

| Approach | Model Size | Dict Recall | RTFx |
|----------|-----------|-------------|------|
| Separate CTC encoder | 97.5 MB | 99.4% | 25.98x |
| **Standalone CTC head** | **1 MB** | **99.4%** | **70.29x** |

The standalone CTC head achieves identical keyword detection quality at 2.7x the speed, using 97x less model weight.

### Loading

The CTC head model auto-downloads from [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) when loading the TDT-CTC-110M model. It also supports manual placement in the TDT model directory.

Two loading paths are supported:
1. **Local (v1):** Place `CtcHead.mlmodelc` in the TDT model directory (`parakeet-tdt-ctc-110m/`)
2. **Auto-download (v2):** Automatically downloaded from the `parakeet-ctc-110m-coreml` HuggingFace repo

```swift
// CTC head loads automatically with TDT-CTC-110M models
let models = try await AsrModels.downloadAndLoad(version: .tdtCtc110m)
// models.ctcHead is non-nil when CtcHead.mlmodelc is available
```

### Conversion

The CTC head is exported using the conversion script in the mobius repo:

```bash
cd mobius/models/stt/parakeet-tdt-ctc-110m/coreml/
uv run python export-ctc-head.py --output-dir ./ctc-head-build
xcrun coremlcompiler compile ctc-head-build/CtcHead.mlpackage ctc-head-build/
```

See [mobius PR #36](https://github.com/FluidInference/mobius/pull/36) for the conversion script.

### Status

This feature is **beta**. The CTC head produces identical keyword detection results to the separate CTC encoder, but the auto-download pathway and integration are new. See [#435](https://github.com/FluidInference/FluidAudio/issues/435) and [PR #450](https://github.com/FluidInference/FluidAudio/pull/450) for details.

## Resources

- **Model:** [FluidInference/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-tdt-ctc-110m-coreml)
- **CTC Head model:** [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) (includes CtcHead.mlmodelc)
- **Benchmark results:** See `benchmarks.md`
- **PR:** [#433 - Add TDT-CTC-110M support](https://github.com/FluidInference/FluidAudio/pull/433)
- **CTC Head PR:** [#450 - Add standalone CTC head for custom vocabulary](https://github.com/FluidInference/FluidAudio/pull/450)
- **Original NVIDIA model:** [nvidia/parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b)
