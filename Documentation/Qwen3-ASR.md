# Qwen3-ASR

Encoder-decoder automatic speech recognition using [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) converted to CoreML.

## Model

**CoreML Model**: [FluidInference/qwen3-asr-0.6b-coreml](https://huggingface.co/FluidInference/qwen3-asr-0.6b-coreml)

Only the **f32** variant is recommended. See [Why not int8?](#why-not-int8) below.

## Architecture

Qwen3-ASR uses an encoder-decoder architecture with autoregressive text generation:

1. **Audio Encoder**: Processes 1-second audio windows (100 mel frames at 10ms hop)
2. **Decoder**: 28-layer transformer with KV-cache for efficient token generation
3. **Generation**: Autoregressive decoding at ~60-80ms per token

## Supported Languages

Qwen3-ASR supports 30 languages with automatic language detection:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| zh | Chinese | en | English | yue | Cantonese |
| ja | Japanese | ko | Korean | vi | Vietnamese |
| th | Thai | id | Indonesian | ms | Malay |
| hi | Hindi | ar | Arabic | tr | Turkish |
| ru | Russian | de | German | fr | French |
| es | Spanish | pt | Portuguese | it | Italian |
| nl | Dutch | pl | Polish | sv | Swedish |
| da | Danish | fi | Finnish | cs | Czech |
| fil | Filipino | fa | Persian | el | Greek |
| hu | Hungarian | mk | Macedonian | ro | Romanian |

### Usage with Language Hint

```swift
// Auto-detect language (default)
let text = try await manager.transcribe(audioSamples: samples)

// Specify language for better accuracy
let text = try await manager.transcribe(audioSamples: samples, language: .chinese)
let text = try await manager.transcribe(audioSamples: samples, language: .japanese)
```

## Usage

### CLI

```bash
# Transcribe a file (auto-detect language)
swift run -c release fluidaudiocli qwen3-transcribe audio.wav

# Transcribe with language hint
swift run -c release fluidaudiocli qwen3-transcribe audio.wav --language zh

# Transcribe with local model
swift run -c release fluidaudiocli qwen3-transcribe audio.wav --model-dir /path/to/model
```

### Swift API

```swift
import FluidAudio

// Initialize manager
let manager = Qwen3AsrManager()

// Load models (auto-downloads if needed)
let modelDir = try await Qwen3AsrModels.download()
try await manager.loadModels(from: modelDir)

// Transcribe audio samples (16kHz mono Float32)
let text = try await manager.transcribe(audioSamples: samples)
```

## Benchmarks

See [Benchmarks.md](Benchmarks.md#qwen3-asr-experimental) for performance results on LibriSpeech and AISHELL-1.

## Files

| Component | Description |
|-----------|-------------|
| `qwen3_asr_audio_encoder.mlmodelc` | Audio feature extraction |
| `qwen3_asr_decoder_stateful.mlmodelc` | Autoregressive decoder with KV-cache |
| `qwen3_asr_embeddings.bin` | Token embedding weights (float16) |
| `vocab.json` | Tokenizer vocabulary (151,936 tokens) |

## Why not int8?

int8 quantization does not improve performance for Qwen3-ASR on Apple Silicon. In testing, int8 was actually **slower** (1.4x RTFx) than f32 (2.8x RTFx).

**Root cause:** Qwen3-ASR uses a 28-layer transformer decoder that runs **once per token** (autoregressive). Each forward pass requires dequantizing ~1GB of weights across all layers. This overhead multiplies with token count.

This differs from parallel decoders like Parakeet TDT, where:
- The decoder is small (~23MB) and runs once
- int8 dequantization cost is amortized across batch output

For autoregressive LLM-style decoders, fp16 compute precision (used by f32 models internally) provides the best speed/accuracy tradeoff on Apple Silicon.
