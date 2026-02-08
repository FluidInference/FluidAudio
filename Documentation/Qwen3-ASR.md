# Qwen3-ASR

Encoder-decoder automatic speech recognition using [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) converted to CoreML.

## Model

**CoreML Model**: [FluidInference/qwen3-asr-0.6b-coreml](https://huggingface.co/FluidInference/qwen3-asr-0.6b-coreml)

| Variant | Size | Use Case |
|---------|------|----------|
| f32 | ~1.2GB | Default, higher precision |
| int8 | ~600MB | Smaller, slightly faster on CPU |

## Architecture

Qwen3-ASR uses an encoder-decoder architecture with autoregressive text generation:

1. **Audio Encoder**: Processes 1-second audio windows (100 mel frames at 10ms hop)
2. **Decoder**: 28-layer transformer with KV-cache for efficient token generation
3. **Generation**: Autoregressive decoding at ~60-80ms per token

## Usage

### CLI

```bash
# Transcribe a file
swift run -c release fluidaudiocli qwen3-transcribe audio.wav

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
