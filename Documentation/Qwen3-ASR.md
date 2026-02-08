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

Hardware: Apple M2, 2022, macOS 26

### LibriSpeech test-clean (English, 2620 files)

| Metric | Value |
|--------|-------|
| WER (Avg) | 4.4% |
| WER (Median) | 0.0% |
| RTFx | 2.8x |
| Per-token | ~75ms |

### AISHELL-1 (Chinese, 6920 files, 9.7h audio)

| Metric | Value |
|--------|-------|
| CER (Avg) | 6.6% |
| WER (Avg) | 10.3% |
| Median RTFx | 4.6x |
| Overall RTFx | 3.8x |

### Running Benchmarks

```bash
# English (LibriSpeech test-clean)
swift run -c release fluidaudiocli qwen3-benchmark --subset test-clean

# Chinese (AISHELL-1)
swift run -c release fluidaudiocli qwen3-benchmark --dataset aishell
```

## Evaluation Methodology

### English (WER)

Word Error Rate calculated using standard whitespace tokenization with text normalization (lowercase, punctuation removal).

### Chinese (CER/WER)

- **CER** (Character Error Rate) is the primary metric for Chinese ASR
- **WER** uses Apple's `NLTokenizer` for word segmentation
- Per [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1): *"We use CER for character-based languages (e.g., Mandarin Chinese, Cantonese, and Korean) and WER for word-delimited languages"*

We were unable to verify the exact tokenization methodology used in official Qwen3-ASR evaluation.

## Comparison with Official Results

| Dataset | Official | CoreML |
|---------|----------|--------|
| LibriSpeech test-clean | 2.11% WER | 4.4% WER |
| AISHELL-2 | 3.15% | - |
| AISHELL-1 | - | 6.6% CER |

The CoreML conversion shows higher error rates, likely due to precision differences during model conversion.

**Sources:**
- [Qwen3-ASR-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)

## Limitations

- **Speed**: Autoregressive decoding (~60-80ms/token) is slower than parallel decoders like Parakeet TDT
- **Accuracy**: CoreML conversion introduces some accuracy loss vs PyTorch original
- **Memory**: Requires ~1.5GB RAM for inference

## Files

| Component | Description |
|-----------|-------------|
| `qwen3_asr_audio_encoder.mlmodelc` | Audio feature extraction |
| `qwen3_asr_decoder_stateful.mlmodelc` | Autoregressive decoder with KV-cache |
| `qwen3_asr_embeddings.bin` | Token embedding weights (float16) |
| `vocab.json` | Tokenizer vocabulary (151,936 tokens) |
