# Qwen3-ForcedAligner

> **Beta**
>
> This implementation is under active development. Timestamp accuracy may vary compared to the original PyTorch model due to int8 quantization. Tested with TTS-generated audio; real-world speech testing is ongoing.

Per-word timestamp alignment using [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) converted to CoreML. Given audio and a known transcript, produces `(startMs, endMs)` for each word.

## Model

**CoreML Model**: [alexwengg/Qwen3-ForcedAligner-0.6B-Coreml](https://huggingface.co/alexwengg/Qwen3-ForcedAligner-0.6B-Coreml) (int8 quantized)

## Architecture

Qwen3-ForcedAligner uses a 3-model pipeline in a single **non-autoregressive** prefill pass (no KV-cache, no decode loop):

```
Audio (16kHz mono) -> Mel Spectrogram -> Audio Encoder -\
                                                         Merge -> Decoder + LM Head -> Timestamps
Text -> BPE Tokenizer -> Embedding ----------------------/
```

| Model | Shape | Quantization |
|-------|-------|-------------|
| Audio Encoder | `[1, 128, 100]` mel -> `[1, N, 1024]` features | f32 |
| Embedding | `[1, seq_len]` token IDs -> `[1, seq_len, 1024]` | int8 |
| Decoder + LM Head | `[1, 1024, 1024]` hidden -> `[1, 1024, 5000]` logits | int8 |

### How It Works

1. Compute Slaney-scale mel spectrogram (128 bins, hop=160, 30s padding)
2. Encode audio in 100-frame chunks
3. Tokenize text with BPE, insert `<timestamp>` delimiters between words
4. Run embedding model on token IDs
5. Merge audio features into `<audio_pad>` positions
6. Pad to fixed sequence length (1024)
7. Compute interleaved MRoPE position embeddings
8. Run fused decoder + LM head (single forward pass)
9. Argmax over 5000-class timestamp vocabulary (each class = 80ms)
10. Fix non-monotonic timestamps via Longest Increasing Subsequence
11. Parse into per-word `(startMs, endMs)` alignments

## Usage

### CLI

```bash
# Align audio with transcript (auto-downloads models)
swift run fluidaudiocli align audio.wav --text "hello world how are you"

# Use local models
swift run fluidaudiocli align audio.wav --text "transcript" --model-dir /path/to/models
```

### Swift API

```swift
import FluidAudio

let manager = ForcedAlignerManager()

// Download and load models (auto-downloads if needed)
try await manager.downloadAndLoadModels()

// Align audio against transcript
let result = try await manager.align(
    audioSamples: samples,  // 16kHz mono Float32
    text: "hello world how are you today"
)

for word in result.alignments {
    print("\(word.word): \(word.startMs) - \(word.endMs) ms")
}
// hello: 0.0 - 240.0 ms
// world: 240.0 - 560.0 ms
// ...
```

### Using with Qwen3-ASR

ForcedAligner pairs naturally with Qwen3-ASR — use ASR to get the transcript, then ForcedAligner to get per-word timestamps:

```swift
let asrManager = Qwen3AsrManager()
try await asrManager.loadModels(from: asrModelDir)
let transcript = try await asrManager.transcribe(audioSamples: samples)

let alignerManager = ForcedAlignerManager()
try await alignerManager.downloadAndLoadModels()
let result = try await alignerManager.align(audioSamples: samples, text: transcript)
```

## Test Results

| Audio Duration | Text | Words | Timestamp Range |
|---------------|------|-------|-----------------|
| 1.39s | "hello world how are you today" | 6 | 0 - 1280ms |
| 2.36s | "the quick brown fox jumps over the lazy dog" | 9 | 0 - 2320ms |
| 2.93s | "artificial intelligence is transforming the world rapidly" | 7 | 0 - 2880ms |
| 4.77s | "I have a dream that one day this nation..." | 21 | 0 - 4720ms |

## Differences from Qwen3-ASR

| | Qwen3-ASR | ForcedAligner |
|---|-----------|---------------|
| Task | Speech-to-text | Text-to-timestamps |
| Input | Audio only | Audio + transcript |
| Decoding | Autoregressive (KV cache) | Non-autoregressive (single pass) |
| Output | Text tokens (full vocab) | Timestamp IDs (5000 classes) |
| Sequence length | Variable | Fixed 1024 |
| Decoder state | Stateful (MLState) | Stateless |

## Limitations

- **Fixed sequence length**: Total sequence (audio features + text tokens) must fit within 1024 tokens. This limits audio duration and transcript length.
- **int8 quantization**: Minor timestamp differences compared to f32 Python reference (~80-160ms on some words).
- **English-focused testing**: Only tested with English text so far. The underlying model supports multiple languages.
- **Inference speed**: First run includes CoreML model compilation (~20-45s). Subsequent runs are faster but still compute-heavy due to the 28-layer decoder.

## Files

| Component | Description |
|-----------|-------------|
| `forced_aligner_audio_encoder.mlmodelc` | Mel spectrogram to 1024-dim audio features |
| `forced_aligner_embedding.mlmodelc` | Token IDs to 1024-dim text embeddings |
| `forced_aligner_decoder_with_lm_head.mlmodelc` | Fused 28-layer decoder + LM head |
| `vocab.json` | BPE tokenizer vocabulary |
| `merges.txt` | BPE merge rules |
