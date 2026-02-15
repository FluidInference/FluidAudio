# Qwen3-ForcedAligner-0.6B

Per-word timestamp alignment using a 3-model CoreML pipeline in a single non-autoregressive prefill pass.

## Architecture

```
Audio (16kHz mono) -> Mel Spectrogram -> Audio Encoder -> \
                                                           Merge -> Decoder + LM Head -> Argmax -> Timestamps
Text -> BPE Tokenizer -> Embedding ---------------------> /
```

Three CoreML models run sequentially:

| Model | Input | Output | Quantization |
|-------|-------|--------|-------------|
| `forced_aligner_audio_encoder` | `[1, 128, 100]` mel chunks | `[1, N, 1024]` audio features | f32 |
| `forced_aligner_embedding` | `[1, seq_len]` token IDs | `[1, seq_len, 1024]` embeddings | int8 |
| `forced_aligner_decoder_with_lm_head` | `[1, 1024, 1024]` hidden + cos/sin | `[1, 1024, 5000]` logits | int8 |

## Pipeline Steps

1. Compute Slaney-scale mel spectrogram (128 bins, hop=160, 30s padding, STFT center padding)
2. Encode audio in 100-frame chunks through the audio encoder
3. Tokenize text with BPE, insert `<timestamp>` delimiters between words
4. Run embedding model on token IDs
5. Merge: replace `<audio_pad>` positions with audio encoder features
6. Pad merged sequence to 1024 (fixed prefill length)
7. Compute interleaved MRoPE position embeddings (sections [24, 20, 20], theta=1M)
8. Run decoder + LM head (single forward pass, no KV cache)
9. Argmax over 5000-class timestamp vocabulary
10. Extract timestamps at `<timestamp>` positions (each class = 80ms)
11. Fix non-monotonic timestamps via Longest Increasing Subsequence
12. Parse into per-word `(startMs, endMs)` alignments

## Usage

### Public API

```swift
let manager = ForcedAlignerManager()
try await manager.downloadAndLoadModels()

let result = try await manager.align(
    audioSamples: samples,  // 16kHz mono Float32
    text: "hello world"
)

for word in result.alignments {
    print("\(word.word): \(word.startMs) - \(word.endMs) ms")
}
```

### CLI

```bash
fluidaudio align audio.wav --text "hello world how are you"
fluidaudio align speech.wav --text "transcript" --model-dir /path/to/models
```

## Key Differences from Qwen3-ASR

| | Qwen3-ASR | ForcedAligner |
|---|-----------|---------------|
| Decoding | Autoregressive (KV cache) | Non-autoregressive (single prefill) |
| Output | Full vocabulary tokens | 5000-class timestamp IDs |
| Seq length | Variable | Fixed 1024 |
| MRoPE | Concatenated halves | Interleaved sections |
| MLState | Stateful decoder | Stateless |
| Input | Audio only | Audio + text transcript |

## Files

| File | Purpose |
|------|---------|
| `ForcedAlignerConfig.swift` | Constants: dimensions, token IDs, seq length |
| `ForcedAlignerTypes.swift` | `WordAlignment`, `ForcedAlignmentResult`, error enum |
| `ForcedAlignerModels.swift` | Download from HuggingFace + CoreML model loading |
| `ForcedAlignerMelSpectrogram.swift` | Slaney-scale mel with STFT center padding |
| `ForcedAlignerMRoPE.swift` | Interleaved rotary position embeddings |
| `ForcedAlignerTokenizer.swift` | BPE tokenizer (vocab.json + merges.txt) |
| `ForcedAlignerInference.swift` | Full 12-step pipeline |
| `ForcedAlignerManager.swift` | Public actor API |

## Model Source

HuggingFace: [`alexwengg/Qwen3-ForcedAligner-0.6B-Coreml`](https://huggingface.co/alexwengg/Qwen3-ForcedAligner-0.6B-Coreml) (subfolder `qwen3-forced-aligner-coreml-int8`)
