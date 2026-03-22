# Qwen3-TTS: Multilingual Text-to-Speech (Beta)

## Overview

Qwen3-TTS is an LLM-based multilingual TTS backend built on the Qwen3 language model. It supports 10 languages including English and Chinese, producing natural speech at 24 kHz via a 4-stage CoreML pipeline.

> **Beta.** Qwen3-TTS does not yet include a built-in text tokenizer. Input must be pre-tokenized externally (e.g., via the Python `qwen-tts` package).

## Quick Start

### CLI

```bash
# English
swift run fluidaudiocli tts --backend qwen3 \
  "Hello world, this is a test of the text to speech system." \
  --output hello.wav

# Chinese
swift run fluidaudiocli tts --backend qwen3 \
  "你好世界，这是一个文字转语音系统的测试。" \
  --output chinese.wav
```

Models are auto-downloaded from HuggingFace on first run.

### Swift

```swift
import FluidAudio

let manager = Qwen3TtsManager()
try await manager.loadIfNeeded()

// Token IDs must be generated externally (e.g., via Python qwen-tts processor)
let tokenIds = [9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 4686, 1331, 39586, 1849, 13]
let result = try await manager.synthesize(text: "Hello world", tokenIds: tokenIds)

let outputURL = URL(fileURLWithPath: "/tmp/qwen3_output.wav")
try result.audio.write(to: outputURL)
```

## Pipeline

```
text tokens ──► Prefill ──► LM Decode Loop ──► Audio Decoder ──► WAV
                  │              │
                  │         ┌────┴────┐
                  │         │ CB0     │ (greedy with repetition penalty)
                  │         │ CB1-15  │ (code predictor, temperature sampling)
                  │         └─────────┘
                  │
             role_ids + text_ids + speaker_embed + TTS special tokens
```

### Stages

| Stage | Model | Description |
|-------|-------|-------------|
| 1. Prefill | `qwen3_tts_lm_prefill_v9` | Encodes text context → initial logits, KV cache, past hidden state |
| 2. LM Decode | `qwen3_tts_lm_decode_v10` | Autoregressive loop generating CB0 tokens (main codebook) |
| 3. Code Predictor | `qwen3_tts_cp_prefill` + `qwen3_tts_cp_decode` | Generates CB1-15 from past hidden + CB0 per step |
| 4. Audio Decoder | `qwen3_tts_decoder_10s` | Converts 16-layer codebook frames to 24 kHz waveform |

## Files

| File | Role |
|------|------|
| `Qwen3TtsManager.swift` | Public API — `loadIfNeeded()`, `synthesize()` |
| `Qwen3TtsSynthesizer.swift` | Core inference pipeline — prefill, decode loop, code predictor, audio decoder |
| `Qwen3TtsModelStore.swift` | Loads and stores 5 CoreML models + embeddings from `.npy` files |
| `Qwen3TtsConstants.swift` | Model dimensions, special token IDs, sampling parameters |
| `Qwen3TtsResourceDownloader.swift` | Auto-downloads models from HuggingFace |

## Sampling

CB0 (main language model) uses greedy decoding with logit processors:
- Repetition penalty (1.05) on all previously generated CB0 tokens
- Token suppression: tokens 2048-3071 masked except EOS (2150)
- `min_new_tokens`: EOS suppressed for first 2 steps

CB1-15 (code predictor) uses temperature sampling:
- Temperature: 0.9
- Top-K: 50
- Greedy code prediction produces silent/broken audio; temperature sampling is required.

## Languages

Qwen3-TTS supports 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

Language IDs are embedded via the codec embedding table during prefill (e.g., English = 2050, Chinese = 2055).

## Limitations

- **No built-in tokenizer.** Text must be pre-tokenized using the Qwen3 tokenizer externally. The CLI currently supports two hardcoded test sentences.
- **Max 128 text tokens.** Longer inputs are truncated.
- **Max 125 codec frames.** Generates up to ~10 seconds of audio per call.
- **CPU+GPU compute.** Models run on `cpuAndGPU` compute units (no ANE optimization yet).

## Model Source

Models are hosted at [alexwengg/qwen3-tts-coreml](https://huggingface.co/alexwengg/qwen3-tts-coreml) on HuggingFace.

Based on [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
