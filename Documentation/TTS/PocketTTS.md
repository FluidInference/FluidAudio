# PocketTTS Swift Inference

How the Swift code generates speech from text.

## Files

| File | Role |
|------|------|
| `PocketTtsManager.swift` | Public API — `initialize()`, `synthesize()`, `synthesizeToFile()` |
| `PocketTtsModelStore.swift` | Loads and stores the 4 CoreML models + constants + voice data |
| `PocketTtsSynthesizer.swift` | Main synthesis loop — chunking, prefill, generation, output |
| `PocketTtsSynthesizer+KVCache.swift` | KV cache state, `prefillKVCache()`, `runCondStep()`, `runFlowLMStep()` |
| `PocketTtsSynthesizer+Flow.swift` | Flow decoder loop, `denormalize()`, `quantize()`, SeededRNG |
| `PocketTtsSynthesizer+Mimi.swift` | Mimi decoder state, `runMimiDecoder()`, `loadMimiInitialState()` |
| `PocketTtsConstantsLoader.swift` | Loads binary constants (embeddings, tokenizer, quantizer weights) |
| `PocketTtsConstants.swift` | All numeric constants (dimensions, thresholds, etc.) |

## Call Flow

```
PocketTtsManager.synthesize(text:)
  |
  v
PocketTtsSynthesizer.synthesize(text:voice:temperature:)
  |
  |-- chunkText()              split text into <=50 token chunks
  |-- loadMimiInitialState()   load 23 streaming state tensors from disk
  |
  |-- FOR EACH CHUNK:
  |     |
  |     |-- tokenizer.encode()     SentencePiece text → token IDs
  |     |-- embedTokens()          table lookup: token ID → [1024] vector
  |     |-- prefillKVCache()       feed 125 voice + N text tokens through cond_step
  |     |     |
  |     |     |-- emptyKVCacheState()   fresh cache (6 layers × [2,1,512,16,64])
  |     |     |-- runCondStep() × ~141  one token per call, updates cache
  |     |
  |     |-- GENERATE LOOP (until EOS or max frames):
  |     |     |
  |     |     |-- runFlowLMStep()       → transformer_out [1,1024] + eos_logit
  |     |     |-- flowDecode()          → 32-dim latent
  |     |     |     |-- randn(32) * sqrt(temperature)
  |     |     |     |-- runFlowDecoderStep() × 8 Euler steps
  |     |     |     |-- latent += velocity * dt each step
  |     |     |
  |     |     |-- denormalize()         latent * std + mean
  |     |     |-- quantize()            matmul [32] × [32,512] → [512]
  |     |     |-- runMimiDecoder()      [512] → 1920 audio samples
  |     |     |     updates 23 streaming state tensors
  |     |     |
  |     |     |-- createSequenceFromLatent()  feed latent back for next frame
  |
  |-- concatenate all frames
  |-- applyTtsPostProcessing() (optional de-essing)
  |-- AudioWAV.data()          wrap in WAV header (24kHz mono)
```

## Key State

### KV Cache (`KVCacheState`)
- 6 cache tensors `[2, 1, 512, 16, 64]` + 6 position counters
- Written during prefill (voice + text tokens)
- Read and extended during generation (one position per frame)
- **Reset per chunk** — each chunk gets a fresh cache

### Mimi State (`MimiState`)
- 23 tensors: convolution history, attention caches, overlap-add buffers
- Loaded once from `mimi_init_state/*.bin` files via `manifest.json`
- Updated after every `runMimiDecoder()` call — outputs feed back as next input
- **Continuous across chunks** — never reset, keeps audio seamless

## Text Chunking

Long text is split into chunks of <=50 tokens to fit the KV cache (512 positions, minus ~125 voice + ~25 overhead).

Splitting priority:
1. Sentence boundaries (`.!?`)
2. Clause boundaries (`,;:`)
3. Word boundaries (fallback)

`normalizeText()` also capitalizes, adds terminal punctuation, and pads short text with leading spaces for better prosody.

## EOS Detection

`runFlowLMStep()` returns an `eos_logit`. When it exceeds `-4.0`, the code generates a few extra frames (3 for short text, 1 for long) then stops.

## CoreML Details

- All 4 models loaded with `.cpuAndGPU` compute units (ANE float16 causes artifacts in Mimi state feedback)
- Models compiled from `.mlpackage` → `.mlmodelc` on first load, cached on disk
- `PocketTtsModelStore` is an actor — thread-safe access to loaded models
- Voice data cached per voice name to avoid reloading

## Usage

```swift
import FluidAudioTTS

let manager = PocketTtsManager()
try await manager.initialize()

let audioData = try await manager.synthesize(text: "Hello, world!")

try await manager.synthesizeToFile(
    text: "Hello, world!",
    outputURL: URL(fileURLWithPath: "/tmp/output.wav")
)
```

## License

CC-BY-4.0, inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts).
