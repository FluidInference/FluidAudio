# PocketTTS Swift Inference

How the Swift code generates speech from text.

## Files

| File | Role |
|------|------|
| `PocketTtsManager.swift` | Public API — `initialize()`, `synthesize()`, `synthesizeToFile()`, `cloneVoice()` |
| `PocketTtsModelStore.swift` | Loads and stores the 4 CoreML models + constants + voice data |
| `PocketTtsVoiceCloner.swift` | Voice cloning — converts audio to voice conditioning embeddings |
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

## Voice Cloning

Clone any voice from a short audio sample (1-30 seconds) using the Mimi encoder model.

### How It Works

1. Audio is loaded and resampled to 24kHz mono using `AudioConverter`
2. The Mimi encoder converts audio to conditioning embeddings `[1, num_frames, 1024]`
3. Embeddings are used at their natural length (no padding) — the KV cache prefill processes the actual number of frames
4. The resulting `PocketTtsVoiceData` can be used directly for synthesis

Variable-length support is important: zero-padding shorter audio would corrupt voice conditioning by feeding meaningless vectors into the transformer.

### Voice Cloning API

```swift
// Clone from audio file (WAV, MP3, M4A, etc.)
let voiceData = try await manager.cloneVoice(from: audioURL)

// Clone from raw samples (24kHz mono Float32)
let voiceData = try await manager.cloneVoice(from: samples)

// Use cloned voice immediately (no file I/O needed)
let audio = try await manager.synthesize(text: "Hello!", voiceData: voiceData)

// Save for later use
try manager.saveClonedVoice(voiceData, to: outputURL)

// Load previously saved voice
let savedVoice = try manager.loadClonedVoice(from: savedVoiceURL)
let audio = try await manager.synthesize(text: "Hello!", voiceData: savedVoice)
```

### CLI Usage

```bash
# Clone voice and synthesize in one step
fluidaudio tts "Hello world" --backend pocket --clone-voice speaker.wav

# Clone, save for later, and synthesize
fluidaudio tts "Hello world" --backend pocket --clone-voice speaker.wav --save-voice my_voice.bin

# Use previously saved voice
fluidaudio tts "Hello world" --backend pocket --voice-file my_voice.bin
```

### Requirements

- Audio duration: 1-30 seconds (capped at 250 frames / ~20s to leave KV cache room)
- The `mimi_encoder.mlmodelc` model is downloaded automatically on first use
- Supports any audio format that AVFoundation can read

## Pipeline and Pronunciation Control

```
text → SentencePiece tokenizer → subword tokens → PocketTTS model → audio
                                                    ↑
                                          pronunciation decisions
                                          happen inside model weights
                                          (no external control)
```

Unlike Kokoro which uses espeak to convert text to IPA phonemes **before** the model, PocketTTS feeds raw text tokens directly into the neural network. The model learned text→pronunciation mappings during training — there is no phoneme stage to intercept.

### Feature Support

| Feature | Supported | Can We Add? | Why |
|---------|-----------|-------------|-----|
| SSML `<phoneme>` | No | No | No IPA layer — model has no phoneme vocabulary |
| Custom lexicon (word → IPA) | No | No | No phoneme stage to apply mappings |
| Markdown `[word](/ipa/)` | No | No | Same — no phoneme input |
| SSML `<sub>` (text substitution) | No | **Yes** | Text-level, can run before tokenizer |
| Text preprocessing (numbers, dates) | Minimal | **Yes** | Text-level, can run before tokenizer |

### What Can Be Added

Text-level preprocessing that runs **before** the SentencePiece tokenizer:

- **Number/date/currency expansion** — "123" → "one hundred twenty three"
- **`<sub>` substitution** — replace abbreviations with full text before tokenization
- **Phonetic spelling workarounds** — spelling out pronunciation ("NVIDIA" → "en-vidia"), though unreliable since the model may not pronounce phonetic spellings consistently

### What Cannot Be Added (Without Retraining)

- **`<phoneme>` tags** — the model has no IPA vocabulary
- **Custom lexicon** — no phoneme stage to apply word → IPA mappings
- **Fine-grained pronunciation control** — the model decides pronunciation from text tokens alone

See [Kokoro.md](Kokoro.md) if you need pronunciation control.

## Usage

```swift
import FluidAudioTTS

let manager = PocketTtsManager()
try await manager.initialize()

// Using built-in voices
let audioData = try await manager.synthesize(text: "Hello, world!")

// Using cloned voice
let voiceData = try await manager.cloneVoice(from: speakerAudioURL)
let audioData = try await manager.synthesize(text: "Hello, world!", voiceData: voiceData)

try await manager.synthesizeToFile(
    text: "Hello, world!",
    outputURL: URL(fileURLWithPath: "/tmp/output.wav")
)
```

## License

CC-BY-4.0, inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts).
