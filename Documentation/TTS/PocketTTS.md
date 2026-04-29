# PocketTTS Swift Inference

How the Swift code generates speech from text.

## Files

| File | Role |
|------|------|
| `PocketTtsManager.swift` | Public API — `initialize()`, `synthesize()`, `synthesizeToFile()`, `makeSession()`, `cloneVoice()` |
| `PocketTtsModelStore.swift` | Loads and stores the 4 CoreML models + constants + voice data |
| `PocketTtsVoiceCloner.swift` | Voice cloning — converts audio to voice conditioning embeddings |
| `PocketTtsSynthesizer.swift` | Main synthesis loop — chunking, prefill, generation, output |
| `PocketTtsSession.swift` | Session actor — persistent voice KV cache, enqueue/finish/cancel API |
| `PocketTtsSynthesizer+KVCache.swift` | KV cache state, `prefillKVCacheVoice()`, `prefillKVCacheText()`, `cloneKVCacheState()` |
| `PocketTtsSynthesizer+Flow.swift` | Flow decoder loop, `denormalize()`, `quantize()`, SeededRNG |
| `PocketTtsSynthesizer+Mimi.swift` | Mimi decoder state, `runMimiDecoder()`, `loadMimiInitialState()` |
| `PocketTtsConstantsLoader.swift` | Loads binary constants (embeddings, tokenizer, quantizer weights) |
| `PocketTtsConstants.swift` | All numeric constants (dimensions, thresholds, etc.) |

## Model Files & Precision

The four CoreML submodels (plus the optional Mimi encoder) and their
auxiliary asset directories. All paths are relative to
[`FluidInference/pocket-tts-coreml`](https://huggingface.co/FluidInference/pocket-tts-coreml)
on HuggingFace; sizes are for the English language pack.

| File | Precision | Size | HF path | Role |
|------|-----------|------|---------|------|
| `cond_step.mlmodelc` | fp16 | 254.3 MB | `v2/<lang>/cond_step.mlmodelc` | KV-cache prefill — runs once per chunk over voice + text tokens (~141 calls); writes the 6-layer KV cache that FlowLM consumes during generation |
| `flowlm_step.mlmodelc` | fp16 | 290.5 MB | `v2/<lang>/flowlm_step.mlmodelc` | Autoregressive transformer — runs **once per audio frame** during generation; outputs a `[1, 1024]` hidden state + EOS logit per step. Loaded when `precision: .fp16` (default) |
| `flowlm_stepv2.mlmodelc` | int8 attn + FFN linears, fp16 elsewhere | 73.5 MB | `v2/<lang>/flowlm_stepv2.mlmodelc` | Drop-in replacement for `flowlm_step` when `precision: .int8` — same I/O signature, ~4× smaller. Quantization recipe per [kyutai-labs/pocket-tts#147](https://github.com/kyutai-labs/pocket-tts/pull/147) |
| `flow_decoder.mlmodelc` | fp16 | 37.3 MB | `v2/<lang>/flow_decoder.mlmodelc` | LSD flow-matching decoder — runs an 8-step Euler loop per audio frame (`latent += velocity · dt`); turns transformer output into a 32-dim audio latent |
| `mimi_decoder.mlmodelc` | fp16 (outputs explicitly fp16) | 40.0 MB | `v2/<lang>/mimi_decoder.mlmodelc` | Mimi VAE audio decoder — runs once per audio frame, takes a 512-dim quantized vector and produces 1920 PCM samples (24 kHz). Maintains 23 streaming-state tensors fed back as next-frame input |
| `mimi_encoder.mlmodelc` | fp16 | optional | `mimi_encoder.mlmodelc` *(repo root)* | **Voice cloning only.** Language-agnostic; lives at the repo root, not under `v2/<lang>/`. Downloaded separately on first `cloneVoice(...)` call |
| `constants_bin/` | binary tensors | 144.2 MB | `v2/<lang>/constants_bin/` | Token embedding table, SentencePiece tokenizer, denormalize/quantize mean+std, per-voice prompts (`alba.safetensors`, etc.) |
| `constants/` | metadata sidecar | 16.7 MB | `v2/<lang>/constants/` | Auxiliary constants referenced by `PocketTtsConstantsLoader` |

`<lang>` is one of: `english`, `french_24l`, `german`, `german_24l`,
`italian`, `italian_24l`, `portuguese`, `portuguese_24l`, `spanish`,
`spanish_24l`.

### Totals (English, on disk)

| Configuration | Total |
|---------------|-------|
| `precision: .fp16` (default) | 766.3 MB |
| `precision: .int8` | 549.3 MB |
| **int8 savings vs fp16** | **−217 MB (28%)** |

The `v2/<lang>/` HF directory ships **both** flowlm variants, so a fresh
download pulls the unused one too. `PocketTtsResourceDownloader` deletes
the unused FlowLM `.mlmodelc` and `.mlpackage` directories after download
completes so only the requested precision occupies disk long-term.

### Why only `flowlm_step` is quantized

The four submodels have different sensitivity to quantization. Only the
FlowLM transformer is published in an int8 variant upstream:

| Submodel | Quantized? | Why |
|----------|------------|-----|
| `cond_step` | No | One-shot prefill; conditioning errors propagate through the entire utterance |
| `flowlm_step` | **Yes** | Per-frame transformer with causal attention; quantization error stays bounded per frame, doesn't compound. Largest file — best size-to-risk trade |
| `flow_decoder` | No | 8-step Euler loop where each step's error feeds the next; small file (37 MB) makes savings marginal anyway |
| `mimi_decoder` | No | Autoregressive feedback loop where 23 streaming-state tensors carry across frames; errors compound frame-over-frame |

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

### Cloning Across Languages

The Mimi encoder is language-agnostic — voice cloning produces a generic
acoustic embedding that any language pack's `cond_step` model can consume.
You can:

- Clone a voice once and reuse the same `PocketTtsVoiceData` across managers
  configured with different languages.
- Clone a voice with a Spanish-only manager without pulling in the English
  language pack — only the encoder subtree is downloaded.

```swift
// Clone with a Spanish manager
let esManager = PocketTtsManager(language: .spanish)
try await esManager.initialize()
let voiceData = try await esManager.cloneVoice(from: speakerAudioURL)

// Use the same cloned voice with a French manager
let frManager = PocketTtsManager(language: .french24L)
try await frManager.initialize()
let frAudio = try await frManager.synthesize(text: "Bonjour", voiceData: voiceData)
```

## Pipeline and Pronunciation Control

```
text → SentencePiece tokenizer → subword tokens → PocketTTS model → audio
                                                    ↑
                                          pronunciation decisions
                                          happen inside model weights
                                          (no external control)
```

Unlike Kokoro which uses a CoreML G2P model to convert text to IPA phonemes **before** the model, PocketTTS feeds raw text tokens directly into the neural network. The model learned text→pronunciation mappings during training — there is no phoneme stage to intercept.

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

## Session API

For streaming input or long-running low-latency sessions, `makeSession()`
performs the voice prefill once, then each enqueued utterance only prefills
text tokens. Mimi state persists across utterances for seamless audio.

```swift
let session = try await manager.makeSession(voice: "alba")
session.enqueue("Hello there.")
session.enqueue("How are you doing today?")
session.finish()
for try await frame in session.frames {
    playAudio(frame.samples)
}
```

| Method | Description |
|--------|-------------|
| `manager.makeSession(voice:temperature:seed:)` | Create session with named voice |
| `manager.makeSession(voiceData:temperature:seed:)` | Create session with cloned voice |
| `session.enqueue(_ text:)` | Add text (non-async, safe from any context) |
| `session.finish()` | End the session and complete the frames stream |
| `session.cancel()` | Stop generation immediately |
| `session.frames` | `AsyncThrowingStream<AudioFrame, Error>` |

| Scenario | API |
|----------|-----|
| One-shot synthesis | `synthesize()` |
| Streaming playback | `synthesizeStreaming()` |
| Streaming text or custom chunking | `makeSession()` |

## Languages

PocketTTS ships with multiple language packs converted from
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts). Pick the one
that matches your input text — there is no automatic language detection.

| ID | Layers | HF Path |
|----|--------|---------|
| `english` | 6 | repo root (legacy layout) |
| `german` | 6 | `v2/german/` |
| `german_24l` | 24 | `v2/german_24l/` |
| `italian` | 6 | `v2/italian/` |
| `italian_24l` | 24 | `v2/italian_24l/` |
| `portuguese` | 6 | `v2/portuguese/` |
| `portuguese_24l` | 24 | `v2/portuguese_24l/` |
| `spanish` | 6 | `v2/spanish/` |
| `spanish_24l` | 24 | `v2/spanish_24l/` |
| `french_24l` | 24 | `v2/french_24l/` |

Notes:
- French only ships a 24-layer pack upstream (no 6-layer variant).
- 24-layer packs are higher quality but slower and larger.
- The 21 voice names (alba, anna, eve, michael, …) are shared across
  languages, but the underlying acoustic embeddings are per-language.
- Mimi encoder weights (used for voice cloning) are language-agnostic and
  always live at the repo root.

### Swift API

```swift
let manager = PocketTtsManager(language: .spanish)
try await manager.initialize()
let audio = try await manager.synthesize(text: "Hola mundo")
```

`PocketTtsManager.language` is immutable per instance. To support multiple
languages in one app, instantiate one manager per language.

### CLI Usage

```bash
# Default (English)
fluidaudio tts "Hello world" --backend pocket --output en.wav

# Spanish (6L)
fluidaudio tts "Hola mundo" --backend pocket --language spanish --output es.wav

# French (24L only)
fluidaudio tts "Bonjour" --backend pocket --language french_24l --output fr.wav
```

## Usage

PocketTTS is part of core `FluidAudio` - no GPL dependencies required.

```swift
import FluidAudio

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

- **PocketTTS models**: CC-BY-4.0, inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- **FluidAudio SDK**: Apache 2.0 licensed (no GPL dependencies)
