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

## v2.1 — Optimized Re-Conversion (current default)

v2.1 is the **same weights** as v2, re-converted for speed (not a finetune).
The loader points at `v2.1/<lang>/`. Measured on M-series / macOS 26:
**~905 ms → ~452–520 ms per utterance (~1.8× RTFx)**; validated end-to-end
(Whisper: english exact, german_24l intelligible).

| stage | v2 | v2.1 |
|---|---|---|
| conditioning | `cond_step`, per-token (~141 calls) | **`cond_prefill`** — whole block in 1 call (122→4.8 ms) |
| flow decoder | `flow_decoder`, 8-step Euler loop (8 calls/frame) | **`flow_decoder_fused`** — 8 steps fused, 1 call/frame (249→46 ms) |
| flowlm | fp16 | fp16 (unchanged; int8 `flowlm_stepv2` is the fastest variant) |
| mimi | unchanged | unchanged (compute-bound floor) |

**ANE residency (measured — corrects earlier claims):** **only
`flow_decoder_fused` runs on the ANE** (0%→100%; scatter-free fp16 graph).
`flowlm` and `cond` run on **GPU** — their rank-5 KV-cache `scatter` is rejected
by the ANE compiler at *any* precision, so the previous "flowlm 1.97× on ANE"
claim did **not** reproduce on-device. `mimi` is **CPU** (fp16 streaming
feedback beeps on ANE, and it is compute-bound regardless).

**Per-model compute units (measured fastest):** `flowlm` `.all`,
`flow_decoder_fused` `.all` (→ANE), `cond_prefill` `.all` (→GPU),
`mimi_decoder` `.cpuOnly`.

The v2 table below documents the original per-step contracts; v2.1 swaps in the
fused/prefill artifacts at `v2.1/<lang>/` and otherwise reuses v2's `mimi`,
`flowlm_stepv2`, and `constants`.

---

## Small models (pocket-tts 4.0.0): 18m / 7m / 2m with the FlowLM on the Neural Engine

Kyutai's 2026-10-20 release added six small English models — `english_18m_2026-10`,
`english_7m_2026-10`, `english_2m_2026-10` and reward-fine-tuned `*_refl_2026-10`
twins. They are a different architecture from the 100M (grouped-query attention,
a `z_proj` before the heads, a low-rank text table, one-step flow heads, a
separate 128-d Mimi decoder), so they have their own packs under
`FluidInference/pocket-tts-coreml` `v4/<model>/` and their own backend:

```swift
let tts = PocketTtsSmallManager(model: .english7m)        // .english18m, .english2m, *Refl
try await tts.initialize()                                 // downloads v4/<model>/ (34–145 MB)
let wav = try await tts.synthesize(text: "Hello, world!", voice: "alba")
for try await frame in try await tts.synthesizeStreaming(text: "…") { /* 80 ms frames */ }
```

```bash
fluidaudiocli tts "Hello world." --backend pocket-small --model 7m --seed 42 -o out.wav
```

Requires macOS 15 / iOS 18. The FlowLM runs on the Neural Engine through one
MLState multifunction package, `pocket_state.mlmodelc`: `prefill` writes the text
block into the FlowLM KV state and `generate` runs the FlowLM step plus the
one-step flow head per frame (`latent_final`, `is_eos`). The Mimi decoder runs as
the fp32 I/O model `mimi_decoder.mlmodelc` on the CPU (`mimiPrecision: .fp32`,
the default); its state (two small KV caches and ~18 KB of conv histories) is
plain inputs/outputs the host feeds back, zeroed per utterance.

**Why the Mimi decoder is not on the ANE by default.** The package also carries
`generate_decode`, which fuses step + head + an fp16 Mimi decoder into a single
ANE dispatch (`mimiPrecision: .fp16Fused`, `--mimi fp16-fused`). It is faster
(18m: 2.9 vs 4.4 ms per frame on an M5 Pro) but the fp16 SEANet adds a white noise
floor at about −72 dBFS, audible as hiss in pauses and at the start of an
utterance; PyTorch and the fp32 path sit at about −80 dBFS with the same spectral
tilt. The same graph in fp16 on the CPU has the same floor, so it is fp16
rounding of O(1) activations against O(1e-3) audio, not an ANE fault; keeping the
SEANet ops in fp32 inside the stateful program is rejected by the Core ML planner
(bad_cast, status −14), hence the separate fp32 model. The fused path keeps the
Mimi KV caches in the MLState; both Mimi models take a one-hot `phase` input per frame
(10 phases, from the frame counter) that selects the Mimi cache-write rows,
relative RoPE rotations and mask tables — the graph has no data-dependent integer
arithmetic because the ANE miscomputes it. Every dimension comes from the pack's
`arch.json` / `pocket_state.json` (`PocketTtsSmallArch`, `PocketTtsSmallStateInfo`)
or the compiled model's own input descriptions; nothing in `PocketTtsSmallEngine`
hard-codes a size.

Text is chunked at sentence boundaries to at most 50 tokens
(`PocketTtsConstants.preferredTokensPerChunk`), the upstream `MAX_TOKEN_PER_CHUNK`.
This matters more than for the 100M: the small models were trained on short
chunks, and fed a whole paragraph in one prefill several voices (eponine, azelma,
anna, estelle, jane, michael, paul with an 81-token passage) never emit EOS and run
to the frame budget — upstream PyTorch does exactly the same with
`max_tokens=512`, so it is model behaviour, not a conversion fault.

Differences from the 100M packs a host must know: the first `sequence` of a chunk
is `bos_emb` (no NaN protocol); `is_eos` is a raw logit with threshold −1.0 and is
ignored for the first 6 frames; temperature defaults to 0.3; the text table is
`[4001, d_model]` with the low-rank projection folded in; voices are the upstream
KV snapshots `[2, 1, n, Hkv, 64]` written as fp16 into `k_cache{i}` / `v_cache{i}`;
`–` and `—` are rewritten to `,` before tokenizing.

Measured on an M5 Pro with the fused fp16 path (`fluidaudiocli`, debug build,
86–91 frames of the verify sentence): 7m 156 ms (44× real time), 18m 260 ms (28×),
2m 125 ms (47×); the default fp32-Mimi path adds about 1.5 ms per frame (18m:
~18× real time). First load pays the ANE compile (10–150 s), warm loads are
sub-second. Whisper transcripts match the PyTorch output. If a pack is replaced in
place with a changed graph, Core ML may keep serving the previously compiled
program from its per-app cache (`~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache`);
a new directory name avoids it. Voice cloning is not wired for the small
models yet (the Mimi encoder is the 100M English one; only `speaker_proj_weight`
differs and ships in `constants_bin/`). Conversion code and the full trial log
(ANE faults in `conv_transpose`, time concat, fp16 RoPE, integer compares) live in
the private model-lab repo, `models/pocket-tts-small-coreml/`.

## Phonon 31m (Gradium)

[Phonon](https://gradium.ai/phonon) (Gradium, 2026-10-20) is the Pocket TTS recipe trained by
the Kyutai founders' spin-off. `phonon-31m-en` is Apache-2.0. It has the 100M architecture at
half width: 512-d, 6 layers, a one-step LSD flow head and the 100M's Mimi decoder. FluidAudio
runs it on the small-model host above:

```swift
let tts = PocketTtsSmallManager(model: .phonon31m)
try await tts.initialize()                          // FluidInference/pocket-tts-coreml phonon/phonon-31m-en
let wav = try await tts.synthesize(text: "Email support@gradium.ai for 24/7 help.", voice: "en_us_Zoey")
```

```bash
fluidaudiocli tts "Hello from Phonon." --backend pocket-small --model phonon --voice en_us_Easton -o out.wav
```

- **Voices.** 103 in all, listed by `availableVoices()`: 25 English voices named `en_us_<Name>` and
  `en_gb_<Name>`, 12 characters named `characters_<Name>`, the checkpoint `default`, and Gradium's
  French, German, Spanish and Portuguese catalogue voices. The released model speaks English only.
  Voices ship as flow-LM conditioning rows (`voices/*.safetensors`). The first use of a voice
  prefills it once and caches the result. `default`, `en_us_Easton`, `en_us_Zoey` and `en_gb_Maeve`
  also ship as KV snapshots.
- **Text.** Text goes through `PhononTextNormalizer`, a Swift port of Gradium's English normalizer
  covering numbers, currency, emails, URLs, abbreviations and symbols. It matches the Rust original
  on 907 test inputs.
- **Pack contract.** `arch.json` carries Phonon's differences:
  - EOS threshold +4.0, with no minimum frame count and a tail of 1 frame (`eos_tail_extra`)
  - a 25 ms raised-cosine fade at each chunk start (`fade_in_*`), which removes the Mimi decoder's
    −40 dBFS cold-start click, present in Gradium's own runtime too
  - the 251-wide Mimi window and the summed-conditioner input bias, which are baked into the graphs
- **Quality.** The fp32 Mimi stays the default. The fp16 Mimi adds near-white noise at about
  −80 dBFS in pauses. Against Gradium's `ptts` runtime at temperature 0, frame counts match, and
  Whisper transcripts are identical for the fp32 and Neural Engine paths.
- **Speed** (M5 Pro): about 20 ms to first audio, 3.3–4.2 ms per 80 ms frame, 19–25x real time.
- **Demo.** `Examples/PhononDemo` is a SwiftUI app with a ring that follows the live spectrum.
- **Conversion code**, parity tools and noise measurements live in model-lab,
  `models/gradium-phonon-coreml/`.

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

`<lang>` is the pack folder: `english_2026-10` (the default `english`), `english`
(`english_2026-04`), `french_24l`, `german`, `german_24l`, `italian`, `italian_24l`,
`portuguese`, `portuguese_24l`, `spanish`, `spanish_24l`.

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

Long text is split at sentence boundaries (`.!?`), and sentences are grouped into chunks of up to 50 tokens
(`PocketTtsConstants.preferredTokensPerChunk`). A longer sentence stays whole as long as it fits the KV cache: the
voice conditioning (about 125 positions for a shipped voice; a cloned voice uses its prompt length), its tokens and
its estimated speech frames must fit in 512 positions (#938).

A sentence that does not fit is cut, and each cut restarts prosody, so cuts go where a speaker pauses anyway (#934):
1. Clause punctuation: `, ; :`, an em dash, a spaced en dash or hyphen, an ellipsis, and the edges of a bracketed
   aside. A bracket inside a word (`f(x)`) is not an aside.
2. Word boundaries, as a fallback: the fewest parts that fit, balanced so no part is left as a short tail.

`normalizeText()` also capitalizes, adds terminal punctuation, and pads short text with leading spaces for better prosody.

## EOS Detection

`runFlowLMStep()` returns an `eos_logit`. When it exceeds `-4.0`, the code generates a few extra frames (3 for short text, 1 for long) then stops.

## CoreML Details

- Per-model compute units (measured fastest, v2.1): `flowlm`/`flow_decoder_fused` `.all`, `cond_prefill` `.all` (GPU), `mimi_decoder` `.cpuOnly`. Only the fused flow decoder reaches the ANE; mimi stays off the ANE (fp16 streaming-state feedback causes audible artifacts there)
- **Overriding compute units (#881).** The defaults above are measured on M-series / macOS 26 and are not tunable through `placement`, which only swaps model variants. Some hardware/OS pairs reject a default placement — an M1 Max on macOS 26.6.2 aborts `flow_decoder_fused` on the ANE (`ANEProgramProcessRequestDirect status=0x16`) at both precisions and both placements. Pass `PocketTtsComputeUnits` to `PocketTtsManager` / `PocketTtsModelStore`; every field is optional and `nil` keeps that stage's default:

  ```swift
  // Keep every stage off the Neural Engine (GPU for the transformer stages, CPU for mimi)
  let tts = PocketTtsManager(computeUnits: .avoidNeuralEngine)
  // Or move just the failing stage
  let tts = PocketTtsManager(computeUnits: PocketTtsComputeUnits(flowDecoder: .cpuAndGPU))
  ```

  CLI: `fluidaudiocli tts "…" --backend pocket --compute-units no-ane` (also `default`, `all`, `cpu-gpu`, `cpu-ane`, `cpu-only`). Under `.aneState` placement the `flowLM` field governs the fused `pocket_state` functions; that placement is ANE-resident by design, so pair `.avoidNeuralEngine` with `.gpu`.
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

Unlike KokoroAne / StyleTTS2 which run a CoreML G2P model to convert text to IPA phonemes **before** the model, PocketTTS feeds raw text tokens directly into the neural network. The model learned text→pronunciation mappings during training — there is no phoneme stage to intercept.

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

See [KokoroAne.md](KokoroAne.md) or [StyleTTS2.md](StyleTTS2.md) if you need pronunciation control.

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

| ID | Swift case | Layers | HF Path |
|----|------------|--------|---------|
| `english` | `.english` | 6 | `v2.1/english_2026-10/` |
| `english_2026-04` | `.english2026Apr` | 6 | `v2.1/english/` |
| `german` | `.german` | 6 | `v2.1/german/` |
| `german_24l` | `.german24L` | 24 | `v2.1/german_24l/` |
| `italian` | `.italian` | 6 | `v2.1/italian/` |
| `italian_24l` | `.italian24L` | 24 | `v2.1/italian_24l/` |
| `portuguese` | `.portuguese` | 6 | `v2.1/portuguese/` |
| `portuguese_24l` | `.portuguese24L` | 24 | `v2.1/portuguese_24l/` |
| `spanish` | `.spanish` | 6 | `v2.1/spanish/` |
| `spanish_24l` | `.spanish24L` | 24 | `v2.1/spanish_24l/` |
| `french_24l` | `.french24L` | 24 | `v2.1/french_24l/` |

Notes:
- `english` is Kyutai's current English model, `english_2026-10` (upstream's `english` id
  serves the same weights). It reads sentence-final numbers and years, and defaults to
  temperature 0.3, the value Kyutai's config sets for it ("human evals preferred this model
  at 0.3"); pass `temperature:` to override. Every other pack defaults to 0.7
  (`PocketTtsLanguage.defaultTemperature`).
- `english_2026-04` is the English pack before #1087: same architecture and tokenizer, older
  weights and voices. Apps that need the previous voice can pin `.english2026Apr`.
- French only ships a 24-layer pack upstream (no 6-layer variant).
- 24-layer packs are higher quality but slower and larger.
- 26 voice names are shared across every language pack, but the underlying
  acoustic embeddings are per-language. The set is 21 English-trained
  "literary" voices (alba, anna, azelma, bill_boerst, caro_davy, charles,
  cosette, eponine, eve, fantine, george, jane, javert, jean, marius, mary,
  michael, paul, peter_yearsley, stuart_bell, vera) plus 5 voices recorded
  natively in their target language: `estelle` (French), `lola` (Spanish),
  `juergen` (German), `rafael` (Portuguese), `giovanni` (Italian). The
  native-language voices generally produce the most idiomatic prosody for
  their matching language pack.
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
