# CosyVoice3 Swift Inference

Mandarin zero-shot voice cloning via Qwen2 LM + CFM Flow + HiFT vocoder,
running on CoreML.

> ⚠️ **Beta / experimental.** End-to-end synthesis is currently slow on
> Apple Silicon — RTFx < 1.0 typical, several seconds of latency for
> short Mandarin utterances. The slowdown is partly the Flow CFM stage
> (fp32, CPU-or-GPU only because fp16 + ANE produces NaNs through the
> fused `layer_norm` — CoreMLTools limitation, tracked upstream) and
> partly HiFT sinegen / windowing ops that fall back to CPU. May be a
> model issue, may be recoverable through better conversion. Treat
> performance numbers as preliminary; the Swift API, model layout, and
> prompt-asset format may change in subsequent releases without
> deprecation aliases.

## Files

| File | Role |
|------|------|
| `CosyVoice3TtsManager.swift` | Public actor — `initialize()`, `synthesize()`, `synthesizeFromFixture()`, `loadVoice()`, `downloadAndCreate()` |
| `CosyVoice3Models.swift` | The 4 CoreML model handles (prefill, decode, flow, hift) |
| `Assets/CosyVoice3ModelStore.swift` | Loads + compiles the four mlpackages, probes flat / nested layouts |
| `Assets/CosyVoice3ResourceDownloader.swift` | HuggingFace pull for `FluidInference/CosyVoice3-0.5B-coreml` |
| `Pipeline/Synthesize/CosyVoice3Synthesizer.swift` | Actor — prefill → decode loop → Flow → HiFT |
| `Pipeline/Synthesize/CosyVoice3RasSampler.swift` | top-p / top-k / repetition mask, seed-tokens bypass |
| `Pipeline/Synthesize/CosyVoice3SpeechEmbeddings.swift` | mmap of 6761×896 fp16 speech-embedding table (12 MB) |
| `Pipeline/Synthesize/CosyVoice3Types.swift` | `CosyVoice3SynthesisOptions`, `CosyVoice3SynthesisResult`, `CosyVoice3ParityOptions` |
| `Pipeline/Preprocess/CosyVoice3TextFrontend.swift` | Special-token splitting + `lm_input_embeds` assembly |
| `Pipeline/Preprocess/Qwen2BpeTokenizer.swift` | tiktoken-compatible byte-level BPE, 151 936 vocab (incl. fileprivate `ByteEncoder` 188-symbol byte→unicode shim) |
| `Pipeline/Preprocess/CosyVoice3TextEmbeddings.swift` | mmap of 151 936×896 fp16 text embedding table |
| `Pipeline/Preprocess/CosyVoice3ChineseNormalizer.swift` | Minimal regex-free port of `frontend_utils.py` |
| `Pipeline/Preprocess/CosyVoice3PromptMel.swift` | 24 kHz 80-bin log-mel matching `matcha audio.py` |
| `Pipeline/Preprocess/CosyVoice3PromptAssets.swift` | Voice-prompt bundle DTO (precomputed IDs / mel / spk-emb) |
| `Pipeline/Preprocess/CosyVoice3FrontendFixture.swift` | Phase 1 parity-fixture loader |
| `CosyVoice3Constants.swift` | Stop-token range, hidden dim, frame counts, etc. |
| `Shared/SafetensorsReader.swift` | ~170 LoC pure-Swift mmap + fp16/fp32/i32 accessors |

## Call Flow

```
CosyVoice3TtsManager.synthesize(text:promptAssets:options:)
  |
  v
CosyVoice3TextFrontend.assembleLmInput(text:promptAssets:)
  |
  |-- normalizeText()           split on <|endofprompt|>, replace_blank, etc.
  |-- Qwen2BpeTokenizer.encode  byte-level BPE → token IDs
  |-- text_embedding lookup     151 936×896 fp16 mmap → [N_text, 896]
  |-- speech_embedding lookup   6761×896 fp16 mmap → [N_speech, 896]
  |-- concat([SOS, text, TASK, prompt_speech_ids]) → lm_input_embeds
  |
  v
CosyVoice3Synthesizer.synthesize(lm_input_embeds:promptAssets:)
  |
  |-- runPrefill()              Qwen2 24L prefill, T <= 256
  |     |-- in: lm_input_embeds, attn_mask
  |     |-- out: logits[1,T,6761], kv_cache[24,1,2,768,64] fp16
  |
  |-- DECODE LOOP (until stop-range hit or maxNewTokens):
  |     |
  |     |-- runDecodeStep()         takes prev token + cached KV
  |     |     |-- in: token_id, kv_cache (in-place state)
  |     |     |-- out: logits[1,1,6761]
  |     |
  |     |-- RasSampler.sample()     top-p/top-k/repetition + seed-tokens bypass
  |     |-- if topId in stopRange (6561...6760): break
  |     |-- decoded.append(topId)
  |
  |-- runFlow()                 CFM 10-step ODE, conditional on prompt mel + spk_emb
  |     |-- in: decoded[N], prompt_mel, spk_embedding
  |     |-- out: full_mel[1, 80, M] fp32
  |
  |-- runHiFT()                 vocoder, chunk-packed (T<=500 frames)
  |     |-- in: full_mel slice from newMelStart..newMelStart+newMelFrames
  |     |-- out: audio samples [N*hop_len] @ 24 kHz
  |
  |-- concatenate chunks → CosyVoice3SynthesisResult.samples
```

## Public API

```swift
import FluidAudio

// One-shot creation that downloads everything to ~/.cache/fluidaudio/
let manager = try await CosyVoice3TtsManager.downloadAndCreate(
    computeUnits: .cpuAndNeuralEngine
)
try await manager.initialize()

// Load a voice prompt bundle (precomputed by mobius/.../bootstrap_aishell3_voices.py)
let voice = try CosyVoice3PromptAssets.load(from: voiceBundleURL)

let result = try await manager.synthesize(
    text: "希望你以后能够做的比我还好用",
    promptAssets: voice,
    options: CosyVoice3SynthesisOptions(maxNewTokens: 1024, seed: 42)
)
// result.samples : [Float]   (mono fp32, 24 kHz)
// result.sampleRate : 24000
```

`CosyVoice3SynthesisOptions`:

| Field | Default | Notes |
|---|---|---|
| `maxNewTokens` | `nil` (cap = 1024) | Hard ceiling on speech-token count |
| `seed` | 42 | Drives the RAS sampler RNG; reproducible runs |

`CosyVoice3SynthesisResult`:

| Field | Type | Notes |
|---|---|---|
| `samples` | `[Float]` | mono, fp32, range ~[-1.0, 1.0] |
| `sampleRate` | `Int` | always 24000 |
| `generatedTokenCount` | `Int` | tokens before EOS |
| `decodedTokens` | `[Int32]` | full speech token sequence (debug) |

## Key State

### KV cache (`kv_cache[24, 1, 2, 768, 64]` fp16)
- 24 transformer layers × `[K,V]` × heads × dim, packed into one `MLState`-style
  `MLMultiArray` that the prefill produces and the decode loop both reads
  and overwrites in-place.
- Reset per `synthesize()` call.

### Prompt assets (`CosyVoice3PromptAssets`)
- `promptText` — Mandarin reference text (must contain `<|endofprompt|>`).
- `promptSpeechIds: [Int32]` — pre-tokenized speech IDs from the
  SpeechTokenizerV3 mlpackage (computed offline, reused across calls).
- `promptMel: [Float]`, `promptMelFrames` — 80-bin log-mel of the reference
  audio at 24 kHz.
- `spkEmbedding: [Float]` — 192-dim speaker embedding from CAMPPlus.

Bundles are produced by
`mobius/models/tts/cosyvoice3/coreml/verify/bootstrap_aishell3_voices.py`
or `extract_voice_prompt.py` for arbitrary speakers.

## CoreML details

- **Compute units:** caller chooses (`.cpuAndNeuralEngine` works for
  prefill + decode + HiFT). Flow is forced to `.cpuAndGPU` regardless —
  fp32 graph, ANE NaNs through the fused `layer_norm`.
- All four mlpackages compiled `.mlpackage → .mlmodelc` on first load and
  cached on disk under `~/.cache/fluidaudio/Models/cosyvoice3/`.
- `CosyVoice3ModelStore` is an actor; `CosyVoice3Synthesizer` is an
  actor. `CosyVoice3Models` (the four-tuple) conforms to `Sendable` via
  `@preconcurrency import CoreML`, matching the existing `TtsModels`
  pattern.

## Stop-token handling

- Speech vocab is `0..<6761`; tokens `6561..<6761` are the EOS range.
- `CosyVoice3Constants.stopRange = 6561...6760` (closed range). The decode
  loop breaks when `topId` falls in that range.
- If the prefill emits a stop token at step 0 the synthesizer throws
  `CosyVoice3Error.predictionFailed` instead of falling through —
  feeding the stop-token embedding into the decode loop would
  accumulate semantically meaningless tokens.

## CLI

```
fluidaudio tts --backend cosyvoice3 \
    --text "希望你以后能够做的比我还好用" \
    --models-dir ~/.cache/fluidaudio/Models/cosyvoice3 \
    --tokenizer-dir … --embeddings-file … --special-tokens-file … \
    --prompt-assets path/to/voice.safetensors \
    --output out.wav
```

`--backend cosyvoice3` (and the `cv3` alias) runs the production
text-driven synthesis path. `--backend` help text flags it as
`[BETA — slow, RTFx < 1.0]` and the dispatcher emits a runtime
`logger.warning` so the beta status shows up without reading docs.

### Dev sub-backends (for debugging the Python ↔ Swift contract)

These are the harnesses future contributors use to bisect divergence
between the Swift port and the upstream Python reference. Each isolates
a distinct stage of the pipeline:

```
fluidaudio tts --backend cosyvoice3-tokenizer-parity \
    --tokenizer-dir … --fixture tokenizer_fixture.json
# Qwen2 BPE encode/decode parity vs tiktoken reference

fluidaudio tts --backend cosyvoice3-frontend-parity \
    --tokenizer-dir … --embeddings-file … \
    --fixture shipping.safetensors --tok-fixture …
# lm_input_embeds assembly parity (text+speech embed lookup, SOS/TASK splice)

fluidaudio tts --backend cosyvoice3-parity \
    --fixture shipping.safetensors --models-dir build/
# Phase 1 fixture parity (Synthesizer: prefill → decode → Flow → HiFT)
```

Recommended bisection order when end-to-end output diverges from
Python: tokenizer-parity → frontend-parity → fixture parity.

The production backend auto-downloads its CoreML mlpackages, tokenizer,
embeddings, and default voice from HuggingFace on first synthesis (cached
under `~/.cache/fluidaudio/Models/cosyvoice3/`) — there is no separate
download CLI mode, matching how Kokoro and PocketTTS work.

## Models

| Component | mlpackage | Precision | Notes |
|---|---|---|---|
| Qwen2 LLM — Prefill (T=256, M=768) | `LLM-Prefill-T256-M768-fp16` | fp16 | KV-cache out |
| Qwen2 LLM — Decode (M=768) | `LLM-Decode-M768-fp16` | fp16 | KV-cache in-place |
| CFM Flow (N=250 → M=500 mel) | `Flow-N250-fp32` | fp32 | CPU/GPU only |
| HiFT vocoder (T=500 → 10 s @ 24 kHz) | `HiFT-T500-fp16` | fp16 | sinegen on CPU |
| Qwen2 + speech embedding tables | `embeddings-fp16.safetensors` | fp16 | mmap'd at runtime |

All shipped at
[`FluidInference/CosyVoice3-0.5B-coreml`](https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml).
The conversion pipeline that produced them lives in
[FluidInference/mobius#42](https://github.com/FluidInference/mobius/pull/42).

## Non-goals / known limits

- **No on-device prompt-asset preparation.** SpeechTokenizerV3 and
  CAMPPlus have CoreML mlpackages but the surrounding DSP isn't ported
  to Swift yet. Callers either use the bundled
  `cosyvoice3-default-zh` voice or run the Python `extract_voice_prompt.py`
  offline.
- **No production-grade Mandarin TN.** `CosyVoice3ChineseNormalizer`
  only mirrors the simple cleanups in upstream `frontend_utils.py`.
  For year / currency / decimal / unit normalization, run
  `wetext.ZhNormalizer` server-side and pass `prenormalized: true` on
  `synthesize()`.
- **Flow stays fp32 (~1.2 GB).** Until CoreMLTools pins fused-`layer_norm`
  fp16 the model NaNs on ANE. Loaded once, kept resident.
- **Streaming API not yet exposed.** The synthesizer runs Phase 1
  (prefill) and Phase 2 (Flow + HiFT) sequentially against the full
  token sequence. Token streaming is internal but not surfaced through
  an `AsyncStream`.

## License

- **CosyVoice3 model weights:** Apache 2.0, inherited from
  [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
  upstream (`speech_300m`, `Fun-CosyVoice3-0.5B-2512`).
- **FluidAudio SDK:** Apache 2.0.
