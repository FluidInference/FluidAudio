# Supertonic-3 (v1.7.3) Swift Inference

Multilingual flow-matching diffusion TTS. 31 languages, 44.1 kHz mono
Float32 output, voice styling via per-speaker JSON tensors. 4-stage
CoreML pipeline at ~398 MB total.

## Overview

Supertonic-3 is a 4-stage CoreML port of
[supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)
v1.7.3. Each utterance runs:

```
text → text_encoder       → text_emb       [1, 256, 128]
text → duration_predictor → duration       [1]
       vector_estimator   ×8 denoising steps (flow-matching diffusion w/ CFG)
       vocoder            → 44.1 kHz mono Float32 wav
```

The `vector_estimator` stage is the cost center: 8 denoising steps with
classifier-free guidance built into the graph (W_COND = 4.0,
W_UNCOND = 3.0). Voice identity rides on two style tensors
(`style_ttl`, `style_dp`) loaded from per-speaker JSON presets — the
upstream repo ships `M1`/`F1`/`M2`/`F2`/… and the FluidInference mirror
re-publishes them under `assets/voice_styles/`.

## Quick Start

### CLI

```bash
swift run fluidaudiocli tts "Hello from Supertonic three." \
    --backend supertonic3 \
    --lang en \
    --voice-style path/to/M1.json \
    --output ~/Desktop/sup3-demo.wav
```

`--voice-style` is required and must point at a Supertonic voice preset
JSON. On first invocation the CLI downloads the four `.mlmodelc`
bundles plus `tts.json` and `unicode_indexer.json`; subsequent calls
reuse the disk cache.

Optional flags:

| Flag | Default | Notes |
|---|---|---|
| `--lang <code>` | `en` | One of the 31 supported ISO codes — see [Languages](#languages) |
| `--total-steps <n>` | `8` | Denoising step count; lower trades quality for latency |
| `--speed <f>` | `1.05` | Speech-rate multiplier; divides predicted duration |

### Swift

```swift
import FluidAudio

let manager = try await Supertonic3Manager.downloadAndCreate(
    computeUnits: .cpuAndNeuralEngine
)

let style = try Supertonic3VoiceStyle.load(
    from: URL(fileURLWithPath: "/path/to/M1.json"))

let (samples, _) = try await manager.synthesize(
    text: "Hello from Supertonic three.",
    language: "en",
    style: style)
// `samples`: Float32 PCM, 44.1 kHz mono.
```

## Pipeline

```
text                       voice style JSON (M1.json …)
  |                                |
  v                                v
Supertonic3TextChunker       Supertonic3VoiceStyle.load
  |  (paragraph → sentence       |  (style_ttl  [1, 50, 256]
  |   → comma → word, 110         |   style_dp  [1,  8,  16])
  |   chars Latin / 90 CJK)       |
  v                                |
Supertonic3UnicodeProcessor        |
  |  (NFKD → strip emoji →         |
  |   <lang>…</lang> tags →        |
  |   65536-entry codepoint        |
  |   indexer → text_ids, mask)    |
  | text_ids   [1, 128]            |
  | text_mask  [1, 1, 128]         |
  v                                v
Supertonic3Synthesizer  (4 CoreML stages, batch = 1)
  ├─ text_encoder       text_ids + style_ttl  → text_emb     [1, 256, 128]
  ├─ duration_predictor text_ids + style_dp   → duration     [1]
  ├─ vector_estimator   ×8 denoising steps    → denoised_latent
  │     (flow-matching diffusion w/ CFG; current_step/total_step
  │      drive σ schedule; style_ttl + text_mask + latent_mask
  │      condition each step)
  └─ vocoder            denoised_latent       → wav (44.1 kHz mono Float32)
```

The flow-matching loop iterates `vector_estimator` `totalSteps` (= 8)
times, feeding each `denoised_latent` back in as the next step's
`noisy_latent`. CFG is fused into the model — Swift only supplies the
`current_step` / `total_step` scalars.

## Files

| File | Role |
|---|---|
| `Supertonic3Manager.swift` | Public actor — `initialize()`, `downloadAndCreate()`, `synthesize(text:language:style:totalSteps:speed:silenceDuration:)`, `cleanup()` |
| `Supertonic3Constants.swift` | Compile-time constants (44.1 kHz, T=128 token cap, 8 default steps, 31-language whitelist, chunker caps) |
| `Supertonic3Error.swift` | Per-module `Error, LocalizedError` enum |
| `Supertonic3Types.swift` | `Supertonic3Config` (`tts.json` schema) + `Supertonic3VoiceStyle` (decoded style tensors) |
| `Assets/Supertonic3ModelStore.swift` | Actor — loads the 4 `.mlmodelc` bundles + `tts.json` + `unicode_indexer.json` |
| `Assets/Supertonic3ResourceDownloader.swift` | HuggingFace pull for `FluidInference/supertonic-3-coreml` |
| `Pipeline/Preprocess/Supertonic3UnicodeProcessor.swift` | NFKD normalize + emoji strip + symbol replace + `<lang>` wrap + 65536-entry codepoint → token-id lookup, pads to fixed T=128 |
| `Pipeline/Preprocess/Supertonic3TextChunker.swift` | Sentence-aware chunker (110 chars Latin, 90 CJK) for long utterances |
| `Pipeline/Synthesize/Supertonic3Synthesizer.swift` | 4-stage CoreML driver — text_encoder + duration_predictor → 8-step vector_estimator → vocoder |
| `Pipeline/Synthesize/Supertonic3LatentSampler.swift` | Noisy-latent + latent-mask sampling for the flow-matching loop |
| `Pipeline/Synthesize/Supertonic3MultiArray.swift` | `MLMultiArray` ergonomics shared by the synthesizer stages |

## Text preprocessing

Mirrors the upstream `UnicodeProcessor` from
[Helper.swift](https://github.com/supertone-inc/supertonic/blob/main/swift/Sources/Helper.swift):

1. NFKD decompose (`decomposedStringWithCompatibilityMapping`).
2. Strip emoji codepoints in the wide Unicode planes.
3. Replace dashes / quotes / brackets / arrows with ASCII equivalents.
4. Drop decorative symbols (`♥`, `☆`, `\\`, …).
5. Expand a small set of abbreviations (`@` → " at ", `e.g.,` →
   "for example,", `i.e.,` → "that is,").
6. Tighten spacing around punctuation; collapse repeated quotes.
7. Append `.` if no terminal punctuation.
8. Wrap with `<lang>…</lang>` tags.
9. Map each Unicode scalar through `unicode_indexer.json` (a flat
   `[Int64]` array keyed by codepoint) — unknown scalars receive `-1`
   so the model can mask them.

The `text_encoder` + `duration_predictor` models pin the T axis at
**128**; longer inputs are truncated and shorter ones zero-padded. The
chunker keeps each pass under 110 (Latin) / 90 (CJK) characters so the
NFKD-expanded codepoint sequence plus `<lang>` overhead fits inside
the 128-token window.

## Voice styles

`Supertonic3VoiceStyle` loads two style tensors from a per-speaker JSON:

| Field | Shape | Consumed by |
|---|---|---|
| `style_ttl` | `[1, 50, 256]` | `text_encoder` + every `vector_estimator` step |
| `style_dp` | `[1, 8, 16]` | `duration_predictor` |

The upstream repo ships `M1` / `M2` / `F1` / `F2` etc. as static JSON
files under
[`FluidInference/supertonic-3-coreml/assets/voice_styles/`](https://huggingface.co/FluidInference/supertonic-3-coreml/tree/main/assets/voice_styles).
Pass any of those to `Supertonic3VoiceStyle.load(from:)`; the decoded
struct is `Sendable` and can be reused across many synthesize calls.

## Languages

31 ISO codes plus `"na"` (language-agnostic / numeric):

```
en  ko  ja  ar  bg  cs  da  de  el  es  et  fi
fr  hi  hr  hu  id  it  lt  lv  nl  pl  pt  ro
ru  sk  sl  sv  tr  uk  vi  na
```

No Chinese (`zh`) — Supertonic-3 was not trained on Mandarin. Use
[`KokoroAne`](KokoroAne.md) with `--variant mandarin` for Mandarin
TTS in FluidAudio.

`Supertonic3Constants.cjkLanguages = ["ko", "ja"]` uses a tighter
90-char chunk cap to leave headroom inside the 128-token window —
CJK uses more codepoints per visible character after NFKD.

## Models

All CoreML packages live under
[`FluidInference/supertonic-3-coreml`](https://huggingface.co/FluidInference/supertonic-3-coreml).

| Stage | mlmodelc | Notes |
|---|---|---|
| Text encoder | `TextEncoder.mlmodelc` | T=128 fixed, outputs `text_emb [1, 256, 128]` |
| Duration predictor | `DurationPredictor.mlmodelc` | T=128 fixed, outputs scalar `duration` |
| Vector estimator | `VectorEstimator.mlmodelc` | 8 denoising steps with fused CFG (W_COND = 4.0, W_UNCOND = 3.0) |
| Vocoder | `Vocoder.mlmodelc` | Outputs `wav` at 44.1 kHz mono Float32 |

Companion assets:

| File | Role |
|---|---|
| `tts.json` | Sample rate, chunk-compress factor, base chunk size — overrides the compile-time defaults in `Supertonic3Constants` |
| `unicode_indexer.json` | Flat `[Int64]` array, indexed by codepoint, mapping every Unicode scalar to a model token id |
| `assets/voice_styles/*.json` | Per-speaker `style_ttl` / `style_dp` tensors |

## Known issues

- **No Chinese.** Supertonic-3's training set does not include Mandarin.
  Use KokoroAne `--variant mandarin` for `zh`.
- **Voice styles are not auto-downloaded.** The four model files +
  `unicode_indexer.json` + `tts.json` come down on first init, but
  voice style JSONs must be supplied by the caller (path on disk or
  fetched separately from the HF repo).
- **Fixed 128-token cap.** Single-pass utterances are bounded by the
  encoder window. The chunker silently splits longer text and
  concatenates per-chunk audio with `silenceDuration` (default 0.3 s)
  in between — fine for paragraphs, but per-chunk prosody continuity
  is not preserved.
- **WER on whitespace-free scripts.** Korean / Japanese have no word
  boundaries; `WERCalculator` splits on whitespace and reports
  near-100% WER. Trust CER on those rows.

## License

- **Supertonic-3 model weights:** Apache 2.0, inherited from
  [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)
  upstream.
- **FluidAudio SDK:** Apache 2.0.
