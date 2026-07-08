# LuxTTS (ZipVoice-Distill) Swift Inference

Zero-shot voice-cloning TTS conditioned on a short prompt clip and its
transcript. 48 kHz mono Float32 output, 3-stage CoreML pipeline
(flow-matching, 4 anchor-Euler steps).

## Overview

LuxTTS is the CoreML port of ZipVoice-Distill (conversion lives in
`mobius/models/tts/zipvoice`). Each utterance runs:

```
espeak-IPA phonemes → TextEncoder            → token_embeds [1, 256, 100]
                      avg-duration expansion → per-frame text_condition
prompt clip (24 kHz) → VocosFbank mel ×0.1   → speech_condition
                      FmDecoder ×4 anchor-Euler steps (guidance 3.0)
                      Vocos vocoder (fixed 282/555-frame bucket) → 48 kHz wav
```

Models: [FluidInference/luxtts-coreml](https://huggingface.co/FluidInference/luxtts-coreml)

| Asset | Purpose |
|---|---|
| `gpu/TextEncoder.mlmodelc`, `gpu/FmDecoder.mlmodelc` | Original graph. **macOS** path, `.cpuAndGPU`. **Never run on the ANE** — the rel-pos attention path corrupts audio there. |
| `ane/TextEncoder.mlmodelc`, `ane/FmDecoder.mlmodelc` | ANE-canonical rewrite. **iOS** path, `.cpuAndNeuralEngine` (FmDecoder 100% ANE-resident, TextEncoder 99%). Same external I/O as `gpu/` — see below. |
| `vocoder/Vocoder282.mlmodelc`, `vocoder/Vocoder555.mlmodelc` | Fixed-shape 48 kHz Vocos vocoders (`mel (1,100,S) → audio`). Shared across variants; run `.cpuAndGPU` **everywhere** (only ~72% ANE-placeable, and CPU_AND_NE compilation is flaky). Smallest bucket ≥ generated frames is used; mel is never truncated. |
| `tokens.txt`, `config.json` | EmiliaTokenizer phoneme→id table (espeak IPA per Unicode scalar + pinyin initials/finals). Shared across variants. |

### Platform variant selection

`ModelNames.LuxTts.defaultVariant` picks `gpu` on macOS and `ane` on iOS;
`LuxTtsModelStore` maps that to `.cpuAndGPU` / `.cpuAndNeuralEngine`. The
downloader pulls **only** the selected variant's two `.mlmodelc` bundles
(plus the shared `vocoder/`, `tokens.txt`, `config.json`) — macOS never
fetches `ane/`, iOS never fetches `gpu/` (ModelHub's `requiredFiles`
pattern filter scopes the tree walk to the variant prefix).

Set `FLUIDAUDIO_LUXTTS_VARIANT=ane` (or `gpu`) to override the platform
default. This is the validation seam: the M-series ANE runs the `ane/`
graph under `.cpuAndNeuralEngine` exactly as an iPhone would, so the iOS
path is fully exercisable on a Mac without an `#if os(iOS)` fork.

### ANE vs GPU graph I/O (identical external contract)

Despite the "ANE-canonical" rewrite (the FmDecoder pre-concatenates
`x|text_condition|speech_condition` on the channel axis and works in
`(1, C, 1, S)` form internally for zero-fallback ANE placement), **the
externally-visible input/output signature is byte-for-byte the same as the
`gpu/` graph** — the cat / transpose / expand-dims are all inside the MIL,
and the output is transposed back to `(1, 1024, 100)` before it leaves the
graph. So no host-side adapter is needed; the same `LuxTtsSynthesizer`
tensor packing drives both graphs.

| model | inputs | output |
|---|---|---|
| **TextEncoder** (gpu + ane) | `tokens (1,256) int32`, `padding_mask (1,256) fp32` | `token_embeds (1,256,100) fp32` |
| **FmDecoder** (gpu + ane) | `x (1,1024,100)`, `text_condition (1,1024,100)`, `speech_condition (1,1024,100)`, `t (1)`, `guidance_scale (1)`, `padding_mask (1,1024)` — all fp32 | `v (1,1024,100) fp32` |

### ANE quality / latency tradeoff

The ANE FmDecoder runs in fp16 (ANE has no fp32 datapath); the gpu graph
keeps fp32 accumulation. This softens the output slightly but stays
transcript-verbatim:

| variant | core RTFx (full bucket) | steady footprint | log-mel cos vs oracle | RMS delta | round-trip transcript |
|---|---|---|---|---|---|
| gpu (`.cpuAndGPU`) | ~92× | ~1 GB | 0.999 | -0.09 dB | verbatim |
| ane (`.cpuAndNeuralEngine`) | ~27× | ~660 MB / **25.5 MB** ANE weights | 0.964 | -0.54 dB | verbatim |

The ANE path is ~0.5 dB softer, not bit-identical, but the Parakeet
round-trip transcript matches the input text exactly. The much smaller
jetsam-visible footprint is why iOS uses it.

All shapes are fixed: ≤ 255 tokens (+1 pad slot), ≤ 1024 mel frames
total, ≤ 555 generated frames (~5.9 s per call; chunking is phase 2).

## Quick Start

### CLI

```bash
swift run fluidaudiocli tts \
  "The quick brown fox jumps over the lazy dog, and honestly, it felt great." \
  --backend luxtts \
  --prompt-audio prompt_clip.wav \
  --prompt-text "quick brown fox jumps over the lazy dog and honestly it felt great." \
  --seed 42 \
  --output out.wav
```

`--prompt-audio` (voice to clone; first 5 s used) is required. Text and
`--prompt-text` are plain English — phonemized in-process by the
espeak-parity G2P (see below). If `--prompt-text` is omitted, the prompt
clip is transcribed with the built-in Parakeet ASR (models download on
first use; TTS-only runs never pay that cost). `--phonemes` bypasses the
G2P: both the text and `--prompt-text` are then espeak IPA (`en-us`).

### Swift API

```swift
let manager = try await LuxTtsManager.downloadAndCreate()
let result = try await manager.synthesize(
    text: "The quick brown fox jumps over the lazy dog.",
    promptAudio: promptURL,          // any format/rate; 24 kHz mono internally
    promptText: "The transcript of the prompt clip.",
    speed: 1.0,
    seed: 42)
// result.samples: 48 kHz mono Float32, prompt-matched loudness
```

Phoneme overloads remain for callers running their own espeak frontend:
`synthesize(phonemes:promptAudio:promptPhonemes:…)` and
`synthesize(tokenIds:promptAudio:promptTokenIds:…)` (raw `tokens.txt` ids).

## Usage notes

- **Keep `speed` at 1.0.** Upstream's `generate()` silently multiplies
  speed by 1.3, which squeezes the ratio-based duration estimate and
  clips sentence onsets. The Swift port never applies it.
- **Trim prompt silence.** Generated length is estimated as
  `prompt_frames / prompt_tokens × text_tokens / speed`; leading or
  trailing silence in the prompt inflates frames-per-token and slows or
  pads the output. Trim with `VadManager` (or any editor) before
  passing the clip.
- **Loudness contract**: prompts quieter than RMS 0.1 are boosted for
  conditioning and the generated wav is scaled back to the prompt's
  original level (upstream `rms_norm`). Don't peak-normalize the output.
- **Determinism**: same phonemes + prompt + seed → same output on the
  same OS/hardware. The noise RNG is not torch's, so waveforms differ
  from the Python pipeline while duration/loudness match (fixture-gated:
  frame accounting exact, RMS within 1 dB).

## G2P (phase 2): espeak-parity English frontend

The model was trained on espeak-ng (`en-us`) phonemes via
EmiliaTokenizer, so `LuxTtsG2p` reproduces **espeak**, not a generic
G2P. It is a lexicon + rules engine, fully offline:

- **Lexicon**: 139k words × up to 7 espeak-probed context variants
  (mid-clause / clause-final / before-only-unstressed / before-vowel /
  before-pause-word / clause-initial / before-r), harvested offline from
  espeak-ng via piper_phonemize. Bundled as a 0.94 MB raw-DEFLATE
  resource (3.7 MB expanded) + 36 KB aux tables — no downloads.
- **Clause rules** (ported from espeak's `translate.c`/`dictionary.c`
  semantics): multi-word merge entries (`in the` → `ɪnðə`, `did not` →
  `dɪdnˌɑːt`), `$strend2` stress resolution (right-to-left), homograph
  verb/noun/past selection via `expect_verb/noun/past` counters
  (`to record` → `ɹᵻkˈoːɹd` vs `the record` → `ɹˈɛkɚd`), position-aware
  `$pause` handling (blocks liaison/flapping), linking-r, `the/to/a/an`
  vowel-context forms, capital-sensitive rows (`I` pronoun vs `i`
  letter, Polish/polish), all-caps spell-out (`FBI` → `ˌɛfbˌiːˈaɪ`),
  camelCase splitting (`FluidAudio` → `flˈuːɪd ˈɔːdɪˌoʊ`).
- **Normalization**: faithful port of the upstream ZipVoice
  `EnglishTextNormalizer` (abbreviations + inflect-parity numbers:
  `$12.50` → `twelve dollars, fifty cents`, `1855` → `eighteen
  fifty-five`, `21st` → `twenty-first`) — hyphens preserved because
  espeak merges them without a space.

**Measured against the espeak oracle** (1,000-sentence corpus:
conversational + LibriSpeech + numbers/dates/currency + names;
regenerate + score via `mobius/models/tts/zipvoice/coreml/g2p/`):

| Approach | Sentence exact match | Token edit rate |
|---|---|---|
| **Lexicon + rules (shipped)** | **99.6%** (gate ≥ 90%) | **0.01%** (gate ≤ 2%) |
| naive word-by-word lexicon | 3.8% | 5.72% |
| Misaki + symbol mapping (rejected) | 0.5% | 9.75% |

The Misaki mapping layer was measured corpus-wide and rejected: the
divergence from espeak is lexical (different vowel choices, stress
positions, missing length marks), not just symbolic, so no mapping can
close it. The gate is reproducible:

```bash
swift run fluidaudiocli luxtts-g2p-dump --corpus corpus_en_1000.txt \
  --tokens tokens.txt --out swift_dump.jsonl
# in mobius/models/tts/zipvoice:
python -m coreml.g2p.validate score --oracle coreml/g2p/oracle_tokens.jsonl \
  --swift swift_dump.jsonl
```

OOV words fall back to possessive/plural suffix rules, camelCase/all-caps
handling, then letter spell-out; OOV token-id scalars are skipped with a
warning, matching upstream.

## Tests

```bash
swift test --filter LuxTts                        # tokenizer/solver/mel/G2P (fixtures)
FLUIDAUDIO_RUN_LUXTTS_E2E=1 swift test --filter LuxTtsE2ETests   # model-dependent e2e

# Exercise the iOS ane/ graph on a Mac's ANE (downloads the ane/ variant):
FLUIDAUDIO_LUXTTS_VARIANT=ane FLUIDAUDIO_RUN_LUXTTS_E2E=1 \
  swift test --filter LuxTtsE2ETests
```

Fixtures are generated by
`mobius/models/tts/zipvoice/coreml/dump_swift_fixtures.py` and live in
`Tests/FluidAudioTests/TTS/LuxTts/Resources/`. G2P expectations in
`LuxTtsG2pTests` are espeak-oracle outputs from
`mobius/models/tts/zipvoice/coreml/g2p/validate.py dump-oracle`; the
corpus-level gate is scored with `luxtts-g2p-dump` + `validate.py score`
(see the G2P section above).

## Remaining TODOs

- Long-input chunking across multiple vocoder windows (> 555 generated
  frames currently errors; mel truncation is not allowed).
- Optional VAD-based automatic prompt-silence trimming.
- Non-English text (the G2P is `en-us` only; Mandarin pinyin tokens
  exist in `tokens.txt` but have no frontend).
