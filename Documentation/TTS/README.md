# TTS Backends — Status & Index

A single page to clarify which text-to-speech backend in FluidAudio is in
what state, what each is good at, and where to read more. For the
canonical model registry across the whole library see
[`../Models.md`](../Models.md); this page is the TTS-only deep view.

## TL;DR Recommendations

| Use case | Recommended backend |
|----------|--------------------|
| Real-time English TTS, multi-voice, SSML, custom lexicon | **Kokoro** |
| Lowest latency / Apple Silicon ANE-resident, single voice | **Kokoro ANE (7-stage)** |
| Streaming frame-by-frame TTS — English + 5 European languages, no phoneme stage | **PocketTTS** |
| English expressive / studio-quality, voice-style cloning | **StyleTTS2** *(beta)* |
| Multilingual (en/es/de/fr/it/vi/zh/hi), 5 built-in speakers | **Magpie** *(slow)* |
| Mandarin zero-shot voice cloning | **CosyVoice3** *(slow, RTFx < 1.0)* |

If you want real-time on Apple Silicon today, pick **Kokoro** or
**PocketTTS**. The other backends are shipped but carry caveats called
out below. The other models were converted base on the request of our community.

## Status Matrix

| Backend | Status | Languages | Voices | RTFx (M-series) | Memory | Highlights | Caveats | Deep dive |
|---------|--------|-----------|--------|-----------------|--------|-----------|---------|-----------|
| **Kokoro** | Ready | English | 48 | > 1.0× | Low (iOS-friendly) | Flow-matching mel + Vocos vocoder. SSML, custom lexicon, long-form chunker. CoreML G2P. | One-shot synthesis (no streaming). | [Kokoro.md](Kokoro.md) |
| **Kokoro ANE (7-stage)** | Ready | English | 1 (`af_heart`) | 3–11× faster than Kokoro | Low | Same 82M weights split into 7 CoreML stages so ANE-friendly layers stay resident on the Neural Engine. | ≤ 510 IPA phonemes per call. No SSML / chunker / custom lexicon. Single voice. | [KokoroAne.md](KokoroAne.md) |
| **PocketTTS** | Ready | English + German / Italian / Portuguese / Spanish / French (v2 packs, 6L and 24L variants; French is 24L only) | 21 built-in voices shared across packs (per-language acoustic embeddings) + voice cloning via Mimi encoder | > 1.0× streaming | Low | Autoregressive frame-by-frame with dynamic audio chunking. No phoneme stage — works directly on text tokens. Streams. Voice cloning is language-agnostic (clone once, reuse across language packs). | One manager per language (no auto language detection); pronunciation is fully model-internal — no IPA / SSML `<phoneme>` / custom-lexicon control. | [PocketTTS.md](PocketTTS.md) |
| **StyleTTS2** | Beta (English) | English (LibriTTS multi-speaker) | Per-utterance ref-style blob (`ref_s.bin`, 256 fp32) | > 1.0× typical | Medium | 4-stage diffusion pipeline: `text_predictor` (ANE) → `diffusion_step_512` (CPU+GPU, ADPM2 + Karras) → `f0n_energy` (ANE) → `decoder` (CPU+GPU, HiFi-GAN). 178-token espeak-ng IPA vocab. English G2P via in-tree Kokoro BART (misaki → espeak remap). | English-only checkpoint. Style-encoder export pending — voices ship as offline `ref_s.bin` blobs. Multilingual G2P fallback exists but is unvalidated. | *(no per-model doc yet — see this README + [`Sources/FluidAudio/TTS/StyleTTS2/`](../../Sources/FluidAudio/TTS/StyleTTS2/))* |
| **Magpie TTS Multilingual** | Not production-ready (slow) | en/es/de/fr/it/vi/zh/hi (8) | 5 built-in | ≈ 0.04× (~25× slower than realtime) | Medium-large | NeMo Magpie 357M, 4-model CoreML pipeline + pure-Swift Local Transformer (Accelerate + BNNS). Custom IPA override via `\|...\|`. ASR-clean on 4/5 speakers. | ~30 s cold first synth; ~96 s warm for an 8-word sentence on M-series. Throughput / MLX backend / CFG perf / Japanese support pending. | [Magpie.md](Magpie.md) |
| **CosyVoice3 (Mandarin)** | Beta (slow) | Mandarin Chinese | Zero-shot from voice prompt | < 1.0× typical | Large | FunAudioLLM CosyVoice3 0.5B zero-shot voice cloning. 4-model CoreML pipeline (Qwen2 LLM prefill + stateful decode + CFM Flow + HiFT vocoder). Swift-native Qwen2 BPE + mmap'd fp16 embedding tables. | Flow stays fp32 / `cpuAndGPU` (fp16 + ANE NaNs through fused `layer_norm`). HiFT sinegen falls back to CPU. Voice prompt assets (speech IDs / mel / spk-emb) precomputed offline. API may change. CLI tags backend `[BETA — slow, RTFx < 1.0]`. | [CosyVoice3.md](CosyVoice3.md) |

> RTFx > 1.0× means faster than realtime (1 s of audio in < 1 s of compute).

## Per-Backend Details

### Kokoro

- **Manager:** `KokoroTtsManager` (`Sources/FluidAudio/TTS/Kokoro/`)
- **Phonemizer:** in-tree CoreML BART G2P (`G2PModel`) for English, Misaki-style IPA, then re-encoded into Kokoro's vocab.
- **Synthesis:** flow matching over mel spectrograms in one pass + Vocos vocoder.
- **Strengths:** multi-voice, SSML, custom lexicon, long-form chunker, low memory.
- **HF:** [`FluidInference/kokoro-82m-coreml`](https://huggingface.co/FluidInference/kokoro-82m-coreml)
- **More:** [Kokoro.md](Kokoro.md), [SSML.md](SSML.md), [voice-quality.md](voice-quality.md)

### Kokoro ANE (7-stage)

- **Manager:** `KokoroAneManager`
- **Same 82M weights** split across 7 CoreML stages (Albert / PostAlbert /
  Alignment / Vocoder on ANE; Prosody / Noise / Tail on CPU+GPU).
- **Origin:** derived (with permission) from
  [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml).
- **Tradeoff:** 3–11× RTFx vs. single-graph Kokoro at the cost of single
  voice (`af_heart`), no chunker (≤ 510 IPA phonemes), no SSML, no
  custom lexicon.
- **HF:** [`FluidInference/kokoro-82m-coreml/tree/main/ANE`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE)
- **More:** [KokoroAne.md](KokoroAne.md)

### PocketTTS

- **Manager:** `PocketTtsManager` / `PocketTtsSynthesizer`
  (`Sources/FluidAudio/TTS/PocketTTS/`)
- **Architecture:** ~155M autoregressive frame-by-frame model, dynamic
  audio chunking, no phoneme stage (works directly on text tokens).
  4 CoreML models (`cond_step`, flow LM, flow decoder, Mimi decoder).
- **Streams:** yes — frame-by-frame, with `makeSession()` for persistent
  voice prefill across multiple utterances.
- **Languages (v2 packs, converted from
  [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)):**

  | Language | Pack IDs | Notes |
  |----------|----------|-------|
  | English | `english` | 6-layer, repo root (legacy layout) |
  | German | `german`, `german_24l` | 6L + 24L |
  | Italian | `italian`, `italian_24l` | 6L + 24L |
  | Portuguese | `portuguese`, `portuguese_24l` | 6L + 24L |
  | Spanish | `spanish`, `spanish_24l` | 6L + 24L |
  | French | `french_24l` | 24L only (no 6L upstream) |

  24-layer packs are higher quality but slower and larger. There is no
  automatic language detection — pick the manager that matches your
  input text. Voice names (`alba`, `anna`, `eve`, `michael`, …) are
  shared across packs; the underlying acoustic embeddings are
  per-language.
- **Voice cloning:** the Mimi encoder is language-agnostic, so you can
  clone a voice once and reuse the resulting `PocketTtsVoiceData`
  across managers configured with different language packs.
- **HF:** [`FluidInference/pocket-tts-coreml`](https://huggingface.co/FluidInference/pocket-tts-coreml)
- **More:** [PocketTTS.md](PocketTTS.md)

### StyleTTS2

- **Manager:** `StyleTTS2Manager` (`Sources/FluidAudio/TTS/StyleTTS2/`)
- **Pipeline (per utterance):**
  1. `text_predictor` (fp16, ANE) → `bert_dur` features + duration
     logits.
  2. `diffusion_step_512` (fp16, CPU+GPU) — ADPM2 sampler with Karras-rho
     schedule, 5 steps + classifier-free guidance.
  3. `f0n_energy` (fp16, ANE) → F0 + energy regression.
  4. `decoder` (fp32, CPU+GPU) — HiFi-GAN waveform synthesis at 24 kHz.
- **Phonemizer:** `StyleTTS2Phonemizer` — in-tree Kokoro BART G2P for
  English (misaki → espeak-ng remap: `A→eɪ I→aɪ O→oʊ W→aʊ Y→ɔɪ ᵊ→ə`).
  Non-English falls back to `MultilingualG2PModel` (CharsiuG2P ByT5),
  unvalidated against the LibriTTS checkpoint.
- **Vocab:** 178-token espeak-ng IPA / stress / length set
  (`text_cleaner_vocab.json`); encode iterates Unicode scalars (not
  grapheme clusters) so combining marks like syllabic-`◌̩` and tie-bar
  `◌͡` are not silently dropped.
- **Voices:** ship offline as `ref_s.bin` blobs (256 fp32 LE, 1024 bytes)
  produced via `mobius-styletts2/scripts/06_dump_ref_s.py`. The
  on-device style encoder export is a follow-up.
- **CLI:**
  ```bash
  swift run fluidaudiocli styletts2 "Hello, world." \
    --voice ref_s.bin --output out.wav
  ```
- **Status notes:**
  - Beta. English-only on the shipped LibriTTS checkpoint.
  - End-to-end ASR roundtrip recognizable but not yet pristine.
  - First-call cold start triggers ANE compilation
    (`anecompilerservice`) for the four CoreML graphs.
  - `initialize()` pre-fetches Kokoro G2P assets so a first-time user
    who has never run Kokoro doesn't hit a cryptic
    `G2PModelError.vocabLoadFailed` deep inside `synthesize`.
- **HF:** [`FluidInference/StyleTTS-2-coreml`](https://huggingface.co/FluidInference/StyleTTS-2-coreml)
- **Conversion repo:** `mobius/models/tts/styletts2/`

### Magpie TTS Multilingual

- **Status:** shipped but **not production-ready** — RTFx ≈ 0.04 on
  M-series; ~30 s cold / ~96 s warm for an 8-word English sentence.
- **Languages:** en, es, de, fr, it, vi, zh, hi.
- **HF:** [`FluidInference/magpie-tts-multilingual-357m-coreml`](https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml)
- **Pending work:** throughput investigation, MLX-backed Local
  Transformer, CFG perf, Japanese support (OpenJTalk + MeCab).
- **More:** [Magpie.md](Magpie.md)

### CosyVoice3 (Mandarin)

- **Status:** beta, slow. End-to-end RTFx < 1.0 typical.
- **Bottlenecks:** Flow stays fp32 / `cpuAndGPU`-only because fp16 + ANE
  NaNs through fused `layer_norm` (CoreMLTools limitation, tracked
  upstream); HiFT sinegen / windowing falls back to CPU.
- **Voice prompts:** precomputed offline via
  `mobius/.../bootstrap_aishell3_voices.py` (speech IDs / mel /
  192-d spk-emb).
- **HF:** [`FluidInference/CosyVoice3-0.5B-coreml`](https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml)
- **More:** [CosyVoice3.md](CosyVoice3.md)

## Evaluated, Not Supported

Backends we converted and tested but chose not to ship:

| Model | Why not |
|-------|---------|
| **KittenTTS** ([#409](https://github.com/FluidInference/FluidAudio/pull/409)) | Inefficient espeak alternatives. Nano (15M) and Mini (82M) variants exist but don't justify the integration cost. |
| **Qwen3-TTS** ([#290](https://github.com/FluidInference/FluidAudio/pull/290), [mobius#20](https://github.com/FluidInference/mobius/pull/20)) | Now 1.1 GB but too slow on-device. |
| **Qwen3-ForcedAligner-0.6B** ([#315](https://github.com/FluidInference/FluidAudio/pull/315), [mobius#21](https://github.com/FluidInference/mobius/pull/21)) | 5-model CoreML pipeline, large footprint, low upstream adoption. |

## Adding a New TTS Backend

The high-level checklist for landing another backend:

1. **Conversion** — sit it in `mobius/models/tts/<backend>/` with a
   reproducible script, a `PRECISION.md`, and a numerical-parity test
   against the PyTorch reference.
2. **CoreML assets** — publish the compiled `.mlmodelc` plus any vocab
   / config under `FluidInference/<backend>-coreml` on HuggingFace.
3. **Swift integration** — add `Sources/FluidAudio/TTS/<Backend>/` with
   a manager actor, an asset downloader (`Repo.<backend>` enum case
   in `ModelNames.swift`), a `TtsBackend` case in
   `Sources/FluidAudio/TTS/TtsBackend.swift`, and a CLI entry under
   `Sources/FluidAudioCLI/Commands/`.
4. **Frontend** — phonemizer / tokenizer; reuse `G2PModel` (English BART)
   or `MultilingualG2PModel` (CharsiuG2P ByT5) where possible.
5. **Tests** — at minimum a smoke test that downloads, synthesizes, and
   asserts the WAV header. Quality tests via ASR roundtrip if feasible.
6. **Docs** — add a deep-dive page in this directory and a row in this
   README's status matrix plus `../Models.md`.

## See Also

- [`../Models.md`](../Models.md) — full model registry across ASR / VAD /
  diarization / TTS.
- [`SSML.md`](SSML.md) — SSML support (Kokoro path).
- [`voice-quality.md`](voice-quality.md) — perceptual quality notes.
