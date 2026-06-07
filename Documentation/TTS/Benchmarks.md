# TTS Benchmarks

> **Setup:** Apple M5 Pro, 24 GB, macOS 26.5 (25F71), on AC.
> Kokoro ANE and StyleTTS2 rows are carried from the M2 reference
> (MacBook Air M2, 16 GB) — they cannot run on this M5 host (see the
> Kokoro libBNNS and StyleTTS2 asset footnotes below).
> **Corpus:** [MiniMax Multilingual TTS Test Set][minimax] (100
> phrases / language, CC-BY-SA-4.0) — the same public corpus used
> by [MiniMax-Speech][mms], seed-tts-eval, and Gradium, so numbers
> here are directly paper-comparable.
> **Status:** PocketTTS (English), Magpie (English), and Supertonic-3
> (English) complete the full 100-phrase MiniMax run on **M5 Pro**.
> Kokoro ANE (English + Mandarin) and StyleTTS2 (English, zero-shot)
> are **M2-only** here — Kokoro segfaults inside Apple's `libBNNS`
> on M5/macOS 26.5 and StyleTTS2's bucketed BERT assets ship without
> `model.mil`; both rows carry their M2 numbers.
>
> [minimax]: https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set
> [mms]: https://arxiv.org/abs/2505.07916

## Why not just RTFx?

RTFx (audio_seconds / synth_seconds) is a useful single number for batch
synthesis, but for conversational use it hides the things users actually
feel:

1. **Cold start** — first model load + ANE compile after install or
   reboot. On Apple Silicon the system's `anecompilerservice` can take
   tens of seconds on first invocation; subsequent loads finish in ~1 s.
2. **TTFT (time-to-first-audio)** — for streaming agents the question
   is "how long until the user hears *something*", not "how long until
   the whole utterance is rendered". For one-shot / batch backends in
   this slice `ttft_ms == synth_ms`. **PocketTTS** is wired through
   its streaming API (`synthesizeStreaming`), so its `ttft_ms` is
   honest first-frame latency. Magpie is batch-only — its `ttft_ms`
   equals `synth_ms`.
3. **Per-stage compute units** — Kokoro ANE / Magpie are pipelines of
   6–7 graphs. Sometimes ANE is *slower per call* but more efficient.
   The "right" compute-unit choice differs per stage.
4. **Memory footprint** — drives whether a backend is mobile-viable.
5. **Quality** — RTFx alone tells you nothing about whether the model
   pronounced "Reykjavík" or "$1,234.56" correctly. We measure WER +
   CER via Parakeet roundtrip on a fixed English corpus; non-English
   backends run with `--skip-asr` for now.

## Methodology

### Corpus

All shipped corpora come from the **MiniMax Multilingual TTS Test
Set** (`MiniMaxAI/TTS-Multilingual-Test-Set` on Hugging Face,
CC-BY-SA-4.0). The fetched files land under
`Benchmarks/tts/corpus/minimax/<lang>.txt` (24 languages × 100 phrases
= 2400 phrases) and are gitignored — populate them on demand with
`swift run fluidaudio minimax-corpus`. Attribution, revision pin,
and WER caveats live in [`MinimaxCorpus.md`](MinimaxCorpus.md).

Reference each language as `--corpus minimax-<lang>`:

| Backend     | Default corpus     | Other supported MiniMax languages              |
|-------------|--------------------|------------------------------------------------|
| Kokoro ANE  | `minimax-english` | `english` (`af_heart`); Kokoro ANE also ships `chinese` (`--variant mandarin`, voice `zf_001`) |
| PocketTTS   | `minimax-english`  | 6L packs: `english`, `german`, `italian`, `portuguese`, `spanish`. 24L packs: `french_24l`, `german_24l`, `italian_24l`, `portuguese_24l`, `spanish_24l` |
| Magpie      | `minimax-english`  | `english`, `spanish`, `german`, `french`, `italian`, `vietnamese`, `chinese`, `hindi` |
| StyleTTS2   | `minimax-english`  | `english` only (LibriTTS iteration_3, zero-shot from `--reference` audio) |
| Supertonic-3 | `minimax-english` | 31 ISO codes minus `zh`: `english`, `korean`, `japanese`, `arabic`, `bulgarian`, `czech`, `danish`, `german`, `greek`, `spanish`, `estonian`, `finnish`, `french`, `hindi`, `croatian`, `hungarian`, `indonesian`, `italian`, `lithuanian`, `latvian`, `dutch`, `polish`, `portuguese`, `romanian`, `russian`, `slovak`, `slovenian`, `swedish`, `turkish`, `ukrainian`, `vietnamese`. Voice styling via `--voice-style <preset.json>` |

Lines beginning with `#` are comments. Custom corpora can still be
passed with `--corpus-path <file.txt>`.

### Metrics

Per phrase:
- `ttft_ms` — time-to-first-audio. The "first audio" granularity is
  backend-defined; see [Audio chunk window
  size](#audio-chunk-window-size) below for the per-backend numbers.
  **PocketTTS** is benchmarked through `synthesizeStreaming`, so its
  `ttft_ms` is the timestamp of the first 80 ms audio frame (1920
  samples @ 24 kHz) — actually-perceptible TTFA. **Kokoro ANE,
  Magpie, StyleTTS2** are batch / one-shot (`synthesize(...)` returns
  the full waveform), so `ttft_ms == synth_ms == time-to-complete-wav`
  for those — interpret it as full-wav latency, not as TTFA.
- `synth_ms` — total synth wall time.
- `audio_ms` — generated audio duration.
- `rtfx` — `audio_ms / synth_ms`.
- `wer`, `cer` — via Parakeet ASR roundtrip on the rendered WAV.
- `stage_ms` — per-stage breakdown (backend-specific keys; populated
  for Kokoro ANE; empty for / PocketTTS / Magpie /
  StyleTTS2 / Supertonic-3 in this report).
- Backend-specific extras: `encoder_tokens`, `acoustic_frames`,
  `chunk_count`, `frame_count`, `code_count`, `generated_token_count`,
  etc.

Aggregates:
- `cold_start_s` — `manager.initialize()` wall time.
- `first_synth_ms` — first synth call after init (still cold-ish).
- `ttft_ms_p50` / `ttft_ms_p95`.
- `warm_synth_ms_p50` / `warm_synth_ms_p95`.
- `agg_rtfx` — `Σ audio_ms / Σ synth_ms` across the corpus.
- `peak_rss_mb` — process-wide peak resident set, via
  `task_vm_info_data_t.resident_size_peak`.
- Per-category macro WER / CER.

### Audio chunk window size

What counts as "first audio" is backend-defined. The vocoder /
codec emits in fixed-size chunks; only **PocketTTS** is wired to
yield those chunks incrementally on `main`. Everything else returns
the full waveform after the full pipeline runs, so `ttft_ms` for
those backends measures full-wav latency rather than perceptual
TTFA. The consolidated [per-backend table](#per-backend-top-line)
below carries the per-backend sample rate, chunk window, and
streaming flag inline alongside the performance metrics.

For batch backends, "average latency" the user perceives is
`synth_ms` (full wav) rather than `ttft_ms` — they're equal in
that case, so the consolidated table just reports them once.

### Reproducibility

```bash
# From the package root.
swift run fluidaudio tts-benchmark \
  --backend kokoro-ane \
  --corpus minimax-english \
  --voice af_heart \
  --compute-units default \
  --output-json bench.json \
  --audio-dir bench-wavs/

# Kokoro ANE Mandarin (skip Parakeet ASR; whisper CER scored separately).
swift run fluidaudio tts-benchmark \
  --backend kokoro-ane --variant mandarin --voice zf_001 \
  --corpus minimax-chinese --skip-asr \
  --output-json bench-zh.json --audio-dir bench-wavs-zh/

# StyleTTS2 zero-shot (LibriTTS iteration_3). The shipped ref_encoder
# is fixed at [1, 1, 80, 231], so the reference must be exactly
# 2.875 s @ 24 kHz mono. Trim externally before invoking, e.g.:
#   ffmpeg -i speaker.wav -t 2.875 -ar 24000 -ac 1 -c:a pcm_s16le ref.wav
swift run fluidaudio tts-benchmark \
  --backend styletts2 --reference ref.wav \
  --corpus minimax-english \
  --output-json bench-styletts2.json --audio-dir bench-wavs-styletts2/

# Supertonic-3 (multilingual, voice-style JSON). M1.json / F1.json / …
# ship under FluidInference/supertonic-3-coreml/assets/voice_styles/.
# `--lang` defaults are inferred from the corpus name; override with
# `--language <iso>` (e.g. ja, ko, fr). No `zh` — Mandarin is Kokoro ANE.
swift run fluidaudio tts-benchmark \
  --backend supertonic3 --voice-style M1.json \
  --corpus minimax-english \
  --output-json bench-supertonic3.json --audio-dir bench-wavs-sup3/
```

The harness writes a JSON report to `--output-json` and (optionally)
keeps WAVs under `--audio-dir`. Pass `--skip-asr` to drop the ASR
roundtrip. The default ASR backend is `parakeet` for English-only
runs; pass `--asr-backend cohere --cohere-model-dir <dir>` to score
Mandarin (or any of the 14 Cohere languages) against
[Cohere Transcribe](../../Sources/FluidAudio/ASR/Cohere/).

## Results

### Per-backend top-line

Reference machine: **Apple M5 Pro, 24 GB unified memory, macOS 26.5
(`25F71`), on AC** for the PocketTTS, Magpie, and Supertonic-3 rows.
The **Kokoro ANE** (en + zh) and **StyleTTS2** rows are carried from
the prior **MacBook Air, Apple M2 (2022), 8-core CPU / 8-core GPU /
16-core Neural Engine, 16 GB, macOS 26** (`Mac14,2`) reference —
they do not run on the M5 host (Kokoro: Apple `libBNNS` SIGSEGV on
longer phrases in every compute mode; StyleTTS2: missing `model.mil`
in the shipped bucketed BERT graphs — see footnotes ᴷ / ˢ). M5
rows use `--compute-units default`; M2 rows used `--compute-units
default` on M2. 100 phrases per language. Voices are backend defaults
(`af_heart` for Kokoro ANE en, `zf_001` for Kokoro ANE zh,
`alba` for PocketTTS, `John` for Magpie, LibriTTS iteration_3 for
StyleTTS2). English WER / CER via Parakeet TDT roundtrip; Mandarin
CER via `whisper-large-v3`.

One consolidated table per backend × language. **Basic info**
(license, language, footprint, sample rate, max chunk per pass,
streaming flag) is merged with **performance** (TTFT, synth, RTFx,
peak RSS, WER, CER) so there is a single source of truth.

| Backend    | License    | Language (voice)          | Footprint                  | Sample rate | Max chunk per pass                                               | Streaming | TTFT p50 / p95\*  | Synth p50 / p95   | Agg RTFx  | Peak RSS | WER    | CER    |
|------------|------------|---------------------------|----------------------------|-------------|------------------------------------------------------------------|-----------|-------------------|-------------------|-----------|----------|--------|--------|
| Kokoro ANE (M2)ᴷ | Apache-2.0 | en (`af_heart`)           | ~0.33 GB                   | 24 kHz      | 510 phonemes / pass (≈25–30 s of audio)                          | No        | **988 / 2068 ms** | 988 / 2068 ms     | **7.47×** | 1027 MB  | 10.8%  | 4.0%   |
| Kokoro ANE (M2)ᴷ | Apache-2.0 | zh (`zf_001`)             | ~0.33 GB                   | 24 kHz      | 510 phonemes / pass (≈25–30 s of audio)                          | No        | **956 / 1802 ms** | 956 / 1802 ms     | 6.37×     | 685 MB   | n/a‡   | 4.0%‡  |
| PocketTTS (v2.1) | research   | en (`alba`, 6L pack)      | fp16 ~330 MB | 24 kHz      | 80 ms Mimi frame, streams until EOS (no fixed cap)               | Yes       | **26 / 27 ms** | 933 / 1233 ms    | **6.51×** | 761 MB   | 0.51%  | 0.08%  |
| Magpie     | research   | en (`John`)               | ~1.3 GB                    | 22.05 kHz   | 256 NanoCodec frames / pass (≈11.9 s); sentence-split for longer | No        | 3765 / 12110 ms∥ | 3765 / 12110 ms∥ | 2.34×∥    | 832 MB∥  | 3.98%  | 2.44%  |
| StyleTTS2 (M2)ˢ | research   | en (LibriTTS iteration_3) | ~0.67 GB¶                  | 24 kHz      | 256 tokens / pass (≈30 s of audio max)                           | No        | 1574 / 3088 ms    | 1574 / 3088 ms    | 4.59×     | 522 MB   | 9.4%   | 4.1%   |
| Supertonic-3 (int4) | Apache-2.0 | en (`M1`, 31-lang)        | int4 ~0.10 GB                   | 44.1 kHz    | 128 codepoints / pass (chunker splits ≥110 char Latin / 90 CJK)  | No        | **80 / 86 ms** | 80 / 86 ms     | **118×** | 201 MB   | 6.86%ᶜ | 4.08%ᶜ |

ᴷ **Kokoro ANE (M2)** — numbers carried from the M2 reference. Kokoro
**cannot be benchmarked on this M5 Pro / macOS 26.5 host**: a *single*
synthesis works (verified, correct audio, en + zh), but the **2nd+
prediction in a process crashes** — even for the identical short
phrase. The mode depends on routing: `--compute-units default` hits a
GPU `MetalPerformanceShadersGraph GPURNNOps … 'JIT not supported'`
assert (SIGABRT) on the prosody RNN; `all-ane` / `cpu-only` segfault in
`libBNNS.dylib` (`BnnsCpuInferenceOperation::ExecuteSync`, SIGSEGV,
nondeterministic). It is not phrase-length, not ASR contention, and no
compute-unit routing avoids it — an Apple-framework instability on
repeated dynamic-shape predictions, not fixable from FluidAudio.
Tracked in [#667][i667]. Re-baseline once Apple ships a fix (or the
Kokoro graphs are re-exported with bucketed shapes).

ᶜ **Supertonic-3 (int4)** is measured on **M5 Pro / macOS 26.5** with
the default **int4 L-bucketed (ANE) VectorEstimator**. The int4-ANE
path drops per-phrase synth to ~80 ms (agg RTFx 118× over 100 phrases)
vs the prior fp16-dynamic-on-CPU path (5.55× on M2 — the speedup is the
elimination of per-length MPSGraph recompiles, not a chip effect). The
**6.86% WER / 4.08% CER are the actual M5 Parakeet roundtrip** on
`minimax-english` — a regression from the **0.84% measured on M2**
(see the per-language table below, same corpus, long phrases present).
**int4 is not the cause**: on M5, fp16 (7.02%), int8 (6.15%), int6
(6.36%), and int4 (6.86%) all land in the same band. The error is
concentrated in the **chunker**: single-chunk phrases (<110 char) score
**3.88% WER** while multi-chunk phrases (≥110 char) score **8.77%**,
with garbling at chunk seams ("hot air"→"Hudir"). Whether the M2→M5
regression is an Apple-framework numerical change or a chunker code
change since the M2 run is under investigation in [#669][i669].

\* TTFT for **PocketTTS** is first-frame emit through the streaming
API (perceptual TTFA). **Kokoro ANE / Magpie / StyleTTS2** all run
one-shot per phrase (no streaming yield on `main`), so for those
rows `ttft_ms == synth_ms == time-to-complete-wav`.

‡ Kokoro ANE Mandarin CER measured on the **full 100-phrase**
`minimax-chinese` corpus via `whisper-large-v3` (Python CPU FP32,
[`Scripts/whisper_zh_cer.py`](../../Scripts/whisper_zh_cer.py))
against the WAVs rendered by `tts-benchmark --backend kokoro-ane
--variant mandarin --voice zf_001 --corpus minimax-chinese
--skip-asr`: **macro CER 4.01% (0.0401)**, **micro CER 4.14%
(0.0414)** across 100 phrases (table reports the macro figure).
WER is omitted because Mandarin has no word boundaries and
`WERCalculator` splits on whitespace — word-level WER reads near
100% and is meaningless. Cohere Transcribe q8 hit a
`MILCompilerForANE` cache failure on this M2 host, so whisper is
the local source of truth for Mandarin CER.

∥ Magpie: batch-only. `synthesize(...)` returns one
`MagpieSynthesisResult` after the full AR + codec pipeline completes,
so `ttft_ms == synth_ms`. Long inputs are sentence-split internally
(NanoCodec 256-frame static cap) and AR(N+1) ‖ codec(N) chunk-level
pipelining overlaps the next chunk's AR loop with the current chunk's
codec pass — wallclock optimization, not incremental yield. The
sub-1.5 s TTFA work referenced in issue #590 (fused sampler +
24-frame cap) lives on `feat/magpie-lt-fusion`, not `main`.

¶ StyleTTS2 footprint is the sum of the shipped iteration_3 mlpackages
(text encoder + bert + ref_encoder + post_albert + alignment + prosody
+ noise + decoder + tail). The shipped ref_encoder is exported with
a fixed `[1, 1, 80, 231]` mel shape, so reference audio must be
exactly 2.875 s @ 24 kHz (300-hop). The benchmark harness expects
the caller to trim externally; mismatched durations error out at
predict time.

ˢ **StyleTTS2 (M2)** — numbers carried from the M2 reference. StyleTTS2
**cannot be re-baselined on M5** with the current assets: the shipped
length-bucketed BERT graphs (`bert_fp16_t64`, `bert_fp16_t128`,
`bert_fp16_t256`) are missing `model.mil` (only `weights` + metadata
present), so CoreML fails with `corruptedModel … Error in reading the
MIL network`. Only the base `bert_fp16.mlmodelc` is intact. This is a
broken upstream asset (re-download does not help), not an M5 issue.
Tracked in [#668][i668]; re-baseline once the bucketed `model.mil`
files are re-uploaded.

### Kokoro ANE — per-stage breakdown (default preset, MiniMax-English)

Means across 100 `minimax-english` phrases on M2 (`af_heart`,
post-laishere 7-graph chain). Stages map to the 7-CoreML-graph split
documented in [KokoroAne.md](KokoroAne.md). Vocoder + noise together
account for ~90% of synth time, which is the natural target for any
further per-stage compute-unit re-tuning.

| Stage         | Mean ms | % of total |
|---------------|---------|------------|
| `albert`      |   24.5  |  2.5%      |
| `post_albert` |    9.3  |  1.0%      |
| `alignment`   |    1.4  |  0.1%      |
| `prosody`     |   40.0  |  4.1%      |
| `noise`       |  169.3  | 17.5%      |
| `vocoder`     |  704.4  | 72.9%      |
| `tail`        |   17.0  |  1.8%      |
| **total**     |  965.9  | 100%       |

### Magpie — per-stage breakdown

Per-stage timings (`text_encoder`, `prefill`, `ar_loop`,
`decoder_step`, `sampler`, `nanocodec`) are still populated on
`MagpieSynthesisResult.timings` for callers that want them — see
[`MagpieTypes.swift`](../../Sources/FluidAudio/TTS/Magpie/MagpieTypes.swift).
This document does not currently re-publish the per-stage table on
`main`: the AR loop dominates and its absolute numbers are
in active flux on `feat/magpie-lt-fusion` (fused sampler + 24-frame
NanoCodec cap). Republish here once that branch lands on `main`.

### Supertonic-3 — per-language breakdown (M2, default preset, `M1` voice)

Same harness, same `M1.json` voice style, same default
(`--total-steps 8 --speed 1.05 --compute-units default`). All 10
languages complete the full 100-phrase `minimax-<lang>` run.

> **Note (M2 vs M5):** this breakdown is the **M2** run. On **M5 Pro /
> macOS 26.5** the English WER regresses to **6.86%** (vs **0.84%**
> here) — concentrated in chunker-split phrases and independent of
> VectorEstimator precision. See the ᶜ footnote and [#669][i669]
> before treating the per-language WER/CER as current on M5.

WER / CER for English is from the in-process Parakeet TDT
roundtrip. The nine non-English rows were synthesized with
`--skip-asr` (Parakeet is English-only), then scored offline by
transcribing the saved WAVs with **`mlx-community/whisper-large-v3-turbo`**
and computing WER / CER against the corpus references with
`jiwer` after NFKC + lowercase + punctuation-strip normalization.
Peak RSS is process-wide so the English row is inflated by the
additional Parakeet models held in memory; the nine non-English
rows reflect Supertonic-3 in isolation.

| Language        | Code | Synth p50 / p95   | Agg RTFx | Peak RSS | WER†    | CER†   |
|-----------------|------|-------------------|----------|----------|---------|--------|
| English         | en   | **479 / 6491 ms** | 5.55×    | 679 MB‡  | 0.84%   | 0.32%  |
| Arabic          | ar   | 427 / 5827 ms     | 6.76×    | 396 MB   | 3.81%   | 1.16%  |
| French          | fr   | 926 / 5897 ms     | 5.58×    | 292 MB   | 3.32%   | 1.13%  |
| German          | de   | **313 / 2318 ms** | 8.75×    | 345 MB   | **0.66%** | **0.45%** |
| Italian         | it   | 691 / 4626 ms     | 11.05×   | 486 MB   | **0.64%** | **0.29%** |
| Japanese        | ja   | 643 / 2058 ms     | 8.69×    | 329 MB   | 98.33%§ | 9.30%  |
| Korean          | ko   | 438 / 1599 ms     | **11.36×** | 370 MB | 11.54%§ | 4.23%  |
| Russian         | ru   | **321 / 6563 ms** | 7.97×    | 356 MB   | 4.06%   | 1.30%  |
| Spanish         | es   | **354 / 583 ms**  | **15.90×** | 351 MB | 1.28%   | 0.57%  |
| Vietnamese      | vi   | 776 / 3934 ms     | 7.02×    | 440 MB   | 9.60%   | 8.33%  |

† Non-English WER / CER produced offline with
`mlx-community/whisper-large-v3-turbo` against the saved WAVs;
text normalized (NFKC + lowercase + punctuation strip) before
scoring. English uses the in-process Parakeet TDT roundtrip.

‡ English row includes Parakeet TDT loaded in-process for the
ASR roundtrip; the other nine rows ran with `--skip-asr` so the
RSS column reflects only the four Supertonic-3 graphs + indexer.

§ Japanese / Korean have no whitespace word boundaries; `jiwer.wer`
splits on whitespace, so the Japanese WER is meaningless (CER 9.3%
is the right metric). Korean's WER is borderline meaningful because
the corpus uses spaced words.

### About the WER / CER numbers

The MiniMax corpus mixes short conversational phrases, medium news
headlines, and long narrative paragraphs. WER on the long tail is
sensitive to the ASR + text-normalizer stack (e.g. `"3,5%"` →
`"three point five percent"` vs. `"three and a half percent"`); per
the [upstream community
discussion](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set/discussions/10),
absolute WER is best read **relatively** (backend A vs. backend B on
the same corpus + same ASR + same normalizer) rather than against
raw paper numbers.

[i667]: https://github.com/FluidInference/FluidAudio/issues/667
[i668]: https://github.com/FluidInference/FluidAudio/issues/668
[i669]: https://github.com/FluidInference/FluidAudio/issues/669
