# TTS Benchmarks

> **Setup:** MacBook Air M2 (2022), 16 GB, macOS 26, on AC.
> **Corpus:** [MiniMax Multilingual TTS Test Set][minimax] (100
> phrases / language, CC-BY-SA-4.0) — the same public corpus used
> by [MiniMax-Speech][mms], seed-tts-eval, and Gradium, so numbers
> here are directly paper-comparable.
> **Status:** Kokoro ANE (English + Mandarin), PocketTTS (English),
> Magpie (English), and StyleTTS2 (English, zero-shot) all complete
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
| Kokoro / Kokoro ANE | `minimax-english` | `english` (`af_heart`); Kokoro ANE also ships `chinese` (`--variant mandarin`, voice `zf_001`) |
| PocketTTS   | `minimax-english`  | 6L packs: `english`, `german`, `italian`, `portuguese`, `spanish`. 24L packs: `french_24l`, `german_24l`, `italian_24l`, `portuguese_24l`, `spanish_24l` |
| Magpie      | `minimax-english`  | `english`, `spanish`, `german`, `french`, `italian`, `vietnamese`, `chinese`, `hindi` |
| StyleTTS2   | `minimax-english`  | `english` only (LibriTTS iteration_3, zero-shot from `--reference` audio) |

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
  Magpie, StyleTTS2** are batch / one-shot
  (`synthesize(...)` returns the full waveform), so `ttft_ms ==
  synth_ms == time-to-complete-wav` for those — interpret it as
  full-wav latency, not as TTFA.
- `synth_ms` — total synth wall time.
- `audio_ms` — generated audio duration.
- `rtfx` — `audio_ms / synth_ms`.
- `wer`, `cer` — via Parakeet ASR roundtrip on the rendered WAV.
- `stage_ms` — per-stage breakdown (backend-specific keys; populated
  for Kokoro ANE; empty for Kokoro / PocketTTS / Magpie /
  StyleTTS2 in this report).
- Backend-specific extras: `encoder_tokens`, `acoustic_frames`,
  `chunk_count`, `frame_count`, `code_count`, `finished_on_eos`,
  `generated_token_count`, etc.

Aggregates:
- `cold_start_s` — `manager.initialize()` wall time. CosyVoice3 also
  includes voice-asset load.
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
```

The harness writes a JSON report to `--output-json` and (optionally)
keeps WAVs under `--audio-dir`. Pass `--skip-asr` to drop the ASR
roundtrip. The default ASR backend is `parakeet` for English-only
runs and is skipped for CosyVoice3; pass `--asr-backend cohere
--cohere-model-dir <dir>` to score Mandarin (or any of the 14
Cohere languages) against [Cohere Transcribe](../../Sources/FluidAudio/ASR/Cohere/).

## Results

### Per-backend top-line

Reference machine: **MacBook Air, Apple M2 (2022), 8-core CPU /
8-core GPU / 16-core Neural Engine, 16 GB unified memory, macOS 26**
(`Mac14,2`, on AC). All runs use `--compute-units default`, 100
phrases per language. Voices are backend defaults
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
| Kokoro ANE | Apache-2.0 | en (`af_heart`)           | ~0.33 GB                   | 24 kHz      | 510 phonemes / pass (≈25–30 s of audio)                          | No        | **988 / 2068 ms** | 988 / 2068 ms     | **7.47×** | 1027 MB  | 10.8%  | 4.0%   |
| Kokoro ANE | Apache-2.0 | zh (`zf_001`)             | ~0.33 GB                   | 24 kHz      | 510 phonemes / pass (≈25–30 s of audio)                          | No        | **956 / 1802 ms** | 956 / 1802 ms     | 6.37×     | 685 MB   | n/a‡   | 4.0%‡  |
| PocketTTS  | research   | en (`alba`, 6L pack)      | fp16 ~0.77 / int8 ~0.55 GB | 24 kHz      | 80 ms Mimi frame, streams until EOS (no fixed cap)               | Yes       | **710 / 1496 ms** | 5160 / 9801 ms    | 1.10×     | 1167 MB  | 1.0%   | 0.4%   |
| Magpie     | research   | en (`John`)               | ~1.3 GB                    | 22.05 kHz   | 256 NanoCodec frames / pass (≈11.9 s); sentence-split for longer | No        | 11470 / 26042 ms∥ | 11470 / 26042 ms∥ | 0.87×∥    | 543 MB∥  | 3.8%   | 2.6%   |
| StyleTTS2  | research   | en (LibriTTS iteration_3) | ~0.67 GB¶                  | 24 kHz      | 256 tokens / pass (≈30 s of audio max)                           | No        | 1574 / 3088 ms    | 1574 / 3088 ms    | 4.59×     | 522 MB   | 9.4%   | 4.1%   |

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
100% and is meaningless. Same caveat applies to the CosyVoice3
zh run reported separately in the [decode budget cap](#cosyvoice3-decode-budget-cap)
section. Cohere Transcribe q8 hit a `MILCompilerForANE` cache
failure on this M2 host, so whisper is the local source of truth.

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
