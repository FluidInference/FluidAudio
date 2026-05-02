# TTS Benchmarks

> **Setup:** MacBook Air M2 (2022), 16 GB, macOS 26, on AC.
> **Corpus:** [MiniMax Multilingual TTS Test Set][minimax] (100
> phrases / language, CC-BY-SA-4.0) — the same public corpus used
> by [MiniMax-Speech][mms], seed-tts-eval, and Gradium, so numbers
> here are directly paper-comparable.
> **Status:** Kokoro, Kokoro ANE, PocketTTS, Magpie, StyleTTS2 all
> complete the English run; CosyVoice3 completes the full Mandarin
> run.
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
   the whole utterance is rendered". For one-shot backends in this
   slice `ttft_ms == synth_ms`. **PocketTTS** and **Magpie** are
   wired through their respective streaming APIs (`synthesizeStreaming`
   / `synthesizeStream`), so their `ttft_ms` is honest first-frame
   latency.
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
| Kokoro / Kokoro ANE | `minimax-english` | `english` only (`af_heart` voice) |
| PocketTTS   | `minimax-english`  | `english`, `german`, `italian`, `portuguese`, `spanish`, `french` |
| StyleTTS2   | `minimax-english`  | `english` only (LibriTTS multi-speaker)        |
| Magpie      | `minimax-english`  | `english`, `spanish`, `german`, `french`, `italian`, `vietnamese`, `chinese`, `hindi` |
| CosyVoice3  | `minimax-chinese`  | `chinese`, `cantonese`                         |

Lines beginning with `#` are comments. Custom corpora can still be
passed with `--corpus-path <file.txt>`.

### Metrics

Per phrase:
- `ttft_ms` — time-to-first-audio. For one-shot backends this equals
  `synth_ms`. **PocketTTS** is benchmarked through
  `synthesizeStreaming`, so its `ttft_ms` is the timestamp of the first
  80 ms audio frame (1920 samples @ 24 kHz). **Magpie** is benchmarked
  through `synthesizeStream`, so its `ttft_ms` is the first
  `MagpieAudioChunk` emit time (typically ~9.6 s on M2 vs ~15 s for
  full synth).
- `synth_ms` — total synth wall time.
- `audio_ms` — generated audio duration.
- `rtfx` — `audio_ms / synth_ms`.
- `wer`, `cer` — via Parakeet ASR roundtrip on the rendered WAV.
- `stage_ms` — per-stage breakdown (backend-specific keys; populated
  for Kokoro ANE + Magpie; empty for Kokoro / PocketTTS /
  StyleTTS2 / CosyVoice3).
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
(`Mac14,2`, on AC). All English runs use `--compute-units default`,
voice = backend default
(`af_heart` for Kokoro, `alba` for PocketTTS, `John` for Magpie),
corpus = `minimax-english` (100 phrases), Parakeet TDT roundtrip for
WER / CER.

| Backend     | License     | Languages              | Footprint | Cold start | TTFT p50 / p95\*   | Synth p50 / p95     | Agg RTFx | Peak RSS | WER     | CER     | Notes |
|-------------|-------------|------------------------|-----------|------------|---------------------|---------------------|----------|----------|---------|---------|-------|
| Kokoro ANE  | Apache-2.0  | en (af_heart only)     | ~330 MB   | 37.9 s     | 1586 / 2515 ms      | 1586 / 2515 ms      | 5.19×    | 738 MB   | 0.108   | 0.040   | one-shot; per-stage CU sweep, 7-graph pipeline |
| Kokoro      | Apache-2.0  | en (af_heart only)     | ~330 MB   | 92.2 s     | 3113 / 4696 ms      | 3113 / 4696 ms      | 2.02×    | 736 MB   | 0.013   | 0.005   | one-shot; cleanest English ASR roundtrip |
| PocketTTS   | research    | en + de + it + pt + es + fr (6L / 24L) | ~140 / ~520 MB | 6.0 s | **1244 / 4749 ms**  | 8757 / 19174 ms     | 0.61×    | 1503 MB  | 0.014   | 0.006   | **streaming**; TTFT is first 80 ms audio frame |
| StyleTTS2   | MIT         | en (LibriTTS multi-spk) | ~280 MB  | 955 s§     | 6671 / 15990 ms§    | 6671 / 15990 ms§    | 2.72×§   | 963 MB§  | 0.440§  | 0.241§  | full 100/100 `minimax-english` via [misaki→espeak post-pass remap](#styletts2-misaki--espeak-post-pass-remap); ref_s = LibriTTS `696_92939_000016_000006.wav` (StyleTTS2 demo voice) |
| Magpie      | research    | en/es/de/fr/it/vi/zh/hi | ~1.3 GB   | 38.5 s∥    | **9580 / 23796 ms**∥ | 15080 / 29895 ms∥   | 0.64×∥   | 762 MB∥  | 0.056   | 0.033   | **streaming TTFT**: first audio chunk at 9.6 s p50 on M2 (full synth 15.1 s); split-K/V decoder; outputBackings fast path with latched fallback |
| CosyVoice3  | Apache-2.0  | zh (mandarin)          | ~1.5 GB   | 29.2 s†    | 14091 / 23679 ms†   | 14091 / 23679 ms†   | 0.357×†  | 3302 MB† | n/a‡    | 0.017‡  | beta; full `minimax-chinese` (100/100 phrases) for latency / RSS and whisper-large-v3 CER‡; cantonese supported via [auto-chunker](#cosyvoice3-auto-chunker) but not benchmarked (no yue ASR) |

\* TTFT for **PocketTTS / Magpie** is first-frame emit through the
streaming API; the others are one-shot, so `ttft_ms == synth_ms`.

† CosyVoice3 chinese: 100/100, 0 errors, ASR skipped. Cold-start
dropped from 302.7 s to 29.2 s on the warm re-run.

‡ CosyVoice3 CER measured on the **full 100-phrase**
`minimax-chinese` corpus via `whisper-large-v3` (Python CPU FP32,
[`Scripts/whisper_zh_cer.py`](../../Scripts/whisper_zh_cer.py)) on
the WAVs rendered by `tts-benchmark --backend cosyvoice3 --corpus
minimax-chinese --skip-asr --audio-dir <dir>`: **macro CER 1.68%
(0.0168)**, **micro CER 1.84% (0.0184)** across 100 phrases.
Whisper is the source of truth here because Cohere Transcribe q8
hit a `MILCompilerForANE` cache failure on this M2 host and ran on
the CPU+GPU fallback path at RTFx ~0.13× (would have taken multiple
hours for the full 100-phrase set vs. ~70 min for whisper). WER is
omitted because Mandarin has no word boundaries and `WERCalculator`
splits on whitespace, so word-level WER reads near 100% and is
meaningless.

∥ Magpie: streamed via `synthesizeStream`. TTFT (9.6 s p50) is
first-chunk emit; synth (15.1 s p50) is full-utterance wall time —
the 5.5 s gap is the streaming win.

§ StyleTTS2 (**beta** — `StyleTTS2Manager.initialize` emits a
runtime warning): warm-cache run; first cold compile of the
bucketed text_predictor / diffusion_step / decoder graphs is
multi-second. ref_s dumped via
[`06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py).
Read WER **relatively** per the
[WER caveat](#about-the-wer--cer-numbers); StyleTTS2's own demo
notebook reports artifacts on long sentences at default
`alpha/beta/diffusion_steps`.

### Kokoro ANE — per-stage breakdown (default preset, MiniMax-English)

Means across 100 `minimax-english` phrases on M2. Stages map to the
7-CoreML-graph split documented in [KokoroAne.md](KokoroAne.md). Vocoder
+ noise together account for ~92% of synth time, which is the natural
target for any further per-stage compute-unit re-tuning. The MiniMax
mean is meaningfully higher than the prior Harvard-sentences run
because phrases 81–100 are paragraph-length news / story sentences.

| Stage         | Mean ms | % of total |
|---------------|---------|------------|
| `albert`      | 28.2    | 2.0%       |
| `post_albert` | 12.1    | 0.9%       |
| `alignment`   | 1.8     | 0.1%       |
| `prosody`     | 49.2    | 3.5%       |
| `noise`       | 242.6   | 17.4%      |
| `vocoder`     | 1039.8  | 74.4%      |
| `tail`        | 24.6    | 1.8%       |
| **total**     | 1398.4  | 100%       |

### Magpie — per-stage breakdown (default preset, MiniMax-English)

Means across 100 `minimax-english` phrases on M2 (`John` voice, en,
default compute units), captured during the original one-shot
profiling run. `ar_loop` is the umbrella for the per-step
`decoder_step` + `sampler` (so it is not added on top in the total).
`nanocodec` runs concurrently with the AR loop in chunked-streaming
mode, which is why the per-stage means do not sum to total warm-synth
mean. The AR loop dominates the wall clock, and its cost grows
super-linearly with phrase length — long news / story phrases drive
the long-tail p95.

| Stage              | Mean ms |
|--------------------|---------|
| `text_encoder`     | 91      |
| `prefill`          | 281     |
| `ar_loop`          | 17946   |
| └── `decoder_step` | 14840   |
| └── `sampler`      | 3081    |
| `nanocodec`        | 17948   |

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

## StyleTTS2 misaki → espeak post-pass remap

StyleTTS2's LibriTTS checkpoint was trained on **espeak-ng-phonemized**
text, but the in-tree BART G2P (shared with Kokoro) emits **misaki**
output. The 178-token vocab accepts both forms, but the acoustic
embeddings for the misaki ligature glyphs are essentially untrained
noise — every training utterance saw the espeak form.

Four systematic divergences vs. `espeak-ng -v en-us --ipa -q`:

| misaki | espeak-ng | example                  |
|--------|-----------|--------------------------|
| `ʧ`    | `tʃ`      | choice → `tʃˈɔɪs`        |
| `ʤ`    | `dʒ`      | jump   → `dʒˈʌmps`       |
| `ɜɹ`   | `ɝ`       | girl   → `ɡˈɝl`          |
| `əɹ`   | `ɚ`       | over   → `ˈoʊvɚ`         |

Fix: 4-rule post-pass remap in `StyleTTS2Phonemizer.phonemize`, gated
on `.americanEnglish`. Result on `minimax-english`: WER 0.581 →
0.440, CER 0.476 → 0.241, agg-RTFx 2.36× → 2.72× (warm-cache
re-run, so latency / RSS deltas are noise — WER / CER are the real
signal). WER is still 30× worse than Kokoro; remaining errors cluster
on word-level BART mispronunciations and long-tail diffusion artifacts.
Further gains likely need a richer remap layer or swapping BART for
libespeak-ng directly.

## CosyVoice3 Decode budget cap

CosyVoice3's Flow CFM was exported with a fixed input shape of
`[1, 250]` speech tokens (`flowTotalTokens` in
`CosyVoice3Constants.swift:45`). The LLM-Decode AR loop is allowed to
emit up to `flowTotalTokens − N_prompt` tokens before being cut off
(typically ~163 generated tokens after the speech-prompt portion).
At `tokenMelRatio=2 × hiftSamplesPerFrame=480 / sampleRate=24000`
that's **40 ms of audio per generated token**, so the loop produces
**at most ~6.5 s of speech per phrase**, regardless of how long the
input text is.

When the AR loop exits because it ran out of budget (i.e. no EOS
token in `stopRange = 6_561…6_760`) instead of natural termination,
`CosyVoice3Synthesizer` now:

1. Logs a `.warning` (one-shot per phrase) naming the
   `decoded.count / maxNew` budget and the produced audio duration.
2. Sets `CosyVoice3SynthesisResult.finishedOnEos = false`, which the
   benchmark harness surfaces as the `finished_on_eos` field on each
   phrase in the JSON report.

Footprint on the cantonese corpus (`minimax-cantonese`,
100 phrases) **without the chunker**: 80 / 100 phrases would hit the
cap, all producing exactly 163 generated tokens / ~6.5 s of audio.
The mandarin corpus sees a much lower truncation rate because
MiniMax-zh phrases are shorter on average.

The structural fix — re-exporting the Flow CFM from
[`mobius-cosyvoice3`](https://github.com/voicelink-ai/mobius-cosyvoice3)
with a larger fixed input shape (e.g. `[1, 500]`) — is upstream
work; bumping the constant in Swift alone would make the Flow
input/output shapes mismatch at predict time. The shipped workaround
is the call-site [auto-chunker](#cosyvoice3-auto-chunker), which
drops cantonese truncation from 80/100 → 5/100 by splitting long
inputs at clause boundaries and crossfading the results.

Surfaced in
`CosyVoice3Synthesizer.synthesize`
(`Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Synthesize/CosyVoice3Synthesizer.swift`)
and
`CosyVoice3SynthesisResult.finishedOnEos`
(`Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Synthesize/CosyVoice3Types.swift`).

## CosyVoice3 auto-chunker

Re-exporting Flow CFM with a larger fixed input shape is gated on
upstream conversion work. Until that lands, `CosyVoice3TtsManager`
splits long inputs at the call site, synthesizes each chunk
independently, and merges with an 8 ms equal-power cosine crossfade.

**Splitter policy** (`CosyVoice3TextChunker`):

- **Hard enders** commit always: `.`, `!`, `?`, `。`, `！`, `？`,
  `\n`.
- **Soft enders** commit only when the running estimate is at or past
  the budget: `，`, `、`, `；`, `：`, `;`, `,`, ASCII space.
- **Force-split** at `budget + 30` tokens of overshoot if no natural
  boundary appeared (rare; mostly continuous CJK with no
  punctuation).

**Token-rate estimate** (calibrated against minimax-zh + minimax-yue
runs):

| Char class | Tokens / char | Rationale                                                    |
|------------|---------------|--------------------------------------------------------------|
| CJK        | 7.5           | worst-case observed in real generation; varies 5.5–9 per char |
| ASCII      | 1.5           | matches BPE rate on English text                              |
| Other      | 2.5           | conservative for accented Latin / non-CJK Unicode             |

`defaultMaxSpeechTokens` is **110**, leaving margin under the
250-token Flow cap minus typical 60–90 token speech-prompt context.

**Concatenation**: 8 ms equal-power cosine crossfade at 24 kHz
between adjacent chunks; single-chunk path short-circuits to plain
copy.

**Validation** (full `minimax-cantonese`, 100 phrases, M2):

| Metric                                    | Pre-chunker | Post-chunker | Δ          |
|-------------------------------------------|-------------|--------------|------------|
| `finished_on_eos=false` (truncated)       | 80 / 100    | **5 / 100**  | −94%       |
| Longest audio output                      | 6.5 s       | **16.1 s**   | +148%      |
| agg-RTFx                                  | 0.245×      | 0.249×       | +1.6%      |
| TTFT p50                                  | 23.9 s      | 35.7 s       | +49%       |
| TTFT p95                                  | 41.2 s      | 60.5 s       | +47%       |
| Peak RSS                                  | 2016 MB     | 3264 MB      | +62%       |

The 5/100 residual is the long-tail token-rate worst case (some
Cantonese characters generate >9 speech tokens); raising the
per-CJK heuristic further would over-fragment short phrases.
Cleaner fix is the upstream Flow re-export.

