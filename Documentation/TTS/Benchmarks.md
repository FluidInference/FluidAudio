# TTS Benchmarks

> **Setup:** MacBook Air M2 (2022), 16 GB, macOS 26, on AC.
> **Corpus:** [MiniMax Multilingual TTS Test Set][minimax] (100
> phrases / language, CC-BY-SA-4.0) — the same public corpus used
> by [MiniMax-Speech][mms], seed-tts-eval, and Gradium, so numbers
> here are directly paper-comparable.
> **Status:** Kokoro, Kokoro ANE, PocketTTS, Magpie, StyleTTS2 all
> complete the English run; CosyVoice3 completes the full Mandarin
> + Cantonese runs after the
> [HiFT-async-timeout fix](#cosyvoice3-hift-timeout-fix) (HiFT pinned
> to `.cpuAndGPU`) **and** the
> [auto-chunker](#cosyvoice3-auto-chunker) drops cantonese long-phrase
> truncation from 80/100 → 5/100. StyleTTS2 needs the
> [`sliceFirstAxis2D` flex-shape fix](#styletts2-flexible-shape-fix)
> to clear long-phrase synthesis.
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
| CosyVoice3  | `minimax-chinese`  | `chinese` only                                 |

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
- `stage_ms` — per-stage breakdown (backend-specific keys; empty for
  Kokoro / PocketTTS / CosyVoice3).
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

### Compute-unit sweep

`--compute-units` selects how each stage is mapped to silicon:

| Preset         | Mapping                                              |
|----------------|------------------------------------------------------|
| `default`      | Backend's empirically-tuned per-stage layout         |
| `all-ane`      | Force every stage to `.cpuAndNeuralEngine`           |
| `cpu-and-gpu`  | Force every stage to `.cpuAndGPU`                    |
| `cpu-only`     | Force every stage to `.cpuOnly`                      |

Reporting all four for each backend exposes the latency-vs-efficiency
tradeoff. Caveats:

- **Kokoro ANE** has a per-stage compute-unit struct
  (`KokoroAneComputeUnits`) and honors all four presets natively.
- **Kokoro / Magpie / CosyVoice3** take a single `MLComputeUnits`; the
  preset maps via `TtsComputeUnitPreset.uniformUnits`.
- **PocketTTS** does not expose per-call compute-unit overrides — the
  preset is logged and ignored.

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
keeps WAVs under `--audio-dir`. Pass `--skip-asr` to drop the Parakeet
roundtrip. CosyVoice3 forces `--skip-asr`.

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
| StyleTTS2   | MIT         | en (LibriTTS multi-spk) | ~280 MB  | 955 s§     | 6671 / 15990 ms§    | 6671 / 15990 ms§    | 2.72×§   | 963 MB§  | 0.440§  | 0.241§  | full 100/100 `minimax-english` after [`sliceFirstAxis2D` flex-shape fix](#styletts2-flexible-shape-fix) **and** [misaki→espeak post-pass remap](#styletts2-misaki--espeak-post-pass-remap); ref_s dumped via [`06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py) from LibriTTS `696_92939_000016_000006.wav` (StyleTTS2 demo voice) |
| Magpie      | research    | en/es/de/fr/it/vi/zh/hi | ~1.3 GB   | 38.5 s∥    | **9580 / 23796 ms**∥ | 15080 / 29895 ms∥   | 0.64×∥   | 762 MB∥  | 0.056   | 0.033   | **streaming TTFT**: first audio chunk at 9.6 s p50 on M2 (full synth 15.1 s); split-K/V decoder; outputBackings fast path with latched fallback |
| CosyVoice3  | Apache-2.0  | zh (mandarin)          | ~1.5 GB   | 29.2 s†    | 14091 / 23679 ms†   | 14091 / 23679 ms†   | 0.357×†  | 3302 MB† | n/a     | n/a     | beta; full `minimax-chinese` (100/100 phrases) after [HiFT fix](#cosyvoice3-hift-timeout-fix) + [LLM-Decode outputBackings fix](#cosyvoice3-llm-decode-outputbackings-fix) |
| CosyVoice3  | Apache-2.0  | yue (cantonese)        | ~1.5 GB   | 219.8 s‡   | 35681 / 60523 ms¶   | 35681 / 60523 ms¶   | 0.249×¶  | 3264 MB¶ | n/a     | n/a     | beta; full `minimax-cantonese` (100/100 phrases) post [auto-chunker](#cosyvoice3-auto-chunker); **truncation 80/100 → 5/100** (`finished_on_eos=false`); longest output 16.1 s (was capped at ~6.5 s) |

\* TTFT = time to first audio frame. PocketTTS streams 80 ms / 1920-sample
frames at 24 kHz, so TTFT < synth_ms; the gap is the streaming
advantage. All other backends are benchmarked one-shot, so
`ttft_ms == synth_ms` for them.

† CosyVoice3: full `minimax-chinese` run, 100 / 100 phrases, 0 errors,
after the HiFT `.cpuAndGPU` fix **and** the LLM-Decode
`outputBackings` double-buffer fix (32.8% agg-RTFx improvement, 31%
TTFT-p50 improvement, 49% max-synth improvement vs the pre-fix
baseline — see
[CosyVoice3 LLM-Decode outputBackings fix](#cosyvoice3-llm-decode-outputbackings-fix)).
ASR roundtrip skipped (no Mandarin / Cantonese ASR backend).
Cold-start dropped from 302.7 s to 29.2 s because ANE compile caches
were warm on the re-run.

¶ CosyVoice3 cantonese: full `minimax-cantonese` run, 100 / 100
phrases, 0 errors, post [auto-chunker](#cosyvoice3-auto-chunker).
Truncation rate dropped from 80/100 to **5/100** because long phrases
are now split at sentence/clause boundaries, synthesized
independently, and merged with an 8 ms equal-power crossfade. Longest
output is **16.1 s** (vs the previous hard cap of ~6.5 s).
Throughput is lower than the chinese row (0.249× vs 0.357×) because
multi-chunk phrases run multiple full synth passes per phrase —
expected trade-off for unblocking long-form output. TTFT-p50 climbed
correspondingly (35.7 s vs 14.1 s). The 5/100 residual truncation is
pre/post calibration: cantonese token-rate distribution has a longer
worst-case tail than chinese (some chars hit ~9 tokens/char). Lifting
the underlying 250-token Flow cap requires re-exporting from
[`mobius-cosyvoice3`](https://github.com/voicelink-ai/mobius-cosyvoice3)
— tracked separately.

‡ Cantonese cold-start of 219.8 s reflects a fresh ANE compile pass
for Decode / Flow / HiFT after a system reboot — the chinese row
above ran with hot caches from a prior session.

∥ Magpie: full `minimax-english` run, 100 / 100 phrases, 0 errors,
through `MagpieTtsManager.synthesizeStream`. **TTFT (9.6 s p50)** is
first-chunk emit time, **synth (15.1 s p50)** is full-utterance wall
time. The 5.5 s gap is the streaming win — agents can start playback
~37% earlier than waiting for the full synth. agg-RTFx 0.64× is the
warm re-run; the originally documented 0.41× was a cold first-pass.

§ StyleTTS2 (**beta / experimental** — `StyleTTS2Manager.initialize`
emits a runtime beta warning): full 100/100 `minimax-english`
phrases, 0 errors, after the
[misaki→espeak post-pass remap](#styletts2-misaki--espeak-post-pass-remap)
(WER 0.581 → 0.440, CER 0.476 → 0.241; agg-RTFx 2.36× → 2.72×;
peak RSS 1428 MB → 963 MB on the re-run with warm ANE caches).
Cold-start of 0.04 s reflects warm ANE caches from prior runs in
the same session; first cold compile of the bucketed text_predictor
/ diffusion_step / decoder graphs is multi-second. Reference voice
is `696_92939_000016_000006.wav` from the upstream
`yl4579/StyleTTS2-LibriTTS/reference_audio.zip` bundle (a known-good
LibriTTS demo speaker the model was fine-tuned on), dumped to a
256-fp32 `ref_s.bin` via the
[`06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py)
helper. WER (44%) and CER (24%) are still worse than Kokoro /
PocketTTS / Magpie — the model's own demo notebook
(`Demo/Inference_LibriTTS.ipynb`) reports artifacts on long
sentences with the default `alpha=0.3, beta=0.7, diffusion_steps=5`,
and a few MiniMax phrases here produce audio with formant breaks
Parakeet doesn't decode cleanly. RTFx is competitive (2.72×) but
absolute WER is best read relatively (per the
[WER caveat](#about-the-wer--cer-numbers)) rather than against the
StyleTTS2 paper's clean numbers.

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
default compute units). `ar_loop` is the umbrella for the per-step
`decoder_step` + `sampler` (so it is not added on top in the total).
`nanocodec` runs concurrently with the AR loop in chunked-streaming
mode, which is why the per-stage means do not sum to total
warm-synth-mean (24.8 s). The AR loop dominates the wall clock, and
its cost grows super-linearly with phrase length — most of MiniMax's
57.5 s p95 latency comes from the long news / story phrases (max
107 s on a single 18 s utterance).

| Stage              | Mean ms |
|--------------------|---------|
| `text_encoder`     | 91      |
| `prefill`          | 281     |
| `ar_loop`          | 17946   |
| └── `decoder_step` | 14840   |
| └── `sampler`      | 3081    |
| `nanocodec`        | 17948   |

### Per-category quality (MiniMax-English, 100 phrases)

MacBook Air M2 (2022), default compute-units, Parakeet TDT roundtrip.
p50 / p95 are warm-synth latency in ms.

| Backend       | Macro WER | Macro CER | TTFT p50 (ms) | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|---------------|-----------|-----------|----------------|-----------------|-----------------|--------|
| Kokoro ANE    | 0.108     | 0.040     | 1586           | 1586            | 2515            | 5.19×  |
| Kokoro        | 0.013     | 0.005     | 3113           | 3113            | 4696            | 2.02×  |
| PocketTTS     | 0.014     | 0.006     | **1244**       | 8757            | 19174           | 0.61×  |
| Magpie        | 0.056     | 0.033     | **9580**       | 15080           | 29895           | 0.64×  |
| StyleTTS2§    | 0.440     | 0.241     | 6671           | 6671            | 15990           | 2.72×  |

The MiniMax corpus mixes short conversational phrases (1–11) with
medium news headlines (81–100) and long narrative paragraphs (101–110
in the upstream split — vendored as part of the 100-phrase total).
WER on the long tail is sensitive to the ASR + text-normalizer stack
(e.g. `"3,5%"` → `"three point five percent"` vs. `"three and a half
percent"`); per the [upstream community
discussion](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set/discussions/10),
absolute WER on this corpus is best read **relatively** (FluidAudio
backend A vs. backend B on the same corpus + same ASR + same
normalizer) rather than against the raw paper numbers.

## StyleTTS2 flexible-shape fix

StyleTTS2's `text_predictor.mlmodelc` was aborting on long MiniMax
phrases with:

```
E5RT: tensor_buffer has known strides while the model has
FlexibleShapeInfo. Strides must be unknown on all dimensions.
```

…raised from `MLMultiArray objectAtIndexedSubscript:` inside
`StyleTTS2Synthesizer.runTextPredictor`. The CoreML runtime rejects
two access patterns on outputs from a model with `FlexibleShapeInfo`:
`arr.strides` reads, and `arr[idx].floatValue` element subscripts.
The original `sliceFirstAxis2D` helper used both.

Fix: `sliceFirstAxis2D` now reads via `arr.dataPointer.bindMemory(...)`
(handling `.float32`, `.float16`, and `.double`) and computes the flat
index from the known `(1, leading, trailing)` row-major layout instead
of querying `arr.strides`. This matches the existing
`readMLMultiArrayPrefix` pattern used elsewhere in the same file.

Status: **verified end-to-end** on a full 100/100 `minimax-english`
run with a `ref_s.bin` dumped from the upstream LibriTTS demo voice
`696_92939_000016_000006.wav` (from
`yl4579/StyleTTS2-LibriTTS/reference_audio.zip`) via the new
[`mobius-styletts2/scripts/06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py)
helper (which wraps the upstream `style_encoder` /
`predictor_encoder` recipe from `99_parity_check.py`). All 100
phrases synthesized successfully. The flex-shape fix only restored
the run; quality on this corpus stayed at 58% macro WER / 48% macro
CER — see [StyleTTS2 misaki → espeak post-pass remap](#styletts2-misaki--espeak-post-pass-remap)
for the follow-up that pulled WER down to 44% / CER to 24%.

Note: the CoreML runtime still emits a non-fatal `E5RT encountered an
STL exception. msg = tensor_buffer has known strides while the model
has FlexibleShapeInfo` line on stdout at process exit (it's printed
during the implicit deinit of one of the flex-shape graphs —
`f0n_energy` or `G2PEncoder`). The process exits 0, all 100 phrases
write valid WAVs, and the JSON summary is correct. The trip is
cosmetic noise from CoreML's lifecycle, not a synthesis failure.

## StyleTTS2 misaki → espeak post-pass remap

After the `sliceFirstAxis2D` flex-shape fix unblocked the full-corpus
run, StyleTTS2 still landed at WER 0.581 / CER 0.476 on
`minimax-english` — an order of magnitude worse than Kokoro
(0.013) or PocketTTS (0.014). The first hypothesis (silent vocab
drops at `StyleTTS2Vocab.encode`) was disproved by instrumenting the
encoder via `--tokenize-only --corpus`: the full 100-phrase corpus
dropped only **11 / 12247 scalars (0.09%)**, all of them ASCII
hyphens. The 178-token espeak-ng vocab covers ~99.9% of the in-tree
G2P's output.

Real root cause: a **G2P convention mismatch**. Both Kokoro and
StyleTTS2 share the in-tree misaki-style BART G2P (`G2PModel`), but
the Kokoro CoreML graph was authored to consume misaki output 1:1,
while StyleTTS2's LibriTTS checkpoint was trained by yl4579 on
**espeak-ng-phonemized** LibriTTS — predating misaki by years. The
178-vocab accepts both forms (e.g. both `ʧ` U+02A7 and `tʃ`
decomposed are valid encodings), but the acoustic embeddings for
the misaki ligature glyphs are essentially untrained noise — every
training utterance saw the espeak form.

Side-by-side comparison against locally-installed `espeak-ng -v en-us
--ipa -q` flagged four systematic divergences:

| misaki | espeak-ng | example                  |
|--------|-----------|--------------------------|
| `ʧ`    | `tʃ`      | choice → `tʃˈɔɪs`        |
| `ʤ`    | `dʒ`      | jump   → `dʒˈʌmps`       |
| `ɜɹ`   | `ɝ`       | girl   → `ɡˈɝl`          |
| `əɹ`   | `ɚ`       | over   → `ˈoʊvɚ`         |

Fix: a 4-rule post-pass remap in `StyleTTS2Phonemizer.phonemize`,
gated on `.americanEnglish` and applied to the assembled phoneme
string after every word has been emitted by the BART G2P. Lives
alongside the existing per-piece misaki diphthong remap (`A→eɪ`,
`I→aɪ`, etc.); see `StyleTTS2Phonemizer.swift` for both tables and
the rationale.

Result on the same `minimax-english` 100-phrase run, same voice
(`libritts_696`), same Parakeet TDT roundtrip:

| Metric              | Pre-fix | Post-fix | Δ        |
|---------------------|---------|----------|----------|
| Macro WER           | 0.581   | 0.440    | −24.2%   |
| Macro CER           | 0.476   | 0.241    | −49.5%   |
| TTFT p50 (ms)       | 8937    | 6671     | −25.4%   |
| TTFT p95 (ms)       | 17351   | 15990    | −7.8%    |
| Agg RTFx            | 2.36×   | 2.72×    | +15.3%   |
| Peak RSS (MB)       | 1428    | 963      | −32.6%   |

Phrase-level: phrase 1 ("…simple **choice**. Get busy living…") went
from `simple voice. Busy dying.` (0.40 WER) to a perfect roundtrip
(0.00 WER). The latency / RSS improvements ride along because the
re-run hit warm ANE compile caches; the WER / CER deltas are the
real signal.

WER is still 30× worse than Kokoro on this corpus. Remaining errors
cluster on word-level mispronunciations the misaki BART itself
emits non-canonically (e.g. `practical → practicckles`,
`separation → expiration`) and on long-tail phrases where the
diffusion sampler produces formant breaks Parakeet doesn't decode
cleanly. Further gains likely require either a richer remap layer
covering espeak's length marks / function-word reductions, or
swapping the BART G2P for libespeak-ng directly. Tracked separately.

## CosyVoice3 HiFT timeout fix

Earlier `minimax-chinese` runs aborted mid-corpus with:

```
E5RT: Submit Async failed for [3:29]: Async task:
HiFT-T500-fp16_main__Op104_BnnsCpuInference has timed out.
@ CancelTimedOutAsyncTask_block_invoke
```

Root cause: HiFT was loaded with `.cpuAndNeuralEngine`, which let the
CoreML planner place most of the graph on ANE but kept at least one
op (`HiFT-T500-fp16_main__Op104`) on the BNNS CPU async-dispatch
path. Long phrases mid-corpus tripped the BNNS async watchdog (the
same ANE+BNNS mixed-compute pathology that already forced Flow and
LLM-Decode off ANE in `CosyVoice3ModelStore.loadIfNeeded`).

Fix: HiFT is now pinned to `.cpuAndGPU` in
`CosyVoice3ModelStore.loadIfNeeded` regardless of the user-supplied
`computeUnits`, removing the BNNS-async path entirely. The model is
fixed-shape `[1, 80, 500]`, so GPU placement is deterministic.
Trade-off: a small per-call latency increase vs. the ANE config that
didn't actually complete the corpus.

Verified end-to-end on full `minimax-chinese` (100 / 100 phrases) and
`minimax-cantonese` (100 / 100 phrases), 0 errors on either; the
watchdog error did not recur on long phrases.

## CosyVoice3 LLM-Decode outputBackings fix

CosyVoice3's autoregressive decode loop (`LLM-Decode-M768-fp16`) runs
once per speech token — typically ~163 steps per phrase to fill the
250-token `flowTotalTokens` cap minus prompt. Each step takes the
previous step's KV cache as `kv_k` / `kv_v` (shape
`[24, 1, 2, 768, 64]` fp32 = 9 MB each) and produces a fresh
`kv_k_out` / `kv_v_out` plus a `[1, 1, 6562]` `speech_logits`. With
the default CoreML `prediction(from:)` API, that's ~36 MB of fresh
host-side `MLMultiArray` allocation **per step**, per phrase — i.e.
~5.9 GB of allocator churn for a single 6.5 s utterance, all on the
critical path of the AR loop.

Fix: bind pre-allocated `MLMultiArray`s as
`MLPredictionOptions.outputBackings` and rotate them front/back/spare
across decode steps (front = read this step, back = written by this
step, spare = next step's write target). Same try/catch fallback
pattern as
[Magpie's outputBackings fast path](#magpie-outputbackings-fast-path):
on the first rejection from a model exported without explicit
MultiArray shape/dtype output constraints, latch a
`useOutputBackings` flag off and route the rest of the corpus through
the fresh-alloc slow path (with explicit `memcpy` into the back
buffers to keep the rest of the AR loop oblivious to the path
taken). One-shot info log on first acceptance, one-shot warning on
first rejection — no per-step log spam.

Measured on full `minimax-chinese` (100 / 100 phrases, MacBook Air
M2, `--compute-units default`):

| Metric           | Before  | After   | Δ       |
|------------------|---------|---------|---------|
| agg RTFx         | 0.269×  | 0.357×  | +32.8%  |
| TTFT p50         | 20547   | 14091   | −31%    |
| TTFT p95         | 31556   | 23679   | −25%    |
| synth_ms mean    | 21243   | 15999   | −25%    |
| synth_ms median  | 20445   | 14044   | −31%    |
| synth_ms max     | 56497   | 28833   | −49%    |
| peak RSS         | 2894 MB | 3302 MB | +14%    |

The +408 MB RSS is the four pre-allocated 9 MB KV back-buffers plus a
6562-element fp32 logits backing — 36 MB nominal — with the
remainder coming from CoreML retaining backing-tensor metadata
across steps. Fair trade for −33% wall-clock latency. Still
beta-class (RTFx < 1.0) but now 1.33× faster end-to-end. The next
RTFx lever is `MLState` (macOS 15+, would keep KV GPU-resident
across steps), which is gated on bumping the library floor from 14.0
to 15.0.

Implemented in
`CosyVoice3Synthesizer.synthesize` and
`CosyVoice3Synthesizer.runDecode`
(`Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Synthesize/CosyVoice3Synthesizer.swift`).

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
100 phrases): **80 / 100 phrases hit the cap**, all producing
exactly 163 generated tokens / ~6.5 s of audio. The mandarin corpus
sees a much lower truncation rate because MiniMax-zh phrases are
shorter on average.

Lifting the cap requires re-exporting the Flow CFM from
[`mobius-cosyvoice3`](https://github.com/voicelink-ai/mobius-cosyvoice3)
with a larger fixed input shape (e.g. `[1, 500]`) — bumping the
constant in Swift alone would make the Flow input/output shapes
mismatch at predict time. Workaround at the call site: split long
phrases at clause boundaries (CJK `，`, `、`, `。`) and concatenate
the per-clause synthesis results.

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

**Concatenation** (`CosyVoice3TtsManager.concatWithCrossfade`): each
adjacent chunk pair fades over `min(8 ms, prefix.count/2,
next.count/2)` samples at 24 kHz, equal-power cosine
(`down = cos(πj/2N)`, `up = sin(πj/2N)`). Single-chunk and zero-fade
paths short-circuit to plain copy / append.

**Validation** (full `minimax-cantonese`, 100 phrases, M2):

| Metric                                    | Pre-chunker | Post-chunker | Δ          |
|-------------------------------------------|-------------|--------------|------------|
| `finished_on_eos=false` (truncated)       | 80 / 100    | **5 / 100**  | −94%       |
| Longest audio output                      | 6.5 s       | **16.1 s**   | +148%      |
| agg-RTFx                                  | 0.245×      | 0.249×       | +1.6%      |
| TTFT p50                                  | 23.9 s      | 35.7 s       | +49%       |
| TTFT p95                                  | 41.2 s      | 60.5 s       | +47%       |
| Peak RSS                                  | 2016 MB     | 3264 MB      | +62%       |

The TTFT regression is the cost of running multiple synth passes per
long phrase — splitting unblocks long-form output at the price of
wall-clock latency. RSS growth tracks the chunker keeping multiple
intermediate audio buffers live during merge. The 5/100 residual
truncation is the long-tail token-rate worst case (some Cantonese
characters generate >9 speech tokens); raising the per-CJK heuristic
further would over-fragment short phrases. Cleaner fix is the Flow
re-export.

Tests (`Tests/FluidAudioTests/TTS/CosyVoice3TextChunkerTests.swift`):
16 tests covering tokenization estimates, hard/soft/force-split
policy, and the crossfade arithmetic. Validated end-to-end on both
`minimax-chinese` (16/100 truncated, longest 16.0 s) and
`minimax-cantonese` (5/100 truncated, longest 16.1 s) corpora.

Implemented in:
- `Sources/FluidAudio/TTS/CosyVoice3/Pipeline/Preprocess/CosyVoice3TextChunker.swift`
- `Sources/FluidAudio/TTS/CosyVoice3/CosyVoice3TtsManager.swift` (`concatWithCrossfade` + multi-call dispatch)

## Magpie outputBackings fast path

Magpie's `decoder_step.mlmodelc` was previously exported without
explicit `MultiArray` shape/dtype constraints on its KV outputs, which
made CoreML reject the `MLPredictionOptions.outputBackings` map that
`MagpieSynthesizer.runDecoderStep` builds:

```
Output feature (null) doesn't support output backing because it
doesn't have a MultiArray constraints.
```

The synthesizer now wraps the fast path in a try/catch that latches a
`MagpieKvCache.useOutputBackings` flag off on the first rejection,
then falls back to fresh-alloc decoding routed through
`MagpieKvCache.absorbOutputs(_:)` for the rest of the run. Cost: a
single throw/catch on the first AR step + ~18.9 MB of fresh fp16
allocation per subsequent step (no per-step exception spam). The
proper fix remains re-exporting `decoder_step.mlmodelc` from the
Python conversion pipeline (`mobius/models/tts/magpie/coreml/...`)
with explicit `MultiArray` shape + dtype constraints on `new_k_*`,
`new_v_*`, `var_*` outputs — that would let the fast path stay live
and avoid the per-step allocation entirely.

