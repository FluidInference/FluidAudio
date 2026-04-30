# TTS Benchmarks

> **Setup:** MacBook Air M2 (2022), 16 GB, macOS 26, on AC.
> **Corpus:** [MiniMax Multilingual TTS Test Set][minimax] (100
> phrases / language, CC-BY-SA-4.0) — the same public corpus used
> by [MiniMax-Speech][mms], seed-tts-eval, and Gradium, so numbers
> here are directly paper-comparable.
> **Status:** Kokoro, Kokoro ANE, PocketTTS, Magpie complete the
> English run; CosyVoice3 completes the full Mandarin run after the
> [HiFT-async-timeout fix](#cosyvoice3-hift-timeout-fix) (HiFT pinned
> to `.cpuAndGPU`); StyleTTS2 verified end-to-end on a 5/5 English
> smoke slice after the
> [`sliceFirstAxis2D` flex-shape fix](#styletts2-flexible-shape-fix)
> (full 100-phrase run pending).
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
   slice `ttft_ms == synth_ms`. **PocketTTS** is wired through
   `synthesizeStreaming`, so its `ttft_ms` is honest first-frame
   latency. Streaming-aware TTFT for Magpie is a follow-up.
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
CC-BY-SA-4.0). The vendored copies live under
`Benchmarks/tts/corpus/minimax/<lang>.txt` (24 languages × 100 phrases
= 2400 phrases). Attribution + reproduction notes are colocated in
`Benchmarks/tts/corpus/minimax/README.md`.

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
  80 ms audio frame (1920 samples @ 24 kHz). Streaming-aware TTFT for
  Magpie is a follow-up.
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
| StyleTTS2   | MIT         | en (LibriTTS multi-spk) | ~280 MB  | 76.1 s§    | 11653 / 22701 ms§   | 11653 / 22701 ms§   | 1.35×§   | 1430 MB§ | 0.420§  | 0.262§  | smoke run only (5/5 MiniMax-en phrases) after [`sliceFirstAxis2D` flex-shape fix](#styletts2-flexible-shape-fix); ref_s dumped via [`06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py) from `af_nicole.wav` — full 100-phrase run pending |
| Magpie      | research    | en/es/de/fr/it/vi/zh/hi | ~1.3 GB   | 19.1 s     | 19834 / 57508 ms    | 19834 / 57508 ms    | 0.41×    | 1233 MB  | 0.056   | 0.033   | streaming-capable but benchmarked one-shot; split-K/V decoder; outputBackings fast path with latched fallback |
| CosyVoice3  | Apache-2.0  | zh (mandarin)          | ~1.5 GB   | 302.7 s†   | 20547 / 31556 ms†   | 20547 / 31556 ms†   | 0.269×†  | 2894 MB† | n/a     | n/a     | beta; full `minimax-chinese` (100/100 phrases) after [HiFT fix](#cosyvoice3-hift-timeout-fix) |
| CosyVoice3  | Apache-2.0  | yue (cantonese)        | ~1.5 GB   | 25.6 s‡    | 20543 / 36133 ms†   | 20543 / 36133 ms†   | 0.270×†  | 3300 MB† | n/a     | n/a     | beta; full `minimax-cantonese` (100/100 phrases); same model as zh, ANE compile cache hot from prior run |

\* TTFT = time to first audio frame. PocketTTS streams 80 ms / 1920-sample
frames at 24 kHz, so TTFT < synth_ms; the gap is the streaming
advantage. All other backends are benchmarked one-shot, so
`ttft_ms == synth_ms` for them.

† CosyVoice3: full `minimax-chinese` and `minimax-cantonese` runs, both
100 / 100 phrases, 0 errors, after the HiFT `.cpuAndGPU` fix. ASR
roundtrip skipped (no Mandarin / Cantonese ASR backend). See
[CosyVoice3 HiFT timeout fix](#cosyvoice3-hift-timeout-fix).

‡ Cantonese cold-start is short because the Mandarin run immediately
beforehand left the ANE compile cache hot. A clean first-time cold
start is dominated by the ANE compile attempts for Decode / Flow that
fall back to `.cpuAndGPU` (~5 min on M2).

§ StyleTTS2: 5/5 `minimax-english` phrases, smoke run only. Cold-start
of 0.06 s reflects warm ANE caches from a prior run in the same
session; first cold compile of the bucketed text_predictor /
diffusion_step / decoder graphs is multi-second. Reference voice
`af_nicole` is a Kokoro voice sample passed through the upstream
`style_encoder` + `predictor_encoder`; WER is high because the voice
isn't in StyleTTS2's training distribution (it's a Kokoro asset
re-purposed as a style ref). The full 100-phrase run is pending.

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
| Magpie        | 0.056     | 0.033     | 19834          | 19834           | 57508           | 0.41×  |
| StyleTTS2§    | 0.420     | 0.262     | 11653          | 11653           | 22701           | 1.35×  |

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

Status: **verified end-to-end** on a 5/5 `minimax-english` smoke slice
with a `ref_s.bin` dumped from `samples/kokoro-english/af_nicole.wav`
via the new
[`mobius-styletts2/scripts/06_dump_ref_s.py`](https://github.com/voicelink-ai/mobius-styletts2/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py)
helper (which wraps the upstream `style_encoder` /
`predictor_encoder` recipe from `99_parity_check.py`). All 5 phrases
synthesized successfully (1.35× agg RTFx, 11.7 s p50 synth, 1.43 GB
peak RSS, 42% macro WER / 26% macro CER on Parakeet TDT roundtrip).

Note: the CoreML runtime still emits a non-fatal `E5RT encountered an
STL exception. msg = tensor_buffer has known strides while the model
has FlexibleShapeInfo` line on stdout at process exit (it's printed
during the implicit deinit of one of the flex-shape graphs —
`f0n_energy` or `G2PEncoder`). The process exits 0, all phrases
write valid WAVs, and the JSON summary is correct. The trip is
cosmetic noise from CoreML's lifecycle, not a synthesis failure.

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

