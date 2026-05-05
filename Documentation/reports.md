# Session Report — Mandarin ASR + Kokoro v1.1-zh Noise Fix

Chronological log of CTC zh-CN ASR benchmarking, the Kokoro v1.1-zh
CoreML noise investigation, and follow-up cleanup.

---

## 1. CTC zh-CN Mandarin ASR — THCHS-30 Benchmark

Full 2,495-sample run on the THCHS-30 test split.

- Mean CER **8.23%**, median **6.45%**, RTFx **14.83×**.
- Distribution: 17.4% perfect, 67.1% under 10%, 93.2% under 20%.
- Genre mix: news 34.8%, literary prose 34%, scientific 12%, classical
  3.2% (CER 20.93% — register mismatch). All simplified Chinese, no
  script mismatch.

---

## 2. CTC zh-CN Error Analysis

Top-100 worst samples → 862 substitution errors.

| Class                                | Share |
|--------------------------------------|-------|
| Exact pinyin + tone match            | 61.7% |
| Same syllable, different tone        | 16.7% |
| **Total homophone / near-homophone** | **78.4%** |

The model decodes acoustic → character directly with no pinyin
intermediate, so confusions land on same-sounding characters that look
nothing alike (厂→场, 饰→世, 腾→藤).

Two confirmed annotation mismatches in THCHS-30: `sampleId` 295 and
307, speaker D12.

---

## 3. Beam Search

Wired `ctcBeamSearch` into `CtcZhCnManager` (replaced the slow private
`greedyCtcDecode`), exposed `--beam-width` on the benchmark and
transcribe CLI commands, and fixed `blankId` to `7000` (default `1024`
makes beam search emit garbage on the zh-CN vocab).

A/B at beam-width 10 on the top-100 worst samples: **identical CER to
greedy**. The model is confident in its (wrong) predictions, so all
beams converge. Beam search alone won't rescue homophone substitutions
without LM rescoring.

---

## 4. PR #476 — Full Digest

`feat: Add experimental CTC zh-CN Mandarin ASR` (merged). 11 commits,
15+ files, +2,116 lines.

- Swift 6 concurrency fixes in `SlidingWindowAsrManager`.
- New CTC zh-CN model layer (`CtcZhCnManager`, `CtcZhCnModels`).
- CLI commands: `ctc-zh-cn-transcribe`, `ctc-zh-cn-benchmark`.
- CI workflow `ctc-zh-cn-benchmark.yml`.
- Unit tests for the pure-function layer (normalization, Levenshtein,
  CER).
- Benchmark docs updated with the full 2,495-sample run.

---

## 5. Cohere Transcribe — Digest

Branch `feature/cohere-transcribe-asr`, landed as PR #479.

- New 14-language multilingual ASR (Conformer encoder + Transformer
  decoder, ~2B params).
- Removed legacy TDT Japanese ASR.
- Rolled back the diarizer `EmbeddingSkipStrategy`.
- Removed the public `computeUnits` parameter from the Kokoro TTS API.

---

## 6. Mobius — Cohere Encoder Fix

`mobius/` is a separate research repo (not git-tracked from the
FluidAudio side) that hosts the Cohere CoreML conversion scripts.

The Cohere encoder wasn't going through `coremltools` because of
dynamic ops in NeMo's `ConformerEncoder` outer wrapper. Fix:
`export-encoder-correct.py` wraps `model.encoder` directly, bypassing
the dynamic wrapper.

| Stage                            | Max diff vs reference   |
|----------------------------------|-------------------------|
| Wrapped encoder vs original NeMo | 0.00 (exact)            |
| CoreML export vs wrapped encoder | 0.0039 (fp16 tolerance) |

Output: `encoder_correct_static.mlpackage`, 3.5 GB.

Five additional bugs fixed in the conversion scripts during this work:

1. Validation script pointed at the wrong forward pass.
2. Decoder mask shape didn't match the encoder output mask.
3. KV-cache layout was incompatible between NeMo and the exported
   graph.
4. Static-shape sentinel values were inconsistent across stages.
5. Tokenizer call site assumed Whisper-style token IDs, not Cohere's.

---

## 7. Disk Cleanup

Recovered ~12.5 GB during the export work (superseded encoder bundles,
ONNX models, spare venv, cached Japanese TDT, pip/uv/npm caches).
Noted but left untouched: ~51 GB in `~/Library/Caches/python*`.

---

## 8. Kokoro v1.1-zh CoreML — Background Noise Investigation

Multi-round root-cause investigation of audible high-frequency noise
on the v1.1-zh CoreML build that survived every weight-quantisation,
compute-precision, and dispatch-mode change tried in earlier rounds.
Full details in
[mobius PR #50](https://github.com/FluidInference/mobius/pull/50)
(`report.md` / `TRIALS.md`). Summary:

**Round 1 — eliminations**

| Hypothesis                            | Result                   |
|---------------------------------------|--------------------------|
| Weight precision (int8 → fp32)        | not it                   |
| Compute precision (FLOAT16 → FLOAT32) | not it                   |
| Dispatch (ANE / GPU / CPU)            | not it                   |
| Voice timbre offset                   | not it                   |
| Snake1D `cos` rewrite                 | not it (math-equivalent) |

**Round 2 — narrowing.** `per_stage_diff.py` localised the noise to
the Noise stage. A first-pass hypothesis blamed `sin`-of-large-args
inside `CoreMLSineGenV2`; ruled out by `probe_sinegen_isolated.py`.
The original probe had conflated `rand_ini` randomness with graph
drift.

**Round 3 — root cause.** `probe_noise_fidelity.py` found phase
`max|diff|` of 6.283 — exactly 2π — pointing at `atan2`. CoreML MIL
`atan2(0, negative)` returns `0` instead of `+π` (PyTorch convention).
The phase error appears at the DC bin (`imag = 0` exactly) and the
Nyquist bin (`imag` at fp32 noise floor ~1e-15) whenever `real < 0`.
The π/2π offset propagates through `noise_convs[0]` (a strided conv
mixing 11 frequency bins) into all 256 channels of `x_source_0`,
surfacing as broadband HF noise above 10 kHz. Reproduces on a pure-fp32
`cpuOnly` build, ruling out everything in Round 1.

**Fix** lives in `convert-coreml.py:432-456`
(`CoreMLForwardSTFT.transform`): clip near-zero `imag` to zero and add
a correction mask that patches phase to `+π` whenever
`imag == 0 ∧ real < 0`. `eps = 1e-5` covers both the exact-zero (DC)
and computational-zero (Nyquist) cases without touching legitimate
spectral imag values (typical audio ≥ 1e-3).

**Headline verification (deterministic PT-vs-CoreML diff)**

| Metric                              | Before     | After          |
|-------------------------------------|-----------:|---------------:|
| `x_source_0` rel-rms                | 0.397      | **0.057** (7×) |
| `x_source_0` corr                   | 0.911      | **0.998**      |
| STFT phase max\|diff\|              | 2π         | **0.000**      |
| HF (≥10 kHz) Δ vs PyTorch (zm_009)  | **+2.50 dB** | **−0.01 dB** |
| HF Δ vs PyTorch (zf_001)            | n/a        | **+0.11 dB**   |

Re-export with int8 palettization preserves the fix; ~4× storage
reduction.

---

## 9. Devin Review Comments — Audit

**FluidAudio PR #547 — `feat(tts/kokoro-ane)`.** 7 unique findings + 6
auto-resolution markers. **Still unresolved:**
`KokoroAneVoicePack.swift:46` off-by-one — uses `phonemeCount - 1`
where the documented contract and upstream `convert.py:get_ref_data`
say `phonemeCount`. Adjacent rows are similar utterance-length
buckets, so impact is mild.

**FluidAudio PR #476 — CTC zh-CN ASR.** 5 findings, all addressed.
`saveResults` empty-guard worth re-verifying (no explicit resolution
marker).

**FluidAudio PR #479 — Cohere Transcribe.** 9 findings, **none have
resolution markers**. Highest-risk:

- `numMelBins == 80` test against actual `128`.
- Production `print()` in `CohereAsrManager.swift:155`.
- `melLength == 1` divide-by-zero in per-feature normalization.
- `language:` parameter computed but never plumbed into `encodeAudio()`
  / `generate()`.
- `Int(Float.infinity)` crash in `CohereEncoderTest`.
- Raw `Logger` instead of `AppLogger` in two files.
- `tests.yml` uses both `branches-ignore` and `branches` on
  `pull_request` — invalid per GH Actions docs, breaks iOS build CI.
- `argmaxFromLogits` hardcoded `vocabSize` capacity — OOB read risk.

**Mobius PR #50.** Zero comments — Devin Review not configured on
mobius.

Coverage gap: Devin summaries reference "N additional findings in
Devin Review" only viewable inside their app. Numbers above cover
GitHub-visible inline comments only.

---

## 10. PRs Created or Updated

| Repo       | PR    | Status | Summary |
|------------|-------|--------|---------|
| FluidAudio | #476  | merged | `feat: Add experimental CTC zh-CN Mandarin ASR` — benchmark docs updated with the full 2,495-sample run. |
| FluidAudio | #547  | merged | `feat(tts/kokoro-ane): add laishere 7-stage CoreML chain` — initial KokoroAne backend. |
| FluidAudio | #570  | merged | `feat(tts/kokoro-ane): add Mandarin (v1.1-zh) variant` — variant plumbing + phonemes-bypass synthesis. Defers atan2 noise docs to #569. |
| FluidAudio | #569  | open   | `fix: KokoroAne zh-CN noise reduction via atan2 phase correction` — `KokoroAne.md` note + cache-invalidation guidance. |
| Mobius     | #50   | open   | `feat(tts/kokoro-v1.1-zh): CoreML conversion + atan2 phase noise fix` — model bundles, before/after/reference WAVs, diagnostic scripts, trials documentation. |

---

## 11. Magpie TTS — Nanocodec v1/v2/v3 Versioning

Renamed on-disk nanocodec decoder bundles:

- **v1** — legacy monolithic `nanocodec_decoder.mlmodelc`, retained on
  HF as fallback.
- **v2** — fp16, fast / ANE, opt-in, audibly noisy.
- **v3** — fp32, default, audibly clean.

**FluidAudio commit `4bd31469f`.** `MagpieNanocodecPrecision.fp16`
selects v2, `.fp32` selects v3. `MagpieModelStore` downloads by
precision and falls back to v1. `ModelNames`, `MagpieNanocodec`, and
`MagpieTtsManager` updated.

**Mobius commits `cec7d1f` + `f72ab69`.** `STATUS.md` updated; Phase E
upload list ships only the compiled `.mlmodelc` bundles.

**HuggingFace.** Only `.mlmodelc` is shipped (FluidAudio loads
compiled bundles only); `.mlpackage` source stays in mobius for
recompilation. Uploaded: `nanocodec_decoder_v2.mlmodelc`,
`nanocodec_decoder_v3.mlmodelc`, plus the legacy
`nanocodec_decoder.mlmodelc` for backward compat.

Phase D fusion deferred.

---

## Open follow-ups

- Verify `CtcZhCnBenchmark.saveResults` empty-guard fix on PR #476.
- Decide whether to address PR #547's voice-pack off-by-one in a
  follow-up PR or in the Manager comment.
- Sweep PR #479's 9 unresolved Devin findings.
- Listening test on the deployment app for the Kokoro v1.1-zh fix
  (last item on mobius PR #50's checklist).
