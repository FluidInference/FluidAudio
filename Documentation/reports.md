# Session Report — Mandarin ASR + Kokoro v1.1-zh Noise Fix

Chronological log of the work spanning the CTC zh-CN ASR benchmark, the
Kokoro v1.1-zh CoreML noise investigation, and follow-up cleanup. Each
section captures what was done, headline numbers, and the artifact that
landed.

---

## 1. CTC zh-CN Mandarin ASR — THCHS-30 Benchmark

Full 2,495-sample run of the experimental CTC zh-CN model on the THCHS-30
test split.

**Aggregate metrics**

| Metric          | Value      |
|-----------------|------------|
| Mean CER        | 8.23%      |
| Median CER      | 6.45%      |
| RTFx            | 14.83×     |
| Total samples   | 2,495      |

**CER distribution**

| Bucket          | Share of samples |
|-----------------|------------------|
| Perfect (0%)    | 17.4%            |
| < 10%           | 67.1%            |
| < 20%           | 93.2%            |

**Dataset composition (effect on CER)**

| Genre            | Share | Notes                         |
|------------------|-------|-------------------------------|
| News             | 34.8% | clean baseline                |
| Literary prose   | 34.0% | clean baseline                |
| Scientific       | 12.0% | technical vocabulary          |
| Classical        | 3.2%  | 20.93% CER — register mismatch |

All audio is simplified Chinese — no script-mismatch contribution to CER.

---

## 2. CTC zh-CN Error Analysis

Drilled into the top-100 worst samples (862 substitution errors) to
understand the error pattern.

**Substitution breakdown**

| Class                              | Share |
|------------------------------------|-------|
| Exact pinyin + tone match          | 61.7% |
| Same syllable, different tone      | 16.7% |
| **Total homophone / near-homophone** | **78.4%** |

The model has no pinyin intermediate representation — it does direct
acoustic-to-character decoding, so confusions land on characters that
sound the same even though they look nothing alike.

Top confusion examples: 厂→场, 饰→世, 腾→藤.

**Data-quality findings**

- 2 confirmed annotation mismatches in the THCHS-30 transcripts —
  `sampleId` 295 and 307, both speaker D12. Audio and transcript do not
  match.

---

## 3. Beam Search Implementation & Testing

Wired CTC beam search into the zh-CN path to test whether it reduces the
homophone error rate.

**Changes**

- Replaced the slow private `greedyCtcDecode` in `CtcZhCnManager` with a
  call into `ctcBeamSearch`.
- Added `--beam-width` flag to both the benchmark and `transcribe` CLI
  commands.
- Fixed `blankId`: must be `7000` for the zh-CN vocab, not the default
  `1024`. The wrong blank ID makes beam search emit garbage.

**A/B result on top-100 worst samples**

| Decoder       | Mean CER (top 100) |
|---------------|---------------------|
| Greedy        | baseline            |
| Beam-width 10 | identical to greedy |

Zero improvement. The model is highly confident in its (wrong)
predictions, so all beams converge to the same character. Without an LM
rescore step, beam search alone cannot rescue homophone substitutions.

---

## 4. PR #476 — Documentation Update

Updated `Documentation/Benchmarks.md` with the full 2,495-sample THCHS-30
results and pushed the commits to PR #476.

---

## 5. PR #476 — Full Digest

`feat: Add experimental CTC zh-CN Mandarin ASR` (merged).

| Aspect          | Value             |
|-----------------|-------------------|
| Commits         | 11                |
| Files changed   | 15+               |
| Net additions   | +2,116 lines      |

Scope:

- Swift 6 concurrency fixes in `SlidingWindowAsrManager`.
- New CTC zh-CN model layer (`CtcZhCnManager`, `CtcZhCnModels`).
- CLI commands: `ctc-zh-cn-transcribe`, `ctc-zh-cn-benchmark`.
- CI workflow `ctc-zh-cn-benchmark.yml`.
- Unit tests for the pure-function layer (normalization, Levenshtein,
  CER).
- Documentation updates.

---

## 6. Cohere Transcribe Branch — Digest

Branch `feature/cohere-transcribe-asr`, eventually landed as PR #479.

- New 14-language multilingual ASR backend.
- Architecture: Conformer encoder + Transformer decoder, ~2B parameters.
- Removed the legacy TDT Japanese ASR.
- Rolled back the diarizer `EmbeddingSkipStrategy`.
- Simplified the Kokoro TTS API by removing the `computeUnits` parameter
  on the public surface.

---

## 7. Mobius Research Folder — Investigation

Mapped the layout of the `mobius/` research sandbox.

- Local-only working space, **not** git-tracked from the FluidAudio repo
  side. Lives as a separate `FluidInference/mobius` GitHub repo.
- Hosts the Cohere Transcribe CoreML conversion scripts.
- **Status at the time of inspection:** encoder export was blocked by
  dynamic ops in NeMo's `ConformerEncoder`. `coremltools` rejected the
  trace.

---

## 8. Cohere Encoder Fix (Option 1)

Got the NeMo Conformer encoder through CoreML conversion.

**Approach**

- Rewrote `export-encoder-correct.py` to wrap `model.encoder` directly,
  bypassing the dynamic outer wrapper that was tripping `coremltools`.

**Validation**

| Stage                             | Max diff vs reference |
|-----------------------------------|-----------------------|
| Wrapped encoder vs original NeMo  | 0.00 (exact)          |
| CoreML export vs wrapped encoder  | 0.0039 (fp16 tolerance) |

**Output**

- `encoder_correct_static.mlpackage` — 3.5 GB.

**Bugs found and fixed in the conversion scripts**

1. Validation script pointed at the wrong forward pass.
2. Decoder mask shape did not match the encoder output mask.
3. KV-cache layout was incompatible between NeMo and the exported graph.
4. Static-shape sentinel values were inconsistent across stages.
5. Tokenizer call site assumed Whisper-style token IDs, not Cohere's.

---

## 9. Disk Cleanup

Recovered space after the Cohere encoder export work.

- Deleted superseded encoder exports — ~10.5 GB.
- Deleted spare ONNX models, an extra venv, and the cached Japanese
  TDT model.
- Cleaned pip / uv / npm caches — ~2 GB.
- Identified ~51 GB sitting in `~/Library/Caches/python*` (left for
  the user to clear at their discretion).

---

## 10. Kokoro v1.1-zh CoreML — Background Noise Investigation

Multi-round investigation into audible high-frequency noise on the
Kokoro v1.1-zh CoreML build that survived every weight-quantisation,
compute-precision, and dispatch-mode change tried in earlier rounds.

**Round 1 — eliminations**

| Hypothesis                                      | Result    |
|-------------------------------------------------|-----------|
| Weight precision (int8 → fp32)                  | not it    |
| Compute precision (FLOAT16 → FLOAT32)           | not it    |
| Dispatch (ANE / GPU / CPU)                      | not it    |
| Voice timbre offset                             | not it    |
| Snake1D `cos` rewrite (verified math-equivalent)| not it    |

**Round 2 — narrowing**

- Five-tier stage-swap (`per_stage_diff.py`) localised the noise to the
  Noise stage.
- A first-pass hypothesis blamed `sin`-of-large-args inside
  `CoreMLSineGenV2`. Was ruled out by `probe_sinegen_isolated.py` (six
  standalone CoreML models). The original probe had conflated `rand_ini`
  randomness with graph drift.

**Round 3 — root cause found**

- `probe_noise_fidelity.py` ran a deterministic PT-vs-CoreML diff on the
  noise stage. Phase `max|diff|` came back at 6.283 — exactly 2π,
  pointing at `atan2`.
- CoreML MIL `atan2(0, negative)` returns `0` instead of `+π` (PyTorch
  convention).
- The phase error appears at the **DC bin** (`imag = 0` exactly) and
  the **Nyquist bin** (`imag` at fp32 noise floor ~1e-15) whenever
  `real < 0`. The π/2π offset propagates through `noise_convs[0]`
  (a strided conv mixing 11 frequency bins) into all 256 channels of
  `x_source_0`, surfacing as broadband HF noise above 10 kHz.
- Reproduces on a pure-fp32 `cpuOnly` build, which is what definitively
  ruled out everything in Round 1.

**Fix**

In `convert-coreml.py:432-456` (`CoreMLForwardSTFT.transform`):

```python
eps = 1e-5
imag_clipped = torch.where(imag_out.abs() < eps,
                           torch.zeros_like(imag_out), imag_out)
magnitude = torch.sqrt(real_out ** 2 + imag_clipped ** 2 + 1e-14)
phase = torch.atan2(imag_clipped, real_out)
correction_mask = (imag_clipped == 0) & (real_out < 0)
phase = torch.where(correction_mask,
                    torch.full_like(phase, math.pi), phase)
```

`eps = 1e-5` covers both the exact-zero (DC) and computational-zero
(Nyquist) cases without affecting any legitimate spectral imag value
(typical audio ≥ 1e-3).

**Verification — conversion fidelity**

| Metric                  | Before fix | After fix      |
|-------------------------|-----------:|---------------:|
| `x_source_0` rel-rms    | 0.397      | **0.057** (7×) |
| `x_source_0` corr       | 0.911      | **0.998**      |
| `x_source_1` rel-rms    | 0.070      | **0.002**      |
| `x_source_1` corr       | 0.996      | **1.000**      |
| STFT phase max\|diff\|  | 6.283 (2π) | **0.000**      |

**Verification — end-to-end audio HF (≥10 kHz) band power, `zm_009`**

| Metric                          | Before fix | After fix     |
|---------------------------------|-----------:|--------------:|
| CoreML output                   | −81.31 dB  | **−83.91 dB** |
| PyTorch reference               | −83.81 dB  | −83.90 dB     |
| Δ(CoreML − PyTorch)             | **+2.50 dB** | **−0.01 dB**  |
| HF residual (CoreML − PyTorch)  | −86.55 dB  | −91.18 dB     |

`zf_001` Δ(CoreML − PyTorch): **+0.11 dB** (no regression on the female
voice).

**Re-export with int8 palettization**

- Re-exported the fixed `KokoroNoise.mlpackage` with int8 palettization.
- Fix preserved through palettization.
- Storage reduction: ~4×.

**Artifact**

- Mobius PR #50 — model bundles, before/after/reference WAVs in
  `docs/audio/`, the diagnostic scripts, and the trials documentation
  (`report.md`, `TRIALS.md`).

---

## 11. Devin Review Comments — Audit

Pulled and read every Devin Review inline comment on the relevant PRs.

**FluidAudio PR #547 — `feat(tts/kokoro-ane)`**

- 7 unique findings + 6 auto-resolution markers (13 comments total).
- **1 still unresolved on the merged code:** `KokoroAneVoicePack.swift:46`
  off-by-one — uses `phonemeCount - 1` where the documented contract and
  upstream `convert.py:get_ref_data` say `phonemeCount`. Impact is mild
  (adjacent rows are similar utterance-length buckets) but every
  synthesis call selects from the row below the intended one.

**FluidAudio PR #476 — `Add experimental CTC zh-CN Mandarin ASR`**

- 5 findings — all addressed (some without explicit resolution markers,
  notably the empty-results NaN guard in `saveResults` is worth
  re-verifying).

**FluidAudio PR #479 — `Add Cohere Transcribe`**

- 9 findings, **none have resolution markers**. Highest-risk items to
  re-verify:
  - `numMelBins == 80` test against actual `128`.
  - Production `print()` in `CohereAsrManager.swift:155`.
  - `melLength == 1` divide-by-zero in per-feature normalization.
  - `language:` parameter computed but never plumbed into
    `encodeAudio()` / `generate()`.
  - `Int(Float.infinity)` crash in `CohereEncoderTest`.
  - Raw `Logger` instead of `AppLogger` in two files.
  - `tests.yml` uses both `branches-ignore` and `branches` on
    `pull_request` — invalid per GH Actions docs, breaks iOS build CI.
  - `argmaxFromLogits` hardcoded `vocabSize` capacity — OOB read risk.

**Mobius PR #50**

- Zero Devin Review comments. Devin Review is not configured on the
  mobius repo.

**Caveat — coverage gap.** Devin Review summaries reference
"N additional findings in Devin Review" only viewable inside the Devin
app. The figures above cover the GitHub-visible inline comments only.

---

## 12. PRs Created or Updated

| Repo       | PR    | Status   | Summary                                                                                       |
|------------|-------|----------|-----------------------------------------------------------------------------------------------|
| FluidAudio | #476  | merged   | `feat: Add experimental CTC zh-CN Mandarin ASR` — benchmark docs updated with full 2,495-sample results. |
| FluidAudio | #569  | open     | `fix: KokoroAne zh-CN noise reduction via atan2 phase correction` — docs note + cache-invalidation guidance for existing users (`KokoroAne.md`). |
| Mobius     | #50   | open     | `feat(tts/kokoro-v1.1-zh): CoreML conversion + atan2 phase noise fix` — model bundles, before/after/reference WAVs, diagnostic scripts, trials documentation. |

---

## 13. Magpie TTS — Nanocodec v1/v2/v3 Versioning

### What was done

- Renamed the on-disk nanocodec decoder models to a v2/v3 scheme.
- **v2** = fp16 (fast / ANE, opt-in, audibly noisy).
- **v3** = fp32 (default, audibly clean).
- **v1** = legacy monolithic `nanocodec_decoder.mlmodelc`, retained on HF
  as a fallback.

### Swift changes (FluidAudio commit `4bd31469f`)

- `MagpieNanocodecPrecision` enum: `.fp16` selects v2, `.fp32` selects v3.
- `MagpieModelStore` updated to download v2 / v3 by precision and fall
  back to the legacy v1 bundle.
- `ModelNames` updated with the v2 / v3 filenames.
- `MagpieNanocodec.swift` docstrings updated for v1 / v2 / v3.
- `MagpieTtsManager.swift` updated for the new selection path.

### Mobius changes (commits `cec7d1f` + `f72ab69`)

- `STATUS.md` updated with the v2 / v3 naming.
- Phase E HF upload list expanded to include both the `.mlpackage` and
  `.mlmodelc` for each version.

### HuggingFace upload (completed)

- `nanocodec_decoder_v2.mlpackage` + `nanocodec_decoder_v2.mlmodelc` (fp16)
- `nanocodec_decoder_v3.mlpackage` + `nanocodec_decoder_v3.mlmodelc` (fp32)
- Legacy `nanocodec_decoder.mlmodelc` + `.mlpackage` retained for
  backward compatibility.

### Build artifacts on disk

- `build/nanocodec_decoder_v2.mlpackage` (fp16) + `compiled/build/nanocodec_decoder_v2.mlmodelc`
- `build/nanocodec_decoder_v3.mlpackage` (fp32) + `compiled/build/nanocodec_decoder_v3.mlmodelc`

### Remaining

- Phase D fusion — deferred.
- Phase E HF upload — completed by user.

---

## Open follow-ups

- Verify `CtcZhCnBenchmark.saveResults` empty-guard fix on PR #476.
- Decide whether to address PR #547's voice-pack off-by-one in a
  follow-up PR or in the Manager comment.
- Sweep PR #479's 9 unresolved Devin findings.
- Schedule the listening test on the deployment app for the Kokoro
  v1.1-zh fix (last item on mobius PR #50's checklist).
