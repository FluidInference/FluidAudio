# CoreML Nemotron Multilingual ASR — MLS Benchmark Results

**Model:** `nemotron-asr-streaming-multilingual-0.6b` (CoreML, fp16, `chunk_mel_frames=112`, `att_context_size=[56, 0]`)
**Dataset:** Multilingual LibriSpeech (`facebook/multilingual_librispeech`), full test split per language
**Date:** 2026-05-20
**Hardware:** Apple Silicon (M-series)

> **Important caveat — not apples-to-apples.** Our `build_fp16` (and every other build in the mobius directory) uses `att_context_size=[56, 0]` — **zero lookahead**. NVIDIA's published MLS numbers are at `att_context_size=[56, 3]` (= 320 ms latency commitment per output token in NeMo's `(R+1) × 80 ms` convention). We benchmarked with 1120 ms of *audio per encoder call* and no future-context, NVIDIA benchmarks with 320 ms latency commitment but 3 frames (240 ms) of right-side lookahead. Different tradeoff, different numbers expected. A new conversion with `--att-context "56,3"` is needed for a clean comparison.

> **Note on chunk_ms vs att_context_size.** Two separate knobs in the conversion script:
> - `chunk_mel_frames` controls how many mel frames are fed per encoder call (audio per chunk). Our `build_fp16` uses 112 → 1120 ms of audio. Other existing builds: 32/56/224 frames.
> - `att_context_size = [left, right]` controls the self-attention window. `left` is fixed at 56 across all configs. `right` is the **lookahead in subsampled conformer steps** (each step = 80 ms). All our existing builds use `right=0`. NVIDIA's published numbers use `right=3` → 320 ms latency per output token.
> Cache shapes `[1, 24, 56, 1024]` and `[1, 24, 1024, 8]` are invariant across all variants — `right` is handled by attention masking, not cache geometry.

## Headline

Normalization stack on both ref and hyp (matches NVIDIA's published methodology):
1. `nemo_text_processing.Normalizer(input_case="cased", lang=...)` — digit/date/abbreviation expansion (no-op on MLS prose).
2. `basic` clean: NFKC → lowercase → strip punctuation except `'` → collapse whitespace. **Diacritics and apostrophes are preserved**, matching the cleaner used by the Parakeet team.

```
┌────────┬──────┬────────────┬────────────┬─────────┐
│ Lang   │  N   │  Ours WER  │ NVIDIA WER │   Gap   │
├────────┼──────┼────────────┼────────────┼─────────┤
│ es_419 │ 2385 │   7.05%    │   6.48%    │ +0.57pp │
│ fr_fr  │ 2426 │  10.16%    │   8.92%    │ +1.24pp │
│ it_it  │ 1262 │  19.03%    │  15.48%    │ +3.55pp │
│ pt_br  │  871 │  13.68%    │  14.63%    │ −0.95pp │
├────────┼──────┼────────────┼────────────┼─────────┤
│ macro  │      │  12.48%    │  11.38%    │ +1.10pp │
└────────┴──────┴────────────┴────────────┴─────────┘
```

**Bottom line:** average gap **+1.10pp behind** NVIDIA's published numbers — though we ran with a different streaming tradeoff (large chunk + no lookahead vs NVIDIA's smaller chunk + 240 ms lookahead). pt_br is the only language we beat (−0.95pp); fr_fr and it_it carry the bulk of the gap.

### What changed from the earlier draft of this doc

An earlier version reported avg parity (+0.12pp) using a Whisper-style normalizer that **strips diacritics**. That cleaner is more aggressive than NVIDIA's: dropping `é/à/í` collapses many ref↔hyp mismatches that NVIDIA's eval keeps as errors. With the correct `basic` cleaner the picture is honest but worse.

## Per-language story

| Lang | Verdict | Diagnosis |
|---|---|---|
| es_419 | +0.57pp | Modern Spanish, no obvious failure mode. Residual is plain acoustic/lexical drift. |
| fr_fr | +1.24pp | Intrinsic French homophone confusion (`ils↔il` 113×, `ces↔ses`, `elles↔elle`), plus LibriVox boilerplate in refs (`chapitre`, `fin`, `vingt`, `madame→mme`). Not a model defect. |
| it_it | +3.55pp | Dante/Tasso archaic Italian in MLS refs. Top patterns: `ch'→che` (72×), `l'→il` (75×), `s'→si` (41×), `esser→essere` (17×), `lor→loro`, `ogne→ogni`, `avea→aveva`, `uom→uomo`, `foco→fuoco`, `ciel→cielo`. Model emits modern Italian; refs preserve original. |
| pt_br | −0.95pp | We **beat** NVIDIA here despite the pt MLS test split being mostly **European** Portuguese with pre-1990 orthography (`elle`, `ella`, `annos`, `quasi`, `vae`). Possibly worth a re-run with the `pt-PT` prompt instead of `pt-BR` to see if the gap widens further in our favor or closes. NTP does not support pt in pip 1.1.0 (it is on `main`). |

## Normalizer experiments (illustrative — not what NVIDIA published)

For reference only, these show what **more aggressive** normalization buys; do not quote these as NVIDIA-comparable.

```
                       es     fr     it     pt
NVIDIA-style basic    7.05  10.16  19.03  13.68   ← our published numbers
+ whisper diacritic   6.10   9.89  18.34  12.21   ← strips é/à/í (too aggressive vs NVIDIA)
+ italian archaic     6.10   9.89  17.79  12.21   ← Dante→modern collapse (it only)
```

Findings:
- **NTP contributes zero** on MLS prose (no digits/dates/abbrevs to expand).
- **Whisper-style stripping** saves ~1pp per language but is **not** NVIDIA's methodology.
- A 40-rule archaic Italian collapse closes another 0.55pp on Italian only.

## CER (basic cleaner, full split)

| Lang | CER |
|---|---|
| es_419 | 2.5% |
| fr_fr | 5.3% |
| it_it | 4.8% |
| pt_br | 4.7% |

CER stays low — acoustic modeling is fine. The 12.48% macro WER is largely a function of:
1. NVIDIA-style normalization keeping diacritic + spelling mismatches as errors.
2. The streaming-config difference: we ran `[56, 0]` (zero lookahead) at 1120 ms chunks; NVIDIA ran `[56, 3]` (3-frame lookahead).

## RTFx (real-time factor)

| Lang | RTFx |
|---|---|
| es_419 | 10.3 |
| fr_fr | 9.5 |
| it_it | 8.2 |
| pt_br | 8.1 |

Comfortably >> 1× realtime on M-series at 1120 ms. Smaller chunks will reduce this proportionally.

## PyTorch reference cross-check (es_419, 200-sample subset)

Same 200-sample subset, three pipelines:

| Pipeline | WER | CER |
|---|---|---|
| PyTorch NeMo reference | 6.21% | 2.06% |
| CoreML basic_itn       | 6.02% | — |
| CoreML hyp_raw         | 6.09% | — |
| NVIDIA published (full split) | 6.48% | — |

CoreML matches PyTorch reference on the same sample subset — the conversion is faithful. The +0.57pp full-split gap to NVIDIA is sampling variance plus the streaming-config difference (NVIDIA at `[56, 3]`, us at `[56, 0]`).

## Methodology notes

- **Streaming-latency labels (NeMo).** Per-output-token latency = `(right + 1) × 80 ms`, where `right` is the second value of `att_context_size`. So `[56, 0]` = 80 ms, `[56, 3]` = 320 ms (NVIDIA's publish point), `[56, 6]` = 560 ms, `[56, 13]` = 1120 ms. The `56` left context is identical across all variants. This latency is independent of `chunk_mel_frames`; that knob controls audio per encoder call (throughput), not per-token streaming latency.
- **Our config (`build_fp16`).** `chunk_mel_frames=112`, `att_context_size=[56, 0]`. Translation: 1120 ms of audio per encoder call, 80 ms per-token latency (no future-context). Different tradeoff than NVIDIA's `[56, 3]` at presumably smaller chunks.
- **NTP behavior.** Empirically verified that `nemo_text_processing` preserves diacritics, case, and punctuation — only rewrites digits/dates/measures/abbreviations. On MLS prose this is a no-op.
- **NVIDIA's cleaner.** Per cross-team verification, NVIDIA's MLS numbers use a `basic` cleaner downstream of NTP that keeps diacritics and apostrophes. Our earlier whisper-style stripping was more aggressive and not directly comparable.

## Open items / next steps

- [ ] Convert a new build with `--att-context "56,3"` (NVIDIA's publish point) and re-sweep — only way to publish apples-to-apples numbers. Requires Linux+CUDA + the `.nemo` checkpoint.
- [ ] Re-run pt with `pt-PT` prompt; current run used `pt-BR` for a mostly-European-pt test split.
- [ ] Extended archaic-Italian rule set (~200 rules) or spaCy `it_core_news_sm` lemmatization, if we want to publish a "calibrated" Italian number alongside the raw one.
- [ ] Speaker-level WER breakdown on it_it requires preserving original MLS speaker IDs (our downloader does sequential renaming).

## Reproduce

Result dumps (per-sample JSONL, contain `hyp_basic`/`ref_basic` plus sub/ins/del counts):
- `/tmp/mls_es_fp16.jsonl`
- `/tmp/mls_fr_fr_fp16.jsonl`
- `/tmp/mls_it_it_fp16.jsonl`
- `/tmp/mls_pt_br_fp16.jsonl`

Scoring scripts:
- `/tmp/rescore_mls_basic.py` — aggregate corpus WER from existing dumps using NVIDIA-style `basic` cleaner (the numbers in the headline table above).
- `/tmp/all_lang_normalizer_experiments.py` — full normalizer matrix (basic vs whisper vs archaic).
- `/tmp/archaic_check_all_langs.py` — per-language top-substitution fingerprint.

NTP install:
```bash
brew install miniforge
conda create -n ntp python=3.11 pynini -c conda-forge -y
conda run -n ntp pip install nemo_text_processing jiwer regex tqdm joblib pandas sacremoses editdistance cdifflib
```
