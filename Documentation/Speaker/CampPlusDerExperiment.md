# CAM++ streaming-diarization DER experiment (2026-06-11)

**Question:** does CAM++ (PR #652) improve diarization DER as the streaming
pipeline's embedding extractor, vs the shipped WeSpeaker model?

**Answer: no — it regresses badly (+21.7 pp avg at default thresholds; still
2.5× worse at its own best threshold).** CAM++ remains a good *verification*
model; it is the wrong shape for this pipeline.

## Setup
7 cached AMI-SDM meetings (EN2002a–d, ES2004a–c; downloader for the rest
hangs — documented), streaming mode, 5 s chunks / 0 s overlap, identical
files both arms. CAM++ wired behind `FLUID_CAMPP_EMBED=1`: per-speaker
embeddings from mask-gathered active audio (≥12 000 samples; the model needs
≥64 fbank frames), 192-d, cosine clustering unchanged.

## Results (assignment threshold 0.84 = WeSpeaker's tuning)

| Meeting | WeSpeaker | CAM++ | Δ |
|---|---|---|---|
| EN2002a | 63.4 | 57.4 | −6.0 |
| EN2002b | 46.4 | 76.8 | +30.4 |
| EN2002c | 40.6 | 61.6 | +21.0 |
| EN2002d | 62.4 | 82.8 | +20.4 |
| ES2004a | 17.0 | 50.0 | +33.0 |
| ES2004b | 57.6 | 67.4 | +9.8 |
| ES2004c | 25.6 | 69.4 | +43.8 |
| **Avg** | **44.8** | **66.5** | **+21.7** |

Threshold sweep (`--assignment-threshold`, ES2004a): 0.55→42.9, 0.65→50.5,
0.75→50.2, 0.84→50.0. Best CAM++ (42.9) vs baseline 17.0 — calibration is
not the dominant cause. (Note: `--threshold` does NOT reach the streaming
assignment path; use `--assignment-threshold`.)

## Why
CAM++ is a statistics-pooling verification model that wants clean,
contiguous, single-speaker clips ≥0.66 s (its own validation: full
utterances, cosine 0.74 same / 0.35 different). The streaming pipeline
feeds it concatenated mask-scraps with splice discontinuities, often
≈1–2 s — inflating within-speaker variance until the greedy clusterer
cannot separate speakers. WeSpeaker takes waveform+mask natively in a
fixed 10 s window and is trained for exactly this input.

Baseline observation worth its own line: shipped streaming DER on the
EN2002 family is 40–63 % (SE-dominated) — far above the published
26.2 % average (which sampled friendlier families). This strengthens the
case for routing streaming diarization to LS-EEND (20.7 %).

## Recommendation for PR #652
Merge as the additive verification/embedding backend it claims to be
(2.8 ms/call, clean separation on clips). Do NOT position it as a
diarizer embedding. If a diarizer A/B is ever revisited, run it in the
OFFLINE pipeline (clean 10 s windows fit its input form) with cosine-AHC
for both embedders.
