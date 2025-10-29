# Kokoro TTS – Quick Verify Guide

This guide shows the minimal, reproducible steps to fetch required assets, render comparison WAVs, and compute objective similarity metrics to the original PyTorch Kokoro model. It assumes macOS 14+, Apple Silicon, and Python 3.10.12 via pyenv.

## Essentials

- Environment files (run `uv` from qualityReport directory):
  - `uv.lock`
  - `pyproject.toml`
- Required scripts:
  - `setup_inference_assets.py` (downloads assets to local directory)
  - `verify_all.py` (generates WAVs + metrics)
- All assets download to the qualityReport directory

## Setup

1) Python + env
- `cd Documentation/TTS/qualityReport`
- `pyenv local 3.10.12`
- `uv sync`

2) Download assets
- `uv run python setup_inference_assets.py`
- Downloads to qualityReport directory:
  - CoreML: `kokoro_24_15s.mlpackage`, `kokoro_21_10s.mlpackage`, `kokoro_24_10s.mlpackage`
  - Voice: `voices/af_heart.pt`
  - ONNX: `kokoro-v1.0.onnx` (only if a local tarball is present)

## Generate + Score (One Command)

- `uv run python verify_all.py`
- What it does:
  - Generates WAVs for a fixed test sentence:
    - PyTorch reference, ONNX (if runtime available), CoreML v24 15s, CoreML v21 10s, and optional v24 10s
  - Computes core-set metrics + CLAP + fused `sim_index`
- Writes outputs under `perf_out/batch_technical_obstacles/`:
  - WAVs: `pytorch.wav`, `onnx.wav` (optional), `v24_15s.wav`, `v21_10s.wav`, `v24_10s.wav`
  - Metrics CSV: `metrics_core_set_plus.csv`
  - Metrics table (Markdown): `metrics_table.md`
  - Speed/Memory CSV: `speed_mem_summary.csv` (average latency, audio seconds, RTFx, peak RSS MB)
  - One-file report: `report.md` (Similarity Index ranking, metrics table, and speed/memory if available)
  - Prints a quick Similarity Index ranking to stdout

## Interpreting Metrics (Cheat Sheet)

| Metric | Higher is better | What it captures | Expected ranking (closest →) | Notes |
|---|:---:|---|---|---|
| ECAPA speaker cosine | Yes | Voice/timbre identity (speaker embedding) | ONNX > v24 10s > v24 15s > v21 | Tracks “same voice” similarity |
| MFCC‑cosine DTW | Yes | Spectral envelope, DTW‑aligned | ONNX > v24 10s > v24 15s ≈ v21 | Uses MFCC c1–c13; timing‑robust |
| MFCC‑cos DTW (norm) | Yes | Spectral (duration‑normalized) | ONNX > v24 10s > v21 ≈ v24 15s | Removes rhythm/length bias |
| LSD DTW | No | Log‑mel spectral distance, DTW‑aligned | ONNX < v24 10s < v24 15s < v21 | Lower is better |
| LSD DTW (norm) | No | Spectral (duration‑normalized) | ONNX < v24 10s < v24 15s < v21 | Lower is better |
| MCD dB | No | Mel‑cepstral distortion | ONNX < v24 10s < v24 15s < v21 | Classic cepstral mismatch |
| F0 RMSE (cents) | No | Absolute pitch error | ONNX < v21 ≈ v24 15s < v24 10s | v24 10s can have offset |
| F0 Pearson r (DTW) | Yes | Pitch contour similarity | ONNX ≈ v24 10s > v21 ≈ v24 15s | Shape agreement over offset |
| Duration drift (%) | No | Tempo/length difference | ONNX < v24 (10s/15s) < v21 | Closer durations are better |
| CLAP audio‑audio cosine | Yes | Content‑aware audio similarity | ONNX > v24 (10s/15s) > v21 | Semantic, timing‑tolerant |
| Similarity Index (fused) | Yes | Composite closeness | ONNX > v24 10s > v24 15s > v21 | Weighted blend of above |



## Metric Details (What each means)

- ECAPA speaker cosine (higher):
  - What: Cosine similarity between speaker embeddings from an ECAPA‑TDNN verifier.
  - Why: Captures voice/timbre identity; higher means the same speaker “impression.”

- MFCC‑cosine DTW (higher):
  - What: Cosine similarity along a Dynamic Time Warping path over MFCCs (c1–c13).
  - Why: Measures spectral envelope similarity while allowing for small timing shifts.

- MFCC‑cosine DTW (duration‑normalized) (higher):
  - What: Same as above but after time‑stretching the candidate to the reference length.
  - Why: Removes rhythm/length bias so spectral match is isolated.

- LSD DTW (lower):
  - What: Log‑spectral distance (on log‑mel features) accumulated along a DTW path.
  - Why: Penalizes spectral envelope differences; lower means closer timbre/brightness.

- LSD DTW (duration‑normalized) (lower):
  - What: LSD after time‑normalizing candidate to reference length.
  - Why: Separates spectral mismatch from duration differences.

- MCD dB (lower):
  - What: Mel‑cepstral distortion in dB (DTW over MFCCs with the conventional scaling).
  - Why: Classic vocoder‑era spectral mismatch metric; lower indicates closer spectra.

- F0 RMSE (cents) (lower):
  - What: Root‑mean‑square error of pitch (F0) in musical cents on overlapping voiced frames.
  - Why: Measures absolute intonation error (offset/scale).

- F0 Pearson r (DTW) (higher):
  - What: Correlation of log‑F0 sequences aligned by DTW, using voiced frames.
  - Why: Captures contour/shape agreement even if there’s an offset.

- Duration drift `dur_pct` (lower):
  - What: Absolute duration delta divided by reference duration.
  - Why: Penalizes tempo/rhythm differences; close durations improve perceived alignment.

- CLAP audio‑audio cosine (higher):
  - What: Cosine similarity of MS‑CLAP embeddings computed on the two WAVs.
  - Why: Content‑aware, robust to small timing shifts; complements speaker/spectral metrics.

- Fused Similarity Index `sim_index` (higher):
  - What: A weighted blend of the normalized metrics above (↑ECAPA/MFCC/CLAP/F0‑r, ↓LSD/MCD/F0‑RMSE/duration).
  - Why: One number that tracks “closest to PyTorch” across timbre, prosody, and content similarity.

## Notes & Troubleshooting (straight to the point)

- ONNX runtime (to include ONNX in verify_all):
  - Apple Silicon: `cd Documentation/TTS/qualityReport && uv run python -m pip install onnxruntime-silicon==1.16.3`
  - Other platforms: `cd Documentation/TTS/qualityReport && uv run python -m pip install onnxruntime`
  - Then rerun: `uv run python verify_all.py`

- Voice asset (no manual step):
  - `setup_inference_assets.py` downloads `voices/af_heart.pt` to the qualityReport directory.
  - Only if you are offline: fetch it once, then optionally `export HF_HUB_OFFLINE=1` to stay offline.

- Speed/RTF: Included by default in `verify_all.py` (writes `speed_mem_summary.csv`). No extra command needed.

## TTSDS (Optional, distributional score)

- What it is: A broader benchmark that compares distributions of features across multiple clips and categories (Speaker, Prosody, Intelligibility, Generic; Environment optional).
- When to use: You want a single, category‑weighted score over many samples or multilingual sets. Not needed for quick “closest to PyTorch” checks on a few utterances.
- Setup: Use a separate venv to avoid torch/torchaudio pin conflicts with the core `uv` env.
