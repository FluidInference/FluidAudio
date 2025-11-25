# wav_compare Progress Log

## 2024-11-25
- Context: focusing on af_heart p0/p1/p2 parity with Python Kokoro reference.
- Changes implemented:
  - Softer trailing-silence trim in `KokoroSynthesizer.swift` (threshold 0.003, keep ~120ms tail) to counter fixed-shape padding from CoreML trace (`audio_length_samples` over-reports).
  - Punctuation-based pauses in `KokoroChunker.swift` (220ms after `.?!`, 140ms after `,;:`, 80ms after `-`, 60ms default) to better match v21 pacing.
- Current cosine distances (speed 1.0):
  - p0: 0.2056 (ref 6.60s, FA 5.99s)
  - p1: 0.2231 (ref 6.52s, FA 5.25s)
  - p2: 0.0842 (ref 2.73s, FA 1.99s)
- Attempts run:
  - Speed sweeps 0.94–1.10 increased distance; best remains speed 1.0.
  - AP-BWE post-process (24k→48k→24k) produced negligible gains and was removed.
- Next possible tweaks (if needed):
  - Add a bit more tail padding for p0/p1.
  - Slightly longer punctuation pauses for single-sentence prompts if we want durations closer to refs.

## 2024-11-26
- What we tried:
  - Speed sweeps (0.94–1.10) for af_heart hurt quality/pacing; retained speed 1.0.
  - Punctuation pauses: 220ms after `.?!`, 140ms after `,;:`, 80ms after hyphens, 60ms default (`KokoroChunker`).
  - Tail trim: soften threshold to 0.003 (~-50dB) and keep ~120ms after last voiced sample (`KokoroSynthesizer`) to cope with CoreML’s baked frame count (traced via `.item()` in v21.py).
  - Random phases: switched from zeroed phases to Gaussian noise (matches PyTorch) and added deterministic seeding to reduce run-to-run variance; zero phases were causing “robotic” artifacts.
  - Embedding loader bug: removed `dict["embedding"]` fallback so numeric indices (1–510) are honored; prior fallback returned the same embedding for all phoneme counts and blurred speaker style.
  - AP-BWE post-process (24k→48k→24k) yielded no improvement; removed outputs.
- Current distances (af_heart, speed 1.0, after fixes):
  - p0: ~0.22–0.24 (ref 6.60s, FA ~6.0s)
  - p1: ~0.24–0.29 (ref 6.52s, FA ~5.2s)
  - p2: ~0.09–0.11 (ref 2.73s, FA ~2.0s) — solid
- Key learnings:
  - HF embedding files match; the mismatch came from Swift’s loader using the wrong key path (fixed).
  - Speed factors and AP-BWE don’t help; pacing/tail/pauses provide modest gains.
  - Remaining gap is p0/p1 duration/trim; likely need a bit more tail padding or pause tuning to get below 0.2 without altering models.
