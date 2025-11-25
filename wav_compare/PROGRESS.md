# wav_compare Progress Log

## 2024-11-25
- Context: focusing on af_heart p0/p1/p2 parity with Python Kokoro reference.
- Changes implemented:
  - Softer trailing-silence trim in `KokoroSynthesizer.swift` (threshold 0.003, keep ~120ms tail).
  - Punctuation-based pauses in `KokoroChunker.swift` (220ms after `.?!`, 140ms after `,;:`, 80ms after `-`, 60ms default).
- Current cosine distances (speed 1.0):
  - p0: 0.2056 (ref 6.60s, FA 5.99s)
  - p1: 0.2231 (ref 6.52s, FA 5.25s)
  - p2: 0.0842 (ref 2.73s, FA 1.99s)
- Speed sweeps (0.94–1.10) increased distance; best remains speed 1.0.
- Next possible tweaks (if needed):
  - Add a bit more tail padding for p0/p1.
  - Slightly longer punctuation pauses for single-sentence prompts if we want durations closer to refs.
