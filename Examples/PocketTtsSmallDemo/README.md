# Pocket TTS small models demo (macOS 15+)

SwiftUI app for the 18m / 7m / 2m Pocket TTS models (`PocketTtsSmallManager`), reduced to an A/B
showcase: three reference sentences, each playable from the bundled upstream PyTorch take or rendered
live by Core ML. Each 80 ms frame is handed to an `AVAudioPlayerNode` as it comes off the Neural
Engine, so audio starts on the first frame and the model keeps ahead of playback. The model and Mimi
mode are chosen with launch flags (`--model`, `--mimi`); the model loads on launch.
The table shows, per sentence, time to first audio, frames, audio length, ms per frame and the
real-time factor; the log goes to stdout (run it from a terminal to see it).

- **The sea** at the top is the speech envelope of what the speaker is
  playing (an audio tap on the player node, 20 ms blocks), drawn as three frost-blue swells at display
  rate — it follows playback, not the generator, which runs 20–50x ahead.
- **Mimi path**: `--mimi fp32` (default; FlowLM on the ANE, Mimi decoder fp32 on the CPU, clean) or
  `--mimi fp16-fused` (one fused ANE dispatch per frame: faster, with the −72 dBFS noise floor audible in
  pauses). Relaunch to switch.
- **Reference takes**: three upstream PyTorch renders (18m, fp32 CPU; voices alba / marius / javert,
  each a different sentence) ship in `Sources/PocketTtsSmallDemo/Reference/`. `▶ PyTorch` plays the
  file through the same player and sea view; `▶ Core ML` speaks the same sentence with the same voice
  on the loaded model, so the two are a fair A/B.
- Colours follow the Nord palette (`Theme.swift`).

Launch flags: `--model 18m|7m|2m[-refl]`, `--mimi fp32|fp16-fused`, `--text "…"`, `--autorun`
(load and speak on launch), `--no-audio` (headless timing).

```bash
cd Examples/PocketTtsSmallDemo
swift run -c release PocketTtsSmallDemo
```

The first load of a model downloads its pack (34–145 MB) and compiles it for the Neural Engine
(10–150 s depending on size); later loads take well under a second.
