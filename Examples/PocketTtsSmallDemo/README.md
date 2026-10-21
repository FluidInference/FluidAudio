# Pocket TTS small models demo (macOS 15+)

SwiftUI app for the 18m / 7m / 2m Pocket TTS models (`PocketTtsSmallManager`). Pick a model and a
voice, type a sentence and press Speak: each 80 ms frame is handed to an `AVAudioPlayerNode` as it
comes off the Neural Engine, so audio starts on the first frame and you hear the model keep ahead of
playback. Press Speak again while it talks — further sentences are enqueued on the same warm session.
The table shows, per sentence, time to first audio, frames, audio length, ms per frame and the
real-time factor; the log mirrors to stdout.

```bash
cd Examples/PocketTtsSmallDemo
swift run -c release PocketTtsSmallDemo
```

The first load of a model downloads its pack (34–145 MB) and compiles it for the Neural Engine
(10–150 s depending on size); later loads take well under a second.
