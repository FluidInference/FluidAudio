# LocalVQE Demo (macOS)

SwiftUI app that exercises `LocalVqeManager` / `LocalVqeStream` from the
parent package: acoustic echo cancellation, noise suppression and
dereverberation on 16 kHz speech.

```bash
cd Examples/LocalVQEDemo
swift run -c release LocalVQEDemo
```

No Xcode project is needed; the executable target builds with SwiftPM and
opens a regular window. The first **Load model** downloads the selected
variant from `FluidInference/localvqe-coreml`.

## Files (offline)

Pick a mic recording and, optionally, the far-end signal the loudspeaker was
playing, then **Enhance**. **Play before → after** plays the mic input then the enhanced output; **Transcribe all** runs Parakeet TDT v3 on every row so the echo words, the near-end words and what survives can be compared as text. The rows are also individually playable and
are written automatically as WAVs to `~/Downloads/LocalVQEDemo/<run>/` (**Show WAVs in Finder** opens the folder). **Use sample pair** loads a benchmark clip (default fileid 1148, a clear win; the menu also offers fileid 0, a hard case where near-end speech is lost) from the
AEC-Challenge synthetic set if `fluidaudiocli enhance-benchmark` has fetched it.

## Live (mic + speaker)

Plays a far-end file through the default output while capturing the default
input, pairs the two by elapsed time in 256 ms steps, and streams them through
`LocalVqeStream`. The meters show mic, far-end and enhanced level per step:
while only the playback is audible, Enhanced should sit well below Mic; when
you speak, it should follow your voice. **Stop** flushes the stream and offers
the three captured clips for listening, written to the same Downloads folder.

Use the built-in speaker and mic without headphones so the mic actually hears
the playback. Microphone permission is attributed to the terminal that
launched the app when it is run with `swift run`.

The far-end reference is the signal rendered by the app's own player, aligned
to the mic by elapsed time. The device round-trip latency is left to the
model's built-in delay search; production integrations should measure it.
