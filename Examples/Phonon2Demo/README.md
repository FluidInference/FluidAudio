# Phonon-2 Demo (macOS)

SwiftUI app that transcribes a file with Phonon-2 (`AsrModelVersion.phonon2`, encoder on the
Neural Engine) and shows audio length, wall time and speed, measured on each run. With "Also time
Parakeet TDT v3 as a reference" on, v3 is timed on the same file right after. The transcript is
clickable: hover a word for its timestamps, click to seek.

`demo.sh` is the showcase launcher: it builds the app, opens one Ghostty window split into two
rows (asitop on top for the Neural Engine / CPU / GPU load, the app's colour-coded model log below)
and opens the app on the one-hour Earnings-22 sample, waiting for the Transcribe button
(`PHONON2_DEMO_AUTORUN=1 ./demo.sh` starts transcribing on launch).

```bash
cd Examples/Phonon2Demo
./demo.sh                                            # Ghostty (asitop + log) and the app on the 1 h sample
./demo.sh ~/audio.wav                                # the same on your own file
swift run -c release Phonon2Demo --file ~/audio.wav  # app only
```

asitop needs sudo; type the password in the top pane. If asitop exits (it crashes on some macOS 27 /
M5 machines), the pane falls back to `macmon`. Wall time covers `AsrManager.transcribe(url, …)`
only; the first run of each model also downloads and compiles it, which is logged separately.
