# Bark Typewriter Demo (macOS)

SwiftUI app that runs a file through Underdog Bark 3.0 (verbatim, default), Bark 3.0 Clean (drops fillers and
repeats) or Parakeet TDT v3 with the encoder on the Neural Engine, and plays the file back as fast as the model reads
it: the waveform lights up to the end of the last decoded window and the transcript types out behind it, one
FluidAudio window (`AsrManager.setWindowHandler`) at a time. Overlapping window text is joined on the last shared
3-word run; when the run ends the seam-merged `ASRResult` text replaces it.

`demo.sh` is the showcase launcher: it builds the app, opens one Ghostty window split into two rows (asitop on top for
the Neural Engine / CPU / GPU load, the app's colour-coded log below: model load, every 10 % of the file, final
speed) and opens the app on the one-hour Earnings-22 sample.

```bash
cd Examples/BarkDemo
./demo.sh                                        # Ghostty (asitop + log) and the app on the 1 h sample
BARK_DEMO_AUTORUN=1 ./demo.sh                    # start transcribing on launch
./demo.sh ~/audio.wav --model clean              # your file, Bark 3.0 Clean preselected (bark | clean | v3)
swift run -c release BarkDemo --file ~/audio.wav # app only
```

Models download from Hugging Face on first use: Bark from
[FluidInference/bark-3.0-coreml](https://huggingface.co/FluidInference/bark-3.0-coreml) (`AsrModelVersion.bark3` /
`.bark3Clean`), v3 from its usual repo. `--bark-dir` / `--clean-dir` load a local folder in the same layout instead.
The first run of each model downloads it and compiles it for the Neural Engine; the app warms the model up before
timing, so wall time covers `AsrManager.transcribe(url, …)` only. asitop needs sudo; type the password in the top
pane. If asitop exits, the pane falls back to `macmon`.

Model credits: Underdog Bark 3.0 / 3.0 Clean by Conway Research (CC-BY-4.0), derived from NVIDIA
parakeet-tdt-0.6b-v3 via moondream/parakeet-ultra.
