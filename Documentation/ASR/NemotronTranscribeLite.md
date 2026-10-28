# Nemotron 3.5 Transcribe Lite

Long-form English transcription with word start times and speakers, from the audio side of NVIDIA
[Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) without its 30B
language model: the ~600M-parameter ASR encoder and the CTC timestamp head, plus
[Nemotron-3-Diarization](../Diarization/Nemotron3.md) for who spoke when.

"Lite" is our name for this setup, not an NVIDIA release. The CTC head was trained by NVIDIA as a timestamp aligner for
the LLM's output; used directly as a greedy CTC recognizer it is much less accurate than the full model (see
Benchmarks). What it is good at is speed: a 49.5-minute meeting in about 5.5 s on an M5 Pro.

## Model

**CoreML Model**: [FluidInference/nemotron-3.5-transcribe-lite-coreml](https://huggingface.co/FluidInference/nemotron-3.5-transcribe-lite-coreml)

| Stage | File | Precision | Compute unit (default) |
|-------|------|-----------|------------------------|
| Mel tables | `assets/mel_window.bin`, `assets/mel_filterbank.bin` | FP32 | host (vDSP) |
| Encoder | `coreml/lite_encoder_w48.mlmodelc` | FP16 | GPU (`.cpuAndGPU`) |
| CTC head | `coreml/lite_ctc_head_w48.mlmodelc` | FP16 | GPU |
| Vocabulary | `assets/vocab.json` (23,862 SentencePiece pieces, blank = 23862) | | |
| Diarizer | `Nemotron3Diarizer_offline` from [nemotron-3-diarization-coreml](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml) | FP16 | Neural Engine |

The models are ML Programs for macOS 15 / iOS 18 and newer.

## Architecture

```
16 kHz audio → NeMo log-mel (checkpoint window + filterbank) → per-feature normalization over the whole recording
  → encoder, 48 s windows: 40 s cores + 4 s context each side → states (1280 x 80 ms frames)
  → CTC head, 600-frame windows: 500-frame cores + 50 context → one token per 40 ms
  → greedy CTC → words, each stamped with its first piece's 40 ms frame

in parallel: Nemotron-3-Diarization (offline preset, Neural Engine) → 10 ms activity for 8 speaker slots
  → averaged to 80 ms → slots with < 2 s of speech dropped → each word takes the most likely active
    speaker within ~1 s of its start, else the previous word's speaker
```

The encoder and CTC head run on the GPU while the diarizer runs on the Neural Engine, so the diarizer costs no extra
wall time. On the Neural Engine the encoder is about 2.5x slower than on the GPU.

## Usage

### CLI

```bash
# Words + speakers: [mm:ss] Speaker N: ... turns, timings and RTFx
swift run -c release fluidaudiocli nemotron35-transcribe meeting.wav

# Words only, plus a <spk:N>-tagged transcript and a JSON of words ({t, spk, w}) and timings
swift run -c release fluidaudiocli nemotron35-transcribe meeting.wav --no-speakers --json words.json
swift run -c release fluidaudiocli nemotron35-transcribe meeting.wav --out transcript.txt
```

The CLI loads and warms all models (and runs the diarizer on the first 10 s, which compiles it for the Neural Engine
on first use, ~80 s) before timing.

### Swift

```swift
let manager = try await NemotronTranscribeLiteManager.load()   // encoder + CTC head on the GPU
try await manager.warmup()

let result = try await manager.transcribeWithSpeakers(audio: samples)   // 16 kHz mono [Float]
for turn in result.turns {
    print(turn.start, turn.speaker ?? -1, turn.text)
}
// Words only: try await manager.transcribe(audio: samples)
```

`transcribeWithSpeakers` needs the `Diarizer` trait (on by default).

## Benchmarks

AMI Meeting Corpus EN2002c (49.5 min, mixed headset, 3 speakers), Apple M5 Pro, models loaded and warm, scored against
the AMI manual transcript with the Whisper English normalizer:

| | |
|---|---|
| total | 5.53 s (537x real time) |
| mel + encoder / CTC head / diarizer (in parallel) | 4.57 s / 0.96 s / 4.95 s |
| WER | 26.4% |
| cpWER (speaker-attributed) | 28.7% |
| speakers found | 3 |

The Swift pipeline matches the reference Python pipeline on the HF repo (`demo/lite_meeting.py`): 9,157 of 9,172
words identical, the rest are near-tie argmax flips from float32 vs float64 mel arithmetic; speakers identical on
every matched word.

For comparison, NVIDIA reports 10.29% cpWER (individual headsets) and 17.34% (single distant mic) on the AMI test set
for the full model with its LLM, and 1.09% WER on LibriSpeech test-clean (this encoder + CTC head: 5.50%).
