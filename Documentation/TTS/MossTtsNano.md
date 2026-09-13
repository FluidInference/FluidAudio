# MOSS-TTS-Nano Swift Inference

0.1B multilingual streaming TTS with zero-shot voice cloning. 20 languages, native
48 kHz **stereo** Float32 output, 80 ms frames streamed as they are decoded.
Six CoreML bundles (≈ 275 MB fp16 on disk; the fp32 voice encoder is fetched only
for custom cloning).

## Overview

MOSS-TTS-Nano (OpenMOSS, Apache-2.0) is a pure autoregressive "audio tokenizer +
LLM" model: a 12-layer GPT-2 (768-d, RoPE) reads rows of `[text token, 16 codec
codes]`, a 1-layer local transformer emits the 16 RVQ codebooks of each 12.5 Hz
frame, and MOSS-Audio-Tokenizer-Nano (22M, causal transformer codec) turns frames
into 48 kHz stereo audio. The conversion lives in
[mobius `models/tts/moss-tts-nano/coreml`](https://github.com/FluidInference/mobius/tree/main/models/tts/moss-tts-nano/coreml);
weights are published at
[FluidInference/moss-tts-nano-coreml](https://huggingface.co/FluidInference/moss-tts-nano-coreml).

```
reference clip ─CodecEncoder (fp32)─► 16×T codes ─┐
text ─SentencePiece BPE─► ids ────────────────────┴─► rows [T,17] ─Prefill─► hidden + KV
                                                                 │  per 80 ms frame
                Frame (local transformer + sampler) ─► 16 codes ─► CodecStep ─► 3840 stereo samples
                Step (global GPT-2, KV update)      ◄── [assistant_slot, 16 codes]
```

Sampling (top-k / top-p / temperature / repetition penalty) is inside the `Frame`
graph; the host only supplies uniform randoms, so a `seed` makes a run
reproducible on the same machine.

## Quick Start

### CLI

```bash
# Preset voice (en_2), stereo 48 kHz WAV
swift run fluidaudiocli tts "Hello from the neural engine." \
    --backend moss-tts-nano --output hello.wav

# Clone a voice from a clip (≤ ~25 s), save the codes, reuse them later
swift run fluidaudiocli tts "Same voice, new words." --backend moss \
    --clone-voice speaker.wav --save-voice speaker.json --output clone.wav
swift run fluidaudiocli tts "Again." --backend moss --voice-file speaker.json

# Deterministic sampling; tokenizer parity check
swift run fluidaudiocli tts "Hello" --backend moss --seed 7 --metrics moss.json
swift run fluidaudiocli tts "Hello" --backend moss --tokens-only
```

| Flag | Default | Notes |
|---|---|---|
| `--voice <name>` | `en_2` | Preset: `en_2` (English), `zh_1` (Mandarin) |
| `--clone-voice <clip>` | – | Any AVFoundation-readable file; resampled to 48 kHz stereo |
| `--save-voice` / `--voice-file <json>` | – | Persist / reuse cloned voice codes |
| `--seed N` | system RNG | Seed for the in-graph sampler |
| `--greedy` | off | Argmax decoding (parity oracle; upstream greedy never stops) |
| `--cpu-only` | off | Everything on CPU |

### Swift

```swift
import FluidAudio

let manager = try await MossTtsNanoManager.downloadAndCreate()
let voice = try await manager.loadVoice(.en2)              // or manager.cloneVoice(audioURL:)

// Batch
let audio = try await manager.synthesize(text: "Local speech synthesis.", voice: voice)
play(left: audio.left, right: audio.right, sampleRate: audio.sampleRate)   // 48 kHz
// audio.mono / audio.interleaved for single-channel or interleaved sinks

// Streaming — 80 ms stereo frames as they are decoded
var options = MossTtsNanoSamplingOptions()
options.seed = 42
for try await frame in try await manager.synthesizeStreaming(text: text, voice: voice, options: options) {
    schedule(left: frame.left, right: frame.right)         // frame.isPause marks inter-chunk silence
}
```

`MossTtsNanoSamplingOptions` exposes the upstream defaults (`textTemperature` 1.5,
`audioTemperature` 1.7, `audioTopP` 0.8, `repetitionPenalty` 1.0, `maxNewFrames`
375 = 30 s per chunk, `maxTextTokens` 50).

## Text handling

- Tokenizer: SentencePiece **BPE** (`MossTtsNanoTokenizer`) with `nmt_nfkc`
  normalization, dummy prefix and byte fallback — verified id-for-id against the
  upstream `sentencepiece` processor. Upstream additionally runs WeTextProcessing
  (number/date verbalization); that step is not ported, so spell out numbers or
  run `NemoTextNormalizer` first for best results.
- Chunking: upstream's sentence → clause → token-budget splitter
  (`MossTtsNanoTextChunker`), 50 tokens per chunk, first letter capitalized and a
  terminal period added when missing; chunks are joined with 0.40 s (≤ 4 words) or
  0.24 s pauses.
- Prompt: `<im_start>user … Reference: <audio_start>[reference rows]<audio_end> …
  Text: {ids} </user_inst><im_end><im_start>assistant <audio_start>`; the template
  ids and special tokens come from the repo's `config.json`.

## Voices

A voice is the codec token sequence of a reference clip (`MossTtsNanoVoice`,
`[frames][16]`). Two presets are published; `cloneVoice(audioURL:)` encodes any
clip with the fp32 codec encoder (fp16 loses a third of the codes through the
residual quantizer). The prefill graph holds 512 rows, so reference frames + text
tokens + 78 template rows must fit: keep clips under ~25 s.

## Performance (M5 Pro, macOS 26.7, warm)

| Stage | ms / call | Unit |
|---|---|---|
| Prefill (512 rows) | 11 | GPU |
| Step (KV M=1024) | 7.6 | GPU |
| Frame | 5.3 | any |
| CodecStep | 5.2 | GPU / ANE |
| CodecEncoder fp32 (8 s clip) | 22 | GPU |

≈ 18 ms of model compute per 80 ms frame. End to end through the Swift host
(release CLI, `en_2` voice, 9–10 s utterances): **3.1–3.8× real time**, first audio
0.22–0.40 s after the call (includes prefill and the first frame); a 3 s Mandarin
utterance with `zh_1` runs at 2.5×. Prefill and Step fail ANE compilation (`ANECCompile FAILED`) and are
pinned to CPU+GPU by `MossTtsNanoModelStore`; `computeUnits` applies to Frame and
CodecStep. The Step graph round-trips a 38 MB KV cache per frame; a `StateType`
(iOS 18) variant is a planned follow-up.

## Parity

- Wrappers vs upstream fp32: hidden states 7e-6, codec 234 dB SNR, encoder codes exact.
- CoreML fp16 greedy replay of a 375-frame upstream reference: 370/375 frames token-exact.
- Streaming codec step vs full decode: 56.6 dB SNR (GPU).
- Parakeet ASR, two English phrases with the `en_2` voice: 8.3 % WER (CoreML) vs
  10.1 % (upstream PyTorch fp32).

## Known limitations

- No text normalization (numbers, dates) — see above.
- `nq < 16` low-bitrate decoding and the batch `CodecDecoder` are not exposed in Swift.
- Sentence chunks are independent generations (upstream behaviour); prosody is not
  carried across chunk boundaries.
