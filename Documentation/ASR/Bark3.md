# Underdog Bark 3.0 and Bark 3.0 Clean (English)

`AsrModelVersion.bark3` and `.bark3Clean` load the two folders of
[FluidInference/bark-3.0-coreml](https://huggingface.co/FluidInference/bark-3.0-coreml), Core ML builds of
[Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) and
[Underdog Bark 3.0 Clean](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Clean) by Conway Research
([Underdog](https://underdog.ai)): post-trainings of `parakeet-tdt-0.6b-v3` (via moondream/parakeet-ultra), English only.
Same tokenizer, 15 s window and `JointDecisionv3` contract as v3, so every v3 decode path applies unchanged
(`isV3Family`).

| | `.bark3` | `.bark3Clean` |
|---|---|---|
| Output | Verbatim: keeps "um", "uh", repeats | Cleaned: drops fillers, false starts and repeated words |
| Use for | Meetings, captions, anything that must match the audio | Dictation and readable notes |
| Repo folder / cache folder | `bark-3.0/` → `Models/bark-3.0` | `bark-3.0-clean/` → `Models/bark-3.0-clean` |
| Encoder on disk (`Encoder.mlmodelc`, ANE) | 399 MB | 410 MB |
| Minimum OS | iOS 18 / macOS 15 | iOS 18 / macOS 15 |

```swift
let models = try await AsrModels.downloadAndLoad(version: .bark3)   // or .bark3Clean
let asr = AsrManager(config: .default)
try await asr.loadModels(models)
var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
let result = try await asr.transcribe(audioURL, decoderState: &state)
```

```bash
swift run -c release fluidaudiocli transcribe meeting.wav --model-version bark3        # or bark3-clean
swift run -c release fluidaudiocli asr-benchmark --subset test-other --model-version bark3
```

Underdog ships a Core ML encoder of its own for Bark 3.0 (with the mel front-end and decoder in Python) and a full Core
ML pipeline for Bark 3.0 Pro; these builds add the whole pipeline for the small models, and Bark 3.0 Clean.

## Encoder encoding

Upstream `model.bark` stores each encoder linear as per-row levels: 260 of the 264 matrices take five values per output
row (`{0, ±a, ±b}`), layer 23's four feed-forward matrices take seven. The 240 that reach the Core ML graph (the
positional projections fold into constants) are re-encoded as iOS 18 `constexpr_lut_to_sparse` +
`constexpr_sparse_to_dense` with one palette per 8 output rows, the layout that runs fastest on the Neural Engine (see
[Phonon2.md](Phonon2.md): the ANE's palette cost scales with the number of palettes). Every palettized weight
reconstructs bit-exactly; nothing is re-quantized. The other tensors (subsampling, depthwise convs, decoder, joint) are
the upstream values converted at fp16. The mel front-end is v3's (Bark uses the original's).

Each folder also carries `Encoder_lut3.mlmodelc` (253 MB, dense 3-bit palette per row, same weights) for GPU use: the
sparse default is rebuilt on every `.cpuAndGPU` load. It is not downloaded by default.

On iOS 17 / macOS 14 `AsrModels` throws before downloading anything and points to `.ultra`.

## `<unk>` tokens

Bark emits SentencePiece `<unk>` (id 0) where its training text had a quotation mark (295 times on LibriSpeech
test-clean with Clean). NeMo / HF decoding skip special tokens; `AsrManager` drops `<unk>` from `ASRResult.text`, the
token timings and the long-form window text, after the decoder has consumed it, so decoding is unchanged.

## Benchmarks

M5 Pro, macOS 27, encoder on the Neural Engine, full LibriSpeech, corpus WER (total edits / reference words):

| Model | test-clean | test-other | RTFx clean / other |
|---|--:|--:|--:|
| `.bark3` | — | 4.96 % | — / 140× |
| `.bark3Clean` | 2.37 %¹ | 4.66 % | 155×¹ / 143× |
| Bark 3.0 Clean, upstream MLX (`bark_mlx.py`, fp16), same files and scorer | 2.39 % | — | 55× / — |
| `.v3` | 2.27 % | 4.12 % | 150× / 140× |

¹ Measured with a dense 6-bit-palette build of the same encoder weights; the shipped sparse build matched it on
test-other (4.66 % both).

One hour of Earnings-22 (four concatenated calls, `Examples/BarkDemo`): `.bark3` 6.5 s (558×, 8,223 words),
`.bark3Clean` 7.0 s (514×, 7,787 words), warm models. LibriSpeech references are verbatim, so Clean's deliberate
deletions count as errors there. Bark 3.0 is evaluated upstream with 0.25 s of leading silence; through the sliding
window it changes nothing (test-other 5.02 % with, 4.96 % without; `asr-benchmark --leading-silence`), so none is
added.

## Demo

`Examples/BarkDemo` plays a file back as fast as the model reads it: the waveform fills to the end of the last decoded
window and the transcript types out behind it (`AsrManager.setWindowHandler`). `./demo.sh` adds a Ghostty window with
the load monitor and the colour-coded log.

Credits: Underdog Bark 3.0 / 3.0 Clean by Conway Research (CC-BY-4.0), derived from NVIDIA parakeet-tdt-0.6b-v3
through moondream/parakeet-ultra.
