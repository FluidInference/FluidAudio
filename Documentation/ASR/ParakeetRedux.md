# Parakeet Redux (2-bit ternary v3)

`AsrModelVersion.redux` loads `FluidInference/parakeet-redux-coreml`, a Core ML build of
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux): a ternary ({-1, 0, +1}) re-training of
`parakeet-tdt-0.6b-v3` with the same 25 languages, tokenizer, 15 s window and `JointDecisionv3` contract. Every v3
decode path applies unchanged (`isV3Family`); only the weights and the download differ.

| | v3 (`Encoder.mlmodelc`, 6-bit) | redux |
|---|---|---|
| Encoder on disk | 445 MB | 183 MB |
| Model directory | ~480 MB | ~220 MB |
| Minimum OS | iOS 17 / macOS 14 | **iOS 18 / macOS 15** |

Redux exists for its download size, which needs the 2-bit encoding and therefore iOS 18 Core ML ops. On iOS 17 /
macOS 14 `AsrModels` throws before downloading anything and points to `.ultra`: an int8 iOS 17 build of Redux is
WER-identical but 595 MB, the same size as Ultra and less accurate, so it is not shipped.

The encoder keeps the checkpoint's exact ternary weights (2-bit codes + the model's own per-row/per-128 fp16 scales),
so nothing is re-quantized on our side. Decoder and joint are re-exported from the redux checkpoint. Recipe:
`mobius/models/stt/parakeet-redux/coreml`.

## Usage

```swift
let models = try await AsrModels.downloadAndLoad(version: .redux)
```

```bash
swift run fluidaudiocli transcribe audio.wav --model-version redux
swift run fluidaudiocli asr-benchmark --subset test-clean --max-files 100 --model-version redux
```

## Compute units

Redux uses the library default, the Neural Engine, like v3 and Ultra, so iOS apps keep background execution (iOS does
not allow GPU work in the background). The first ANE load compiles the 2-bit weights for several minutes (~7 min on
an M-series Mac); Core ML caches it and later loads take seconds. Pass `encoderComputeUnits: .cpuAndGPU` to avoid the
first compile when background execution does not matter: the GPU decompresses the weights in-kernel, loads in about
a second, and is faster per window (≈21 ms vs 45–52 ms).

Seven weight encodings were measured on the ANE (see the conversion README); none beats the shipped 6-bit v3 encoder
there:

| Encoder weights on ANE | First load | Latency |
|---|---:|---:|
| fp16 (uncompressed) | 8.7 s | 20.8 ms |
| 2-bit / int4, per-(row, block) scale (shipped) | 320–1585 s | 45–52 ms |
| 2-bit, per-row scale | 45 s | 70 ms |

## Accuracy and speed

Full LibriSpeech, `asr-benchmark`, M-series Mac. WER is corpus-level (total edit distance over total reference
words, which is what the published leaderboards report); RTFx is total audio divided by total processing time.

| Set | v3 WER | redux WER | v3 RTFx | redux RTFx |
|---|------:|----------:|--------:|-----------:|
| test-clean (2620 files) | **2.27 %** | 2.71 % | **128.6×** | 83.9× |
| test-other (2939 files) | **4.12 %** | 5.12 % | **114.7×** | 76.0× |

Default compute units (ANE), v3 and redux run back to back on the same machine, both on the v3-family long-form path
(no mel context, silence-aligned window starts).

**v3 is the more accurate model on English**, by 0.44 points on test-clean and 1.00 on test-other, which reproduces
the upstream model card's own deltas (+0.44 and +1.21). The absolute values are higher than the card's because
FluidAudio decodes in 15 s windows and scores with a simpler normalizer than the Open ASR Leaderboard, which
penalises both models equally. Compute placement is WER-neutral for both models.

Redux is also slower end to end: ~34 % behind v3 on the ANE. On GPU (`.cpuAndGPU`) the gap closes to a few percent
(its encoder is 21.3 ms/window vs 18.3 ms), but redux keeps the ANE default for iOS background execution.

Conversion fidelity is not the issue — the Core ML redux transcripts match a PyTorch fp32 decode of the redux
checkpoint to 0.19 % WER. Everything above is the checkpoint's own behaviour. **Choose redux for download size, and
for the languages where the ternary re-training genuinely helps; do not choose it for English accuracy or speed.**

### Multilingual (FLEURS)

`fleurs-benchmark --samples 100`, 24 of the model's 25 languages (FluidAudio's FLEURS set has no `es_es`), same
build, encoder on GPU. **This is where redux earns its place: it wins the average by 1.74 points.**

| | v3 | redux |
|---|---:|---:|
| Mean WER over 24 languages | 14.81 % | **13.06 %** |
| Duration-weighted WER | 14.65 % | **12.89 %** |
| Languages won | 11 | **13** |
| RTFx | **149×** | 134× |

The split is systematic rather than random. Redux is much better on the low-resource end and gives back ground on
the high-resource languages:

| Redux better | Δ | Redux worse | Δ |
|---|---:|---|---:|
| Latvian | −11.14 | French | +3.66 |
| Maltese | −7.72 | Russian | +2.57 |
| Slovene | −7.17 | Dutch | +1.92 |
| Estonian | −7.02 | English | +1.86 |
| Greek | −5.19 | Polish | +1.68 |
| Lithuanian | −5.10 | Ukrainian | +1.43 |

The per-language direction agrees with the upstream model card in 23 of 24 languages (only Bulgarian flips, by about
a point either way). Our deltas run larger than the card's in both directions, consistent with the harder decoding
conditions; the ranking is what transfers.

### Which to ship

* **English-only** → v3. It is more accurate and faster.
* **Multilingual, especially Baltic / Maltese / Slovene / Greek** → redux, which is both more accurate on average and
  260 MB smaller.
* **Size-constrained** → redux, at a known English cost.
* **iOS 17 / macOS 14** → ultra; redux needs iOS 18.
* **iOS background transcription** → any of the three on the ANE default; redux pays a one-time multi-minute first compile.
