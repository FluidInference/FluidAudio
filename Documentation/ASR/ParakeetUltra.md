# Parakeet Ultra (post-trained v3)

`AsrModelVersion.ultra` loads `FluidInference/parakeet-ultra-coreml`, a Core ML build of
[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra): a full-precision post-training of
`parakeet-tdt-0.6b-v3` with the same architecture, 25 languages, tokenizer, 15 s window and `JointDecisionv3`
contract. Every v3 decode path applies unchanged (`isV3Family`); only the weights differ.

| | v3 (`Encoder.mlmodelc`, 6-bit) | redux | ultra |
|---|---|---|---|
| Encoder on disk | 445 MB | 183 MB | 595 MB |
| Encoder weights | 6-bit LUT | 2-bit ternary | int8 linear per-channel |
| Default encoder units | ANE | ANE | ANE |
| Minimum OS | iOS 17 / macOS 14 | iOS 18 / macOS 15 | iOS 17 / macOS 14 |

The encoder is int8 rather than v3's 6-bit LUT, which flips tokens on some windows (#760). It is a single iOS 17
/ macOS 14 export that also serves iOS 18+: paired back to back on full test-other against an iOS 18 export of the
same weights it matches in WER (3.79 / 3.80 %) and speed (ANE 96.4× vs 93.3×, GPU 96.7× vs 96.5×). Recipe:
`mobius/models/stt/parakeet-ultra/coreml`.

## Usage

```swift
let models = try await AsrModels.downloadAndLoad(version: .ultra)
```

```bash
swift run fluidaudiocli transcribe audio.wav --model-version ultra
swift run fluidaudiocli asr-benchmark --subset test-clean --model-version ultra
swift run fluidaudiocli fleurs-benchmark --languages en_us,de_de --samples 100 --model-version ultra
```

## Accuracy

Full corpora, corpus-level WER (total edits over total reference words), same build and scorer for every column.

| Set | v3 | redux | ultra |
|---|---:|---:|---:|
| LibriSpeech test-clean (2620 files) | 2.27 % | 2.67 % | **2.12 %** |
| LibriSpeech test-other (2939 files) | 4.12 % | 5.15 % | **3.79 %** |
| FLEURS, 24 languages × 100, mean | 14.81 % | 13.06 % | **11.67 %** |
| FLEURS, duration-weighted | 14.65 % | 12.89 % | **11.51 %** |

Ultra beats v3 in all 24 FLEURS languages (FluidAudio's FLEURS set has no `es_es`), most on the low-resource end:
Latvian −8.4, Lithuanian −7.4, Slovene −7.4, Maltese −6.9, Finnish −5.3, Greek −5.0, Slovak −5.0. The per-language
direction agrees with the upstream card in 24/24. Unlike redux, it does not give back ground on English or the
high-resource languages (French −0.8, Russian −1.3, English −0.2).

The int8 encoder is WER-identical to an fp16 export (test-clean 2.12 vs 2.13 %, test-other 3.79 vs 3.79 %), and
compute placement is WER-neutral (ANE 2.12 %, GPU 2.13 %).

## Speed

RTFx = total audio / total processing time. v3, redux and ultra run back to back per row on the same machine (load
3–6), full test sets:

| Set | Encoder | v3 | redux | ultra |
|---|---|---:|---:|---:|
| test-clean | ANE | 88.3× | 67.8× | **89.5×** |
| test-clean | GPU | 93.1× | 86.6× | **94.2×** |
| test-other | ANE | 92.2× | 68.0× | **96.5×** |
| test-other | GPU | 93.4× | 89.2× | **100.7×** |

FLEURS (24 languages × 100, `fleurs-benchmark --encoder-compute-units`), same protocol:

| Encoder | v3 | redux | ultra |
|---|---:|---:|---:|
| ANE | 92.8× | 70.8× | **93.0×** |
| GPU | 89.8× | 87.0× | **122.5×** |

FLEURS clips are short, so its RTFx swings more with machine load than LibriSpeech; compare within a row.

The shipped iOS 17 encoder was re-checked against v3 on test-clean: ANE 96.7× vs 94.9×, GPU 102.6× vs 103.4×, WER
unchanged (2.12 %). In isolation it is within 3–5 % of the iOS 18 export on GPU (16.6–17.4 vs 16.1–16.5 ms/window).

## Which to ship

* **Default choice for new integrations** → ultra: more accurate than v3 on English and every FLEURS language, same
  speed, runs on the ANE (so iOS background transcription works).
* **Download size matters most** → redux (220 MB model dir vs ~630 MB).
* **Existing v3 integrations** → switching is a one-line change (`version: .ultra`); v3 stays the library default so
  nothing changes for current users until they opt in.
