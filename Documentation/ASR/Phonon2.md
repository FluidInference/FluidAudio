# Phonon-2 (five-value v3, English)

`AsrModelVersion.phonon2` loads `FluidInference/phonon-2-coreml`, a Core ML build of
[FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2): Fermion Research's quantization-aware
re-training of `parakeet-tdt-0.6b-v3` in which every encoder weight takes one of five learned values per output row
(`{0, ±lo, ±hi}`, about 2.1 bits each in the upstream 164 MB download). English only; same tokenizer, 15 s window and
`JointDecisionv3` contract as v3, so every v3 decode path applies unchanged (`isV3Family`).

| | v3 (`Encoder.mlmodelc`, 6-bit) | phonon2 (`Encoder.mlmodelc`) | phonon2 (`Encoder_lut3.mlmodelc`) |
|---|---|---|---|
| Encoder on disk | 445 MB | 470 MB | 253 MB |
| Model directory | ~480 MB | ~510 MB | ~290 MB |
| Minimum OS | iOS 17 / macOS 14 | **iOS 18 / macOS 15** | **iOS 18 / macOS 15** |
| Languages | 25 | English | English |

The encoder keeps the checkpoint's exact five-value weights as fp16 palettes (iOS 18 `constexpr_lut_to_dense`, one
palette per 8 output rows), so nothing is re-quantized on our side; decoder and joint are re-exported from the
checkpoint's int6 tables. Recipe: `mobius/models/stt/phonon-2/coreml`. On iOS 17 / macOS 14 `AsrModels` throws before
downloading anything and points to `.ultra`.

## Usage

```swift
let models = try await AsrModels.downloadAndLoad(version: .phonon2)
```

```bash
swift run fluidaudiocli transcribe audio.wav --model-version phonon2
swift run fluidaudiocli asr-benchmark --subset test-clean --model-version phonon2
```

## Compute units and the two encoder files

Phonon-2 uses the library default, the Neural Engine. Its shipped encoder (`Encoder.mlmodelc`, 6-bit palettes shared by
8 rows) is the fastest v3-family encoder we have measured there, and its first load compiles in about 15 s:

| Encoder, one 15 s window | GPU | ANE | ANE first load |
|---|---:|---:|---:|
| v3 6-bit (reference) | 18.3 ms | 23.5 ms | 11.5 s |
| phonon2 `Encoder.mlmodelc` (lut6, 470 MB) | 16.2 ms | **18.6 ms** | 14 s |
| phonon2 `Encoder_lut3.mlmodelc` (lut3, 253 MB) | 16.2 ms | 72.4 ms | 72 s |

The Neural Engine's palette cost grows with the number of palettes, not with their bit width, so the 3-bit per-row
variant is 3× slower there while the GPU does not care. `Encoder_lut3.mlmodelc` exists for apps that run the encoder
on the GPU (`encoderComputeUnits: .cpuAndGPU`) and want the smaller download: copy the model directory, rename
`Encoder_lut3.mlmodelc` to `Encoder.mlmodelc`, and load it with `AsrModels.loadLocal(from:version: .phonon2)`. The
transcripts are identical (both files hold the same exact weights).

## Accuracy and speed

Full LibriSpeech, `asr-benchmark`, M5 Pro (macOS 27), Phonon-2 and v3 run back to back on the same machine, default
compute units (ANE) unless noted. WER is corpus-level (total edit distance over total reference words); RTFx is total
audio divided by total processing time.

| Set | v3 WER | phonon2 WER | v3 RTFx | phonon2 RTFx (`Encoder.mlmodelc`) | phonon2 RTFx (`Encoder_lut3`) |
|---|------:|------------:|--------:|--------------------------------:|------------------------------:|
| test-clean (2620 files), ANE | **2.27 %** | 2.47 % | 148.7× | **155.1×** | 70.0× |
| test-other (2939 files), ANE | **4.12 %** | 4.62 % | 138.1× | **143.0×** | 64.6× |
| test-clean, GPU (`.cpuAndGPU`) | **2.30 %** | 2.46 % | **171.9×** | 150.6× | 154.1× |

On LibriSpeech **v3 is the more accurate model** by 0.20 (clean) and 0.50 (other) points, which reproduces the upstream
card's own deltas against its teacher (+0.20 / +0.79 under the Open ASR Leaderboard protocol; the card wins against v3
on AMI meetings and VoxPopuli, which we have not measured). Phonon-2 beats Redux on English (2.71 / 5.12 %). Absolute
values are above the card's because FluidAudio decodes in 15 s windows with a simpler normalizer; both models pay it
equally. The two encoder files produce identical transcripts (2620 / 2620 files).

Conversion fidelity: on the first 100 test-clean files the Core ML transcripts differ from a NeMo fp32 full-context
decode of the same checkpoint by 0.34 % WER (corpus WER 1.83 % vs 1.79 %), so the gap above is the checkpoint's, not
the conversion's.

**Choose phonon2 for English on iOS 18+ when you want the fastest Neural Engine encoder in the v3 family, or the
253 MB GPU encoder; choose v3 / Ultra for multilingual audio or the last half point on English.**

