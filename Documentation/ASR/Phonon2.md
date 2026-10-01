# Phonon-2 (five-value v3, English)

`AsrModelVersion.phonon2` loads `FluidInference/phonon-2-coreml`, a Core ML build of
[FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2): Fermion Research's quantization-aware
re-training of `parakeet-tdt-0.6b-v3` in which every encoder weight takes one of five learned values per output row
(`{0, ±lo, ±hi}`, about 2.1 bits each in the upstream 164 MB download). English only; same tokenizer, 15 s window and
`JointDecisionv3` contract as v3, so every v3 decode path applies unchanged (`isV3Family`).

| | v3 (`Encoder.mlmodelc`, 6-bit) | phonon2 (`Encoder.mlmodelc`) | phonon2 (`Encoder_lut3.mlmodelc`) |
|---|---|---|---|
| Encoder on disk | 445 MB | **321 MB** | 253 MB |
| Model directory | ~480 MB | ~360 MB | ~290 MB |
| Minimum OS | iOS 17 / macOS 14 | **iOS 18 / macOS 15** | **iOS 18 / macOS 15** |
| Languages | 25 | English | English |

The default encoder keeps the checkpoint's exact five-value weights as a sparsity mask (51 % of the weights are zero)
plus fp16 palettes over the non-zeros (iOS 18 `constexpr_lut_to_sparse` + `constexpr_sparse_to_dense`, one palette per
8 output rows), so nothing is re-quantized on our side; decoder and joint are re-exported from the checkpoint's int6
tables. Recipe: `mobius/models/stt/phonon-2/coreml`. On iOS 17 / macOS 14 `AsrModels` throws before downloading
anything and points to `.ultra`.

## Usage

```swift
let models = try await AsrModels.downloadAndLoad(version: .phonon2)
```

```bash
swift run fluidaudiocli transcribe audio.wav --model-version phonon2
swift run fluidaudiocli asr-benchmark --subset test-clean --model-version phonon2
```

## Compute units and the encoder files

Phonon-2 uses the library default, the Neural Engine. Its default encoder is the fastest v3-family encoder we have
measured there; the first ANE load compiles the sparse weights for about a minute, and Core ML caches the result.
The HF repo carries four more exact encoders (same transcripts) for other trade-offs:

| Encoder, one 15 s window | Size | ANE | ANE RTFx | GPU |
|---|---:|---:|---:|---|
| v3 6-bit (reference) | 445 MB | 23.5 ms | 149–152× | 18 ms |
| phonon2 `Encoder.mlmodelc` (sparse, 8 rows/palette) | 321 MB | **18.6 ms** | **159×** | 16 ms, but ~150 s load every launch |
| `Encoder_sparse-g4.mlmodelc` | 246 MB | 24.3 ms | 140× | same load caveat |
| `Encoder_sparse-g1.mlmodelc` | 176 MB | 70 ms | ~70× | same load caveat |
| `Encoder_lut6.mlmodelc` (dense) | 470 MB | 18.6 ms | 155× | 16 ms, 0.6 s load |
| `Encoder_lut3.mlmodelc` (dense) | 253 MB | 72 ms | 70× | 16 ms, 0.7 s load |

The Neural Engine's palette cost grows with the number of palettes, not their bit width, which is why 8 rows per
palette beats v3's encoder while per-row palettes are 3× slower. The GPU materializes *sparse* weights at every load
(~150 s of CPU, never cached), so apps that run the encoder on the GPU (`encoderComputeUnits: .cpuAndGPU`) should use
a dense file: download the model directory, rename `Encoder_lut3.mlmodelc` (small) or `Encoder_lut6.mlmodelc` (fast)
to `Encoder.mlmodelc`, and load it with `AsrModels.loadLocal(from:version: .phonon2)`.

## Accuracy and speed

Full LibriSpeech, `asr-benchmark`, M5 Pro (macOS 27), Phonon-2 and v3 run back to back on the same machine, default
compute units (ANE) unless noted. WER is corpus-level (total edit distance over total reference words); RTFx is total
audio divided by total processing time.

| Set | v3 WER | phonon2 WER | v3 RTFx | phonon2 RTFx (default / lut6) | phonon2 RTFx (`Encoder_lut3`) |
|---|------:|------------:|--------:|--------------------------------:|------------------------------:|
| test-clean (2620 files), ANE | **2.27 %** | 2.47 % | 148.7–151.5× | **159.0×** | 70.0× |
| test-other (2939 files), ANE | **4.12 %** | 4.62 % | 138.1× | **143.0×** (lut6) | 64.6× |
| test-clean, GPU (`.cpuAndGPU`) | **2.30 %** | 2.46 % | **171.9×** | 150.6× (lut6) | 154.1× |

On LibriSpeech **v3 is the more accurate model** by 0.20 (clean) and 0.50 (other) points, which reproduces the upstream
card's own deltas against its teacher (+0.20 / +0.79 under the Open ASR Leaderboard protocol; the card wins against v3
on AMI meetings and VoxPopuli, which we have not measured). Phonon-2 beats Redux on English (2.71 / 5.12 %). Absolute
values are above the card's because FluidAudio decodes in 15 s windows with a simpler normalizer; both models pay it
equally. The two encoder files produce identical transcripts (2620 / 2620 files).

Conversion fidelity: on the first 100 test-clean files the Core ML transcripts differ from a NeMo fp32 full-context
decode of the same checkpoint by 0.34 % WER (corpus WER 1.83 % vs 1.79 %), so the gap above is the checkpoint's, not
the conversion's.

**Choose phonon2 for English on iOS 18+ when you want the fastest Neural Engine encoder in the v3 family, or the
176–253 MB encoder options; choose v3 / Ultra for multilingual audio or the last half point on English.**

