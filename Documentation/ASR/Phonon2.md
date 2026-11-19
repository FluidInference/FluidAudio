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
tables. On iOS 17 / macOS 14 `AsrModels` throws before downloading anything and points to `.ultra`.

### Encoder ladder (multifunction `Encoder.mlmodelc`)

`Encoder.mlmodelc` is a Core ML **multifunction** model: three encoder programs — `enc_5040ms`, `enc_10160ms`,
`enc_15000ms` (5.04 / 10.16 / 15 s windows) — over one shared weight blob (341 MB for the three; the extra functions
cost ~10 MB each, mostly their constant-folded relative-position tables). `AsrModels.load` reads the function names
(`MLModelAsset.functionNames`), loads each as its own `MLModel`, and `AsrManager` runs every window on the smallest
function that holds its mel frames, slicing the 15 s preprocessor output to the function's length. Each function
computes the same length-masked encoder, so transcripts do not change (LibriSpeech test-clean: 2.47 % WER with either
build) — only the time does:

| Function | Encoder frames | ANE, one call (M5 Pro) |
|---|---:|---:|
| `enc_5040ms` | 63 | ~6 ms |
| `enc_10160ms` | 126 | 12.2 ms |
| `enc_15000ms` | 188 | 19.3 ms |

The lengths are not arbitrary: Neural Engine time for this encoder is quantized in **64-frame tiles** of the time
axis — 11, 12, 13, 14 and 15 s windows (138–188 frames) all cost 18–21 ms, 10 s (126 frames) costs 12 ms. A finer
ladder (1 s steps) was built and measured first and bought nothing; 64 / 128 / 188 frames are the three tiles the 15 s
preprocessor can feed. Two consequences for the host, both in `ChunkProcessor` (see "Pause-Cut Windows" and
"Post-Merge Repair Pass" in LongTranscription.md): long recordings are cut at the **latest pause** in each window with
no overlap at that seam (fewer windows at the same per-window price), and seam-gap repair probes run on the 5 s
function. A single-shape encoder loads exactly as before (`encoderLadder` is empty) and keeps the overlap chunker.

Core ML compiles each function for the Neural Engine on first load (about 55 s each on an M5 Pro, cached afterwards),
so the first launch takes roughly three minutes instead of one. Function names must not contain a `.` — Core ML's ANE
compiler crashes on `enc_5.04s`, hence the millisecond spelling.

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
a dense file. FluidAudio only downloads `Encoder.mlmodelc`, so fetch the alternate encoder yourself, swap it in, and
load the directory with `AsrModels.loadLocal(from:version: .phonon2, encoderComputeUnits: .cpuAndGPU)`:

```bash
hf download FluidInference/phonon-2-coreml --include "Encoder_lut3.mlmodelc/*" --local-dir /tmp/phonon2
cp -R ~/Library/Application\ Support/FluidAudio/Models/phonon-2 /tmp/phonon2-gpu
rm -rf /tmp/phonon2-gpu/Encoder.mlmodelc && mv /tmp/phonon2/Encoder_lut3.mlmodelc /tmp/phonon2-gpu/Encoder.mlmodelc
```

## Accuracy and speed

Full LibriSpeech, `asr-benchmark`, M5 Pro (macOS 27), Phonon-2 and v3 run back to back on the same machine, default
compute units (ANE) unless noted. WER is corpus-level (total edit distance over total reference words); RTFx is total
audio divided by total processing time.

| Set (ANE) | v3 | Ultra | Phonon-2 default |
|---|---|---|---|
| test-clean (2620 files) WER | 2.27 % | **2.13 %** | 2.47 % |
| test-other (2939 files) WER | 4.12 % | **3.81 %** | 4.62 % |
| test-clean RTFx | 149–152× | 151× | **193×** (ladder; 155–159× single-shape) |
| test-other RTFx | 138× | 142× | **146×** (single-shape) |

The ladder encoder's test-clean figure was measured back to back with the single-shape build in one session (193.3×
vs 155.5×, 100.6 s vs 125.1 s for the 2620 files, corpus WER 2.47 % for both): most LibriSpeech utterances fit the
10.16 s or 5.04 s function, so a 2620-file run stops paying for 2620 fifteen-second encoder passes.

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

### 60-minute long-form file (Earnings-22, four concatenated calls)

`transcribe` on the 3600 s `earnings22_top4_1h.wav` (M5 Pro, default ANE encoder, best of 2 runs in one session,
processing time excludes model load). Reference = the concatenated Earnings-22 chunk transcripts, same normalizer as
above. These are the same runs as the ANE column of the compute-unit table below and as Benchmarks.md.

| Model | Processing time | RTFx | WER |
|---|---:|---:|---:|
| v3 | 10.7 s | 335× | 16.5 % |
| Ultra | 7.9 s | 457× | **13.5 %** |
| Phonon-2 single-shape encoder (sparse, 321 MB), overlap chunker | 7.5 s | 480× | 17.2 % |
| Phonon-2 `Encoder_lut6` (470 MB), overlap chunker | 7.4 s | 484× | 17.2 % |
| **Phonon-2 ladder encoder (default), pause-cut chunker** | **5.55 s** | **649×** | 16.7 % |

On conversational long-form audio Phonon-2 is the fastest model we ship (1.9× v3's throughput, ahead of Ultra) but the
least accurate of the three: Ultra beats v3 here while Phonon-2 trails it, consistent with the upstream card's
Earnings-22 row (6.96 % vs its teacher's 5.85 %). All Phonon-2 encoder files produce the same transcript on the same
chunk layout; the ladder row's lower WER comes from the pause-cut windows, which remove the seam merge wherever a
pause exists.

Where the ladder row's time goes (3 runs, 5.54–5.57 s): ~267 windows × 19.3 ms of Neural Engine time is 5.1 s of it,
planning 0.1 s, the repair pass 0.35 s. The chunk loop is Neural Engine-bound at four workers (more workers do not
help), so the remaining lever is encoding fewer windows, not host work.

For reference, Fermion Research's own Core ML runtime for the same checkpoint
([phonon-coreml](https://github.com/fermionresearch/phonon-coreml) 1.1.2 with
[Phonon-2-CoreML](https://huggingface.co/FermionResearch/Phonon-2-CoreML): multifunction 5/10/15/35 s encoder, pause-cut
15 s windows without overlap, C greedy-TDT decoder) transcribes the same file on the same machine, interleaved with the
runs above, in **5.74 s (627×) at 17.26 % WER**; its 15 s encoder call costs the same 18.6–19.3 ms as ours. The
difference is window packing (its 288 windows average 12.5 s, ours ~13.5 s) and the 5 s repair probes.

Encoder compute units on the same 60-minute file (`transcribe --encoder-compute-units`, best of 2, processing time
excludes model load):

| Model | ANE | GPU | `.all` |
|---|---:|---:|---:|
| v3 | 10.7 s / 335× | 9.2 s / 391× | 9.3 s / 387× |
| Ultra | 7.9 s / 457× | 11.1 s / 326× | 11.2 s / 322× |
| Phonon-2 default (sparse, 321 MB) | 7.5 s / 480× | 8.2 s / 437× | 7.9 s / 454× |
| Phonon-2 `Encoder_lut6` (470 MB) | 7.4 s / 484× | 10.4 s / 347× | 10.9 s / 332× |

The sparse default is the fastest on every unit. On the GPU it runs as plain fp16 after its ~150 s load-time
expansion, which is why it beats the dense palette there; `.all` lands on the GPU path for the encoder.

