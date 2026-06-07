# ANE Profiler

| | |
|---|---|
| **Measured** | 2026-06-05 |
| **Machine** | MacBook, Apple Silicon M5, macOS (Darwin 25.x) |
| **Config** | `computeUnits = .cpuAndNeuralEngine` (ANE allowed; capability, not production override) |
| **Metric** | `MLComputePlan` `preferredComputeDevice` per op, counted (mlprogram ops / nn layers) |
| **Size** | on-disk `.mlmodelc` bundle size (MB) |
| **Lat** | per-call latency (ms). **Real audio, warm** for TDT v3, EOU, Nemotron, Pyannote offline, TTS. **Synthetic** (zero-input) for ja / zh / CTC-110M. One-time measurement (see Latency below). |
| **Tool** | `Scripts/ane_profile.swift` (device split, reproducible) |

> **Rule of thumb: the smaller the model, the less the ANE matters. Under ~50 MB it often isn't worth
> it at all.** On a small graph the fixed cost of moving tensors to the Neural Engine and back
> outweighs the speedup, so CoreML keeps it on CPU. Put ANE effort on the big graphs.
>
> All measured on Macbooks like M5.
>
> **Reading the Lat column.** Stages that run once per call show per-call time: `/ chunk`
> (encoder/preprocessor, once per audio chunk) or `/ call` (single-shot). Stages that loop show
> `T ms (N× @ p)` = ran N times at p ms each, T ms total. The loop count N **scales with output
> length** (ASR decoders/joints, PocketTTS, so N is for this test clip), except Supertonic's
> VectorEstimator, which is **fixed** at the denoising-step count (8).

---

# ASR

| Model | Type | Chunk | ANE | GPU | CPU | ops | Size | Heavy graph → device |
|-------|------|------:|----:|----:|----:|----:|-----:|----------------------|
| Parakeet CTC 110M | batch (sliding-window) | 15 s (2 s overlap) | **97%** | 0% | 3% | 1353 | 101 MB | AudioEncoder → ANE |
| Parakeet CTC Chinese | batch (sliding-window) | 15 s (2 s overlap) | **96%** | 0% | 4% | 1443 | 583 MB | Encoder → ANE |
| Parakeet TDT v3 | batch (sliding-window) | 15 s (2 s overlap) | **93%** | 0% | 7% | 1484 | 463 MB | Encoder → ANE¹ |
| Parakeet TDT Japanese | batch (sliding-window) | 15 s (2 s overlap) | **93%** | 0% | 7% | 1490 | 611 MB | Encoder → ANE |
| Parakeet EOU | streaming | 160 / 320 / 1280 ms | **92%** | 0% | 8% | 1243 | 233 MB | streaming_encoder → ANE |
| Nemotron Multilingual | streaming | 1120 / 2240 ms | **92%** | 0% | 8% | 1786 | 636 MB | encoder → ANE |
| Nemotron EN | streaming | 560 / 1120 / 2240 ms | **90%** | 0% | 10% | 1788 | 602 MB | encoder_int8 → ANE |

¹ v3's encoder *can* run 99% ANE but **ships on `.cpuAndGPU`** (+8% RTFx vs ANE on M-series). In
production it runs on GPU, not ANE. The table shows ANE *capability* with the default config.

**Component detail**

### Parakeet TDT v3
Latency here is **measured on real audio** (7.8 s clip, production config, 5-run average), not synthetic.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| Encoder | 99% | 0% | 1% | 1385 | 426 MB | 28.2 / chunk |
| Decoder | 0% | 0% | 100% | 24 | 23 MB | 9 ms (40× @ 0.23) |
| Joint | 0% | 0% | 100% | 24 | 13 MB | 23 ms (49× @ 0.46) |
| Preprocessor | 0% | 0% | 100% | 51 | 1 MB | 3.2 / chunk |

> Per 7.8 s of audio: 1 encoder call, ~40 decoder + ~49 joint steps. The joint loop totals ~22 ms,
> rivaling the single 28 ms encoder call.

### Parakeet TDT Japanese
Lat is **synthetic** (zero-input), not real audio (no CLI transcribe path). Runs ~2x low; treat as a
lower bound.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| Encoder | 99% | 0% | 1% | 1386 | 580 MB | 23.3 |
| CtcDecoder | 100% | 0% | 0% | 4 | 7 MB | 0.25 |
| Decoderv2 | 0% | 0% | 100% | 24 | 17 MB | 0.13 |
| Jointerv2 | 0% | 0% | 100% | 24 | 6 MB | 0.16 |
| Preprocessor | 0% | 0% | 100% | 52 | 1 MB | 0.92 |

> `CtcDecoder` lands on ANE despite being tiny (7 MB), the exception to the rule that small models stay
> on CPU.

### Parakeet CTC Chinese
Lat is **synthetic** (zero-input), not real audio; the zh CTC pipeline uses a separate manager not
yet instrumented. Runs ~2x low; treat as a lower bound.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| Encoder fp32 | 99% | 0% | 1% | 1385 | 1130 MB | 24.2 |
| Encoder int8 | 99% | 0% | 1% | 1385 | 568 MB | 23.4 |
| Decoder | 100% | 0% | 0% | 6 | 14 MB | 0.69 |
| Preprocessor | 0% | 0% | 100% | 52 | 1 MB | 0.95 |

### Parakeet CTC 110M
Lat is **synthetic** (zero-input); this is the keyword-spotting CTC variant with no transcribe CLI.
Runs ~2x low; treat as a lower bound.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| AudioEncoder | 100% | 0% | 0% | 1315 | 98 MB | 8.5 |
| CtcHead | 0% | 0% | 100% | 6 | 2 MB | 0.18 |
| MelSpectrogram | 0% | 0% | 100% | 32 | 1 MB | 0.57 |

### Parakeet EOU (1280ms)
Latency **measured on real audio**, warm (the device-split/size columns are the 1280ms bundle; the Lat
column was run on the **160ms** default CLI variant, so the encoder figure is that variant's).

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| streaming_encoder | 97% | 0% | 3% | 1174 | 220 MB | 6.5 / chunk |
| decoder | 0% | 0% | 100% | 14 | 8 MB | 30 ms (229× @ 0.13) |
| joint_decision | 0% | 0% | 100% | 21 | 3 MB | 28 ms (229× @ 0.12) |

> EOU doesn't ship a CoreML `preprocessor`. It computes mel features in native Swift
> (`AudioMelSpectrogram`), so there's no preprocessor row. (Nemotron, by contrast, *does* run its
> CoreML preprocessor.) decoder/joint go through the shared `RnntDecoder` (~229 steps for 7.8 s audio).

### Nemotron EN
Latency **measured on real audio**, warm (default 1120ms chunk, 7 chunks for 7.8 s). Uses the separate
decoder then joint path.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| encoder_int8 | 97% | 0% | 3% | 1672 | 564 MB | ~13 / chunk (1st ~290 cold) |
| preprocessor | 0% | 0% | 100% | 47 | 1 MB | 2.4 / chunk |
| decoder | 0% | 0% | 100% | 24 | 15 MB | 44 ms (148× @ 0.30) |
| joint | 0% | 0% | 100% | 12 | 4 MB | 23 ms (148× @ 0.15) |

### Nemotron Multilingual
Latency **measured on real audio**, warm (2240ms chunk, 4 chunks for 7.8 s). Default decode is the
fused `decoder_joint` (B1).

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| encoder | 96% | 0% | 4% | 1680 | 540 MB | 9.7 / chunk |
| preprocessor | 0% | 0% | 100% | 47 | 1 MB | 1.9 / chunk |
| decoder_joint | 54% | 0% | 46% | 28 | 47 MB | 79 ms (168× @ 0.47) |
| joint | 100% | 0% | 0% | 12 | 19 MB | unused (B1 default) |
| decoder | 0% | 0% | 100% | 19 | 29 MB | unused (B1 default) |

> Unlike EN, the multilingual `joint` is fully ANE and `decoder_joint` is mixed (54% ANE).

---

# VAD

| Model | Type | Chunk | ANE | GPU | CPU | ops | Size | Lat ms |
|-------|------|------:|----:|----:|----:|----:|-----:|-------:|
| Silero VAD (single graph) | streaming | 256 ms | 0% | 0% | 100% | 357 | 2 MB | 0.19 |

---

# Diarization (offline)

| Pipeline | Type | Chunk | ANE | GPU | CPU | ops | Size |
|----------|------|------:|----:|----:|----:|----:|-----:|
| Pyannote offline | batch (offline) | 10 s window | 49% | 0% | 51% | 233 | 22 MB |

**Component detail**

### Pyannote offline
Latency **measured on real audio**, warm (`process --mode offline`, 7.8 s clip). Cold first-call is far
higher (see warmup note below): segmentation was 1293 ms and embedding 410 ms/call on the cold run.

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| Embedding (speaker embed) | 93% | 0% | 7% | 124 | 13 MB | 7.9 / call |
| Segmentation | 0% | 0% | 100% | 58 | 6 MB | 55.4 / call |
| FBank | 0% | 0% | 100% | 33 | 2 MB | 3.2 / call |
| PldaRho | 0% | 0% | 100% | 18 | 1 MB | 0.15 / call |

> Warm segmentation (55 ms) is the heaviest stage and is 100% CPU. The **cold-start penalty is the real
> story here**: the first ANE/CPU call paid ~1.3 s (segmentation) and ~0.4 s (embedding) for model
> compile + residency, 10-25x the warm cost.

---

# TTS

Latency **measured on real synthesis**, warm (one short sentence; `tts --backend …`).

**Summary**

| Model | Type | ANE | GPU | CPU | ops | Size | Heavy graph → device |
|-------|------|----:|----:|----:|----:|-----:|----------------------|
| Kokoro ANE (7-stage) | batch (per utterance) | 75% | 0% | 25% | 1472 | 83 MB | Vocoder → ANE |
| Supertonic (`--ve-variant fp16`, legacy) | batch (8-step diffusion) | 30% | 0% | 70% | 1365 | 192 MB | VectorEstimator → **CPU** (dynamic shapes can't use ANE) |
| Supertonic (default, int4 L-bucketed) | batch (8-step diffusion) | ~90% | 0% | ~10% | 1289 | 102 MB | VectorEstimator → **ANE** (fixed L-buckets) |
| PocketTTS (v2.1) | streaming (autoregressive) | ~9% | ~31% | ~60% | 2629 | ~330 MB | flow_decoder_fused → **ANE**; flowlm/cond → GPU; mimi → CPU |

**Component detail**

### Kokoro ANE (7-stage)
| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| Albert | 94% | 0% | 6% | 310 | 6 MB | 6.5 |
| PostAlbert | 22% | 0% | 78% | 98 | 14 MB | 3.7 |
| Alignment | 0% | 0% | 100% | 19 | 1 MB | 0.8 |
| Prosody | 99% | 0% | 1% | 138 | 9 MB | 56.5 |
| Noise | 0% | 0% | 100% | 239 | 5 MB | 61.6 |
| Vocoder | 99% | 0% | 1% | 655 | 47 MB | 71.8 |
| Tail | 0% | 0% | 100% | 13 | 1 MB | 9.6 |

### PocketTTS (v2.1)
Autoregressive: stages run many steps per utterance. Measured on M-series /
macOS 26 with the v2.1 optimized packs. Only the fused flow decoder reaches the
ANE; flowlm/cond run on GPU (the rank-5 KV-cache `scatter` is rejected by the
ANE compiler at **any** precision), and mimi is CPU (fp16 streaming-state
feedback produces audible artifacts on the ANE, and it is compute-bound anyway).

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| cond_prefill | 0% | 100% | 0% | 550¹ | 127 MB | ~5 ms (1× @ 4.8) |
| flowlm_step (fp16) | 0% | 100% | 0% | 556 | 145 MB | 149 ms (43× @ 3.46) |
| flow_decoder_fused | **100%** | 0% | 0% | 1252 | 18 MB | 46 ms (42× @ 1.09) |
| mimi_decoder | 0% | 0% | 100% | 271¹ | 41 MB | 302 ms (42× @ 7.2) |

¹ `MLComputePlan` crashes on `cond_prefill` (ANE compile) and `mimi_decoder`
(streaming state), so their device split is inferred from the runtime config
(GPU / CPU) and the op counts are from `model.mil` (non-const ops, the same
metric MLComputePlan reports — verified equal on the fused decoder: 1252).

> v2.1 cut the per-utterance pipeline ~905 ms → ~452–520 ms (**~1.8× RTFx**),
> device-verified end-to-end (Whisper exact). Wins: **fused flow decoder** — the
> 8-step LSD Euler loop unrolled into one call (336→42 dispatches/utt), and the
> fat scatter-free fp16 graph flips **0% → 100% ANE** (the single-step kernel was
> always rejected); **cond_prefill** — whole conditioning block in one call
> (18→1); **fp16 flowlm**. The earlier "flowlm 1.97× on ANE" claim did **not**
> reproduce — flowlm is GPU. mimi is the remaining floor (~60% of wall-time),
> compute-bound (not overhead-bound — an MLState micro-bench showed state
> marshalling is only ~0.5 ms/call). Voice cloning uses the unchanged v2
> `mimi_encoder` (repo-root, language-agnostic; still crashes
> `MLComputePlan.load`) — not part of the v2.1 synthesis path.

### Supertonic
`VectorEstimator` runs once per denoising step (default 8) and is the heaviest stage. The **default is
now the fixed-length int4 (L-bucketed) build** (`.aneBucketed(.int4)`): ~94% on the ANE, ~2.7× faster
end-to-end, with 4-bit k-means palettization that is perceptually clean. The synthesizer pads each
chunk's latent up to the smallest bucket ≥ its length (L ∈ {128, 256, 512}; 128 covers the common
case). The legacy **fp16 dynamic** build (`--ve-variant fp16`) uses RangeDim shapes Core ML **cannot
place on the ANE**, so it stays on CPU; the `ANECCompile() FAILED` line it emits is non-fatal noise.
Verified M5 Pro / macOS 26.5; see [Supertonic3 docs](TTS/Supertonic3.md#vectorestimator-variants).

| Component | ANE | GPU | CPU | ops | Size | Lat ms |
|-----------|----:|----:|----:|----:|-----:|-------:|
| TextEncoder | 98% | 0% | 2% | 308 | 18 MB | 1.2 |
| DurationPredictor | 0% | 0% | 100% | 195 | 2 MB | 2.5 |
| VectorEstimator (fixed L128, int4, **default**) | 94% | 0% | 6% | 679 | 33 MB | ~31 ms (8× @ 3.8)² |
| Vocoder | 100% | 0% | 0% | 107 | 49 MB | 10.5 |

² Fixed-build split + per-step latency from `MLComputePlan` + warm CPU-vs-NE timing at L128 (NE 3.8 ms
vs CPU-only 14.2 ms/step). A cold first call additionally pays a one-time ANE compile.

---


## How to measure

### Method 1: `MLComputePlan` (device split, what produced this report)

`MLComputePlan` (macOS 14.4+ / iOS 17.4+) loads a compiled model and reports, per operation, the
compute device CoreML will prefer. Counting ops by device gives the ANE/GPU/CPU split with no Xcode
and no instrumentation. Implemented in [`Scripts/ane_profile.swift`](../Scripts/ane_profile.swift);
the core is:

```swift
let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
// walk plan.modelStructure (.program ops / .neuralNetwork layers / .pipeline submodels),
// call plan.deviceUsage(for: op)?.preferred, and tally .neuralEngine / .gpu / .cpu
```

Reproduce the device split:

```bash
swiftc -O -target arm64-apple-macos14.4 Scripts/ane_profile.swift -o /tmp/ane_profile
/tmp/ane_profile path/to/Encoder.mlmodelc                # multiple args print a TOTAL
/tmp/ane_profile --units gpu path/to/Encoder.mlmodelc    # force a compute-unit policy
```

### Latency (one-time, real audio)

The latency numbers were a **one-time measurement**: temporary env-gated timers wrapped each model's
`prediction` call while running the real pipelines on a real benchmark clip (`transcribe`,
`nemotron-transcribe`, `parakeet-eou`, `process --mode offline`, `tts`). That instrumentation was
removed after measuring (not retained in the codebase). To refresh, re-add a timer around the
prediction sites in the relevant manager and re-run.

> **Warm vs cold matters a lot.** The first call to each model pays ANE compile + weight residency,
> which can be 10-50x the warm per-call cost (Pyannote segmentation: 1293 ms cold, 55 ms warm). The
> doc's numbers are warm (second run). Many-call stages (encoders, decode loops) amortize this;
> few-call stages (diarization) are dominated by it, so cold start is the real cost on a fresh launch.

### Method 2: Xcode Core ML Performance Report

Open the `.mlpackage`/`.mlmodelc` in Xcode → **Performance** tab → **+** → pick a connected device.
Per-layer Neural Engine / GPU / CPU breakdown on real hardware. Best for *which* layers fall off the
ANE and for per-device numbers (iPhone ANE differs from M-series). GUI only, not automatable.

### Method 3: `powermetrics` (runtime confirmation)

Methods 1 and 2 show the plan; this shows what the silicon actually did:

```bash
sudo powermetrics --samplers ane_power -i 200   # watch ANE power while inference runs
```

If ANE power stays near 0 mW during transcription, the model is not really on the ANE regardless of
config.

## Gotchas that knock work off the ANE

- **Small graphs (under ~50 MB)** stay on CPU by design; transfer overhead beats the speedup. Not a bug.
- **fp16 only.** fp32 ops fall to CPU/GPU (and watch fp16 **NaN** on some encoders).
- **Dynamic shapes / big reshapes** land on CPU; static-shape graphs stay on ANE. Cohere's v2 decoder
  fixed its attention mask to a literal shape specifically to stay ANE-resident.
- **`MLState` is ANE-incompatible on iOS 18** for some configs; stateful decoders can get bumped off.
- **ANE means fast inference, slow load.** First ANE init is multi-second; a model can be 90% ANE and
  still feel slow cold. That's load, not inference. Separate the two.
- **GPU sometimes wins.** Parakeet v3's encoder is deliberately on GPU (+8% RTFx, WER-neutral on
  M-series). "More ANE" is the goal for **iOS power**, not automatically for Mac throughput.
- **Optimization hints can backfire.** `MLOptimizationHints(reshapeFrequency: .infrequent,
  specializationStrategy: .fastPrediction)` regressed RTFx 26% on a static-shape encoder. Re-bench
  before enabling. See `Sources/FluidAudio/Shared/MLModelConfigurationUtils.swift`.
