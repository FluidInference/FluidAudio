# Magpie TTS Multilingual (Swift Port)

Swift port of NVIDIA NeMo Magpie TTS Multilingual 357M, exported to CoreML.
Lives under `Sources/FluidAudio/TTS/Magpie/`.

## Status

Functional but **quite slow — needs significant perf work, not for real-time
or latency-sensitive use.** First synth on a fresh process is dominated by
CoreML model load + first-call ANE compile (~30 s); warm synths run at
~96 s wall for an 8-word English sentence on M-series, i.e. RTFx ≈ **0.04**
(~25× slower than realtime). Whether the throughput ceiling is a model
characteristic, a CoreML conversion limitation, or both is still being
investigated and is expected to improve in subsequent iterations. For
real-time use prefer Kokoro (~20× RTFx) or PocketTTS (~1.5–2× RTFx);
Magpie's value prop is multilingual coverage and the 5 built-in speaker
contexts, not throughput.

Audio quality is perceptually clean across all 5 speakers and ASR-clean on
4/5; speaker 0 has a single trailing-word artifact ("…and") attributable
to fp16 sampler-trajectory drift, not a structural bug.

Not yet covered: Japanese (deferred — needs OpenJTalk XCFramework + MeCab
dict), CFG performance optimization, MLX-backed LocalTransformer,
throughput investigation (the headline gap).

## Architecture

```
text → MagpieTokenizer (per-language) → text_encoder.mlmodelc
                                          ↓
speaker_N.npy (110×768) → decoder_prefill.mlmodelc (1 batched call) ──┐
                                                                      ↓
                            ┌──── KV cache (12 layers × [2,1,512,12,64] fp16)
                            ↓
                   AR loop (decoder_step.mlmodelc, ≤500 steps):
                     ├─ LocalTransformer (Swift, Accelerate+BNNS)
                     ├─ Sampler (top-k=80, temp=0.6, forbidden mask)
                     ├─ embed sampled (8) codes → next decoder_step input
                     └─ stop on audio_eos_id (2017) or maxSteps
                            ↓
                   nanocodec_decoder.mlmodelc → 22 050 Hz Float32 PCM
```

## Compute placement (verified end-to-end)

| Model              | Compute units            | Reasoning                                                                                                    |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `text_encoder`     | `.cpuAndNeuralEngine`    | Runs on ANE; ~3.5× vs CPU.                                                                                   |
| `decoder_prefill`  | `.cpuAndNeuralEngine`    | Runs on ANE; ~3.2× vs CPU. One batched call replaces 110 sequential `decoder_step` calls.                    |
| `decoder_step`     | **`.cpuAndGPU`**         | Pinned. ANE compile fails (`MILCompilerForANE: ANECCompile() FAILED`) due to rank-4 split-K/V scatter; on `.cpuAndNeuralEngine` it falls back to CPU at ~hundreds-of-ms cost per call. GPU (Metal MPS) is fastest. Verified: 96 s warm vs 103 s warm on `.cpuAndNeuralEngine`. |
| `nanocodec_decoder`| `.cpuAndNeuralEngine`    | Runs on ANE.                                                                                                 |

The pin is implemented in `MagpieModelStore.swift:60` — caller-supplied
`computeUnits` is honored for all models *except* `decoder_step`, which is
forced to `.cpuAndGPU` (or `.cpuOnly` if the caller asked for `.cpuOnly`).

## Performance journey

Three optimizations landed during the port; numbers are warm-avg wall time on
M-series for an 8-word English sentence.

| Stage                                                   | Wall (warm) | Speedup |
| ------------------------------------------------------- | ----------- | ------- |
| Baseline: 110-step prefill loop, ANE on decoder_step    | ~420 s      | 1.0×    |
| **Wire `decoder_prefill.mlmodelc` (1 batched call)**    | ~110 s      | 3.8×    |
| **Pin decoder_step to `.cpuAndGPU`**                    | ~96 s       | 4.4×    |

Asset was already on HF (`FluidInference/magpie-tts-multilingual-357m-coreml`)
and downloaded by `MagpieResourceDownloader`, just unused. `prefillFast`
(`MagpiePrefill.swift:23`) replaces 110 sequential `decoder_step` calls with
one `decoder_prefill` call whose 12 stacked-K/V outputs (`var_208`, `var_374`,
… `var_1958`, each `[2, 1, 512, 12, 64]` fp16) are sliced via two `memcpy`s
per layer into the KV cache (`MagpieKvCache.seedFromPrefillOutputs`).

## Public API

```swift
let manager = try await MagpieTtsManager.downloadAndCreate(
    languages: [.english],
    cacheDirectory: nil,
    computeUnits: .cpuAndNeuralEngine,   // decoder_step pinned to GPU internally
    progressHandler: nil
)

let result = try await manager.synthesize(
    text: "Hello world.",
    speaker: .john,
    language: .english,
    options: .default
)
// result.samples : [Float]   (22 050 Hz)
// result.codeCount : Int
// result.durationSeconds : Double
```

## CLI

```bash
# Download all assets eagerly
swift run fluidaudiocli magpie download

# Synth
swift run fluidaudiocli magpie text "Hello world." --speaker 0 --output hello.wav
```

Parity, probe, and compute-plan tooling live upstream in `mobius` (Python) —
they exercise the export pipeline and are out of scope for the Swift runtime.

## Known issues

1. **spk0 trailing-word drift.** ASR shows a stray word at the end (e.g.
   "…seashore, and"). Stage-by-stage parity probe (in `mobius`) localizes it
   to fp16 sampler-trajectory non-determinism between Python+CoreML reference
   and Swift+CoreML host: prefill SNR degrades L0=64 dB → L11=44 dB through
   the 12-layer cache, then compounds in the AR loop. CoreML itself is
   consistent between languages; the drift is host-floating-point + RNG/sampler
   ordering. Not user-perceptible on speakers 1–4.

2. **`decoder_step` ANE compile failure is real.** Earlier benchmark with
   zeroed `position` scalars showed a 3× ANE speedup; that was misleading —
   with real incrementing positions the ANEF compile fails at runtime per
   call. Keep the `.cpuAndGPU` pin.

## File map

```
Sources/FluidAudio/TTS/Magpie/
├── MagpieTtsManager.swift                # public actor
├── MagpieConstants.swift                 # shapes, ids, file names, HF repo id
├── MagpieError.swift
├── MagpieTypes.swift
├── Assets/
│   ├── MagpieModelStore.swift            # actor; loads 4 mlmodelcs, per-model compute units
│   ├── MagpieResourceDownloader.swift    # HF download via DownloadUtils
│   ├── MagpieConstantsStore.swift
│   └── MagpieLocalTransformerWeights.swift
├── LocalTransformer/
│   ├── MagpieLocalTransformer.swift      # 1-layer transformer (attention + FFN) via Accelerate (cblas_sgemm) + BNNS (GELU)
│   └── MagpieSampler.swift               # top-k + temp + forbidden mask + CFG merge
├── Pipeline/
│   ├── Preprocess/                       # per-language tokenizers + IPA override
│   └── Synthesize/
│       ├── MagpieSynthesizer.swift       # orchestrates encode → prefill → AR → nanocodec
│       ├── MagpieKvCache.swift           # 12 layers × (cache, position); seedFromPrefillOutputs
│       ├── MagpiePrefill.swift           # prefillFast (batched) + prefill (110-step fallback)
│       └── MagpieNanocodec.swift
└── Shared/
    ├── NpyReader.swift                   # .npy v1 (fp32/fp16/int)
    └── MagpieMT19937.swift               # deterministic RNG matching Python reference

Sources/FluidAudioCLI/Commands/
└── MagpieCommand.swift                   # dispatch (download / text)
```
