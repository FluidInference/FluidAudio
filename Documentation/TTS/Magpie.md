# Magpie TTS Multilingual (Swift Port)

Swift port of NVIDIA NeMo Magpie TTS Multilingual 357M, exported to CoreML.
Lives under `Sources/FluidAudio/TTS/Magpie/`.

## Status

> ⚠️ **Batch model. Beta / experimental.** Magpie is a **batch / offline
> TTS model** — not a streaming model. Below real-time on Apple Silicon
> (agg-RTFx ~0.41× on M2). Not for latency-sensitive use; prefer
> Kokoro / Kokoro ANE or PocketTTS for real-time. Initializing
> `MagpieTtsManager` logs a runtime beta warning at `.warning` level.

## Batch / offline inference (not streaming)

Magpie is a **batch model**. Despite NVIDIA marketing copy describing
Magpie as "targeting streaming applications," the released NeMo
inference path is batch-only:

- `MagpieTTSModel.infer_batch` / `do_tts` in
  `nemo/collections/tts/models/magpietts.py` runs the AR loop to
  completion, accumulating every audio code into `state.all_predictions`
  (≈ lines 6850–6860).
- After the loop, codes are concatenated
  (`torch.cat(state.all_predictions, dim=-1)`) and the codec is invoked
  exactly once per utterance via
  `self._codec_helper.codes_to_audio(predicted_codes, predicted_codes_lens)`
  (≈ lines 5334–5351 / 6889–6891).
- There is no incremental codec dispatch, no per-step yield, and no
  callback hook anywhere in the released NeMo path.

The Swift port preserves these batch semantics:

- `MagpieTtsManager.synthesize(text:...)` is the only entry point — it
  returns a single `MagpieSynthesisResult` after the full AR loop and
  codec pass complete. There is no streaming variant; long inputs are
  sentence-split internally via `MagpieChunker` so each chunk fits
  NanoCodec's 256-frame static-shape cap, but the full result is
  concatenated and returned as one buffer.
- Within `synthesize(...)`, AR(N+1) ‖ codec(N) chunk-level pipelining
  overlaps the next chunk's AR loop with the current chunk's codec
  pass. This is a wallclock optimization, not incremental yield.
- `MagpieNanocodec`'s v2/v3 24-frame chunked sliding-window dispatch is
  a memory + first-audio-latency optimization on the CoreML side; it is
  not codec-level streaming.

If you need a TTS model that genuinely streams audio frame-by-frame as
the AR loop emits codes, use **PocketTTS** (Mimi, ~1.2 s TTFT) or
**Kokoro** (parallel, no AR loop). Magpie's value prop is multilingual
coverage and the 5 built-in speaker contexts, not throughput or latency.

Functional but **below real-time — not for latency-sensitive use.**
On the full `minimax-english` 100-phrase corpus (M2, default compute
units), Magpie posts agg-RTFx **0.41×** with p50 warm synth ~19.8 s
and p95 ~57.5 s — most of the long tail comes from paragraph-length
news / story phrases (max 107 s on a single 18 s utterance). Cold
start ~19 s on warm ANE caches, dominated by first-call decoder_step
compile. The AR loop (`decoder_step` + sampler) dominates wall clock
and grows super-linearly with phrase length; the
[`outputBackings` fast path](Benchmarks.md#magpie-outputbackings-fast-path)
already eliminated the per-step KV reallocation cost. Further gains
likely need an MLX-backed LocalTransformer or a smaller-K/V variant.
For real-time use prefer Kokoro / Kokoro ANE (2–5× RTFx) or PocketTTS
(streaming, TTFT ~1.2 s); Magpie's value prop is multilingual coverage
(en/es/de/fr/it/vi/zh/hi) and 5 built-in speaker contexts, not
throughput.

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

## Cold-start mitigation (`warmup()`)

Magpie's CoreML graphs are **ANE-resident**. Apple's `anecompilerservice`
caches compiled graphs per-process, but invalidates that cache on
**system sleep / wake** and after long idle periods. The next
`synthesize` call after a wake event then stalls 10–20 s while the ANE
graphs recompile — this is the user-visible "TTS hang" downstream apps
see in cold-start scenarios (cf. VoiceInk issue #321).

`MagpieTtsManager` runs an automatic warmup at the end of `initialize()`,
so the first `synthesize` after app launch is fast. To mitigate the
post-wake recompile, call `warmup()` from your app's wake-handler:

```swift
public func warmup() async throws  // throws .notInitialized if called pre-init
```

Recommended pattern:

```swift
// 1. App launch — no manual warmup needed; initialize() runs it.
let manager = try await MagpieTtsManager.downloadAndCreate(languages: [.english])

// 2. App wake / re-foreground — re-warm in the background so the next
// synthesize() doesn't pay ANE recompile cost.
NotificationCenter.default.addObserver(
    forName: NSApplication.didBecomeActiveNotification, object: nil, queue: nil
) { _ in
    Task { try? await manager.warmup() }
}
```

Implementation runs a 16-step throwaway synthesis on a single `.` input
(CFG off, output discarded) to force first-dispatch specialization across
`text_encoder` → `prefill` → `decoder_step` → `nanocodec_decoder`. Total
wall time ≈ 1.5–2 s on M2.

`warmup()` is **safe to call repeatedly**. There is no internal
"already warm" guard — if the ANE state is still warm, the predict()
calls run fast; if the cache was invalidated, the method pays the
recompile so your next user-facing synthesize doesn't.

> **Note:** `warmup()` does not dispatch to a background queue
> internally. Wrap the call in `Task { ... }` if you don't want to
> block the calling context. The same applies to `initialize()` — both
> are normal `async` functions that run on whatever executor the caller
> awaits them from.

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
