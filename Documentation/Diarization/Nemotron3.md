# Nemotron 3 Diarization

FluidAudio support for NVIDIA's **Nemotron 3 Diarization** (streaming Sortformer
successor): up to **8 speakers**, arrival-order speaker channels, 10 ms output
resolution, streaming and offline profiles from a single checkpoint.

> **Model availability:** the converted CoreML presets are published at
> [`FluidInference/nemotron-3-diarization-coreml`](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml)
> (gated until NVIDIA's public release — request access, then set `HF_TOKEN`;
> ungated afterwards). `Nemotron3Models.loadFromHuggingFace` downloads one preset
> bundle on first use; `Nemotron3Models.load(config:directory:)` loads a local copy.

## Quick start

```swift
import FluidAudio

let config = Nemotron3Config.fast32  // recommended default
let models = try await Nemotron3Models.loadFromHuggingFace(config: config)
let diarizer = Nemotron3Diarizer(config: config, models: models)

let (probs, frames) = try diarizer.processComplete(audioSamples)  // 16 kHz mono
let segments = Nemotron3Diarizer.segments(probabilities: probs, frameCount: frames)
// arrival-ordered speaker segments at 10 ms resolution, up to 8 speakers
```

Optional VAD gating for silence-heavy audio (skips inference over non-speech while
preserving the output timeline):

```swift
let (probs, frames) = try diarizer.processComplete(audioSamples, speechMask: mask)
```

## Streaming (microphone / live audio)

Feed 16 kHz samples as they arrive; a chunk runs as soon as its right context is
buffered (`config.latencySeconds` after the chunk starts) and returns
`config.chunkSeconds` of new 10 ms frames. The path is frame-exact with
`processComplete` on the same audio.

```swift
diarizer.reset()
for samples in microphoneSteps {            // any granularity, e.g. 320 ms
    diarizer.appendAudio(samples)
    for chunk in try diarizer.processBufferedAudio() {
        timeline.append(contentsOf: chunk.probabilities)   // [frames * 8]
    }
}
for chunk in try diarizer.finishStream() {  // flushes the trailing partial chunk
    timeline.append(contentsOf: chunk.probabilities)
}
```

`Nemotron3Diarizer` is not thread-safe: own it from one actor or task. For
word→speaker attribution pair it with
`StreamingUnifiedAsrManager.consumeWordTimings()` and pick, per word, the speaker
slot with the most activity over the word's span.

## Choosing a preset

Latency = (chunk + right context) x 80 ms — the audio buffered before a result is
final. Audio chunk = new audio consumed per model call; larger chunks amortize the
fixed speaker-cache cost, which *improves* accuracy while increasing throughput.

| Preset | Size | Audio chunk/call | Latency | Pros | Cons |
|---|---|---|---|---|---|
| `low` | 190 MB | 0.72 s | 1.04 s | Best quality at real streaming latency; NVIDIA's reference config | Heaviest ANE use per second of audio |
| `fast` | 190 MB | 0.72 s | 1.04 s | ~3x cheaper per call than `low` — leaves ANE room for concurrent ASR | Slightly lower accuracy than `low` |
| `fast32` | 190 MB | 2.56 s | 2.88 s | **Recommended default** — `low`-level accuracy at near-`fast` cost | Latency too high for live-caption UX |
| `fast128` | 190 MB | 10.24 s | 10.56 s | Best accuracy of the streaming lineup; highest streaming throughput | Near-live only; results trail by ~10 s |
| `offline` | 190 MB | 27.2 s | 30.4 s | Highest accuracy; fastest batch profile | GPU-only (ANE compiler limit); 30 s latency |
| `s32-split-w8a8`* | **95 MB** | 2.56 s | 2.88 s | Half size, 100% ANE-resident graph, zero GPU use — the iOS pick | Requires `pre_encode_proj_t.bin` alongside the model |
| `c128-split-w8a8`* | **95 MB** | 10.24 s | 10.56 s | Batch throughput without touching the GPU | Same split-mode requirement; ~10 s latency |

\* Split-graph mode (`splitGraph` config flag): feature stacking and the 1024→512
projection run host-side (one reshape + one `cblas_sgemm`), leaving a pure
floating-point transformer graph that is fully ANE-resident and quantizes cleanly
to W8A8. `Nemotron3Models.runSplit` handles the host-side work transparently.

Quick chooser: hard ~1 s latency → `fast` (sharing the ANE with ASR) or `low`
(diarizer owns the ANE) · general use → `fast32` · latency-flexible quality →
`fast128` · recorded archives on a Mac → `offline` · iPhone/iPad, battery, or
GPU-busy systems → the `split-w8a8` pair.

Additional card profiles (`verylow`, `ultra`) and intermediate configurations exist
via `Nemotron3Config.preset(named:)` / custom initializers but are dominated by the
presets above for typical use.

## CLI

```bash
# Diarize a file (prints segments; --output writes RTTM). Downloads the preset on first use.
swift run fluidaudiocli nemotron3-diarize audio.wav --variant fast32

# Same audio through the live path (appendAudio / processBufferedAudio / finishStream)
swift run fluidaudiocli nemotron3-diarize audio.wav --variant fast32 --streaming

# Benchmark against AMI / VoxConverse harnesses
swift run fluidaudiocli nemotron3-benchmark --variant fast32 --collar 0

# Batch processing with concurrent GPU workers
swift run fluidaudiocli nemotron3-batch --workers 2 --files a,b,c
```

Useful flags: `--models <dir>` (local bundles instead of the HF download; required for
`verylow`/`ultra` and sweep variants), `--compute-units ane|gpu|all`, `--profile`
(per-stage wall breakdown), `--vad` (Silero-gated processing), sweep flags
(`--chunk-len`, `--fifo`, `--spkcache`, `--rc`, `--update-period`) for
custom-converted models.

## Implementation notes

- **State lives host-side**: the CoreML model is a pure forward pass over
  `[speaker cache | FIFO | chunk]`; `Nemotron3StateUpdater` ports NeMo's
  `streaming_update_async` (cache compression, learned silence embedding, FIFO
  eviction) in Swift. Closed-loop output matches the NeMo reference at 99.995%
  frame agreement on real audio.
- Model outputs are fp16 with padded rows; readback uses a stride-aware
  `vDSP_mmov` compaction (naive reads silently scramble or run ~40x slower —
  see `Nemotron3TensorLayoutTests`).
- Long ANE-route runs require the per-chunk autoreleasepool in `processComplete`
  (IOSurface-backed outputs otherwise exhaust the pool after thousands of calls).
- The mel frontend is the shared `AudioMelSpectrogram` (128 mel, 10 ms hop,
  no normalization) — the same family as the Nemotron ASR models.
