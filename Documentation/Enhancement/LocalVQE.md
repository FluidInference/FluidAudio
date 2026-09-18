# LocalVQE — Echo Cancellation & Noise Suppression

`LocalVqeManager` runs [LocalVQE](https://github.com/localai-org/LocalVQE)
(Apache-2.0), a compact neural model for acoustic echo cancellation (AEC),
noise suppression and dereverberation of 16 kHz speech. It is a streaming,
CPU-tuned derivative of DeepVQE (Indenbom et al., Interspeech 2023). Typical
use: cleaning up call audio captured without headphones, where the mic picks
up what the loudspeaker plays.

**Beta.** Numerically equivalent to the upstream PyTorch and GGML engines and
scored identically to GGML on the 800-clip AEC-Challenge blind set (see
[Quality](#quality-aec-challenge-blind-test-set)); not yet exercised inside
production call pipelines.

## Inputs

The model takes two 16 kHz mono signals of equal length:

- **mic** — the microphone capture.
- **reference** — the far-end signal: a loopback of what the loudspeaker
  played. Without it the model still denoises and dereverberates; pass
  silence (`process(mic:)` does this for you).

Output is 16 kHz mono, same length as the input, sample-aligned. Level
matches the upstream GGML engine (the OBS plugin and HF demo).

## Quick start

```swift
import FluidAudio

let vqe = try await LocalVqeManager()               // downloads v1.3 (256 ms chunk) on first use
let clean = try await vqe.process(mic: micSamples, reference: farEndSamples)

// Files (any format / rate; converted to 16 kHz mono)
let cleanFile = try await vqe.process(micURL: micURL, referenceURL: speakerURL)
```

### Streaming

```swift
let vqe = try await LocalVqeManager(config: LocalVqeConfig(chunk: .realtime16ms))
let stream = try await vqe.makeStream()

// Push buffers of any size as they arrive (mic and reference must be equal length).
let out = try await stream.enhance(mic: micBuffer, reference: refBuffer)

// End of clip: drain the delay line so total output == total input.
let tail = try await stream.flush()
```

Streams from one manager share its model and may run concurrently: inference
uses Core ML's async prediction API, which Apple documents as thread-safe.

`enhance` returns samples as whole model calls complete. Output sample `i`
corresponds to input sample `i`, delivered one hop (256 samples, 16 ms)
after the input that produced it plus whatever is still buffered toward the
next call. `flush()` resets the stream; call `reset()` to start a new clip
without flushing.

## Configuration

```swift
LocalVqeConfig(
    variant: .v13,          // .v13 (4.8M params, default) or .v12 (1.3M, ~1/4 the cost)
    chunk: .batch256ms,     // .batch256ms (files) or .realtime16ms (live capture)
    computeUnits: .cpuOnly  // fp32 models; CPU is fastest for the 16 ms chunk
)
```

Both variants are joint AEC + NS + dereverb models. The chunk size only
changes how many 16 ms hops each Core ML call consumes; the audio is
bit-identical either way.

| Variant | Chunk | Compute | Per-call p50 | RTFx |
|---|---|---|---:|---:|
| v1.3 | 256 ms | CPU | 7.1 ms | 36× |
| v1.3 | 16 ms | CPU | 1.2 ms | 14× |
| v1.2 | 256 ms | CPU | 4.2 ms | 60× |
| v1.2 | 16 ms | CPU | 0.7 ms | 24× |

Apple M5 Pro, release build, `fluidaudiocli enhance --streaming`. RTFx is
audio-per-call ÷ p50 latency. GPU gives ~15% on the 256 ms chunk at the cost
of a ~110 ms first-call compile; ANE is not used (see below).

## CLI

```bash
swift run -c release fluidaudiocli enhance mic.wav --reference speaker.wav --output clean.wav
swift run -c release fluidaudiocli enhance mic.wav --output clean.wav            # NS/dereverb only
swift run -c release fluidaudiocli enhance mic.wav -r speaker.wav --chunk 16ms --streaming
```

`--variant v1.2`, `--compute-units gpu`, `--buffer-samples N` (streaming
buffer size) and `--model-dir DIR` (load local `.mlmodelc` bundles) are also
available; `--help` lists everything.

## Models

HuggingFace: [FluidInference/localvqe-coreml](https://huggingface.co/FluidInference/localvqe-coreml).
One `.mlmodelc` per (variant, chunk); only the configured one is downloaded
(19 MB for v1.3, 5 MB for v1.2). Cached under
`~/Library/Application Support/FluidAudio/Models/localvqe/`.

Manual loading:

```swift
let vqe = try LocalVqeManager(config: config, modelDirectory: URL(fileURLWithPath: "/path/with/mlmodelc"))
```

The models are fp32 streaming exports with explicit state: every call takes
`mic`/`ref` plus 33 `in_*` state tensors and returns `enhanced` plus the
matching `out_*` tensors. fp16 was rejected: it drops parity with the
reference from 102 dB to 5 dB (CPU) / 33 dB (ANE) because the power-law
front-end epsilons underflow and the S4D recurrence accumulates error.
Conversion lives in the [mobius](https://github.com/FluidInference/mobius)
repo under `models/enhancement/localvqe/coreml`.

## Parity

Upstream double-talk demo clip (10 s), Swift `LocalVqeStream` output:

| Against | max abs diff | SNR |
|---|---:|---:|
| Upstream PyTorch reference (fp32, ×2 to the GGML level) | 3.8e-5 | 74 dB (16-bit WAV limited) |
| Upstream GGML CLI (`localvqe-v1.3-4.8M-f32.gguf`) | 2.8e-5 | 80 dB |

Streaming in 100 / 256 / 1000 / 4096-sample buffers and whole-clip
processing produce the same audio to 1e-5.

## Quality: AEC-Challenge blind test set

The upstream quality table is AECMOS on the ICASSP 2022 AEC-Challenge blind
set (800 real device recordings). The Swift port was rendered over all 800
clips and scored with Microsoft's local AECMOS model (echo and degradation
MOS, 1–5, higher is better), blind ERLE and DNSMOS OVRL, using the
challenge's segment rules. Scripts live in the mobius repo
(`models/enhancement/localvqe/coreml/score_blind.py`).

| Scenario | n | Unprocessed echo | v1.3 echo / deg | v1.3 ERLE | v1.2 echo / deg | v1.2 ERLE |
|---|--:|--:|---|--:|---|--:|
| doubletalk | 115 | 2.17 | 4.35 / 3.93 | – | 4.20 / 3.63 | – |
| doubletalk-with-movement | 185 | 2.21 | 4.35 / 3.86 | – | 4.13 / 3.57 | – |
| farend-singletalk | 107 | 1.95 | 2.49 / 5.00 | 54.1 dB | 3.92 / 5.00 | 45.7 dB |
| farend-singletalk-with-movement | 193 | 2.23 | 3.08 / 5.00 | 55.0 dB | 4.13 / 5.00 | 38.2 dB |
| nearend-singletalk | 200 | 5.00 | 4.99 / 4.14 | – | 4.99 / 4.09 | – |

**Port fidelity.** The upstream GGML engine was run on the same 800 clips
and scored on identical, aligned whole-hop samples: every per-scenario mean
matches the Core ML port to two decimals, the per-clip echo-MOS delta has
mean +0.0002 (95th percentile 0.017), degradation-MOS 95th percentile 0.0004,
and the aligned waveforms agree at a median 84 dB SNR (numerically
equivalent within 16-bit quantisation, not bit-identical). The only
differences found are artefacts of the upstream CLI (256-sample output
delay, zero-filled trailing hop, and a 16-bit writer that wraps samples above
full scale); the Swift CLI writes float32.

**Against the published table.** v1.2 agrees with the upstream single-talk
rows to within about 0.15 MOS and reproduces the far-end ERLE exactly
(45.7 dB), with one exception (with-movement ERLE 38.2 dB vs 40.6 dB
published). The double-talk rows disagree for every model including the
unprocessed baseline (2.17 vs 2.67), and no segment rule tried reproduces
them: an undocumented evaluation-protocol difference, cause not established.
The published v1.3 far-end echo MOS is about 1 point above what the
published v1.3 weights produce under the documented protocol, at higher
ERLE. Treat the table above, not the upstream README, as the reference for
this port.

## Benchmark: near-end recall / far-end leakage

`fluidaudiocli enhance-benchmark` scores the enhancer with the in-repo
Parakeet TDT v3 ASR on the Microsoft AEC-Challenge synthetic set (mic +
loopback + clean near-end triples; 200-example subset at
[FluidInference/aec-challenge-synthetic-mini](https://huggingface.co/datasets/FluidInference/aec-challenge-synthetic-mini),
auto-downloaded). The ASR transcript of the clean near-end clip is the
reference; the loopback transcript gives the far-end words.

- **Recall**: reference words kept by the hypothesis, `1 - (D + S) / N`.
- **WER**: `(S + D + I) / N` against the clean-near-end transcript. Above
  100% on unprocessed audio because the ASR transcribes the echo as well.
- **Leakage**: far-end words that appear in the hypothesis without being
  near-end words, over the far-end word count.

200 examples, signal-to-echo ratio (SER) −10…+10 dB, M5 Pro, 256 ms chunk, CPU:

| Condition | Recall | WER | Leakage | RTFx |
|---|---:|---:|---:|---:|
| Unprocessed mic | 39.5% | 134.2% | 33.8% | – |
| LocalVQE v1.3 | **87.5%** | 43.4% | **1.8%** | 36× |
| LocalVQE v1.2 | 86.3% | 49.4% | 1.9% | 62× |
| v1.3, silent reference (NS only) | 45.3% | 122.6% | 24.4% | 36× |

By SER: at SER ≤ 0 dB (echo louder than speech, 110 files) v1.3 lifts recall
32.0% → 87.0% and cuts leakage 41.5% → 2.3%; at SER > 0 dB (90 files)
49.5% → 88.3% and 25.2% → 1.3%. The silent-reference row shows the model
needs the loopback to cancel echo; without it, it only denoises.

```bash
swift run -c release fluidaudiocli enhance-benchmark                      # both variants, 200 files
swift run -c release fluidaudiocli enhance-benchmark --max-files 50 --variants v1.3 --no-reference --output results.json
```

## Not included

Upstream's `v1.4-AEC` (echo-only, keeps room and noise) and the low-power
GTCRN line are GGUF-only and depend on a C++ adaptive-filter front-end with
no PyTorch reference; they are not converted.
