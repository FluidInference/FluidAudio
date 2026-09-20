# LocalVQE — Echo Cancellation & Noise Suppression

`LocalVqeManager` runs [LocalVQE](https://github.com/localai-org/LocalVQE)
(Apache-2.0), a compact neural model for acoustic echo cancellation (AEC),
noise suppression and dereverberation of 16 kHz speech. It is a streaming,
CPU-tuned derivative of DeepVQE (Indenbom et al., Interspeech 2023). Typical
use: cleaning up call audio captured without headphones, where the mic picks
up what the loudspeaker plays.

**Beta.** Port fidelity validated (numerically equivalent to the upstream
PyTorch and GGML engines, scored identically to GGML on the 800-clip
AEC-Challenge blind set); the published benchmark is only partially
reproduced, and its ERLE protocol is ambiguous in the public artifacts (see
[Quality](#quality-aec-challenge-blind-test-set)). Not yet exercised inside
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
Within each stream, `enhance`, `flush`, and `reset` execute in the order they
reach the actor, including across inference suspension points. Feed a stream
from one consumer task and await each push before submitting the next: tasks
launched independently from audio callbacks can arrive out of capture order
and accumulate queued audio. Run enhancement outside the audio render callback.

`enhance` returns samples as whole model calls complete. Output sample `i`
corresponds to input sample `i`, delivered one hop (256 samples, 16 ms)
after the input that produced it plus whatever is still buffered toward the
next call. `flush()` resets the stream; call `reset()` to start a new clip
without flushing.

Cancelling an operation while it is queued removes it without changing the
current clip. Cancelling a running push/flush, or an inference error, discards
the unfinished clip and clears recurrent state before the next operation.
Treat this as an audio discontinuity. `reset()` waits for earlier operations;
to abandon an active push, cancel its task, await its completion, then resume
with the next clip. A cancelled queued reset does not reset the stream.

For live capture, supply continuous 16 kHz mono mic/reference buffers covering
the same time intervals. The reference must be the actual far-end playback
signal. Equal buffer lengths alone do not establish correct timing. Reset on
capture/playback discontinuities, and validate reference timing, route changes,
sustained latency and recovery in the application on its target devices before
enabling enhancement by default. The offline benchmark does not exercise that
live integration.

Real-model streaming regression tests can be enabled locally or in CI with:

```bash
FLUIDAUDIO_LOCALVQE_MODEL_DIR=/path/to/compiled/models swift test --filter LocalVqe
```

The explicit directory must contain both v1.3 chunk variants; a missing bundle
fails the tests. Without that setting, model tests skip in CI or when the local
cache is absent. The suite covers overlapping pushes, flush/reset ordering,
queued and active cancellation, fresh-clip recovery, and independent streams.
See the [streaming validation report](LocalVQEValidation.md) for the completed
real-model checks, the local XCTest environment limitation and remaining
live-integration coverage.

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

**Summary: port fidelity validated; the published benchmark is only partially
reproduced, with unresolved v1.2 far-end differences and ERLE protocol.**

The upstream quality table is AECMOS on the ICASSP 2022 AEC-Challenge blind
set (800 real device recordings). The Swift port was rendered over all 800
clips and scored two ways, kept separate because they answer different
questions. Scripts and per-clip results live in the mobius repo
(`models/enhancement/localvqe/coreml/score_blind.py`).

**Challenge protocol (reference).** Microsoft's current scenario-aware AECMOS
model with the challenge's segment rules (convergence portions excluded),
DNSMOS OVRL on the same rated segment, and blind ERLE with the LocalVQE
technical-report gating (far-end-dominated frames only). AECMOS is 1–5,
higher is better.

| Scenario | n | Unprocessed echo | v1.3 echo / deg / ERLE / OVRL | v1.2 echo / deg / ERLE / OVRL |
|---|--:|--:|---|---|
| doubletalk | 115 | 2.17 | 4.35 / 3.93 / 6.3 dB / 2.89 | 4.20 / 3.63 / 6.2 dB / 2.77 |
| doubletalk-with-movement | 185 | 2.21 | 4.35 / 3.86 / 6.1 dB / 2.84 | 4.13 / 3.57 / 6.0 dB / 2.73 |
| farend-singletalk | 107 | 1.95 | 2.49 / 5.00 / 54.2 dB / 1.95 | 3.92 / 5.00 / 53.2 dB / 1.89 |
| farend-singletalk-with-movement | 193 | 2.23 | 3.08 / 5.00 / 55.9 dB / 1.96 | 4.13 / 5.00 / 47.3 dB / 1.80 |
| nearend-singletalk | 200 | 5.00 | 4.99 / 4.14 / 2.3 dB / 3.17 | 4.99 / 4.09 / 2.1 dB / 3.17 |

**Upstream protocol (HF model-card reproduction).** The published table was
compared using the legacy AECMOS model over the first 20 s of each clip; that
protocol reproduces the card's unprocessed baseline exactly
(2.67 / 2.56 / 1.90 / 2.13 / 5.00). The card defines ERLE as a plain
whole-signal energy ratio, but its numbers resemble a separately reconstructed
gated metric. `gERLE*` below is that reconstruction, not a confirmed
interpretation of the card's protocol. Under it, the Core ML port gives:

| Scenario | HF card v1.3 | Core ML v1.3 (echo / deg / gERLE* / OVRL) | HF card v1.2 | Core ML v1.2 (echo / deg / gERLE* / OVRL) |
|---|---|---|---|---|
| doubletalk | 4.73 / 2.62 | 4.73 / 2.62 | 4.72 / 2.37 | 4.72 / 2.39 |
| doubletalk-with-movement | 4.67 / 2.43 | 4.66 / 2.44 | 4.65 / 2.30 | 4.64 / 2.31 |
| farend-singletalk | 3.69 / 4.83 | 3.54 / 4.82 | 3.78 / 4.91 | 4.07 / 4.93 |
| farend-singletalk-with-movement | 3.88 / 4.98 | 3.75 / 4.96 | 4.12 / 4.96 | 4.27 / 4.96 |
| nearend-singletalk | 5.00 / 4.18 | 5.00 / 4.18 | 5.00 / 4.16 | 5.00 / 4.17 |

The ERLE definition conflict is material: current GGML gives plain / gated
far-end ERLE of 43.0 / 50.9 and 40.1 / 49.4 dB for v1.3, versus card values
50.9 / 49.9; v1.2 gives 38.7 / 48.0 and 30.5 / 41.1 dB, versus 45.7 / 40.6.
Selecting the closer gated result does not prove that upstream used this gate.

Unprocessed baseline: exact. v1.3: double-talk and near-end within 0.01
echo MOS; far-end 0.15 low from aligned float output and within 0.04 when
the upstream CLI's raw 16-bit, one-hop-late output is scored instead; gated
gated ERLE within 0.8 dB and OVRL within 0.01 (full columns in the mobius
README). v1.2: double-talk and near-end within 0.02 echo, 0.02 deg, 0.1 dB
gated ERLE and 0.06 OVRL; the far-end rows are not reproduced on any metric (echo
+0.29 / +0.15, gated ERLE +1.9 / +0.7 dB, OVRL +0.09 / +0.05, from either
runtime). Those values are above the published ones, which is not evidence
that the port outperforms upstream; +0.29 echo MOS is not rounding noise.
Rendering v1.2 at the pre-v1.2 delay window (dmax 32, which the reference
config left on the day that row was published) brings far-end ERLE
(44.9 / 40.5 vs 45.7 / 40.6 dB) and degradation (4.88 / 4.96 vs 4.91 / 4.96)
close to the card, but does not establish which configuration upstream used.
Its echo MOS moves farther from the card;
softmax temperature 1.0, the ReLU6 reference, the upstream CLI's output
format and every scorer/segment variation were also tested and rejected
(details in the mobius README). None of the tested configurations reproduces
the whole table. Upstream's evaluation configuration, rendered audio or
scoring script would help resolve the remaining echo cells. An earlier revision of
this page said the v1.3 far-end row could not have come from the published
weights; that was a protocol mismatch and is retracted.

The [follow-up investigation in mobius](https://github.com/FluidInference/mobius/blob/a066485e4c65790fa21b180b7ddf5e22e0f2d044/models/enhancement/localvqe/coreml/REPRODUCTION.md)
also compares the published PT/GGUF tensors and isolates a historical
upstream state-copy defect. Testing the original engine on all 300 far-end
recordings still did not reproduce the card's echo MOS. Its per-recording
results and precision diagnostics are retained separately from the main
800-clip benchmark.

**Port fidelity.** The upstream GGML engine was run on the same 800 clips
and scored on identical, aligned whole-hop samples: every per-scenario mean
matches the Core ML port to two decimals, the per-clip echo-MOS delta has
mean +0.0002 (95th percentile 0.017), degradation-MOS 95th percentile 0.0004,
and the aligned waveforms agree at a median 84 dB SNR (numerically
equivalent within 16-bit quantisation, not bit-identical). The only
differences found are artefacts of the upstream CLI (256-sample output
delay, zero-filled trailing hop, and a 16-bit writer that wraps samples above
full scale); the Swift CLI writes float32.

## Benchmark: near-end recall / far-end leakage

**Exploratory.** `fluidaudiocli enhance-benchmark` scores the enhancer with
the in-repo Parakeet TDT v3 ASR on a 200-example subset (the first 200 of
shard 0) of the Microsoft AEC-Challenge synthetic *training* set (mic +
loopback + clean near-end triples;
[FluidInference/aec-challenge-synthetic-mini](https://huggingface.co/datasets/FluidInference/aec-challenge-synthetic-mini),
revision `1f3714b5a3f98cedef1bbb017f21bbd7ae688596`, archive SHA256
`45ff5d7acfce499558c25a0eace45eb819cec8aa76420fe733de7ee116ae548d`).
The default download and cached metadata are verified before use; malformed
rows, duplicate IDs and missing audio now fail instead of silently shrinking
the benchmark. The ASR transcript of the clean near-end clip is the
reference and the loopback transcript gives the far-end words, so the
metrics are relative to machine transcripts, not human ones; examples whose
clean-near-end transcript is empty (33 of 200) are excluded and reported.
Use it to compare conditions, not as an absolute quality figure; the
AEC-Challenge blind-set table above is the quality reference.

- **Recall**: reference words kept by the hypothesis, `1 - (D + S) / N`.
- **WER**: `(S + D + I) / N` against the clean-near-end transcript. Above
  100% on unprocessed audio because the ASR transcribes the echo as well.
- **Leakage**: far-end words that appear in the hypothesis without being
  near-end words, over the far-end word count.

167 scored examples, signal-to-echo ratio (SER) −10…+10 dB, M5 Pro, 256 ms
chunk, CPU:

| Condition | Recall | WER | Leakage | RTFx |
|---|---:|---:|---:|---:|
| Unprocessed mic | 44.1% | 112.2% | 34.0% | – |
| LocalVQE v1.3 | **77.6%** | 29.6% | **1.1%** | 34× |
| LocalVQE v1.2 | 73.0% | 33.9% | 1.0% | 57× |
| v1.3, silent reference (NS only) | 42.3% | 101.8% | 23.9% | 34× |

By SER: at SER ≤ 0 dB (echo louder than speech, 93 files) v1.3 lifts recall
36.4% → 74.5% and cuts leakage 41.1% → 1.6%; at SER > 0 dB (74 files)
54.4% → 81.7% and 25.9% → 0.5%. The silent-reference row shows the model
needs the loopback to cancel echo; without it, it only denoises.

An earlier revision of this table reported 87.5% / 86.3% recall: the shared
WER scorer had its insertion/deletion labels swapped, so an empty hypothesis
scored 100% recall. Fixed in `WERCalculator` (WER itself was unaffected).
When `--output` is used, the JSON includes the dataset revision and hashes,
numeric-file-ID selection order, exact selected/excluded IDs, both machine
reference transcripts and the raw word-count numerators and denominators.

```bash
swift run -c release fluidaudiocli enhance-benchmark                      # both variants, 200 files
swift run -c release fluidaudiocli enhance-benchmark --max-files 50 --variants v1.3 --no-reference --output results.json
```

## Not included

Upstream's `v1.4-AEC` (echo-only, keeps room and noise) and the low-power
GTCRN line are GGUF-only and depend on a C++ adaptive-filter front-end with
no PyTorch reference; they are not converted.
