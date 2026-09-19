# LocalVQE streaming validation

The stream wrapper serializes each complete `enhance`, `flush`, and `reset`
operation, including across asynchronous prediction. Previously another
operation could enter during inference and overwrite shared input buffers,
consume the same pending audio, or reset state before the first call resumed.
Independent streams still run concurrently on a shared manager.

Queued cancellation leaves the active clip untouched. Active cancellation or
inference failure clears the unfinished clip, recurrent state and sample
counters before reuse. Reset waits for earlier operations. Callers should
submit capture buffers through one consumer task, in capture order, and await
each push to bound queued audio. See [usage and recovery](LocalVQE.md#streaming).

## Verification

Validation performed on Apple Silicon with cached real v1.3 Core ML models:

- Debug library/CLI compilation and `swift build -c release` passed.
- Strict swift-format lint and `git diff --check` passed.
- A standalone harness linked to the compiled FluidAudio module passed 13
  audio comparisons using the repository's real
  `01-validation-request-21.4s.wav` fixture. It exercised the same concurrency
  and recovery scenarios as the added regression tests; it did not replace
  the model with a mock.

| Check | Result |
|---|---|
| Overlapping pushes followed by flush | Exact match to sequential processing |
| Reset queued between pushes | Next clip exactly matches a fresh stream |
| Cancelled queued push, flush or reset | Three comparisons; active clip unchanged |
| Active push cancellation, then reuse | `CancellationError`; next clip exactly matches a fresh stream |
| Two independent concurrent streams | Each exactly matches its sequential result |
| Streaming buffers of 100, 256, 1000 and 4096 samples | Four exact matches to whole-clip output |
| 16 ms versus 256 ms model chunk | Maximum absolute difference 8.20e-8 |

Three predetermined real AEC-Challenge mic/reference pairs also passed a
release CLI smoke check, using v1.3 CPU inference, 16 ms model chunks and
256-sample input buffers. Output was finite, retained the microphone input
length, and was bit-identical between streaming and whole-clip processing.

| Recording stem | Duration | Samples |
|---|---:|---:|
| `t2U2oyODeEuQhnAt3oCksQ_doubletalk` | 36.58 s | 585280 |
| `Du0RI678G0yhsNVU5AGKTw_farend-singletalk-with-movement` | 23.90 s | 382400 |
| `f2HsvN51L0ygRLcFvf3udg_nearend-singletalk` | 16.53 s | 264480 |

This smoke check did not rerun the full 800-clip quality benchmark or tune
the model/scorer. The existing CLI reported p50 1.10 ms, p99 1.18 ms, and
maximum 1.98–2.22 ms on these recordings. These are offline timing observations:
the CLI collects timing only for pushes emitting samples, excluding the
initial dropped-hop call and flush. They do not cover startup or certify
live-call deadlines under device contention.

## XCTest and CI

The local `swift test --filter LocalVqe` attempt could not execute: this Mac
has Command Line Tools 6.2.3 without Xcode's XCTest framework (`no such module
'XCTest'`). The standalone checks above are reported separately from XCTest.

On an Xcode-equipped machine, run the regression suite with real models:

```bash
FLUIDAUDIO_LOCALVQE_MODEL_DIR=/path/to/compiled/models swift test --filter LocalVqe
```

The directory must contain the v1.3 16 ms and 256 ms bundles. Supplying it
enables model tests in CI and makes a missing bundle a failure. Without an
explicit directory, model tests retain the existing CI/missing-cache skip.

## Production boundary

The stream-wrapper concurrency issue is fixed and its recovery behavior is
verified locally. Live microphone/playback clock alignment, route changes,
audio-callback scheduling, and sustained performance under contention or
thermal load still require the application's capture/playback pipeline on
its target devices. LocalVQE remains beta pending that integration validation.
