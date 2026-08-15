# Encoder Compute Placement (Parakeet TDT v3)

The Parakeet v3 conformer encoder defaults to `.cpuAndNeuralEngine` (ANE). On Apple
Silicon, running it on the GPU (`.cpuAndGPU`) is **~8% faster end-to-end and WER-neutral**.
This is exposed as an opt-in so latency/throughput-oriented workloads can take the win while
ANE remains the default for power efficiency on iOS.

## How to opt in

API — `AsrModels.load` / `loadFromCache` / `loadWithAutoRecovery` / `downloadAndLoad` take an
optional `encoderComputeUnits` (default `nil` → uses the configuration's compute units, ANE):

```swift
let models = try await AsrModels.downloadAndLoad(
    version: .v3,
    encoderComputeUnits: .cpuAndGPU   // opt into the GPU encoder
)
```

CLI — `asr-benchmark` accepts `--encoder-compute-units <ane|gpu|cpu|all>`:

```bash
swift run fluidaudiocli asr-benchmark --model-version v3 --subset test-clean \
    --encoder-compute-units gpu
```

`nil` / omitting the flag preserves today's behavior exactly (ANE).

## Benchmark

LibriSpeech `test-clean`, 100 files, M-series Mac, 6-bit palettized encoder, back-to-back:

| Encoder placement | Median RTFx | Overall RTFx | Avg WER | Avg CER |
| ----------------- | ----------- | ------------ | ------- | ------- |
| ANE (default)     | ~117        | ~131         | 2.6%    | 0.5%    |
| **GPU**           | **~127**    | **~137**     | 2.6%    | 0.5%    |

WER/CER are identical; the speedup is pure compute-placement.

## Why

Isolated encoder latency for one 15s window (`[1, 128, 1501]` mel input):

| Compute units | Encoder latency |
| ------------- | --------------- |
| `.cpuAndGPU`  | 17.8 ms         |
| `.cpuAndNeuralEngine` | 23.5 ms |
| `.cpuOnly`    | 85.4 ms         |

The encoder is the single largest compute component (~23 ms vs ~1 ms preprocessor and
~17 ms for the per-step TDT decode loop), so a ~6 ms encoder saving moves the whole pipeline.

### Why only the encoder

The decoder (LSTM prediction net) and joint network are called many times per window
(~40 and ~60 respectively) and are dispatch-bound at ~0.1 ms and ~0.22 ms per call — GPU
does not help them and would add per-call dispatch overhead, so they stay on ANE. The
preprocessor is pinned to `.cpuOnly` (its STFT ops map to CPU; GPU is ~2× slower there).

## Caveats

- Measured on an M-series **Mac**. The GPU draws more power than the ANE, so for
  battery- or thermal-sensitive **iOS** streaming, ANE is likely still the right default —
  hence this is opt-in rather than a global default change.
- The win is largest for short-utterance workloads where the encoder dominates. For
  continuous long-form audio the relative encoder share is similar, so the ~8% holds in
  practice, but always measure on your target device and audio profile.

## Not worth pursuing: encoder weight quantization

For completeness — lowering the encoder's weight precision does **not** speed it up. The
shipped 6-bit palettized encoder is already optimal:

| Encoder weights | Median RTFx | Avg WER |
| --------------- | ----------- | ------- |
| 6-bit (shipped) | ~120        | 2.6%    |
| INT4            | ~109        | 5.4%    |

INT4 is both slower in the full pipeline (its degraded outputs make the TDT decoder skip
fewer frames → more decode work) and substantially worse on WER. The encoder is
compute-bound, not weight-bandwidth-bound, so fewer weight bits buy no speed. Compute
placement (this doc) is the lever that actually moves; precision is not.
