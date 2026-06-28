# Nemotron Speech Streaming 0.6B — LibriSpeech test-clean Benchmark

Measured through `StreamingNemotronAsrManager` over the **full** LibriSpeech
`test-clean` split (2620 files, 53,120 reference words, ~5.4 h audio), scored
with the benchmark's word-level edit distance. Encoder precision: **int8**.
Mel features are computed **natively in Swift** (`NemotronMelExtractor` →
`AudioMelSpectrogram`, NeMo `normalize: NA` raw log-mel); there is no CoreML
preprocessor stage (removed in the issue #739 fix).

Reproduce per tier:

```bash
swift run -c release fluidaudiocli nemotron-benchmark --subset test-clean --chunk <560|1120|2240>
```

| Chunk tier | WER | RTFx | Errors / words | Processing time |
|------------|-----|------|----------------|-----------------|
| 560 ms (lowest latency) | 2.71% | 40.7x | 1442 / 53120 | 478.4 s |
| 1120 ms (trained chunk) | **2.58%** | 24.3x | 1369 / 53120 | 801.6 s |
| 2240 ms (default) | 2.64% | 87.4x | 1403 / 53120 | 222.5 s |

- **WER** is aggregate (total errors ÷ total words across all 2620 files), matching
  the value the benchmark reports and writes to `/tmp/nemotron_<tier>ms_benchmark.json`.
- **RTFx** is end-to-end single-stream per the whole set (Swift mel + int8 ANE encode +
  greedy RNN-T decode), release build, on the dev machine (Apple Silicon). Absolute RTFx is
  machine- and load-dependent; the *relative* ordering is stable across runs.
- The 1120 ms tier has the lowest WER but also the lowest RTFx: it runs more encoder calls
  than 2240 ms without the larger chunk's batching headroom. 2240 ms is the default — it is
  the throughput sweet spot and within ~0.06 pp WER of the best tier.
- All tiers share one encoder; the tier only changes the streaming chunk geometry
  (`chunk_mel_frames`). Accuracy is essentially flat across tiers (2.58–2.71%).

## Native-Swift mel front-end (issue #739)

The CoreML `preprocessor` model was removed from the Nemotron streaming path. Its flexible
`RangeDim` audio input made CoreML build the ANE `default_function` against a 1-sample lower
bound, producing `ios17.slice_by_index: zero shape error` / "Skipped adding default_function
to entry point: main" — the warning behind the iPadOS cold-start empty-transcript failure.
`NemotronMelExtractor` reproduces NeMo's `AudioToMelSpectrogramPreprocessor` exactly
(`n_fft=512`, `win=400`, `hop=160`, 128 mels, symmetric Hann, `preemph=0.97`, `normalize: NA`).
Python parity vs NeMo PyTorch: max |Δ| ≈ 9e-3 (float tolerance); the 2.58–2.71% WER above
confirms end-to-end correctness (a wrong mel front-end would collapse WER).
