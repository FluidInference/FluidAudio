# Parakeet Unified 0.6B — LibriSpeech test-clean Benchmark

Measured through the FluidAudio Swift managers (`UnifiedAsrManager` for batch,
`StreamingUnifiedAsrManager` for streaming) over all 2620 `test-clean` files,
scored with the repo's `TextNormalizer` (same normalization as `asr-benchmark`).
Encoder precision: **int8**. Run with `swift run -c release fluidaudiocli unified-benchmark`.

| Mode | Avg WER | Aggregate WER | Median WER | Median RTFx | Overall RTFx | Long files (>15s) |
|------|---------|---------------|------------|-------------|--------------|-------------------|
| batch | 2.15% | 1.68% | 0.00% | 111.5x | 123.3x | 238 |
| streaming | 2.21% | 1.79% | 0.00% | 53.1x | 29.1x | 238 |

- **Avg WER** is the mean of per-file WER (matches `asr-benchmark`'s "Average WER").
- **Aggregate WER** is total errors ÷ total words across the set.
- Long files (> 15 s) are transcribed with overlapping 15 s windows merged on a 2 s overlap (batch),
  or as one continuous session (streaming) — none are skipped. Streaming's overall RTFx drops on
  long files because it re-encodes a 7.68 s window per 1.04 s chunk (the latency tax); batch only
  re-encodes the 2 s overlap, so its throughput stays flat.
- RTFx is end-to-end per file (preprocess + encode + greedy RNNT decode) on the run machine.

## Comparison vs Parakeet TDT v3 (same harness)

Parakeet TDT v3 measured via `asr-benchmark --subset test-clean --model-version v3` on the same
machine and `TextNormalizer`: **Average WER 2.6%**, Median 0.0%, Overall RTFx 110.

| Model | Mode | Avg WER | Overall RTFx | Punctuation/caps | Languages |
|-------|------|---------|--------------|------------------|-----------|
| Parakeet TDT v3 | batch (sliding window) | 2.6% | 110 | no | 25 + Japanese |
| Parakeet Unified | batch | 2.15% | 123 | yes | English |
| Parakeet Unified | streaming | 2.21% | 29 | yes | English |

For English file transcription, Unified batch beats TDT v3 on both WER and throughput and adds
punctuation/capitalization. TDT v3 remains the choice for non-English audio.
