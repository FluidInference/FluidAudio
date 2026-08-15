# Parakeet Unified 0.6B — LibriSpeech test-clean Benchmark

Measured through the FluidAudio Swift managers (`UnifiedAsrManager` for batch,
`StreamingUnifiedAsrManager` for streaming) over all 2620 `test-clean` files,
scored with the repo's `TextNormalizer` (same normalization as `asr-benchmark`).
Encoder precision: **int8**. Run with `swift run -c release fluidaudiocli unified-benchmark`.

| Mode | Avg WER | Aggregate WER | Median WER | Median RTFx | Overall RTFx | Long files (>15s) |
|------|---------|---------------|------------|-------------|--------------|-------------------|
| batch | 2.16% | 1.68% | 0.00% | 130.0x | 143.6x | 238 |
| streaming | 2.21% | 1.79% | 0.00% | 66.2x | 65.9x | 238 |

- **Avg WER** is the mean of per-file WER (matches `asr-benchmark`'s "Average WER").
- **Aggregate WER** is total errors ÷ total words across the set.
- Long files (> 15 s) are transcribed with overlapping 15 s windows merged on a 2 s overlap (batch),
  or as one continuous session (streaming) — none are skipped. Streaming's overall RTFx drops on
  long files because it re-encodes a 7.68 s window per 1.04 s chunk (the latency tax); batch only
  re-encodes the 2 s overlap, so its throughput stays flat.
- RTFx is end-to-end per file (Swift mel + encode + greedy RNNT decode) on the run machine.
  Mel features are computed natively in Swift (`AudioMelSpectrogram` + NeMo per_feature
  normalization); there is no CoreML preprocessor stage.

## Streaming latency tiers

The streaming encoder bakes a `[left, chunk, right]` chunked-attention window (in 80 ms
encoder frames) at conversion time, so each latency tier is a distinct encoder selected by
`--parakeet-variant`. **Latency = (chunk + right) × 80 ms.** The checkpoint is multi-context
trained, so every tier is the same weights re-exported with a different mask.

| Variant | `[L,C,R]` | Latency | WER | RTFx | Notes |
|---------|-----------|---------|-----|------|-------|
| `parakeet-unified-320ms` | 70,2,2 | 0.32 s | 2.37% | 10x | lowest latency |
| `parakeet-unified-640ms` | 70,7,1 | 0.64 s | 2.40% | 27x | efficiency — same WER as 320ms, big chunk re-encodes less often |
| `parakeet-unified-1120ms` | 70,7,7 | 1.12 s | 2.25% | 33x | best streaming WER |
| `parakeet-unified-2080ms` (default) | 70,13,13 | 2.08 s | 2.47% | 54x | default |

Numbers above are **aggregate WER on a 150-file `test-clean` sweep** (identical files across tiers,
so the *relative* ordering is what matters); the default tier's full-2620 streaming WER is 1.79%
aggregate (table above). Two rules of thumb from the sweep:

- **Look-ahead (right context) drives WER; chunk size drives RTFx.** A big chunk re-encodes less
  often (higher RTFx) but adds latency; more right context improves WER but also adds latency.
- **Look-ahead saturates around right≈2 frames** — beyond that you pay latency without WER gain,
  and `right=0` (no look-ahead) tanks WER. Intermediate tiers like 0.48 s (`70,2,4`) are therefore
  dominated and not shipped.

## Comparison vs Parakeet TDT v3 (same harness)

Parakeet TDT v3 measured via `asr-benchmark --subset test-clean --model-version v3` on the same
machine and `TextNormalizer`: **Average WER 2.6%**, Median 0.0%, Overall RTFx 110.

| Model | Mode | Avg WER | Overall RTFx | Punctuation/caps | Languages |
|-------|------|---------|--------------|------------------|-----------|
| Parakeet TDT v3 | batch (sliding window) | 2.6% | 110 | no | 25 + Japanese |
| Parakeet Unified | batch | 2.16% | 144 | yes | English |
| Parakeet Unified | streaming | 2.21% | 66 | yes | English |

For English file transcription, Unified batch beats TDT v3 on both WER and throughput and adds
punctuation/capitalization. TDT v3 remains the choice for non-English audio.