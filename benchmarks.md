# Parakeet TDT-CTC-110M Benchmark Results

## LibriSpeech test-clean (Full Dataset)

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| **Average WER** | **3.01%** |
| **Median WER** | **0.0%** |
| Average CER | 1.09% |
| Audio duration | 19,452.5s (~5.4 hours) |
| Processing time | 201.5s (~3.4 minutes) |
| **Overall RTFx** | **96.5x** |
| **Median RTFx** | **86.4x** |

## Configuration

- Model: Parakeet TDT-CTC-110M (CoreML)
- Architecture: Hybrid TDT-CTC with fused preprocessor+encoder
- Platform: Apple Silicon (M2)
- Date: March 26, 2026

## Key Features

- **96.5x real-time factor** - 1 hour of audio transcribes in 37 seconds
- **3.01% WER** - Competitive accuracy on LibriSpeech test-clean
- **0% median WER** - Most files transcribed perfectly
- **iOS compatible** - Runs on iPhone with full CoreML optimization
- **Stateless processing** - No encoder state carryover needed

## Running the Benchmark

```bash
# Build release
swift build -c release

# Run full benchmark (auto-downloads dataset and models)
.build/release/fluidaudiocli asr-benchmark --subset test-clean --model-version tdt-ctc-110m

# Run with limited files
.build/release/fluidaudiocli asr-benchmark --subset test-clean --model-version tdt-ctc-110m --max-files 100

# Process single file
.build/release/fluidaudiocli asr-benchmark --single-file 1089-134686-0000 --model-version tdt-ctc-110m
```

## Notes

- TDT (Token-and-Duration Transducer) decoder with CTC-constrained beam search
- Fused preprocessor+encoder reduces model load time and memory usage
- Models available at: [FluidInference/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-tdt-ctc-110m-coreml)
- iOS test app validates on-device performance with LibriSpeech ground truth

---

# Nemotron Speech Streaming 0.6B Benchmark Results

## LibriSpeech test-clean (Full Dataset)

| Metric | Value |
|--------|-------|
| Files processed | 2,620 |
| Total words | 53,120 |
| Total errors | 1,334 |
| **WER** | **2.51%** |
| Audio duration | 19,452.5s (~5.4 hours) |
| Processing time | 3,393.7s (~56.6 minutes) |
| **RTFx** | **5.7x** |
| Peak memory | 1.452 GB |

## Configuration

- Model: Nemotron Speech Streaming 0.6B (CoreML)
- Encoder variant: int8
- Platform: Apple Silicon (M4 Pro)
- Date: January 15, 2026

## Running the Benchmark

```bash
# Build release
swift build -c release

# Run full benchmark (auto-downloads dataset and models)
.build/release/fluidaudiocli nemotron-benchmark --subset test-clean

# Run with limited files
.build/release/fluidaudiocli nemotron-benchmark --subset test-clean --max-files 100

# Use float32 encoder variant
.build/release/fluidaudiocli nemotron-benchmark --encoder float32 --max-files 50
```

## Notes

- True streaming with 1.12s audio chunks and encoder state carryover
- RNNT greedy decoding with proper decoder LSTM state management
- Models available at: [alexwengg/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/alexwengg/nemotron-speech-streaming-en-0.6b-coreml)

---

# Parakeet TDT 0.6B v3 Benchmark Results

## LibriSpeech test-clean (2,620 files, 19,452.5 s audio)

| Encoder           | On-disk | Avg WER | Avg CER | Overall RTFx | Peak RAM |
|-------------------|--------:|--------:|--------:|-------------:|---------:|
| 8-bit palettized  | 425 MB  | 2.64%   | 1.03%   | 47.1×        | 153 MB   |
| int4 linear/ch    | 285 MB  | 3.76%   | 1.59%   | 43.1×        | 139 MB   |

Apple M2, `.cpuAndNeuralEngine`. Decoder/joint/preprocessor fp16 in both. Models: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml).

---

# Parakeet Unified 0.6B Benchmark Results

Unified FastConformer-RNNT — one checkpoint serves both offline batch and
chunked-attention streaming. English, with punctuation and capitalization.

## LibriSpeech test-clean (Full Dataset, 2,620 files, ~5.4 h audio)

| Mode      | Avg WER | Aggregate WER | Median WER | Overall RTFx | Median RTFx | Long files (>15s) |
|-----------|--------:|--------------:|-----------:|-------------:|------------:|------------------:|
| Batch     | 2.15%   | 1.68%         | 0.00%      | 123.3×       | 111.5×      | 238               |
| Streaming | 2.21%   | 1.79%         | 0.00%      | 29.1×        | 53.1×       | 238               |

- **Avg WER** is the mean of per-file WER (matches `asr-benchmark`'s "Average WER"); **Aggregate WER** is total errors ÷ total words.
- Long files (> 15 s) are not skipped: batch uses overlapping 15 s windows merged on a 2 s overlap; streaming runs them as one continuous session.
- Streaming's overall RTFx falls below its median because it re-encodes a 7.68 s window per 1.04 s chunk (the latency tax); long files amortize that poorly. Batch only re-encodes the 2 s overlap, so throughput stays flat.

## Configuration

- Model: Parakeet Unified 0.6B (CoreML)
- Architecture: Unified FastConformer-RNNT (`chunked_limited_with_rc` attention); greedy RNNT, no TDT duration head
- Encoder variant: int8 (per-channel linear symmetric) — WER-lossless vs fp16, half the size
- Platform: Apple Silicon (M5 Pro), `.cpuAndNeuralEngine`
- Date: June 12, 2026

## Running the Benchmark

```bash
# Build release
swift build -c release

# Run full benchmark (batch + streaming; auto-downloads dataset and models)
.build/release/fluidaudiocli unified-benchmark --mode both

# Single mode, limited files, or fp16 encoder
.build/release/fluidaudiocli unified-benchmark --mode streaming --max-files 100
.build/release/fluidaudiocli unified-benchmark --mode batch --precision fp16
```

## Notes

- Same harness and `TextNormalizer` as `asr-benchmark`, so directly comparable: Parakeet TDT v3 = 2.6% Avg WER / 110× RTFx (multilingual, no punctuation). For English files, Unified batch wins on WER, throughput, and punctuation; TDT v3 remains the multilingual option.
- Batch and streaming share the same greedy RNNT decoder; they differ only in the encoder window (offline 15 s full-attention vs streaming 7.68 s chunked).
- Models available at: [FluidInference/parakeet-unified-en-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-unified-en-0.6b-coreml)
