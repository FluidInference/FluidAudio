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

# Parakeet TDT 0.6B v3 — Encoder Quantization Sweep

## LibriSpeech test-clean (Full Dataset, 2,620 files)

Clean-ANE re-run (no other workloads touching the Neural Engine, ~60 s cooldown between runs):

| Encoder variant | Encoder file | On-disk | Avg WER | Median WER | Avg CER | Total proc. time | Overall RTFx | Median RTFx |
|-----------------|--------------|--------:|--------:|-----------:|--------:|-----------------:|-------------:|------------:|
| **fp16** | `Encoder.mlmodelc` | 425 MB | **2.64%** | 0.0% | 1.03% | 413.4 s (~6.9 min) | **47.1×** | 41.0× |
| **int4 (linear/channel)** | `EncoderInt4.mlmodelc` | 285 MB | **3.76%** | 0.0% | 1.59% | 451.0 s (~7.5 min) | **43.1×** | 39.6× |

Audio duration for both runs: 19,452.5 s (~5.4 h). Decoder / joint / preprocessor are fp16 in both runs.

### Cross-stack comparison: mweinbach/parakeet-coreml-swift on the same M2

Same hardware, same dataset, same `.cpuAndNeuralEngine` compute units, but a different Parakeet TDT v3 Swift implementation (`mweinbach/parakeet-coreml-swift` @ HEAD, model `mweinbach1/parakeet-tdt-0.6b-v3-coreml`, 4-bit *palettized* encoder + fp16 decoder/joint).

| Stack | Encoder | Avg WER | Median WER | Avg CER | Overall RTFx | Median RTFx | Total proc. time |
|-------|---------|--------:|-----------:|--------:|-------------:|------------:|-----------------:|
| **OURS** (fp16) | int4-channel ✗, fp16 | **2.64%** | 0.0% | 1.03% | **47.1×** | 41.0× | 413.4 s |
| **OURS** (int4) | int4 linear-per-channel | **3.76%** | 0.0% | 1.59% | **43.1×** | 39.6× | 451.0 s |
| **THEIRS** (mweinbach) | 4-bit palettized | **12.77%** | 0.0% | 9.35% | **24.7×** | 21.7× | 787.5 s |

**Honest disclosure on the THEIRS row:** their median WER is 0% — most utterances transcribe cleanly. The 12.77% average is dragged up by ~100 catastrophic decoder-spillover cases (3.8% of the set), where their TDT greedy loop keeps emitting tokens past the actual audio (English hallucinations like "you know, you know, you know, …" or Cyrillic / random unicode spillover). The worst case is `7127-75947-0004` (1.9 s clip "EXPLAIN YOURSELF" → 700+ token hypothesis). The remaining 96.2% of files have WER ≤ 50% and the 66.9% bulk have WER ≤ 5%.

WER bucket distribution for theirs:

| WER bucket | Files | Share |
|---|---:|---:|
| ≤ 5% | 1,753 | 66.9% |
| 5–50% | 767 | 29.3% |
| > 50% | 100 | 3.8% |

## Configuration

- Model: Parakeet TDT 0.6B v3 (CoreML)
- Architecture: TDT (Token-and-Duration Transducer), sliding-window encoder (15 s window, 188 enc frames)
- Platform: Apple Silicon (M2, 16 GB)
- Date: April 30, 2026
- Runs (release build, no `--max-files`):
  - fp16: main branch (`feat/tts-benchmark` @ `8f9e42fd9`) — `fluidaudiocli asr-benchmark --subset test-clean --model-version v3`
  - int4: PR #560 branch (`feat/parakeet-encoder-int4-default` @ `22845e10f`, int4 is the v3 default) — same command

## Notes

- **No int8 v3 encoder is published.** `FluidInference/parakeet-tdt-0.6b-v3-coreml` only ships fp16 (`Encoder.mlmodelc`) and int4 (`EncoderInt4.mlmodelc`). The `supportsInt8Encoder` selector in `ParakeetLanguageModels` is wired for CTC Zh-Cn (`Encoder-v2-int8`) and Qwen3, not Parakeet TDT v3.
- After PR #560 int4 becomes the v3 default and `requiredModelsV3` swaps `encoderFile` → `encoderInt4File`. Existing v3 callers will start downloading the int4 bundle on first use.
- **Quality**: int4 regresses by **+1.13 pp WER** vs fp16 on full test-clean (3.76 vs 2.64). Smaller than the +2.6 pp gap reported on the 100-file slice in `wer_variants/staging/.../UPLOAD_NOTES.md`, so the upload-notes table is pessimistic.
- **Speed**: on M2 with a clean ANE, fp16 is **slightly faster** than int4 end-to-end (47.1× vs 43.1× overall RTFx). The int4 advantage demonstrated on other Apple Silicon generations does not show up here — the encoder isn't the dominant cost in this pipeline (decoder/joint and FLAC I/O contribute meaningfully), and the int4 dequant path on M2 ANE doesn't beat the fp16 path. **The win for int4 is purely on-disk size (~33% smaller, 425 MB → 285 MB)**, not RTFx.
- A previous run on this machine showed fp16 at 28.0× and int4 at 21.1× — that session had a competing ANE workload (another model held the Neural Engine), which throttled both. Numbers above are the corrected clean-ANE measurements.
- Median RTFx ≈ overall RTFx for both variants → per-utterance latency is consistent, no long-tail outliers.
- Models: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
