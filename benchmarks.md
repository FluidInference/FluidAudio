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
