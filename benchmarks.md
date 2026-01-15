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
