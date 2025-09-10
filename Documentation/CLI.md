# Command Line Interface (CLI)

This guide collects commonly used `fluidaudio` CLI commands for ASR, diarization, VAD, and datasets.

## ASR

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# Transcribe multiple files in parallel
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudio asr-benchmark --subset test-clean --num-files 50

# Multilingual ASR (FLEURS) benchmark
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10
```

## Diarization

```bash
# Run AMI benchmark (auto-download dataset)
swift run fluidaudio diarization-benchmark --auto-download

# Tune threshold and save results
swift run fluidaudio diarization-benchmark --threshold 0.7 --output results.json

# Quick test on a single AMI file
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.8

# Real-time-ish streaming benchmark (~3s chunks with 2s overlap)
swift run fluidaudio diarization-benchmark --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2

# Balanced throughput/quality (~10s chunks with 5s overlap)
swift run fluidaudio diarization-benchmark --dataset ami-sdm \
  --chunk-seconds 10 --overlap-seconds 5
```

## VAD

```bash
# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3

# Save results and enable debug output
swift run fluidaudio vad-benchmark --all-files --output vad_results.json --debug
```

## Datasets

```bash
# Download test sets
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
swift run fluidaudio download --dataset ami-sdm
swift run fluidaudio download --dataset vad
```

