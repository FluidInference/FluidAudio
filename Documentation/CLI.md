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

**Evaluation Modes:**
- `--mode streaming`: Real-time processing with first-occurrence mapping (production performance)
- `--mode offline`: Full-file processing with Hungarian algorithm (research comparison)

```bash
# Streaming evaluation (default) - production performance
swift run fluidaudio diarization-benchmark --mode streaming --auto-download

# Offline evaluation - comparable to research papers  
swift run fluidaudio diarization-benchmark --mode offline --auto-download

# Compare both modes on same file
swift run fluidaudio diarization-benchmark --single-file ES2004a --mode streaming --output streaming.json
swift run fluidaudio diarization-benchmark --single-file ES2004a --mode offline --output offline.json

# Tune threshold and save results (streaming)
swift run fluidaudio diarization-benchmark --mode streaming --threshold 0.7 --output results.json

# Real-time streaming benchmark (~3s chunks with 2s overlap)
swift run fluidaudio diarization-benchmark --mode streaming --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2

# Balanced streaming throughput/quality (~10s chunks with 5s overlap)
swift run fluidaudio diarization-benchmark --mode streaming --dataset ami-sdm \
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

