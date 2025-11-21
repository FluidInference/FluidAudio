# Command Line Interface (CLI)

This guide collects commonly used `fluidaudio` CLI commands for ASR, diarization, VAD, and datasets.

## TTS

TTS is built into the CLI. Run it directly:

```bash
swift run fluidaudio tts "Hello from FluidAudio" --output out.wav
```

## ASR

```bash
# Transcribe an audio file (batch)
swift run fluidaudio transcribe audio.wav

# English-only run with higher accuracy
swift run fluidaudio transcribe audio.wav --model-version v2

# Transcribe multiple files in parallel
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudio asr-benchmark --subset test-clean --max-files 50

# English benchmark preset (LibriSpeech)
swift run fluidaudio asr-benchmark --subset test-clean --max-files 50 --model-version v2

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

# Run the full VBx offline pipeline
swift run fluidaudio diarization-benchmark --mode offline --dataset ami-sdm --threshold 0.6

# Process a single file with streaming vs. offline inference
swift run fluidaudio process meeting.wav --mode streaming --threshold 0.7
swift run fluidaudio process meeting.wav --mode offline --threshold 0.6 --debug
```

- `--mode offline` switches the CLI to `OfflineDiarizerManager`, running the full VBx pipeline with PLDA refinement. Expect DER ≈ 18–20 % on AMI-SDM with threshold 0.6.
- Add `--rttm /path/to/ground_truth.rttm` to `process` to compute DER/JER in-place, or `--export-embeddings embeddings.json` for debugging speaker vectors.
- GitHub Actions workflow `offline-pipeline.yml` replays `fluidaudio diarization-benchmark --mode offline --single-file ES2004a` on every PR so failures in model downloads or clustering logic are caught early.

## VAD

```bash
# Offline segmentation with seconds output (default mode)
swift run fluidaudio vad-analyze path/to/audio.wav

# Streaming only with 128 ms chunks and a custom threshold (timestamps emitted in seconds)
swift run fluidaudio vad-analyze path/to/audio.wav --streaming --threshold 0.65 --min-silence-ms 400

# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3

# Save benchmark results and enable debug output
swift run fluidaudio vad-benchmark --all-files --output vad_results.json --debug
```

`swift run fluidaudio vad-analyze --help` lists every tuning option (padding,
negative threshold overrides, max-duration splitting, etc.).

## Datasets

```bash
# Download test sets
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
swift run fluidaudio download --dataset ami-sdm
swift run fluidaudio download --dataset vad
```
