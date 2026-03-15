# Command Line Interface (CLI)

This guide collects commonly used `fluidaudio` CLI commands for ASR, diarization, VAD, and datasets.

## TTS

TTS is built into the CLI. Run it directly:

```bash
swift run fluidaudiocli tts "Hello from FluidAudio" --output out.wav

# Multilingual G2P benchmark
swift run fluidaudiocli g2p-benchmark
```

## ASR

```bash
# Transcribe an audio file (batch)
swift run fluidaudiocli transcribe audio.wav

# English-only run with higher accuracy
swift run fluidaudiocli transcribe audio.wav --model-version v2

# Transcribe with Qwen3 ASR
swift run fluidaudiocli qwen3-transcribe audio.wav

# Streaming ASR with Parakeet EOU
swift run fluidaudiocli parakeet-eou --input audio.wav

# Transcribe multiple files in parallel
swift run fluidaudiocli multi-stream audio1.wav audio2.wav

# Benchmark ASR on LibriSpeech
swift run fluidaudiocli asr-benchmark --subset test-clean --max-files 50

# English benchmark preset (LibriSpeech)
swift run fluidaudiocli asr-benchmark --subset test-clean --max-files 50 --model-version v2

# Multilingual ASR (FLEURS) benchmark
swift run fluidaudiocli fleurs-benchmark --languages en_us,fr_fr --samples 10

# Qwen3 ASR benchmark
swift run fluidaudiocli qwen3-benchmark

# CTC keyword spotting benchmark on Earnings22
swift run fluidaudiocli ctc-earnings-benchmark
```

## Diarization

```bash
# Run AMI benchmark (auto-download dataset)
swift run fluidaudiocli diarization-benchmark --auto-download

# Tune threshold and save results
swift run fluidaudiocli diarization-benchmark --threshold 0.7 --output results.json

# Quick test on a single AMI file
swift run fluidaudiocli diarization-benchmark --single-file ES2004a --threshold 0.8

# Real-time-ish streaming benchmark (~3s chunks with 2s overlap)
swift run fluidaudiocli diarization-benchmark --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2

# Balanced throughput/quality (~10s chunks with 5s overlap)
swift run fluidaudiocli diarization-benchmark --dataset ami-sdm \
  --chunk-seconds 10 --overlap-seconds 5

# Run the full VBx offline pipeline
swift run fluidaudiocli diarization-benchmark --mode offline --dataset ami-sdm --threshold 0.6

# Process a single file with streaming vs. offline inference
swift run fluidaudiocli process meeting.wav --mode streaming --threshold 0.7
swift run fluidaudiocli process meeting.wav --mode offline --threshold 0.6 --debug

# Sortformer streaming diarization
swift run fluidaudiocli sortformer audio.wav

# Sortformer benchmark on AMI dataset
swift run fluidaudiocli sortformer-benchmark
```

- `--mode offline` switches the CLI to `OfflineDiarizerManager`, running the full VBx pipeline with PLDA refinement. Expect DER ≈ 18–20 % on AMI-SDM with threshold 0.6.
- Add `--rttm /path/to/ground_truth.rttm` to `process` to compute DER/JER in-place, or `--export-embeddings embeddings.json` for debugging speaker vectors.
- GitHub Actions workflow `offline-pipeline.yml` replays `fluidaudio diarization-benchmark --mode offline --single-file ES2004a` on every PR so failures in model downloads or clustering logic are caught early.

## VAD

```bash
# Offline segmentation with seconds output (default mode)
swift run fluidaudiocli vad-analyze path/to/audio.wav

# Streaming only with 128 ms chunks and a custom threshold (timestamps emitted in seconds)
swift run fluidaudiocli vad-analyze path/to/audio.wav --streaming --threshold 0.65 --min-silence-ms 400

# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudiocli vad-benchmark --num-files 50 --threshold 0.3

# Save benchmark results and enable debug output
swift run fluidaudiocli vad-benchmark --all-files --output vad_results.json --debug
```

`swift run fluidaudiocli vad-analyze --help` lists every tuning option (padding,
negative threshold overrides, max-duration splitting, etc.).

## Datasets

```bash
# Download test sets
swift run fluidaudiocli download --dataset librispeech-test-clean
swift run fluidaudiocli download --dataset librispeech-test-other
swift run fluidaudiocli download --dataset ami-sdm
swift run fluidaudiocli download --dataset vad
```
