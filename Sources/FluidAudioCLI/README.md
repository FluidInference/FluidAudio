# FluidAudio CLI

The FluidAudio CLI provides various commands for audio processing, transcription, and benchmarking.

## Installation

Build the CLI tool:
```bash
swift build -c release
```

## Commands Overview

### 1. `process` - Audio Diarization
Process a single audio file to identify speakers and their segments.

```bash
# Basic usage
swift run fluidaudio process audio.wav

# With custom output and threshold
swift run fluidaudio process audio.wav --output results.json --threshold 0.7

# With debug mode
swift run fluidaudio process audio.wav --debug
```

**Options:**
- `--output <file>`: Output JSON file (default: prints to console)
- `--threshold <value>`: Clustering threshold 0.0-1.0 (default: 0.7)
- `--debug`: Enable debug output

### 2. `transcribe` - Audio Transcription
Transcribe audio files using streaming ASR with real-time updates.

```bash
# Basic transcription
swift run fluidaudio transcribe audio.wav

# With low-latency configuration
swift run fluidaudio transcribe audio.wav --config low-latency

# With debug output
swift run fluidaudio transcribe audio.wav --debug

# Compare with direct ASR API
swift run fluidaudio transcribe audio.wav --compare
```

**Options:**
- `--config <type>`: Configuration type: `default`, `low-latency`, `high-accuracy`
- `--debug`: Show debug information
- `--compare`: Compare streaming API with direct ASR API
- `--help, -h`: Show help message

**Configurations:**
- `default`: 2.5s chunks, 0.85 confirmation threshold
- `low-latency`: 2.0s chunks, 0.75 confirmation threshold
- `high-accuracy`: 3.0s chunks, 0.90 confirmation threshold

### 3. `multi-stream` - Parallel Transcription
Transcribe multiple audio files in parallel using shared ASR models.

```bash
# Process two different files
swift run fluidaudio multi-stream mic_audio.wav system_audio.wav

# Process same file on both streams
swift run fluidaudio multi-stream audio.wav

# With debug output
swift run fluidaudio multi-stream audio1.wav audio2.wav --debug
```

**Options:**
- `--debug`: Show debug information
- `--help, -h`: Show help message

### 4. `diarization-benchmark` - Speaker Diarization Benchmark
Run comprehensive benchmarks on evaluation datasets.

```bash
# Run on AMI dataset with auto-download
swift run fluidaudio diarization-benchmark --auto-download

# Test single file
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.7

# Run on specific dataset
swift run fluidaudio diarization-benchmark --dataset ami-sdm --max-files 10

# Save results to file
swift run fluidaudio diarization-benchmark --output benchmark_results.json
```

**Options:**
- `--dataset <name>`: Dataset to use (ami-sdm, ami-mdm, voxconverse)
- `--auto-download`: Automatically download required datasets
- `--single-file <id>`: Test a single file (e.g., ES2004a)
- `--threshold <value>`: Clustering threshold (default: 0.7)
- `--max-files <n>`: Maximum files to process
- `--output <file>`: Save results to JSON file
- `--verbose`: Show detailed progress

### 5. `vad-benchmark` - Voice Activity Detection Benchmark
Benchmark VAD performance on test datasets.

```bash
# Run VAD benchmark
swift run fluidaudio vad-benchmark --num-files 40

# With custom threshold 
swift run fluidaudio vad-benchmark --threshold 0.8

# Test on specific dataset
swift run fluidaudio vad-benchmark --dataset voices-subset
```

**Options:**
- `--num-files <n>`: Number of files to test (or `--all-files`)
- `--threshold <value>`: VAD threshold (default: 0.3)
- `--dataset <name>`: Dataset to use (e.g., `mini50`, `voices-subset`, `musan-full`)
- `--debug`: Verbose logging and per-file RTFx

### 6. `asr-benchmark` - ASR Benchmark
Benchmark ASR performance on LibriSpeech or other datasets.

```bash
# Run on LibriSpeech test-clean
swift run fluidaudio asr-benchmark --subset test-clean --max-files 100

# Run on test-other subset
swift run fluidaudio asr-benchmark --subset test-other --max-files 50

# With verbose output
swift run fluidaudio asr-benchmark --verbose
```

**Options:**
- `--subset <name>`: LibriSpeech subset (test-clean, test-other)
- `--max-files <n>`: Maximum files to process
- `--verbose`: Show detailed progress

### 7. `parakeet-eou` - Streaming ASR
Real-time streaming transcription with end-of-utterance detection.

```bash
swift run fluidaudio parakeet-eou --input audio.wav --use-cache
swift run fluidaudio parakeet-eou --benchmark --chunk-size 160 --max-files 100 --use-cache
```

**Options:** `--input <path>`, `--benchmark`, `--max-files <n>`, `--chunk-size <160|320|1600>`, `--eou-debounce <ms>`, `--use-cache`, `--models <path>`, `--output <path>`, `--verbose`

### 8. `download` - Download Datasets

```bash
swift run fluidaudio download --dataset ami-sdm
swift run fluidaudio download --list
```

**Options:** `--dataset <name>`, `--list`

## Output Formats

### Diarization Output (JSON)
```json
{
  "audioFile": "audio.wav",
  "durationSeconds": 300.5,
  "speakerCount": 3,
  "segments": [
    {
      "speakerId": "Speaker 1",
      "startTimeSeconds": 0.0,
      "endTimeSeconds": 45.2,
      "qualityScore": 0.85
    }
  ],
  "processingTimeSeconds": 15.2,
  "realTimeFactor": 0.05
}
```

### Transcription Output
- Real-time updates showing volatile and confirmed text
- Final transcription with performance metrics
- RTFx (Real-Time Factor) showing processing speed

## Performance Notes

- All commands process audio as fast as possible (no artificial delays)
- Multi-stream command demonstrates parallel processing with shared models
- Benchmarks provide detailed performance metrics including DER, WER, and RTFx

## Examples

### Complete Workflow Example
```bash
# 1. Download dataset
swift run fluidaudio download --dataset ami-sdm

# 2. Run diarization benchmark
swift run fluidaudio diarization-benchmark --dataset ami-sdm --output results.json

# 3. Process individual file
swift run fluidaudio process audio.wav --threshold 0.7

# 4. Transcribe audio
swift run fluidaudio transcribe audio.wav --config low-latency

# 5. Multi-stream transcription
swift run fluidaudio multi-stream mic.wav system.wav
```

### Quick Test
```bash
# Test with included sample files
swift run fluidaudio transcribe medical.wav
swift run fluidaudio process IS1001a.Mix-Headset.wav --threshold 0.7
```

## Troubleshooting

1. **Model Download Issues**: The CLI will automatically download required models on first use. Ensure you have internet connectivity.

2. **Memory Usage**: For long audio files, ensure sufficient memory is available.

3. **Performance**: Use release build (`swift build -c release`) for best performance.

4. **Audio Format**: The CLI automatically handles various audio formats and sample rates.
