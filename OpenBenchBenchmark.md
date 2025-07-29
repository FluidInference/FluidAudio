# OpenBench Speaker Diarization Benchmark Guide

This guide walks you through benchmarking FluidAudio on the OpenBench dataset suite - 8 standardized datasets for evaluating speaker diarization systems.

## Prerequisites

- Python 3.8+ with pip
- Swift 5.5+ (comes with Xcode)
- ~50GB free disk space
- 8GB+ RAM recommended

## Step 1: Clone and Setup OpenBench

```bash
# Clone OpenBench repository
git clone https://github.com/argmaxinc/OpenBench.git
cd OpenBench

# Create Python virtual environment
python3 -m venv venv_new
source venv_new/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pandas soundfile
```

## Step 2: Download Test Datasets

```bash
# Download all 8 test datasets (~30GB)
python download_datasets.py --dataset all --split test

# Or download specific datasets
python download_datasets.py --dataset ami-ihm,ami-sdm,voxconverse --split test
```

Available datasets:
- `earnings21` - Earnings calls
- `ami-ihm` - AMI Individual Headset Mix
- `ami-sdm` - AMI Single Distant Microphone
- `aishell-4` - Chinese meeting recordings
- `voxconverse` - Broadcast media
- `ava-avd` - Movie segments
- `ali-meetings` - AliMeeting dataset
- `icsi-meetings` - ICSI Meeting corpus

## Step 3: Extract Audio Files for Swift

The datasets are in parquet format. We need to extract them to WAV files:

```bash
python extract_all_for_swift.py
```

**Important**: The extraction script automatically saves files directly to `/Users/kikow/brandon/FluidAudioSwift/OpenBenchData/`. No manual copying is needed! The script:
- Reads from: `/Users/kikow/brandon/OpenBench/datasets/`
- Saves to: `/Users/kikow/brandon/FluidAudioSwift/OpenBenchData/`
- Creates WAV files and ground_truth.json files in the correct location for Swift

## Step 4: Run FluidAudio Benchmark

```bash
# Navigate to FluidAudioSwift directory
cd /Users/kikow/brandon/FluidAudioSwift

# Run benchmark on all datasets (use caffeinate to prevent sleep)
caffeinate -i swift run -c release fluidaudio openbench-benchmark --all

# Run a single dataset
swift run -c release fluidaudio openbench-benchmark --datasets ami-ihm

# Run multiple specific datasets
swift run -c release fluidaudio openbench-benchmark --datasets ami-ihm,ami-sdm,voxconverse

# Run with limited files for testing
swift run -c release fluidaudio openbench-benchmark --all --max-files 5
```

## Benchmark Options

- `--all` - Run all 8 datasets
- `--datasets <list>` - Comma-separated list of datasets
- `--max-files <int>` - Limit files per dataset (for testing)
- `--threshold <float>` - Clustering threshold (default: 0.7)
- `--output <file>` - Output JSON file (default: openbench_results.json)
- `--debug` - Enable debug mode

## Expected Runtime

Based on the benchmark results:
- **5 files per dataset**: ~6 minutes
- **All test files (477 total)**: ~5 hours
- **Processing speed**: 0.017 RTF (60x real-time)

## Results Format

The benchmark outputs:
1. **Console summary** with per-dataset metrics
2. **JSON file** with detailed results

Example output:
```
Dataset         Files    Avg RTF      Avg DER      Duration
-----------------------------------------------------------------
ami-ihm         16       0.017        23.5%        543.7min
ami-sdm         16       0.017        26.6%        543.7min
voxconverse     232      0.017        20.5%        2612.2min
...

📈 Performance Summary:
  Total files processed: 477
  Total audio duration: 200.5 hours
  Overall Average RTF: 0.017
  Overall Average DER: 26.0%
```

## Metrics Explained

- **RTF (Real-Time Factor)**: Processing time / Audio duration
  - < 1.0 = Faster than real-time
  - 0.017 = 60x faster than real-time

- **DER (Diarization Error Rate)**: Sum of:
  - False Alarm: Speech detected when none exists
  - Missed Speech: Speech not detected
  - Speaker Error: Wrong speaker attributed

  Lower is better, < 30% is considered good

## Troubleshooting

1. **Segmentation fault**: Usually memory-related when processing many files
   - Solution: Process datasets individually or use `--max-files`

2. **Missing audio files**: Ensure extraction completed successfully
   - Check: `ls -la /Users/kikow/brandon/FluidAudioSwift/OpenBenchData/*/`

3. **Python package errors**:
   ```bash
   pip install --upgrade pandas soundfile numpy
   ```

4. **Swift build errors**:
   ```bash
   swift package clean
   swift build -c release
   ```

## Comparing Results

To compare with other systems:
- **Pyannote 3.0**: ~23% average DER (published benchmarks)
- **FluidAudio**: ~26% average DER at 60x real-time
- **SpeakerKit**: Claims to match pyannote at 9.6x speed

The trade-off is ~3% accuracy for exceptional speed, making FluidAudio ideal for real-time applications.

## 🎯 DER Comparison: FluidAudio vs Pyannote 3.0

| Dataset      | Pyannote 3.0 | FluidAudio | Difference | Better? |
|--------------|--------------|------------|------------|---------|
| ami-ihm      | 19.0%        | 23.5%      | +4.5%      | ❌       |
| ami-sdm      | 22.2%        | 26.6%      | +4.4%      | ❌       |
| aishell-4    | 12.3%        | 22.1%      | +9.8%      | ❌       |
| voxconverse  | 11.3%        | 20.5%      | +9.2%      | ❌       |
| ava-avd      | 49.1%        | 32.6%      | -16.5%     | ✅       |
| ali-meetings | 24.3%*       | 27.7%      | +3.4%      | ❌       |
| **Average**  | **23.0%**    | **25.5%**  | **+2.5%**  | -       |

*AliMeeting (channel 1) in pyannote
