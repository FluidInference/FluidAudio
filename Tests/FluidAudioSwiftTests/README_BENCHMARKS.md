# FluidAudioSwift Research Benchmarks

This directory contains benchmark tests that evaluate FluidAudioSwift's speaker diarization performance using standard research datasets and metrics.

## Overview

The benchmark tests follow the same evaluation protocols used in academic research papers, particularly those evaluating speaker diarization systems on the AMI Meeting Corpus. This ensures that FluidAudioSwift's performance can be directly compared with published research results.

## Official AMI Meeting Corpus Benchmarks

### What is the AMI Corpus?

The AMI Meeting Corpus is the gold standard dataset for speaker diarization research, containing 100 hours of meeting recordings with manual annotations. It's used in virtually all speaker diarization research papers for evaluation.

### Getting the Official Data

**Important**: These benchmarks require the official AMI Meeting Corpus data, which must be downloaded from the University of Edinburgh (the same source used by research papers).

#### Step 1: Download Audio Files

1. Visit the official AMI corpus download page: https://groups.inf.ed.ac.uk/ami/download/
2. Select test meetings (recommended for benchmarking):
   - **Scenario meetings**: ES2002a, ES2003a, ES2004a, ES2005a
   - **Non-scenario meetings**: IS1000a, IS1001a, IS1002a
   - **TNO meetings**: TS3003a, TS3004a

3. Select audio stream types:
   - **Individual headsets** (IHM): For close-talking microphone evaluation
   - **Headset mix** (SDM): For single distant microphone evaluation
   - **Microphone array** (MDM): For multiple distant microphone evaluation

4. Download the WAV files to your local machine

#### Step 2: Download Ground Truth Annotations

1. On the same download page, download:
   - **AMI manual annotations v1.6.2** (22MB) - Contains the ground truth speaker segments
   - **AMI automatic annotations v1.5.1** (68MB) - Contains additional metadata

#### Step 3: Organize Files

Create the following directory structure in your home directory:

```
~/FluidAudioSwift_Datasets/ami_official/
├── ihm/                    # Individual Headset Microphones
│   ├── ES2002a.Headset-0.wav
│   ├── ES2003a.Headset-0.wav
│   └── ...
├── sdm/                    # Single Distant Microphone
│   ├── ES2002a.Mix-Headset.wav
│   ├── ES2003a.Mix-Headset.wav
│   └── ...
├── mdm/                    # Multiple Distant Microphones
│   ├── ES2002a.Array1-01.wav
│   ├── ES2003a.Array1-01.wav
│   └── ...
└── annotations/            # Ground truth annotations
    ├── AMI_manual_1.6.2/
    └── AMI_automatic_1.5.1/
```

### Running the Benchmarks

Once you have the official data set up, you can run the benchmark tests:

```bash
# Run all benchmark tests
swift test --filter BenchmarkTests

# Run specific benchmark variants
swift test --filter testAMI_Official_IHM_Benchmark
swift test --filter testAMI_Official_SDM_Benchmark
swift test --filter testAMI_Research_Protocol_Evaluation
```

### Understanding the Results

The benchmarks report standard research metrics:

#### Diarization Error Rate (DER)
- **Primary metric** used in all speaker diarization papers
- Combines three error types:
  - **False Alarm**: System detects speech when there is silence
  - **Missed Detection**: System misses actual speech
  - **Speaker Error**: System assigns wrong speaker identity

#### Jaccard Error Rate (JER)
- Measures temporal overlap accuracy between predicted and ground truth segments
- Complementary metric to DER

#### Research Baselines

The benchmarks compare against published research results:

| System | AMI IHM DER | AMI SDM DER | Paper |
|--------|-------------|-------------|-------|
| Powerset BCE | 18.5% | ~25% | Interspeech 2023 |
| EEND | 25.3% | ~32% | ICASSP 2019 |
| x-vector clustering | 28.7% | ~35% | Traditional baseline |

### Expected Performance

- **IHM (Individual Headset)**: 15-25% DER for modern systems
- **SDM (Single Distant Mic)**: 25-35% DER for modern systems
- **MDM (Multiple Distant Mic)**: 20-30% DER for modern systems

SDM is typically 5-10% higher DER than IHM due to far-field audio challenges.

## Technical Details

### Evaluation Protocol

The benchmarks follow the standard research evaluation protocol:

1. **Frame-based evaluation**: 10ms frames (0.01 seconds)
2. **No collar**: No forgiveness regions around speaker boundaries
3. **Standard test sets**: Uses the same AMI test meetings as research papers
4. **Multiple audio conditions**: Evaluates both close-talking (IHM) and far-field (SDM/MDM)

### Audio Processing

- **Sample rate**: 16 kHz (standard for speech processing)
- **Chunk size**: 10 seconds (matches FluidAudioSwift's processing window)
- **Audio format**: WAV files from official AMI corpus

### Model Architecture

FluidAudioSwift uses:
- **Segmentation**: PyAnnote-based neural segmentation
- **Embedding**: WeSpeaker embedding extraction
- **Powerset encoding**: 7-class powerset for up to 3 simultaneous speakers
- **Backend**: CoreML for efficient on-device processing

## Troubleshooting

### Tests Skip with "No Official AMI Data Found"

This means the benchmark couldn't find the official AMI corpus files. Make sure:

1. Files are in the correct directory: `~/FluidAudioSwift_Datasets/ami_official/`
2. Audio files have the correct naming convention (e.g., `ES2002a.Headset-0.wav`)
3. Files are in the appropriate subdirectories (`ihm/`, `sdm/`, `mdm/`)

### Models Not Available

If tests skip with "models not available":

1. Ensure you have internet connectivity for model download
2. Models are downloaded from Hugging Face on first run
3. Check available disk space (models are ~100MB each)

### Performance Lower Than Expected

If DER is higher than research baselines:

1. **Audio quality**: Ensure you're using official AMI corpus files
2. **Ground truth**: Verify annotations are properly loaded
3. **Model version**: Check if you're using the latest FluidAudioSwift models
4. **Test set**: Ensure you're testing on the same meetings as research papers

## Research Usage

These benchmarks are designed for:

- **Academic research**: Comparing FluidAudioSwift with other diarization systems
- **Algorithm development**: Measuring improvements to the diarization pipeline
- **Performance validation**: Ensuring consistent results across different hardware
- **Reproducible research**: Following standard evaluation protocols

## Contributing

When adding new benchmark tests:

1. Follow the same evaluation protocol as existing tests
2. Use official research datasets when possible
3. Report standard metrics (DER, JER)
4. Include comparison with published baselines
5. Document the evaluation setup clearly

## References

- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [Powerset Multi-class Cross Entropy Loss for Neural Speaker Diarization](https://www.isca-speech.org/archive/interspeech_2023/)
- [End-to-End Neural Speaker Diarization with Permutation-free Objectives](https://arxiv.org/abs/1909.05952)
