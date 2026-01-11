# NeMo Sortformer AMI Benchmark

This directory contains tools for comparing the Swift/CoreML Sortformer implementation against NVIDIA's original NeMo Sortformer model.

## Overview

The `nemo_ami_benchmark.py` script runs NVIDIA's Sortformer model on the AMI SDM dataset to provide a baseline comparison for the Swift/CoreML implementation.

## Requirements

### Python Environment

```bash
# Create virtual environment with Python 3.10+
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchaudio torchcodec
pip install nemo_toolkit[asr] pyannote.metrics
```

### HuggingFace Authentication

The NVIDIA Sortformer model is gated and requires HuggingFace authentication:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model license at [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### AMI Dataset

Download the AMI SDM test set audio files and RTTM ground truth:

```bash
# Audio files should be in:
~/FluidAudioDatasets/ami_official/sdm/

# RTTM files should be in:
~/FluidAudioDatasets/ami_official/rttm/
```

RTTM files can be downloaded from [pyannote AMI diarization setup](https://github.com/pyannote/AMI-diarization-setup).

## Usage

### Basic Usage

```bash
# Run on single file
HF_TOKEN="your_token" python nemo_ami_benchmark.py --single-file ES2004a --device cpu

# Run on all 16 AMI test meetings
HF_TOKEN="your_token" python nemo_ami_benchmark.py --device cpu

# Save results to JSON
HF_TOKEN="your_token" python nemo_ami_benchmark.py --output results.json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio-dir` | Path to AMI audio files | `~/FluidAudioDatasets/ami_official/sdm` |
| `--rttm-dir` | Path to RTTM ground truth files | `~/FluidAudioDatasets/ami_official/rttm` |
| `--output`, `-o` | Output JSON file path | None |
| `--single-file` | Run on single meeting (e.g., ES2004a) | All 16 meetings |
| `--device` | Device to use (cpu, cuda, mps) | mps if available, else cpu |
| `--batch` | Use batch mode instead of streaming | False |
| `--model-path` | Path to local .nemo model file | Downloads from HuggingFace |

## Configuration Settings

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `nvidia/diar_sortformer_4spk-v1` | NVIDIA Sortformer 4-speaker model |
| Sample Rate | 16000 Hz | Audio sample rate |
| Frame Duration | 80 ms | Duration per output frame |
| Num Speakers | 4 | Maximum number of speakers |

### High-Latency Streaming Config

These settings match the Swift `SortformerConfig.nvidiaHighLatency`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk Length | 48 frames | Core chunk length in encoder frames |
| Left Context | 56 frames | Left context in encoder frames |
| Right Context | 56 frames | Right context in encoder frames |
| Subsampling Factor | 8 | Mel frames per encoder frame |
| **Total Context** | **30.4 seconds** | (48 + 56 + 56) * 8 * 10ms |

### Post-Processing Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| Onset Threshold | 0.5 | Threshold for speaker activity detection |
| Offset Threshold | 0.5 | Threshold for speaker activity end |

## AMI Test Meetings

The benchmark runs on 16 AMI SDM test meetings:

| Series | Meetings |
|--------|----------|
| EN2002 | EN2002a, EN2002b, EN2002c, EN2002d |
| ES2004 | ES2004a, ES2004b, ES2004c, ES2004d |
| IS1009 | IS1009a, IS1009b, IS1009c, IS1009d |
| TS3003 | TS3003a, TS3003b, TS3003c, TS3003d |

## Output Metrics

| Metric | Description |
|--------|-------------|
| DER | Diarization Error Rate (Miss + FA + SE) |
| Miss % | Missed speech (false negatives) |
| FA % | False alarm (false positives) |
| SE % | Speaker error (wrong speaker assigned) |
| Speakers | Detected / Ground truth speaker count |
| RTFx | Real-time factor (audio duration / processing time) |

## Example Output

```
================================================================================
NEMO SORTFORMER AMI BENCHMARK
================================================================================
Device: cpu
Mode: Streaming (30.4s chunks)
Audio dir: /Users/user/FluidAudioDatasets/ami_official/sdm
RTTM dir: /Users/user/FluidAudioDatasets/ami_official/rttm
Meetings: 1

Loading Sortformer model...
Model loaded in 2.35s

----------------------------------------------------------------------
Meeting         DER %   Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
ES2004a         34.0%    30.7%     0.9%     2.3% 4/       4     0.2x
----------------------------------------------------------------------
AVERAGE         34.0%    30.7%     0.9%     2.3%          -     0.2x
======================================================================
```

## Comparison with Swift/CoreML

| Metric | NeMo Python (CPU) | Swift/CoreML (ANE) |
|--------|-------------------|---------------------|
| DER | 34.0% | 32.3% |
| Miss Rate | 30.7% | ~29% |
| False Alarm | 0.9% | ~1% |
| Speaker Error | 2.3% | ~2% |
| RTFx | 0.2x | ~5x |

The Swift/CoreML implementation achieves comparable accuracy while being significantly faster due to Apple Neural Engine acceleration.

## Notes

- CPU inference is slow (~0.2x real-time). Use CUDA for faster inference if available.
- MPS (Apple Silicon GPU) may have memory issues with long audio files.
- The NeMo model runs in batch mode; Swift implements true streaming chunking on top.

## References

- [NVIDIA Sortformer Model](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [pyannote AMI Diarization Setup](https://github.com/pyannote/AMI-diarization-setup)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)
