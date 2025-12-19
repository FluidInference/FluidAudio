# Earnings22 Keyword Test Suite

Test suite for verifying FluidAudio transcription accuracy using the Earnings22 keyword-spotting dataset.

## Prerequisites

- Python 3.8+
- pandas (`pip install pandas`)
- FluidAudio built in release mode

Build FluidAudio:
```bash
swift build -c release
```

Copy Joint Decision Single Step Model from mobius:
```bash
cp -R .../mobius/models/stt/parakeet-tdt-v3-0.6b/coreml/compiled/parakeet_coreml/parakeet_joint_decision_single_step.mlmodelc JointDecisionSingleStep.mlmodelc
```

**NOTE**
If JointDecisionSingleStep.mlmodelc doesn't exist, beam search is unavailable and greedy decoding is used instead.


## Setup

### 1. Download the Dataset

```bash
cd test-earnings
python download_parquet.py
```

This downloads the test and validation parquet files from `argmaxinc/earnings22-kws-golden` on Hugging Face.

### 2. Extract the Dataset

```bash
python extract_parquet.py test-00000-of-00001.parquet test-dataset
```

This extracts audio files and metadata into the `test-dataset` directory:
- `{file_id}.wav` - Audio file
- `{file_id}.text.txt` - Original transcription text
- `{file_id}.keywords.txt` - Keywords for custom vocabulary
- `{file_id}.dictionary.txt` - Dictionary words to verify in output

## Running Tests

### Run a Single Test

Test a specific file (default: `4468654_chunk45`):
```bash
python test_keyword.py
```

Test a specific file by ID:
```bash
python test_keyword.py 4468654_chunk45
```

### Run All Tests

Run tests on all files with non-empty dictionaries:
```bash
python test_keyword.py --all
```

### Options

| Option              | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| `file_id`           | File ID to test (default: `4468654_chunk45`)                        |
| `--all`             | Run tests on all files with dictionary entries                      |
| `--data-dir DIR`    | Directory containing extracted files (default: `test-dataset`)      |
| `--fluidaudio PATH` | Path to FluidAudio binary (default: `../.build/release/fluidaudio`) |

### Examples

```bash
# Test specific file with custom FluidAudio path
python test_keyword.py 4468654_chunk45 --fluidaudio /path/to/fluidaudio

# Test all files from a different data directory
python test_keyword.py --all --data-dir my-dataset
```

## Test Output

Each test displays:
- Expected vs transcribed text
- WER (Word Error Rate)
- Word-level analysis (matching, missing, extra words)
- Dictionary word verification (pass/fail for each word)

Summary includes:
- Average WER across all tests
- Dictionary check pass rate
- Most frequently missing words
