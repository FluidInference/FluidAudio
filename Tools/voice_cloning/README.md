# Voice Cloning Evaluation Scripts

Tools for evaluating PocketTTS voice cloning quality using spectral similarity.

## evaluate_voice.py

Compares a reference voice sample with synthesized TTS output using mel-spectrogram and MFCC similarity metrics. No neural network required.

### Install

```bash
pip install librosa numpy
# Or minimal (scipy fallback):
pip install scipy numpy

# Optional for plotting:
pip install matplotlib
```

### Usage

```bash
# Basic comparison
python evaluate_voice.py reference.wav synthesized.wav

# With visualization
python evaluate_voice.py reference.wav synthesized.wav --plot

# JSON output
python evaluate_voice.py reference.wav synthesized.wav --json
```

### Metrics

| Metric | Description |
|--------|-------------|
| Mel Similarity | Cosine similarity of mean mel spectrum (voice timbre) |
| MFCC Similarity | Cosine similarity of mean MFCCs (voice characteristics) |
| MFCC Std Similarity | Similarity of MFCC dynamics |
| Combined Score | Weighted average (0.4 mel + 0.4 mfcc + 0.2 mfcc_std) |

### Quality Thresholds

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.90+ | Excellent | Very close spectral match |
| 0.80+ | Good | Similar voice characteristics |
| 0.70+ | Fair | Some similarity |
| <0.70 | Poor | Different spectral characteristics |

### Example Workflow

```bash
# 1. Clone a voice using FluidAudio CLI
fluidaudio tts "Hello, this is a test." --backend pocket --clone-voice speaker.wav -o output.wav

# 2. Evaluate the result
python Tools/voice_cloning/evaluate_voice.py speaker.wav output.wav --plot
```

### Output Example

```
Reference:   speaker.wav
Synthesized: output.wav

Reference duration:   5.23s
Synthesized duration: 2.15s

Computing spectral similarity...

  Mel Similarity:      0.9234
  MFCC Similarity:     0.8876
  MFCC Std Similarity: 0.8543
  Combined Score:      0.8951
  Quality:             Good
```
