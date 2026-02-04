# Voice Cloning Evaluation Scripts

Tools for evaluating PocketTTS voice cloning quality.

## evaluate_voice.py

Compares a reference voice sample with synthesized TTS output using neural speaker embeddings (Resemblyzer).

### Install

```bash
pip install resemblyzer
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

### Quality Thresholds

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.85+ | Excellent | Very close voice match |
| 0.75+ | Good | Clearly same speaker |
| 0.65+ | Fair | Some similarity |
| <0.65 | Poor | Different speaker characteristics |

### Example Workflow

```bash
# 1. Clone a voice using FluidAudio CLI
fluidaudio tts "Hello, this is a test." --backend pocket --clone-voice speaker.wav -o output.wav

# 2. Evaluate the result
python Scripts/voice_cloning/evaluate_voice.py speaker.wav output.wav
```
