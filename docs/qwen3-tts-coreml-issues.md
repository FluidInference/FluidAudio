# Qwen3-TTS CoreML Implementation Issues & Fixes

This document captures the issues encountered during the CoreML port of Qwen3-TTS and their solutions.

## Issue 1: CB0 Token Repetition (Stuck LM)

### Symptoms
- Chinese audio was silent or unintelligible
- English audio sometimes degraded
- CB0 tokens getting stuck at same values (e.g., `[1657, 1657, 1657, ...]`)
- Only 27 unique CB0 values out of 125 frames (should be ~98% unique)
- Audio RMS was 9x quieter than PyTorch reference

### Root Cause
The PyTorch implementation uses `repetition_penalty=1.3` by default, which penalizes recently generated tokens to prevent the LM from getting stuck in repetitive loops. The CoreML port was missing this.

### Fix
Added repetition penalty to the LM decode loop in `Qwen3TtsSynthesizer.swift`:

```swift
// Apply repetition penalty (matching PyTorch default of 1.3)
let repetitionPenalty: Float = 1.3
let recentTokens = allCodebooks.suffix(20).map { $0[0] }  // Last 20 CB0 tokens
for token in recentTokens {
    if token < logits.count && logits[token] > 0 {
        logits[token] /= repetitionPenalty
    } else if token < logits.count {
        logits[token] *= repetitionPenalty
    }
}
```

### Results After Fix
- English: 57/58 unique CB0 (98%), natural EOS at frame 58
- Chinese: 64/65 unique CB0 (98%), natural EOS at frame 65
- Both transcribe correctly with Whisper

## Issue 2: Temperature/TopK Tuning

### Original Values (from PyTorch defaults)
- Temperature: 0.9
- TopK: 50

### Adjusted Values (better quality)
- Temperature: 0.7
- TopK: 30

Lower temperature produces more deterministic, cleaner audio with less noise artifacts.

## Issue 3: Audio Post-Processing

### Symptoms
- Raw audio had sibilance (harsh "s" sounds)
- Some high-frequency artifacts

### Fix
Added `AudioPostProcessor.applyTtsPostProcessing()` with:
- De-essing: -4.0 dB reduction
- Smoothing: enabled

## Verification

### Spectral Comparison (English)
- Mel spectrogram cosine similarity: 93.7%
- MFCC cosine similarity: 94.2%

### Whisper Transcription
- English: "Hello world, this is a test of the text to speech system." ✓
- Chinese: "您好,世界,这是一个文字转语音系统的测试" ✓

## Key Learnings

1. **Repetition penalty is critical** - Without it, autoregressive LMs can get stuck in loops, especially for languages with different token distributions (Chinese)

2. **CB0 drives CB1-15** - When CB0 gets stuck, the code predictor (CB1-15) also produces repetitive patterns, leading to silent/broken audio

3. **Debug with token diversity metrics** - Monitoring unique CB0/CB1 counts and consecutive repeats quickly reveals stuck patterns

4. **Temperature sampling is required** - Greedy decoding (argmax) never produces EOS because codec tokens always have higher logits than EOS token
