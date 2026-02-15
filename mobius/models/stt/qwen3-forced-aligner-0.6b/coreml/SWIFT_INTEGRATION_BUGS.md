# Qwen3-ForcedAligner-0.6B CoreML — Swift Integration Bugs & Fixes

Documented issues encountered while integrating the Qwen3-ForcedAligner-0.6B CoreML int8 models into FluidAudio (Swift). These are pitfalls that any future CoreML integration should watch for.

---

## Bug 1: MLMultiArray Non-Contiguous Strides (Critical)

**Symptom**: Decoder output logits were completely wrong. At position 21, Python showed `[15.27, 9.05, 7.61, ...]` (argmax=0), Swift showed `[-5.58, -6.20, -5.19, ...]` (argmax=168). All timestamps were 10x too high (13440-24960ms for 1.4s audio).

**Root Cause**: CoreML `MLMultiArray` outputs can have non-contiguous memory layouts with padding between rows. The decoder output shape was `[1, 1024, 5000]` but actual strides were `[5128192, 5008, 1]` — the stride for the sequence dimension was **5008**, not 5000 (the vocab dimension). There are 8 padding elements between each row.

The Swift code assumed contiguous layout:
```swift
// WRONG: assumes contiguous memory
let offset = i * vocabDim  // vocabDim=5000, but actual stride=5008
```

**Fix**: Always use `strides[].intValue` from the MLMultiArray:
```swift
// CORRECT: use actual stride
let logitsStride = logits.strides[1].intValue  // 5008
let logitsPtr = logits.dataPointer.bindMemory(
    to: Float.self, capacity: logits.strides[0].intValue
)
for i in 0..<seqLen {
    let offset = i * logitsStride
    // ...
}
```

**Impact**: This affected all three model outputs (audio encoder, embedding, decoder). All MLMultiArray parsing must be stride-aware.

**Lesson**: Never assume `shape[n] == strides[n-1]`. CoreML may add padding for memory alignment. Always use the stride values.

---

## Bug 2: Encoder Output Shape Dimensionality (Critical — caused SIGSEGV)

**Symptom**: Segfault (SIGSEGV) when processing audio longer than ~1.4 seconds. AddressSanitizer reported heap-buffer-overflow at the encoder feature extraction loop.

**Root Cause**: The audio encoder outputs a **3D** tensor `[1, N, 1024]`, not 2D `[N, 1024]`. The code used `strides[0]` (the batch stride = `N * 1024`), but needed `strides[1]` (the frame stride = `1024`).

```
Encoder output: shape=[1, 13, 1024], strides=[13312, 1024, 1]
```

Using `strides[0] = 13312` as the per-frame stride meant:
- Frame 0: offset 0 (OK)
- Frame 1: offset 13312 (out of bounds! total buffer = 13312)

For the first test (1.4s audio, 1 encoder chunk), this happened to not crash because the out-of-bounds memory was still mapped. With longer audio (2.36s, 3 chunks), different memory layout triggered the crash.

**Fix**: Check shape dimensionality and use the correct stride index:
```swift
let featStride: Int
if features.shape.count == 3 {
    featStride = features.strides[1].intValue  // 3D: [batch, frames, dim]
} else {
    featStride = features.strides[0].intValue  // 2D: [frames, dim]
}
```

**Lesson**: Don't assume the output tensor dimensionality. CoreML models may or may not include a batch dimension. Always check `shape.count` and index strides accordingly.

---

## Bug 3: Mel Spectrogram Scale (Slaney vs HTK)

**Symptom**: Swift mel values differed significantly from Python reference. mel[0,:5] was `[-0.695, -0.695, -0.695, ...]` (constant) vs Python's `[-0.285, -0.208, ...]`.

**Root Cause**: The existing `WhisperMelSpectrogram` in FluidAudio uses the **HTK** mel scale (the `librosa` default). However, HuggingFace's `WhisperFeatureExtractor` (used by the forced aligner) uses `mel_scale="slaney"` + `norm="slaney"`.

HTK mel scale:
```
mel = 2595 * log10(1 + f/700)
```

Slaney mel scale:
```
if f < 1000: mel = 3 * f / 200          (linear)
if f >= 1000: mel = 15 + 27/ln(6.4) * ln(f/1000)  (logarithmic)
```

Slaney normalization: `2 / (f_high - f_low)` per filterbank band (area normalization).

**Fix**: Created `ForcedAlignerMelSpectrogram` with Slaney mel scale and normalization, separate from the existing HTK-based `WhisperMelSpectrogram`.

**Lesson**: "Whisper mel spectrogram" is not a single thing. The mel scale and normalization depend on the upstream library and model. Always verify which variant the model was trained with.

---

## Bug 4: Missing STFT Center Padding

**Symptom**: After fixing the mel scale, values were closer but still wrong. Swift mel[0,:5] = `[-0.217, 0.007, ...]` vs Python `[-0.285, -0.208, ...]`.

**Root Cause**: PyTorch's `torch.stft()` uses `center=True` by default, which reflect-pads the audio by `nFFT/2` on each side before computing the STFT. The Swift implementation had no center padding.

Without center padding, the first frame window starts at sample 0 and only sees the right half of data, leading to different spectral content for early frames.

**Fix**: Added reflect padding before STFT:
```swift
private static func reflectPad(_ input: [Float], padLen: Int) -> [Float] {
    let n = input.count
    var result = [Float](repeating: 0, count: padLen + n + padLen)
    // Left reflect: input[padLen], input[padLen-1], ..., input[1]
    for i in 0..<padLen { result[i] = input[padLen - i] }
    // Center: copy input
    for i in 0..<n { result[padLen + i] = input[i] }
    // Right reflect: input[n-2], input[n-3], ..., input[n-1-padLen]
    for i in 0..<padLen { result[padLen + n + i] = input[n - 2 - i] }
    return result
}
```

**Lesson**: When porting Python audio code, check `torch.stft()` defaults carefully. `center=True` is the default and omitting it silently produces wrong results.

---

## Bug 5: MRoPE Position IDs for Padded Positions

**Symptom**: Timestamps were 10x too high (13440-24960ms for 1.4s audio). This was a contributing factor alongside the stride bug.

**Root Cause**: Position IDs for padded positions (beyond the actual sequence length) incremented continuously `[0, 1, 2, ..., 1023]` instead of repeating the last valid position `[0, 1, ..., 37, 37, 37, ...]`.

The Python reference computes positions as `cumsum(attention_mask) - 1`, where `attention_mask` is 1 for real tokens and 0 for padding. This means padded positions all get the same position ID as the last real token.

**Fix**: Changed MRoPE computation to clamp positions:
```swift
func compute(totalLen: Int, contentLen: Int) -> (cos: [Float], sin: [Float]) {
    let lastValidPos = max(contentLen - 1, 0)
    for p in 0..<totalLen {
        let posId = min(p, lastValidPos)  // Padded positions repeat last valid
        // ...
    }
}
```

**Lesson**: Attention mask handling matters for position encodings. Padded positions should not get unique position IDs.

---

## Summary of Fixes Applied

| Bug | Severity | Symptom | Root Cause |
|-----|----------|---------|------------|
| MLMultiArray strides | Critical | Wrong logits, wrong timestamps | Non-contiguous memory layout |
| Encoder 3D shape | Critical | SIGSEGV on longer audio | Wrong stride index for 3D tensor |
| Slaney mel scale | High | Wrong mel values | HTK vs Slaney mel scale |
| STFT center padding | High | Wrong mel values | Missing reflect padding |
| MRoPE position IDs | Medium | Inflated timestamps | Padded positions not clamped |

## General CoreML Integration Lessons

1. **Always use stride-aware MLMultiArray parsing.** Never assume `shape[n] * shape[n+1] * ... = strides[n-1]`. CoreML adds alignment padding.
2. **Check output tensor dimensionality.** Models may or may not include a batch dimension. Use `shape.count` to determine stride indexing.
3. **Verify mel spectrogram parameters against the training pipeline.** "Whisper-compatible" has at least two variants (HTK and Slaney).
4. **Check STFT defaults.** `torch.stft(center=True)` is the default and adds reflect padding.
5. **Test with multiple input lengths.** Short inputs may mask buffer overflows that only manifest with longer sequences.
6. **Use AddressSanitizer.** `swift build -Xswiftc -sanitize=address` catches buffer overflows that would otherwise be silent memory corruption.
