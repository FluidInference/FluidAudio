# Streaming EOU Cache Stride Fix

## Issue

The streaming Parakeet EOU 120M model was producing ~68% WER instead of the expected 5-12% WER.

## Root Cause

**MLMultiArray stride mismatch between CoreML encoder output and state cache.**

The encoder outputs `newCacheTime` with strides `[8192, 8192, 16, 1]`, but the state cache `cacheLastTime` has strides `[4096, 4096, 8, 1]`. Using `memcpy` copied data at incorrect offsets, corrupting every other value in the cache.

Evidence after copy showed zeros at alternating indices:
```
cache_time[0,0,:5,0] = [-1.668, 0.0, 1.677, 0.0, -2.549]
                              ^^^        ^^^
                           corrupted   corrupted
```

## Fix

Replaced `memcpy` with stride-aware element-by-element copy in `StreamingEncoderState.swift`:

```swift
private func copyMLMultiArrayWithStrides(from src: MLMultiArray, to dst: MLMultiArray) {
    let srcStrides = src.strides.map { $0.intValue }
    let dstStrides = dst.strides.map { $0.intValue }
    let shape = dst.shape.map { $0.intValue }

    let srcPtr = src.dataPointer.bindMemory(to: Float.self, capacity: src.count)
    let dstPtr = dst.dataPointer.bindMemory(to: Float.self, capacity: dst.count)

    for i0 in 0..<shape[0] {
        for i1 in 0..<shape[1] {
            for i2 in 0..<shape[2] {
                for i3 in 0..<shape[3] {
                    let srcIdx = i0 * srcStrides[0] + i1 * srcStrides[1] + i2 * srcStrides[2] + i3 * srcStrides[3]
                    let dstIdx = i0 * dstStrides[0] + i1 * dstStrides[1] + i2 * dstStrides[2] + i3 * dstStrides[3]
                    dstPtr[dstIdx] = srcPtr[srcIdx]
                }
            }
        }
    }
}
```

## Results

Benchmark on 100 LibriSpeech test-clean files:

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Average WER | ~68% | **44.6%** |
| Median WER | - | **40.6%** |
| Average CER | - | **22.5%** |
| RTFx | ~38x | **48.7x** |
| Avg Chunk Latency | - | **14.3ms** |
| EOU Detection Rate | - | **74%** |

## Remaining Issue

WER of 45% is still far from the expected 5-12%. The output shows token repetition patterns:
- `"sain sain sain sain sain saint"`
- `"that that that that that"`
- `"into into into into into"`

This indicates the RNNT decoder may have issues with:
- Decoder LSTM state management across chunks
- Max symbols per step limiting
- Blank token probability thresholds

Further investigation needed in `RnntDecoder.swift`.

## Files Modified

- `Sources/FluidAudio/ASR/RNNT/StreamingEncoderState.swift` - Added stride-aware cache copy
- `Sources/FluidAudio/ASR/RNNT/StreamingEouAsrManager.swift` - Removed debug logging
