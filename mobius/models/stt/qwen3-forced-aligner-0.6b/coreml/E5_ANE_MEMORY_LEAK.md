# E5/ANE IOSurface Memory Leak Investigation

## Problem

CoreML's Apple Neural Engine (ANE) runtime leaks IOSurface buffers on repeated predictions via the audio encoder model. The audio encoder contains 3 `Conv2d` layers that CoreML routes to ANE regardless of the `.cpuAndGPU` compute units setting. After ~500 predictions, the process crashes with IOSurface allocation failures.

The leak originates in Apple's `e5rt` runtime (the ANE execution backend). Each prediction allocates IOSurface buffers that are never fully released, accumulating until the system refuses to allocate more.

## Root Cause

- The 3 `Conv2d` layers in `Qwen3ASRAudioEncoder` (conv2d1, conv2d2, conv2d3) are the only ops CoreML routes to ANE
- `.cpuAndGPU` is a **preference**, not a constraint -- CoreML still sends conv ops to ANE
- The embedding and decoder models have no conv ops and run entirely on GPU with no leak

## Approaches Tested

### Approach 1: ObjC @autoreleasepool Batch Drain
Wrap predictions in `@autoreleasepool` via an ObjC wrapper (`CoreMLPredictionWrapper`), draining every 50 predictions.

- **Result**: 5.7x RTFx, no crashes through 200 samples
- **Issue**: Slower than native API due to forced drain overhead

### Approach 2: Native Batch API (WINNER)
Use `MLModel.predictions(fromBatch:)` instead of individual `predict()` calls for the audio encoder. Batch all mel chunks into a single `MLArrayBatchProvider`.

- **Result**: 7.2x RTFx, no crashes through 200 samples, 306MB RAM
- **Why it works**: The batch API manages IOSurface buffers internally with a single allocation/release cycle per batch, avoiding the per-prediction leak accumulation
- **Periodic model reload every 300 samples** as a safety net for very long runs (1000+)

### Approach 3: Model Surgery (Remove Conv Ops)
Re-export the audio encoder with conv2d decomposed into pad + slice + reshape + matmul, eliminating all conv ops so nothing routes to ANE.

- **Conversion**: Successful -- 0 conv ops, equivalence verified (max diff 1.79e-06)
- **GPU execution**: FAILED -- MPS graph compiler crashes on the replacement ops (`slice_by_index`, `reshape` patterns) with `MLIR pass manager failed`
- **CPU-only fallback**: Works but 4.5x RTFx -- 38% slower than approach 2

Variants tried:
1. `F.unfold` + matmul -- coremltools fuses it back to conv at PyTorch frontend stage
2. Disabled 9 conv fusion passes -- conv still created at frontend, not optimization stage
3. `F.pad` + fancy indexing (gather) + matmul -- 0 conv ops but MPS crashes on gather
4. `F.pad` + strided slicing + matmul -- 0 conv ops but MPS crashes on strided slice_by_index
5. `F.pad` + simple slice + reshape + select + matmul -- 0 conv ops but MPS crashes (rank 5 slice patterns still incompatible)

### Approach 4: Bake compute_units at Conversion
Set `compute_units=CPU_AND_GPU` in the CoreML model spec to prevent ANE routing at the model level.

- **Result**: Not viable -- this is already a hint, not a constraint. CoreML ignores it for conv ops.

## Final Solution

**Approach 2: Native Batch API** in `ForcedAlignerInference.encodeAudio()`:

```swift
// Collect all mel chunk inputs
var melInputs: [MLDictionaryFeatureProvider] = []
for chunk in chunks {
    melInputs.append(try createMelInput(chunk))
}

// Single batch prediction -- avoids per-prediction IOSurface leak
let batchProvider = MLArrayBatchProvider(array: melInputs)
let batchResults = try models.audioEncoder.predictions(fromBatch: batchProvider)
```

Plus periodic model reload every 300 samples in `AlignBenchmark` as a safety net:

```swift
if idx > 0 && idx % 300 == 0 {
    try await manager.loadModels(from: modelDir)
}
```

## Benchmark Results (Buckeye Corpus, 200 samples, M2)

| Approach | RTFx | RAM | Crash? |
|----------|------|-----|--------|
| ObjC batch drain (every 50) | 5.7x | 156-207MB | No |
| **Native batch API** | **7.2x** | **306MB** | **No** |
| No-conv encoder + cpuOnly | 4.5x | 2.3GB | No |
| No-conv encoder + cpuAndGPU | N/A | N/A | MPS crash |
| ObjC per-pred + reload/300 (baseline) | 7.4x | 300MB-2GB | No |

## Key Findings

1. `.cpuAndGPU` does NOT prevent ANE usage -- conv ops always get routed to ANE
2. MPS graph compiler cannot handle non-standard op patterns replacing conv (gather, strided slice, reshape+select)
3. CoreML's native batch API (`predictions(fromBatch:)`) manages IOSurface lifecycle better than individual predictions
4. The E5 leak is specific to repeated individual predictions, not to the model architecture itself
