# Failed Optimization Attempts in FluidAudio

Here's a comprehensive list of the optimization attempts that failed to deliver significant improvements or were abandoned during the development of the FluidAudio speaker diarization system:

## 1. Segmentation Model Post-Processing Integration

**Approach**: Moving Swift post-processing operations to CoreML for GPU acceleration
**Status**: Implemented but later removed
**Issues**:
- SincNet layer in pyannote model prevented full integration
- Softmax applied to log probabilities (incorrect implementation)
- Segmentation was not the primary bottleneck (only 32.4% of pipeline time)

**Files Created/Deleted**:
- ❌ `segmentation_postprocessor.mlpackage` - Deleted (had softmax bug, never integrated)
- ❌ `segmentation_postprocessor_fixed.mlpackage` - Never created
- ❌ `SegmentationProcessorOptimized.swift` - Never created

## 2. Embedding Model Direct Optimization

**Approach**: Modifying the wespeaker model to eliminate 1001 SliceByIndex operations
**Status**: Failed due to compiled model format
**Issues**:
- Couldn't modify compiled .mlmodelc files
- Access to original PyTorch source required
- Custom layers (SincNet) complicated conversion

**Scripts Purged**:
- ❌ `convert_wespeaker_proper.py`
- ❌ `download_and_convert_wespeaker.py`
- ❌ `extract_wespeaker_weights.py`

## 3. Demo Optimized Model Approach

**Approach**: Creating simplified version of wespeaker without SliceByIndex operations
**Status**: Created but deleted due to poor performance
**Issues**:
- DER jumped to 50.3% (from 17.8%)
- Simplified demo lacked actual weights
- Poor quality made it unusable

**Models Deleted**:
- ❌ `optimized_embedding_no_slice.mlpackage`

## 4. Preprocessing Pipeline Optimization

**Approach**: Creating specialized preprocessing models for wespeaker
**Status**: Created but deleted due to compatibility issues
**Issues**:
- Shape mismatch issues made them unusable
- Incompatible with existing wespeaker model

**Models Deleted**:
- ❌ `wespeaker_preprocessing.mlpackage`
- ❌ `wespeaker_preprocessing_flexible.mlpackage`

## 5. Batch Frame Extractors

**Approach**: Creating models to batch extract frames before feeding to wespeaker
**Status**: Implemented but minimal impact due to architecture mismatch
**Issues**:
- Wespeaker still expected waveform input, not pre-extracted frames
- Integration provided minimal benefits (~5% improvement)
- Would require wespeaker model retraining to accept frames directly

**Models Deleted**:
- ❌ `batch_frame_extractor_conv1d.mlpackage`
- ❌ `batch_frame_extractor_dynamic.mlpackage`

## 6. Wespeaker Variant Models

**Approach**: Creating multiple optimized versions of wespeaker
**Status**: All experimental versions removed
**Issues**:
- Compilation issues
- Performance degradation
- Architecture incompatibilities

**Models Deleted**:
- ❌ `wespeaker_float16.mlpackage`
- ❌ `wespeaker_optimized.mlpackage`
- ❌ `wespeaker_optimized_flexible.mlpackage`
- ❌ `wespeaker_realistic_optimized.mlpackage`
- ❌ `wespeaker_optimized.onnx`

## 7. Powerset Decoders

**Approach**: Optimizing post-processing for segmentation
**Status**: Deleted during cleanup
**Issues**:
- Minimal impact on overall pipeline
- Added complexity for little gain

**Models Deleted**:
- ❌ `powerset_decoder.mlpackage`
- ❌ `powerset_decoder_old.mlpackage`

## 8. CoreML Pipeline Approach

**Approach**: Creating a 3-model pipeline (wespeaker → embedding_renamer → unified_post_embedding)
**Status**: Catastrophic performance degradation
**Issues**:
- Before: 60x RTFx (17.5s for ES2004a)
- After: 24x RTFx (43.7s for ES2004a)
- Pipeline overhead: ~19s (373% overhead!)

**Root Cause**: CoreML pipeline infrastructure adds massive overhead for model chaining

## 9. Direct Model Quantization

**Approach**: Attempting to quantize compiled .mlmodelc files
**Status**: Failed due to compiled model format
**Issues**:
- Cannot quantize compiled .mlmodelc files
- Need original .mlpackage for quantization

## 10. PyTorch Model Conversion

**Approach**: Loading original pyannote model with HF token for direct optimization
**Status**: Failed due to model architecture complexities
**Issues**:
- PyTorch Lightning wrapper complications
- Custom SincNet layer prevents CoreML conversion
- ModuleList handling complexities
- ONNX conversion not supported in coremltools

**Scripts Created but Failed**:
- ❌ `optimize_pyannote_segmentation.py`
- ❌ `optimize_pyannote_via_onnx.py`
- ❌ `optimize_pyannote_direct.py`
- ❌ `optimize_pyannote_torchscript.py`
- ❌ `optimize_pyannote_final.py`

## 11. Embedding Modes System

**Approach**: Creating flexible embedding extraction modes (skip, fast, cached, full)
**Status**: Implemented but later purged during code cleanup
**Issues**:
- Added complexity without sufficient performance gains
- Skip mode degraded DER to ~56%
- Cached mode useful only in specific scenarios

**Files Deleted**:
- ❌ `FastEmbeddingExtractor.swift`
- ❌ `ConcurrentEmbeddingExtractor.swift`
- ❌ `EmbeddingCache.swift`

## 12. Merged Embedding + Unified Model

**Approach**: Single inference for embedding extraction + post-processing
**Status**: Model compilation hangs (too complex for CoreML compiler)
**Issues**:
- When working: ~2x slower than separate models (36x vs 60x RTFx)
- Pipeline overhead negates single-inference benefits

**Status**: Disabled in production

## 13. INT8 Quantization (Initial Attempt)

**Approach**: Used xcrun coremlcompiler with --compute-precision Mixed
**Status**: Failed to provide actual quantization
**Issues**:
- No actual quantization occurred
- Weights remained 27MB
- No performance improvement

**Note**: Later succeeded with 8-bit palettization, but this initial approach failed

## 14. Unified Fbank Model

**Approach**: Complete end-to-end diarization in one model
**Status**: Initial implementation had severe accuracy issues (78.8% DER vs 17.8% expected)
**Issues**:
- Swift fbank implementation didn't match Python's torchaudio.compliance.kaldi.fbank
- Infinite loop during model loading
- Complex integration challenges

**Status**: Under investigation, not deployed

## Key Takeaways from Failed Attempts

### Model Architecture Constraints:
- Cannot modify compiled .mlmodelc files
- Custom layers (SincNet) prevent direct CoreML conversion
- 1001 SliceByIndex operations remain fundamental bottleneck

### CoreML Limitations:
- Pipelines add massive overhead (373%)
- Limited dynamic shape support
- Shape flexibility requires careful design
- Complex models may not compile

### Integration Challenges:
- Model interfaces must align for optimization benefits
- Preprocessing must match exactly (e.g., fbank extraction)
- Architecture mismatch limits optimization potential

### Optimization Trade-offs:
- Some optimizations reduced model size but not speed
- Others increased complexity without sufficient performance gains
- Architectural simplicity often outperforms forced unification

Despite these failures, the project eventually achieved success with the combination of INT8 quantization and the OptimizedWeSpeaker wrapper, improving performance by 39% while maintaining accuracy.

# Analysis of INT8 Quantization Script

This document analyzes the `quantize_wespeaker_int8.py` script which provides a comprehensive approach to quantizing the CoreML models to INT8 precision.

## Key Features

### 1. Multiple Quantization Methods

The script tries three different approaches in sequence if earlier attempts fail:

- **Method 1**: Weight quantization using `quantize_weights`
- **Method 2**: Compute precision optimization for MLProgram models
- **Method 3**: Manual/experimental quantization as fallback

### 2. Comprehensive Error Handling

- Gracefully handles failures in each quantization method
- Provides detailed error messages and diagnostics
- Falls back to alternative approaches automatically

### 3. Size Reduction Tracking

- Calculates and reports model size reduction
- Helps evaluate storage benefits of quantization

### 4. Testing Framework

- Automatically generates a test script (`test_quantized_models.py`)
- Tests both wespeaker and segmentation models
- Measures inference time for quantized models

### 5. Integration Guide

- Creates a Swift integration guide (`INT8_INTEGRATION_GUIDE.md`)
- Provides code snippets for loading quantized models
- Includes performance monitoring and rollback options

## How This Differs from Previous Attempts

### More Comprehensive Approach

- Previous attempts may have focused on a single quantization method
- This script tries multiple approaches systematically

### Better Error Recovery

- Handles failures gracefully and tries alternative methods
- Provides clear diagnostics about why each approach may fail

### Integration Support

- Includes guidance for Swift integration
- Creates testing utilities automatically

### Performance Expectations Management

- Explicitly warns about potential accuracy degradation
- Sets realistic expectations (DER increase from 17.8% to 25-30%)

## Potential Advantages

### 1. Wider Compatibility

- The multiple approaches make it more likely to work with different model types
- Handles both MLProgram and neural network models

### 2. Easier Testing and Integration

- Automated test script generation
- Clear integration guidelines for Swift

### 3. Fallback Mechanisms

- If optimal methods fail, falls back to more aggressive approaches
- Increases chances of producing a working quantized model

## Potential Limitations

### 1. Accuracy Concerns

- The script warns about significant accuracy degradation
- DER could increase from 17.8% to 25-30%, which may be unacceptable

### 2. Compatibility Issues

- Some methods require specific CoreML versions or model formats
- May not work with all model architectures

### 3. Linear Quantization Only

- Doesn't explore more advanced quantization approaches like k-means clustering
- The successful INT8 + OptimizedWeSpeaker approach mentioned in the documents used palettization rather than linear quantization

## How This Could Be Integrated with Successful Approaches

This script could potentially be enhanced by:

### 1. Adding Palettization

- Incorporate the successful 8-bit palettization approach
- Use k-means clustering instead of linear quantization

### 2. Combining with OptimizedWeSpeaker

- Add integration with the OptimizedWeSpeaker wrapper
- Implement the buffer reuse and direct memory operations

### 3. Selective Quantization

- Add options to quantize only specific layers
- Preserve critical layers at higher precision

## Comparison with Successful INT8 Implementation

The documents indicate that the successful INT8 quantization approach used:

- **8-bit palettization** (k-means clustering with 256-entry lookup tables)
- **OptimizedWeSpeaker wrapper** for efficient processing
- Achieved **39% performance improvement** (RTF: 83.22x)
- Maintained **excellent accuracy** (DER: 17.4%)

In contrast, this script appears to use:

- **Linear quantization** methods
- **No specialized wrapper** optimization
- Warns about **accuracy degradation** (DER: 25-30%)

## Conclusion

This script represents a more systematic approach to quantization than some previous attempts, though it appears to use linear quantization rather than the palettization method that was ultimately successful when combined with the OptimizedWeSpeaker wrapper. The key difference is that the successful approach used 8-bit palettization (k-means clustering) which preserves accuracy better than linear quantization while still achieving significant performance improvements.
