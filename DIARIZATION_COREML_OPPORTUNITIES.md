# Speaker Diarization CoreML Optimization Opportunities

## Current State (After Purge)

### Active Models in Pipeline

1. **Segmentation Model** (`pyannote_segmentation.mlmodelc`)
   - Input: Audio (1, 1, 160000)
   - Output: Speaker probabilities (1, 589, 7)
   - Status: ✅ Production ready

2. **Embedding Preprocessor** (`embedding_preprocessor.mlmodelc`)
   - Input: Audio + Speaker masks
   - Output: Prepared waveforms and masks for embedding
   - Status: ✅ Integrated and optimized

3. **Embedding Model** (`wespeaker.mlmodelc`)
   - Input: Waveforms + Masks
   - Output: Speaker embeddings (3, 256)
   - Bottleneck: Contains 1001 SliceByIndex operations
   - Status: ⚠️ Performance bottleneck (10.5s)

4. **Unified Post-Embedding Model** (`unified_post_embedding.mlmodelc`)
   - Handles cosine distance calculations
   - Computes speaker activities
   - Basic segment filtering
   - Status: ✅ GPU-accelerated

### Removed/Disabled Models

The following models were purged as they were either broken, experimental, or provided no benefit:

- ❌ `batch_frame_extractor.mlpackage.disabled` - Shape mismatch issues
- ❌ `segmentation_postprocessor.mlpackage` - Unused, had softmax bug
- ❌ `optimized_embedding_no_slice` - Demo model, poor quality
- ❌ Various experimental models (v2 versions, etc.)

### Available Source Models

- **segmentation.mlpackage** - Original pyannote segmentation (uncompiled)
- **wespeaker.mlpackage** - Original wespeaker embedding (uncompiled)

These can be modified and recompiled for optimization attempts.

## Current Performance Profile

```
Pipeline Timing Breakdown (17-minute audio):
- Model Compilation: 4.8s (18.2%)
- Segmentation: 8.5s (32.4%)
- Embedding Extraction: 10.5s (46.5%) ← Main bottleneck
- Speaker Clustering: 0.7s (2.7%)
- Post Processing: <0.1s

Total: 26.2s (RTFx: 60x)
DER: 17.7% (optimal)
```

## Optimization Opportunities

### 1. Embedding Model Optimization (High Priority)

**Problem**: 1001 SliceByIndex operations in wespeaker.mlpackage

**Solutions**:
- Batch frame extraction to reduce operations
- Float16 quantization for Neural Engine
- Custom frame extraction layer

**Expected Impact**: 3-5x speedup (10.5s → 2-3s)

### 2. Clean Frame Calculation Integration

**Current**: CPU-based nested loops between segmentation and preprocessing

**Solution**: Add to embedding_preprocessor model:
```python
clean_frames = (sum(masks, axis=1) < 2.0).float()
clean_masks = masks * clean_frames
```

**Expected Impact**: Minor but eliminates CPU-GPU transfer

### 3. Enhanced Unified Post-Embedding

**Current**: Basic distance calculation and filtering

**Add**:
- EMA-based embedding updates
- Advanced clustering decisions
- Speaker tracking logic

**Expected Impact**: Reduce CPU clustering overhead

### 4. Pipeline Fusion Opportunities

**Segmentation → Preprocessor Fusion**:
- Combine segmentation output directly with preprocessing
- Eliminate intermediate CPU processing

**Embedding → Post-Processing Fusion**:
- Direct embedding to clustering pipeline
- Reduce memory transfers

## Implementation Priority

1. **Fix Embedding Bottleneck** (Critical)
   - Modify wespeaker.mlpackage to batch operations
   - Test Float16 quantization
   - Expected: 60x → 180x RTFx

2. **Integrate Clean Frame Calculation** (Easy Win)
   - Update embedding_preprocessor
   - Simple addition to existing model
   - Expected: Minor improvement

3. **Enhance Post-Embedding Model** (Medium)
   - Add EMA updates
   - More sophisticated clustering
   - Expected: Better speaker tracking

## Technical Constraints

- CoreML doesn't support dynamic shapes well
- Custom layers (SincNet) prevent some optimizations
- Must maintain 17.7% DER accuracy
- Compiled models (.mlmodelc) cannot be modified

## Next Steps

1. Attempt wespeaker.mlpackage optimization
2. Profile exact overhead sources in embedding extraction
3. Consider custom Metal implementation for critical paths
4. Investigate CoreML Pipeline API for model chaining