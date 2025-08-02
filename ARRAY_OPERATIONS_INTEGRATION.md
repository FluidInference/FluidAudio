# Array Operations Integration for CoreML Models

This document outlines all the array operations that can be integrated directly into the CoreML models to eliminate Swift-side array manipulations.

## Current Array Operations in Swift

### Segmentation Model Operations

Currently in `SegmentationProcessor.swift`:

1. **Reshape Operation** (lines 30-43)
   - Converts flat output (1, 4123) to 3D array (1, 589, 7)
   - Manual iteration through frames and combinations

2. **Powerset Conversion** (lines 48-86)
   - Maps 7 combinations to 3 binary speaker masks
   - Uses predefined powerset mapping

### Embedding Model Operations

Currently in `EmbeddingExtractor.swift`:

1. **Clean Frame Calculation** (lines 22-29)
   - Sums speaker activities per frame
   - Creates mask where sum < 2.0 (no overlap)

2. **Clean Mask Generation** (lines 31-50)
   - Applies clean frame mask to speaker segments
   - Transposes from [batch][frame][speaker] to [speaker][frame]

3. **Waveform Duplication** (lines 61-65)
   - Duplicates audio chunk for each speaker
   - Creates (3, 160000) tensor from single audio

4. **Array Conversion** (lines 125-142)
   - Converts MLMultiArray to [[Float]]
   - Handles stride calculations

## Operations That Can Be Integrated

### For Segmentation Model

| Operation | Current Location | Integration Method |
|-----------|------------------|-------------------|
| Reshape (4123→589×7) | Swift manual loop | Model output layer |
| Softmax normalization | Not applied | Add softmax layer |
| Argmax selection | Part of powerset | Add argmax layer |
| One-hot encoding | Part of powerset | Add one-hot layer |
| Powerset mapping | Swift function | Matrix multiplication layer |
| Final reshape | Swift arrays | Model output shape |

### For Embedding Model

| Operation | Current Location | Integration Method |
|-----------|------------------|-------------------|
| Speaker sum | Swift reduce | Sum reduction layer |
| Clean mask (< 2.0) | Swift conditional | Comparison layer |
| Mask broadcasting | Swift loops | Broadcast multiply |
| Array transpose | Swift nested loops | Transpose layer |
| Waveform broadcast | Swift loops | Tile/broadcast layer |

## Benefits of Integration

### Performance
- **GPU Acceleration**: All operations run on GPU/Neural Engine
- **Parallelization**: Matrix operations are highly parallel
- **Memory Efficiency**: No intermediate CPU arrays

### Code Simplicity
- **Before**: 100+ lines of array manipulation
- **After**: Single model.predict() call

### Optimization
- **Fusion**: CoreML can fuse operations
- **Quantization**: Can quantize entire pipeline
- **Pruning**: Can prune unnecessary operations

## Implementation Approaches

### Approach 1: Modify During Conversion
Modify the PyTorch models during CoreML conversion to include post-processing layers.

```python
# Add layers during conversion
def add_postprocessing_layers(pytorch_model):
    return nn.Sequential(
        pytorch_model,
        ReshapeLayer(589, 7),
        PowersetConverter(),
        BinaryMaskOutput()
    )
```

### Approach 2: Create Wrapper Models
Create new models that wrap existing ones with post-processing.

```python
class SegmentationWithPostProcessing(nn.Module):
    def __init__(self, base_model):
        self.base = base_model
        self.powerset = PowersetLayer()
    
    def forward(self, x):
        x = self.base(x)
        x = x.reshape(1, 589, 7)
        return self.powerset(x)
```

### Approach 3: Chain Multiple Models
Use the created post-processing models in sequence (current solution).

## Recommended Architecture

### Integrated Segmentation Pipeline
```
Input Audio (1, 1, 160000)
    ↓
Segmentation CNN/RNN
    ↓
Reshape Layer (1, 589, 7)
    ↓
Softmax Activation
    ↓
Argmax + One-hot
    ↓
Powerset Matrix Multiply
    ↓
Output Binary Masks (1, 589, 3)
```

### Integrated Embedding Pipeline
```
Input: Waveform (3, 160000) + Mask (3, 589)
    ↓
Clean Mask Generation
    - Sum reduction
    - Threshold < 2.0
    - Broadcast multiply
    ↓
Feature Extraction
    ↓
Embedding Computation
    ↓
Output: Clean Embeddings (3, 256)
```

## Migration Path

1. **Phase 1**: Use separate post-processing models (✓ Completed)
2. **Phase 2**: Create integrated models for testing
3. **Phase 3**: Modify original model conversion
4. **Phase 4**: Deploy fully integrated models

## Code Examples

### Current Swift Code (Complex)
```swift
// Many lines of array manipulation
for f in 0..<frames {
    for c in 0..<combinations {
        let index = f * combinations + c
        segments[0][f][c] = segmentOutput[index].floatValue
    }
}
return powersetConversion(segments)
```

### New Swift Code (Simple)
```swift
// Single model call
let speakerMasks = integratedModel.predict(audio)
return speakerMasks
```

## Conclusion

Moving array operations into CoreML models provides:
- 🚀 **Performance**: GPU/Neural Engine acceleration
- 🎯 **Simplicity**: Cleaner Swift code
- 🔧 **Maintainability**: Logic in one place
- ⚡ **Optimization**: Better compiler optimizations

The three created models (`segmentation_reshaper`, `powerset_converter`, `clean_mask_generator`) demonstrate this approach and can be further integrated into the base models for maximum efficiency.