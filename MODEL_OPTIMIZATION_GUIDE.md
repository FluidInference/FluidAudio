# Model Optimization Guide for FluidAudio

This guide outlines optimization techniques to improve the performance of the integrated segmentation and embedding models used in FluidAudio's speaker diarization system.

## Current Model Architecture

### Integrated Segmentation Model
- **Input**: Audio waveform (1, 1, 160000)
- **Output**: Binary speaker masks (1, 589, 3)
- **Operations**: Segmentation → Reshape → Softmax → Argmax → One-hot → Powerset mapping

### Integrated Embedding Model
- **Input**: Waveform (1, 160000) + Binary mask (1, 589, 3)
- **Output**: Speaker embeddings (3, 256)
- **Operations**: Clean mask generation → Waveform duplication → Embedding extraction

## Optimization Techniques

### 1. Model Quantization

Reduce model size and improve inference speed by using lower precision.

```python
# In convert_segmentation_integrated.py or convert_embedding_integrated.py

# Option 1: Float16 precision (2x smaller, ~1.5x faster)
mlmodel = ct.convert(
    traced_model,
    inputs=[...],
    outputs=[...],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14
)

# Option 2: INT8 quantization (4x smaller, 2-3x faster)
from coremltools.optimize.coreml import OpPalettizerConfig, palettize_weights

config = OpPalettizerConfig(
    mode="kmeans",
    nbits=8,
    granularity="per_channel"
)
quantized_model = palettize_weights(mlmodel, config)
```

**Expected improvements**:
- Model size: 50-75% reduction
- Inference speed: 1.5-3x faster
- Minimal accuracy loss (<1% DER increase)

### 2. Operation Fusion

Combine multiple operations into single fused layers.

```python
class FusedSegmentationWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.register_buffer('powerset_mapping', self._create_powerset_mapping())
    
    @torch.jit.script_method
    def fused_postprocess(self, segments_flat: torch.Tensor) -> torch.Tensor:
        # Fuse all operations into a single JIT-compiled function
        segments_3d = segments_flat.view(1, 589, 7)
        probs = F.softmax(segments_3d, dim=-1)
        indices = torch.argmax(probs, dim=-1)
        one_hot = F.one_hot(indices, 7).float()
        masks = torch.matmul(one_hot.view(-1, 7), self.powerset_mapping)
        return masks.view(1, 589, 3)
    
    def forward(self, audio):
        segments = self.base(audio)
        return self.fused_postprocess(segments)
```

### 3. Batch Processing

Process multiple audio chunks simultaneously for better GPU utilization.

```python
class BatchedDiarizationModel(nn.Module):
    def __init__(self, segmentation_model, embedding_model):
        super().__init__()
        self.segmentation = segmentation_model
        self.embedding = embedding_model
    
    def forward(self, audio_batch, return_intermediates=False):
        # audio_batch: (batch_size, 1, 160000)
        batch_size = audio_batch.shape[0]
        
        # Batch segmentation
        segments = self.segmentation(audio_batch)
        
        # Batch embedding with automatic broadcasting
        waveforms = audio_batch.repeat_interleave(3, dim=0)
        masks = segments.view(-1, 589, 1).expand(-1, -1, 256)
        
        embeddings = self.embedding(waveforms, masks)
        embeddings = embeddings.view(batch_size, 3, 256)
        
        if return_intermediates:
            return embeddings, segments
        return embeddings
```

### 4. Model Pruning

Remove unnecessary weights to reduce model size and computation.

```python
import torch.nn.utils.prune as prune

def prune_model(model, sparsity=0.3):
    """
    Apply structured pruning to reduce model size.
    
    Args:
        model: PyTorch model to prune
        sparsity: Fraction of weights to remove (0.3 = 30%)
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            # Structured pruning (remove entire channels)
            prune.ln_structured(
                module, 
                name='weight', 
                amount=sparsity, 
                n=2, 
                dim=0
            )
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    return model

# Usage
pruned_model = prune_model(model, sparsity=0.3)
```

### 5. Hardware-Specific Optimizations

Configure models for optimal performance on Apple Silicon.

```swift
// In DiarizerManager.swift
private func configureModelForHardware() -> MLModelConfiguration {
    let config = MLModelConfiguration()
    
    #if arch(arm64)
    // Apple Silicon optimization
    config.computeUnits = .all  // Use CPU, GPU, and Neural Engine
    config.allowLowPrecisionAccumulationOnGPU = true
    #else
    // Intel Mac optimization
    config.computeUnits = .cpuAndGPU
    #endif
    
    // Enable async prediction for better throughput
    config.parameters[.enableAsyncPrediction] = true
    
    return config
}
```

### 6. Caching and Memoization

Cache intermediate results to avoid redundant computations.

```swift
// In DiarizerManager.swift
private actor ResultCache {
    private var segmentationCache: [Data: [[[Float]]]] = [:]
    private var embeddingCache: [Data: [[Float]]] = [:]
    private let maxCacheSize = 100
    
    func getCachedSegmentation(for audio: Data) -> [[[Float]]]? {
        return segmentationCache[audio]
    }
    
    func cacheSegmentation(_ result: [[[Float]]], for audio: Data) {
        if segmentationCache.count >= maxCacheSize {
            segmentationCache.removeFirst()
        }
        segmentationCache[audio] = result
    }
}
```

### 7. Streaming/Online Processing

Enable real-time processing with minimal latency.

```python
class StreamingDiarizationModel(nn.Module):
    def __init__(self, base_model, context_size=50):
        super().__init__()
        self.model = base_model
        self.context_size = context_size
        self.register_buffer('context', torch.zeros(1, 1, context_size * 160))
    
    def forward(self, audio_chunk, update_context=True):
        # Concatenate with previous context
        audio_with_context = torch.cat([self.context, audio_chunk], dim=-1)
        
        # Process
        output = self.model(audio_with_context)
        
        # Update context for next chunk
        if update_context:
            self.context = audio_chunk[..., -self.context_size * 160:]
        
        return output
```

### 8. Compilation Optimizations

Use CoreML's advanced compilation features.

```python
# Enable all optimizations during conversion
mlmodel = ct.convert(
    traced_model,
    inputs=[...],
    outputs=[...],
    convert_to="mlprogram",
    pass_pipeline=ct.PassPipeline([
        "common::const_elimination",
        "common::dead_code_elimination", 
        "common::loop_invariant_elimination",
        "common::fuse_pad_conv",
        "compression::palettize_weights",
        "compression::prune_weights"
    ])
)

# Save with memory mapping for faster loading
mlmodel.save(
    "optimized_model.mlpackage",
    save_mode=ct.models.MLModel.SaveMode.MEMORY_MAPPED
)
```

### 9. Parallel Processing

Process multiple audio files concurrently.

```swift
// In CLI or batch processing
func processBatch(_ audioFiles: [URL]) async throws -> [DiarizationResult] {
    try await withThrowingTaskGroup(of: DiarizationResult.self) { group in
        // Limit concurrent tasks to avoid memory pressure
        let maxConcurrent = ProcessInfo.processInfo.activeProcessorCount
        
        for (index, file) in audioFiles.enumerated() {
            if index >= maxConcurrent {
                _ = try await group.next()  // Wait for one to complete
            }
            
            group.addTask {
                try await self.processFile(file)
            }
        }
        
        return try await group.reduce(into: []) { $0.append($1) }
    }
}
```

### 10. Model Ensemble Optimization

Combine multiple models efficiently.

```python
class EnsembleDiarizationModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, audio):
        # Run all models in parallel (PyTorch handles this)
        outputs = [model(audio) for model in self.models]
        
        # Weighted average (learnable weights)
        weights = F.softmax(self.ensemble_weights, dim=0)
        output = sum(w * out for w, out in zip(weights, outputs))
        
        return output
```

## Implementation Priority

1. **High Impact, Easy Implementation**:
   - Model quantization (FP16)
   - Hardware-specific configuration
   - Basic caching

2. **High Impact, Medium Complexity**:
   - Operation fusion
   - Batch processing
   - INT8 quantization

3. **Medium Impact, Complex Implementation**:
   - Model pruning
   - Streaming processing
   - Model ensemble

## Performance Benchmarks

| Optimization | Speed Improvement | Size Reduction | DER Impact |
|--------------|------------------|----------------|------------|
| FP16 Quantization | 1.5x | 50% | < 0.5% |
| INT8 Quantization | 2-3x | 75% | < 1% |
| Operation Fusion | 1.2x | 0% | 0% |
| Batch Processing | 2-4x* | 0% | 0% |
| Pruning (30%) | 1.3x | 30% | < 1% |

*When processing multiple files

## Testing Optimizations

```bash
# Benchmark original model
swift run fluidaudio benchmark --output baseline.json

# Apply optimizations and re-benchmark
swift run fluidaudio benchmark --output optimized.json

# Compare results
swift run fluidaudio compare-benchmarks baseline.json optimized.json
```

## Best Practices

1. **Always test accuracy** after applying optimizations
2. **Profile before and after** to measure actual improvements
3. **Start with lossless optimizations** (fusion, caching)
4. **Apply lossy optimizations carefully** (quantization, pruning)
5. **Consider deployment target** (iOS vs macOS, device capabilities)

## Next Steps

1. Implement FP16 quantization for immediate 50% size reduction
2. Add operation fusion for cleaner computation graphs
3. Enable batch processing in CLI for multi-file scenarios
4. Test INT8 quantization impact on accuracy
5. Implement streaming mode for real-time applications

## References

- [CoreML Optimization](https://coremltools.readme.io/docs/optimization-overview)
- [PyTorch Mobile Optimization](https://pytorch.org/tutorials/recipes/script_optimized.html)
- [Apple Neural Engine](https://developer.apple.com/documentation/coreml/optimizing-core-ml-performance)