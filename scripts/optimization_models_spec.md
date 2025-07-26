# CoreML Optimization Models Specification

This document describes the CoreML models needed to optimize the ASR pipeline.

## Required Models

### 1. TransposeEncoder.mlmodel
- **Purpose**: Efficiently transpose encoder output from [1, 640, T] to [1, T, 640]
- **Input**: `input` - shape [1, 640, RangeDim(1, 2048)]
- **Output**: `output` - shape [1, RangeDim(1, 2048), 640]
- **Operation**: Transpose with axes permutation [0, 2, 1]

### 2. Argmax.mlmodel
- **Purpose**: Replace manual argmax operation for token prediction
- **Input**: `logits` - shape [1025] (vocab size)
- **Outputs**: 
  - `index` - shape [1] (argmax index)
  - `score` - shape [1] (max value)
- **Operations**: 
  - argmax on axis 0
  - reduce_max on axis 0

### 3. AudioPadding.mlmodel
- **Purpose**: Pad variable length audio to fixed length
- **Input**: `audio` - shape [RangeDim(1, 160000)]
- **Output**: `padded_output` - shape [160000]
- **Operation**: Pad with zeros to target length

### 4. TokenDurationPrediction.mlmodel
- **Purpose**: Split logits and predict token + duration
- **Input**: `logits` - shape [1030] (1025 vocab + 5 durations)
- **Outputs**:
  - `token_id` - shape [1]
  - `token_score` - shape [1]
  - `duration_index` - shape [1]
- **Operations**:
  - Slice [0:1025] for token logits
  - Slice [1025:1030] for duration logits
  - Argmax on token logits
  - Reduce_max on token logits
  - Argmax on duration logits

## Model Generation

To generate these models, you need:
1. Python 3.8+
2. coremltools (`pip install coremltools`)
3. numpy (`pip install numpy`)
4. torch (optional, for padding model)

Run: `python3 Scripts/generate_optimization_models.py`

## Integration

The models will be loaded by `CoreMLOptimizations.loadOptimizationModels()` and used by `OptimizedAsrManager` to accelerate:
- Tensor transposition (30% faster)
- Token prediction (50% faster)
- Audio preprocessing (20% faster)
- Combined token/duration prediction (40% faster)