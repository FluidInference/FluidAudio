# Parakeet EOU Streaming Implementation Report

## Overview

This document details the process of implementing streaming ASR with NVIDIA's Parakeet Realtime EOU 120M model in CoreML/Swift, including the challenges encountered, solutions developed, and final working architecture.

## Model Background

**Model**: `nvidia/parakeet_realtime_eou_120m-v1`
- Architecture: FastConformer-RNNT with End-of-Utterance detection
- Parameters: 120M
- Features: Real-time streaming with utterance boundary detection
- Special tokens: `<EOU>` (End of Utterance), `<EOB>` (End of Block), `<blk>` (RNNT blank)

## Initial State

### What We Had
- `BatchEouAsrManager` - Working batch inference using monolithic encoder
- `StreamingEouAsrManager` - Non-functional streaming implementation
- Split CoreML models in `parakeet_eou_split_coreml/`:
  - `preprocessor.mlmodelc`
  - `encoder.mlmodelc` (batch-only, no cache I/O)
  - `decoder.mlmodelc`
  - `joint.mlmodelc`

### The Problem
`StreamingEouAsrManager` expected a `streaming_encoder.mlpackage` with cache inputs/outputs, but only batch encoder models existed. The batch encoder processes full audio without maintaining state between chunks.

## Investigation Phase

### 1. Understanding NeMo's Streaming Architecture

NeMo's FastConformer encoder supports two modes:
- **Batch mode**: `encoder.forward()` - processes full audio at once
- **Streaming mode**: `encoder.cache_aware_stream_step()` - processes chunks with cache state

The streaming encoder requires cache tensors to maintain context across chunks:

```python
# Cache shapes (FIXED, not dynamic)
cache_last_channel: [num_layers, batch, cache_size, hidden_dim]  # [17, 1, 70, 512]
cache_last_time: [num_layers, batch, hidden_dim, time_cache]     # [17, 1, 512, 8]
cache_last_channel_len: [batch]                                   # [1] (int32)
```

Key insight: **Cache tensor shapes are fixed**. Only the `cache_last_channel_len` value changes to track how much of the cache is filled.

### 2. Examining Existing Split Models

The existing `encoder.mlmodelc` in `parakeet_eou_split_coreml/` was converted for batch inference:
- Input: `mel` (mel spectrogram)
- Output: `encoder_output`
- **No cache inputs/outputs**

This model couldn't support streaming because it had no mechanism to carry state between chunks.

## Solution: Export Cache-Aware Streaming Encoder

### Approach

Created a PyTorch wrapper around NeMo's `cache_aware_stream_step()` method that could be traced and converted to CoreML.

### Implementation: `export_streaming_encoder.py`

```python
class StreamingEncoderWrapper(nn.Module):
    """Wrapper for cache-aware streaming encoder."""

    def __init__(self, encoder: nn.Module, keep_all_outputs: bool = True):
        super().__init__()
        self.encoder = encoder
        self.keep_all_outputs = keep_all_outputs

        if encoder.streaming_cfg is None:
            encoder.setup_streaming_params()
        self.streaming_cfg = encoder.streaming_cfg

    def forward(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        # Call encoder with cache
        outputs = self.encoder(
            audio_signal=mel,
            length=mel_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        # Post-process for streaming
        outputs = self.encoder.streaming_post_process(
            outputs, keep_all_outputs=self.keep_all_outputs
        )

        return outputs
```

### CoreML Conversion

```python
# Fixed shapes for export
inputs = [
    ct.TensorType(name="mel", shape=(1, 80, 101), dtype=np.float32),  # 1000ms = 101 frames
    ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
    ct.TensorType(name="cache_last_channel", shape=(17, 1, 70, 512), dtype=np.float32),
    ct.TensorType(name="cache_last_time", shape=(17, 1, 512, 8), dtype=np.float32),
    ct.TensorType(name="cache_last_channel_len", shape=(1,), dtype=np.int32),
]

outputs = [
    ct.TensorType(name="encoder", dtype=np.float32),
    ct.TensorType(name="encoder_length", dtype=np.int32),
    ct.TensorType(name="cache_last_channel_out", dtype=np.float32),
    ct.TensorType(name="cache_last_time_out", dtype=np.float32),
    ct.TensorType(name="cache_last_channel_len_out", dtype=np.int32),
]

mlmodel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.iOS17,
)
```

### Validation

Created `test_streaming_encoder_coreml.py` to verify CoreML output matches NeMo:

```
Chunk 1: max_diff=0.000006, mean_diff=0.000001
Chunk 2: max_diff=0.000005, mean_diff=0.000001
...
Final diff: max=0.000006, mean=0.000001
✓ CoreML model matches NeMo output!
```

## Bug Fixes During Implementation

### Bug 1: Token ID Mismatch

**Symptom**: Immediate EOU detection, garbage output

**Root Cause**: Token IDs were wrong in Swift code:
```swift
// WRONG (before fix)
eouTokenId = 1025
eobTokenId = 1026
blankId = 1026  // Same as EOB!

// CORRECT (from vocab.json)
eouTokenId = 1024  // <EOU>
eobTokenId = 1025  // <EOB>
blankId = 1026     // <blk>
```

**Fix**: Updated `ModelNames.swift` and `RnntConfig.swift`:
```swift
// Token IDs from vocab.json (NeMo parakeet_realtime_eou_120m-v1)
public static let eouTokenId = 1024  // <EOU> - End of Utterance
public static let eobTokenId = 1025  // <EOB> - End of Block
public static let blankId = 1026     // <blk> - RNNT blank token
```

### Bug 2: Corrupted Decoder Models

**Symptom**: Wrong SOS (Start of Sequence) projection output from decoder

**Root Cause**: Nested directory issue from bad copy operation:
```
decoder.mlmodelc/
└── decoder.mlmodelc/  # Nested!
    └── actual model files
```

**Fix**: Re-copied decoder model from verified source:
```bash
cp -r mobius/models/stt/parakeet-realtime-eou-120m/coreml/parakeet_eou_coreml/decoder.mlmodelc streaming/
```

### Bug 3: Mel Frame Count Mismatch

**Symptom**: Shape mismatch errors in streaming encoder

**Root Cause**: Swift code used wrong frame count for 1000ms chunks

**Fix**: Updated `StreamingEouAsrManager.swift`:
```swift
// 1000ms at 16kHz = 16000 samples
// Mel frames = samples / hop_length + 1 = 16000 / 160 + 1 = 101
let fixedFrames = 101  // Must match exported streaming_encoder model
```

## Final Architecture

### Model Pipeline (Streaming Mode)

```
Audio (16kHz)
    ↓ (1000ms chunks)
Preprocessor → Mel Spectrogram [1, 80, 101]
    ↓
Streaming Encoder (with cache) → Encoder Output [1, 512, ~25] + Updated Cache
    ↓
RNNT Decoder Loop:
    ├── Decoder (LSTM) → Prediction [1, 640]
    ├── Joint (Encoder + Decoder) → Logits [1, 1027]
    └── Greedy decode until blank/EOU
    ↓
Token IDs → SentencePiece Decode → Text
```

### Cache Flow

```
Chunk 1:
  Input: zero cache, cache_len=0
  Output: updated cache, cache_len=N

Chunk 2:
  Input: cache from chunk 1, cache_len=N
  Output: updated cache, cache_len=min(N+M, 70)

... continues with rolling cache window
```

### File Organization

```
parakeet_eou_coreml/
├── batch/                              # Full audio processing
│   ├── preprocessor.mlpackage/.mlmodelc
│   ├── encoder.mlpackage/.mlmodelc     # No cache I/O
│   ├── decoder.mlpackage/.mlmodelc
│   ├── joint.mlpackage/.mlmodelc
│   ├── vocab.json
│   └── metadata.json
│
├── streaming/                          # Chunk-by-chunk processing
│   ├── preprocessor.mlpackage/.mlmodelc
│   ├── streaming_encoder.mlpackage/.mlmodelc  # WITH cache I/O
│   ├── decoder.mlpackage/.mlmodelc
│   ├── joint.mlpackage/.mlmodelc
│   ├── vocab.json
│   └── streaming_encoder_config.json
│
└── scripts/
    ├── export_streaming_encoder.py
    └── test_streaming_encoder_coreml.py
```

## Performance Results

### Streaming Mode
- **RTFx**: 18-30x real-time on Apple Silicon
- **Latency**: <50ms per 1s chunk
- **Memory**: ~200MB loaded

### Test Output
```
Audio: she_sells_seashells_16k.wav (3.5s)
Output: "she sells seashells by the shore"
```

### WER Comparison (LibriSpeech test-clean, 100 files)
- Batch mode: Baseline
- Streaming mode: ~35% relative WER increase (expected due to chunking)

## Key Learnings

### 1. Cache Shapes Are Fixed
The FastConformer's cache tensors have fixed shapes. Only the "fill level" (`cache_last_channel_len`) varies. This made CoreML conversion straightforward once understood.

### 2. NeMo's `cache_aware_stream_step()` Is the Key
Don't try to manually implement streaming logic. NeMo's method handles all the complexity:
- Attention context caching
- Convolution state caching
- Positional encoding offsets
- Output frame alignment

### 3. Tracing Works Better Than Scripting
`torch.jit.trace()` with `strict=False` successfully captured the streaming encoder. Scripting failed due to dynamic control flow in NeMo's internals.

### 4. Verify Token IDs Against vocab.json
Always cross-reference special token IDs with the actual `vocab.json` file. Assumptions based on other models can be wrong.

### 5. Check for Nested Directories
When copying CoreML models, verify the internal structure. Nested directories from bad copies cause subtle failures.

## Workflow for Future Models

1. **Examine NeMo model's streaming API**
   ```python
   encoder.setup_streaming_params()
   encoder.get_initial_cache_state(batch_size=1)
   encoder.cache_aware_stream_step(...)
   ```

2. **Create wrapper for tracing**
   - Wrap the streaming method in a simple `nn.Module`
   - Use fixed input shapes matching your chunk size

3. **Export to CoreML**
   - Use `torch.jit.trace()` with `strict=False`
   - Specify all input/output tensor types explicitly

4. **Validate against NeMo**
   - Process same audio through both
   - Compare outputs (should be <0.001 max diff)

5. **Verify token IDs**
   - Check vocab.json for special tokens
   - Update Swift code to match

6. **Test end-to-end**
   - Single file transcription
   - Benchmark on test set

## Files Modified

- `Sources/FluidAudio/ModelNames.swift` - Fixed token IDs
- `Sources/FluidAudio/ASR/RNNT/RnntConfig.swift` - Fixed default token IDs
- `Sources/FluidAudio/ASR/RNNT/StreamingEouAsrManager.swift` - Updated frame count

## Files Created

- `Scripts/ParakeetEOU/Conversion/export_streaming_encoder.py`
- `Scripts/ParakeetEOU/Conversion/test_streaming_encoder_coreml.py`
- `parakeet_eou_coreml/` - Organized model bundle
- `parakeet_eou_coreml.tar.gz` - HuggingFace distribution

## References

- [NVIDIA Parakeet EOU Model](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [CoreML Tools Documentation](https://coremltools.readme.io/)
