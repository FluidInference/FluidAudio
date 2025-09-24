# Last Chunk Handling in FluidAudio ASR

## Overview

Implementation of NVIDIA RNNT streaming patterns for last chunk handling in FluidAudio's Token-and-Duration Transducer (TDT) decoder.

## NVIDIA Reference Implementation

Based on `speech_to_text_streaming_infer_rnnt.py` from NeMo Parakeet TDT documentation:

### Key Components from NVIDIA

1. **Last Chunk Detection**
```python
# NVIDIA approach
is_last_chunk_batch = chunk_length >= rest_audio_lengths
is_last_chunk = right_sample >= audio_batch.shape[1]
```

2. **Variable Chunk Length Handling**
```python
chunk_lengths_batch = torch.where(
    is_last_chunk_batch,
    rest_audio_lengths,  # Use remaining audio length
    torch.full_like(rest_audio_lengths, fill_value=chunk_length),
)
```

3. **Buffer Processing with Last Chunk Flag**
```python
buffer.add_audio_batch_(
    audio_batch[:, left_sample:right_sample],
    audio_lengths=chunk_lengths_batch,
    is_last_chunk=is_last_chunk,
    is_last_chunk_batch=is_last_chunk_batch,
)
```

4. **State Management Across Chunks**
```python
if current_batched_hyps is None:
    current_batched_hyps = chunk_batched_hyps
else:
    current_batched_hyps.merge_(chunk_batched_hyps)
```

## FluidAudio Implementation

### 1. Last Chunk Detection
```swift
// ChunkProcessor.swift
let isLastChunk = (centerStart + centerSamples) >= audioSamples.count
```

### 2. Flag Propagation Through Pipeline
```swift
// Method signatures updated throughout
func executeMLInferenceWithTimings(
    _ paddedAudio: [Float],
    isLastChunk: Bool = false
) async throws -> (tokens: [Int], timestamps: [Int], encoderSequenceLength: Int)

func decodeWithTimings(
    encoderOutput: MLMultiArray,
    isLastChunk: Bool = false
) async throws -> (tokens: [Int], timestamps: [Int])
```

### 3. Last Chunk Finalization Logic
```swift
// TdtDecoder.swift - Based on NVIDIA consecutive blank pattern
if isLastChunk {
    var additionalSteps = 0
    var consecutiveBlanks = 0
    let maxConsecutiveBlanks = 2  // Exit after 2 blanks in a row
    
    while additionalSteps < maxSymbolsPerStep && consecutiveBlanks < maxConsecutiveBlanks {
        // Use last valid encoder frame if beyond bounds
        let frameIndex = min(timeIndices, encoderFrames.count - 1)
        let encoderStep = encoderFrames[frameIndex]
        
        // Continue processing beyond encoder length
        // Exit naturally when decoder produces consecutive blanks
    }
}
```

### 4. State Finalization
```swift
// TdtDecoderState.swift
mutating func finalizeLastChunk() {
    predictorOutput = nil  // Clear cache
    timeJump = nil        // No more chunks
    // Keep lastToken and LSTM states for context
}
```

## FluidAudio Constraints

### 1. Model Limitations
- **15-second hard limit**: Models cannot process audio longer than 240,000 samples (15s at 16kHz)
- **No single-pass mode**: Audio > 15s must use ChunkProcessor
- **Fixed model architecture**: Cannot modify Core ML models at runtime

### 2. Memory Constraints
- **ANE alignment required**: All arrays must be ANE-aligned for optimal performance
- **Zero-copy operations**: Minimize memory allocations during streaming
- **State persistence**: LSTM states must be maintained across chunk boundaries

### 3. Threading Requirements
- **No @unchecked Sendable**: All code must be properly thread-safe
- **Actor-based concurrency**: Use Swift actors for thread safety
- **Persistent decoder states**: States maintained across async boundaries

## Critical Implementation Points

### 1. Chunk Boundary Handling
```swift
// Current chunking parameters (frame-aligned)
centerSeconds: 11.2     // 140 encoder frames
leftContextSeconds: 1.6  // 20 encoder frames  
rightContextSeconds: 1.6 // 20 encoder frames
// Total: 14.4s (within 15s limit)
```

### 2. Time Index Management
- **timeIndices**: Current position in encoder frames
- **timeJump**: Tracks processing beyond current chunk for streaming
- **Bounds checking**: Always clamp indices to prevent crashes

### 3. Decoder State Continuity
- **lastToken**: Maintains linguistic context between chunks
- **predictorOutput**: Cached LSTM output for performance
- **hiddenState/cellState**: LSTM memory across boundaries

### 4. Token Emission Rules
- Only emit tokens beyond `startFrameOffset` to avoid duplicates
- Update decoder state regardless of emission for context preservation
- Force advancement after `maxSymbolsPerStep` to prevent infinite loops

## Processing Flow
```
Audio Input (>15s)
    ↓
ChunkProcessor
    ├─ Chunk 0: isLastChunk=false
    └─ Chunk N: isLastChunk=true
        ↓
TdtDecoder.decodeWithTimings(isLastChunk: true)
    ├─ Main decoding loop
    └─ Last chunk finalization (if isLastChunk=true)
        ├─ Continue beyond encoder frames
        ├─ Use consecutive blank detection
        └─ Natural termination
```

## Code Locations
- `Sources/FluidAudio/ASR/ChunkProcessor.swift`: Chunk detection logic
- `Sources/FluidAudio/ASR/TDT/TdtDecoder.swift`: Finalization logic (lines 331-416) 
- `Sources/FluidAudio/ASR/TDT/TdtDecoderState.swift`: State management
- `Sources/FluidAudio/ASR/AsrTranscription.swift`: Pipeline routing
