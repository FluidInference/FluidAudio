# FluidAudio - Claude Code Instructions

## Project Overview

FluidAudio is a speaker diarization system for Apple platforms using Core ML models. The system processes audio to identify "who spoke when" by segmenting audio and clustering speaker embeddings.

## Critical Development Rules

### ⚠️ NEVER USE "unchecked Sendable"

- **DO NOT** use `@unchecked Sendable` under any circumstances
- Always properly implement thread-safe code with proper synchronization
- Use actors, `@MainActor`, or proper locking mechanisms instead
- If you encounter Sendable conformance issues, fix them properly rather than bypassing with `@unchecked`

## Current Performance Status

- **Achieved**: 17.7% DER
- **Target**: < 30% DER
- **Competitive with**: State-of-the-art research (Powerset BCE: 18.5%)

### Optimal Configuration

```swift
DiarizerConfig(
    clusteringThreshold: 0.7,     // Optimal value: 17.7% DER
    minDurationOn: 1.0,           // Minimum speaker segment duration
    minDurationOff: 0.5,          // Minimum silence between speakers
    minActivityThreshold: 10.0,   // Minimum activity threshold
    debugMode: false
)
```

## Key Features

### 1. Speaker Diarization
- **Status**: Production ready with 17.7% DER
- **Models**: CoreML-based speaker segmentation and embedding
- **Auto-recovery**: Handles corrupted model downloads automatically

### 2. Voice Activity Detection (VAD)
- **Status**: Merged in PR #9, 98% accuracy on MUSAN dataset
- **Models**: Custom CoreML pipeline with enhanced fallback
- **Config**: Optimal threshold of 0.445

### 3. Auto-Recovery Mechanism
- Automatic detection and recovery from CoreML compilation failures
- Re-downloads corrupted models from Hugging Face
- Up to 3 retry attempts with comprehensive logging

## CLI Commands

### Benchmarking
```bash
# Full AMI benchmark
swift run fluidaudio diarization-benchmark --auto-download

# Single file with custom parameters
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.7 --output results.json

# Process individual audio
swift run fluidaudio process meeting.wav --output results.json
```

### VAD Commands (Pending CLI Integration)
```bash
# Once integrated:
swift run fluidaudio vad-benchmark --num-files 40 --threshold 0.445
```

### Development
```bash
# Build
swift build
swift build -c release

# Test
swift test
swift test --filter CITests

# Package management
swift package update
swift package resolve
```

## Parameter Tuning Guide

### DiarizerConfig Parameters

1. **clusteringThreshold** (0.0-1.0)
   - Sweet spot: 0.7-0.8
   - Impact: Speaker separation accuracy
   - 0.7 = 17.7% DER (optimal)

2. **minDurationOn** (seconds)
   - Default: 1.0
   - Filters short speech segments

3. **minDurationOff** (seconds)
   - Default: 0.5
   - Minimum gap between speakers

4. **minActivityThreshold** (frames)
   - Default: 10.0
   - Affects missed speech detection

### VADConfig Parameters
```swift
VADConfig(
    threshold: 0.445,             // 98% accuracy
    chunkSize: 512,
    sampleRate: 16000,
    adaptiveThreshold: true,
    minThreshold: 0.1,
    maxThreshold: 0.7,
    enableSNRFiltering: true,
    minSNRThreshold: 6.0,
    useGPU: true
)
```

## Optimization Results Summary

| Configuration | DER | Notes |
|--------------|-----|-------|
| threshold=0.1 | 75.8% | Over-clustering (153+ speakers) |
| threshold=0.5 | 20.6% | Better but still too many speakers |
| **threshold=0.7** | **17.7%** | **OPTIMAL - Production config** |
| threshold=0.8 | 18.0% | Very close to optimal |
| threshold=0.9 | 40.2% | Under-clustering |

## Architecture Notes

- **Online diarization**: Works well with chunk-based processing
- **10-second chunks**: No significant performance impact
- **Speaker tracking**: Effective across chunks
- **DER calculation**: Fixed with optimal speaker mapping (Hungarian algorithm)

## Development Guidelines

1. **Testing**: Always run benchmarks on multiple files for validation
2. **Logging**: Use comprehensive logging for debugging
3. **Error Handling**: Implement graceful degradation
4. **Performance**: Keep RTF < 1.0x for real-time capability
5. **Thread Safety**: Never use `@unchecked Sendable` - implement proper synchronization

## Next Steps

1. **Multi-file validation**: Test optimal config on all AMI files
2. **CLI integration**: Complete VAD command exposure
3. **Real-world testing**: Validate on non-AMI audio
4. **Documentation**: Update API documentation

## Model Sources

- **Diarization**: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **VAD CoreML**: [alexwengg/coreml_silero_vad](https://huggingface.co/alexwengg/coreml_silero_vad)
- **Test Data**: [alexwengg/musan_mini*](https://huggingface.co/datasets/alexwengg) variants
