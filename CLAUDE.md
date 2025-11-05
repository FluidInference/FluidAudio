# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluidAudio is a comprehensive Swift framework for local, low-latency audio processing on Apple platforms. It provides state-of-the-art speaker diarization, automatic speech recognition (ASR), and voice activity detection (VAD) through open-source models converted to Core ML. The system processes audio to identify "who spoke when" by segmenting audio and clustering speaker embeddings, with industry-competitive performance (17.7% DER).

## Critical Development Rules

### ⚠️ NEVER USE "unchecked Sendable"

- **DO NOT** use `@unchecked Sendable` under any circumstances
- Always properly implement thread-safe code with proper synchronization
- Use actors, `@MainActor`, or proper locking mechanisms instead
- If you encounter Sendable conformance issues, fix them properly rather than bypassing with `@unchecked`

### ⚠️ NEVER CREATE DUMMY MODELS OR SYNTHETIC DATA

- **DO NOT** create dummy, mock, or fake models for testing or development
- **DO NOT** generate synthetic audio data for testing
- **DO NOT** use random/fake models as placeholders
- **DO NOT** create "demonstration" or "simulated" models that don't contain real weights
- Always use the actual models required by the code
- If model authentication is required, inform the user rather than creating dummy versions
- Mock models produce meaningless results and waste development time
- Placeholder models with random weights will destroy performance (e.g., 17.8% → 77.1% DER)

### ⚠️ MODEL OPERATIONS - CONSULT BEFORE IMPLEMENTING

- When asked to merge, convert, or modify models:
  - If it seems impossible or I have significant objections, CONSULT YOU FIRST
  - Explain the concerns and let you decide whether to proceed
  - If you say proceed, then DO IT IMMEDIATELY without further objections
- **DO NOT** create demonstration or placeholder models without permission
- **DO NOT** implement alternatives without asking
- Only after your approval: Implementation, then explanation of results

### Code Style and Formatting

- **Swift Format**: This project uses swift-format for consistent code style
- **Configuration**: See `.swift-format` for style rules
- **Auto-formatting**: PRs are automatically checked for formatting compliance
- **Local formatting**: Run `swift format --in-place --recursive --configuration .swift-format Sources/ Tests/`

#### Style Guidelines
- **Line length**: 120 characters
- **Indentation**: 4 spaces
- **Import order**: `import CoreML`, `import Foundation`, `import OSLog` (OrderedImports rule)
- **Naming conventions**:
  - lowerCamelCase for variables/functions
  - UpperCamelCase for types
- **Error handling**: Use proper Swift error handling, no force unwrapping in production
- **Documentation**: Triple-slash comments (`///`) for public APIs
- **Thread safety**: Use actors, `@MainActor`, or proper locking - never `@unchecked Sendable`
- **Control flow**: Prefer flattened if statements with early returns/continues over nested if statements. Use guard statements and inverted conditions to exit early. Nested if statements should be absolutely avoided to improve readability and reduce cognitive complexity.

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

### 2. Auto-Recovery Mechanism
- Automatic detection and recovery from CoreML compilation failures
- Re-downloads corrupted models from Hugging Face
- Up to 3 retry attempts with comprehensive logging

## Essential Development Commands

### Core Commands
```bash
# Build
swift build                             # Debug build
swift build -c release                 # Release build (recommended for benchmarks)

# Test
swift test                             # Run all tests
swift test --parallel                  # Parallel test execution
swift test --filter CITests           # Run CI-specific tests only
swift test --filter AsrManagerTests   # Run specific test class

# Package management
swift package update                   # Update dependencies
swift package resolve                 # Resolve dependencies
swift package clean                   # Clean build cache
```

### Code Quality
```bash
# Format code (requires Swift 6+ for development)
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/

# Check formatting without modifying
swift format lint --recursive --configuration .swift-format Sources/ Tests/

# Verify formatting compliance (CI-style check)
swift format --configuration .swift-format Sources/ Tests/
```

### CLI Commands

#### Benchmarking
```bash
# Diarization benchmarks
swift run fluidaudio diarization-benchmark --auto-download
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.7 --output results.json

# ASR benchmarks
swift run fluidaudio asr-benchmark --subset test-clean --max-files 100
swift run fluidaudio asr-benchmark --subset test-other --output asr_results.json

# FLEURS multilingual benchmark
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

# VAD benchmark
swift run fluidaudio vad-benchmark --num-files 40 --threshold 0.5
```

#### Audio Processing
```bash
# Transcription
swift run fluidaudio transcribe audio.wav
swift run fluidaudio transcribe audio.wav --low-latency

# Multi-stream processing
swift run fluidaudio multi-stream audio1.wav audio2.wav

# Diarization processing
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6
```

#### Dataset Management
```bash
# Download evaluation datasets
swift run fluidaudio download --dataset ami-sdm
swift run fluidaudio download --dataset librispeech-test-clean
swift run fluidaudio download --dataset librispeech-test-other
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

## High-Level Architecture

### Project Structure
```
FluidAudio/
├── Sources/
│   ├── FluidAudio/           # Main library
│   │   ├── ASR/             # Automatic Speech Recognition
│   │   ├── Diarizer/        # Speaker diarization system
│   │   ├── VAD/             # Voice Activity Detection
│   │   └── Shared/          # Common utilities (audio conversion, memory optimization)
│   └── FluidAudioCLI/       # Command-line interface (macOS only)
├── Tests/                   # Comprehensive test suite
└── Datasets/               # Evaluation datasets (AMI corpus)
```

### Core Components

#### 1. ASR (Automatic Speech Recognition)
- **AsrManager**: Main class for speech-to-text processing
- **TDT (Token Duration Transducer)**: Advanced decoding architecture with Token Duration tracking
- **Architecture**: Stateless processing with automatic decoder state reset after each transcription
- **Chunking Strategy**: Fixed-size chunks (~14.96s) with 2.0s overlap, stateless decoding per chunk
- **Token Merging**: Sophisticated algorithm using contiguous pair matching, LCS, or midpoint splitting
- **Models**: Parakeet TDT v3 (0.6b) supporting 25 European languages
- **Performance**: ~209.8x RTF on M4 Pro, 55.7% WER improvement with stateless approach

#### 2. Diarization System
- **DiarizerManager**: Main orchestrator for speaker separation
- **SegmentationProcessor**: Voice activity detection and segmentation
- **EmbeddingExtractor**: Speaker embedding generation for clustering
- **SpeakerManager**: Consistent speaker ID tracking across chunks
- **Performance**: 17.7% DER on AMI dataset (competitive with research)

#### 3. Voice Activity Detection
- **VadManager**: Voice activity detection with CoreML models
- **VadAudioProcessor**: additional filtering process 

#### 4. Shared Infrastructure
- **ANEMemoryOptimizer**: Apple Neural Engine memory management
- **AudioConverter**: Universal audio format conversion to 16kHz mono Float32
- **ModelDownloader**: Automatic model retrieval from HuggingFace with recovery
- **Zero-Copy Processing**: Efficient model chaining without data duplication

### Processing Pipeline
1. **Audio Input** → AudioConverter (16kHz mono Float32)
2. **VAD Processing** → Voice activity segments
3. **Diarization** → Speaker embeddings + clustering
4. **ASR Processing** → Speech-to-text transcription
5. **Output** → Timestamped speaker-attributed transcripts

### Threading and Concurrency
- **Actor-based Architecture**: Thread-safe processing without `@unchecked Sendable`
- **Stateless Decoding**: Each chunk transcribed independently with fresh decoder state
- **Automatic State Reset**: Decoder state reset after each `transcribe()` call for independent processing
- **Memory Management**: Automatic cleanup and ANE optimization
- **Batch Processing**: Optimized for sequential transcription of multiple files without state carryover

### Model Management
- **Automatic Downloads**: Models fetched from HuggingFace on first use
- **Auto-Recovery**: Corrupt model detection and re-download
- **CoreML Compilation**: Optimized for Apple Neural Engine
- **Caching**: Local model storage with validation

## Architecture Notes

- **Stateless ASR**: Each chunk transcribed independently with automatic state reset
- **Chunk-based Processing**: Fixed ~14.96s chunks with 2.0s overlap for merging
- **Token Merging**: Three-tier strategy (contiguous pairs, LCS, midpoint split) handles chunk boundaries
- **Quality Improvement**: Stateless approach eliminates context pollution, yielding 55.7% WER reduction
- **Batch Processing**: Optimized for transcribing multiple files sequentially
- **Online diarization**: Works well with chunk-based processing
- **Speaker tracking**: Effective across chunks
- **DER calculation**: Fixed with optimal speaker mapping (Hungarian algorithm)
- **Cross-platform**: Supports macOS 14.0+, iOS 17.0+ (library), CLI macOS-only

## Streaming Diarization (Work in Progress)

### Goal
Develop a custom streaming speaker diarization manager that maintains consistent speaker IDs across chunks WITHOUT using the Hungarian algorithm for retroactive remapping (which is "cheating" in real-time scenarios).

## Development Environment

### Requirements
- **Swift**: 5.10+ (Swift 6+ required for contributors using swift-format)
- **Platforms**: macOS 14.0+, iOS 17.0+ 
- **Xcode**: Latest stable version for iOS development
- **Hardware**: Apple Silicon recommended for optimal performance

### CI/CD Pipeline
The project uses GitHub Actions with the following workflows:
- **swift-format.yml**: Code formatting compliance checks
- **tests.yml**: Cross-platform build and test execution
- **asr-benchmark.yml**: ASR performance validation
- **diarizer-benchmark.yml**: Speaker diarization benchmarks
- **vad-benchmark.yml**: Voice activity detection validation

### Code Style Configuration
- **Swift Format**: Enforced via `.swift-format` config
- **Line Length**: 120 characters
- **Indentation**: 4 spaces
- **Formatting Rules**: Automatic via swift-format, CI enforced

## User Preferences

- Never start responses with positive re-affirming text like "You're absolutely right!", "Good change!", "Excellent progress!", or similar
- Get straight to the point with technical facts
- For debugging, use print statements and delete them at the end when instructed
- Never create fallbacks or simplified solutions that don't actually solve the problem
- Always go for the proper solution over the "simplified" solution
- When asked to implement something specific, DO IT FIRST before explaining why it might not be optimal - implementation first, explanation second
- Just do as instructed - don't try to over-do things that aren't asked

## Development Guidelines

1. **Testing**: Always run benchmarks on multiple files for validation
2. **Logging**: Use comprehensive logging for debugging
3. **Error Handling**: Implement graceful degradation
4. **Performance**: Keep RTFx > 1.0x for real-time capability
5. **Thread Safety**: Never use `@unchecked Sendable` - implement proper synchronization
6. **Follow Instructions**: When the user asks to implement something specific, DO IT FIRST before explaining why it might not be optimal. Implementation first, explanation second.
7. **Avoid Deprecated Code**: Do not add support for deprecated models or features unless explicitly requested. Keep the codebase clean by only supporting current versions.
8. **Testing Policy**: ONLY add or run tests when explicitly requested by the user
9. **Git Operations**: NEVER run `git push` unless explicitly requested by the user. Only commit when asked.
10. **Code Formatting**: All code must pass swift-format checks before merge

## Next Steps

1. **Multi-file validation**: Test optimal config on all AMI files
2. **Real-world testing**: Validate on non-AMI audio
3. **Documentation**: Update API documentation

## Testing Strategy

### Test Categories
- **Unit Tests**: Component-level testing for individual classes
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking against standard datasets
- **Memory Tests**: ANE memory optimization validation
- **Edge Case Tests**: Boundary condition handling
- **CI Tests**: Smoke tests for continuous integration

### Key Test Classes
- **CITests**: Lightweight tests for CI pipeline
- **AsrManagerTests**: ASR functionality validation
- **DiarizerMemoryTests**: Memory management validation
- **SendableTests**: Thread safety compliance
- **SegmentationProcessorTests**: Audio segmentation accuracy

### Running Specific Tests
```bash
# CI-specific tests (lightweight)
swift test --filter CITests

# ASR component tests
swift test --filter AsrManagerTests

# Memory optimization tests
swift test --filter ANEMemoryOptimizerTests

# Edge case validation
swift test --filter EdgeCaseTests
```

## Model Sources

- **Diarization**: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **VAD CoreML**: [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml)
- **ASR Models**: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Test Data**: [alexwengg/musan_mini*](https://huggingface.co/datasets/alexwengg) variants