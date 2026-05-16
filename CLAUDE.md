# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluidAudio is a Swift framework for local, low-latency audio processing on Apple platforms. It provides speaker diarization, automatic speech recognition (ASR), and voice activity detection (VAD) through open-source models converted to Core ML.

## Critical Development Rules

### NEVER USE `@unchecked Sendable`

- Always implement thread-safe code with proper synchronization
- Use actors, `@MainActor`, or proper locking mechanisms instead
- If you encounter Sendable conformance issues, fix them properly

### NEVER CREATE DUMMY MODELS OR SYNTHETIC DATA

- Do not create dummy, mock, or fake models for testing or development
- Do not generate synthetic audio data for testing
- Always use the actual models required by the code
- If model authentication is required, inform the user rather than creating dummy versions

### NEVER UPLOAD TO HUGGINGFACE

- Do not upload models, datasets, or any files to HuggingFace
- Do not create HuggingFace repos
- Prepare files locally and let the user handle all HF uploads themselves

### MODEL OPERATIONS - CONSULT BEFORE IMPLEMENTING

- When asked to merge, convert, or modify models:
  - If it seems impossible or there are significant objections, consult the user first
  - If they say proceed, do it immediately without further objections
- Do not create placeholder models or implement alternatives without asking

## User Preferences

- Never start responses with positive re-affirming text ("You're absolutely right!", "Good change!", etc.)
- Get straight to the point with technical facts
- For debugging, use print statements and delete them at the end when instructed
- Never create fallbacks or simplified solutions that don't actually solve the problem
- When asked to implement something specific, do it first before explaining why it might not be optimal
- Don't over-do things that aren't asked

## Development Guidelines

1. **Follow Instructions**: Implementation first, explanation second
2. **Testing Policy**: Add unit tests when writing new code.
3. **Git Operations**: Never run `git push` unless explicitly requested.
   - **No Co-Author Tags**: Do not add `Co-Authored-By` lines for Claude, Copilot, or any AI assistant in commit messages.
   - **No GitHub comments**: Never post comments, reviews, or reactions on issues or PRs via `gh`. Reading issues, PRs, and comments is fine. Creating PRs and editing PR titles/bodies is fine.
4. **Multi-Agent Workflow**: This repo is worked on by multiple coding agents
   in parallel. Switching branches in a shared working tree drags unrelated
   WIP changes (and their build artifacts) into your compile and surfaces
   "file was modified during the build" errors. Use `git worktree` instead
   — shared `.git`, isolated working tree + `.build/`, no collisions.

   ```bash
   # From the primary checkout, create an isolated tree for your branch
   git worktree add ../FluidAudio-<task> -b <branch> origin/main
   cd ../FluidAudio-<task>
   # Independent working tree, independent .build/, shared .git
   ```

   One worktree per active task. Remove with `git worktree remove <path>` when
   done. List active worktrees with `git worktree list`.
5. **Code Formatting**: All code must pass swift-format checks before merge
6. **Avoid Deprecated Code**: Do not add support for deprecated models or features unless explicitly requested
7. **Performance**: Keep RTFx > 1.0x for real-time capability

## Code Style

- **Swift Format**: Enforced via `.swift-format` config, CI checked
- **Local formatting**: `swift format --in-place --recursive --configuration .swift-format Sources/ Tests/`
- **Line length**: 120 characters
- **Indentation**: 4 spaces
- **Import order**: Alphabetical preferred, but OrderedImports rule is disabled due to Swift 6.1 (GitHub Actions CI) vs 6.3 (local) formatter incompatibility. Swift 6.3 is unavailable in GitHub Actions runners.
- **Naming**: lowerCamelCase for variables/functions, UpperCamelCase for types
- **Error handling**: Proper Swift error handling, no force unwrapping in production. Per-module error enums conforming to `Error, LocalizedError` (e.g. `ASRError`, `VadError`, `OfflineDiarizationError`, `Qwen3AsrError`)
- **Logging**: Use `AppLogger(category:)` from `Shared/AppLogger.swift` — not `print()` in production code. One logger per component (e.g. `AppLogger(category: "VadManager")`)
- **Documentation**: Triple-slash comments (`///`) for public APIs
- **Control flow**: Prefer guard statements and early returns over nested if statements

## Build Commands

```bash
# Build
swift build                             # Debug build
swift build -c release                 # Release build (recommended for benchmarks)

# Test
swift test                             # Run all tests
swift test --filter CITests           # Run CI-specific tests only
swift test --filter AsrManagerTests   # Run specific test class

# Format
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
swift format lint --recursive --configuration .swift-format Sources/ Tests/

# Package management
swift package update
swift package resolve
swift package clean
```

### CLI Commands

```bash
# Transcription
swift run fluidaudiocli transcribe audio.wav
swift run fluidaudiocli transcribe audio.wav --low-latency
swift run fluidaudiocli qwen3-transcribe audio.wav
swift run fluidaudiocli multi-stream audio1.wav audio2.wav

# TTS
swift run fluidaudiocli tts "Hello world" --output hello.wav

# Diarization
swift run fluidaudiocli process meeting.wav --output results.json --threshold 0.6
swift run fluidaudiocli sortformer audio.wav
swift run fluidaudiocli parakeet-eou --input audio.wav

# Benchmarks
swift run fluidaudiocli asr-benchmark --subset test-clean --max-files 100
swift run fluidaudiocli diarization-benchmark --auto-download
swift run fluidaudiocli vad-benchmark --num-files 40 --threshold 0.5
swift run fluidaudiocli fleurs-benchmark --languages en_us,fr_fr --samples 10
swift run fluidaudiocli sortformer-benchmark
swift run fluidaudiocli qwen3-benchmark
swift run fluidaudiocli ctc-earnings-benchmark
swift run fluidaudiocli g2p-benchmark

# Dataset downloads
swift run fluidaudiocli download --dataset ami-sdm
swift run fluidaudiocli download --dataset librispeech-test-clean
```

## Project Structure

```
FluidAudio/
├── Sources/
│   ├── FluidAudio/           # Main library (single product)
│   │   ├── ASR/             # Automatic Speech Recognition
│   │   │   ├── Parakeet/    # Parakeet TDT (Decoder/, SlidingWindow/, Streaming/)
│   │   │   └── Qwen3/       # Qwen3 ASR
│   │   ├── Diarizer/        # Speaker diarization (segmentation, embedding, clustering)
│   │   ├── TTS/             # Text-to-speech (KokoroAne, PocketTTS, StyleTTS2, Magpie)
│   │   ├── VAD/             # Voice Activity Detection (Silero VAD)
│   │   └── Shared/          # Common utilities (audio conversion, model downloading)
│   └── FluidAudioCLI/       # Command-line interface (macOS only)
├── Tests/                   # Test suite
├── Scripts/                 # Python utilities (benchmarks, evaluation tools)
├── mobius/                  # Research submodule: model conversions, trials, and known issues
├── Documentation/           # Reference documentation
├── Frameworks/              # Vendored frameworks
└── ThirdPartyLicenses/      # Third-party license files
```

## Architecture Overview

### Core Components
- **AsrManager** (`ASR/Parakeet/`): Speech-to-text via TDT (Token Duration Transducer) decoding. Stateless per-chunk processing with automatic decoder state reset.
- **SlidingWindowAsrManager** (`ASR/Parakeet/SlidingWindow/`): Real-time ASR with sliding window processing and cancellation support.
- **StreamingAsrManager** (`ASR/Parakeet/Streaming/`): Protocol for true streaming ASR engines (EOU, Nemotron) with cache-aware encoders.
- **Qwen3AsrManager** (`ASR/Qwen3/`): Qwen3-based ASR with Whisper mel spectrogram frontend.
- **OfflineDiarizerManager** (`Diarizer/`): Speaker separation via segmentation, embedding extraction, and VBx clustering. 17.7% DER on AMI dataset.
- **VadManager** (`VAD/`): Voice activity detection with CoreML models.
- **KokoroAneManager** (`TTS/KokoroAne/`): ANE-resident Kokoro 82M (7-stage CoreML chain) — English + Mandarin.
- **PocketTtsSynthesizer** (`TTS/PocketTTS/`): PocketTTS streaming text-to-speech synthesis.
- **StyleTTS2Manager** (`TTS/StyleTTS2/`): StyleTTS2 LibriTTS zero-shot voice cloning.
- **MagpieManager** (`TTS/Magpie/`): Magpie multilingual TTS (experimental, RTFx < 1.0).

### Key Patterns
- **Actor-based concurrency**: Thread-safe processing, no `@unchecked Sendable`
- **Stateless ASR**: Each chunk transcribed independently (~14.96s chunks, 2.0s overlap)
- **Auto-recovery**: Corrupt CoreML model detection and re-download from HuggingFace
- **Model management**: Models auto-download from HuggingFace on first use. Can be pre-fetched via `swift run fluidaudiocli download`.
- **Cross-platform**: macOS 14.0+, iOS 17.0+ (library), CLI macOS-only

## Platform Requirements

- **Swift**: 5.10+ (Swift 6+ for swift-format)
- **C++17**: Required for `FastClusterWrapper` (set via `cxxLanguageStandard: .cxx17` in Package.swift)
- **Platforms**: macOS 14.0+, iOS 17.0+
- **Hardware**: Apple Silicon recommended

## CI/CD

GitHub Actions workflows:
- **swift-format.yml**: Code formatting compliance
- **tests.yml**: Build and test execution
- **asr-benchmark.yml**: ASR performance validation
- **diarizer-benchmark.yml**: Diarization benchmarks
- **vad-benchmark.yml**: VAD validation

## Model Sources

- **Diarization**:
  - Online/Streaming (DiarizerManager): [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) (based on pyannote/speaker-diarization-3.1)
  - Offline Batch (OfflineDiarizerManager): [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) (based on pyannote/speaker-diarization-community-1)
- **VAD CoreML**: [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml)
- **ASR Models**: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)
- **Test Data**: [alexwengg/musan_mini*](https://huggingface.co/datasets/alexwengg) variants


Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
