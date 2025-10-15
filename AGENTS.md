# FluidAudio - Agent Development Guide

## Build & Test Commands

```bash
swift build                                    # Build project
swift build -c release                        # Release build
swift test                                     # Run all tests
swift test --filter CITests                   # Run single test class
swift test --filter CITests.testPackageImports # Run single test method
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
```

## Architecture

- **FluidAudio/**: Main library (ASR/, Diarizer/, VAD/, Shared/ modules)
- **FluidAudioCLI/**: CLI tool with benchmarking and processing commands
- **Tests/FluidAudioTests/**: Comprehensive test suite
- **Models**: Auto-downloaded from HuggingFace with CoreML compilation
- **Processing Pipeline**: Audio → VAD → Diarization → ASR → Timestamped transcripts

## Critical Rules

- **NEVER** use `@unchecked Sendable` - implement proper thread safety with actors/MainActor
- **NEVER** create dummy/mock models or synthetic audio data - use real models only
- **NEVER** create simplified versions - implement full solutions or consult first
- **NEVER** run `git push` unless explicitly requested by user
- **ONLY** add or run tests when explicitly requested by the user

## Code Style (swift-format config)

- Line length: 120 chars, 4-space indentation
- Import order: `import CoreML`, `import Foundation`, `import OSLog` (OrderedImports rule)
- Naming: lowerCamelCase for variables/functions, UpperCamelCase for types
- Error handling: Use proper Swift error handling, no force unwrapping in production
- Documentation: Triple-slash comments (`///`) for public APIs
- Thread safety: Use actors, `@MainActor`, or proper locking - never `@unchecked Sendable`
- Control flow: Prefer flattened if statements with early returns/continues over nested if statements. Use guard statements and inverted conditions to exit early. Nested if statements should be absolutely avoided.

## Clean code

- When adding new interfaces, make sure that the API is consistent with the other model managers
- Files should be isolated and the code should contain a single responsibility for each

## Mobius Plan

When users ask you to perform tasks that might be more compilcated, make sure you look at PLANS.md and follow the instructions there to plan the change out first and follow the instructions there. The plans should be in a .mobius/ folder and never committed directly to Github
