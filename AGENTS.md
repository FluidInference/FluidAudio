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

## Swift 6 Concurrency Migration

### Common Warning Patterns & Fixes

#### 1. Non-Sendable Struct Types
When a struct is used across actor boundaries or in concurrent contexts, add `Sendable` conformance:

```swift
// Before
struct MyConfig {
    let timeout: Int
}

// After
struct MyConfig: Sendable {
    let timeout: Int
}
```

**Examples from codebase:**
- `TdtDecoderState` (ASR/TDT/TdtDecoderState.swift) - LSTM state management
- `AssignmentConfig` (Diarizer/Clustering/SpeakerOperations.swift) - speaker assignment config
- `DownloadConfig` (DownloadUtils.swift) - download timeout settings

#### 2. Function Type Aliases
Closure types that cross actor boundaries must be marked `@Sendable`:

```swift
// Before
public typealias DataWriter = (Data, URL) throws -> Void

// After
public typealias DataWriter = @Sendable (Data, URL) throws -> Void
```

**Applied to:** AssetDownloader.swift - `DataWriter` and `FileMover` typealias

#### 3. Singleton/Shared Static Properties
Global actor isolation for non-Sendable classes:

```swift
// Before
static let shared = EspeakG2P()

// After
@MainActor static let shared = EspeakG2P()
```

**Applied to:** EspeakG2P.swift - eSpeak NG wrapper singleton

#### 4. Mutable Global State (#MutableGlobalVariable)
Mutable static properties require proper synchronization or actor isolation. Options:

**Option A: Use @MainActor for entire type**
```swift
@MainActor
class MyService {
    static var cache: [String: Data] = [:]
    static let cacheLock = NSLock()
}
```

**Option B: Use actor for concurrent access**
```swift
actor CacheManager {
    private var cache: [String: Data] = [:]

    func set(_ key: String, _ value: Data) {
        cache[key] = value
    }

    func get(_ key: String) -> Data? {
        cache[key]
    }
}
```

**Current cases:** KokoroSynthesizer.swift - voiceEmbeddingPayloads, voiceEmbeddingVectors

#### 5. Non-Sendable Framework Types (MLMultiArray)
CoreML's `MLMultiArray` doesn't conform to Sendable. When passing across actor boundaries:
- Wrap in a Sendable struct/class
- Use `@MainActor` for related processing
- Create separate Sendable representations for cross-actor data

### Migration Checklist

When fixing concurrency warnings:
1. Run `swift build` and capture full warning output
2. Identify warning categories (Sendable, @MainActor, #MutableGlobalVariable, etc.)
3. Start with "low-hanging fruit": simple struct/typealias additions
4. Address mutable state with proper synchronization
5. Handle framework non-Sendable types last (most complex)
6. Run `swift build` again to verify warning reduction
7. Never use `@unchecked Sendable` as a shortcut

## Clean code

- When adding new interfaces, make sure that the API is consistent with the other model managers
- Files should be isolated and the code should contain a single responsibility for each

## Mobius Plan

When users ask you to perform tasks that might be more compilcated, make sure you look at PLANS.md and follow the instructions there to plan the change out first and follow the instructions there. The plans should be in a .mobius/ folder and never committed directly to Github
