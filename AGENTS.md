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

## Critical Rules
- **NEVER** use `@unchecked Sendable` - implement proper thread safety with actors/MainActor
- **NEVER** create dummy/mock models or synthetic audio data - use real models only
- **NEVER** create simplified versions - implement full solutions or consult first

## Code Style (swift-format config)
- Line length: 120 chars, 4-space indentation
- Import order: `import CoreML`, `import Foundation`, `import OSLog` (OrderedImports rule)
- Naming: lowerCamelCase for variables/functions, UpperCamelCase for types
- Error handling: Use proper Swift error handling, no force unwrapping in production
- Documentation: Triple-slash comments (`///`) for public APIs
- Thread safety: Use actors, `@MainActor`, or proper locking - never `@unchecked Sendable`