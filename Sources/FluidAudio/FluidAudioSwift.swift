import Foundation
import OSLog

// MARK: - Re-exports

// Re-export all types and classes from the separate module files
// Since they're in the same module, they're already available when importing FluidAudio

// MARK: - Backward Compatibility

/// Backward compatibility alias for the old config name
public typealias SpeakerDiarizationConfig = DiarizerConfig

/// Backward compatibility alias for the old error type
public typealias SpeakerDiarizationError = DiarizerError

// The Swift Programming Language
// https://docs.swift.org/swift-book

/// A library for fluid audio processing on Apple platforms.
///
/// This package provides speaker diarization and embedding extraction capabilities
/// optimized for macOS and iOS using Apple's machine learning framework.

// Note: The FluidAudio struct below creates a namespace collision
// Types like RawEmbedding and SendableSpeaker are already public in their respective files
// and will be available when importing FluidAudio module

public struct FluidAudio {
    // Empty struct for namespace - all functionality is in the module's types
}
