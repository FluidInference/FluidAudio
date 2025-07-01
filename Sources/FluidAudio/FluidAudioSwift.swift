import Foundation
import OSLog

// MARK: - Re-exports

// Re-export all types and classes from the separate module files
// Since they're in the same module, they're already available when importing SeamlessAudioSwift

// MARK: - Backward Compatibility

/// Backward compatibility alias for the old config name
@available(macOS 13.0, iOS 16.0, *)
public typealias SpeakerDiarizationConfig = DiarizerConfig

/// Backward compatibility alias for the old error type
public typealias SpeakerDiarizationError = DiarizerError

// The Swift Programming Language
// https://docs.swift.org/swift-book

/// A library for fluid audio processing on Apple platforms.
///
/// This package provides speaker diarization and embedding extraction capabilities
/// optimized for macOS and iOS using Apple's machine learning framework.

public struct FluidAudio {

}
