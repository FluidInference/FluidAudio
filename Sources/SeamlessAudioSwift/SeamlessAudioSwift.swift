import Foundation
import OSLog

// MARK: - Re-exports

// Re-export all types and classes from the separate module files
// Since they're in the same module, they're already available when importing SeamlessAudioSwift

// MARK: - Backward Compatibility

/// Backward compatibility alias for the old class name
@available(macOS 13.0, iOS 16.0, *)
public typealias SpeakerDiarizationManager = SherpaOnnxDiarizerManager

/// Backward compatibility alias for the old config name
public typealias SpeakerDiarizationConfig = DiarizerConfig

/// Backward compatibility alias for the old error type
public typealias SpeakerDiarizationError = DiarizerError

