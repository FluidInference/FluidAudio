import Foundation

/// Kokoro inference architecture. Existing applications retain the seven-stage default.
public enum KokoroAneVersion: String, CaseIterable, Sendable {
    /// Original seven-stage pipeline, retaining the SDK's existing platform support.
    case legacy
    /// Hybrid ANE/GPU fast path, requiring macOS 15 or iOS 18.
    case v3
}
