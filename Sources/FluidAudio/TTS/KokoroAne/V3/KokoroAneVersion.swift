#if TTS
import Foundation

/// Kokoro inference architecture. Existing applications retain the seven-stage default.
public enum KokoroAneVersion: String, CaseIterable, Sendable {
    /// Original seven-stage pipeline, retaining the SDK's existing platform support.
    case legacy
    /// Static-bucket pipeline, requiring macOS 15 or iOS 18. Its fast vocoder
    /// runs only on a GPU routing; the OS 27 default skips it (#1103).
    case v3
}
#endif
