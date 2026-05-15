import Foundation

/// Shared TTS constants. Backend-specific tuning lives next to each backend
/// (e.g. `PocketTtsConstants`, `KokoroAneConstants`); this enum only carries
/// values that are genuinely cross-backend.
public enum TtsConstants {

    /// Default voice identifier when callers don't pass an explicit one.
    /// Currently matches the KokoroAne English default (`af_heart`).
    public static let recommendedVoice = "af_heart"
}
