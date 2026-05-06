import Foundation

/// Named-voice resolution for `StyleTTS2VoiceStyle`.
///
/// Once `StyleTTS2AneManager.initialize` has staged the shared assets (and
/// the `voices/` subdirectory), `named(_:in:)` resolves a preset id from
/// `StyleTTS2VoicePresets` to its `ref_s_<voice>.bin` file inside the cached
/// repo root and loads the 256-fp32 blob.
///
/// Callers that already have an absolute `ref_s.bin` path keep using
/// `StyleTTS2VoiceStyle.load(from:)`. This API is purely a convenience for
/// "just give me Vinay" CLI / app integrations.
extension StyleTTS2VoiceStyle {

    /// Load a named preset voice from the repo root produced by
    /// `StyleTTS2ResourceDownloader.ensureModels(...)`.
    ///
    /// - Parameters:
    ///   - id: Preset id from `StyleTTS2VoicePresets.allIDs` (case-insensitive).
    ///   - repoRoot: Directory returned by
    ///     `StyleTTS2ModelStore.ensureAssetsAvailable()`.
    /// - Returns: The decoded 256-fp32 voice style.
    /// - Throws: `StyleTTS2Error.modelNotFound` for unknown ids, or any of
    ///   the underlying `load(from:)` errors.
    public static func named(_ id: String, in repoRoot: URL) throws -> StyleTTS2VoiceStyle {
        guard let voice = StyleTTS2VoicePresets.voice(forID: id) else {
            throw StyleTTS2Error.modelNotFound(
                "Unknown StyleTTS2 voice id '\(id)'. "
                    + "Known ids: \(StyleTTS2VoicePresets.allIDs.joined(separator: ", "))")
        }
        let url = voiceURL(for: voice, in: repoRoot)
        return try StyleTTS2VoiceStyle.load(from: url)
    }

    /// Resolve the on-disk URL of a preset voice without loading it. Useful
    /// for callers that want to pass a URL straight into `synthesize(...)`.
    public static func voiceURL(for voice: StyleTTS2VoicePresets.Voice, in repoRoot: URL) -> URL {
        repoRoot
            .appendingPathComponent(StyleTTS2VoicePresets.directoryName, isDirectory: true)
            .appendingPathComponent(voice.filename)
    }

    /// Convenience: same as the other `voiceURL(for:in:)` but takes the id
    /// directly. Returns `nil` for unknown ids.
    public static func voiceURL(forID id: String, in repoRoot: URL) -> URL? {
        guard let voice = StyleTTS2VoicePresets.voice(forID: id) else { return nil }
        return voiceURL(for: voice, in: repoRoot)
    }
}
