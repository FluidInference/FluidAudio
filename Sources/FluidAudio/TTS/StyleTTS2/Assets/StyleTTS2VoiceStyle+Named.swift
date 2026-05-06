import Foundation

/// Named-voice resolution for `StyleTTS2VoiceStyle`.
///
/// Once `StyleTTS2Manager.initialize` has staged the shared assets (and
/// the `voices/` subdirectory), `named(_:in:)` resolves a preset id from
/// `StyleTTS2VoicePresets` to its `ref_s_<voice>.bin` file inside the cached
/// repo root and loads the 256-fp32 blob.
///
/// Callers that already have an absolute `ref_s.bin` path keep using
/// `StyleTTS2VoiceStyle.load(from:)`. This API is purely a convenience for
/// "just give me Vinay" CLI / app integrations.
extension StyleTTS2VoiceStyle {

    /// Maximum number of preset ids embedded directly in an
    /// "unknown voice id" error message before truncating with a "+N more"
    /// summary. Keeps the message readable as the catalog grows.
    private static let unknownIDExampleLimit = 10

    /// Load a named preset voice from the repo root produced by
    /// `StyleTTS2AssetDownloader.ensureModels(...)`.
    ///
    /// - Parameters:
    ///   - id: Preset id from `StyleTTS2VoicePresets.allIDs` (case-insensitive).
    ///   - repoRoot: Directory returned by
    ///     `StyleTTS2AssetStore.ensureAssetsAvailable()`.
    /// - Returns: The decoded 256-fp32 voice style.
    /// - Throws: `StyleTTS2Error.modelNotFound` in **two distinct shapes**
    ///   that callers may want to differentiate by message contents:
    ///     1. Unknown id — message is `"Unknown StyleTTS2 voice id '<id>'…"`
    ///        and the file was never looked at. Fix the caller-supplied id.
    ///     2. Known id but `ref_s_<voice>.bin` is missing on disk — message
    ///        is just the filename, raised by
    ///        `StyleTTS2VoiceStyle.load(from:)`. Indicates the cache was
    ///        evicted or the asset bundle failed verification; re-run
    ///        `StyleTTS2AssetStore.ensureAssetsAvailable()`.
    ///   `StyleTTS2Error.invalidConfiguration` is also possible if the blob
    ///   is present but malformed (wrong size or unreadable bytes).
    public static func named(_ id: String, in repoRoot: URL) throws -> StyleTTS2VoiceStyle {
        guard let voice = StyleTTS2VoicePresets.voice(forID: id) else {
            throw StyleTTS2Error.modelNotFound(formatUnknownIDMessage(id))
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
    ///
    /// This shape coexists with `named(_:in:)` on purpose:
    /// - `named(_:in:)` is the load-or-fail entry point (caller wants the
    ///   blob in memory; an unknown id is exceptional).
    /// - `voiceURL(forID:in:)` is the lookup-or-validate shape (caller is
    ///   normalizing user input or checking whether to fall back to a
    ///   default voice; an unknown id is expected and expressed as `nil`).
    public static func voiceURL(forID id: String, in repoRoot: URL) -> URL? {
        guard let voice = StyleTTS2VoicePresets.voice(forID: id) else { return nil }
        return voiceURL(for: voice, in: repoRoot)
    }

    // MARK: - Private

    private static func formatUnknownIDMessage(_ id: String) -> String {
        let allIDs = StyleTTS2VoicePresets.allIDs
        let total = allIDs.count
        let prefix = "Unknown StyleTTS2 voice id '\(id)'."
        if total <= unknownIDExampleLimit {
            return "\(prefix) Known ids: \(allIDs.joined(separator: ", "))"
        }
        let head = allIDs.prefix(unknownIDExampleLimit).joined(separator: ", ")
        let remaining = total - unknownIDExampleLimit
        return
            "\(prefix) Known ids (\(total) total): \(head), +\(remaining) more. "
            + "Use `StyleTTS2VoicePresets.allIDs` for the full list."
    }
}
