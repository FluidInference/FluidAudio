import Foundation

/// Cross-backend TTS cache root.
///
/// Mirrors the cache root previously exposed by the now-removed
/// `TtsModels.cacheDirectoryURL()` so callers (currently `G2PModel` and
/// `MultilingualG2PModel`) can still resolve
/// `~/.cache/fluidaudio/Models/<repo>/` paths without depending on a
/// backend-specific manager.
///
/// macOS resolves to `~/.cache/fluidaudio`; other platforms use the
/// caches directory under `fluidaudio/`.
public enum TtsCacheDirectory {

    /// Ensures the platform-appropriate cache root exists and returns it.
    public static func ensure() throws -> URL {
        let base: URL
        #if os(macOS)
        base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(
                for: .cachesDirectory, in: .userDomainMask
            ).first
        else {
            throw TTSError.processingFailed("Failed to locate caches directory")
        }
        base = first
        #endif

        let cacheDirectory = base.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory
    }
}
