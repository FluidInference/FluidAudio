import Foundation

/// Shared Misaki lexicon JSON cache loader.
///
/// Reads the preprocessed `us_lexicon_cache.json` payload (word →
/// phoneme tokens) shipped under the kokoro HF repo root and exposes the
/// resulting maps to callers (currently `StyleTTS2Manager`). The on-disk
/// schema is `{ lower: [String: [String]], caseSensitive: [String:
/// [String]] }`; consumers pass an `allowedTokens` set so phonemes outside
/// the target backend's character vocab are filtered out at load time.
public actor LexiconAssetCache {

    private static let logger = AppLogger(category: "LexiconAssetCache")

    private var wordToPhonemes: [String: [String]] = [:]
    private var caseSensitiveWordToPhonemes: [String: [String]] = [:]
    private var isLoaded = false

    private struct CachePayload: Codable {
        let lower: [String: [String]]
        let caseSensitive: [String: [String]]
    }

    public init() {}

    /// Load `us_lexicon_cache.json` from `kokoroDirectory`, filtering each
    /// entry's phoneme list to `allowedTokens`. Throws if the file is
    /// missing or unparseable.
    public func ensureLoaded(
        kokoroDirectory: URL, allowedTokens: Set<String>
    ) async throws {
        if isLoaded && !caseSensitiveWordToPhonemes.isEmpty { return }

        let cacheURL = kokoroDirectory.appendingPathComponent("us_lexicon_cache.json")
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            throw TTSError.processingFailed(
                "Missing lexicon cache (expected us_lexicon_cache.json)")
        }

        do {
            let data = try Data(contentsOf: cacheURL)
            let payload = try JSONDecoder().decode(CachePayload.self, from: data)
            let filteredLower = payload.lower.mapValues {
                $0.filter { allowedTokens.contains($0) }
            }
            let filteredCase = payload.caseSensitive.mapValues {
                $0.filter { allowedTokens.contains($0) }
            }

            guard !filteredLower.isEmpty else {
                throw TTSError.processingFailed(
                    "us_lexicon_cache.json had no entries within the allowed token set")
            }

            wordToPhonemes = filteredLower
            caseSensitiveWordToPhonemes = filteredCase
            isLoaded = true
            Self.logger.info("Loaded lexicon cache: \(filteredLower.count) entries")
        } catch let error as TTSError {
            throw error
        } catch {
            wordToPhonemes = [:]
            caseSensitiveWordToPhonemes = [:]
            isLoaded = false
            throw TTSError.processingFailed(
                "Failed to load lexicon cache: \(error.localizedDescription)")
        }
    }

    /// Returns the lower-cased and case-sensitive lexicon maps.
    public func lexicons() -> (word: [String: [String]], caseSensitive: [String: [String]]) {
        (wordToPhonemes, caseSensitiveWordToPhonemes)
    }
}
