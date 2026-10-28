#if TTS
import Foundation

/// Shared Misaki lexicon JSON cache loader.
///
/// Reads a preprocessed Misaki lexicon payload (word → phoneme tokens)
/// shipped under the kokoro HF repo root — `us_lexicon_cache.json`, or
/// `gb_lexicon_cache.json` for British English — and exposes the resulting
/// maps to callers (StyleTTS2, KokoroAne, Inflect, Paradee). The on-disk
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

    /// Misaki's American (`us_`) or British (`gb_`) lexicon cache, built the
    /// same way from that dialect's `_gold` + `_silver` dictionaries.
    public static func englishFileName(british: Bool) -> String {
        british ? "gb_lexicon_cache.json" : "us_lexicon_cache.json"
    }

    /// Load `fileName` (default `us_lexicon_cache.json`) from
    /// `kokoroDirectory`, filtering each entry's phoneme list to
    /// `allowedTokens`. Throws if the file is missing or unparseable.
    public func ensureLoaded(
        kokoroDirectory: URL, allowedTokens: Set<String>,
        fileName: String = LexiconAssetCache.englishFileName(british: false)
    ) async throws {
        if isLoaded && !caseSensitiveWordToPhonemes.isEmpty { return }

        let cacheURL = kokoroDirectory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            throw TTSError.processingFailed(
                "Missing lexicon cache (expected \(fileName))")
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
                    "\(fileName) had no entries within the allowed token set")
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
#endif
