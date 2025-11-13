import Foundation
import OSLog

/// Mandarin per‑character phoneme lexicon loader and encoder.
///
/// Loads a JSON mapping of single CJK characters to Kokoro‑zh phoneme strings
/// (Bopomofo + optional tone digits), and encodes arbitrary text with
/// punctuation normalization to produce a phoneme string compatible with the
/// Kokoro zh model vocabulary.
public actor ZhCharLexicon {
    public static let shared = ZhCharLexicon()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "ZhCharLexicon")
    private var mapping: [String: String] = [:]
    private var isLoaded = false
    private var overrideURL: URL? = nil

    public func setOverrideURL(_ url: URL?) {
        overrideURL = url
        isLoaded = false
    }

    /// Ensure the lexicon mapping is loaded.
    /// Falls back to cache path: ~/.cache/fluidaudio/Models/kokoro/zh_char_phonemes.json
    public func ensureLoaded() async throws {
        if isLoaded { return }

        let url: URL = try {
            if let overrideURL, FileManager.default.fileExists(atPath: overrideURL.path) {
                return overrideURL
            }
            let base = try TtsModels.cacheDirectoryURL().appendingPathComponent("Models/kokoro")
            return base.appendingPathComponent("zh_char_phonemes.json")
        }()

        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.warning("ZhCharLexicon file not found at: \(url.path)")
            mapping = [:]
            isLoaded = true
            return
        }

        do {
            let data = try Data(contentsOf: url)
            let json = try JSONSerialization.jsonObject(with: data)
            guard let dict = json as? [String: Any] else {
                throw TTSError.processingFailed("Invalid zh_char_phonemes.json format")
            }
            var parsed: [String: String] = [:]
            parsed.reserveCapacity(dict.count)
            for (k, v) in dict {
                if let s = v as? String, k.count == 1 {
                    parsed[k] = s
                }
            }
            mapping = parsed
            isLoaded = true
            logger.info("Loaded zh_char_phonemes.json with \(mapping.count) entries")
        } catch {
            mapping = [:]
            isLoaded = true
            throw TTSError.processingFailed("Failed to load zh_char_phonemes.json: \(error.localizedDescription)")
        }
    }

    /// Encode text to a Kokoro‑zh phoneme string using the lexicon only.
    /// Unknown characters are skipped; punctuation is normalized.
    /// Returned string contains only tokens present in `allowedTokens`.
    public func encode(
        text raw: String,
        allowedTokens: Set<String>
    ) async throws -> String {
        try await ensureLoaded()
        let text = ZhCharLexicon.normalize(raw)
        var out = String()
        out.reserveCapacity(text.count * 2)

        for ch in text {
            if let norm = ZhCharLexicon.PUNCT_MAP[ch] {
                if allowedTokens.contains(String(norm)) { out.append(norm) }
                continue
            }
            if ch.isWhitespace {
                out.append(" ")
                continue
            }
            if let tok = mapping[String(ch)], !tok.isEmpty {
                // Only admit codepoints that exist in vocab
                for u in tok {
                    let s = String(u)
                    if allowedTokens.contains(s) { out.append(u) }
                }
            }
        }

        return out
    }

    /// Compute coverage of CJK characters present in the lexicon mapping.
    /// Returns (covered, total, missingUnique).
    public func coverage(
        text raw: String,
        allowedTokens: Set<String>
    ) async throws -> (covered: Int, total: Int, missing: [String]) {
        try await ensureLoaded()
        let text = ZhCharLexicon.normalize(raw)
        var total = 0
        var covered = 0
        var missing: Set<String> = []

        for ch in text {
            if isCJK(ch) {
                total += 1
                if let tok = mapping[String(ch)], !tok.isEmpty {
                    if tok.contains(where: { allowedTokens.contains(String($0)) }) {
                        covered += 1
                    } else {
                        missing.insert(String(ch))
                    }
                } else {
                    missing.insert(String(ch))
                }
            }
        }
        return (covered, total, Array(missing))
    }

    private func isCJK(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF:
            return true
        default:
            return false
        }
    }

    /// Chinese punctuation normalization used by the zh model.
    private static let PUNCT_MAP: [Character: Character] = [
        "，": ",", "。": ".", "？": "?", "！": "!", "：": ":", "；": ";",
        "（": "(", "）": ")", "“": "\"", "”": "\"", "、": ",",
        "—": "—", "…": "…",
    ]

    private static func normalize(_ s: String) -> String {
        return s.precomposedStringWithCompatibilityMapping
    }
}
