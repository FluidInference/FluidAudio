import Foundation

/// Hanzi → pinyin lookup, loaded from the binary `.bin` files shipped at
/// `FluidInference/kokoro-82m-coreml/ANE-zh/assets/`.
///
/// Two artefacts are consumed:
///
///   * `pinyin_single.bin` — single-character map. Format:
///     ```
///     repeat:
///       u32_le  unicode codepoint
///       u8      pinyin_count
///       repeat pinyin_count:
///         u8    utf8_byte_length
///         N×u8  utf8 bytes (pinyin syllable with diacritic, e.g. "líng")
///     ```
///
///   * `pinyin_phrases.bin` — multi-character phrase map. Format:
///     ```
///     repeat:
///       u16_le  utf8_byte_length of phrase
///       N×u8    utf8 phrase bytes (e.g. "一丁不识")
///       u8      pinyin_count
///       repeat pinyin_count:
///         u8    utf8_byte_length
///         M×u8  utf8 pinyin bytes
///     ```
///
/// Pinyins are stored with **diacritic tone marks** (e.g. `níhǎo`), which
/// `MandarinPinyinNormalizer.normalize` converts to the
/// `<base><digit>` form (`ni2`, `hao3`) the bopomofo mapper expects.
public struct MandarinPinyinDict: Sendable {

    /// Phrase (≥1 chars) → ordered list of pinyin syllables.
    public let phrases: [String: [String]]
    /// Single Hanzi codepoint → ordered list of pinyin syllables.
    /// Multiple readings indicate a polyphone; index 0 is the canonical
    /// pypinyin choice.
    public let singles: [UInt32: [String]]
    /// Length (in `Character`s) of the longest phrase entry. Bounds the
    /// forward-maximum-match search in `MandarinG2P.segment`.
    public let maxPhraseCharCount: Int

    public init(phrases: [String: [String]], singles: [UInt32: [String]]) {
        self.phrases = phrases
        self.singles = singles
        var maxLen = 1
        for key in phrases.keys where key.count > maxLen {
            maxLen = key.count
        }
        self.maxPhraseCharCount = maxLen
    }

    public enum LoadError: Swift.Error, LocalizedError {
        case truncated(String)

        public var errorDescription: String? {
            switch self {
            case .truncated(let what): return "Mandarin G2P dict \(what) is truncated"
            }
        }
    }

    /// Parse the single-character `.bin` payload.
    public static func parseSingles(_ data: Data) throws -> [UInt32: [String]] {
        var result: [UInt32: [String]] = [:]
        var pos = 0
        while pos < data.count {
            guard pos + 5 <= data.count else { throw LoadError.truncated("singles") }
            let cp =
                UInt32(data[pos])
                | (UInt32(data[pos + 1]) << 8)
                | (UInt32(data[pos + 2]) << 16)
                | (UInt32(data[pos + 3]) << 24)
            pos += 4
            let count = Int(data[pos])
            pos += 1
            var pyList: [String] = []
            pyList.reserveCapacity(count)
            for _ in 0..<count {
                guard pos < data.count else { throw LoadError.truncated("singles pinyin") }
                let length = Int(data[pos])
                pos += 1
                guard pos + length <= data.count else {
                    throw LoadError.truncated("singles pinyin payload")
                }
                let utf8 = data.subdata(in: pos..<(pos + length))
                guard let s = String(data: utf8, encoding: .utf8) else {
                    throw LoadError.truncated("singles pinyin utf8")
                }
                pyList.append(s)
                pos += length
            }
            result[cp] = pyList
        }
        return result
    }

    /// Parse the phrase `.bin` payload.
    public static func parsePhrases(_ data: Data) throws -> [String: [String]] {
        var result: [String: [String]] = [:]
        var pos = 0
        while pos < data.count {
            guard pos + 3 <= data.count else { throw LoadError.truncated("phrases") }
            let phraseLen = Int(data[pos]) | (Int(data[pos + 1]) << 8)
            pos += 2
            guard pos + phraseLen + 1 <= data.count else {
                throw LoadError.truncated("phrases payload")
            }
            let phraseBytes = data.subdata(in: pos..<(pos + phraseLen))
            guard let phrase = String(data: phraseBytes, encoding: .utf8) else {
                throw LoadError.truncated("phrases utf8")
            }
            pos += phraseLen
            let count = Int(data[pos])
            pos += 1
            var pyList: [String] = []
            pyList.reserveCapacity(count)
            for _ in 0..<count {
                guard pos < data.count else { throw LoadError.truncated("phrases pinyin") }
                let length = Int(data[pos])
                pos += 1
                guard pos + length <= data.count else {
                    throw LoadError.truncated("phrases pinyin payload")
                }
                let utf8 = data.subdata(in: pos..<(pos + length))
                guard let s = String(data: utf8, encoding: .utf8) else {
                    throw LoadError.truncated("phrases pinyin utf8")
                }
                pyList.append(s)
                pos += length
            }
            result[phrase] = pyList
        }
        return result
    }

    /// Convenience loader for the uncompressed `.bin` payloads shipped at
    /// `FluidInference/kokoro-82m-coreml/ANE-zh/assets/`.
    public static func load(singlesURL: URL, phrasesURL: URL) throws -> MandarinPinyinDict {
        let singles = try parseSingles(try Data(contentsOf: singlesURL))
        let phrases = try parsePhrases(try Data(contentsOf: phrasesURL))
        return MandarinPinyinDict(phrases: phrases, singles: singles)
    }
}
