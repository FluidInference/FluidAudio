import Foundation

/// Character-level tokenizer used by French and Hindi.
///
/// NeMo's `ChineseCharsTokenizer` equivalent maps each Unicode character to its
/// id via `token2id.json`. Unknown characters are silently dropped (matching the
/// NeMo default `add_blank_at = False` behavior). Whitespace is mapped to `" "`
/// when present in the vocab.
public struct MagpieCharTokenizer: MagpieLanguageTokenizer {

    public let language: MagpieLanguage
    private let token2id: [String: Int32]

    public init(language: MagpieLanguage, tokenizerDir: URL) throws {
        self.language = language
        let base = MagpieTokenizerFiles.tokenizerName(for: language)
        self.token2id = try MagpiePhonemeTokenizer.loadTokenMap(
            tokenizerDir.appendingPathComponent("\(base)_token2id.json"))
    }

    public func encode(_ text: String) throws -> [Int32] {
        var ids: [Int32] = []
        for ch in text {
            let key = String(ch)
            if let id = token2id[key] {
                ids.append(id)
            }
        }
        return ids
    }

    public func encodeIpaTokens(_ tokens: [String]) throws -> [Int32] {
        var ids: [Int32] = []
        for tok in tokens {
            guard let id = token2id[tok] else {
                throw MagpieError.invalidConstants(
                    "IPA override token '\(tok)' is not in \(language.rawValue) token2id map")
            }
            ids.append(id)
        }
        return ids
    }
}
