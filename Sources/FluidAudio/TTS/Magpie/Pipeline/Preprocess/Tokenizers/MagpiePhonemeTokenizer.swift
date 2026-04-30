import Foundation

/// Phoneme / G2P based tokenizer shared by English, Spanish, German, Italian, and
/// Vietnamese (all use IPA phoneme dictionaries emitted by NeMo).
///
/// Behavior:
/// 1. Normalize text (lowercase for English/German, keep case otherwise — matches
///    NeMo's `grapheme_case` defaults).
/// 2. Word-tokenize on whitespace + punctuation.
/// 3. For each word:
///    - If the phoneme_dict has the word → emit `" "` separator then each IPA
///      phoneme as its own token id (via `token2id`).
///    - Otherwise emit the raw characters as individual ids.
/// 4. Preserve punctuation as literal token ids when present in `token2id`.
///
/// This is a pragmatic port of NeMo's EnglishPhonemesTokenizer / IPATokenizer that
/// trades the full feature set for deterministic Swift-side lookup. Callers who
/// need bit-exact parity should supply `MagpiePhonemeTokens` directly.
public struct MagpiePhonemeTokenizer: MagpieLanguageTokenizer {

    public let language: MagpieLanguage
    private let phonemeDict: [String: [String]]
    private let heteronyms: Set<String>
    private let token2id: [String: Int32]

    public init(language: MagpieLanguage, tokenizerDir: URL) throws {
        self.language = language
        let base = MagpieTokenizerFiles.tokenizerName(for: language)
        self.token2id = try Self.loadTokenMap(
            tokenizerDir.appendingPathComponent("\(base)_token2id.json"))
        self.phonemeDict = try Self.loadPhonemeDict(
            tokenizerDir.appendingPathComponent("\(base)_phoneme_dict.json"))

        if language == .german {
            let hetURL = tokenizerDir.appendingPathComponent("\(base)_heteronyms.json")
            self.heteronyms = (try? Self.loadHeteronyms(hetURL)) ?? []
        } else {
            self.heteronyms = []
        }
    }

    public func encode(_ text: String) throws -> [Int32] {
        var ids: [Int32] = []
        let normalized = normalize(text)
        let tokens = splitWords(normalized)

        for piece in tokens {
            switch piece {
            case .word(let word):
                let key = caseKey(for: word)
                if heteronyms.contains(key) {
                    // Heteronym: fall back to grapheme-level encoding.
                    appendGraphemes(word, into: &ids)
                } else if let phones = phonemeDict[key] {
                    // Inter-word spaces come from `.separator(" ")` below; do not
                    // prepend an extra space here. NeMo's IPATokenizer relies on
                    // raw whitespace from the input (`pad_with_space=false` for
                    // english_phoneme) for word boundaries.
                    for p in phones {
                        if let id = token2id[p] { ids.append(id) }
                    }
                } else {
                    appendGraphemes(word, into: &ids)
                }
            case .separator(let sep):
                if let id = token2id[sep] { ids.append(id) }
            }
        }
        return ids
    }

    public func encodeIpaTokens(_ tokens: [String]) throws -> [Int32] {
        var ids: [Int32] = []
        appendSpace(&ids)
        for p in tokens {
            guard let id = token2id[p] else {
                throw MagpieError.invalidConstants(
                    "IPA override token '\(p)' is not in \(language.rawValue) token2id map")
            }
            ids.append(id)
        }
        return ids
    }

    // MARK: - Helpers

    /// Match NeMo `tokenizer_metadata.json` `grapheme_case`:
    ///   english_phoneme: upper, spanish_phoneme: upper, german_phoneme: mixed.
    private func normalize(_ text: String) -> String {
        switch language {
        case .english, .spanish:
            return text.uppercased()
        default:
            return text
        }
    }

    private func caseKey(for word: String) -> String {
        switch language {
        case .english, .spanish:
            return word.uppercased()
        default:
            return word
        }
    }

    private enum Piece {
        case word(String)
        case separator(String)
    }

    /// Split input into word pieces and punctuation/whitespace separators.
    private func splitWords(_ text: String) -> [Piece] {
        var pieces: [Piece] = []
        var current = ""
        for ch in text {
            if ch.isLetter {
                current.append(ch)
            } else {
                if !current.isEmpty {
                    pieces.append(.word(current))
                    current = ""
                }
                let s = String(ch)
                if ch.isWhitespace {
                    pieces.append(.separator(" "))
                } else {
                    pieces.append(.separator(s))
                }
            }
        }
        if !current.isEmpty { pieces.append(.word(current)) }
        return pieces
    }

    private func appendSpace(_ ids: inout [Int32]) {
        if let id = token2id[" "] { ids.append(id) }
    }

    private func appendGraphemes(_ word: String, into ids: inout [Int32]) {
        for ch in word {
            if let id = token2id[String(ch)] {
                ids.append(id)
            }
        }
    }

    // MARK: - JSON loaders

    static func loadTokenMap(_ url: URL) throws -> [String: Int32] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieError.tokenizerDataMissing(
                language: url.deletingPathExtension().lastPathComponent, file: url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: Int].self, from: data)
        var out: [String: Int32] = [:]
        out.reserveCapacity(raw.count)
        for (k, v) in raw { out[k] = Int32(v) }
        return out
    }

    static func loadPhonemeDict(_ url: URL) throws -> [String: [String]] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieError.tokenizerDataMissing(
                language: url.deletingPathExtension().lastPathComponent, file: url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: [String]].self, from: data)
    }

    static func loadHeteronyms(_ url: URL) throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        let data = try Data(contentsOf: url)
        let list = try JSONDecoder().decode([String].self, from: data)
        return Set(list)
    }
}
