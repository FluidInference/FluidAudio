import Foundation

/// Mandarin tokenizer: jieba segmentation → pypinyin lookup → tone/letter split →
/// phoneme ids via `mandarin_phoneme_token2id.json`.
///
/// This Swift port uses the pre-built dictionaries emitted by
/// `mobius/.../export_pypinyin.py` and `export_tokenizers.py`:
///
///   - `mandarin_pypinyin_phrase_dict.json` — phrase → [pinyin] multi-char hits.
///   - `mandarin_pypinyin_char_dict.json`   — single char → [pinyin] fallback.
///   - `mandarin_jieba_dict.json`           — user-dict entries with frequencies.
///   - `mandarin_phoneme_pinyin_dict.json`  — pinyin (with tone digit) → [IPA phonemes].
///   - `mandarin_phoneme_tone_dict.json`    — tone digit → tone token.
///   - `mandarin_phoneme_ascii_letter_dict.json` — ASCII letter → token string.
///   - `mandarin_phoneme_token2id.json`     — final token string → id.
///
/// Segmentation strategy: forward maximum-matching over the phrase dict, with a
/// per-character fallback. Full jieba HMM fallback is not ported here — OOV
/// characters collapse to their single-char pypinyin entry. This handles the
/// majority of real-world text; tricky edge cases (unseen words) should use the
/// `MagpiePhonemeTokens` bypass path.
public struct MagpieMandarinTokenizer: MagpieLanguageTokenizer {

    public let language: MagpieLanguage = .mandarin

    private let phraseDict: [String: [String]]
    private let charDict: [String: [String]]
    private let pinyinDict: [String: [String]]
    private let toneDict: [String: String]
    private let asciiLetterDict: [String: String]
    private let token2id: [String: Int32]
    /// Characters (max length) covered by phraseDict — used to bound MaxMatch search.
    private let maxPhraseLength: Int

    public init(tokenizerDir: URL) throws {
        let base = MagpieTokenizerFiles.tokenizerName(for: .mandarin)
        self.phraseDict =
            (try? Self.loadDict(tokenizerDir.appendingPathComponent("mandarin_pypinyin_phrase_dict.json"))) ?? [:]
        self.charDict =
            (try? Self.loadDict(tokenizerDir.appendingPathComponent("mandarin_pypinyin_char_dict.json"))) ?? [:]
        self.pinyinDict = try Self.loadDict(
            tokenizerDir.appendingPathComponent("\(base)_pinyin_dict.json"))
        self.toneDict = try Self.loadStringDict(
            tokenizerDir.appendingPathComponent("\(base)_tone_dict.json"))
        self.asciiLetterDict = try Self.loadStringDict(
            tokenizerDir.appendingPathComponent("\(base)_ascii_letter_dict.json"))
        self.token2id = try MagpiePhonemeTokenizer.loadTokenMap(
            tokenizerDir.appendingPathComponent("\(base)_token2id.json"))

        var maxLen = 1
        for key in phraseDict.keys where key.count > maxLen {
            maxLen = key.count
        }
        self.maxPhraseLength = maxLen
    }

    public func encode(_ text: String) throws -> [Int32] {
        var ids: [Int32] = []
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            // Forward-maximum-match against phraseDict.
            var matched = false
            let upper = min(maxPhraseLength, chars.count - i)
            if upper > 1 {
                for len in stride(from: upper, through: 2, by: -1) {
                    let phrase = String(chars[i..<(i + len)])
                    if let pinyin = phraseDict[phrase] {
                        appendPinyin(pinyin, into: &ids)
                        i += len
                        matched = true
                        break
                    }
                }
            }
            if matched { continue }

            let single = String(chars[i])
            if let pinyin = charDict[single] {
                appendPinyin(pinyin, into: &ids)
            } else if let letter = asciiLetterDict[single], let id = token2id[letter] {
                ids.append(id)
            } else if let id = token2id[single] {
                ids.append(id)
            }
            // else: silently drop (matches NeMo behavior for punctuation / unknown).
            i += 1
        }
        return ids
    }

    public func encodeIpaTokens(_ tokens: [String]) throws -> [Int32] {
        var ids: [Int32] = []
        for tok in tokens {
            guard let id = token2id[tok] else {
                throw MagpieError.invalidConstants(
                    "IPA override token '\(tok)' is not in mandarin token2id map")
            }
            ids.append(id)
        }
        return ids
    }

    // MARK: - Pinyin → phoneme expansion

    private func appendPinyin(_ pinyinList: [String], into ids: inout [Int32]) {
        for pinyin in pinyinList {
            // pinyin is usually "ni3" (initial+final+tone digit). Split off trailing digit.
            let (stem, tone) = splitTone(pinyin)
            if let phones = pinyinDict[stem] {
                for p in phones {
                    if let id = token2id[p] { ids.append(id) }
                }
            } else {
                // Fallback: emit stem as-is if present in token2id.
                if let id = token2id[stem] { ids.append(id) }
            }
            if let toneDigit = tone, let toneTok = toneDict[toneDigit], let id = token2id[toneTok] {
                ids.append(id)
            }
        }
    }

    private func splitTone(_ pinyin: String) -> (stem: String, tone: String?) {
        guard let last = pinyin.last, last.isNumber else { return (pinyin, nil) }
        let stem = String(pinyin.dropLast())
        return (stem, String(last))
    }

    // MARK: - Loaders

    private static func loadDict(_ url: URL) throws -> [String: [String]] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieError.tokenizerDataMissing(
                language: "mandarin", file: url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: [String]].self, from: data)
    }

    private static func loadStringDict(_ url: URL) throws -> [String: String] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieError.tokenizerDataMissing(
                language: "mandarin", file: url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: String].self, from: data)
    }
}
