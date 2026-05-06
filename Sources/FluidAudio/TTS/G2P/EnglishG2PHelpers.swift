import Foundation

/// Phoneme-resolution helpers shared by the Kokoro and StyleTTS2 G2P paths.
///
/// Both backends consume the same `us_lexicon_cache.json` (misaki gold dict
/// flattened into a JSON map) and need the same morphological-stemming and
/// spell-out fallbacks for OOV inputs. Each backend keeps its own orchestration
/// (Kokoro filters against an allowed token set + supports multilingual G2P;
/// StyleTTS2 is English-only with a misaki→espeak postpass) and calls into
/// these helpers as building blocks.
///
/// Helpers are pure — no model loading, no I/O. The lexicon is passed in by
/// the caller. `allowed: Set<String>?` is optional: pass nil to accept every
/// phoneme (StyleTTS2's behavior), or a vocab set to filter (Kokoro's
/// behavior, where the vocab is restricted to ~150 IPA tokens).
public enum EnglishG2PHelpers {

    // MARK: - Stem-based inflection

    /// Try to derive phonemes for an inflected word by stripping `-s`/`-ed`/
    /// `-ing` and looking up the stem in the lexicon, then reapplying the
    /// suffix phonetically. Returns nil if no inflected form matches or the
    /// stem yields no allowed phonemes.
    ///
    /// - Parameters:
    ///   - word: Lowercased, apostrophe-preserved word (e.g. `"jumped"`,
    ///     `"running"`, `"buses"`).
    ///   - lexicon: Lower-cased lexicon mapping `word` → phoneme array (the
    ///     `lower` map from `us_lexicon_cache.json`).
    ///   - allowed: Optional vocab filter applied to the stem phonemes before
    ///     suffixing. `nil` accepts everything.
    public static func stemInflected(
        _ word: String,
        lexicon: [String: [String]],
        allowed: Set<String>? = nil
    ) -> [String]? {
        if let result = stemS(word, lexicon: lexicon, allowed: allowed) { return result }
        if let result = stemEd(word, lexicon: lexicon, allowed: allowed) { return result }
        if let result = stemIng(word, lexicon: lexicon, allowed: allowed) { return result }
        return nil
    }

    /// Strip `-s`/`-es`/`-ies` and apply the phonetic plural / 3rd-person rule.
    private static func stemS(
        _ word: String,
        lexicon: [String: [String]],
        allowed: Set<String>?
    ) -> [String]? {
        guard word.count >= 3, word.hasSuffix("s") else { return nil }

        var stem: String?
        if !word.hasSuffix("ss"), lexicon[String(word.dropLast(1))] != nil {
            // word-s → word
            stem = String(word.dropLast(1))
        } else if word.hasSuffix("'s") || (word.count > 4 && word.hasSuffix("es") && !word.hasSuffix("ies")),
            lexicon[String(word.dropLast(2))] != nil
        {
            // word-es → word, word's → word
            stem = String(word.dropLast(2))
        } else if word.count > 4, word.hasSuffix("ies"),
            lexicon[String(word.dropLast(3)) + "y"] != nil
        {
            // word-ies → word-y
            stem = String(word.dropLast(3)) + "y"
        }

        guard let stem, var stemPhonemes = lexicon[stem] else { return nil }
        stemPhonemes = filtered(stemPhonemes, allowed: allowed)
        guard !stemPhonemes.isEmpty else { return nil }
        return appendSSuffix(to: stemPhonemes)
    }

    /// Strip `-ed`/`-d` and apply the phonetic past-tense rule.
    private static func stemEd(
        _ word: String,
        lexicon: [String: [String]],
        allowed: Set<String>?
    ) -> [String]? {
        guard word.count > 4, word.hasSuffix("ed") else { return nil }

        var stem: String?
        if lexicon[String(word.dropLast(1))] != nil {
            // word-e-d → word-e (e.g. "phrased" → "phrase", "rated" → "rate")
            // Check silent-e stems first to avoid matching shorter stems
            // like "rat" for "rated".
            stem = String(word.dropLast(1))
        } else if !word.hasSuffix("eed"), lexicon[String(word.dropLast(2))] != nil {
            // word-ed → word (e.g. "jumped" → "jump")
            stem = String(word.dropLast(2))
        }

        guard let stem, var stemPhonemes = lexicon[stem] else { return nil }
        stemPhonemes = filtered(stemPhonemes, allowed: allowed)
        guard !stemPhonemes.isEmpty else { return nil }
        return appendEdSuffix(to: stemPhonemes)
    }

    /// Strip `-ing` and apply the phonetic progressive rule.
    private static func stemIng(
        _ word: String,
        lexicon: [String: [String]],
        allowed: Set<String>?
    ) -> [String]? {
        guard word.count >= 5, word.hasSuffix("ing") else { return nil }

        var stem: String?
        if word.count > 5, lexicon[String(word.dropLast(3))] != nil {
            // word-ing → word (e.g. "jumping" → "jump")
            stem = String(word.dropLast(3))
        } else if lexicon[String(word.dropLast(3)) + "e"] != nil {
            // word-ing → word-e (e.g. "making" → "make")
            stem = String(word.dropLast(3)) + "e"
        } else if word.count > 5 {
            // Doubled consonant: word-Xing → word (e.g. "running" → "run")
            let base = String(word.dropLast(3))
            if base.count >= 2 {
                let lastChar = base.last!
                let secondLastIdx = base.index(base.endIndex, offsetBy: -2)
                let secondLastChar = base[secondLastIdx]
                let doublingConsonants: Set<Character> = Set("bcdgklmnprstvxz")
                if (lastChar == secondLastChar && doublingConsonants.contains(lastChar))
                    // "Xcking" → stem ends in c (e.g. "trafficking" → "traffic",
                    // "panicking" → "panic"). The k is an orthographic insert
                    // to preserve the /k/ sound, not part of the lexical stem.
                    || (lastChar == "k" && secondLastChar == "c")
                {
                    let stemCandidate = String(base.dropLast(1))
                    if lexicon[stemCandidate] != nil {
                        stem = stemCandidate
                    }
                }
            }
        }

        guard let stem, var stemPhonemes = lexicon[stem] else { return nil }
        stemPhonemes = filtered(stemPhonemes, allowed: allowed)
        guard !stemPhonemes.isEmpty else { return nil }
        return appendIngSuffix(to: stemPhonemes)
    }

    // MARK: - Phonetic suffix rules (US English)

    /// Vowel phoneme tokens that trigger `t→ɾ` flapping in American English
    /// (before `-ed`/`-ing`). Includes misaki single-codepoint diphthong
    /// shortcuts (`A=eɪ`, `I=aɪ`, `O=oʊ`, `W=aʊ`, `Y=ɔɪ`) and the
    /// corresponding espeak two-codepoint forms.
    private static let usTaus: Set<String> = [
        "A", "I", "O", "W", "Y",
        "i", "u", "æ", "ɑ", "ə", "ɛ", "ɪ", "ɹ", "ʊ", "ʌ",
        "eɪ", "aɪ", "oʊ", "aʊ", "ɔɪ",
    ]

    /// Voiceless consonant phoneme tokens that take the `-s` suffix.
    private static let voicelessSTokens: Set<String> = ["p", "t", "k", "f", "θ"]

    /// Sibilant phoneme tokens that take the `-ᵻz` suffix. Covers misaki
    /// (`ʧ`, `ʤ`) and espeak (`tʃ`, `dʒ`) affricate spellings.
    private static let sibilantSTokens: Set<String> = [
        "s", "z", "ʃ", "ʒ", "ʧ", "ʤ", "tʃ", "dʒ",
    ]

    /// Voiceless consonant phoneme tokens that take the `-t` past-tense suffix.
    private static let voicelessEdTokens: Set<String> = [
        "p", "k", "f", "θ", "ʃ", "s", "ʧ", "tʃ",
    ]

    /// Append `-s`/`-z`/`-ᵻz` based on the final phoneme of the stem.
    private static func appendSSuffix(to stem: [String]) -> [String] {
        guard let last = stem.last else { return stem }
        if voicelessSTokens.contains(last) {
            return stem + ["s"]
        } else if sibilantSTokens.contains(last) {
            return stem + ["ᵻ", "z"]
        }
        return stem + ["z"]
    }

    /// Append `-t`/`-d`/`-ᵻd` based on the final phoneme of the stem, with
    /// `t→ɾ` flapping before `-ᵻd`.
    private static func appendEdSuffix(to stem: [String]) -> [String] {
        guard let last = stem.last else { return stem }
        if voicelessEdTokens.contains(last) {
            return stem + ["t"]
        } else if last == "d" {
            return stem + ["ᵻ", "d"]
        } else if last != "t" {
            return stem + ["d"]
        }
        // Ends in "t": check for flapping (t → ɾ before ᵻd).
        if stem.count >= 2 {
            let secondLast = stem[stem.count - 2]
            if usTaus.contains(secondLast) {
                var result = Array(stem.dropLast())
                result.append("ɾ")
                result.append("ᵻ")
                result.append("d")
                return result
            }
        }
        return stem + ["ᵻ", "d"]
    }

    /// Append `-ɪŋ` with `t→ɾ` flapping when applicable.
    private static func appendIngSuffix(to stem: [String]) -> [String] {
        guard let last = stem.last else { return stem }
        // Flapping: vowel + t → vowel + ɾɪŋ
        if last == "t", stem.count >= 2 {
            let secondLast = stem[stem.count - 2]
            if usTaus.contains(secondLast) {
                var result = Array(stem.dropLast())
                result.append("ɾ")
                result.append("ɪ")
                result.append("ŋ")
                return result
            }
        }
        return stem + ["ɪ", "ŋ"]
    }

    // MARK: - Numeric spell-out

    private static let decimalDigits = CharacterSet.decimalDigits

    private static let spellOutFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .spellOut
        f.locale = Locale(identifier: "en_US")
        return f
    }()

    /// If `token` is a pure numeric string (e.g. `"2025"`), return its
    /// spell-out per-word components (`["two", "thousand", "twenty", "five"]`).
    /// Returns `nil` for empty, non-numeric, or non-decimal inputs.
    public static func spelledOutTokens(for token: String) -> [String]? {
        guard !token.isEmpty else { return nil }
        if token.rangeOfCharacter(from: decimalDigits.inverted) != nil {
            return nil
        }
        guard let value = Int(token) else { return nil }
        guard let spelled = spellOutFormatter.string(from: NSNumber(value: value)) else { return nil }
        let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "-"))
        let components =
            spelled
            .lowercased()
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }
        return components.isEmpty ? nil : components
    }

    // MARK: - Per-letter fallback

    /// Hardcoded single-letter pronunciation table used as a last-resort
    /// fallback when the lexicon, stemmer, BART G2P, and spell-out paths all
    /// miss. Phonemes are in misaki convention.
    public static let letterPronunciations: [String: [String]] = [
        "a": ["e", "ɪ"],
        "b": ["b", "i"],
        "c": ["s", "i"],
        "d": ["d", "i"],
        "e": ["i"],
        "f": ["ɛ", "f"],
        "g": ["ʤ", "i"],
        "h": ["e", "ɪ", "ʧ"],
        "i": ["a", "ɪ"],
        "j": ["ʤ", "e"],
        "k": ["k", "e"],
        "l": ["ɛ", "l"],
        "m": ["ɛ", "m"],
        "n": ["ɛ", "n"],
        "o": ["o"],
        "p": ["p", "i"],
        "q": ["k", "j", "u"],
        "r": ["ɑ", "ɹ"],
        "s": ["ɛ", "s"],
        "t": ["t", "i"],
        "u": ["j", "u"],
        "v": ["v", "i"],
        "w": ["d", "ʌ", "b", "əl", "j", "u"],
        "x": ["ɛ", "k", "s"],
        "y": ["w", "a", "ɪ"],
        "z": ["z", "i"],
    ]

    // MARK: - Internal

    /// Filter `tokens` against `allowed`. If `allowed` is nil, returns
    /// `tokens` unchanged.
    private static func filtered(_ tokens: [String], allowed: Set<String>?) -> [String] {
        guard let allowed else { return tokens }
        return tokens.filter { allowed.contains($0) }
    }
}
