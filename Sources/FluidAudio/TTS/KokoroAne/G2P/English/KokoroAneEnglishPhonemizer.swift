import Foundation

/// English text frontend for the KokoroAne 7-stage chain.
///
/// Word resolution order (mirrors `StyleTTS2Phonemizer` and Kokoro's
/// Misaki frontend):
///   1. caller-supplied custom lexicon (case-sensitive, then lower-cased)
///   2. letter-name overrides for bundled entries that don't read as
///      letter names (`AI`, `US`) — spelled out from per-letter entries
///      (issue #710)
///   3. case-sensitive Misaki lexicon hit on the original spelling
///      (proper nouns, abbreviations like `NATO`)
///   4. case-sensitive hit on the normalized lower-case form
///   5. lower-cased Misaki lexicon hit — this is what gives function
///      words their weak forms (`to` → `tu`), instead of the BART G2P
///      citation form (`tˈO`) that over-stresses them (issue #691)
///   6. strict ASCII all-caps initialisms (`FBI`, `ATP`) spelled as
///      letter names after a full lexicon miss (issue #710)
///   7. BART G2P CoreML fallback for OOV words (injected by the caller)
///
/// Punctuation supported by the chain's `vocab.json` (`, . ! ? ; …` etc.)
/// is preserved and attached to the preceding word — Kokoro treats those
/// tokens as prosody/pause cues, matching upstream `KPipeline.g2p` output.
/// Unlike the StyleTTS2 frontend, Misaki diphthong shorthand (`A O I Y W`)
/// is NOT expanded: the laishere vocab carries those tokens directly.
struct KokoroAneEnglishPhonemizer: Sendable {

    private static let logger = AppLogger(category: "KokoroAneEnglishPhonemizer")

    /// Lower-cased word → ordered Misaki phoneme tokens (pre-filtered
    /// against the chain vocab at load time by `LexiconAssetCache`).
    let wordToPhonemes: [String: [String]]

    /// Original-case word → phoneme tokens (`"AI"`, `"iPhone"`, …).
    let caseSensitiveWordToPhonemes: [String: [String]]

    /// Caller-supplied overrides (word → IPA string), checked before the
    /// Misaki lexicon. Exact spelling wins over the lower-cased form.
    let customLexicon: [String: String]

    /// Punctuation characters the loaded `vocab.json` can encode.
    /// Characters outside this set are dropped (they would be silently
    /// skipped at `KokoroAneVocab.encode` anyway).
    let allowedPunctuation: Set<Character>

    init(
        wordToPhonemes: [String: [String]] = [:],
        caseSensitiveWordToPhonemes: [String: [String]] = [:],
        customLexicon: [String: String] = [:],
        allowedPunctuation: Set<Character> = []
    ) {
        self.wordToPhonemes = wordToPhonemes
        self.caseSensitiveWordToPhonemes = caseSensitiveWordToPhonemes
        self.customLexicon = customLexicon
        self.allowedPunctuation = allowedPunctuation
    }

    /// Convert text to a Misaki-style IPA string. Words are joined with
    /// single spaces; kept punctuation attaches to the preceding word
    /// (`"Hello, world!"` → `"həlˈO, wˈɜɹld!"` shape).
    ///
    /// - Parameter fallback: per-word G2P for words missing from every
    ///   lexicon. Receives the normalized (lower-cased) spelling. `nil`
    ///   return skips the word with a warning; a thrown error aborts.
    /// - Throws: `KokoroAneError.inputProcessingFailed` when the input is
    ///   empty or nothing could be resolved.
    func phonemize(
        _ text: String,
        fallback: (String) async throws -> [String]?
    ) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }

        var parts: [String] = []

        for token in Self.splitWords(trimmed) {
            if token.isEmpty { continue }

            // Punctuation token (single non-word char from the splitter).
            if token.count == 1, let ch = token.first, !ch.isLetter, !ch.isNumber {
                guard allowedPunctuation.contains(ch) else { continue }
                // Attach to the preceding word — Kokoro's vocab encodes
                // punctuation as its own prosody token, but Misaki output
                // never puts a space before it.
                if parts.isEmpty {
                    parts.append(String(ch))
                } else {
                    parts[parts.count - 1].append(ch)
                }
                continue
            }

            if let ipa = try await resolveWord(token, fallback: fallback) {
                parts.append(ipa)
            }
        }

        let joined = parts.joined(separator: " ")
        if joined.isEmpty {
            throw KokoroAneError.inputProcessingFailed(
                "produced no phonemes for input '\(trimmed)'")
        }
        return joined
    }

    // MARK: - Word resolution

    private func resolveWord(
        _ word: String,
        fallback: (String) async throws -> [String]?
    ) async throws -> String? {
        let normalized = Self.normalizeKey(word)

        if let custom = customLexicon[word] ?? customLexicon[normalized] {
            return custom
        }

        // A few bundled case-sensitive entries don't read as letter names
        // even though uppercase callers expect them to (`AI` → blended
        // `ˈAˌI`, `US` → the lowercase-pronoun `ʌs` shape). Spell those out
        // before consulting the lexicon so they sound like `A I` / `U S`
        // (issue #710). Lowercase `us`/`ai` are untouched — the override
        // only matches the exact uppercase spelling.
        if Self.letterNameOverrides.contains(word), let spelled = spellAsLetterNames(word) {
            return spelled
        }

        if let phonemes = caseSensitiveWordToPhonemes[word]
            ?? caseSensitiveWordToPhonemes[normalized]
            ?? wordToPhonemes[normalized],
            !phonemes.isEmpty
        {
            return phonemes.joined()
        }

        // After a full lexicon miss, read strict ASCII all-caps tokens of a
        // small length range as letter-name initialisms (`FBI`, `ATP`)
        // instead of letting BART G2P sound them out as a word (issue #710).
        // Known acronyms (`NASA`, `FIFA`, `OK`, `COVID`) keep their bundled
        // pronunciations because they're resolved above as lexicon hits.
        if Self.isInitialismCandidate(word), let spelled = spellAsLetterNames(word) {
            return spelled
        }

        guard !normalized.isEmpty else { return nil }
        do {
            if let phonemes = try await fallback(normalized), !phonemes.isEmpty {
                return phonemes.joined()
            }
            Self.logger.warning("G2P returned nil for word '\(normalized)' — skipping")
            return nil
        } catch {
            Self.logger.warning("G2P failed on word '\(normalized)': \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: - Letter-name initialisms (issue #710)

    /// Exact uppercase spellings whose bundled lexicon entry is not the
    /// letter-name reading callers expect. These bypass the Misaki lexicon
    /// and are spelled out from the per-letter entries instead.
    private static let letterNameOverrides: Set<String> = ["AI", "US"]

    /// Length bounds for spelling unknown all-caps tokens as letter names.
    private static let initialismLengthRange = 2...5

    /// A strict ASCII all-caps token (`FBI`, `ATP`) within the length range
    /// — the only shape we spell out after a lexicon miss. Anything with a
    /// digit, hyphen, apostrophe, or non-ASCII letter is excluded so brand
    /// and proper-name pronunciations aren't disturbed.
    static func isInitialismCandidate(_ word: String) -> Bool {
        guard Self.initialismLengthRange.contains(word.count) else { return false }
        return word.allSatisfy { $0.isASCII && $0.isUppercase && $0.isLetter }
    }

    /// Spell a token as a sequence of letter names, reading each uppercase
    /// letter's pronunciation from the case-sensitive lexicon and joining
    /// them with spaces so each letter is its own prosodic unit (`FBI` →
    /// `ˈɛf bˈi ˈI`). Returns `nil` if any letter is missing from the
    /// lexicon (e.g. the G2P-only degraded path) so the caller falls
    /// through to its normal fallback rather than emitting a partial word.
    private func spellAsLetterNames(_ word: String) -> String? {
        var letters: [String] = []
        letters.reserveCapacity(word.count)
        for ch in word {
            guard let phonemes = caseSensitiveWordToPhonemes[String(ch)], !phonemes.isEmpty else {
                return nil
            }
            letters.append(phonemes.joined())
        }
        guard !letters.isEmpty else { return nil }
        return letters.joined(separator: " ")
    }

    /// Lowercase + strip non-letter/digit/apostrophe chars so we hit the
    /// same Misaki cache entries the preprocessor wrote.
    static func normalizeKey(_ word: String) -> String {
        let lowered = word.lowercased()
        let allowedSet = CharacterSet.letters.union(.decimalDigits)
            .union(CharacterSet(charactersIn: "'"))
        let filtered = lowered.unicodeScalars.filter { allowedSet.contains($0) }
        return String(String.UnicodeScalarView(filtered))
    }

    // MARK: - Word splitter

    private static let knownLeadingApostropheWords: Set<String> = [
        "'cause", "'em", "'til", "'tis", "'twas", "'twere",
    ]

    /// Emit runs of letters/digits (internal apostrophes and hyphens stay
    /// inside words: `don't`, `twenty-one`), single punctuation chars as
    /// their own tokens, and drop whitespace. Same shape as the StyleTTS2
    /// frontend's imitation of `nltk.word_tokenize`.
    static func splitWords(_ text: String) -> [String] {
        var out: [String] = []
        var current: String = ""

        @inline(__always) func flushCurrent() {
            if !current.isEmpty {
                out.append(current)
                current.removeAll(keepingCapacity: true)
            }
        }

        for index in text.indices {
            let ch = text[index]
            if ch.isWhitespace {
                flushCurrent()
            } else if ch == "'" {
                let nextIndex = text.index(after: index)
                let nextIsWord =
                    nextIndex < text.endIndex
                    && (text[nextIndex].isLetter || text[nextIndex].isNumber)
                if !current.isEmpty && nextIsWord {
                    current.append(ch)
                } else if current.isEmpty && Self.startsKnownLeadingApostropheWord(in: text, at: index) {
                    current.append(ch)
                } else {
                    flushCurrent()
                    out.append(String(ch))
                }
            } else if ch.isLetter || ch.isNumber || ch == "-" {
                current.append(ch)
            } else {
                flushCurrent()
                out.append(String(ch))
            }
        }
        flushCurrent()
        return out
    }

    private static func startsKnownLeadingApostropheWord(
        in text: String,
        at apostropheIndex: String.Index
    ) -> Bool {
        let nextIndex = text.index(after: apostropheIndex)
        guard nextIndex < text.endIndex, text[nextIndex].isLetter else {
            return false
        }

        var endIndex = nextIndex
        while endIndex < text.endIndex, text[endIndex].isLetter {
            endIndex = text.index(after: endIndex)
        }

        let candidate = "'" + text[nextIndex..<endIndex].lowercased()
        return knownLeadingApostropheWords.contains(candidate)
    }
}
