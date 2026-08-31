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
///   7. `-'s` stem + clitic for possessives whose stem is a known word
///      (`today's` → `today` + /z/), mirroring Misaki's `stem_s`
///   8. BART G2P CoreML fallback for OOV words (injected by the caller)
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

        // Fold typographic apostrophes to ASCII so in-word smart-quote
        // contractions (`we’re`) survive tokenization intact (issue #774).
        let prepared = Self.normalizeApostrophes(trimmed)

        var parts: [String] = []

        for token in Self.splitWords(prepared) {
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

    /// - Parameter allowFallback: when `false`, the BART G2P fallback is
    ///   skipped and an OOV word resolves to `nil`. Used by the possessive
    ///   rule, which — like Misaki's `stem_s` — only fires when the stem is a
    ///   *known* word; an OOV stem must leave the whole token on its original
    ///   path instead of quietly re-shaping it.
    private func resolveWord(
        _ word: String,
        allowFallback: Bool = true,
        fallback: (String) async throws -> [String]?
    ) async throws -> String? {
        let normalized = Self.normalizeKey(word)
        // Raw lower-cased spelling with hyphens intact. `normalizeKey` strips
        // hyphens, so this is the only form that can reach the lexicon's 3,459
        // hyphenated keys (`twenty-one`, `a-frame`, …) (issue #775).
        let lowered = word.lowercased()

        if let custom = customLexicon[word] ?? customLexicon[normalized] {
            return custom
        }

        // A few bundled case-sensitive entries don't read as letter names
        // even though uppercase callers expect them to (`AI` → blended
        // `ˈAˌI`, `US` → the lowercase-pronoun `ʌs` shape). Spell those out
        // before consulting the lexicon so they sound like `A I` / `U S`
        // (issue #710). Lowercase `us`/`ai` are untouched — the override
        // only matches the exact uppercase spelling.
        if EnglishInitialisms.letterNameOverrides.contains(word) {
            if let spelled = spellAsLetterNames(word) {
                return spelled
            }
            // Per-letter entries should always be present when the full
            // lexicon is loaded; if they aren't (e.g. a letter was filtered
            // out of the cache) the override below silently becomes the
            // blended shape it was meant to bypass — log so it isn't silent.
            Self.logger.warning(
                "Letter-name override '\(word)' unspellable (missing per-letter lexicon entries); "
                    + "falling back to the bundled pronunciation")
        }

        if let phonemes = caseSensitiveWordToPhonemes[word]
            ?? caseSensitiveWordToPhonemes[normalized]
            ?? wordToPhonemes[lowered]
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
        if EnglishInitialisms.isCandidate(word), let spelled = spellAsLetterNames(word) {
            return spelled
        }

        // A possessive / `-'s` clitic whose stem is a known word (`today's`,
        // `someone's`, `the boss's`). The lexicon stores the clitic `'s` as its
        // own entry and carries no glued `today's` key, so these always miss
        // above and the whole inflected token goes to BART G2P, which mangles
        // it (`someone's` → "Samian's"). Resolve the stem and append the clitic
        // by rule instead — same shape as Misaki's `Lexicon.stem_s`.
        if let possessive = try await resolvePossessive(
            word, lowered: lowered, fallback: fallback)
        {
            return possessive
        }

        // A hyphenated compound that missed every lexicon as a whole
        // (`tales-to-amaze`) — resolve each part and join, so it reads as
        // `tales to amaze` instead of BART G2P on the glued `talestoamaze`
        // (issue #775). Real lexicon compounds (`twenty-one`) already returned
        // above, so only genuine misses reach here.
        if word.contains("-"),
            let compound = try await resolveHyphenatedCompound(
                word, allowFallback: allowFallback, fallback: fallback)
        {
            return compound
        }

        guard allowFallback, !normalized.isEmpty else { return nil }
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

    /// Resolve a hyphenated compound that missed the lexicon by splitting on
    /// hyphens and resolving each part, joining the phoneme strings with a
    /// space (word boundary). Returns `nil` if the token isn't a multi-part
    /// compound or any part is unresolvable, so the caller falls back to
    /// whole-word G2P (issue #775). Parts contain no hyphens, so this does not
    /// recurse back into itself.
    private func resolveHyphenatedCompound(
        _ word: String,
        allowFallback: Bool = true,
        fallback: (String) async throws -> [String]?
    ) async throws -> String? {
        let parts = word.split(separator: "-", omittingEmptySubsequences: true).map(String.init)
        guard parts.count >= 2 else { return nil }

        var resolved: [String] = []
        resolved.reserveCapacity(parts.count)
        for part in parts {
            guard
                let ipa = try await resolveWord(
                    part, allowFallback: allowFallback, fallback: fallback),
                !ipa.isEmpty
            else {
                return nil
            }
            resolved.append(ipa)
        }
        return resolved.joined(separator: " ")
    }

    // MARK: - Possessive / `-'s` clitic

    /// Resolve a lower-cased token ending in `'s` as stem + `-s` clitic.
    ///
    /// Mirrors Misaki's `Lexicon.stem_s`, which only accepts the split when
    /// the stem is a known word — so an OOV stem returns `nil` here and the
    /// caller falls through to whole-token G2P exactly as before. The stem is
    /// resolved through the normal chain minus the G2P fallback, which keeps
    /// custom-lexicon overrides, letter-name spelling and hyphen splitting
    /// working for compounds like `mother-in-law's`.
    ///
    /// - Parameters:
    ///   - word: the token as written (apostrophes already folded to ASCII by
    ///     ``normalizeApostrophes``). Original case is preserved so the stem
    ///     can still reach case-sensitive entries (`NASA's`, `iPhone's`).
    ///   - lowered: `word.lowercased()`, so `TODAY'S` matches too.
    private func resolvePossessive(
        _ word: String,
        lowered: String,
        fallback: (String) async throws -> [String]?
    ) async throws -> String? {
        // `len(word) < 3` in Misaki: a bare `'s` (and anything shorter than
        // three characters) never stems.
        guard lowered.count >= 3, lowered.hasSuffix("'s") else { return nil }
        let stem = String(word.dropLast(2))
        guard !stem.isEmpty, !stem.hasSuffix("'") else { return nil }

        guard
            let stemIPA = try await resolveWord(stem, allowFallback: false, fallback: fallback),
            !stemIPA.isEmpty
        else {
            return nil
        }
        return stemIPA + Self.clitic(after: stemIPA)
    }

    /// Voiceless non-sibilant obstruents — the `-s` clitic devoices after
    /// these (`cat's` → `kˈæts`).
    private static let voicelessNonSibilants: Set<Character> = ["p", "t", "k", "f", "θ"]

    /// Sibilants — the clitic takes an epenthetic vowel after these
    /// (`boss's` → `bˈɑsᵻz`). Note the Misaki lexicon spells the affricates
    /// with the single-scalar ligatures `ʧ` / `ʤ`, not `tʃ` / `dʒ`.
    private static let sibilants: Set<Character> = ["s", "z", "ʃ", "ʒ", "ʧ", "ʤ"]

    /// The `-s` clitic phoneme for a stem, by English phonology — a direct
    /// port of Misaki's `Lexicon._s`. The US form of the epenthetic vowel is
    /// `ᵻ` (Misaki uses `ɪ` only when `british`); this frontend loads the US
    /// lexicon, and `ᵻ` is in the chain's `vocab.json`.
    static func clitic(after stemIPA: String) -> String {
        guard let last = stemIPA.last else { return "z" }
        if voicelessNonSibilants.contains(last) { return "s" }
        if sibilants.contains(last) { return "ᵻz" }
        return "z"
    }

    // MARK: - Letter-name initialisms (issue #710)

    /// Spell a token as a sequence of letter names using the per-letter
    /// entries in the case-sensitive lexicon (`FBI` → `ˈɛf bˈi ˈI`). See
    /// ``EnglishInitialisms/spell(_:letterTokens:render:separator:)`` —
    /// returns `nil` if any letter is missing so the caller falls through
    /// to its normal fallback rather than emitting a partial word.
    private func spellAsLetterNames(_ word: String) -> String? {
        EnglishInitialisms.spell(word) { caseSensitiveWordToPhonemes[$0] }
    }

    /// Typographic apostrophes that iOS smart punctuation and web/ebook
    /// content use in place of the ASCII `'` (issue #774).
    private static let smartApostrophes: Set<Character> = ["\u{2019}", "\u{2018}", "\u{02BC}"]

    /// Fold typographic apostrophes (`’` `‘` `ʼ`) to the ASCII apostrophe so
    /// `splitWords` and `normalizeKey` — which keep only U+0027 word-internal
    /// — don't split contractions like `we’re` into `we` + `re` (issue #774).
    static func normalizeApostrophes(_ text: String) -> String {
        guard text.contains(where: { smartApostrophes.contains($0) }) else {
            return text
        }
        return String(text.map { smartApostrophes.contains($0) ? "'" : $0 })
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
