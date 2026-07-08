import Foundation

/// English-only StyleTTS2 phonemizer.
///
/// Per-word resolution uses the preprocessed Misaki lexicon cache
/// (`us_lexicon_cache.json`) loaded via `LexiconAssetCache`. Token
/// strings in those maps are pre-filtered against StyleTTS2's character
/// vocabulary at load time, so anything that comes back is directly
/// encodable by `StyleTTS2TextCleaner`.
///
/// Raw text is first passed through `EnglishTextNormalizer` (strict
/// standalone numbers / ordinals / decimals / 12-hour times → words, #711).
///
/// Per-word resolution order:
///   1. letter-name overrides for bundled entries that don't read as letter
///      names (`AI`, `US`) — spelled from per-letter entries (#710)
///   2. case-sensitive lexicon hit on the original spelling (proper nouns,
///      abbreviations like `NATO`)
///   3. case-sensitive lexicon hit on the normalized lower-case form
///   4. lower-case lexicon hit
///   5. strict ASCII all-caps initialisms (`FBI`, `ATP`) spelled as letter
///      names after a full lexicon miss (#710)
///   6. BART grapheme-to-phoneme CoreML model
///      (`G2PEncoder.mlmodelc` / `G2PDecoder.mlmodelc`, fetched from the
///      kokoro repo) — last resort for OOV words.
///
/// > Important: callers with a higher-quality phonemizer (e.g. server-side
/// > espeak) can still bypass everything via
/// > `StyleTTS2Manager.synthesize(ipa:referenceAudioURL:...)`.
public struct StyleTTS2Phonemizer: Sendable {

    private let logger = AppLogger(category: "StyleTTS2Phonemizer")

    /// Lower-cased word → ordered list of phoneme tokens (already filtered
    /// against StyleTTS2's character vocab by `LexiconCache`).
    private let wordToPhonemes: [String: [String]]

    /// Original-case word → token list. Misaki ships proper nouns and
    /// abbreviations in their canonical case (`"AI"`, `"iPhone"`); we hit
    /// this map first so we get the right pronunciation before falling
    /// back to the lower-cased view.
    private let caseSensitiveWordToPhonemes: [String: [String]]

    public init(
        wordToPhonemes: [String: [String]] = [:],
        caseSensitiveWordToPhonemes: [String: [String]] = [:]
    ) {
        self.wordToPhonemes = wordToPhonemes
        self.caseSensitiveWordToPhonemes = caseSensitiveWordToPhonemes
    }

    /// Phonemize a sentence and encode it into the StyleTTS2 token IDs.
    /// Returns the encoded token list (with the leading-pad token already
    /// inserted) ready for the `text_encoder` stage.
    ///
    /// - Throws: `StyleTTS2Error.phonemizationFailed` if no word in the
    ///   input can be resolved by the lexicon or the G2P fallback.
    public func encode(_ text: String) async throws -> [Int32] {
        let phonemeString = try await phonemize(text)
        return StyleTTS2TextCleaner.encode(phonemeString)
    }

    /// Convert text to a plain IPA string (no token IDs). Words are
    /// joined with single spaces — `StyleTTS2TextCleaner` accepts space
    /// as a real symbol so the model gets word boundaries.
    public func phonemize(_ text: String) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return ""
        }

        // Raw-text normalization first: the native NeMo TN pass when the host
        // links `libtext_processing_rs`, else the conservative always-on
        // baseline (issue #711). Shared with KokoroAne via
        // ``EnglishTextNormalizer/normalizeForFrontend(_:)``.
        let normalized = EnglishTextNormalizer.normalizeForFrontend(trimmed)

        let words = splitWords(normalized)
        var ipaParts: [String] = []
        ipaParts.reserveCapacity(words.count)

        var anyResolved = false
        var lexiconHits = 0
        var g2pHits = 0

        for word in words {
            if word.isEmpty { continue }
            // Punctuation passes through verbatim — TextCleaner has direct
            // entries for `; : , . ! ? ¡ ¿ — … " « » " " ` and space.
            // Counts as "resolved" so a punctuation-only utterance doesn't
            // trigger the phonemization-failed throw.
            if word.allSatisfy({ StyleTTS2TextCleaner.punctuation.contains($0) }) {
                ipaParts.append(word)
                anyResolved = true
                continue
            }

            // 1. Letter-name overrides for bundled entries that don't read
            //    as letter names (`AI` → blended, `US` → pronoun shape).
            //    Spell from the per-letter entries before the lexicon so they
            //    sound like `A I` / `U S` (issue #710). Lowercase `us`/`ai`
            //    are untouched — the override is exact-uppercase only.
            if EnglishInitialisms.letterNameOverrides.contains(word),
                let spelled = spellAsLetterNames(word)
            {
                ipaParts.append(spelled)
                anyResolved = true
                lexiconHits += 1
                continue
            }
            if EnglishInitialisms.letterNameOverrides.contains(word) {
                logger.notice(
                    "Letter-name override '\(word)' unspellable (missing per-letter lexicon "
                        + "entries); falling back to the bundled pronunciation")
            }

            // 2. Misaki lexicon-cache lookup (Kokoro pattern).
            if let phonemes = lookupLexicon(word: word), !phonemes.isEmpty {
                ipaParts.append(Self.expandMisakiShorthand(phonemes.joined()))
                anyResolved = true
                lexiconHits += 1
                continue
            }

            // 3. Strict ASCII all-caps initialisms (`FBI`, `ATP`) spelled as
            //    letter names after a full lexicon miss, before G2P sounds
            //    them out as a word (issue #710). Lexicon-backed acronyms
            //    (`NASA`, `FIFA`) keep their bundled pronunciation above.
            if EnglishInitialisms.isCandidate(word), let spelled = spellAsLetterNames(word) {
                ipaParts.append(spelled)
                anyResolved = true
                lexiconHits += 1
                continue
            }

            // 4. BART G2P CoreML fallback for OOV words (Kokoro's
            //    `G2PEncoder.mlmodelc` + `G2PDecoder.mlmodelc`).
            do {
                let phonemes = try await G2PModel.shared.phonemize(word: word)
                if let phonemes, !phonemes.isEmpty {
                    ipaParts.append(Self.expandMisakiShorthand(phonemes.joined()))
                    anyResolved = true
                    g2pHits += 1
                } else {
                    // Degraded fallback: pass the grapheme through. The
                    // decoder's vocab includes ASCII letters, so this still
                    // produces *something* rather than dropping the word
                    // outright (which would shift alignment).
                    logger.notice("G2P returned nil for '\(word)'; passing graphemes")
                    ipaParts.append(word)
                }
            } catch {
                logger.warning("G2P failed for '\(word)': \(error); passing graphemes")
                ipaParts.append(word)
            }
        }

        if !wordToPhonemes.isEmpty {
            logger.debug(
                "Phonemized \(words.count) tokens — lexicon hits: \(lexiconHits), G2P fallback: \(g2pHits)"
            )
        }

        if !anyResolved {
            throw StyleTTS2Error.phonemizationFailed(
                "no words resolved by lexicon or G2P (input='\(text.prefix(40))')")
        }

        return ipaParts.joined(separator: " ")
    }

    // MARK: - Misaki → espeak shorthand expansion
    //
    // Misaki/Kokoro phonemes use 5 ASCII uppercase chars as single-character
    // shorthand for English diphthongs. StyleTTS2 was trained on espeak
    // transcriptions where the same diphthongs are written as their two
    // component IPA chars. Without this expansion the encoder treats e.g.
    // `O` as the Latin uppercase letter (token id 30) instead of /oʊ/, and
    // every word containing /eɪ/, /oʊ/, /aɪ/, /ɔɪ/, /aʊ/ is rendered as
    // gibberish. Confirmed by ASR round-trip — expansion turns garbled
    // output into intelligible speech.
    private static let misakiShorthand: [Character: String] = [
        "A": "eɪ",
        "O": "oʊ",
        "I": "aɪ",
        "Y": "ɔɪ",
        "W": "aʊ",
    ]

    static func expandMisakiShorthand(_ ipa: String) -> String {
        var out = ""
        out.reserveCapacity(ipa.count)
        for ch in ipa {
            if let exp = misakiShorthand[ch] {
                out.append(exp)
            } else {
                out.append(ch)
            }
        }
        return out
    }

    // MARK: - Letter-name initialisms (issue #710)

    /// Spell `word` as a sequence of letter names using the per-letter
    /// entries in the case-sensitive lexicon (`FBI` → `ˈɛf bˈi ˈaɪ`). The
    /// single-letter entries carry Misaki shorthand (`I` for /aɪ/), so each
    /// letter is run through ``expandMisakiShorthand(_:)`` like every other
    /// lexicon hit. Returns `nil` if any letter is missing so the caller
    /// falls through rather than emitting a partial word.
    private func spellAsLetterNames(_ word: String) -> String? {
        EnglishInitialisms.spell(
            word,
            letterTokens: { caseSensitiveWordToPhonemes[$0] },
            render: { Self.expandMisakiShorthand($0.joined()) }
        )
    }

    // MARK: - Lexicon lookup

    /// Lexicon resolution: case-sensitive hit on the original spelling
    /// first, then on the lower-cased form, then the lower-cased-only
    /// map.
    private func lookupLexicon(word: String) -> [String]? {
        if let phones = caseSensitiveWordToPhonemes[word] { return phones }
        let normalized = normalizeKey(word)
        if normalized.isEmpty { return nil }
        if let phones = caseSensitiveWordToPhonemes[normalized] { return phones }
        return wordToPhonemes[normalized]
    }

    /// Lowercase + strip non-letter/digit/apostrophe chars so we hit the
    /// same Misaki cache entries the preprocessor wrote.
    private func normalizeKey(_ word: String) -> String {
        let lowered = word.lowercased()
        let allowedSet = CharacterSet.letters.union(.decimalDigits)
            .union(CharacterSet(charactersIn: "'"))
        let filtered = lowered.unicodeScalars.filter { allowedSet.contains($0) }
        return String(String.UnicodeScalarView(filtered))
    }

    // MARK: - Word splitter
    //
    // The Python side uses `nltk.tokenize.word_tokenize`, which separates
    // punctuation from adjacent words and splits on whitespace. This is a
    // small in-house imitation: it walks the string and emits runs of
    // letters, runs of digits, single punctuation chars, and ignores
    // whitespace. Good enough for parity at the StyleTTS2 token level.
    private func splitWords(_ text: String) -> [String] {
        var out: [String] = []
        var current: String = ""

        @inline(__always) func flushCurrent() {
            if !current.isEmpty {
                out.append(current)
                current.removeAll(keepingCapacity: true)
            }
        }

        for ch in text {
            if ch.isWhitespace {
                flushCurrent()
            } else if ch.isLetter || ch.isNumber || ch == "'" || ch == "-" {
                current.append(ch)
            } else {
                // Treat any other char (punctuation, symbol) as its own token.
                flushCurrent()
                out.append(String(ch))
            }
        }
        flushCurrent()
        return out
    }
}
