import Foundation

/// English-only StyleTTS2 phonemizer.
///
/// The Python pipeline runs `phonemizer.backend.EspeakBackend(language="en-us",
/// preserve_punctuation=True, with_stress=True)` followed by
/// `nltk.tokenize.word_tokenize` and finally `TextCleaner`. FluidAudio cannot
/// ship the espeak C library, so the Swift backend mirrors Kokoro's
/// gold-first lookup pattern instead:
///
///   1. The Misaki gold lexicon (`us_gold.json`) — high-confidence,
///      hand-curated, includes stress markers (`ˈ`, `ˌ`).
///   2. The Misaki silver lexicon (`us_silver.json`) — lower confidence
///      but still curated.
///   3. CharsiuG2P (`MultilingualG2PModel`) — neural fallback for words
///      not in either lexicon. CharsiuG2P uses a different IPA convention
///      (no stress markers, slightly different vowel choices), so this
///      branch is reserved for genuine misses to keep the synthesizer
///      close to the espeak training distribution.
///
/// > Important: callers with a reliable phonemizer (e.g. server-side
/// > espeak) can still bypass everything via
/// > `StyleTTS2Manager.synthesize(ipa:referenceAudioURL:...)`.
public struct StyleTTS2Phonemizer: Sendable {

    private let logger = AppLogger(category: "StyleTTS2Phonemizer")
    private let language: MultilingualG2PLanguage
    private let lexicon: StyleTTS2GoldLexicon?

    public init(
        language: MultilingualG2PLanguage = .americanEnglish,
        lexicon: StyleTTS2GoldLexicon? = nil
    ) {
        self.language = language
        self.lexicon = lexicon
    }

    /// Phonemize a sentence and encode it into the StyleTTS2 token IDs.
    /// Returns the encoded token list (with the leading-pad token already
    /// inserted) ready for the `text_encoder` stage.
    ///
    /// - Throws: `StyleTTS2Error.phonemizationFailed` if the underlying G2P
    ///   model isn't loadable or returns nothing for every word.
    public func encode(_ text: String) async throws -> [Int32] {
        let phonemeString = try await phonemize(text)
        return StyleTTS2TextCleaner.encode(phonemeString)
    }

    /// Convert text to a plain IPA string (no token IDs). Mirrors
    /// `" ".join(word_tokenize(espeak.phonemize([text])))`.
    public func phonemize(_ text: String) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return ""
        }

        let words = splitWords(trimmed)
        var ipaParts: [String] = []
        ipaParts.reserveCapacity(words.count)

        var anyResolved = false
        var lexiconHits = 0
        var g2pHits = 0

        for word in words {
            if word.isEmpty { continue }
            // Punctuation passes through verbatim — TextCleaner has direct
            // entries for `; : , . ! ? ¡ ¿ — … " « » " " ` and space.
            if word.allSatisfy({ StyleTTS2TextCleaner.punctuation.contains($0) }) {
                ipaParts.append(word)
                continue
            }

            // 1. Misaki gold/silver lexicon (case-sensitive → normalized).
            if let lexicon = lexicon, let ipa = await lexicon.phonemes(for: word) {
                ipaParts.append(ipa)
                anyResolved = true
                lexiconHits += 1
                continue
            }

            // 2. CharsiuG2P fallback for unknown words.
            do {
                let phonemes = try await MultilingualG2PModel.shared.phonemize(
                    word: word, language: language)
                if let phonemes, !phonemes.isEmpty {
                    ipaParts.append(phonemes.joined())
                    anyResolved = true
                    g2pHits += 1
                } else {
                    // Degraded fallback: pass the grapheme through. The
                    // decoder's vocab includes ASCII letters, so this still
                    // produces *something* rather than dropping the word
                    // outright (which would shift alignment).
                    logger.notice("CharsiuG2P returned nil for '\(word)'; passing graphemes")
                    ipaParts.append(word)
                }
            } catch {
                logger.warning("CharsiuG2P failed for '\(word)': \(error); passing graphemes")
                ipaParts.append(word)
            }
        }

        if lexicon != nil {
            logger.debug(
                "Phonemized \(words.count) tokens — gold/silver hits: \(lexiconHits), G2P fallback: \(g2pHits)"
            )
        }

        if !anyResolved {
            throw StyleTTS2Error.phonemizationFailed(
                "no words resolved by lexicon or CharsiuG2P (input='\(text.prefix(40))')")
        }

        return ipaParts.joined(separator: " ")
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
