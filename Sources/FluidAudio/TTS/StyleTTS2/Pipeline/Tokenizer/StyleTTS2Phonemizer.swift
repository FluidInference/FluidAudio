import Foundation

/// English-only StyleTTS2 phonemizer.
///
/// The Python pipeline runs `phonemizer.backend.EspeakBackend(language="en-us",
/// preserve_punctuation=True, with_stress=True)` followed by
/// `nltk.tokenize.word_tokenize` and finally `TextCleaner`. FluidAudio cannot
/// ship the espeak C library, so the Swift backend uses the existing
/// `MultilingualG2PModel` (CharsiuG2P ByT5) actor instead.
///
/// > Important: CharsiuG2P uses a different IPA convention than espeak —
/// > most notably it lacks stress markers (`ˈ`, `ˌ`) and yields slightly
/// > different vowel choices for some words. The model produces
/// > intelligible speech but quality is meaningfully below the espeak
/// > reference. Callers with a reliable phonemizer should pass IPA
/// > directly via `StyleTTS2Manager.synthesizePhonemes(...)`.
public struct StyleTTS2Phonemizer: Sendable {

    private let logger = AppLogger(category: "StyleTTS2Phonemizer")
    private let language: MultilingualG2PLanguage

    public init(language: MultilingualG2PLanguage = .americanEnglish) {
        self.language = language
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
        for word in words {
            if word.isEmpty { continue }
            // Punctuation passes through verbatim — TextCleaner has direct
            // entries for `; : , . ! ? ¡ ¿ — … " « » " " ` and space.
            if word.allSatisfy({ StyleTTS2TextCleaner.punctuation.contains($0) }) {
                ipaParts.append(word)
                continue
            }

            do {
                let phonemes = try await MultilingualG2PModel.shared.phonemize(
                    word: word, language: language)
                if let phonemes, !phonemes.isEmpty {
                    ipaParts.append(phonemes.joined())
                    anyResolved = true
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

        if !anyResolved {
            throw StyleTTS2Error.phonemizationFailed(
                "no words resolved by CharsiuG2P (input='\(text.prefix(40))')")
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
