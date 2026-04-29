import Foundation

/// Text → IPA phoneme string pipeline for StyleTTS2.
///
/// Wires the existing `MultilingualG2PModel` (CharsiuG2P ByT5) through to the
/// 178-token espeak-ng IPA vocabulary that ships in
/// `constants/text_cleaner_vocab.json`.
///
/// The output is a phoneme **string** (not yet token ids) so that callers
/// can splice in custom IPA, log, or post-process before encoding through
/// `StyleTTS2Vocab.encode(_:)`.
///
/// Pipeline:
///  1. `TtsTextPreprocessor.preprocess` — number/currency/time/unit expansion
///     and smart-quote normalization (shared with the Kokoro frontend).
///  2. Walk the cleaned text character-by-character. Letter runs are
///     accumulated as words and flushed through G2P. Non-letter characters
///     (punctuation, whitespace) are passed through verbatim — the
///     178-vocab includes `.,;:!?…—""«»¡¿"` and `' '` so this round-trips
///     cleanly.
///  3. Returned string is one IPA grapheme per character, ready to be fed
///     to `StyleTTS2Vocab.encode(_:)`.
public enum StyleTTS2Phonemizer {

    private static let logger = AppLogger(category: "StyleTTS2Phonemizer")

    /// Convert raw text to an IPA phoneme string for StyleTTS2.
    ///
    /// - Parameters:
    ///   - text: Source text in the natural orthography of `language`.
    ///   - language: Target language for `MultilingualG2PModel`. The
    ///     LibriTTS checkpoint shipped on HF is English-only; non-English
    ///     output will run but is not validated.
    /// - Returns: A phoneme string. Pass this to `StyleTTS2Vocab.encode(_:)`
    ///   to obtain `[Int32]` token ids.
    public static func phonemize(
        text: String,
        language: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> String {
        let cleaned = TtsTextPreprocessor.preprocess(text)

        var output = ""
        output.reserveCapacity(cleaned.count * 2)

        var wordBuffer = ""
        wordBuffer.reserveCapacity(64)

        for ch in cleaned {
            if ch.isLetter || ch == "'" || ch == "'" {
                // Treat ASCII apostrophe + typographic apostrophe as part
                // of words (don't, won't, they're) — they're stripped before
                // G2P input but kept as word boundaries.
                wordBuffer.append(ch)
                continue
            }
            // Non-letter — flush the buffered word, then emit the char.
            if !wordBuffer.isEmpty {
                try await flushWord(&wordBuffer, language: language, into: &output)
            }
            // Punctuation/whitespace passes through verbatim — vocab covers
            // the common set; unmapped glyphs are silently dropped at encode
            // time (matches upstream `text_utils.TextCleaner.__call__`).
            output.append(ch)
        }
        if !wordBuffer.isEmpty {
            try await flushWord(&wordBuffer, language: language, into: &output)
        }

        return output
    }

    // MARK: - Private

    /// Phonemize one word and append its IPA characters to `output`.
    /// Drops apostrophes from the G2P input (CharsiuG2P sometimes produces
    /// noise for `'s`/`'t` enclitics; upstream espeak-ng treats them as
    /// part of the word, but ByT5's clean handling is to strip).
    private static func flushWord(
        _ buffer: inout String,
        language: MultilingualG2PLanguage,
        into output: inout String
    ) async throws {
        defer { buffer.removeAll(keepingCapacity: true) }

        let stripped = buffer.replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "'", with: "")
        guard !stripped.isEmpty else { return }

        let phonemes = try await MultilingualG2PModel.shared.phonemize(
            word: stripped,
            language: language
        )
        guard let phonemes else {
            // G2P unavailable (e.g. CI). Fall back to passing the cleaned
            // word through verbatim — the vocab covers ASCII letters so
            // it produces an audibly-degraded but non-empty output.
            logger.warning(
                "G2P unavailable for word \"\(stripped)\"; passing through verbatim")
            output.append(stripped)
            return
        }
        for piece in phonemes {
            output.append(piece)
        }
    }
}
