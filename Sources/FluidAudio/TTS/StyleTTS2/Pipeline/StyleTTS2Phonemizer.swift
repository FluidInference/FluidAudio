import Foundation

/// Text → IPA phoneme string pipeline for StyleTTS2.
///
/// For English (`.americanEnglish`), uses the in-tree `G2PModel` (BART
/// encoder-decoder, misaki-style IPA) and remaps the misaki conventions to
/// the espeak-ng convention that StyleTTS2's LibriTTS checkpoint expects.
///
/// **Per-piece (single glyph) remap** — applied as misaki emits each piece:
///
///   misaki → espeak-ng
///   A → eɪ   I → aɪ   O → oʊ   W → aʊ   Y → ɔɪ
///   ᵊ → ə   (tiny-schwa offglide; not in StyleTTS2's 178-vocab)
///
/// **Post-pass (multi-glyph) remap** — applied to the assembled phoneme
/// string after every word has been emitted. Both the ligature and the
/// decomposed forms exist as distinct tokens in the 178-vocab, but the
/// LibriTTS checkpoint was trained against espeak-ng output, so the model's
/// embeddings for the misaki ligature glyphs (`ʧ`, `ʤ`) are essentially
/// untrained noise. Same story for the schwa+r digraphs that espeak collapses
/// into single rhotic vowels (`ɝ`, `ɚ`):
///
///   misaki → espeak-ng         word example
///   ʧ      → tʃ               choice  → tʃˈɔɪs
///   ʤ      → dʒ               jump    → dʒˈʌmps
///   ɜɹ     → ɝ  (U+025D)      girl    → ɡˈɝl
///   əɹ     → ɚ  (U+025A)      over    → ˈoʊvɚ
///
/// Other glyphs (`ˈ`, `ˌ`, `ð`, `θ`, `ɹ`, `ɾ`, etc.) are already in the
/// 178-token espeak-ng vocabulary and pass through unchanged.
///
/// Non-English languages fall back to `MultilingualG2PModel` (CharsiuG2P
/// ByT5). Output quality there is unvalidated — the LibriTTS checkpoint is
/// English-only.
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

    /// Misaki single-glyph diphthongs / offglides → espeak-ng IPA.
    /// Applied per-piece on the output of `G2PModel.phonemize`.
    private static let misakiToEspeak: [String: String] = [
        "A": "eɪ",
        "I": "aɪ",
        "O": "oʊ",
        "W": "aʊ",
        "Y": "ɔɪ",
        "ᵊ": "ə",
    ]

    /// Post-pass multi-glyph remap applied to the assembled phoneme string
    /// after all word pieces have been concatenated. Decomposes misaki's
    /// affricate ligatures and collapses the schwa+r digraphs into the
    /// single rhotic vowels espeak-ng emits — see the type-level docs for
    /// rationale. Order matters only insofar as `əɹ` and `ɜɹ` must be
    /// applied before any rule that would consume the trailing `ɹ` (none
    /// exist today; left ordered for future-proofing).
    private static let misakiToEspeakPostPass: [(String, String)] = [
        ("ʧ", "tʃ"),
        ("ʤ", "dʒ"),
        ("ɜɹ", "ɝ"),
        ("əɹ", "ɚ"),
    ]

    /// Apply `misakiToEspeakPostPass` rules to a phoneme string in order.
    /// Exposed `internal` for unit tests.
    internal static func applyEspeakPostPass(_ s: String) -> String {
        var out = s
        for (from, to) in misakiToEspeakPostPass {
            out = out.replacingOccurrences(of: from, with: to)
        }
        return out
    }

    /// Convert raw text to an IPA phoneme string for StyleTTS2.
    ///
    /// - Parameters:
    ///   - text: Source text in the natural orthography of `language`.
    ///   - language: Target language. English uses the in-tree BART G2P;
    ///     all others fall back to `MultilingualG2PModel`.
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

        // Multi-glyph misaki → espeak normalization. Only meaningful for
        // English (the LibriTTS checkpoint is English-only); skipping for
        // other languages avoids touching CharsiuG2P output we don't have
        // a model contract for.
        if language == .americanEnglish {
            output = applyEspeakPostPass(output)
        }
        return output
    }

    // MARK: - Private

    /// Phonemize one word and append its IPA characters to `output`.
    /// Drops apostrophes from the G2P input (`'s`/`'t` enclitics roundtrip
    /// poorly through ByT5; the BART G2P also handles them best stripped).
    private static func flushWord(
        _ buffer: inout String,
        language: MultilingualG2PLanguage,
        into output: inout String
    ) async throws {
        defer { buffer.removeAll(keepingCapacity: true) }

        let stripped = buffer.replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "'", with: "")
        guard !stripped.isEmpty else { return }

        if language == .americanEnglish {
            try await flushWordEnglish(stripped, into: &output)
        } else {
            try await flushWordMultilingual(stripped, language: language, into: &output)
        }
    }

    /// English path: BART G2P (misaki IPA) → espeak-ng IPA via small remap.
    private static func flushWordEnglish(
        _ word: String,
        into output: inout String
    ) async throws {
        // Lazy first-call download/load; subsequent calls hit the cache.
        try await G2PModel.shared.ensureModelsAvailable()
        let phonemes = try await G2PModel.shared.phonemize(word: word.lowercased())
        guard let phonemes else {
            logger.warning(
                "G2P unavailable for English word \"\(word)\"; passing through verbatim")
            output.append(word)
            return
        }
        for piece in phonemes {
            if let mapped = misakiToEspeak[piece] {
                output.append(mapped)
            } else {
                output.append(piece)
            }
        }
    }

    /// Non-English path: CharsiuG2P fallback (unvalidated for StyleTTS2).
    /// The shipped LibriTTS checkpoint is English-only, so this branch is
    /// best-effort. `MultilingualG2PModel.loadIfNeeded` only reads from
    /// cache — `StyleTTS2Manager.initialize` does not pre-fetch this repo,
    /// so callers who hit this path must have downloaded the kokoro
    /// multilingual G2P some other way (e.g. via Kokoro init).
    private static func flushWordMultilingual(
        _ word: String,
        language: MultilingualG2PLanguage,
        into output: inout String
    ) async throws {
        try await MultilingualG2PModel.shared.ensureModelsAvailable()
        let phonemes = try await MultilingualG2PModel.shared.phonemize(
            word: word,
            language: language
        )
        guard let phonemes else {
            logger.warning(
                "G2P unavailable for word \"\(word)\" (\(language)); passing through verbatim")
            output.append(word)
            return
        }
        for piece in phonemes {
            output.append(piece)
        }
    }
}
