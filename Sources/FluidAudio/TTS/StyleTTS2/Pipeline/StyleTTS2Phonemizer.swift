import Foundation

/// Text → IPA phoneme string pipeline for StyleTTS2.
///
/// English (`.americanEnglish`) resolution order, per word:
///   1. Kokoro `us_lexicon_cache.json` lookup (case-sensitive form first,
///      then lowercase). This is misaki's gold dictionary flattened into a
///      JSON map; common words and contractions (`this`, `is`, `the`,
///      `of`, `don't`, `won't`, `i'm`, …) only resolve correctly through
///      this path. Apostrophes are preserved on the lookup key.
///   2. Acronym fallback: if the word is 2..6 ASCII uppercase letters
///      (e.g. `ANE`, `IBM`, `USA`), spell it letter-by-letter through the
///      lexicon's case-sensitive single-letter entries.
///   3. In-tree `G2PModel` BART encoder-decoder (misaki-style IPA) for
///      OOV words, with the same misaki→espeak remap as before.
///
/// **Per-piece (single glyph) remap** — applied to every emitted phoneme
/// piece, regardless of source (lexicon, acronym, or BART):
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
public enum StyleTTS2Phonemizer {

    private static let logger = AppLogger(category: "StyleTTS2Phonemizer")

    /// Misaki single-glyph diphthongs / offglides → espeak-ng IPA.
    /// Applied per-piece on every phoneme source (lexicon, acronym, BART).
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

    /// Lazily-loaded shared Kokoro lexicon (`us_lexicon_cache.json`).
    /// Phoneme tokens are stored in misaki convention; pieces are remapped
    /// to espeak-ng on emit via `emitPieces`.
    private actor LexiconHolder {
        private var lower: [String: [String]] = [:]
        private var caseSensitive: [String: [String]] = [:]
        private var loaded = false

        private struct Payload: Decodable {
            let lower: [String: [String]]
            let caseSensitive: [String: [String]]
        }

        /// Download the lexicon if needed and parse it once. Subsequent
        /// calls are no-ops. Errors are surfaced; callers may swallow.
        func ensureLoaded() async throws {
            if loaded { return }
            let url = try await TtsResourceDownloader.ensureLexiconFile(
                named: "us_lexicon_cache.json")
            let data = try Data(contentsOf: url)
            let payload = try JSONDecoder().decode(Payload.self, from: data)
            self.lower = payload.lower
            self.caseSensitive = payload.caseSensitive
            self.loaded = true
        }

        /// Look up `word` in the lexicon. Tries the case-sensitive map
        /// first (catches NASA/IBM/etc. and capital-aware variants),
        /// falls back to the lowercase map. Returns nil if missing or
        /// the lexicon failed to load.
        func lookup(_ word: String) -> [String]? {
            if let v = caseSensitive[word] { return v }
            return lower[word.lowercased()]
        }

        /// If `word` is an acronym (2..6 ASCII uppercase letters), return
        /// per-letter pronunciations concatenated. Letters are resolved
        /// case-sensitive first (e.g. `A` → `ˈeɪ`, `N` → `ˈɛn`), with the
        /// lowercase map as fallback. Returns nil if the pattern doesn't
        /// match or any letter is missing from the lexicon.
        func lookupAcronym(_ word: String) -> [String]? {
            let chars = Array(word)
            guard chars.count >= 2 && chars.count <= 6 else { return nil }
            for ch in chars {
                guard ch.isASCII, ch.isLetter, ch.isUppercase else { return nil }
            }
            var result: [String] = []
            result.reserveCapacity(chars.count * 4)
            for ch in chars {
                let s = String(ch)
                guard let pieces = caseSensitive[s] ?? lower[s.lowercased()] else {
                    return nil
                }
                result.append(contentsOf: pieces)
            }
            return result
        }
    }

    private static let lexicon = LexiconHolder()

    /// Convert raw text to an IPA phoneme string for StyleTTS2.
    ///
    /// - Parameters:
    ///   - text: Source text in the natural orthography of `language`.
    ///   - language: Target language. English consults the Kokoro lexicon
    ///     first, then falls back to the in-tree BART G2P. All other
    ///     languages route through `MultilingualG2PModel`.
    /// - Returns: A phoneme string. Pass this to `StyleTTS2Vocab.encode(_:)`
    ///   to obtain `[Int32]` token ids.
    public static func phonemize(
        text: String,
        language: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> String {
        let cleaned = TtsTextPreprocessor.preprocess(text)

        // Best-effort lexicon load for English. If this fails (offline,
        // missing asset), lookups return nil and we degrade gracefully
        // to BART G2P — the original behavior.
        if language == .americanEnglish {
            do {
                try await lexicon.ensureLoaded()
            } catch {
                logger.warning(
                    "Lexicon unavailable (\(error.localizedDescription)); "
                        + "falling through to BART G2P only")
            }
        }

        var output = ""
        output.reserveCapacity(cleaned.count * 2)

        var wordBuffer = ""
        wordBuffer.reserveCapacity(64)

        for ch in cleaned {
            if ch.isLetter || ch == "'" || ch == "\u{2019}" {
                // Treat ASCII apostrophe + typographic apostrophe as part
                // of words (don't, won't, they're) — preserved on the
                // lexicon lookup key, stripped only for the BART fallback.
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
    /// Keeps the original (with apostrophes) for lexicon lookup; the
    /// BART G2P fallback receives the apostrophe-stripped form because
    /// `'s`/`'t` enclitics roundtrip poorly through that encoder.
    private static func flushWord(
        _ buffer: inout String,
        language: MultilingualG2PLanguage,
        into output: inout String
    ) async throws {
        defer { buffer.removeAll(keepingCapacity: true) }

        let original = buffer
        let stripped =
            original
            .replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "\u{2019}", with: "")
        guard !stripped.isEmpty else { return }

        if language == .americanEnglish {
            try await flushWordEnglish(
                original: original, stripped: stripped, into: &output)
        } else {
            try await flushWordMultilingual(
                stripped, language: language, into: &output)
        }
    }

    /// English path: lexicon → acronym fallback → BART G2P.
    private static func flushWordEnglish(
        original: String,
        stripped: String,
        into output: inout String
    ) async throws {
        // 1. Lexicon (apostrophes preserved).
        if let pieces = await lexicon.lookup(original) {
            emitPieces(pieces, into: &output)
            return
        }

        // 2. Acronym fallback (apostrophe-stripped — apostrophes don't
        //    appear in acronyms).
        if let pieces = await lexicon.lookupAcronym(stripped) {
            emitPieces(pieces, into: &output)
            return
        }

        // 3. BART G2P.
        try await G2PModel.shared.ensureModelsAvailable()
        let phonemes = try await G2PModel.shared.phonemize(word: stripped.lowercased())
        guard let phonemes else {
            logger.warning(
                "G2P unavailable for English word \"\(stripped)\"; passing through verbatim")
            output.append(stripped)
            return
        }
        emitPieces(phonemes, into: &output)
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

    /// Append phoneme pieces to `output`, applying the per-piece misaki →
    /// espeak-ng remap. Used for lexicon, acronym, and BART results — they
    /// all live in misaki convention before this point.
    private static func emitPieces(_ pieces: [String], into output: inout String) {
        for piece in pieces {
            if let mapped = misakiToEspeak[piece] {
                output.append(mapped)
            } else {
                output.append(piece)
            }
        }
    }
}
