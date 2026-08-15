import Foundation

/// Shared helpers for reading uppercase initialisms as letter names in
/// English TTS frontends (issue #710).
///
/// A few bundled Misaki entries don't read as letter names even though
/// uppercase callers expect them to (`AI` → blended `ˈAˌI`, `US` → the
/// lowercase-pronoun `ʌs` shape), and unknown all-caps tokens (`FBI`,
/// `ATP`) miss the lexicon and fall through to BART G2P, which sounds them
/// out as a word. Both ``KokoroAneEnglishPhonemizer`` and (in a follow-up)
/// StyleTTS2 can use these helpers to spell such tokens from the per-letter
/// lexicon entries instead.
///
/// The actual letter phonemes come from the caller's lexicon, so this type
/// holds only the policy (which tokens to spell) and the spelling
/// mechanics; the data stays with each frontend.
enum EnglishInitialisms {

    /// Exact uppercase spellings whose bundled lexicon entry is not the
    /// letter-name reading callers expect. These should bypass the Misaki
    /// lexicon and be spelled out from the per-letter entries instead.
    /// Lowercase `us`/`ai` are intentionally absent — only the exact
    /// uppercase spelling is overridden.
    static let letterNameOverrides: Set<String> = ["AI", "US"]

    /// Length bounds for spelling unknown all-caps tokens as letter names.
    private static let lengthRange = 2...5

    /// A strict ASCII all-caps token (`FBI`, `ATP`) within the length range
    /// — the only shape worth spelling out after a lexicon miss. Anything
    /// with a digit, hyphen, apostrophe, or non-ASCII letter is excluded so
    /// brand and proper-name pronunciations aren't disturbed.
    static func isCandidate(_ word: String) -> Bool {
        guard lengthRange.contains(word.count) else { return false }
        return word.allSatisfy { $0.isASCII && $0.isUppercase && $0.isLetter }
    }

    /// Spell `word` as a sequence of letter names, resolving each uppercase
    /// letter to its phoneme tokens via `letterTokens` and joining the
    /// rendered letters with `separator` so each is its own prosodic unit
    /// (`FBI` → `ˈɛf bˈi ˈI`).
    ///
    /// - Parameters:
    ///   - letterTokens: resolves a single-letter key (`"F"`) to its
    ///     phoneme tokens, or `nil` if absent (e.g. a G2P-only degraded
    ///     path, or a letter filtered out of the lexicon cache).
    ///   - render: turns one letter's tokens into its IPA fragment.
    ///     Defaults to a plain join; frontends that post-process Misaki
    ///     shorthand can pass their own.
    ///   - separator: inserted between letters (default a single space).
    /// - Returns: the spelled string, or `nil` if any letter is missing —
    ///   so the caller falls through to its normal fallback rather than
    ///   emitting a partial word.
    static func spell(
        _ word: String,
        letterTokens: (String) -> [String]?,
        render: ([String]) -> String = { $0.joined() },
        separator: String = " "
    ) -> String? {
        var letters: [String] = []
        letters.reserveCapacity(word.count)
        for character in word {
            guard let tokens = letterTokens(String(character)), !tokens.isEmpty else {
                return nil
            }
            letters.append(render(tokens))
        }
        guard !letters.isEmpty else { return nil }
        return letters.joined(separator: separator)
    }
}
