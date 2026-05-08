import Foundation

/// A phonetic encoder used to bypass the Levenshtein similarity gate
/// when two words *sound* alike but spell differently.
///
/// FluidAudio's vocabulary rescorer uses Levenshtein similarity as its
/// primary candidate-acceptance gate. That works well when the TDT
/// decoder produces a near-correct spelling of a vocabulary term, but
/// fails when TDT mangles an unfamiliar word into a phonetically-
/// similar but textually-distinct English word — e.g., decoding the
/// drug name `Crenessity` as `chronicity` (Levenshtein similarity 0.40,
/// well below the 0.60 gate, but phonetically identical).
///
/// Conformers provide a string-keyed phonetic code (e.g., Metaphone3,
/// Soundex, NYSIIS). When configured, the rescorer will additionally
/// accept candidate replacements whose `soundsAlike` returns `true`,
/// provided their Levenshtein similarity still clears
/// `ContextBiasingConstants.phoneticSimilarityFloor`.
///
/// FluidAudio does not ship a phonetic encoder implementation — the
/// protocol is dependency-injected at `VocabularyRescorer.create(...)`
/// time. Paid encoders (Metaphone3, etc.) can be wired in by clients
/// without bringing third-party code into FluidAudio core.
public protocol PhoneticEncoder: Sendable {

    /// Return the primary phonetic code for `word`. An empty string is
    /// reserved as "no code" — pairs where either side encodes to empty
    /// must not be considered a phonetic match.
    func encode(_ word: String) -> String

    /// Return an optional alternate phonetic code, or `nil`/empty when
    /// none exists. Some encoders (e.g., Metaphone3) emit a second
    /// pronunciation for words with multiple plausible pronunciations
    /// (e.g., `read` past tense vs present tense).
    func encodeAlternate(_ word: String) -> String?
}

extension PhoneticEncoder {

    /// Default: no alternate code.
    public func encodeAlternate(_ word: String) -> String? { nil }

    /// True iff `a` and `b` share any phonetic code (primary or alternate).
    /// Empty codes never match — a word whose pronunciation is unknown
    /// to the encoder cannot be claimed to sound like anything.
    public func soundsAlike(_ a: String, _ b: String) -> Bool {
        let pa = encode(a)
        let pb = encode(b)
        guard !pa.isEmpty, !pb.isEmpty else { return false }
        if pa == pb { return true }

        let aa = encodeAlternate(a)
        let ab = encodeAlternate(b)
        if let aa, !aa.isEmpty {
            if aa == pb { return true }
            if let ab, !ab.isEmpty, aa == ab { return true }
        }
        if let ab, !ab.isEmpty, ab == pa { return true }
        return false
    }
}
