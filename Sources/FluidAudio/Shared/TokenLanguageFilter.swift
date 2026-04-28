import Foundation

/// Supported languages for script-based filtering.
public enum Language: String, Sendable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case romanian = "ro"  // uses ș, ț (Latin Extended-B)

    // Latin-script Slavic — prone to Cyrillic confusion in multilingual ASR.
    case polish = "pl"
    case czech = "cs"
    case slovak = "sk"
    case slovenian = "sl"
    case croatian = "hr"
    case bosnian = "bs"

    case russian = "ru"
    case ukrainian = "uk"
    case belarusian = "be"
    case bulgarian = "bg"
    case serbian = "sr"

    public var script: Script {
        switch self {
        case .english, .spanish, .french, .german, .italian, .portuguese, .romanian,
            .polish, .czech, .slovak, .slovenian, .croatian, .bosnian:
            return .latin
        case .russian, .ukrainian, .belarusian, .bulgarian, .serbian:
            return .cyrillic
        }
    }
}

/// Writing script categories. Latin and Cyrillic letter ranges are
/// non-overlapping, which the `matches` impl relies on.
public enum Script: Sendable {
    case latin
    case cyrillic
}

/// Filters ASR decoder tokens against the target language's alphabet.
///
/// Used by the v3 TDT decoder to stop the joint network from emitting
/// wrong-language tokens — e.g. Cyrillic letters while transcribing Polish
/// (issue #512). Partitions by Unicode script (Latin/Cyrillic) only; a
/// per-language token allowlist (Polish vs Czech etc.) could plug in here
/// later without changing the call-site API.
internal struct TokenLanguageFilter: Sendable {

    /// SentencePiece word-boundary marker (▁, U+2581). Stripped before script
    /// checks — carries no script information.
    private static let sentencePieceBoundary: Unicode.Scalar = "\u{2581}"

    /// Whether every character in `text` is compatible with `script`.
    ///
    /// The Latin path has no "reject Cyrillic" guard because the Cyrillic
    /// block sits outside every Latin range. The Cyrillic path *does* need an
    /// explicit ASCII-letter reject — ASCII overlaps with the Latin range, so
    /// without the guard a token like `"cat"` would pass as Cyrillic.
    static func matches(_ text: String, script: Script) -> Bool {
        let cleanedText = text.replacingOccurrences(of: String(sentencePieceBoundary), with: "")

        // Pure boundary markers are script-neutral — let the caller's argmax
        // decide on logit alone rather than excluding them here.
        guard !cleanedText.isEmpty else { return true }

        let chars = cleanedText.unicodeScalars
        switch script {
        case .latin:
            return chars.allSatisfy {
                ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII
                    || ($0.value >= 0x00A0 && $0.value <= 0x00FF)  // Latin-1
                    || ($0.value >= 0x0100 && $0.value <= 0x017F)  // Latin Extended-A
                    || ($0.value >= 0x0180 && $0.value <= 0x024F)  // Latin Extended-B
                    || ($0.value >= 0x0300 && $0.value <= 0x036F)  // Combining Diacritical Marks (NFD)
                    || ($0.value >= 0x1E00 && $0.value <= 0x1EFF)  // Latin Extended Additional
            }
        case .cyrillic:
            return chars.allSatisfy { char in
                let value = char.value
                if value >= 0x0400 && value <= 0x04FF { return true }
                // ASCII is script-neutral except for letters A-Z/a-z, which
                // must be rejected (see function doc).
                if value >= 0x0020 && value <= 0x007F {
                    if (value >= 0x41 && value <= 0x5A) || (value >= 0x61 && value <= 0x7A) {
                        return false
                    }
                    return true
                }
                return false
            }
        }
    }

    /// Filter top-K candidates by script and return the highest-logit match.
    ///
    /// Returned probability is softmax **over the top-K logits only**, not the
    /// full vocabulary. It is systematically larger than a full-vocab softmax
    /// — use it for relative ranking, not as a drop-in probability.
    ///
    /// - Returns: `nil` if no right-language match exists or inputs are empty.
    static func filterTopK(
        topKIds: [Int],
        topKLogits: [Float],
        vocabulary: [Int: String],
        preferredScript: Script
    ) -> (tokenId: Int, probability: Float)? {
        let count = min(topKIds.count, topKLogits.count)
        guard count > 0 else { return nil }

        // Explicit argmax over right-language candidates. CoreML top-K is not
        // guaranteed sorted. The `bestIdx < 0` clause forces the first match
        // to win even when its logit is -infinity (otherwise it would never
        // beat the sentinel).
        var bestIdx: Int = -1
        var bestLogit: Float = -.infinity
        for idx in 0..<count {
            let tokenId = topKIds[idx]
            guard let tokenText = vocabulary[tokenId] else { continue }
            guard matches(tokenText, script: preferredScript) else { continue }

            let logit = topKLogits[idx]
            if bestIdx < 0 || logit > bestLogit {
                bestLogit = logit
                bestIdx = idx
            }
        }
        guard bestIdx >= 0 else { return nil }

        // Top-K softmax with max-logit stability.
        var maxLogit = -Float.infinity
        for i in 0..<count where topKLogits[i] > maxLogit {
            maxLogit = topKLogits[i]
        }
        guard maxLogit.isFinite else { return (topKIds[bestIdx], 0) }

        var sumExp: Float = 0
        for i in 0..<count {
            sumExp += expf(topKLogits[i] - maxLogit)
        }
        guard sumExp > 0 else { return (topKIds[bestIdx], 0) }

        let prob = expf(topKLogits[bestIdx] - maxLogit) / sumExp
        return (topKIds[bestIdx], max(0, min(1, prob)))
    }
}
