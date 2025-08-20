import Foundation

/// Text normalizer for standard ASR evaluation
/// Follows HuggingFace Open ASR Leaderboard standards for LibriSpeech benchmarking
struct TextNormalizer {

    /// Standard normalization for WER calculation aligned with HuggingFace evaluation
    /// This follows the standard approach used in ASR leaderboard evaluations:
    /// 1. Convert to lowercase
    /// 2. Remove punctuation except apostrophes in contractions
    /// 3. Normalize British to American spelling (standard in ASR benchmarks)
    /// 4. Collapse multiple spaces
    /// 5. Trim leading/trailing whitespace
    static func normalize(_ text: String) -> String {
        var normalized = text

        // Step 1: Convert to lowercase
        normalized = normalized.lowercased()

        // Step 2: Normalize British to American spelling variations
        // This is standard practice in ASR benchmarking (e.g., HuggingFace Open ASR Leaderboard)
        // to ensure consistent evaluation across different English variants.
        // LibriSpeech transcriptions use American spelling, so normalizing ensures fair comparison.
        let spellingVariations = [
            "counselled": "counseled",
            "counselling": "counseling",
            "distil": "distill",
            "distilled": "distilled",
            "distilling": "distilling",
            "distils": "distills",
            "fulfil": "fulfill",
            "fulfilled": "fulfilled",
            "fulfilling": "fulfilling",
            "fulfils": "fulfills",
            "gruelling": "grueling",
            "levelled": "leveled",
            "levelling": "leveling",
            "marshalled": "marshaled",
            "marshalling": "marshaling",
            "modelled": "modeled",
            "modelling": "modeling",
            "quarrelled": "quarreled",
            "quarrelling": "quarreling",
            "revelled": "reveled",
            "revelling": "reveling",
            "travelling": "traveling",
            "travelled": "traveled",
            "yodelling": "yodeling",
            "yodelled": "yodeled",
        ]

        for (british, american) in spellingVariations {
            normalized = normalized.replacingOccurrences(of: british, with: american)
        }

        // Step 3: Remove punctuation but keep apostrophes
        // This regex keeps letters, numbers, spaces, and apostrophes
        let punctuationPattern = try! NSRegularExpression(
            pattern: "[^a-z0-9\\s']",
            options: []
        )
        normalized = punctuationPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        // Step 4: Normalize whitespace
        let whitespacePattern = try! NSRegularExpression(
            pattern: "\\s+",
            options: []
        )
        normalized = whitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        // Step 5: Trim leading/trailing whitespace
        normalized = normalized.trimmingCharacters(in: .whitespaces)

        return normalized
    }
}
