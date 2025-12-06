import Foundation
import OSLog

/// A single custom vocabulary entry.
public struct CustomVocabularyTerm: Codable, Sendable {
    public let text: String
    public let weight: Float?
    public let aliases: [String]?
    /// Optional pre-tokenized Parakeet TDT vocabulary IDs for this phrase.
    /// When present, decode-time biasing can operate directly on RNNT/TDT token IDs.
    public let tokenIds: [Int]?
    /// Optional pre-tokenized CTC vocabulary IDs for this phrase.
    /// This is used by the auxiliary CTC keyword spotter path and is distinct from
    /// the Parakeet TDT token IDs above.
    public let ctcTokenIds: [Int]?

    /// Create a custom vocabulary term.
    /// - Parameters:
    ///   - text: The word or phrase to boost
    ///   - weight: Optional weight for the term
    ///   - aliases: Optional alternative spellings
    ///   - tokenIds: Optional pre-tokenized Parakeet TDT token IDs
    ///   - ctcTokenIds: Optional pre-tokenized CTC token IDs
    public init(
        text: String,
        weight: Float? = nil,
        aliases: [String]? = nil,
        tokenIds: [Int]? = nil,
        ctcTokenIds: [Int]? = nil
    ) {
        self.text = text
        self.weight = weight
        self.aliases = aliases
        self.tokenIds = tokenIds
        self.ctcTokenIds = ctcTokenIds
    }
}

/// Raw JSON model for on‑disk config.
public struct CustomVocabularyConfig: Codable, Sendable {
    public let alpha: Float?
    public let contextScore: Float?
    public let depthScaling: Float?
    public let scorePerPhrase: Float?
    public let terms: [CustomVocabularyTerm]

    // CTC keyword boosting confidence thresholds
    public let minCtcScore: Float?
    public let minSimilarity: Float?
    public let minCombinedConfidence: Float?
}

/// Runtime context used by the decoder biasing system.
public struct CustomVocabularyContext: Sendable {
    public let terms: [CustomVocabularyTerm]
    public let alpha: Float
    public let contextScore: Float
    public let depthScaling: Float
    public let scorePerPhrase: Float

    // CTC keyword boosting confidence thresholds
    public let minCtcScore: Float
    public let minSimilarity: Float
    public let minCombinedConfidence: Float

    public init(
        terms: [CustomVocabularyTerm],
        alpha: Float = 0.5,
        contextScore: Float = 1.2,
        depthScaling: Float = 2.0,
        scorePerPhrase: Float = 0.0,
        minCtcScore: Float = -12.0,
        minSimilarity: Float = 0.52,
        minCombinedConfidence: Float = 0.54
    ) {
        self.terms = terms
        self.alpha = alpha
        self.contextScore = contextScore
        self.depthScaling = depthScaling
        self.scorePerPhrase = scorePerPhrase
        self.minCtcScore = minCtcScore
        self.minSimilarity = minSimilarity
        self.minCombinedConfidence = minCombinedConfidence
    }

    /// Load a custom vocabulary JSON file produced by the analysis tooling.
    public static func load(from url: URL) throws -> CustomVocabularyContext {
        let logger = Logger(subsystem: "com.fluidaudio", category: "CustomVocabulary")
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(CustomVocabularyConfig.self, from: data)

        let alpha = config.alpha ?? 0.5
        let contextScore = config.contextScore ?? 1.2
        let depthScaling = config.depthScaling ?? 2.0
        let scorePerPhrase = config.scorePerPhrase ?? 0.0
        let minCtcScore = config.minCtcScore ?? -12.0
        let minSimilarity = config.minSimilarity ?? 0.52
        let minCombinedConfidence = config.minCombinedConfidence ?? 0.54

        // Validate and normalize vocabulary terms
        var validatedTerms: [CustomVocabularyTerm] = []
        for term in config.terms {
            let (sanitized, warnings) = sanitizeVocabularyTerm(term.text)

            if !warnings.isEmpty {
                logger.warning("Term '\(term.text)': \(warnings.joined(separator: ", "))")
            }

            // Skip empty terms after sanitization
            guard !sanitized.isEmpty else {
                logger.warning("Term '\(term.text)' is empty after sanitization, skipping")
                continue
            }

            validatedTerms.append(term)
        }

        return CustomVocabularyContext(
            terms: validatedTerms,
            alpha: alpha,
            contextScore: contextScore,
            depthScaling: depthScaling,
            scorePerPhrase: scorePerPhrase,
            minCtcScore: minCtcScore,
            minSimilarity: minSimilarity,
            minCombinedConfidence: minCombinedConfidence
        )
    }

    /// Load a custom vocabulary from simple text format.
    /// Format: one word per line, optionally "word: alias1, alias2, ..."
    public static func loadFromSimpleFormat(from url: URL) throws -> CustomVocabularyContext {
        let contents = try String(contentsOf: url, encoding: .utf8)
        var terms: [CustomVocabularyTerm] = []

        for line in contents.split(whereSeparator: { $0.isNewline }) {
            let trimmed = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

            // Parse "word: alias1, alias2, ..." format
            if let colonIndex = trimmed.firstIndex(of: ":") {
                let word = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespaces)
                let aliasesPart = String(trimmed[trimmed.index(after: colonIndex)...])
                let aliases = aliasesPart.split(separator: ",").map {
                    String($0).trimmingCharacters(in: .whitespaces)
                }.filter { !$0.isEmpty }

                terms.append(
                    CustomVocabularyTerm(
                        text: word,
                        weight: 10.0, // Aggressive default weight for text list
                        aliases: aliases.isEmpty ? nil : aliases
                    ))
            } else {
                terms.append(CustomVocabularyTerm(text: trimmed, weight: 10.0)) // Aggressive default weight
            }
        }

        return CustomVocabularyContext(terms: terms)
    }

    /// Sanitize a vocabulary term and return warnings about potential issues.
    private static func sanitizeVocabularyTerm(_ text: String) -> (sanitized: String, warnings: [String]) {
        var warnings: [String] = []
        var result = text

        // 1. Check for control characters
        if result.rangeOfCharacter(from: .controlCharacters) != nil {
            warnings.append("contains control characters")
            result = result.filter { !$0.isNewline && !$0.isWhitespace || $0 == " " }
        }

        // 2. Check for diacritics (informational, not blocking)
        if result.folding(options: .diacriticInsensitive, locale: nil) != result {
            warnings.append("contains diacritics - consider adding ASCII alias")
        }

        // 3. Check for numbers (informational)
        if result.rangeOfCharacter(from: .decimalDigits) != nil {
            warnings.append("contains numbers")
        }

        // 4. Check for unusual characters (not letters, spaces, hyphens, apostrophes)
        let allowedChars = CharacterSet.letters
            .union(.whitespaces)
            .union(CharacterSet(charactersIn: "-'"))

        if result.rangeOfCharacter(from: allowedChars.inverted) != nil {
            warnings.append("contains unusual characters")
        }

        return (result, warnings)
    }
}
