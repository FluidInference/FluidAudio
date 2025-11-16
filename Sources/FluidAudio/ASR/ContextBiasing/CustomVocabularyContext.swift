import Foundation

/// A single custom vocabulary entry.
public struct CustomVocabularyTerm: Codable, Sendable {
    public let text: String
    public let weight: Float?
}

/// Raw JSON model for on‑disk config.
public struct CustomVocabularyConfig: Codable, Sendable {
    public let alpha: Float?
    public let contextScore: Float?
    public let depthScaling: Float?
    public let scorePerPhrase: Float?
    public let terms: [CustomVocabularyTerm]
}

/// Runtime context used by the decoder biasing system.
public struct CustomVocabularyContext: Sendable {
    public let terms: [CustomVocabularyTerm]
    public let alpha: Float
    public let contextScore: Float
    public let depthScaling: Float
    public let scorePerPhrase: Float

    public init(
        terms: [CustomVocabularyTerm],
        alpha: Float = 0.5,
        contextScore: Float = 1.2,
        depthScaling: Float = 2.0,
        scorePerPhrase: Float = 0.0
    ) {
        self.terms = terms
        self.alpha = alpha
        self.contextScore = contextScore
        self.depthScaling = depthScaling
        self.scorePerPhrase = scorePerPhrase
    }

    /// Load a custom vocabulary JSON file produced by the analysis tooling.
    public static func load(from url: URL) throws -> CustomVocabularyContext {
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(CustomVocabularyConfig.self, from: data)

        let alpha = config.alpha ?? 0.5
        let contextScore = config.contextScore ?? 1.2
        let depthScaling = config.depthScaling ?? 2.0
        let scorePerPhrase = config.scorePerPhrase ?? 0.0
        return CustomVocabularyContext(
            terms: config.terms,
            alpha: alpha,
            contextScore: contextScore,
            depthScaling: depthScaling,
            scorePerPhrase: scorePerPhrase
        )
    }
}
