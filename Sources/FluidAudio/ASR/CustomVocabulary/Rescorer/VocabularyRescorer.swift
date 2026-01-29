import Foundation

/// CTC-based vocabulary rescoring for principled vocabulary integration.
///
/// Instead of blindly replacing words based on phonetic similarity, this rescorer
/// uses CTC log-probabilities to verify that vocabulary terms actually match the audio.
/// Only replaces when the vocabulary term has significantly higher acoustic evidence.
///
/// This implements "shallow fusion" or "CTC rescoring" - a standard technique in ASR.
/// The rescorer computes ACTUAL CTC scores for both vocabulary terms AND original words,
/// enabling a fair comparison rather than relying on heuristics.
public struct VocabularyRescorer: Sendable {

    let logger = AppLogger(category: "VocabularyRescorer")

    let spotter: CtcKeywordSpotter
    let vocabulary: CustomVocabularyContext
    let ctcTokenizer: CtcTokenizer?
    let debugMode: Bool

    // BK-tree for efficient approximate string matching (USE_BK_TREE=1 to enable)
    // When enabled, uses BK-tree to find candidate vocabulary terms within edit distance
    // instead of iterating all terms. Provides O(log n) vs O(n) for large vocabularies.
    let useBKTree: Bool
    let bkTree: BKTree?
    let bkTreeMaxDistance: Int

    /// Configuration for rescoring behavior
    public struct Config: Sendable {
        /// Minimum CTC score advantage needed to replace original word
        /// Higher = more conservative (fewer replacements)
        public let minScoreAdvantage: Float

        /// Minimum absolute CTC score for vocabulary term to be considered
        public let minVocabScore: Float

        /// Maximum CTC score for original word to allow replacement
        /// If original word scores very high, don't replace it
        public let maxOriginalScoreForReplacement: Float

        /// Weight for vocabulary term boost (added to CTC score)
        public let vocabBoostWeight: Float

        /// Enable adaptive thresholds based on token count
        /// When true, thresholds are adjusted for longer vocabulary terms
        public let useAdaptiveThresholds: Bool

        /// Reference token count for adaptive scaling (tokens beyond this get adjusted thresholds)
        public let referenceTokenCount: Int

        public static let `default` = Config(
            minScoreAdvantage: ContextBiasingConstants.defaultMinScoreAdvantage,
            minVocabScore: ContextBiasingConstants.defaultMinVocabScore,
            maxOriginalScoreForReplacement: ContextBiasingConstants.defaultMaxOriginalScoreForReplacement,
            vocabBoostWeight: ContextBiasingConstants.defaultVocabBoostWeight,
            useAdaptiveThresholds: ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: ContextBiasingConstants.defaultReferenceTokenCount
        )

        public init(
            minScoreAdvantage: Float = ContextBiasingConstants.defaultMinScoreAdvantage,
            minVocabScore: Float = ContextBiasingConstants.defaultMinVocabScore,
            maxOriginalScoreForReplacement: Float = ContextBiasingConstants.defaultMaxOriginalScoreForReplacement,
            vocabBoostWeight: Float = ContextBiasingConstants.defaultVocabBoostWeight,
            useAdaptiveThresholds: Bool = ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: Int = ContextBiasingConstants.defaultReferenceTokenCount
        ) {
            self.minScoreAdvantage = minScoreAdvantage
            self.minVocabScore = minVocabScore
            self.maxOriginalScoreForReplacement = maxOriginalScoreForReplacement
            self.vocabBoostWeight = vocabBoostWeight
            self.useAdaptiveThresholds = useAdaptiveThresholds
            self.referenceTokenCount = referenceTokenCount
        }

        // MARK: - Adaptive Threshold Functions

        /// Compute adaptive minimum vocabulary score based on token count.
        /// Longer keywords naturally have lower CTC scores, so we relax the threshold.
        ///
        /// Formula: `minVocabScore - (extraTokens * 1.0)`
        /// - 3 tokens: no adjustment (reference)
        /// - 5 tokens: threshold lowered by 2.0
        /// - 8 tokens: threshold lowered by 5.0
        ///
        /// - Parameters:
        ///   - tokenCount: Number of tokens in the vocabulary term
        /// - Returns: Adjusted minimum vocabulary score threshold
        public func adaptiveMinVocabScore(tokenCount: Int) -> Float {
            guard useAdaptiveThresholds else { return minVocabScore }
            let extraTokens = max(0, tokenCount - referenceTokenCount)
            return minVocabScore - Float(extraTokens) * 1.0
        }

        /// Compute adaptive context-biasing weight based on token count.
        /// Longer keywords need more boost to compensate for accumulated scoring error.
        ///
        /// Formula: `cbw * (1 + log2(tokenCount / referenceTokenCount) * 0.3)`
        /// - 3 tokens: cbw * 1.0 (reference)
        /// - 6 tokens: cbw * 1.3
        /// - 12 tokens: cbw * 1.6
        ///
        /// - Parameters:
        ///   - baseCbw: Base context-biasing weight
        ///   - tokenCount: Number of tokens in the vocabulary term
        /// - Returns: Adjusted context-biasing weight
        public func adaptiveCbw(baseCbw: Float, tokenCount: Int) -> Float {
            guard useAdaptiveThresholds, tokenCount > referenceTokenCount else { return baseCbw }
            let ratio = Float(tokenCount) / Float(referenceTokenCount)
            let scaleFactor = 1.0 + log2(ratio) * 0.3
            return baseCbw * scaleFactor
        }

        /// Compute adaptive minimum score advantage based on token count.
        /// Longer keywords may need less advantage since they're more distinctive.
        ///
        /// Formula: `minScoreAdvantage / sqrt(tokenCount / referenceTokenCount)`
        /// - 3 tokens: no adjustment (reference)
        /// - 6 tokens: advantage reduced to ~70%
        /// - 12 tokens: advantage reduced to ~50%
        ///
        /// - Parameters:
        ///   - tokenCount: Number of tokens in the vocabulary term
        /// - Returns: Adjusted minimum score advantage threshold
        public func adaptiveMinScoreAdvantage(tokenCount: Int) -> Float {
            guard useAdaptiveThresholds, tokenCount > referenceTokenCount else { return minScoreAdvantage }
            let ratio = Float(tokenCount) / Float(referenceTokenCount)
            return minScoreAdvantage / sqrt(ratio)
        }
    }

    let config: Config

    // MARK: - Async Factory

    /// Create rescorer asynchronously with CTC spotter and vocabulary.
    /// This is the recommended API as it avoids blocking during tokenizer initialization.
    ///
    /// - Parameters:
    ///   - spotter: CTC keyword spotter for generating log probabilities
    ///   - vocabulary: Custom vocabulary context with terms to detect
    ///   - config: Rescoring configuration (default: .default)
    ///   - ctcModelDirectory: Directory containing tokenizer.json (default: nil uses 110m model)
    /// - Returns: Initialized VocabularyRescorer
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public static func create(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config = .default,
        ctcModelDirectory: URL? = nil
    ) async throws -> VocabularyRescorer {
        let tokenizer: CtcTokenizer
        if let modelDir = ctcModelDirectory {
            tokenizer = try await CtcTokenizer.load(from: modelDir)
        } else {
            tokenizer = try await CtcTokenizer.load()
        }

        return VocabularyRescorer(
            spotter: spotter,
            vocabulary: vocabulary,
            config: config,
            ctcTokenizer: tokenizer
        )
    }

    /// Private initializer for async factory
    private init(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config,
        ctcTokenizer: CtcTokenizer
    ) {
        self.spotter = spotter
        self.vocabulary = vocabulary
        self.config = config
        self.ctcTokenizer = ctcTokenizer
        #if DEBUG
        self.debugMode = true  // Verbose logging in DEBUG builds
        #else
        self.debugMode = false
        #endif

        // BK-tree for efficient approximate string matching (disabled by default)
        // Enable for large vocabularies (>100 terms) where O(log V) lookup helps
        self.useBKTree = ContextBiasingConstants.useBkTree
        self.bkTreeMaxDistance = ContextBiasingConstants.bkTreeMaxDistance
        if useBKTree {
            self.bkTree = BKTree(terms: vocabulary.terms)
        } else {
            self.bkTree = nil
        }
    }

    // MARK: - Result Types

    /// Result of rescoring a word
    public struct RescoringResult: Sendable {
        public let originalWord: String
        public let originalScore: Float
        public let replacementWord: String?
        public let replacementScore: Float?
        public let shouldReplace: Bool
        public let reason: String
    }

    /// Output from rescoring operation
    public struct RescoreOutput: Sendable {
        public let text: String
        public let replacements: [RescoringResult]
        public let wasModified: Bool
    }

    // MARK: - Word Timing Utilities

    /// Word timing information built from TDT token timings
    public struct WordTiming: Sendable {
        public let word: String
        public let startTime: Double
        public let endTime: Double
        public let confidence: Float
        public let wordIndex: Int
    }

    /// Build word-level timings from token timings.
    /// Tokens starting with space " " or "▁" (SentencePiece) begin new words.
    func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
        var wordTimings: [WordTiming] = []
        var currentWord = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0
        var minConfidence: Float = 1.0
        var wordIndex = 0

        for timing in tokenTimings {
            let token = timing.token

            // Skip special tokens
            if token.isEmpty || token == "<blank>" || token == "<pad>" {
                continue
            }

            // Check if this starts a new word (space or ▁ prefix, or first token)
            let startsNewWord = isWordBoundary(token) || currentWord.isEmpty

            if startsNewWord && !currentWord.isEmpty {
                // Save previous word (trim any leading/trailing whitespace)
                let trimmedWord = currentWord.trimmingCharacters(in: .whitespaces)
                if !trimmedWord.isEmpty {
                    wordTimings.append(
                        WordTiming(
                            word: trimmedWord,
                            startTime: wordStart,
                            endTime: wordEnd,
                            confidence: minConfidence,
                            wordIndex: wordIndex
                        ))
                    wordIndex += 1
                }
                minConfidence = 1.0
                currentWord = ""
            }

            if startsNewWord {
                currentWord = stripWordBoundaryPrefix(token)
                wordStart = timing.startTime
            } else {
                currentWord += token
            }
            wordEnd = timing.endTime
            minConfidence = min(minConfidence, timing.confidence)
        }

        // Save final word
        let trimmedWord = currentWord.trimmingCharacters(in: .whitespaces)
        if !trimmedWord.isEmpty {
            wordTimings.append(
                WordTiming(
                    word: trimmedWord,
                    startTime: wordStart,
                    endTime: wordEnd,
                    confidence: minConfidence,
                    wordIndex: wordIndex
                ))
        }

        return wordTimings
    }

}
