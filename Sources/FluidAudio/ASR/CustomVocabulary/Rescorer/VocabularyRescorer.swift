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

    /// Configuration for rescoring behavior
    public struct Config: Sendable {
        /// Enable adaptive thresholds based on token count
        /// When true, thresholds are adjusted for longer vocabulary terms
        public let useAdaptiveThresholds: Bool

        /// Reference token count for adaptive scaling (tokens beyond this get adjusted thresholds)
        public let referenceTokenCount: Int

        public static let `default` = Config(
            useAdaptiveThresholds: ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: ContextBiasingConstants.defaultReferenceTokenCount
        )

        public init(
            useAdaptiveThresholds: Bool = ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: Int = ContextBiasingConstants.defaultReferenceTokenCount
        ) {
            self.useAdaptiveThresholds = useAdaptiveThresholds
            self.referenceTokenCount = referenceTokenCount
        }

        // MARK: - Adaptive Threshold Functions

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
    }

    /// Build word-level timings from token timings.
    /// Tokens starting with space " " or "▁" (SentencePiece) begin new words.
    func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
        var wordTimings: [WordTiming] = []
        var currentWord = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0

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
                            endTime: wordEnd
                        ))
                }
                currentWord = ""
            }

            if startsNewWord {
                currentWord = stripWordBoundaryPrefix(token)
                wordStart = timing.startTime
            } else {
                currentWord += token
            }
            wordEnd = timing.endTime
        }

        // Save final word
        let trimmedWord = currentWord.trimmingCharacters(in: .whitespaces)
        if !trimmedWord.isEmpty {
            wordTimings.append(
                WordTiming(
                    word: trimmedWord,
                    startTime: wordStart,
                    endTime: wordEnd
                ))
        }

        return wordTimings
    }

}
