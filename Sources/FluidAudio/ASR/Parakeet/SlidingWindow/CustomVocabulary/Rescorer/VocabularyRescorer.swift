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

    // BK-tree for efficient approximate string matching (experimental)
    // When enabled, uses BK-tree to find candidate vocabulary terms within edit distance
    // instead of iterating all terms. Provides O(log n) vs O(n) for large vocabularies.
    let useBKTree: Bool
    let bkTree: BKTree?
    let bkTreeMaxDistance: Int

    /// Configuration for rescoring behavior
    public struct Config: Sendable {
        /// Enable adaptive thresholds based on token count
        /// When true, thresholds are adjusted for longer vocabulary terms
        public let useAdaptiveThresholds: Bool

        /// Reference token count for adaptive scaling (tokens beyond this get adjusted thresholds)
        public let referenceTokenCount: Int

        /// Token-count pivot for the short-term cbw taper (#702, opt-in).
        /// A value `<= 1` disables the taper (default). When enabled (e.g. 5),
        /// terms with fewer tokens than the pivot get a reduced boost so they
        /// can't beat a correctly transcribed common word on the flat boost
        /// alone. Trades short-vocab precision for some recall on
        /// distinctive-name vocabularies — see #702.
        public let shortTermCbwTaperPivot: Int

        /// Exponent for the short-term cbw taper. Higher = more conservative.
        public let shortTermCbwTaperExponent: Float

        /// Minimum string similarity for a single-word spotter-anchored rescue
        /// (#702, opt-in). `0.0` disables (default), preserving the
        /// acoustic-only rescue; a positive value (e.g. 0.30) suppresses
        /// near-zero-similarity over-fires at some recall cost.
        public let spotterRescueMinSimilarity: Float

        /// Minimum string similarity for a multi-word spotter-anchored rescue.
        /// `0.0` disables (default). Recommended opt-in: 0.50.
        public let spotterRescueMultiWordMinSimilarity: Float

        /// Whether the spotter-anchored acoustic rescue pass runs at all (#724).
        /// `true` (default) = current behavior. Setting `false` skips the
        /// acoustic rescue entirely, reproducing the pre-#634 (0.14.5) behavior
        /// — far fewer short-keyword over-fires (#702) at no recall cost on
        /// distinctive-name vocabularies, but loses recovery of brand names TDT
        /// mangles past the string-similarity gate. Prefer this for short-vocab
        /// KWS. (Composes with `spotterRescueMinSimilarity`: this is the on/off
        /// switch, that is the in-between similarity floor.)
        public let spotterRescueEnabled: Bool

        public static let `default` = Config()

        public init(
            useAdaptiveThresholds: Bool = ContextBiasingConstants.defaultUseAdaptiveThresholds,
            referenceTokenCount: Int = ContextBiasingConstants.defaultReferenceTokenCount,
            shortTermCbwTaperPivot: Int = ContextBiasingConstants.defaultShortTermCbwTaperPivot,
            shortTermCbwTaperExponent: Float = ContextBiasingConstants.defaultShortTermCbwTaperExponent,
            spotterRescueMinSimilarity: Float = ContextBiasingConstants.defaultSpotterRescueMinSimilarity,
            spotterRescueMultiWordMinSimilarity: Float = ContextBiasingConstants
                .defaultSpotterRescueMultiWordMinSimilarity,
            spotterRescueEnabled: Bool = ContextBiasingConstants.defaultSpotterRescueEnabled
        ) {
            self.useAdaptiveThresholds = useAdaptiveThresholds
            self.referenceTokenCount = referenceTokenCount
            self.shortTermCbwTaperPivot = shortTermCbwTaperPivot
            self.shortTermCbwTaperExponent = shortTermCbwTaperExponent
            self.spotterRescueMinSimilarity = spotterRescueMinSimilarity
            self.spotterRescueMultiWordMinSimilarity = spotterRescueMultiWordMinSimilarity
            self.spotterRescueEnabled = spotterRescueEnabled
        }

        // MARK: - Adaptive Threshold Functions

        /// Compute adaptive context-biasing weight based on token count.
        ///
        /// Longer keywords need more boost to compensate for accumulated
        /// scoring error; shorter keywords need *less* boost because their
        /// per-token DP score is inflated by free-start alignment (#702).
        ///
        /// - Long terms (`tokenCount > referenceTokenCount`):
        ///   `cbw * (1 + log2(tokenCount / referenceTokenCount) * 0.3)`
        ///   - 6 tokens: cbw * 1.3, 12 tokens: cbw * 1.6
        /// - Reference length (`== referenceTokenCount`): `cbw` unchanged.
        /// - Short terms (`tokenCount < referenceTokenCount`):
        ///   `cbw * (tokenCount / referenceTokenCount) ** shortTermCbwTaperExponent`,
        ///   tapering the boost toward zero so a short keyword cannot beat a
        ///   correctly transcribed common word on the flat boost alone — it
        ///   must earn the margin from acoustic evidence.
        ///
        /// - Parameters:
        ///   - baseCbw: Base context-biasing weight
        ///   - tokenCount: Number of tokens in the vocabulary term
        /// - Returns: Adjusted context-biasing weight
        public func adaptiveCbw(baseCbw: Float, tokenCount: Int) -> Float {
            guard useAdaptiveThresholds else { return baseCbw }

            // Short-term taper (#702, opt-in): below the pivot, scale the
            // boost down toward zero so a short keyword cannot beat a correctly
            // transcribed common word on the flat boost alone. Disabled when
            // pivot <= 1 (default), which leaves the original behavior intact.
            let pivot = shortTermCbwTaperPivot
            if pivot > 1 && tokenCount < pivot {
                let ratio = Float(max(1, tokenCount)) / Float(pivot)
                return baseCbw * pow(ratio, shortTermCbwTaperExponent)
            }

            // Long terms: boost grows to compensate for accumulated per-token
            // scoring error.
            guard tokenCount > referenceTokenCount else { return baseCbw }
            let ratio = Float(tokenCount) / Float(referenceTokenCount)
            return baseCbw * (1.0 + log2(ratio) * 0.3)
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

        let useBKTree = ContextBiasingConstants.useBkTree
        let bkTree: BKTree? = useBKTree ? BKTree(terms: vocabulary.terms) : nil

        return VocabularyRescorer(
            spotter: spotter,
            vocabulary: vocabulary,
            config: config,
            ctcTokenizer: tokenizer,
            useBKTree: useBKTree,
            bkTree: bkTree,
            bkTreeMaxDistance: ContextBiasingConstants.bkTreeMaxDistance
        )
    }

    /// Private initializer for async factory
    private init(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config,
        ctcTokenizer: CtcTokenizer,
        useBKTree: Bool,
        bkTree: BKTree?,
        bkTreeMaxDistance: Int
    ) {
        self.spotter = spotter
        self.vocabulary = vocabulary
        self.config = config
        self.ctcTokenizer = ctcTokenizer
        self.useBKTree = useBKTree
        self.bkTree = bkTree
        self.bkTreeMaxDistance = bkTreeMaxDistance
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

    /// The discovery path that produced a vocabulary candidate.
    public enum CandidateOrigin: String, Sendable, Equatable {
        /// Candidate discovered while searching vocabulary terms around one TDT word.
        case wordCentric

        /// Candidate discovered from a single-word term-centric match.
        case termCentricSingleWord

        /// Candidate discovered from a multi-word term-centric match.
        case termCentricMultiWord

        /// Candidate discovered from the CTC keyword-spotter rescue pass.
        case spotterRescue
    }

    /// The final action taken by the legacy CTC token rescorer for a candidate.
    public enum LegacyApplicationOutcome: String, Sendable, Equatable {
        /// The candidate passed comparison and was applied to the legacy transcript.
        case applied

        /// The legacy floating-point comparison ran and rejected the candidate.
        case rejectedByComparison

        /// The legacy comparison could not be completed because required evidence was unavailable.
        case unavailableEvidence

        /// The candidate passed comparison but an overlapping candidate won final arbitration.
        case supersededByOverlap
    }

    /// Diagnostic evidence for one candidate evaluated by the legacy CTC token rescorer.
    public struct CandidateEvidence: Sendable, Equatable {
        /// Non-negative correlation identifier unique only within one evidence output.
        public let candidateID: Int

        /// Discovery path that produced this candidate evaluation.
        public let origin: CandidateOrigin

        /// The untouched phrase from the base transcript that was evaluated.
        public let basePhrase: String

        /// The canonical vocabulary term proposed as a replacement.
        public let canonicalTerm: String

        /// The exact alias that produced the best string match, or `nil` when the canonical term matched.
        public let matchedAlias: String?

        /// String similarity used to rank and gate the candidate.
        public let similarity: Float

        /// Raw CTC score for the vocabulary term, before context biasing, when available.
        public let rawVocabularyCTCScore: Float?

        /// Raw CTC score for the original phrase, when available.
        public let rawOriginalCTCScore: Float?

        /// Context-biasing boost added to the vocabulary score, when available.
        public let effectiveBoost: Float?

        /// Half-open word-index range into ``CandidateEvidenceOutput/baseWords``.
        public let wordRange: Range<Int>

        /// Half-open token-index range in the supplied token timings, when available and contiguous.
        public let tokenRange: Range<Int>?

        /// Half-open UTF-8 byte range into ``CandidateEvidenceOutput/baseText``, when exact alignment is available.
        public let baseTextUTF8Range: Range<Int>?

        /// Start time of the evaluated base phrase, when available.
        public let startTime: TimeInterval?

        /// End time of the evaluated base phrase, when available.
        public let endTime: TimeInterval?

        /// Whether the boosted vocabulary score passed the pre-arbitration acoustic comparison.
        public let comparisonPassed: Bool

        /// Final action taken by the legacy rescorer after overlap arbitration.
        public let legacyOutcome: LegacyApplicationOutcome

        /// Human-readable reason emitted by the existing replacement behavior.
        public let reason: String

        /// Creates diagnostic evidence for one evaluated vocabulary candidate.
        public init(
            candidateID: Int,
            origin: CandidateOrigin,
            basePhrase: String,
            canonicalTerm: String,
            matchedAlias: String?,
            similarity: Float,
            rawVocabularyCTCScore: Float?,
            rawOriginalCTCScore: Float?,
            effectiveBoost: Float?,
            wordRange: Range<Int>,
            tokenRange: Range<Int>?,
            baseTextUTF8Range: Range<Int>?,
            startTime: TimeInterval?,
            endTime: TimeInterval?,
            comparisonPassed: Bool,
            legacyOutcome: LegacyApplicationOutcome,
            reason: String
        ) {
            self.candidateID = candidateID
            self.origin = origin
            self.basePhrase = basePhrase
            self.canonicalTerm = canonicalTerm
            self.matchedAlias = matchedAlias
            self.similarity = similarity
            self.rawVocabularyCTCScore = rawVocabularyCTCScore
            self.rawOriginalCTCScore = rawOriginalCTCScore
            self.effectiveBoost = effectiveBoost
            self.wordRange = wordRange
            self.tokenRange = tokenRange
            self.baseTextUTF8Range = baseTextUTF8Range
            self.startTime = startTime
            self.endTime = endTime
            self.comparisonPassed = comparisonPassed
            self.legacyOutcome = legacyOutcome
            self.reason = reason
        }

        /// Returns the same evidence with its final legacy arbitration outcome replaced.
        func replacingLegacyOutcome(_ outcome: LegacyApplicationOutcome) -> CandidateEvidence {
            CandidateEvidence(
                candidateID: candidateID,
                origin: origin,
                basePhrase: basePhrase,
                canonicalTerm: canonicalTerm,
                matchedAlias: matchedAlias,
                similarity: similarity,
                rawVocabularyCTCScore: rawVocabularyCTCScore,
                rawOriginalCTCScore: rawOriginalCTCScore,
                effectiveBoost: effectiveBoost,
                wordRange: wordRange,
                tokenRange: tokenRange,
                baseTextUTF8Range: baseTextUTF8Range,
                startTime: startTime,
                endTime: endTime,
                comparisonPassed: comparisonPassed,
                legacyOutcome: outcome,
                reason: reason
            )
        }
    }

    /// Non-mutating diagnostic output from candidate evaluation.
    public struct CandidateEvidenceOutput: Sendable, Equatable {
        /// The untouched transcript supplied to the rescorer.
        public let baseText: String

        /// Exact internal word sequence that every candidate ``CandidateEvidence/wordRange`` indexes.
        public let baseWords: [String]

        /// Every candidate that reached the legacy CTC evaluation, including rejections.
        public let candidates: [CandidateEvidence]

        /// Creates diagnostic output for an untouched base transcript.
        public init(baseText: String, baseWords: [String], candidates: [CandidateEvidence]) {
            self.baseText = baseText
            self.baseWords = baseWords
            self.candidates = candidates
        }
    }

    // MARK: - Word Timing Utilities

    /// Word timing information built from TDT token timings
    public struct WordTiming: Sendable {
        public let word: String
        public let startTime: Double
        public let endTime: Double
        let tokenRange: Range<Int>?
    }

    /// Build word-level timings from token timings.
    /// Tokens starting with space " " or "▁" (SentencePiece) begin new words.
    static func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
        var wordTimings: [WordTiming] = []
        var currentWord = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0
        var wordTokenStart = 0
        var wordTokenEnd = 0
        var wordTokensAreContiguous = true

        for (tokenIndex, timing) in tokenTimings.enumerated() {
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
                            tokenRange: wordTokensAreContiguous ? wordTokenStart..<wordTokenEnd : nil
                        ))
                }
                currentWord = ""
            }

            if startsNewWord {
                currentWord = stripWordBoundaryPrefix(token)
                wordStart = timing.startTime
                wordTokenStart = tokenIndex
                wordTokensAreContiguous = true
            } else {
                if tokenIndex != wordTokenEnd {
                    wordTokensAreContiguous = false
                }
                currentWord += token
            }
            wordEnd = timing.endTime
            wordTokenEnd = tokenIndex + 1
        }

        // Save final word
        let trimmedWord = currentWord.trimmingCharacters(in: .whitespaces)
        if !trimmedWord.isEmpty {
            wordTimings.append(
                WordTiming(
                    word: trimmedWord,
                    startTime: wordStart,
                    endTime: wordEnd,
                    tokenRange: wordTokensAreContiguous ? wordTokenStart..<wordTokenEnd : nil
                ))
        }

        return wordTimings
    }

}
