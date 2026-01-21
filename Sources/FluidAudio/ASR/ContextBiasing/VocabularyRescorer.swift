import Foundation
import OSLog

/// CTC-based vocabulary rescoring for principled vocabulary integration.
///
/// Instead of blindly replacing words based on phonetic similarity, this rescorer
/// uses CTC log-probabilities to verify that vocabulary terms actually match the audio.
/// Only replaces when the vocabulary term has significantly higher acoustic evidence.
///
/// This implements "shallow fusion" or "CTC rescoring" - a standard technique in ASR.
/// The rescorer computes ACTUAL CTC scores for both vocabulary terms AND original words,
/// enabling a fair comparison rather than relying on heuristics.
public struct VocabularyRescorer {

    let logger = Logger(subsystem: "com.fluidaudio", category: "VocabularyRescorer")

    /// Log debug message with lazy evaluation (only formats string when debugMode is true)
    @inline(__always)
    private func debugLog(_ message: @escaping @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug("\(message(), privacy: .public)")
    }

    let spotter: CtcKeywordSpotter
    let vocabulary: CustomVocabularyContext
    let ctcTokenizer: CtcTokenizer?
    private let ctcModelDirectory: URL?
    let debugMode: Bool

    // BK-tree for efficient approximate string matching (USE_BK_TREE=1 to enable)
    // When enabled, uses BK-tree to find candidate vocabulary terms within edit distance
    // instead of iterating all terms. Provides O(log n) vs O(n) for large vocabularies.
    let useBKTree: Bool
    private let bkTree: BKTree?
    private let bkTreeMaxDistance: Int

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
            minScoreAdvantage: 2.0,  // Vocab term must score 2.0 better than original
            minVocabScore: -12.0,  // Vocab term must have reasonable CTC score (lowered for alias support)
            maxOriginalScoreForReplacement: -4.0,  // Don't replace very confident words
            vocabBoostWeight: 1.5,  // Boost for vocabulary terms
            useAdaptiveThresholds: true,  // Enable adaptive thresholds by default
            referenceTokenCount: 3  // Tokens beyond 3 get adjusted thresholds
        )

        public init(
            minScoreAdvantage: Float = 2.0,
            minVocabScore: Float = -8.0,
            maxOriginalScoreForReplacement: Float = -4.0,
            vocabBoostWeight: Float = 1.5,
            useAdaptiveThresholds: Bool = true,
            referenceTokenCount: Int = 3
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
        self.ctcModelDirectory = nil
        self.ctcTokenizer = ctcTokenizer
        self.debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        // Initialize BK-tree for efficient approximate string matching (optional)
        self.useBKTree = ProcessInfo.processInfo.environment["USE_BK_TREE"] == "1"
        self.bkTreeMaxDistance =
            ProcessInfo.processInfo.environment["BK_TREE_MAX_DISTANCE"]
            .flatMap { Int($0) } ?? 3
        if useBKTree {
            self.bkTree = BKTree(terms: vocabulary.terms)
            if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                let initLogger = Logger(subsystem: "com.fluidaudio", category: "VocabularyRescorer")
                let termCount = vocabulary.terms.count
                let maxDist = bkTreeMaxDistance
                initLogger.debug("[BK-TREE] Initialized with \(termCount) terms, maxDistance=\(maxDist)")
            }
        } else {
            self.bkTree = nil
        }
    }

    /// Result of rescoring a word
    public struct RescoringResult: Sendable {
        public let originalWord: String
        public let originalScore: Float
        public let replacementWord: String?
        public let replacementScore: Float?
        public let shouldReplace: Bool
        public let reason: String
    }

    /// Rescore a transcript using CTC evidence with principled scoring.
    /// This method computes ACTUAL CTC scores for original words using the cached log-probs,
    /// enabling a fair comparison between vocabulary terms and original transcript words.
    ///
    /// - Parameters:
    ///   - transcript: Original transcript from TDT decoder
    ///   - spotResult: Result from spotKeywordsWithLogProbs containing detections and cached log-probs
    /// - Returns: Rescored transcript with replacements only where acoustically justified
    public func rescore(
        transcript: String,
        spotResult: CtcKeywordSpotter.SpotKeywordsResult
    ) -> RescoreOutput {
        let words = transcript.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        guard !words.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        let detections = spotResult.detections
        let logProbs = spotResult.logProbs
        let hasLogProbs = !logProbs.isEmpty

        if debugMode {
            logger.info("=== VocabularyRescorer ===")
            logger.info("Transcript: \(transcript)")
            logger.info("Detections: \(detections.count)")
            logger.info("CTC log-probs available: \(hasLogProbs) (frames: \(logProbs.count))")
            if hasLogProbs && ctcTokenizer != nil {
                logger.info("Using ACTUAL CTC scoring for original words")
            } else {
                logger.info("Using heuristic scoring (CTC log-probs or tokenizer unavailable)")
            }
        }

        var replacements: [RescoringResult] = []
        var modifiedWords = words

        // Track which word indices have already been replaced to avoid double replacements
        var replacedIndices = Set<Int>()

        // For each detection, search the ENTIRE transcript for the best matching word
        // Time-based indexing is unreliable, so we scan all words
        for detection in detections {
            let vocabTerm = detection.term.text
            let vocabScore = detection.score

            // Get token count for adaptive threshold calculation
            let vocabTokenCount = detection.term.ctcTokenIds?.count ?? detection.term.tokenIds?.count ?? 3

            // Compute adaptive minimum vocab score based on token count
            let adaptiveMinScore = config.adaptiveMinVocabScore(tokenCount: vocabTokenCount)

            // Skip if vocab term doesn't meet minimum score
            guard vocabScore >= adaptiveMinScore else {
                if debugMode {
                    let baseScore = config.minVocabScore
                    let thresholdInfo =
                        config.useAdaptiveThresholds
                        ? "adaptive=\(String(format: "%.2f", adaptiveMinScore)) (base=\(String(format: "%.2f", baseScore)), tokens=\(vocabTokenCount))"
                        : String(format: "%.2f", baseScore)
                    logger.debug(
                        "Skipping '\(vocabTerm)': CTC score \(String(format: "%.2f", vocabScore)) < min \(thresholdInfo)"
                    )
                }
                continue
            }

            var bestCandidate:
                (wordIndex: Int, originalWord: String, similarity: Float, isHighConfidenceAlias: Bool, spanLength: Int)?

            // Build list of all forms to check (canonical + aliases)
            // Look up the canonical term in the vocabulary to get ALL aliases
            // (since detections may come from entries without alias info)
            var allForms = [vocabTerm]
            let vocabTermLower = detection.term.textLowercased
            for term in vocabulary.terms {
                if term.textLowercased == vocabTermLower {
                    if let aliases = term.aliases {
                        allForms.append(contentsOf: aliases)
                    }
                }
            }
            // Also add aliases from the detection itself (in case it has unique ones)
            if let aliases = detection.term.aliases {
                for alias in aliases where !allForms.contains(alias.lowercased()) {
                    allForms.append(alias)
                }
            }

            // Use the standard similarity threshold
            let effectiveMinSimilarity = vocabulary.minSimilarity

            // Search the ENTIRE transcript for matching words or phrases
            for idx in 0..<words.count {
                // Skip already replaced words
                guard !replacedIndices.contains(idx) else { continue }

                // Check for similarity match against canonical term and aliases
                var bestSimilarity: Float = 0
                var isHighConfidenceAliasMatch = false
                var matchedSpanLength = 1

                for form in allForms {
                    let formWords = form.split(whereSeparator: { $0.isWhitespace })
                    let spanLength = formWords.count

                    // Ensure span fits in transcript
                    if idx + spanLength > words.count { continue }

                    // Construct transcript span
                    let transcriptSpan = words[idx..<idx + spanLength]
                        .map { $0.trimmingCharacters(in: .punctuationCharacters) }
                        .joined(separator: " ")

                    let similarity = Self.stringSimilarity(transcriptSpan, form)
                    if similarity >= bestSimilarity {
                        // Update best if strictly better, OR if equal and this is an alias match
                        if similarity > bestSimilarity {
                            bestSimilarity = similarity
                            matchedSpanLength = spanLength
                        }
                        // High confidence if similarity >= 0.65 and matching an alias
                        if similarity >= 0.65 && form != vocabTerm {
                            isHighConfidenceAliasMatch = true
                        }
                    }

                    // COMPOUND WORD MATCHING: For single-word vocabulary terms, also try
                    // matching against concatenated adjacent transcript words.
                    // This handles cases like "Newrez" being transcribed as "new res".
                    if spanLength == 1 {
                        // Try concatenating 2 adjacent words: "new res" → "newres"
                        if idx + 2 <= words.count {
                            let word1 = words[idx].trimmingCharacters(in: .punctuationCharacters)
                            let word2 = words[idx + 1].trimmingCharacters(in: .punctuationCharacters)
                            let concatenated = word1 + word2  // No space
                            let concatSimilarity = Self.stringSimilarity(concatenated, form)

                            if concatSimilarity > bestSimilarity {
                                bestSimilarity = concatSimilarity
                                matchedSpanLength = 2  // Replacing 2 words with 1
                            }
                        }

                        // Try concatenating 3 adjacent words for longer compound words
                        if idx + 3 <= words.count && form.count >= 8 {
                            let word1 = words[idx].trimmingCharacters(in: .punctuationCharacters)
                            let word2 = words[idx + 1].trimmingCharacters(in: .punctuationCharacters)
                            let word3 = words[idx + 2].trimmingCharacters(in: .punctuationCharacters)
                            let concatenated = word1 + word2 + word3
                            let concatSimilarity = Self.stringSimilarity(concatenated, form)

                            if concatSimilarity > bestSimilarity {
                                bestSimilarity = concatSimilarity
                                matchedSpanLength = 3
                            }
                        }
                    }
                }

                // Debug: show all similarity calculations for high-similarity matches
                if bestSimilarity >= 0.50 {
                    let wordClean = words[idx].trimmingCharacters(in: .punctuationCharacters)
                    debugLog("    [SIM] '\(wordClean)' vs '\(vocabTerm)' = \(String(format: "%.2f", bestSimilarity))")
                }

                // LENGTH RATIO CHECK: Prevent short common words from matching longer vocab terms
                // e.g., "and" (3 chars) should not match "Andre" (5 chars) even with ~60% similarity
                let originalWord = words[idx].trimmingCharacters(in: .punctuationCharacters).lowercased()
                let lengthRatio = Float(originalWord.count) / Float(detection.term.textLowercased.count)

                // If original word is much shorter than vocab term, require higher similarity
                // Ratio < 0.75 means original is 25%+ shorter (e.g., "and"=3 / "andre"=5 = 0.6)
                var adjustedMinSimilarity = effectiveMinSimilarity
                if lengthRatio < 0.75 && originalWord.count <= 4 {
                    // For short words with low length ratio, require much higher similarity
                    adjustedMinSimilarity = max(effectiveMinSimilarity, 0.80)
                    if bestSimilarity >= effectiveMinSimilarity {
                        debugLog(
                            "    [LENGTH] '\(originalWord)' too short (ratio=\(String(format: "%.2f", lengthRatio))), "
                                + "raising threshold to \(String(format: "%.2f", adjustedMinSimilarity))"
                        )
                    }
                }

                if bestSimilarity >= adjustedMinSimilarity {
                    let originalSpan = words[idx..<idx + matchedSpanLength].joined(separator: " ")

                    if let existing = bestCandidate {
                        if bestSimilarity > existing.similarity {
                            bestCandidate = (
                                idx, originalSpan, bestSimilarity, isHighConfidenceAliasMatch, matchedSpanLength
                            )
                        }
                    } else {
                        bestCandidate = (
                            idx, originalSpan, bestSimilarity, isHighConfidenceAliasMatch, matchedSpanLength
                        )
                    }
                }
            }

            guard let candidate = bestCandidate else {
                continue
            }

            debugLog(
                "  [CANDIDATE] '\(candidate.originalWord)' -> '\(vocabTerm)' (sim=\(String(format: "%.2f", candidate.similarity)), isHighConfAlias=\(candidate.isHighConfidenceAlias), span=\(candidate.spanLength))"
            )

            // Now the key decision: Should we replace?
            // We need to compare CTC score for the vocabulary term vs the original word

            // The vocab term already has a CTC score from the detection
            let vocabCtcScore = vocabScore + config.vocabBoostWeight

            // Compute the ACTUAL CTC score for the original word if we have log-probs and tokenizer
            let originalScore: Float
            let scoringMethod: String

            if hasLogProbs, let tokenizer = ctcTokenizer {
                // PRINCIPLED APPROACH: Tokenize original word and run CTC DP
                let cleanedOriginal = candidate.originalWord.trimmingCharacters(in: .punctuationCharacters)
                let originalTokenIds = tokenizer.encode(cleanedOriginal)

                if !originalTokenIds.isEmpty {
                    let (ctcScore, _, _) = spotter.scoreWord(logProbs: logProbs, keywordTokens: originalTokenIds)
                    originalScore = ctcScore
                    scoringMethod = "actual"

                    if debugMode {
                        logger.debug(
                            "    Original '\(cleanedOriginal)' tokenized to \(originalTokenIds), CTC score: \(String(format: "%.2f", ctcScore))"
                        )
                    }
                } else {
                    // Tokenization failed, fall back to heuristic
                    originalScore = estimateOriginalWordScore(
                        detection: detection,
                        originalWord: candidate.originalWord,
                        similarity: candidate.similarity
                    )
                    scoringMethod = "heuristic (tokenization failed)"
                }
            } else {
                // FALLBACK: Use heuristic when log-probs or tokenizer unavailable
                originalScore = estimateOriginalWordScore(
                    detection: detection,
                    originalWord: candidate.originalWord,
                    similarity: candidate.similarity
                )
                scoringMethod = "heuristic"
            }

            let scoreAdvantage = vocabCtcScore - originalScore

            debugLog(
                "  Vocab CTC: \(String(format: "%.2f", vocabCtcScore)), Original (\(scoringMethod)): \(String(format: "%.2f", originalScore)), Advantage: \(String(format: "%.2f", scoreAdvantage))"
            )

            // Decision criteria:
            // 1. HIGH CONFIDENCE ALIAS: If similarity >= 0.85 to a user-defined alias, trust the mapping
            // 2. Otherwise: Vocabulary term must have significant score advantage AND original not too confident
            let shouldReplace: Bool
            let reason: String

            if candidate.isHighConfidenceAlias && candidate.similarity >= 0.65 {
                // User explicitly defined this alias mapping - trust it with moderate similarity
                shouldReplace = true
                reason = "High-confidence alias match (sim: \(String(format: "%.2f", candidate.similarity)))"

                // Replace span
                // We need to replace words[idx..<idx+span] with [replacement]
                // But we are modifying a copy. We can't easily do spans with simple array assignment if we are tracking indices.
                // For simplicity, we replace the first word and clear the others.
                modifiedWords[candidate.wordIndex] = preserveCapitalization(
                    original: candidate.originalWord,
                    replacement: vocabTerm
                )
                for i in 1..<candidate.spanLength {
                    modifiedWords[candidate.wordIndex + i] = ""  // Mark for removal
                }

                for i in 0..<candidate.spanLength {
                    replacedIndices.insert(candidate.wordIndex + i)
                }

            } else if candidate.spanLength >= 2 && candidate.similarity >= 0.80 && scoreAdvantage >= 0.5 {
                // COMPOUND WORD MATCH: For multi-word spans (e.g., "new res" -> "Newrez"),
                // high string similarity (>=0.80) is strong evidence even with lower CTC advantage.
                // Threshold raised from 0.75 to 0.80 to avoid false positives like "and I" -> "Audi" (sim=0.75)
                shouldReplace = true
                reason =
                    "Compound word match (span=\(candidate.spanLength), sim=\(String(format: "%.2f", candidate.similarity)), advantage=\(String(format: "%.2f", scoreAdvantage)))"

                modifiedWords[candidate.wordIndex] = preserveCapitalization(
                    original: candidate.originalWord,
                    replacement: vocabTerm
                )
                for i in 1..<candidate.spanLength {
                    modifiedWords[candidate.wordIndex + i] = ""
                }

                for i in 0..<candidate.spanLength {
                    replacedIndices.insert(candidate.wordIndex + i)
                }

            } else {
                // Compute adaptive score advantage threshold based on token count
                let adaptiveMinAdvantage = config.adaptiveMinScoreAdvantage(tokenCount: vocabTokenCount)

                if scoreAdvantage >= adaptiveMinAdvantage
                    && originalScore <= config.maxOriginalScoreForReplacement
                {
                    // Similarity threshold depends on span length and word length:
                    // - Multi-word (span≥2): higher threshold (0.80) - prevents "want to"→"Santoro", "and I"→"Audi"
                    // - Single word, short (≤3 chars): very high threshold (0.85) - prevents "you"→"Yu"
                    // - Single word, longer (>3 chars): lower threshold (0.55) - allows "NECI"→"Nequi"
                    let minSimilarityForSpan: Float
                    if candidate.spanLength >= 2 {
                        minSimilarityForSpan = 0.80
                    } else if candidate.originalWord.count <= 3 {
                        // Short words are often common English words - require very high similarity
                        minSimilarityForSpan = 0.85
                    } else {
                        minSimilarityForSpan = 0.55
                    }

                    if candidate.similarity >= minSimilarityForSpan {
                        // Standard CTC-based replacement
                        shouldReplace = true
                        let thresholdInfo =
                            config.useAdaptiveThresholds
                            ? "adaptive=\(String(format: "%.2f", adaptiveMinAdvantage)) (base=\(String(format: "%.2f", config.minScoreAdvantage)), tokens=\(vocabTokenCount))"
                            : String(format: "%.2f", config.minScoreAdvantage)
                        reason =
                            "Vocab score advantage: \(String(format: "%.2f", scoreAdvantage)) >= \(thresholdInfo), sim=\(String(format: "%.2f", candidate.similarity))"

                        modifiedWords[candidate.wordIndex] = preserveCapitalization(
                            original: candidate.originalWord,
                            replacement: vocabTerm
                        )
                        for i in 1..<candidate.spanLength {
                            modifiedWords[candidate.wordIndex + i] = ""
                        }

                        for i in 0..<candidate.spanLength {
                            replacedIndices.insert(candidate.wordIndex + i)
                        }
                    } else {
                        shouldReplace = false
                        reason =
                            "Similarity too low for span \(candidate.spanLength): \(String(format: "%.2f", candidate.similarity)) < \(String(format: "%.2f", minSimilarityForSpan))"
                    }
                } else if originalScore > config.maxOriginalScoreForReplacement {
                    shouldReplace = false
                    reason = "Original word too confident: \(String(format: "%.2f", originalScore))"
                } else {
                    shouldReplace = false
                    let thresholdInfo =
                        config.useAdaptiveThresholds
                        ? "adaptive=\(String(format: "%.2f", adaptiveMinAdvantage)) (base=\(String(format: "%.2f", config.minScoreAdvantage)), tokens=\(vocabTokenCount))"
                        : String(format: "%.2f", config.minScoreAdvantage)
                    reason = "Score advantage too low: \(String(format: "%.2f", scoreAdvantage)) < \(thresholdInfo)"
                }
            }

            replacements.append(
                RescoringResult(
                    originalWord: candidate.originalWord,
                    originalScore: originalScore,
                    replacementWord: shouldReplace ? vocabTerm : nil,
                    replacementScore: shouldReplace ? vocabCtcScore : nil,
                    shouldReplace: shouldReplace,
                    reason: reason
                ))

            let action = shouldReplace ? "REPLACE" : "KEEP"
            debugLog("  [\(action)] '\(candidate.originalWord)' -> '\(vocabTerm)': \(reason)")
        }

        // Remove empty words (cleared spans)
        let finalWords = modifiedWords.filter { !$0.isEmpty }
        let modifiedText = finalWords.joined(separator: " ")
        let wasModified = modifiedText != transcript

        debugLog("Final: \(modifiedText)")
        debugLog("Modified: \(wasModified)")
        debugLog("===========================")

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
    }

    /// Output from rescoring operation
    public struct RescoreOutput: Sendable {
        public let text: String
        public let replacements: [RescoringResult]
        public let wasModified: Bool
    }

    // MARK: - Timestamp-Based Rescoring (NeMo CTC-WS Algorithm)

    /// Word timing information built from TDT token timings
    public struct WordTiming: Sendable {
        public let word: String
        public let startTime: Double
        public let endTime: Double
        public let confidence: Float
        public let wordIndex: Int
    }

    /// Rescore using timestamp-based matching (NeMo CTC-WS algorithm).
    /// Instead of string similarity, matches CTC detections to TDT words by overlapping timestamps.
    ///
    /// - Parameters:
    ///   - transcript: Original transcript from TDT decoder
    ///   - tokenTimings: Token-level timings from TDT decoder
    ///   - spotResult: CTC keyword spotting result with detections and timestamps
    ///   - cbw: Context-biasing weight (default 3.0 per NeMo paper)
    /// - Returns: Rescored transcript with timestamp-based replacements and insertions
    public func rescoreWithTimings(
        transcript: String,
        tokenTimings: [TokenTiming],
        spotResult: CtcKeywordSpotter.SpotKeywordsResult,
        cbw: Float = 3.0
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty else {
            // Fall back to string-similarity based rescoring
            return rescore(transcript: transcript, spotResult: spotResult)
        }

        let detections = spotResult.detections
        guard !detections.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        debugLog("=== VocabularyRescorer (Timestamp-Based) ===")
        debugLog("Words: \(wordTimings.count), Detections: \(detections.count)")
        debugLog("CBW (context-biasing weight): \(cbw)")

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var insertions: [(word: String, insertAfterIndex: Int, time: Double)] = []
        var replacedIndices = Set<Int>()

        // Process each CTC detection
        for detection in detections {
            let vocabTerm = detection.term.text
            let vocabScore = detection.score + cbw  // Apply context-biasing weight
            let detectionStart = detection.startTime
            let detectionEnd = detection.endTime

            debugLog(
                "  Detection: '\(vocabTerm)' [\(String(format: "%.2f", detectionStart))-\(String(format: "%.2f", detectionEnd))s] score=\(String(format: "%.2f", vocabScore))"
            )

            // Find overlapping TDT words
            var overlappingWords: [(index: Int, timing: WordTiming, overlapRatio: Double)] = []

            for (idx, timing) in wordTimings.enumerated() {
                guard !replacedIndices.contains(idx) else { continue }

                // Calculate overlap
                let overlapStart = max(detectionStart, timing.startTime)
                let overlapEnd = min(detectionEnd, timing.endTime)
                let overlapDuration = max(0, overlapEnd - overlapStart)

                if overlapDuration > 0 {
                    let detectionDuration = detectionEnd - detectionStart
                    let overlapRatio = detectionDuration > 0 ? overlapDuration / detectionDuration : 0
                    overlappingWords.append((index: idx, timing: timing, overlapRatio: overlapRatio))
                }
            }

            if !overlappingWords.isEmpty {
                // Found overlapping words - decide whether to replace
                // Sort by overlap ratio (highest first)
                let sorted = overlappingWords.sorted { $0.overlapRatio > $1.overlapRatio }
                let bestMatch = sorted[0]

                // Convert TDT confidence (0-1) to log-probability scale for comparison
                // TDT confidence is softmax probability, CTC score is log-probability
                let tdtLogProb = log(max(bestMatch.timing.confidence, 1e-10))

                let shouldReplace = vocabScore > tdtLogProb

                debugLog(
                    "    Overlap with '\(bestMatch.timing.word)' (conf=\(String(format: "%.2f", bestMatch.timing.confidence)), logP=\(String(format: "%.2f", tdtLogProb)))"
                )
                debugLog(
                    "    CTC score: \(String(format: "%.2f", vocabScore)) vs TDT: \(String(format: "%.2f", tdtLogProb)) -> \(shouldReplace ? "REPLACE" : "KEEP")"
                )

                if shouldReplace {
                    modifiedWords[bestMatch.index].word = vocabTerm
                    replacedIndices.insert(bestMatch.index)

                    replacements.append(
                        RescoringResult(
                            originalWord: bestMatch.timing.word,
                            originalScore: tdtLogProb,
                            replacementWord: vocabTerm,
                            replacementScore: vocabScore,
                            shouldReplace: true,
                            reason:
                                "Timestamp overlap, CTC score \(String(format: "%.2f", vocabScore)) > TDT \(String(format: "%.2f", tdtLogProb))"
                        ))
                }
            } else {
                // No overlapping words - find insertion point (gap detection)
                var insertAfterIndex = -1

                for (idx, timing) in wordTimings.enumerated() {
                    if timing.endTime <= detectionStart {
                        insertAfterIndex = idx
                    }
                }

                // Check if there's actually a gap (not overlapping with next word)
                let nextWordStart: Double
                if insertAfterIndex + 1 < wordTimings.count {
                    nextWordStart = wordTimings[insertAfterIndex + 1].startTime
                } else {
                    nextWordStart = Double.infinity
                }

                let gapExists = detectionEnd <= nextWordStart

                if gapExists {
                    insertions.append((word: vocabTerm, insertAfterIndex: insertAfterIndex, time: detectionStart))
                    debugLog("    No overlap - INSERT after index \(insertAfterIndex) (gap detected)")

                    replacements.append(
                        RescoringResult(
                            originalWord: "",
                            originalScore: 0,
                            replacementWord: vocabTerm,
                            replacementScore: vocabScore,
                            shouldReplace: true,
                            reason: "Inserted into gap at \(String(format: "%.2f", detectionStart))s"
                        ))
                } else {
                    debugLog("    No gap found for insertion (would overlap with existing word)")
                }
            }
        }

        // Build final transcript
        // First, collect all words with their positions
        var finalWords: [(word: String, time: Double)] = modifiedWords.map { ($0.word, $0.startTime) }

        // Add insertions
        for insertion in insertions {
            finalWords.append((word: insertion.word, time: insertion.time))
        }

        // Sort by time
        finalWords.sort { $0.time < $1.time }

        var modifiedText = finalWords.map { $0.word }.joined(separator: " ")

        // HYBRID FALLBACK: If no timestamp-based replacements were made,
        // try string-similarity matching for detections (especially for "Boz" -> "Bose")
        if replacements.isEmpty && !detections.isEmpty {
            debugLog("  No timestamp matches - trying string-similarity fallback")
            modifiedText = applyStringSimilarityFallback(
                text: modifiedText,
                detections: detections,
                cbw: cbw
            )
        }

        let wasModified = modifiedText != transcript

        debugLog("Final: \(modifiedText)")
        debugLog("Modified: \(wasModified)")
        debugLog("===========================================")

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
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

            // Check if this starts a new word:
            // - Has space prefix " " (TDT format)
            // - Has "▁" prefix (SentencePiece format)
            // - Is first token
            let startsNewWord = token.hasPrefix(" ") || token.hasPrefix("▁") || currentWord.isEmpty

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
                // Remove prefix (space or ▁)
                if token.hasPrefix(" ") {
                    currentWord = String(token.dropFirst())
                } else if token.hasPrefix("▁") {
                    currentWord = String(token.dropFirst())
                } else {
                    currentWord = token
                }
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

    /// String-similarity fallback for when timestamp-based matching fails.
    /// Replaces words that are phonetically similar to detected vocabulary terms.
    private func applyStringSimilarityFallback(
        text: String,
        detections: [CtcKeywordSpotter.KeywordDetection],
        cbw: Float
    ) -> String {
        var modifiedText = text
        let words = text.split(whereSeparator: { $0.isWhitespace }).map { String($0) }

        for detection in detections {
            let vocabTerm = detection.term.text
            let vocabTermLower = detection.term.textLowercased

            // Find all words similar to the vocabulary term
            for word in words {
                let wordClean = word.trimmingCharacters(in: .punctuationCharacters)
                let similarity = Self.stringSimilarity(wordClean, vocabTerm)

                // Threshold for fallback - replace phonetically similar words
                // "Boz" vs "Bose" = 0.50, need to catch these cases
                // Require: same first letter AND similar length to avoid false positives
                let wordCleanLower = wordClean.lowercased()
                let sameFirstLetter = wordCleanLower.first == vocabTermLower.first
                let lengthDiff = abs(wordClean.count - vocabTerm.count)
                let lengthMatch = lengthDiff <= 1
                let shouldReplace =
                    similarity >= 0.50 && sameFirstLetter && lengthMatch
                    && wordCleanLower != vocabTermLower
                if shouldReplace {
                    // Replace this word with the vocabulary term
                    let replacement = preserveCapitalization(original: wordClean, replacement: vocabTerm)

                    // Use word boundary regex to avoid partial replacements
                    let pattern = "\\b\(NSRegularExpression.escapedPattern(for: wordClean))\\b"
                    if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
                        modifiedText = regex.stringByReplacingMatches(
                            in: modifiedText,
                            options: [],
                            range: NSRange(modifiedText.startIndex..., in: modifiedText),
                            withTemplate: replacement
                        )
                    }

                    debugLog(
                        "    [FALLBACK] '\(wordClean)' -> '\(replacement)' (sim=\(String(format: "%.2f", similarity)))"
                    )
                }
            }
        }

        return modifiedText
    }

    // MARK: - Private Helpers

    /// Candidate vocabulary term match with span information
    struct CandidateMatch {
        let term: CustomVocabularyTerm
        let similarity: Float
        let spanLength: Int  // Number of TDT words matched (1 for single, 2+ for compound)
        let matchedText: String  // The normalized text that matched
    }

    /// Find candidate vocabulary terms for a TDT word, including compound word detection.
    ///
    /// This method queries the BK-tree (or performs linear scan) for:
    /// 1. Single word matches
    /// 2. Two-word compound matches (word + next word concatenated)
    /// 3. Three-word compound matches (for longer vocabulary terms)
    ///
    /// - Parameters:
    ///   - normalizedWord: The normalized TDT word
    ///   - adjacentNormalized: Array of normalized adjacent words (for compound detection)
    ///   - minSimilarity: Minimum similarity threshold
    /// - Returns: Array of candidate matches sorted by similarity (descending)
    func findCandidateTermsForWord(
        normalizedWord: String,
        adjacentNormalized: [String],
        minSimilarity: Float
    ) -> [CandidateMatch] {
        guard !normalizedWord.isEmpty else { return [] }

        var candidates: [CandidateMatch] = []

        if useBKTree, let tree = bkTree {
            // BK-tree path: O(log V) per query

            // 1. Single word query
            let maxLen1 = max(normalizedWord.count, 3)
            let maxDist1 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen1)))
            let results1 = tree.search(query: normalizedWord, maxDistance: maxDist1)

            for result in results1 {
                let similarity = Self.stringSimilarity(normalizedWord, result.normalizedText)
                if similarity >= minSimilarity {
                    candidates.append(
                        CandidateMatch(
                            term: result.term,
                            similarity: similarity,
                            spanLength: 1,
                            matchedText: normalizedWord
                        ))
                }
            }

            // 2. Two-word compound query (e.g., "new" + "res" -> "newres" matches "Newrez")
            if !adjacentNormalized.isEmpty, let word2 = adjacentNormalized.first, !word2.isEmpty {
                let compound2 = normalizedWord + word2
                let maxLen2 = max(compound2.count, 3)
                let maxDist2 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen2)))
                let results2 = tree.search(query: compound2, maxDistance: maxDist2)

                for result in results2 {
                    let similarity = Self.stringSimilarity(compound2, result.normalizedText)
                    if similarity >= minSimilarity {
                        candidates.append(
                            CandidateMatch(
                                term: result.term,
                                similarity: similarity,
                                spanLength: 2,
                                matchedText: compound2
                            ))
                    }
                }
            }

            // 3. Three-word compound query (for longer terms like "livmarli" from "liv" + "mar" + "li")
            if adjacentNormalized.count >= 2,
                let word2 = adjacentNormalized.first, !word2.isEmpty,
                let word3 = adjacentNormalized.dropFirst().first, !word3.isEmpty
            {
                let compound3 = normalizedWord + word2 + word3
                // Only search for 3-word compounds if the compound is long enough
                if compound3.count >= 6 {
                    let maxLen3 = compound3.count
                    let maxDist3 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen3)))
                    let results3 = tree.search(query: compound3, maxDistance: maxDist3)

                    for result in results3 {
                        let similarity = Self.stringSimilarity(compound3, result.normalizedText)
                        if similarity >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: result.term,
                                    similarity: similarity,
                                    spanLength: 3,
                                    matchedText: compound3
                                ))
                        }
                    }
                }
            }

            // 4. Multi-word phrase query (e.g., "bank of america" as space-separated phrase)
            // This handles multi-word vocabulary terms
            // Guard: only attempt if we have adjacent words (need at least 1 for a 2-word phrase)
            if !adjacentNormalized.isEmpty {
                for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                    let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                    let phrase = phraseWords.joined(separator: " ")
                    let maxLenPhrase = max(phrase.count, 3)
                    let maxDistPhrase = min(
                        bkTreeMaxDistance + 1, Int((1.0 - minSimilarity) * Float(maxLenPhrase)))
                    let resultsPhrase = tree.search(query: phrase, maxDistance: maxDistPhrase)

                    for result in resultsPhrase {
                        let similarity = Self.stringSimilarity(phrase, result.normalizedText)
                        if similarity >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: result.term,
                                    similarity: similarity,
                                    spanLength: spanLen,
                                    matchedText: phrase
                                ))
                        }
                    }
                }
            }

        } else {
            // Linear scan fallback: O(V) per word
            for term in vocabulary.terms {
                let termNormalized = Self.normalizeForSimilarity(term.text)
                guard !termNormalized.isEmpty else { continue }

                let termWordCount = termNormalized.split(separator: " ").count

                if termWordCount == 1 {
                    // Single word term - check single word and compounds
                    let similarity1 = Self.stringSimilarity(normalizedWord, termNormalized)
                    if similarity1 >= minSimilarity {
                        candidates.append(
                            CandidateMatch(
                                term: term,
                                similarity: similarity1,
                                spanLength: 1,
                                matchedText: normalizedWord
                            ))
                    }

                    // Check 2-word compound
                    if !adjacentNormalized.isEmpty, let word2 = adjacentNormalized.first, !word2.isEmpty {
                        let compound2 = normalizedWord + word2
                        let similarity2 = Self.stringSimilarity(compound2, termNormalized)
                        if similarity2 >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: term,
                                    similarity: similarity2,
                                    spanLength: 2,
                                    matchedText: compound2
                                ))
                        }
                    }

                    // Check 3-word compound
                    if adjacentNormalized.count >= 2 {
                        let word2 = adjacentNormalized[0]
                        let word3 = adjacentNormalized[1]
                        if !word2.isEmpty && !word3.isEmpty {
                            let compound3 = normalizedWord + word2 + word3
                            if compound3.count >= 6 {
                                let similarity3 = Self.stringSimilarity(compound3, termNormalized)
                                if similarity3 >= minSimilarity {
                                    candidates.append(
                                        CandidateMatch(
                                            term: term,
                                            similarity: similarity3,
                                            spanLength: 3,
                                            matchedText: compound3
                                        ))
                                }
                            }
                        }
                    }
                } else {
                    // Multi-word term - check phrases
                    // Guard: only attempt if we have adjacent words
                    if !adjacentNormalized.isEmpty {
                        for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                            let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                            let phrase = phraseWords.joined(separator: " ")
                            let similarity = Self.stringSimilarity(phrase, termNormalized)
                            if similarity >= minSimilarity {
                                candidates.append(
                                    CandidateMatch(
                                        term: term,
                                        similarity: similarity,
                                        spanLength: spanLen,
                                        matchedText: phrase
                                    ))
                            }
                        }
                    }
                }
            }
        }

        // Sort by similarity (descending), then by span length (prefer longer matches)
        return candidates.sorted {
            if $0.similarity != $1.similarity {
                return $0.similarity > $1.similarity
            }
            return $0.spanLength > $1.spanLength
        }
    }

    /// Estimate the CTC score for the original word based on detection characteristics
    /// This is a heuristic - ideally we'd run full CTC DP on the original word's tokens
    private func estimateOriginalWordScore(
        detection: CtcKeywordSpotter.KeywordDetection,
        originalWord: String,
        similarity: Float
    ) -> Float {
        // If words are very similar phonetically, the original word likely scores similarly
        // to the vocabulary term. Adjust based on similarity.

        // Start with the vocabulary term's score
        let vocabScore = detection.score

        // Higher similarity = original word likely scores close to vocab term
        // Lower similarity = vocab term might be wrong match, original could score higher

        // Heuristic: If similarity is low, boost original word estimate
        // (because low similarity means acoustic evidence might favor original)
        let similarityPenalty = (1.0 - similarity) * 4.0  // 0-4 point boost for original

        // The original word's estimated score
        let estimatedScore = vocabScore + similarityPenalty

        return estimatedScore
    }

    /// Compute string similarity using Levenshtein distance
    static func stringSimilarity(_ a: String, _ b: String) -> Float {
        let aLower = a.lowercased()
        let bLower = b.lowercased()

        let distance = StringUtils.levenshteinDistance(aLower, bLower)
        let maxLen = max(aLower.count, bLower.count)

        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Represents a normalized form of a vocabulary term (canonical or alias)
    struct NormalizedForm: Hashable {
        let raw: String
        let normalized: String
        let wordCount: Int
    }

    /// Build all normalized forms (canonical + aliases) for a vocabulary term
    func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        var rawForms: [String] = [term.text]
        let termLower = term.textLowercased

        // Look up canonical term in vocabulary to get ALL aliases
        for vocabTerm in vocabulary.terms where vocabTerm.textLowercased == termLower {
            if let aliases = vocabTerm.aliases {
                rawForms.append(contentsOf: aliases)
            }
        }
        // Also add aliases from the term itself
        if let aliases = term.aliases {
            rawForms.append(contentsOf: aliases)
        }

        var seen = Set<String>()
        var forms: [NormalizedForm] = []

        for raw in rawForms {
            let normalized = Self.normalizeForSimilarity(raw)
            guard !normalized.isEmpty else { continue }
            guard !seen.contains(normalized) else { continue }
            seen.insert(normalized)

            let wordCount = normalized.split(separator: " ").count
            forms.append(NormalizedForm(raw: raw, normalized: normalized, wordCount: wordCount))
        }

        return forms
    }

    /// Determine required similarity threshold based on span length and word length
    /// Note: Using permissive thresholds to avoid rejecting valid matches
    func requiredSimilarity(minSimilarity: Float, spanLength: Int, normalizedText: String) -> Float {
        // Multi-word spans: slightly higher threshold to avoid false positives
        if spanLength >= 2 {
            return max(minSimilarity, 0.55)
        }

        // Single words: use the configured minimum similarity
        // Note: The 0.85 threshold for short words was too aggressive (caused regression)
        return minSimilarity
    }

    /// Preserve capitalization from original word in replacement
    func preserveCapitalization(original: String, replacement: String) -> String {
        guard let firstChar = original.first else { return replacement }

        if firstChar.isUppercase && replacement.first?.isLowercase == true {
            return replacement.prefix(1).uppercased() + replacement.dropFirst()
        }
        return replacement
    }

    /// Normalize text for similarity checks: lowercase, collapse whitespace,
    /// and strip punctuation while preserving letters, numbers, apostrophes, and hyphens.
    static func normalizeForSimilarity(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'-"))
        var result = ""
        var lastWasSpace = true

        for scalar in text.lowercased().unicodeScalars {
            if allowed.contains(scalar) {
                result.append(Character(scalar))
                lastWasSpace = false
            } else if scalar == " " || scalar == "\t" || scalar == "\n" {
                if !lastWasSpace && !result.isEmpty {
                    result.append(" ")
                    lastWasSpace = true
                }
            }
            // Skip other characters (punctuation)
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Build set of normalized vocabulary terms for guard checks
    func buildVocabularyNormalizedSet() -> Set<String> {
        var normalizedSet = Set<String>()
        for term in vocabulary.terms {
            let normalized = Self.normalizeForSimilarity(term.text)
            if !normalized.isEmpty {
                normalizedSet.insert(normalized)
            }
            // Also add aliases if present
            if let aliases = term.aliases {
                for alias in aliases {
                    let normalizedAlias = Self.normalizeForSimilarity(alias)
                    if !normalizedAlias.isEmpty {
                        normalizedSet.insert(normalizedAlias)
                    }
                }
            }
        }
        return normalizedSet
    }

}
