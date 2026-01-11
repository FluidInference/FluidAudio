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

    private let logger = Logger(subsystem: "com.fluidaudio", category: "VocabularyRescorer")
    private let spotter: CtcKeywordSpotter
    private let vocabulary: CustomVocabularyContext
    private let trie: VocabularyTrie
    private let ctcTokenizer: SentencePieceCtcTokenizer?
    private let debugMode: Bool

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

        public static let `default` = Config(
            minScoreAdvantage: 2.0,  // Vocab term must score 2.0 better than original
            minVocabScore: -12.0,  // Vocab term must have reasonable CTC score (lowered for alias support)
            maxOriginalScoreForReplacement: -4.0,  // Don't replace very confident words
            vocabBoostWeight: 1.5  // Boost for vocabulary terms
        )

        public init(
            minScoreAdvantage: Float = 2.0,
            minVocabScore: Float = -8.0,
            maxOriginalScoreForReplacement: Float = -4.0,
            vocabBoostWeight: Float = 1.5
        ) {
            self.minScoreAdvantage = minScoreAdvantage
            self.minVocabScore = minVocabScore
            self.maxOriginalScoreForReplacement = maxOriginalScoreForReplacement
            self.vocabBoostWeight = vocabBoostWeight
        }
    }

    private let config: Config

    /// Initialize rescorer with CTC spotter and vocabulary
    public init(
        spotter: CtcKeywordSpotter,
        vocabulary: CustomVocabularyContext,
        config: Config = .default
    ) {
        self.spotter = spotter
        self.vocabulary = vocabulary
        self.trie = VocabularyTrie(vocabulary: vocabulary)
        self.config = config
        self.debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        // Initialize CTC tokenizer for scoring original words
        do {
            self.ctcTokenizer = try SentencePieceCtcTokenizer()
        } catch {
            self.ctcTokenizer = nil
            let logger = Logger(subsystem: "com.fluidaudio", category: "VocabularyRescorer")
            logger.warning("Failed to initialize CTC tokenizer: \(error). Will fall back to heuristic scoring.")
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

            // Skip if vocab term doesn't meet minimum score
            guard vocabScore >= config.minVocabScore else {
                if debugMode {
                    logger.debug(
                        "Skipping '\(vocabTerm)': CTC score \(String(format: "%.2f", vocabScore)) < min \(String(format: "%.2f", config.minVocabScore))"
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
            let vocabTermLower = vocabTerm.lowercased()
            for term in vocabulary.terms {
                if term.text.lowercased() == vocabTermLower {
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
                if debugMode && bestSimilarity >= 0.50 {
                    let wordClean = words[idx].trimmingCharacters(in: .punctuationCharacters)
                    print("    [SIM] '\(wordClean)' vs '\(vocabTerm)' = \(String(format: "%.2f", bestSimilarity))")
                }

                if bestSimilarity >= effectiveMinSimilarity {
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

            if debugMode {
                print(
                    "  [CANDIDATE] '\(candidate.originalWord)' -> '\(vocabTerm)' (sim=\(String(format: "%.2f", candidate.similarity)), isHighConfAlias=\(candidate.isHighConfidenceAlias), span=\(candidate.spanLength))"
                )
            }

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

            if debugMode {
                print(
                    """
                      Vocab CTC: \(String(format: "%.2f", vocabCtcScore)), Original (\(scoringMethod)): \(String(format: "%.2f", originalScore)), Advantage: \(String(format: "%.2f", scoreAdvantage))
                    """)
            }

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

            } else if scoreAdvantage >= config.minScoreAdvantage
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
                    reason =
                        "Vocab score advantage: \(String(format: "%.2f", scoreAdvantage)), sim=\(String(format: "%.2f", candidate.similarity))"

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
                reason = "Score advantage too low: \(String(format: "%.2f", scoreAdvantage))"
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

            if debugMode {
                let action = shouldReplace ? "REPLACE" : "KEEP"
                print("  [\(action)] '\(candidate.originalWord)' -> '\(vocabTerm)': \(reason)")
            }
        }

        // Remove empty words (cleared spans)
        let finalWords = modifiedWords.filter { !$0.isEmpty }
        let modifiedText = finalWords.joined(separator: " ")
        let wasModified = modifiedText != transcript

        if debugMode {
            logger.info("Final: \(modifiedText)")
            logger.info("Modified: \(wasModified)")
            logger.info("===========================")
        }

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

        if debugMode {
            print("=== VocabularyRescorer (Timestamp-Based) ===")
            print("Words: \(wordTimings.count), Detections: \(detections.count)")
            print("CBW (context-biasing weight): \(cbw)")
        }

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

            if debugMode {
                print(
                    "  Detection: '\(vocabTerm)' [\(String(format: "%.2f", detectionStart))-\(String(format: "%.2f", detectionEnd))s] score=\(String(format: "%.2f", vocabScore))"
                )
            }

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

                if debugMode {
                    print(
                        "    Overlap with '\(bestMatch.timing.word)' (conf=\(String(format: "%.2f", bestMatch.timing.confidence)), logP=\(String(format: "%.2f", tdtLogProb)))"
                    )
                    print(
                        "    CTC score: \(String(format: "%.2f", vocabScore)) vs TDT: \(String(format: "%.2f", tdtLogProb)) -> \(shouldReplace ? "REPLACE" : "KEEP")"
                    )
                }

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

                    if debugMode {
                        print("    No overlap - INSERT after index \(insertAfterIndex) (gap detected)")
                    }

                    replacements.append(
                        RescoringResult(
                            originalWord: "",
                            originalScore: 0,
                            replacementWord: vocabTerm,
                            replacementScore: vocabScore,
                            shouldReplace: true,
                            reason: "Inserted into gap at \(String(format: "%.2f", detectionStart))s"
                        ))
                } else if debugMode {
                    print("    No gap found for insertion (would overlap with existing word)")
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
            if debugMode {
                print("  No timestamp matches - trying string-similarity fallback")
            }
            modifiedText = applyStringSimilarityFallback(
                text: modifiedText,
                detections: detections,
                cbw: cbw
            )
        }

        let wasModified = modifiedText != transcript

        if debugMode {
            print("Final: \(modifiedText)")
            print("Modified: \(wasModified)")
            print("===========================================")
        }

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
    }

    /// Build word-level timings from token timings.
    /// Tokens starting with space " " or "▁" (SentencePiece) begin new words.
    private func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
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
            let vocabTermLower = vocabTerm.lowercased()

            // Find all words similar to the vocabulary term
            for word in words {
                let wordClean = word.trimmingCharacters(in: .punctuationCharacters)
                let similarity = Self.stringSimilarity(wordClean, vocabTerm)

                // Threshold for fallback - replace phonetically similar words
                // "Boz" vs "Bose" = 0.50, need to catch these cases
                // Require: same first letter AND similar length to avoid false positives
                let sameFirstLetter = wordClean.lowercased().first == vocabTermLower.first
                let lengthDiff = abs(wordClean.count - vocabTerm.count)
                let lengthMatch = lengthDiff <= 1
                let shouldReplace =
                    similarity >= 0.50 && sameFirstLetter && lengthMatch
                    && wordClean.lowercased() != vocabTermLower
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

                    if debugMode {
                        print(
                            "    [FALLBACK] '\(wordClean)' -> '\(replacement)' (sim=\(String(format: "%.2f", similarity)))"
                        )
                    }
                }
            }
        }

        return modifiedText
    }

    // MARK: - Private Helpers

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
    private static func stringSimilarity(_ a: String, _ b: String) -> Float {
        let aLower = a.lowercased()
        let bLower = b.lowercased()

        let distance = levenshteinDistance(aLower, bLower)
        let maxLen = max(aLower.count, bLower.count)

        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Represents a normalized form of a vocabulary term (canonical or alias)
    private struct NormalizedForm: Hashable {
        let raw: String
        let normalized: String
        let wordCount: Int
    }

    /// Build all normalized forms (canonical + aliases) for a vocabulary term
    private func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        var rawForms: [String] = [term.text]
        let termLower = term.text.lowercased()

        // Look up canonical term in vocabulary to get ALL aliases
        for vocabTerm in vocabulary.terms where vocabTerm.text.lowercased() == termLower {
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
    private func requiredSimilarity(minSimilarity: Float, spanLength: Int, normalizedText: String) -> Float {
        // Multi-word spans: slightly higher threshold to avoid false positives
        if spanLength >= 2 {
            return max(minSimilarity, 0.55)
        }

        // Single words: use the configured minimum similarity
        // Note: The 0.85 threshold for short words was too aggressive (caused regression)
        return minSimilarity
    }

    /// Levenshtein distance between two strings
    private static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let m = aChars.count
        let n = bChars.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                let cost = aChars[i - 1] == bChars[j - 1] ? 0 : 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            }
        }

        return dp[m][n]
    }

    /// Preserve capitalization from original word in replacement
    private func preserveCapitalization(original: String, replacement: String) -> String {
        guard let firstChar = original.first else { return replacement }

        if firstChar.isUppercase && replacement.first?.isLowercase == true {
            return replacement.prefix(1).uppercased() + replacement.dropFirst()
        }
        return replacement
    }

    /// Normalize text for similarity checks: lowercase, collapse whitespace,
    /// and strip punctuation while preserving letters, numbers, apostrophes, and hyphens.
    private static func normalizeForSimilarity(_ text: String) -> String {
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
    private func buildVocabularyNormalizedSet() -> Set<String> {
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

    // MARK: - Constrained CTC Rescoring

    /// Rescore using constrained CTC search around TDT word locations.
    ///
    /// This approach fixes the timing mismatch issue where global CTC search finds
    /// vocabulary terms at completely different locations than TDT transcription.
    ///
    /// Algorithm:
    /// 1. Find TDT words phonetically similar to vocabulary terms (string similarity)
    /// 2. For each match, run constrained CTC DP within the TDT word's timestamp window
    /// 3. Compare constrained CTC score with TDT confidence to decide replacement
    ///
    /// - Parameters:
    ///   - transcript: Original transcript from TDT decoder
    ///   - tokenTimings: Token-level timings from TDT decoder
    ///   - logProbs: CTC log-probabilities from spotter
    ///   - frameDuration: Duration of each CTC frame in seconds
    ///   - cbw: Context-biasing weight (default 3.0 per NeMo paper)
    ///   - marginSeconds: Temporal margin around TDT word for CTC search (default 0.5s)
    ///   - minSimilarity: Minimum string similarity to consider a match (default 0.5)
    /// - Returns: Rescored transcript with constrained CTC replacements
    public func rescoreWithConstrainedCTC(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = 3.0,
        marginSeconds: Double = 0.5,
        minSimilarity: Float = 0.5
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        if debugMode {
            print("=== VocabularyRescorer (Constrained CTC) ===")
            print("Words: \(wordTimings.count), Frames: \(logProbs.count)")
            print("Frame duration: \(String(format: "%.4f", frameDuration))s")
            print("CBW: \(cbw), Margin: \(marginSeconds)s, MinSimilarity: \(minSimilarity)")
        }

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var replacedIndices = Set<Int>()

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // For each vocabulary term, find similar TDT words and run constrained CTC
        for term in vocabulary.terms {
            let vocabTerm = term.text

            // Skip short vocabulary terms (per NeMo CTC-WS paper)
            guard vocabTerm.count >= vocabulary.minTermLength else {
                if debugMode {
                    print(
                        "  Skipping '\(vocabTerm)': too short (\(vocabTerm.count) < \(vocabulary.minTermLength) chars)")
                }
                continue
            }

            let vocabTokens = term.ctcTokenIds ?? term.tokenIds

            guard let vocabTokens, !vocabTokens.isEmpty else {
                continue
            }

            // Build all normalized forms (canonical + aliases) for this term
            let normalizedForms = buildNormalizedForms(for: term)
            guard !normalizedForms.isEmpty else { continue }

            let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
            let normalizedCurrentSet = Set(normalizedForms.map { $0.normalized })

            // Split forms by word count for appropriate matching
            let multiWordForms = normalizedForms.filter { $0.wordCount > 1 }
            let singleWordForms = normalizedForms.filter { $0.wordCount == 1 }

            if !multiWordForms.isEmpty {
                // Multi-word phrase matching: look for consecutive TDT words that match the phrase
                let maxWordCount = multiWordForms.map { $0.wordCount }.max() ?? 0
                let minWordCount = multiWordForms.map { $0.wordCount }.min() ?? 0
                let maxSpan = min(4, maxWordCount + 1)  // Allow some flexibility
                let minSpan = max(2, minWordCount)

                guard minSpan <= maxSpan else { continue }

                for spanLength in minSpan...maxSpan {
                    for startIdx in 0..<(wordTimings.count - spanLength + 1) {
                        // Check if any word in the span is already replaced
                        let spanIndices = Array(startIdx..<(startIdx + spanLength))
                        guard spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else { continue }

                        // Build concatenated phrase from consecutive TDT words
                        let spanWords = spanIndices.map { wordTimings[$0].word }
                        let tdtPhrase = spanWords.joined(separator: " ")
                        let normalizedPhrase = Self.normalizeForSimilarity(tdtPhrase)
                        guard !normalizedPhrase.isEmpty else { continue }

                        // Check similarity against ALL forms (canonical + aliases)
                        var bestSimilarity: Float = 0
                        for form in multiWordForms {
                            let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                            bestSimilarity = max(bestSimilarity, similarity)
                        }

                        // Skip if already exact match to canonical (no replacement needed)
                        if normalizedPhrase == normalizedCanonical {
                            continue
                        }

                        // Guard: Skip if original phrase matches a DIFFERENT vocabulary term
                        if vocabularyNormalizedSet.contains(normalizedPhrase)
                            && !normalizedCurrentSet.contains(normalizedPhrase)
                        {
                            if debugMode {
                                print(
                                    "  [MULTI] Skipping '\(vocabTerm)': phrase '\(tdtPhrase)' matches another vocab term"
                                )
                            }
                            continue
                        }

                        // Use adaptive similarity threshold
                        let minSimilarityForSpan = requiredSimilarity(
                            minSimilarity: minSimilarity,
                            spanLength: spanLength,
                            normalizedText: normalizedPhrase
                        )
                        guard bestSimilarity >= minSimilarityForSpan else { continue }

                        // Get temporal window for the entire span
                        let spanStartTime = wordTimings[spanIndices.first!].startTime
                        let spanEndTime = wordTimings[spanIndices.last!].endTime

                        let marginFrames = Int(marginSeconds / frameDuration)
                        let spanStartFrame = Int(spanStartTime / frameDuration)
                        let spanEndFrame = Int(spanEndTime / frameDuration)

                        let searchStart = max(0, spanStartFrame - marginFrames)
                        let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

                        // Score vocabulary phrase using constrained CTC
                        let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
                            logProbs: logProbs,
                            keywordTokens: vocabTokens,
                            searchStartFrame: searchStart,
                            searchEndFrame: searchEnd
                        )

                        // Score original TDT phrase using constrained CTC
                        var originalCtcScore: Float = -Float.infinity
                        if let tokenizer = ctcTokenizer {
                            let originalTokens = tokenizer.encode(tdtPhrase)
                            if !originalTokens.isEmpty {
                                let (score, _, _) = spotter.ctcWordSpotConstrained(
                                    logProbs: logProbs,
                                    keywordTokens: originalTokens,
                                    searchStartFrame: searchStart,
                                    searchEndFrame: searchEnd
                                )
                                originalCtcScore = score
                            }
                        }

                        let boostedVocabScore = vocabCtcScore + cbw
                        let shouldReplace = boostedVocabScore > originalCtcScore

                        if debugMode {
                            print(
                                "  [MULTI] '\(tdtPhrase)' vs '\(vocabTerm)' (sim=\(String(format: "%.2f", bestSimilarity)))"
                            )
                            print(
                                "    TDT span: [\(String(format: "%.2f", spanStartTime))-\(String(format: "%.2f", spanEndTime))s] words=\(spanLength)"
                            )
                            print(
                                "    CTC('\(tdtPhrase)'): \(String(format: "%.2f", originalCtcScore))"
                            )
                            print(
                                "    CTC('\(vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbw) = \(String(format: "%.2f", boostedVocabScore))"
                            )
                            print(
                                "    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(boostedVocabScore > originalCtcScore ? ">" : "<=") original)"
                            )
                        }

                        if shouldReplace {
                            // Replace first word with full phrase, mark rest as empty
                            let replacement = preserveCapitalization(
                                original: spanWords.first ?? tdtPhrase,
                                replacement: vocabTerm
                            )
                            modifiedWords[spanIndices.first!].word = replacement
                            for idx in spanIndices.dropFirst() {
                                modifiedWords[idx].word = ""  // Will be filtered out
                            }
                            // Mark all indices as replaced
                            for idx in spanIndices {
                                replacedIndices.insert(idx)
                            }

                            replacements.append(
                                RescoringResult(
                                    originalWord: tdtPhrase,
                                    originalScore: originalCtcScore,
                                    replacementWord: replacement,
                                    replacementScore: boostedVocabScore,
                                    shouldReplace: true,
                                    reason:
                                        "CTC-vs-CTC (multi-word): '\(vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) > '\(tdtPhrase)'=\(String(format: "%.2f", originalCtcScore))"
                                ))
                        }
                    }
                }
            }

            if !singleWordForms.isEmpty {
                // Single-word matching (includes compound word detection)
                for (wordIdx, timing) in wordTimings.enumerated() {
                    guard !replacedIndices.contains(wordIdx) else { continue }

                    let tdtWord = timing.word
                    let normalizedWord = Self.normalizeForSimilarity(tdtWord)
                    guard !normalizedWord.isEmpty else { continue }

                    // Skip if already exact match to canonical (no replacement needed)
                    if normalizedWord == normalizedCanonical {
                        continue
                    }

                    // Guard: Skip if original word matches a DIFFERENT vocabulary term
                    if vocabularyNormalizedSet.contains(normalizedWord)
                        && !normalizedCurrentSet.contains(normalizedWord)
                    {
                        if debugMode {
                            print("  Skipping '\(vocabTerm)': word '\(tdtWord)' matches another vocab term")
                        }
                        continue
                    }

                    // Check similarity against ALL forms (single word)
                    var bestSimilarity: Float = 0
                    var matchedSpanLength = 1
                    var matchedConcatenation = normalizedWord
                    for form in singleWordForms {
                        let similarity = Self.stringSimilarity(normalizedWord, form.normalized)
                        bestSimilarity = max(bestSimilarity, similarity)
                    }

                    // COMPOUND WORD MATCHING: For single-word vocabulary terms, also try
                    // matching against concatenated adjacent TDT words.
                    // This handles cases like "Livmarli" being transcribed as "Liv Mali".
                    // Minimum vocab length of 4 for 2-word matching to avoid false positives on short words.
                    let minLengthFor2Word = 4
                    let minLengthFor3Word = 8

                    // Pre-compute normalized adjacent words (only if needed)
                    let normalized2: String? =
                        (wordIdx + 1 < wordTimings.count && !replacedIndices.contains(wordIdx + 1))
                        ? Self.normalizeForSimilarity(wordTimings[wordIdx + 1].word)
                        : nil
                    let normalized3: String? =
                        (wordIdx + 2 < wordTimings.count && !replacedIndices.contains(wordIdx + 2))
                        ? Self.normalizeForSimilarity(wordTimings[wordIdx + 2].word)
                        : nil

                    // 2-word compound matching
                    if let norm2 = normalized2, !norm2.isEmpty, vocabTerm.count >= minLengthFor2Word {
                        let concatenated = normalizedWord + norm2  // No space
                        for form in singleWordForms {
                            let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                            if concatSimilarity > bestSimilarity {
                                bestSimilarity = concatSimilarity
                                matchedSpanLength = 2
                                matchedConcatenation = concatenated
                            }
                        }
                    }

                    // 3-word compound matching (for longer vocabulary terms only)
                    if let norm2 = normalized2, let norm3 = normalized3,
                        !norm2.isEmpty, !norm3.isEmpty, vocabTerm.count >= minLengthFor3Word
                    {
                        let concatenated = normalizedWord + norm2 + norm3
                        for form in singleWordForms {
                            let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                            if concatSimilarity > bestSimilarity {
                                bestSimilarity = concatSimilarity
                                matchedSpanLength = 3
                                matchedConcatenation = concatenated
                            }
                        }
                    }

                    // Use adaptive similarity threshold (use concatenated text for multi-word spans)
                    let minSimilarityForSpan = requiredSimilarity(
                        minSimilarity: minSimilarity,
                        spanLength: matchedSpanLength,
                        normalizedText: matchedConcatenation
                    )
                    guard bestSimilarity >= minSimilarityForSpan else { continue }

                    // Build the original phrase (single word or concatenated span)
                    let spanIndices = Array(wordIdx..<(wordIdx + matchedSpanLength))
                    let originalPhrase =
                        matchedSpanLength == 1
                        ? tdtWord
                        : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")

                    // Get temporal window for the span (use direct indexing to avoid force unwraps)
                    let spanStartTime = wordTimings[wordIdx].startTime
                    let spanEndTime = wordTimings[wordIdx + matchedSpanLength - 1].endTime

                    let marginFrames = Int(marginSeconds / frameDuration)
                    let spanStartFrame = Int(spanStartTime / frameDuration)
                    let spanEndFrame = Int(spanEndTime / frameDuration)

                    let searchStart = max(0, spanStartFrame - marginFrames)
                    let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

                    // Score vocabulary term using constrained CTC
                    let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
                        logProbs: logProbs,
                        keywordTokens: vocabTokens,
                        searchStartFrame: searchStart,
                        searchEndFrame: searchEnd
                    )

                    // Score original phrase using constrained CTC (same window)
                    var originalCtcScore: Float = -Float.infinity
                    if let tokenizer = ctcTokenizer {
                        let originalTokens = tokenizer.encode(originalPhrase)
                        if !originalTokens.isEmpty {
                            let (score, _, _) = spotter.ctcWordSpotConstrained(
                                logProbs: logProbs,
                                keywordTokens: originalTokens,
                                searchStartFrame: searchStart,
                                searchEndFrame: searchEnd
                            )
                            originalCtcScore = score
                        }
                    }

                    // Apply context-biasing weight to vocabulary term (per NeMo paper)
                    let boostedVocabScore = vocabCtcScore + cbw

                    // CTC-vs-CTC comparison (same scale, per NeMo paper)
                    let shouldReplace = boostedVocabScore > originalCtcScore

                    if debugMode {
                        print(
                            "  '\(originalPhrase)' vs '\(vocabTerm)' (sim=\(String(format: "%.2f", bestSimilarity)), span=\(matchedSpanLength))"
                        )
                        print(
                            "    TDT span: [\(String(format: "%.2f", spanStartTime))-\(String(format: "%.2f", spanEndTime))s]"
                        )
                        print(
                            "    CTC('\(originalPhrase)'): \(String(format: "%.2f", originalCtcScore))"
                        )
                        print(
                            "    CTC('\(vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbw) = \(String(format: "%.2f", boostedVocabScore))"
                        )
                        print(
                            "    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(boostedVocabScore > originalCtcScore ? ">" : "<=") original)"
                        )
                    }

                    if shouldReplace {
                        let replacement = preserveCapitalization(original: tdtWord, replacement: vocabTerm)
                        modifiedWords[wordIdx].word = replacement
                        // Mark additional words in span as empty (will be filtered out)
                        for idx in spanIndices.dropFirst() {
                            modifiedWords[idx].word = ""
                        }
                        // Mark all indices as replaced
                        for idx in spanIndices {
                            replacedIndices.insert(idx)
                        }

                        replacements.append(
                            RescoringResult(
                                originalWord: originalPhrase,
                                originalScore: originalCtcScore,
                                replacementWord: replacement,
                                replacementScore: boostedVocabScore,
                                shouldReplace: true,
                                reason:
                                    "CTC-vs-CTC: '\(vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) > '\(originalPhrase)'=\(String(format: "%.2f", originalCtcScore))"
                            ))
                    }
                }
            }
        }

        // Reconstruct transcript from modified words (filter empty strings from multi-word replacements)
        let modifiedText = modifiedWords.map { $0.word }.filter { !$0.isEmpty }.joined(separator: " ")
        let wasModified = !replacements.isEmpty

        if debugMode {
            print("Final: \(modifiedText)")
            print("Replacements: \(replacements.count)")
            print("===========================================")
        }

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
    }
}
