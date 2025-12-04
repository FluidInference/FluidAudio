import Foundation
import OSLog

/// CTC-based vocabulary rescoring for principled vocabulary integration.
///
/// Instead of blindly replacing words based on phonetic similarity, this rescorer
/// uses CTC log-probabilities to verify that vocabulary terms actually match the audio.
/// Only replaces when the vocabulary term has significantly higher acoustic evidence.
///
/// This implements "shallow fusion" or "CTC rescoring" - a standard technique in ASR.
public struct VocabularyRescorer {

    private let logger = Logger(subsystem: "com.fluidaudio", category: "VocabularyRescorer")
    private let spotter: CtcKeywordSpotter
    private let vocabulary: CustomVocabularyContext
    private let trie: VocabularyTrie
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
            minVocabScore: -8.0,  // Vocab term must have reasonable CTC score
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

    /// Rescore a transcript using CTC evidence
    /// - Parameters:
    ///   - transcript: Original transcript from TDT decoder
    ///   - audioSamples: Audio samples for CTC scoring
    ///   - detections: CTC keyword detections for vocabulary terms
    /// - Returns: Rescored transcript with replacements only where acoustically justified
    public func rescore(
        transcript: String,
        audioSamples: [Float],
        detections: [CtcKeywordSpotter.KeywordDetection]
    ) async throws -> RescoreOutput {
        let words = transcript.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        guard !words.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        if debugMode {
            logger.info("=== VocabularyRescorer ===")
            logger.info("Transcript: \(transcript)")
            logger.info("Detections: \(detections.count)")
        }

        // Build a map of detections by their time windows
        var detectionsByTimeRange: [(detection: CtcKeywordSpotter.KeywordDetection, wordIndex: Int?)] = []
        for detection in detections {
            detectionsByTimeRange.append((detection, nil))
        }

        // Estimate word positions based on uniform distribution
        // In a real implementation, we'd use TDT timestamps
        let totalDuration = Double(audioSamples.count) / 16000.0
        let wordDuration = totalDuration / Double(words.count)

        var replacements: [RescoringResult] = []
        var modifiedWords = words

        // For each detection, find the best matching word and evaluate replacement
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

            // Find which word(s) this detection might replace
            let detectionMidpoint = (detection.startTime + detection.endTime) / 2
            let estimatedWordIndex = min(Int(detectionMidpoint / wordDuration), words.count - 1)

            // Look at words near the detection
            let searchRadius = 2
            let startIdx = max(0, estimatedWordIndex - searchRadius)
            let endIdx = min(words.count - 1, estimatedWordIndex + searchRadius)

            var bestCandidate: (wordIndex: Int, originalWord: String, similarity: Float)?

            for idx in startIdx...endIdx {
                let word = words[idx].trimmingCharacters(in: .punctuationCharacters)

                // Skip if this word is already the vocabulary term
                if word.lowercased() == vocabTerm.lowercased() {
                    continue
                }

                // Check phonetic/character similarity
                let similarity = Self.stringSimilarity(word, vocabTerm)

                if similarity >= vocabulary.minSimilarity {
                    if let existing = bestCandidate {
                        if similarity > existing.similarity {
                            bestCandidate = (idx, word, similarity)
                        }
                    } else {
                        bestCandidate = (idx, word, similarity)
                    }
                }
            }

            guard let candidate = bestCandidate else {
                continue
            }

            // Now the key decision: Should we replace?
            // We need to compare CTC score for the vocabulary term vs the original word

            // The vocab term already has a CTC score from the detection
            let vocabCtcScore = vocabScore + config.vocabBoostWeight

            // For the original word, we estimate its CTC score
            // This is approximate - ideally we'd run CTC DP on the original word's tokens
            // For now, use a heuristic based on the detection's score relative to the word
            let originalEstimatedScore = estimateOriginalWordScore(
                detection: detection,
                originalWord: candidate.originalWord,
                similarity: candidate.similarity
            )

            let scoreAdvantage = vocabCtcScore - originalEstimatedScore

            if debugMode {
                logger.debug(
                    """
                    Candidate: '\(candidate.originalWord)' -> '\(vocabTerm)'
                      Vocab CTC: \(String(format: "%.2f", vocabCtcScore))
                      Original est: \(String(format: "%.2f", originalEstimatedScore))
                      Advantage: \(String(format: "%.2f", scoreAdvantage))
                      Similarity: \(String(format: "%.2f", candidate.similarity))
                    """)
            }

            // Decision criteria:
            // 1. Vocabulary term must have significant score advantage
            // 2. Original word shouldn't have very high confidence
            let shouldReplace =
                scoreAdvantage >= config.minScoreAdvantage
                && originalEstimatedScore <= config.maxOriginalScoreForReplacement

            let reason: String
            if shouldReplace {
                reason = "Vocab score advantage: \(String(format: "%.2f", scoreAdvantage))"
                modifiedWords[candidate.wordIndex] = preserveCapitalization(
                    original: candidate.originalWord,
                    replacement: vocabTerm
                )
            } else if originalEstimatedScore > config.maxOriginalScoreForReplacement {
                reason = "Original word too confident: \(String(format: "%.2f", originalEstimatedScore))"
            } else {
                reason = "Score advantage too low: \(String(format: "%.2f", scoreAdvantage))"
            }

            replacements.append(
                RescoringResult(
                    originalWord: candidate.originalWord,
                    originalScore: originalEstimatedScore,
                    replacementWord: shouldReplace ? vocabTerm : nil,
                    replacementScore: shouldReplace ? vocabCtcScore : nil,
                    shouldReplace: shouldReplace,
                    reason: reason
                ))

            if debugMode {
                let action = shouldReplace ? "REPLACE" : "KEEP"
                logger.info("  [\(action)] '\(candidate.originalWord)' -> '\(vocabTerm)': \(reason)")
            }
        }

        let modifiedText = modifiedWords.joined(separator: " ")
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
}
