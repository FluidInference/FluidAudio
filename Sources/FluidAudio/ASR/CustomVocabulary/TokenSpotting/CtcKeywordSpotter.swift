import CoreML
import Foundation

/// Swift implementation of CTC keyword spotting for Parakeet-TDT CTC 110M,
/// mirroring the NeMo `ctc_word_spot` dynamic programming algorithm.
///
/// This engine:
/// - Runs the MelSpectrogram + AudioEncoder CoreML models from `CtcModels`.
/// - Extracts CTC logits and converts them to log-probabilities over time.
/// - Applies DP to score each keyword independently (no beam search competition).
public struct CtcKeywordSpotter: Sendable {

    let logger = AppLogger(category: "CtcKeywordSpotter")
    let models: CtcModels
    public let blankId: Int

    /// Computed property to avoid storing non-Sendable MLPredictionOptions.
    /// Creating on demand is cheap (just init + empty dict).
    var predictionOptions: MLPredictionOptions {
        AsrModels.optimizedPredictionOptions()
    }

    let sampleRate: Int = ASRConstants.sampleRate
    let maxModelSamples: Int = ASRConstants.maxModelSamples

    // Chunking parameters for audio longer than maxModelSamples
    // 2s overlap at 16kHz = 32,000 samples (matches TDT chunking pattern)
    let chunkOverlapSamples: Int = 32_000

    // Debug flag - enabled only in DEBUG builds
    #if DEBUG
    let debugMode: Bool = true  // Set to true locally for verbose logging
    #else
    let debugMode: Bool = false
    #endif

    // Temperature for CTC softmax (higher = softer distribution, lower = more peaked)
    let temperature: Float = ContextBiasingConstants.ctcTemperature

    // Blank bias applied to log probabilities (positive values penalize blank token)
    let blankBias: Float = ContextBiasingConstants.blankBias

    // MARK: - Result Types

    struct CtcLogProbResult: Sendable {
        let logProbs: [[Float]]
        let frameDuration: Double
        let totalFrames: Int
        let audioSamplesUsed: Int
        let frameTimes: [Double]?
    }

    /// Public result type containing detections and cached CTC log-probabilities.
    /// The log-probs can be reused for scoring additional words without re-running the CTC model.
    public struct SpotKeywordsResult: Sendable {
        /// Keyword detections for vocabulary terms
        public let detections: [KeywordDetection]
        /// CTC log-probabilities [T, V] for reuse in rescoring
        public let logProbs: [[Float]]
        /// Duration of each CTC frame in seconds
        public let frameDuration: Double
        /// Total number of CTC frames
        public let totalFrames: Int
    }

    /// Result for a single keyword detection.
    public struct KeywordDetection: Sendable {
        public let term: CustomVocabularyTerm
        public let score: Float
        public let totalFrames: Int
        public let startFrame: Int
        public let endFrame: Int
        public let startTime: TimeInterval
        public let endTime: TimeInterval

        public init(
            term: CustomVocabularyTerm,
            score: Float,
            totalFrames: Int,
            startFrame: Int,
            endFrame: Int,
            startTime: TimeInterval,
            endTime: TimeInterval
        ) {
            self.term = term
            self.score = score
            self.totalFrames = totalFrames
            self.startFrame = startFrame
            self.endFrame = endFrame
            self.startTime = startTime
            self.endTime = endTime
        }
    }

    public init(models: CtcModels, blankId: Int = ContextBiasingConstants.defaultBlankId) {
        self.models = models
        self.blankId = blankId
        // predictionOptions is now a computed property - no assignment needed
    }

    /// Convenience helper to create a spotter using the default cache location.
    public static func makeDefault(
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) async throws -> CtcKeywordSpotter {
        let models = try await CtcModels.downloadAndLoad()
        return CtcKeywordSpotter(models: models, blankId: blankId)
    }

    // MARK: - Public API

    /// Spot all keywords defined in a `CustomVocabularyContext` that provide `tokenIds`.
    ///
    /// This is Phase 1 support: phrases must be pre-tokenized offline so that
    /// CTC keyword spotting can operate directly on vocabulary IDs.
    public func spotKeywords(
        audioSamples: [Float],
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) async throws -> [KeywordDetection] {
        let ctcResult = try await computeLogProbs(for: audioSamples)
        let logProbs = ctcResult.logProbs
        guard !logProbs.isEmpty else { return [] }

        if debugMode {
            logger.debug("=== CTC Keyword Spotter Debug ===")
            logger.debug("Audio samples: \(audioSamples.count), frames: \(logProbs.count)")
            logger.debug("Vocab size: \(logProbs[0].count), blank ID: \(blankId)")
            logger.debug("Terms to spot: \(customVocabulary.terms.count)")
        }

        // Each CTC frame spans a fixed slice of the original audio.
        // Derive frame duration from the trimmed logProbs and original sample count.
        let frameDuration = ctcResult.frameDuration
        let totalFrames = ctcResult.totalFrames

        var results: [KeywordDetection] = []

        for term in customVocabulary.terms {
            // Prefer CTC-specific token IDs when present; fall back to the shared
            // tokenIds only if ctcTokenIds is not provided. This keeps the RNNT/TDT
            // and CTC vocabularies logically separated.
            let ids = term.ctcTokenIds ?? term.tokenIds
            guard let ids, !ids.isEmpty else {
                if debugMode {
                    logger.debug("  Skipping '\(term.text)': no CTC token IDs")
                }
                continue
            }

            let (score, start, end) = ctcWordSpot(logProbs: logProbs, keywordTokens: ids)

            if debugMode {
                let scoreText = String(format: "%.4f", score)
                let startText = String(format: "%.3f", TimeInterval(start) * frameDuration)
                let endText = String(format: "%.3f", TimeInterval(end) * frameDuration)
                logger.debug(
                    "  '\(term.text)': score=\(scoreText), frames=[\(start), \(end)], time=[\(startText)s, \(endText)s]"
                )
            }

            // Adjust threshold for multi-token phrases (they naturally have lower scores)
            let tokenCount = ids.count
            let adjustedThreshold: Float? = minScore.map { base in
                let extraTokens = max(0, tokenCount - ContextBiasingConstants.baselineTokenCountForThreshold)
                return base - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
            }

            if let threshold = adjustedThreshold, score <= threshold {
                if debugMode {
                    let thresholdText = String(format: "%.4f", threshold)
                    let baseText = minScore.map { String(format: "%.4f", $0) } ?? "nil"
                    logger.debug(
                        "    REJECTED: score \(String(format: "%.4f", score)) <= threshold \(thresholdText) (base: \(baseText), tokens: \(tokenCount))"
                    )
                }
                continue
            }

            let startTime =
                ctcResult.frameTimes.flatMap { start < $0.count ? $0[start] : nil }
                ?? TimeInterval(start) * frameDuration
            let endTime =
                ctcResult.frameTimes.flatMap { end < $0.count ? $0[end] : nil }
                ?? TimeInterval(end) * frameDuration

            let detection = KeywordDetection(
                term: term,
                score: score,
                totalFrames: totalFrames,
                startFrame: start,
                endFrame: end,
                startTime: startTime,
                endTime: endTime
            )
            results.append(detection)

            if debugMode {
                logger.debug("    ACCEPTED: adding detection")
            }
        }

        if debugMode {
            logger.debug("Total detections: \(results.count)")
            logger.debug("=================================")
        }

        return results
    }

    /// Spot keywords and return both detections and cached log-probabilities.
    /// The log-probs can be reused for scoring additional words (e.g., original transcript words)
    /// without re-running the expensive CTC model inference.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono audio samples.
    ///   - customVocabulary: Vocabulary context with pre-tokenized terms.
    ///   - minScore: Optional minimum score threshold for detections.
    /// - Returns: SpotKeywordsResult containing detections and reusable log-probs.
    public func spotKeywordsWithLogProbs(
        audioSamples: [Float],
        customVocabulary: CustomVocabularyContext,
        minScore: Float? = nil
    ) async throws -> SpotKeywordsResult {
        let ctcResult = try await computeLogProbs(for: audioSamples)
        let logProbs = ctcResult.logProbs
        guard !logProbs.isEmpty else {
            return SpotKeywordsResult(detections: [], logProbs: [], frameDuration: 0, totalFrames: 0)
        }

        let frameDuration = ctcResult.frameDuration
        let totalFrames = ctcResult.totalFrames

        var results: [KeywordDetection] = []

        for term in customVocabulary.terms {
            // Skip short terms to reduce false positives (per NeMo CTC-WS paper)
            guard term.text.count >= customVocabulary.minTermLength else {
                if debugMode {
                    logger.debug(
                        "  Skipping '\(term.text)': too short (\(term.text.count) < \(customVocabulary.minTermLength) chars)"
                    )
                }
                continue
            }

            let ids = term.ctcTokenIds ?? term.tokenIds
            guard let ids, !ids.isEmpty else { continue }

            // Adjust threshold for multi-token phrases
            let tokenCount = ids.count
            let adjustedThreshold: Float =
                minScore.map { base in
                    let extraTokens = max(0, tokenCount - ContextBiasingConstants.baselineTokenCountForThreshold)
                    return base - Float(extraTokens) * ContextBiasingConstants.thresholdRelaxationPerToken
                } ?? ContextBiasingConstants.defaultMinSpotterScore

            // Find ALL occurrences of this keyword (not just the best one)
            let multipleDetections = ctcWordSpotMultiple(
                logProbs: logProbs,
                keywordTokens: ids,
                minScore: adjustedThreshold,
                mergeOverlap: true
            )

            for (score, start, end) in multipleDetections {
                let startTime =
                    ctcResult.frameTimes.flatMap { start < $0.count ? $0[start] : nil }
                    ?? TimeInterval(start) * frameDuration
                let endTime =
                    ctcResult.frameTimes.flatMap { end < $0.count ? $0[end] : nil }
                    ?? TimeInterval(end) * frameDuration

                let detection = KeywordDetection(
                    term: term,
                    score: score,
                    totalFrames: totalFrames,
                    startFrame: start,
                    endFrame: end,
                    startTime: startTime,
                    endTime: endTime
                )
                results.append(detection)
            }
        }

        return SpotKeywordsResult(
            detections: results,
            logProbs: logProbs,
            frameDuration: frameDuration,
            totalFrames: totalFrames
        )
    }

    /// Score a single word against cached CTC log-probabilities.
    /// This allows scoring arbitrary words (e.g., original transcript words) without re-running the CTC model.
    ///
    /// - Parameters:
    ///   - logProbs: Cached CTC log-probabilities from spotKeywordsWithLogProbs.
    ///   - keywordTokens: Token IDs for the word to score.
    /// - Returns: Tuple (score, startFrame, endFrame) where score is average log-prob per token.
    public func scoreWord(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        return ctcWordSpot(logProbs: logProbs, keywordTokens: keywordTokens)
    }

    // MARK: - NeMo-compatible DP (delegated to CtcDPAlgorithm)

    func ctcWordSpot(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: keywordTokens)
    }

    func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            searchStartFrame: searchStartFrame,
            searchEndFrame: searchEndFrame
        )
    }

    func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        CtcDPAlgorithm.ctcWordSpotMultiple(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            minScore: minScore,
            mergeOverlap: mergeOverlap
        )
    }

}
