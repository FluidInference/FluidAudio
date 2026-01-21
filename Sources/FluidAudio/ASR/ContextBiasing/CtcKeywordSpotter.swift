import CoreML
import Foundation
import OSLog

/// Swift implementation of CTC keyword spotting for Parakeet-TDT CTC 110M,
/// mirroring the NeMo `ctc_word_spot` dynamic programming algorithm.
///
/// This engine:
/// - Runs the MelSpectrogram + AudioEncoder CoreML models from `CtcModels`.
/// - Extracts CTC logits and converts them to log‑probabilities over time.
/// - Applies DP to score each keyword independently (no beam search competition).
public struct CtcKeywordSpotter {

    let logger = AppLogger(category: "CtcKeywordSpotter")
    private let models: CtcModels
    public let blankId: Int
    private let predictionOptions: MLPredictionOptions

    private let sampleRate: Int = ASRConstants.sampleRate
    private let maxModelSamples: Int = ASRConstants.maxModelSamples

    // Chunking parameters for audio longer than maxModelSamples
    // 2s overlap at 16kHz = 32,000 samples (matches TDT chunking pattern)
    private let chunkOverlapSamples: Int = 32_000

    // Debug flag - enabled only in DEBUG builds
    #if DEBUG
    let debugMode: Bool = false  // Set to true locally for verbose logging
    #else
    let debugMode: Bool = false
    #endif

    // Flag to enable/disable chunked CTC processing
    private let useChunkedCTC: Bool = true

    // Temperature for CTC softmax (higher = softer distribution, lower = more peaked)
    private let temperature: Float = ContextBiasingConstants.ctcTemperature

    // Blank bias applied to log probabilities (positive values penalize blank token)
    private let blankBias: Float = ContextBiasingConstants.blankBias

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
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
        // Input is fixed-size (240k samples padded), so ANE/GPU should work fine.
    }

    /// Convenience helper to create a spotter using the default cache location.
    public static func makeDefault(
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) async throws -> CtcKeywordSpotter {
        let models = try await CtcModels.downloadAndLoad()
        return CtcKeywordSpotter(models: models, blankId: blankId)
    }

    // MARK: - Public API

    /// Spot a single keyword given its token IDs.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono audio samples.
    ///   - keywordTokenIds: Model vocabulary IDs for the keyword.
    /// - Returns: Tuple `(score, startFrame, endFrame)` where `score` is average log‑prob per token.
    public func spotKeyword(
        audioSamples: [Float],
        keywordTokenIds: [Int]
    ) async throws -> (score: Float, startFrame: Int, endFrame: Int) {
        let ctcResult = try await computeLogProbs(for: audioSamples)
        let (score, start, end) = ctcWordSpot(logProbs: ctcResult.logProbs, keywordTokens: keywordTokenIds)
        return (score, start, end)
    }

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
                let detectionSummary =
                    "  '\(term.text)': score=\(scoreText), frames=[\(start), \(end)], time=[\(startText)s, \(endText)s]"
                logger.debug("\(detectionSummary)")
                logger.debug("    CTC token IDs: \(ids)")

                // Sample log-probs for this term's tokens at the detected window
                if start < logProbs.count && end <= logProbs.count {
                    let windowSize = min(5, end - start)
                    for frameIdx in start..<min(start + windowSize, end) {
                        let frame = logProbs[frameIdx]
                        var tokenLogProbs: [String] = []
                        for tokenId in ids {
                            if tokenId < frame.count {
                                let logProb = frame[tokenId]
                                tokenLogProbs.append("id\(tokenId)=\(String(format: "%.4f", logProb))")
                            }
                        }
                        logger.debug("      frame[\(frameIdx)]: \(tokenLogProbs.joined(separator: ", "))")

                        // Show top 10 most likely tokens at this frame
                        let topK = 10
                        let sortedIndices = frame.enumerated()
                            .sorted { $0.element > $1.element }
                            .prefix(topK)
                        let topTokens = sortedIndices.map { "id\($0.offset)=\(String(format: "%.4f", $0.element))" }
                        logger.debug("      top-\(topK) tokens: \(topTokens.joined(separator: ", "))")
                    }
                }
            }

            // Adjust threshold for multi-token phrases (they naturally have lower scores)
            // Each additional token beyond 3 relaxes the threshold by 1.0
            let tokenCount = ids.count
            let adjustedThreshold: Float? = minScore.map { base in
                let extraTokens = max(0, tokenCount - 3)
                return base - Float(extraTokens) * 1.0
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
                    let extraTokens = max(0, tokenCount - 3)
                    return base - Float(extraTokens) * 1.0
                } ?? -15.0

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

    // MARK: - CoreML pipeline

    func computeLogProbs(for audioSamples: [Float]) async throws -> CtcLogProbResult {
        guard !audioSamples.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0, frameTimes: nil)
        }

        // For audio longer than model limit, use chunked processing
        if useChunkedCTC && audioSamples.count > maxModelSamples {
            return try await computeLogProbsChunked(audioSamples: audioSamples)
        }

        // Use staged models (mel spectrogram + encoder) for short audio
        return try await computeWithStagedModels(audioSamples: audioSamples)
    }

    /// Process long audio in chunks with overlap, concatenating log-probs.
    ///
    /// Algorithm:
    /// 1. Split audio into chunks of maxModelSamples with chunkOverlapSamples overlap
    /// 2. Run CTC inference on each chunk
    /// 3. Concatenate log-probs, averaging overlapping frames
    private func computeLogProbsChunked(audioSamples: [Float]) async throws -> CtcLogProbResult {
        let totalSamples = audioSamples.count
        let chunkSize = maxModelSamples
        let overlap = chunkOverlapSamples
        let stride = chunkSize - overlap

        // Calculate number of chunks needed
        var chunks: [(start: Int, end: Int)] = []
        var start = 0
        while start < totalSamples {
            let end = min(start + chunkSize, totalSamples)
            chunks.append((start: start, end: end))
            if end >= totalSamples { break }
            start += stride
        }

        if debugMode {
            logger.debug("=== Chunked CTC Processing ===")
            logger.debug(
                "Total samples: \(totalSamples) (\(String(format: "%.2f", Double(totalSamples) / Double(sampleRate)))s)"
            )
            logger.debug("Chunk size: \(chunkSize), overlap: \(overlap), stride: \(stride)")
            logger.debug("Number of chunks: \(chunks.count)")
        }

        // Process each chunk
        var chunkResults: [CtcLogProbResult] = []
        for (idx, chunk) in chunks.enumerated() {
            let chunkAudio = Array(audioSamples[chunk.start..<chunk.end])

            if debugMode {
                let startTime = Double(chunk.start) / Double(sampleRate)
                let endTime = Double(chunk.end) / Double(sampleRate)
                logger.debug(
                    "  Chunk \(idx + 1)/\(chunks.count): samples [\(chunk.start)-\(chunk.end)] = [\(String(format: "%.2f", startTime))-\(String(format: "%.2f", endTime))s]"
                )
            }

            let result = try await computeWithStagedModels(audioSamples: chunkAudio)
            chunkResults.append(result)

            if debugMode {
                logger.debug(
                    "    -> \(result.totalFrames) frames, frameDuration=\(String(format: "%.4f", result.frameDuration))s"
                )
            }
        }

        guard !chunkResults.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0, frameTimes: nil)
        }

        // Use frame duration from first chunk (should be consistent)
        let frameDuration = chunkResults[0].frameDuration
        guard frameDuration > 0 else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0, frameTimes: nil)
        }

        // Calculate overlap in frames
        let overlapFrames = Int(Double(overlap) / Double(sampleRate) / frameDuration)

        // Concatenate log-probs with overlap averaging
        var concatenatedLogProbs: [[Float]] = []

        for (chunkIdx, result) in chunkResults.enumerated() {
            let logProbs = result.logProbs
            guard !logProbs.isEmpty else { continue }

            if chunkIdx == 0 {
                // First chunk: take all frames
                concatenatedLogProbs.append(contentsOf: logProbs)
            } else {
                // Subsequent chunks: average overlap region, then append non-overlapping part
                let overlapCount = min(overlapFrames, concatenatedLogProbs.count, logProbs.count)

                if overlapCount > 0 {
                    // Average the overlapping frames
                    let existingStart = concatenatedLogProbs.count - overlapCount
                    for i in 0..<overlapCount {
                        let existingIdx = existingStart + i
                        let newFrame = logProbs[i]
                        let existingFrame = concatenatedLogProbs[existingIdx]

                        // Element-wise average of log-probs
                        var averaged = [Float](repeating: 0, count: existingFrame.count)
                        for v in 0..<existingFrame.count {
                            averaged[v] = (existingFrame[v] + newFrame[v]) / 2.0
                        }
                        concatenatedLogProbs[existingIdx] = averaged
                    }
                }

                // Append non-overlapping frames from this chunk
                if overlapCount < logProbs.count {
                    concatenatedLogProbs.append(contentsOf: logProbs.suffix(from: overlapCount))
                }
            }
        }

        if debugMode {
            logger.debug("Concatenated: \(concatenatedLogProbs.count) total frames")
            logger.debug("Overlap frames averaged: \(overlapFrames) per boundary")
            logger.debug("==============================")
        }

        return CtcLogProbResult(
            logProbs: concatenatedLogProbs,
            frameDuration: frameDuration,
            totalFrames: concatenatedLogProbs.count,
            audioSamplesUsed: totalSamples,
            frameTimes: nil
        )
    }

    private func computeWithStagedModels(audioSamples: [Float]) async throws -> CtcLogProbResult {
        // Prepare fixed-length audio input expected by MelSpectrogram.
        let (audioInput, clampedCount) = try prepareAudioArray(audioSamples)
        let melInput = try makeFeatureProvider(name: "audio", array: audioInput, length: clampedCount)

        let melModel = models.melSpectrogram
        let encoderModel = models.encoder

        let melOutput = try await melModel.compatPrediction(
            from: melInput,
            options: predictionOptions
        )

        guard let melFeatures = melOutput.featureValue(for: "melspectrogram_features")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing melspectrogram_features from CTC MelSpectrogram model")
        }

        // Prefer explicit mel_length; otherwise infer from shape (frames axis).
        var melLengthValue =
            melOutput.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue
            ?? melFeatures.shape.last?.intValue
        if melFeatures.shape.count == 4 {
            melLengthValue = melFeatures.shape[2].intValue
        }

        if debugMode {
            logger.debug("Mel features shape: \(melFeatures.shape)")
            let lengthText = melLengthValue.map(String.init) ?? "nil"
            logger.debug("mel_length: \(lengthText)")

            // Print mel feature statistics to compare with Python NeMo
            let melCount = melFeatures.count
            var melMin: Float = Float.infinity
            var melMax: Float = -Float.infinity
            var melSum: Float = 0

            for i in 0..<melCount {
                let val = melFeatures[i].floatValue
                melMin = min(melMin, val)
                melMax = max(melMax, val)
                melSum += val
            }
            let melMean = melSum / Float(melCount)

            let statsSummary =
                String(format: "Mel features stats: min=%.4f, max=%.4f, mean=%.4f", melMin, melMax, melMean)
            logger.debug("\(statsSummary)")

            // Print first frame (first 10 features) for comparison
            if melFeatures.shape.count >= 4 {
                logger.debug("First mel frame (first 10 features):")
                var firstFrameVals: [String] = []
                for i in 0..<min(10, melFeatures.shape[3].intValue) {
                    let idx = [0, 0, 0, i] as [NSNumber]
                    let val = melFeatures[idx].floatValue
                    firstFrameVals.append(String(format: "%.4f", val))
                }
                logger.debug("  [\(firstFrameVals.joined(separator: ", "))]")
            }
        }

        // Build encoder input (mel features + length placeholder).
        let encoderInput = try makeEncoderInput(melFeatures: melFeatures, melLength: melLengthValue)

        // Run AudioEncoder to obtain CTC logits.
        let encoderOutput = try await encoderModel.compatPrediction(
            from: encoderInput,
            options: predictionOptions
        )

        // Check which output is available
        let hasRaw = encoderOutput.featureValue(for: "ctc_head_raw_output")?.multiArrayValue != nil
        let hasSoftmax = encoderOutput.featureValue(for: "ctc_head_output")?.multiArrayValue != nil

        if debugMode {
            logger.debug("CTC outputs available: ctc_head_raw_output=\(hasRaw), ctc_head_output=\(hasSoftmax)")
        }

        // Use ctc_head_raw_output (raw logits), NOT ctc_head_output (which contains post-softmax probabilities)
        // From debugging: ctc_head_output produces nonsense scores when passed through log-softmax again
        let ctcRaw =
            encoderOutput.featureValue(for: "ctc_head_raw_output")?.multiArrayValue
            ?? encoderOutput.featureValue(for: "ctc_head_output")?.multiArrayValue

        guard let ctcRaw else {
            throw ASRError.processingFailed(
                "Missing CTC head output from encoder model (expected ctc_head_raw_output or ctc_head_output)"
            )
        }

        if debugMode {
            logger.debug("CTC raw output shape: \(ctcRaw.shape)")
            let usedOutput = hasRaw ? "ctc_head_raw_output (raw logits)" : "ctc_head_output (post-softmax)"
            logger.debug("Using output: \(usedOutput)")
        }

        // Convert logits → log‑probabilities and trim padding frames.
        // Apply temperature scaling (CTC_TEMPERATURE) and blank bias (BLANK_BIAS)
        let allLogProbs = try makeLogProbs(from: ctcRaw, temperature: temperature, blankBias: blankBias)
        let trimmed = trimLogProbs(allLogProbs, audioSampleCount: clampedCount)
        let frameCount = trimmed.count

        if debugMode {
            logger.debug(
                "Log-probs computed: \(trimmed.count) frames (total: \(allLogProbs.count)), vocab size: \(trimmed.first?.count ?? 0)"
            )
            // Sample a few frames to check log-prob distribution
            if trimmed.count > 0 {
                let sampleFrameIndices = [0, trimmed.count / 2, trimmed.count - 1]
                for idx in sampleFrameIndices where idx < trimmed.count {
                    let frame = trimmed[idx]
                    let maxLogProb = frame.max() ?? -Float.infinity
                    let maxIdx = frame.firstIndex(of: maxLogProb) ?? -1
                    let blankLogProb = blankId < frame.count ? frame[blankId] : -Float.infinity
                    let maxText = String(format: "%.4f", maxLogProb)
                    let blankText = String(format: "%.4f", blankLogProb)
                    logger.debug("  frame[\(idx)]: max_logprob=\(maxText) at idx=\(maxIdx), blank_logprob=\(blankText)")
                }
            }
        }

        let frameDuration =
            frameCount > 0
            ? Double(clampedCount) / Double(frameCount) / Double(sampleRate)
            : 0

        return CtcLogProbResult(
            logProbs: trimmed,
            frameDuration: frameDuration,
            totalFrames: frameCount,
            audioSamplesUsed: clampedCount,
            frameTimes: nil
        )
    }

    private func prepareAudioArray(_ audioSamples: [Float]) throws -> (MLMultiArray, Int) {
        let clampedCount = min(audioSamples.count, maxModelSamples)

        // Detect expected input rank from the MelSpectrogram model's 'audio' feature description.
        // Canary-1b-v2 expects rank 1 [samples], parakeet-ctc-0.6b expects rank 2 [1, samples].
        let melModel = models.melSpectrogram
        let audioDesc = melModel.modelDescription.inputDescriptionsByName["audio"]
        let expectedRank = audioDesc?.multiArrayConstraint?.shape.count ?? 1

        // Determine data type - prefer float16 if model expects it, otherwise float32
        let dataType: MLMultiArrayDataType =
            audioDesc?.multiArrayConstraint?.dataType == .float16 ? .float16 : .float32

        let array: MLMultiArray
        if expectedRank == 2 {
            // Rank 2: [1, maxSamples]
            array = try MLMultiArray(shape: [1, NSNumber(value: maxModelSamples)], dataType: dataType)
        } else {
            // Rank 1: [maxSamples]
            array = try MLMultiArray(shape: [NSNumber(value: maxModelSamples)], dataType: dataType)
        }

        // Copy actual samples.
        for i in 0..<clampedCount {
            array[i] = NSNumber(value: audioSamples[i])
        }

        // Remaining positions are left as zero (default) to represent padding.
        if clampedCount < maxModelSamples {
            for i in clampedCount..<maxModelSamples {
                array[i] = 0
            }
        }

        if debugMode {
            let midpoint = clampedCount / 2
            var sampleVals: [String] = []
            for i in midpoint..<min(midpoint + 5, clampedCount) {
                sampleVals.append(String(format: "%.4f", audioSamples[i]))
            }
            let absMax = audioSamples.prefix(clampedCount).map { abs($0) }.max() ?? 0
            let mean = audioSamples.prefix(clampedCount).reduce(0.0, +) / Float(clampedCount)
            let statsText = String(
                format: "  Audio input: count=%d/%d, abs_max=%.4f, mean=%.6f",
                clampedCount, maxModelSamples, absMax, mean)
            logger.debug("\(statsText)")
            logger.debug("  mid_5=[\(sampleVals.joined(separator: ", "))]")
        }

        return (array, clampedCount)
    }

    private func makeFeatureProvider(
        name: String, array: MLMultiArray, length: Int? = nil
    ) throws
        -> MLFeatureProvider
    {
        var dict: [String: MLFeatureValue] = [
            name: MLFeatureValue(multiArray: array)
        ]

        if let length, name == "audio" {
            let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
            lengthArray[0] = NSNumber(value: length)
            dict["audio_length"] = MLFeatureValue(multiArray: lengthArray)
        }
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func makeEncoderInput(melFeatures: MLMultiArray, melLength: Int?) throws -> MLFeatureProvider {
        // The encoder expects:
        // - "melspectrogram_features": passthrough from MelSpectrogram
        // - "mel_length": [1] int32 frame count
        // Some exports also require a dummy "input_1": [1,1,1,1] fp16 flag.
        let lengthValue = melLength ?? melFeatures.shape.last?.intValue ?? 0
        guard lengthValue > 0 else {
            throw ASRError.processingFailed("Invalid mel_length for CTC encoder input")
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: lengthValue)

        var dict: [String: MLFeatureValue] = [
            "melspectrogram_features": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ]

        // Optional placeholder accepted by some staged exports.
        if let input1 = try? MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16) {
            input1[0] = 1
            dict["input_1"] = MLFeatureValue(multiArray: input1)
        }

        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func makeLogProbs(
        from ctcOutput: MLMultiArray,
        applyLogSoftmax: Bool = true,
        temperature: Float = 1.0,
        blankBias: Float = 0.0
    ) throws -> [[Float]] {
        let rank = ctcOutput.shape.count
        guard rank == 3 || rank == 4 else {
            throw ASRError.processingFailed("Unexpected CTC output rank: \(ctcOutput.shape)")
        }

        let vocabSize: Int
        let timeSteps: Int
        let indexBuilder: (Int, Int) -> [NSNumber]

        if rank == 3 {
            // Expected shape: [1, timeSteps, vocabSize]
            timeSteps = ctcOutput.shape[1].intValue
            vocabSize = ctcOutput.shape[2].intValue
            indexBuilder = { t, v in [0, t, v].map { NSNumber(value: $0) } }
        } else {
            // Expected shape: [1, vocabSize, 1, timeSteps]
            vocabSize = ctcOutput.shape[1].intValue
            timeSteps = ctcOutput.shape[3].intValue
            indexBuilder = { t, v in [0, v, 0, t].map { NSNumber(value: $0) } }
        }

        if vocabSize <= 0 || timeSteps <= 0 {
            return []
        }

        var logProbs: [[Float]] = Array(
            repeating: Array(repeating: 0, count: vocabSize),
            count: timeSteps
        )

        // Iterate over time/vocab dimensions, read logits or log-probabilities.
        // Apply log-softmax per frame when needed.
        for t in 0..<timeSteps {
            var logits = [Float](repeating: 0, count: vocabSize)

            for v in 0..<vocabSize {
                logits[v] = ctcOutput[indexBuilder(t, v)].floatValue
            }

            // Show detailed logit analysis for sample frames
            if debugMode && (t == 0 || t == timeSteps / 4 || t == timeSteps / 2 || t == 3 * timeSteps / 4) {
                let maxLogit = logits.max() ?? 0
                let minLogit = logits.min() ?? 0
                let blankLogit = logits.indices.contains(blankId) ? logits[blankId] : 0

                // Show top 5 non-blank tokens
                var nonBlankLogits: [(id: Int, logit: Float)] = []
                for v in 0..<blankId {
                    nonBlankLogits.append((id: v, logit: logits[v]))
                }
                nonBlankLogits.sort { $0.logit > $1.logit }
                let bestNonBlank = nonBlankLogits.first?.logit ?? 0
                let gap = blankLogit - bestNonBlank
                let top3 = nonBlankLogits.prefix(3)
                    .map { "id=\($0.id):\(String(format: "%.1f", $0.logit))" }
                    .joined(separator: ", ")
                logger.debug(
                    "  frame[\(t)]: blank=\(String(format: "%.1f", blankLogit)), gap=\(String(format: "%.1f", gap)), top3=[\(top3)]"
                )
            }

            var row = applyLogSoftmax ? logSoftmax(logits, temperature: temperature) : logits

            // Apply blank bias: subtract from blank token log prob to penalize it
            if blankBias != 0.0 && blankId < row.count {
                row[blankId] -= blankBias
            }

            logProbs[t] = row
        }

        return logProbs
    }

    private func logSoftmax(_ logits: [Float], temperature: Float = 1.0) -> [Float] {
        guard !logits.isEmpty else { return [] }

        // Apply temperature scaling: divide logits by temperature before softmax
        // Higher temperature = softer distribution (spreads probability mass)
        // Lower temperature = sharper distribution (more peaked)
        let scaledLogits = temperature != 1.0 ? logits.map { $0 / temperature } : logits

        let maxLogit = scaledLogits.max() ?? 0
        var sumExp: Float = 0

        for i in 0..<scaledLogits.count {
            sumExp += expf(scaledLogits[i] - maxLogit)
        }

        let logSumExp = logf(sumExp)
        var result: [Float] = Array(repeating: 0, count: scaledLogits.count)

        // log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
        for i in 0..<scaledLogits.count {
            result[i] = (scaledLogits[i] - maxLogit) - logSumExp
        }

        return result
    }

    private func trimLogProbs(_ logProbs: [[Float]], audioSampleCount: Int) -> [[Float]] {
        guard !logProbs.isEmpty else { return logProbs }

        let totalFrames = logProbs.count
        if audioSampleCount >= maxModelSamples {
            return logProbs
        }

        let samplesPerFrame = Double(maxModelSamples) / Double(totalFrames)
        let validFrames = Int(ceil(Double(audioSampleCount) / samplesPerFrame))
        let clampedFrames = max(1, min(validFrames, totalFrames))

        if debugMode {
            logger.debug("[DEBUG] Trimming CTC frames:")
            logger.debug(
                "[DEBUG]   totalFrames=\(totalFrames), sampleCount=\(audioSampleCount), maxModelSamples=\(maxModelSamples)"
            )
            logger.debug(
                "[DEBUG]   samplesPerFrame=\(String(format: "%.2f", samplesPerFrame)), validFrames=\(validFrames), clampedFrames=\(clampedFrames)"
            )
        }

        return Array(logProbs.prefix(clampedFrames))
    }

    // MARK: - NeMo-compatible DP

    /// Dynamic programming keyword alignment, ported from
    /// `NeMo/scripts/asr_context_biasing/ctc_word_spotter.py:ctc_word_spot`.
    // Wildcard token ID: represents "*" that matches anything at zero cost
    private static let WILDCARD_TOKEN_ID = ContextBiasingConstants.wildcardTokenId

    /// Core DP table construction shared by all CTC word spotting variants.
    /// Returns filled DP, backtrack, and lastMatch arrays for downstream interpretation.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities [T, vocab_size]
    ///   - keywordTokens: Token IDs for the keyword (may include WILDCARD_TOKEN_ID)
    /// - Returns: Tuple of (dp, backtrack, lastMatch) where:
    ///   - dp[t][n] = best score to match first n tokens by time t
    ///   - backtrack[t][n] = start frame for the best alignment ending at t with n tokens matched
    ///   - lastMatch[t][n] = frame where token n was last matched (actual end frame)
    private func fillDPTable(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let T = logProbs.count
        let N = keywordTokens.count

        // dp[t][n] = best score to match first n tokens by time t
        var dp = Array(
            repeating: Array(repeating: -Float.greatestFiniteMagnitude, count: N + 1),
            count: T + 1
        )
        var backtrack = Array(
            repeating: Array(repeating: 0, count: N + 1),
            count: T + 1
        )
        // lastMatch[t][n] = the frame where the nth token was last matched
        // This tracks the ACTUAL end frame, not the DP evaluation frame
        var lastMatch = Array(
            repeating: Array(repeating: 0, count: N + 1),
            count: T + 1
        )

        // Initialize: keyword of length 0 has score 0 at any time
        for t in 0...T {
            dp[t][0] = 0.0
        }

        for t in 1...T {
            let frame = logProbs[t - 1]

            for n in 1...N {
                let tokenId = keywordTokens[n - 1]

                // Wildcard token: matches any symbol (including blank) at zero cost
                if tokenId == Self.WILDCARD_TOKEN_ID {
                    let wildcardSkip = dp[t - 1][n - 1]  // Move to next token
                    let wildcardStay = dp[t - 1][n]  // Stay on wildcard
                    let wildcardScore = max(wildcardSkip, wildcardStay)
                    dp[t][n] = wildcardScore
                    if wildcardScore == wildcardSkip {
                        backtrack[t][n] = t - 1
                        lastMatch[t][n] = t  // Wildcard consumed at this frame
                    } else {
                        backtrack[t][n] = backtrack[t - 1][n]
                        lastMatch[t][n] = lastMatch[t - 1][n]  // Propagate from previous
                    }
                    continue
                }

                if tokenId < 0 || tokenId >= frame.count {
                    continue
                }

                let tokenScore = frame[tokenId]

                // Option 1: match this token at this timestep (new token or repeat)
                let matchScore = max(
                    dp[t - 1][n - 1] + tokenScore,
                    dp[t - 1][n] + tokenScore
                )

                // Option 2: skip this timestep (blank or other token)
                let skipScore = dp[t - 1][n]

                if matchScore > skipScore {
                    dp[t][n] = matchScore
                    backtrack[t][n] = t - 1
                    lastMatch[t][n] = t  // Token matched at this frame
                } else {
                    dp[t][n] = skipScore
                    backtrack[t][n] = backtrack[t - 1][n]
                    lastMatch[t][n] = lastMatch[t - 1][n]  // Propagate last match frame
                }
            }
        }

        return (dp, backtrack, lastMatch)
    }

    /// Count non-wildcard tokens for score normalization.
    private func nonWildcardCount(_ keywordTokens: [Int]) -> Int {
        keywordTokens.filter { $0 != Self.WILDCARD_TOKEN_ID }.count
    }

    func ctcWordSpot(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return (-Float.infinity, 0, 0)
        }

        let (dp, backtrack, lastMatch) = fillDPTable(logProbs: logProbs, keywordTokens: keywordTokens)

        // Find best end position for the full keyword
        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude

        if T >= N {
            for t in N...T {
                if dp[t][N] > bestScore {
                    bestScore = dp[t][N]
                    bestEnd = t
                }
            }
        }

        let bestStart = backtrack[bestEnd][N]
        // Use lastMatch to get the actual end frame where the last token matched
        let actualEndFrame = lastMatch[bestEnd][N]

        // Normalize score by non-wildcard tokens
        let normFactor = nonWildcardCount(keywordTokens)
        let normalizedScore = normFactor > 0 ? bestScore / Float(normFactor) : bestScore

        return (normalizedScore, bestStart, actualEndFrame)
    }

    /// Constrained CTC word spotting within a temporal window.
    ///
    /// Unlike `ctcWordSpot` which searches the entire audio for the best alignment,
    /// this method restricts the search to a specific frame range. This is useful when
    /// you already know approximately where a word should be (e.g., from TDT timestamps)
    /// and want to verify/refine the detection within that window.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities [T, vocab_size]
    ///   - keywordTokens: Token IDs for the keyword
    ///   - searchStartFrame: Start of search window (inclusive)
    ///   - searchEndFrame: End of search window (exclusive)
    /// - Returns: Tuple `(score, startFrame, endFrame)` in global frame coordinates
    func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        // Clamp search window to valid range
        let clampedStart = max(0, searchStartFrame)
        let clampedEnd = min(T, searchEndFrame)

        if N == 0 || clampedEnd <= clampedStart {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        // Slice logProbs to the search window
        let windowLogProbs = Array(logProbs[clampedStart..<clampedEnd])
        let windowT = windowLogProbs.count

        if windowT < N {
            // Window too small for keyword
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let (dp, backtrack, lastMatch) = fillDPTable(logProbs: windowLogProbs, keywordTokens: keywordTokens)

        // Find best end position within the window
        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude

        for t in N...windowT {
            if dp[t][N] > bestScore {
                bestScore = dp[t][N]
                bestEnd = t
            }
        }

        let bestStart = backtrack[bestEnd][N]
        // Use lastMatch to get the actual end frame where the last token matched
        let actualEndFrame = lastMatch[bestEnd][N]

        // Normalize score by non-wildcard tokens
        let normFactor = nonWildcardCount(keywordTokens)
        let normalizedScore = normFactor > 0 ? bestScore / Float(normFactor) : bestScore

        // Convert window-relative indices back to global frame coordinates
        let globalStart = clampedStart + bestStart
        let globalEnd = clampedStart + actualEndFrame

        return (normalizedScore, globalStart, globalEnd)
    }

    /// Find ALL occurrences of a keyword in the audio, not just the best one.
    /// Returns multiple (score, startFrame, endFrame) tuples for each detection.
    /// Per NeMo CTC-WS paper: finds all candidates, merges overlapping intervals.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities [T, vocab_size]
    ///   - keywordTokens: Token IDs for the keyword
    ///   - minScore: Minimum normalized score threshold (default: -15.0)
    ///   - mergeOverlap: Whether to merge overlapping detections (default: true)
    /// - Returns: Array of (score, startFrame, endFrame) tuples
    func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = -15.0,
        mergeOverlap: Bool = true
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return []
        }

        let (dp, backtrack, lastMatch) = fillDPTable(logProbs: logProbs, keywordTokens: keywordTokens)

        // Normalize score factor
        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        // Find all positions where the complete keyword has good score
        // Look for local maxima in the score
        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []

        guard T >= N else { return [] }

        for t in N...T {
            let rawScore = dp[t][N]
            let normalizedScore = rawScore / normFactor

            // Check if this is a local maximum (better than neighbors)
            let prevScore = t > N ? dp[t - 1][N] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? dp[t + 1][N] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                let startFrame = backtrack[t][N]
                // Use lastMatch to get the actual end frame where the last token matched
                let actualEndFrame = lastMatch[t][N]
                candidates.append((score: normalizedScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        // If no local maxima found but there are valid scores, take the global best
        if candidates.isEmpty {
            var bestEnd = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for t in N...T {
                let normalizedScore = dp[t][N] / normFactor
                if normalizedScore > bestScore {
                    bestScore = normalizedScore
                    bestEnd = t
                }
            }
            if bestScore >= minScore {
                let startFrame = backtrack[bestEnd][N]
                // Use lastMatch to get the actual end frame where the last token matched
                let actualEndFrame = lastMatch[bestEnd][N]
                candidates.append((score: bestScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        guard mergeOverlap else { return candidates }

        // Merge overlapping intervals, keeping the best score
        let sorted = candidates.sorted { $0.startFrame < $1.startFrame }
        var merged: [(score: Float, startFrame: Int, endFrame: Int)] = []

        for candidate in sorted {
            if let last = merged.last {
                // Check for overlap
                if candidate.startFrame <= last.endFrame {
                    // Overlapping - keep the one with better score
                    if candidate.score > last.score {
                        merged[merged.count - 1] = candidate
                    }
                    // Extend end frame if needed
                    if candidate.endFrame > merged[merged.count - 1].endFrame {
                        merged[merged.count - 1].endFrame = candidate.endFrame
                    }
                } else {
                    merged.append(candidate)
                }
            } else {
                merged.append(candidate)
            }
        }

        return merged
    }

}
