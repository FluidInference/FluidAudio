import CoreML
import Foundation

/// Swift implementation of CTC keyword spotting for Parakeet-TDT CTC 110M,
/// mirroring the NeMo `ctc_word_spot` dynamic programming algorithm.
///
/// This engine:
/// - Runs the MelSpectrogram + AudioEncoder CoreML models from `CtcModels`.
/// - Extracts CTC logits and converts them to log‑probabilities over time.
/// - Applies DP to score each keyword independently (no beam search competition).
public struct CtcKeywordSpotter: Sendable {

    let logger = AppLogger(category: "CtcKeywordSpotter")
    private let models: CtcModels
    public let blankId: Int

    /// Computed property to avoid storing non-Sendable MLPredictionOptions.
    /// Creating on demand is cheap (just init + empty dict).
    private var predictionOptions: MLPredictionOptions {
        AsrModels.optimizedPredictionOptions()
    }

    private let sampleRate: Int = ASRConstants.sampleRate
    private let maxModelSamples: Int = ASRConstants.maxModelSamples

    // Chunking parameters for audio longer than maxModelSamples
    // 2s overlap at 16kHz = 32,000 samples (matches TDT chunking pattern)
    private let chunkOverlapSamples: Int = 32_000

    // Debug flag - enabled only in DEBUG builds
    #if DEBUG
    let debugMode: Bool = true  // Set to true locally for verbose logging
    #else
    let debugMode: Bool = false
    #endif

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

    // MARK: - CoreML pipeline

    func computeLogProbs(for audioSamples: [Float]) async throws -> CtcLogProbResult {
        guard !audioSamples.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0, frameTimes: nil)
        }

        // For audio longer than model limit, use chunked processing
        if audioSamples.count > maxModelSamples {
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
        let melInput = try makeAudioFeatureProvider(array: audioInput, length: clampedCount)

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
            logger.debug(
                "Mel features shape: \(melFeatures.shape), mel_length: \(melLengthValue.map(String.init) ?? "nil")")
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
                "Log-probs: \(trimmed.count) frames (total: \(allLogProbs.count)), vocab size: \(trimmed.first?.count ?? 0)"
            )
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

        // Copy actual samples (MLMultiArray is zero-initialized, so padding is implicit).
        for i in 0..<clampedCount {
            array[i] = NSNumber(value: audioSamples[i])
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

    private func makeAudioFeatureProvider(array: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: length)
        return try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: array),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])
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
