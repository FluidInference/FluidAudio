import CoreML
import Foundation
import OSLog

// Helper for stderr output
fileprivate struct StderrOutputStream: TextOutputStream {
    func write(_ string: String) {
        if let data = string.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
fileprivate var standardError = StderrOutputStream()

/// Swift implementation of CTC keyword spotting for Parakeet-TDT CTC 110M,
/// mirroring the NeMo `ctc_word_spot` dynamic programming algorithm.
///
/// This engine:
/// - Runs the MelSpectrogram + AudioEncoder CoreML models from `CtcModels`.
/// - Extracts CTC logits and converts them to log‑probabilities over time.
/// - Applies DP to score each keyword independently (no beam search competition).
public struct CtcKeywordSpotter {

    private let logger = AppLogger(category: "CtcKeywordSpotter")
    private let models: CtcModels
    private let blankId: Int
    private let predictionOptions: MLPredictionOptions

    // Parakeet CTC 110M expects up to 240_000 samples (≈15s at 16kHz).
    private let sampleRate: Int = 16_000
    private let maxModelSamples: Int = 240_000

    // Debug flag controlled by environment variable FLUIDAUDIO_DEBUG_CTC_BOOSTING
    private let debugMode: Bool = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

    private struct CtcLogProbResult {
        let logProbs: [[Float]]
        let frameDuration: Double
        let totalFrames: Int
        let audioSamplesUsed: Int
        let frameTimes: [Double]?
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
    }

    public init(models: CtcModels, blankId: Int = 1024) {
        self.models = models
        self.blankId = blankId
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
        // CTC staged models have dynamic shapes; force CPU to avoid CoreML NE/GPU crashes.
        self.predictionOptions.usesCPUOnly = true
    }

    /// Convenience helper to create a spotter using the default cache location.
    public static func makeDefault(blankId: Int = 1024) async throws -> CtcKeywordSpotter {
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
            print("=== CTC Keyword Spotter Debug ===", to: &standardError)
            print(
                "Audio samples: \(audioSamples.count), frames: \(logProbs.count)",
                to: &standardError)
            print("Vocab size: \(logProbs[0].count), blank ID: \(blankId)", to: &standardError)
            print("Terms to spot: \(customVocabulary.terms.count)", to: &standardError)
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
                    print("  Skipping '\(term.text)': no CTC token IDs", to: &standardError)
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
                print(detectionSummary, to: &standardError)
                print("    CTC token IDs: \(ids)", to: &standardError)

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
                        print("      frame[\(frameIdx)]: \(tokenLogProbs.joined(separator: ", "))", to: &standardError)

                        // Show top 10 most likely tokens at this frame
                        let topK = 10
                        let sortedIndices = frame.enumerated()
                            .sorted { $0.element > $1.element }
                            .prefix(topK)
                        print("      top-\(topK) tokens: ", terminator: "", to: &standardError)
                        let topTokens = sortedIndices.map { "id\($0.offset)=\(String(format: "%.4f", $0.element))" }
                        print(topTokens.joined(separator: ", "), to: &standardError)
                    }
                }
            }

            if let threshold = minScore, score <= threshold {
                if debugMode {
                    let thresholdText = String(format: "%.4f", threshold)
                    print(
                        "    REJECTED: score \(String(format: "%.4f", score)) <= threshold \(thresholdText)",
                        to: &standardError)
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
                print("    ACCEPTED: adding detection", to: &standardError)
            }
        }

        if debugMode {
            print("Total detections: \(results.count)", to: &standardError)
            print("=================================", to: &standardError)
        }

        return results
    }

    // MARK: - CoreML pipeline

    private func computeLogProbs(for audioSamples: [Float]) async throws -> CtcLogProbResult {
        guard !audioSamples.isEmpty else {
            return CtcLogProbResult(
                logProbs: [], frameDuration: 0, totalFrames: 0, audioSamplesUsed: 0, frameTimes: nil)
        }

        // Prefer staged path, but fall back to fused if staged fails at runtime.
        if models.melSpectrogram != nil, models.encoder != nil {
            do {
                return try await computeWithStagedModels(audioSamples: audioSamples)
            } catch {
                if debugMode {
                    print("[DEBUG] Staged CTC failed, falling back to fused: \(error)", to: &standardError)
                }
            }
        }

        if let fused = models.fusedMelEncoder {
            return try await computeWithFusedModel(fused, audioSamples: audioSamples)
        }

        throw ASRError.processingFailed("No CTC model available (staged or fused)")
    }

    private func computeWithFusedModel(
        _ model: MLModel,
        audioSamples: [Float]
    ) async throws -> CtcLogProbResult {
        let clampedCount = min(audioSamples.count, maxModelSamples)
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: clampedCount)], dataType: .float32)

        for i in 0..<clampedCount {
            audioArray[i] = NSNumber(value: audioSamples[i])
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: clampedCount)

        let dict: [String: MLFeatureValue] = [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ]
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: dict)

        let output = try await model.compatPrediction(
            from: inputProvider,
            options: predictionOptions
        )

        guard let logProbsArray = output.featureValue(for: "log_probs")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing log_probs output from fused CTC model")
        }
        guard let encoderLengthArray = output.featureValue(for: "encoder_length")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing encoder_length output from fused CTC model")
        }
        let frameTimesArray = output.featureValue(for: "frame_times")?.multiArrayValue

        let encoderFrames = encoderLengthArray[0].intValue
        let allLogProbs = try makeLogProbs(from: logProbsArray, applyLogSoftmax: false)
        let frameCount = max(0, min(encoderFrames, allLogProbs.count))
        let trimmed = frameCount > 0 ? Array(allLogProbs.prefix(frameCount)) : []

        var frameDuration: Double = 0
        var frameTimes: [Double]? = nil
        if let frameTimesArray {
            var times: [Double] = []
            times.reserveCapacity(frameTimesArray.count)
            for i in 0..<frameTimesArray.count {
                times.append(frameTimesArray[i].doubleValue)
            }
            if frameCount > 0 {
                times = Array(times.prefix(frameCount))
            }
            frameTimes = times
            if times.count > 1 {
                frameDuration = max(0, times[1] - times[0])
            }
        }
        if frameDuration == 0 {
            frameDuration =
                frameCount > 0
                ? Double(clampedCount) / Double(frameCount) / Double(sampleRate)
                : 0
        }

        if debugMode {
            let fusedSummary =
                "Fused CTC log-probs: frames=\(trimmed.count)/\(allLogProbs.count), encoder_length=\(encoderFrames)"
            let vocabSummary = "  vocab size \(trimmed.first?.count ?? 0)"
            print(fusedSummary + "," + vocabSummary, to: &standardError)
        }

        return CtcLogProbResult(
            logProbs: trimmed,
            frameDuration: frameDuration,
            totalFrames: frameCount,
            audioSamplesUsed: clampedCount,
            frameTimes: frameTimes
        )
    }

    private func computeWithStagedModels(audioSamples: [Float]) async throws -> CtcLogProbResult {
        // Prepare fixed-length audio input expected by MelSpectrogram.
        let (audioInput, clampedCount) = try prepareAudioArray(audioSamples)
        let melInput = try makeFeatureProvider(name: "audio", array: audioInput, length: clampedCount)

        guard
            let melModel = models.melSpectrogram,
            let encoderModel = models.encoder
        else {
            throw ASRError.processingFailed("CTC MelSpectrogram/AudioEncoder models are unavailable")
        }

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
            print("Mel features shape: \(melFeatures.shape)", to: &standardError)
            let lengthText = melLengthValue.map(String.init) ?? "nil"
            print("mel_length: \(lengthText)", to: &standardError)

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
            print(statsSummary, to: &standardError)

            // Print first frame (first 10 features) for comparison
            if melFeatures.shape.count >= 4 {
                print("First mel frame (first 10 features):", to: &standardError)
                var firstFrameVals: [String] = []
                for i in 0..<min(10, melFeatures.shape[3].intValue) {
                    let idx = [0, 0, 0, i] as [NSNumber]
                    let val = melFeatures[idx].floatValue
                    firstFrameVals.append(String(format: "%.4f", val))
                }
                print("  [\(firstFrameVals.joined(separator: ", "))]", to: &standardError)
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
            print(
                "CTC outputs available: ctc_head_raw_output=\(hasRaw), ctc_head_output=\(hasSoftmax)",
                to: &standardError)
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
            print("CTC raw output shape: \(ctcRaw.shape)", to: &standardError)
            let usedOutput = hasRaw ? "ctc_head_raw_output (raw logits)" : "ctc_head_output (post-softmax)"
            print(
                "Using output: \(usedOutput)",
                to: &standardError)
        }

        // Convert logits → log‑probabilities and trim padding frames.
        let allLogProbs = try makeLogProbs(from: ctcRaw)
        let trimmed = trimLogProbs(allLogProbs, audioSampleCount: clampedCount)
        let frameCount = trimmed.count

        if debugMode {
            print(
                "Log-probs computed: \(trimmed.count) frames (total: \(allLogProbs.count)), vocab size: \(trimmed.first?.count ?? 0)",
                to: &standardError)
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
                    print(
                        "  frame[\(idx)]: max_logprob=\(maxText) at idx=\(maxIdx), blank_logprob=\(blankText)",
                        to: &standardError)
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
        // Use Float16 to match the CoreML model's expected input type.
        // Current staged Mel expects 1-D: [maxSamples].
        let array = try MLMultiArray(shape: [NSNumber(value: maxModelSamples)], dataType: .float16)

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
            print(statsText, to: &standardError)
            print("  mid_5=[\(sampleVals.joined(separator: ", "))]", to: &standardError)
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

        let dict: NSMutableDictionary = [
            "melspectrogram_features": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ]

        // Optional placeholder accepted by some staged exports.
        if let input1 = try? MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16) {
            input1[0] = 1
            dict["input_1"] = MLFeatureValue(multiArray: input1)
        }

        return try MLDictionaryFeatureProvider(dictionary: dict as! [String: MLFeatureValue])
    }

    private func makeLogProbs(
        from ctcOutput: MLMultiArray,
        applyLogSoftmax: Bool = true
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

            if debugMode && t == 0 {
                let maxLogit = logits.max() ?? 0
                let minLogit = logits.min() ?? 0
                let blankLogit = logits.indices.contains(blankId) ? logits[blankId] : 0
                let minText = String(format: "%.4f", minLogit)
                let maxText = String(format: "%.4f", maxLogit)
                let blankText = String(format: "%.4f", blankLogit)
                print(
                    "  Raw logits frame[0]: min=\(minText), max=\(maxText), blank=\(blankText)",
                    to: &standardError)
            }

            let row = applyLogSoftmax ? logSoftmax(logits) : logits
            logProbs[t] = row
        }

        return logProbs
    }

    private func logSoftmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }

        let maxLogit = logits.max() ?? 0
        var sumExp: Float = 0
        var shifted: [Float] = Array(repeating: 0, count: logits.count)

        for i in 0..<logits.count {
            let v = expf(logits[i] - maxLogit)
            shifted[i] = v
            sumExp += v
        }

        let logSumExp = logf(sumExp)
        var result: [Float] = Array(repeating: 0, count: logits.count)

        for i in 0..<logits.count {
            result[i] = logf(shifted[i]) - logSumExp
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
            print("[DEBUG] Trimming CTC frames:", to: &standardError)
            print(
                "[DEBUG]   totalFrames=\(totalFrames), sampleCount=\(audioSampleCount), maxModelSamples=\(maxModelSamples)",
                to: &standardError)
            print(
                "[DEBUG]   samplesPerFrame=\(String(format: "%.2f", samplesPerFrame)), validFrames=\(validFrames), clampedFrames=\(clampedFrames)",
                to: &standardError)
        }

        return Array(logProbs.prefix(clampedFrames))
    }

    // MARK: - NeMo-compatible DP

    /// Dynamic programming keyword alignment, ported from
    /// `NeMo/scripts/asr_context_biasing/ctc_word_spotter.py:ctc_word_spot`.
    // Wildcard token ID: -1 represents "*" that matches anything at zero cost
    private static let WILDCARD_TOKEN_ID = -1

    func ctcWordSpot(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return (-Float.infinity, 0, 0)
        }

        // dp[t][n] = best score to match first n tokens by time t
        var dp = Array(
            repeating: Array(repeating: -Float.greatestFiniteMagnitude, count: N + 1),
            count: T + 1
        )
        var backtrackTime = Array(
            repeating: Array(repeating: 0, count: N + 1),
            count: T + 1
        )

        // Initialize: keyword of length 0 has score 0 at any time.
        for t in 0...T {
            dp[t][0] = 0.0
        }

        for t in 1...T {
            let frame = logProbs[t - 1]

            for n in 1...N {
                let tokenId = keywordTokens[n - 1]

                // Wildcard token: matches any symbol (including blank) at zero cost
                if tokenId == Self.WILDCARD_TOKEN_ID {
                    // Wildcard can skip this frame at zero cost
                    let wildcardSkip = dp[t - 1][n - 1]  // Move to next token
                    let wildcardStay = dp[t - 1][n]  // Stay on wildcard

                    let wildcardScore = max(wildcardSkip, wildcardStay)
                    dp[t][n] = wildcardScore
                    backtrackTime[t][n] = wildcardScore == wildcardSkip ? t - 1 : backtrackTime[t - 1][n]
                    continue
                }

                if tokenId < 0 || tokenId >= frame.count {
                    continue
                }

                let tokenScore = frame[tokenId]

                // Option 1: match this token at this timestep (new token or repeat).
                let matchScore = max(
                    dp[t - 1][n - 1] + tokenScore,
                    dp[t - 1][n] + tokenScore
                )

                // Option 2: skip this timestep (blank or other token).
                let skipScore = dp[t - 1][n]

                if matchScore > skipScore {
                    dp[t][n] = matchScore
                    backtrackTime[t][n] = t - 1
                } else {
                    dp[t][n] = skipScore
                    backtrackTime[t][n] = backtrackTime[t - 1][n]
                }
            }
        }

        // Find best end position for the full keyword.
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

        let bestStart = backtrackTime[bestEnd][N]

        // Normalize score only by non-wildcard tokens
        let nonWildcardCount = keywordTokens.filter { $0 != Self.WILDCARD_TOKEN_ID }.count
        let normalizedScore = nonWildcardCount > 0 ? bestScore / Float(nonWildcardCount) : bestScore

        return (normalizedScore, bestStart, bestEnd)
    }
}
