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

    /// Result for a single keyword detection.
    public struct KeywordDetection: Sendable {
        public let term: CustomVocabularyTerm
        public let score: Float
        public let startFrame: Int
        public let endFrame: Int
        public let startTime: TimeInterval
        public let endTime: TimeInterval
    }

    public init(models: CtcModels, blankId: Int = 1024) {
        self.models = models
        self.blankId = blankId
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
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
        let (logProbs, _) = try await computeLogProbs(for: audioSamples)
        let (score, start, end) = ctcWordSpot(logProbs: logProbs, keywordTokens: keywordTokenIds)
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
        let (logProbs, totalFramesBeforeTrim) = try await computeLogProbs(for: audioSamples)
        guard !logProbs.isEmpty else { return [] }

        if debugMode {
            print("=== CTC Keyword Spotter Debug ===", to: &standardError)
            print(
                "Audio samples: \(audioSamples.count), frames: \(logProbs.count) (total before trim: \(totalFramesBeforeTrim))",
                to: &standardError)
            print("Vocab size: \(logProbs[0].count), blank ID: \(blankId)", to: &standardError)
            print("Terms to spot: \(customVocabulary.terms.count)", to: &standardError)
        }

        // Each CTC frame spans a fixed slice of audio based on the model's downsampling.
        // The model outputs a fixed number of frames (totalFramesBeforeTrim, typically 188)
        // for maxModelSamples (240000), so frame duration is constant.
        let samplesPerFrame = Double(maxModelSamples) / Double(totalFramesBeforeTrim)
        let frameDuration = samplesPerFrame / Double(sampleRate)

        if debugMode {
            print("[DEBUG] CTC frame timing:", to: &standardError)
            print(
                "[DEBUG]   samplesPerFrame=\(String(format: "%.2f", samplesPerFrame)), frameDuration=\(String(format: "%.5f", frameDuration))s (~\(String(format: "%.2f", 1.0 / frameDuration)) fps)",
                to: &standardError)
        }

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
                print(
                    "  '\(term.text)': score=\(String(format: "%.4f", score)), frames=[\(start), \(end)], time=[\(String(format: "%.3f", TimeInterval(start) * frameDuration))s, \(String(format: "%.3f", TimeInterval(end) * frameDuration))s]",
                    to: &standardError)
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
                    print(
                        "    REJECTED: score \(String(format: "%.4f", score)) <= threshold \(String(format: "%.4f", threshold))",
                        to: &standardError)
                }
                continue
            }

            let startTime = TimeInterval(start) * frameDuration
            let endTime = TimeInterval(end) * frameDuration

            let detection = KeywordDetection(
                term: term,
                score: score,
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

    private func computeLogProbs(for audioSamples: [Float]) async throws -> (logProbs: [[Float]], totalFrames: Int) {
        guard !audioSamples.isEmpty else { return ([], 0) }

        // 1) Prepare fixed-length audio input expected by MelSpectrogram.
        let audioInput = try prepareAudioArray(audioSamples)
        let melInput = try makeFeatureProvider(name: "audio", array: audioInput)

        // 2) Run MelSpectrogram.
        let melOutput = try await models.melSpectrogram.compatPrediction(
            from: melInput,
            options: predictionOptions
        )

        guard
            let melFeatures = melOutput.featureValue(for: "melspectrogram_features")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Missing melspectrogram_features from CTC MelSpectrogram model")
        }

        if debugMode {
            print("Mel features shape: \(melFeatures.shape)", to: &standardError)

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

            print(
                "Mel features stats: min=\(String(format: "%.4f", melMin)), max=\(String(format: "%.4f", melMax)), mean=\(String(format: "%.4f", melMean))",
                to: &standardError)

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

        // 3) Build encoder input (mel features + length placeholder).
        let encoderInput = try makeEncoderInput(melFeatures: melFeatures)

        // 4) Run AudioEncoder to obtain CTC logits.
        let encoderOutput = try await models.encoder.compatPrediction(
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

        // 5) Convert logits → log‑probabilities and trim padding frames.
        let allLogProbs = try makeLogProbs(from: ctcRaw)
        let totalFrames = allLogProbs.count
        let trimmed = trimLogProbs(allLogProbs, for: audioSamples.count)

        if debugMode {
            print(
                "Log-probs computed: \(trimmed.count) frames (total: \(totalFrames)), vocab size: \(trimmed.first?.count ?? 0)",
                to: &standardError)
            // Sample a few frames to check log-prob distribution
            if trimmed.count > 0 {
                let sampleFrameIndices = [0, trimmed.count / 2, trimmed.count - 1]
                for idx in sampleFrameIndices where idx < trimmed.count {
                    let frame = trimmed[idx]
                    let maxLogProb = frame.max() ?? -Float.infinity
                    let maxIdx = frame.firstIndex(of: maxLogProb) ?? -1
                    let blankLogProb = blankId < frame.count ? frame[blankId] : -Float.infinity
                    print(
                        "  frame[\(idx)]: max_logprob=\(String(format: "%.4f", maxLogProb)) at idx=\(maxIdx), blank_logprob=\(String(format: "%.4f", blankLogProb))",
                        to: &standardError)
                }
            }
        }

        return (trimmed, totalFrames)
    }

    private func prepareAudioArray(_ audioSamples: [Float]) throws -> MLMultiArray {
        let clampedCount = min(audioSamples.count, maxModelSamples)
        // Use Float16 to match the CoreML model's expected input type
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
            print(
                "  Audio input: count=\(clampedCount)/\(maxModelSamples), mid_5=[\(sampleVals.joined(separator: ", "))], abs_max=\(String(format: "%.4f", absMax)), mean=\(String(format: "%.6f", mean))",
                to: &standardError)
        }

        return array
    }

    private func makeFeatureProvider(name: String, array: MLMultiArray) throws -> MLFeatureProvider {
        let dict: [String: MLFeatureValue] = [
            name: MLFeatureValue(multiArray: array)
        ]
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func makeEncoderInput(melFeatures: MLMultiArray) throws -> MLFeatureProvider {
        // The encoder expects:
        // - "melspectrogram_features": [1, 1, T, 80]
        // - "input_1": scalar placeholder (length/flag). Use 1 as a simple "valid" marker.
        // Use Float16 to match the CoreML model's expected input type
        let lengthArray = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        lengthArray[0] = 1

        let dict: [String: MLFeatureValue] = [
            "melspectrogram_features": MLFeatureValue(multiArray: melFeatures),
            "input_1": MLFeatureValue(multiArray: lengthArray),
        ]
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    private func makeLogProbs(from ctcOutput: MLMultiArray) throws -> [[Float]] {
        // Expected shape: [1, vocabSize, 1, timeSteps]
        guard ctcOutput.shape.count == 4 else {
            throw ASRError.processingFailed("Unexpected CTC output rank: \(ctcOutput.shape)")
        }

        let vocabSize = ctcOutput.shape[1].intValue
        let timeSteps = ctcOutput.shape[3].intValue

        if vocabSize <= 0 || timeSteps <= 0 {
            return []
        }

        var logProbs: [[Float]] = Array(
            repeating: Array(repeating: 0, count: vocabSize),
            count: timeSteps
        )

        // Iterate over time and vocab dimensions, read logits, and apply log-softmax per frame.
        for t in 0..<timeSteps {
            var logits = [Float](repeating: 0, count: vocabSize)

            for v in 0..<vocabSize {
                let index = [0, v, 0, t].map { NSNumber(value: $0) }
                logits[v] = ctcOutput[index].floatValue
            }

            if debugMode && t == 0 {
                let maxLogit = logits.max() ?? 0
                let minLogit = logits.min() ?? 0
                let blankLogit = logits[blankId]
                print(
                    "  Raw logits frame[0]: min=\(String(format: "%.4f", minLogit)), max=\(String(format: "%.4f", maxLogit)), blank=\(String(format: "%.4f", blankLogit))",
                    to: &standardError)
            }

            let row = logSoftmax(logits)
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

    private func trimLogProbs(_ logProbs: [[Float]], for sampleCount: Int) -> [[Float]] {
        guard !logProbs.isEmpty else { return logProbs }

        let totalFrames = logProbs.count
        if sampleCount >= maxModelSamples {
            return logProbs
        }

        let samplesPerFrame = Double(maxModelSamples) / Double(totalFrames)
        let validFrames = Int(ceil(Double(sampleCount) / samplesPerFrame))
        let clampedFrames = max(1, min(validFrames, totalFrames))

        if debugMode {
            print("[DEBUG] Trimming CTC frames:", to: &standardError)
            print(
                "[DEBUG]   totalFrames=\(totalFrames), sampleCount=\(sampleCount), maxModelSamples=\(maxModelSamples)",
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
