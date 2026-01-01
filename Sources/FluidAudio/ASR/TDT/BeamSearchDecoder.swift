import Accelerate
import CoreML
import Foundation
import OSLog

/// Beam search decoder with vocabulary biasing for TDT models.
///
/// Uses the JointDecisionSingleStep model which outputs top-K candidates per frame,
/// enabling proper beam search with vocabulary boosting. Unlike the deprecated Joint
/// model, this processes frames one at a time with correct decoder state updates.
///
/// The JointDecisionSingleStep model takes:
/// - encoder_step: [1, 1024, 1] - single encoder frame
/// - decoder_step: [1, 640, 1] - single decoder state
/// And produces:
/// - token_id: [1, 1, 1] - greedy choice
/// - token_prob: [1, 1, 1] - probability
/// - duration: [1, 1, 1] - duration bin
/// - top_k_ids: [1, 1, 1, 64] - top-64 token candidates
/// - top_k_logits: [1, 1, 1, 64] - logits for scoring
public final class BeamSearchDecoder {

    private let logger = Logger(subsystem: "com.fluidaudio", category: "BeamSearchDecoder")

    /// Configuration for beam search
    public struct Config: Sendable {
        /// Number of hypotheses to maintain
        public let beamWidth: Int

        /// Boost applied when a token continues a vocabulary term prefix
        public let partialMatchBoost: Float

        /// Boost applied when a token completes a vocabulary term
        public let completeMatchBoost: Float

        /// Minimum log probability to consider a token
        public let minLogProb: Float

        /// Number of duration bins in the model
        public let numDurationBins: Int

        /// Vocabulary size (excluding blank)
        public let vocabSize: Int

        /// Blank token ID
        public let blankId: Int

        /// Maximum symbols per step (prevents infinite loops)
        public let maxSymbolsPerStep: Int

        /// Number of top-K candidates from joint model
        public let topK: Int

        /// Penalty applied to non-greedy tokens (to prevent spurious divergence)
        public let nonGreedyPenalty: Float

        /// Maximum acoustic gap from greedy to consider a token
        /// Tokens with logProb more than this below greedy are ignored
        /// This prevents vocabulary boosting of acoustically implausible tokens
        public let maxAcousticGap: Float

        public static let `default` = Config(
            beamWidth: 10,
            partialMatchBoost: 5.0,
            completeMatchBoost: 2.0,
            minLogProb: -15.0,
            numDurationBins: 5,
            vocabSize: 8192,
            blankId: 8192,
            maxSymbolsPerStep: 10,
            topK: 64,  // Use all top-K candidates from joint model
            nonGreedyPenalty: 3.0,  // Penalty for non-greedy token selection
            maxAcousticGap: 5.0  // Only consider tokens within ~0.7% probability of greedy
        )

        public init(
            beamWidth: Int = 10,
            partialMatchBoost: Float = 5.0,
            completeMatchBoost: Float = 2.0,
            minLogProb: Float = -15.0,
            numDurationBins: Int = 5,
            vocabSize: Int = 8192,
            blankId: Int = 8192,
            maxSymbolsPerStep: Int = 10,
            topK: Int = 64,
            nonGreedyPenalty: Float = 3.0,
            maxAcousticGap: Float = 5.0
        ) {
            self.beamWidth = beamWidth
            self.partialMatchBoost = partialMatchBoost
            self.completeMatchBoost = completeMatchBoost
            self.minLogProb = minLogProb
            self.numDurationBins = numDurationBins
            self.vocabSize = vocabSize
            self.blankId = blankId
            self.maxSymbolsPerStep = maxSymbolsPerStep
            self.topK = topK
            self.nonGreedyPenalty = nonGreedyPenalty
            self.maxAcousticGap = maxAcousticGap
        }
    }

    /// A hypothesis in the beam
    public struct Hypothesis {
        /// Token IDs decoded so far
        var tokens: [Int]

        /// Cumulative log probability (acoustic score)
        var acousticScore: Float

        /// Cumulative vocabulary boost score
        var vocabBoost: Float

        /// Combined score for ranking (raw, not length-normalized during beam search)
        /// Length normalization is applied only when selecting the final best hypothesis
        var combinedScore: Float {
            acousticScore + vocabBoost
        }

        /// Current decoder hidden state (h)
        var hState: MLMultiArray

        /// Current decoder cell state (c)
        var cState: MLMultiArray

        /// Cached decoder output for reuse
        var decoderOutput: MLMultiArray?

        /// Last emitted token (for decoder input)
        var lastToken: Int

        /// Token-level timing information (frame indices)
        var timestamps: [Int]

        /// Current frame index
        var frameIndex: Int

        /// Number of non-blank symbols emitted without advancing frames
        var symbolsAtCurrentFrame: Int

        /// Trie cursor for vocabulary biasing
        var trieCursor: VocabularyTrie.Cursor?
    }

    private let config: Config
    private let vocabularyTrie: VocabularyTrie?

    /// Initialize beam search decoder
    /// - Parameters:
    ///   - config: Beam search configuration
    ///   - vocabularyTrie: Optional vocabulary trie for biasing
    public init(config: Config = .default, vocabularyTrie: VocabularyTrie? = nil) {
        self.config = config
        self.vocabularyTrie = vocabularyTrie
    }

    /// Decode using beam search with vocabulary biasing
    ///
    /// Uses the JointDecisionSingleStep model for frame-by-frame decoding with
    /// proper decoder state updates after each non-blank emission.
    ///
    /// - Parameters:
    ///   - encoderOutput: Encoder output [1, D, T] where D=1024, T=num_frames
    ///   - jointSingleStepModel: JointDecisionSingleStep model with top-K outputs
    ///   - decoderModel: Decoder model for prediction network
    ///   - initialHState: Initial decoder hidden state
    ///   - initialCState: Initial decoder cell state
    /// - Returns: Best hypothesis tokens and frame timestamps
    public func decode(
        encoderOutput: MLMultiArray,
        jointSingleStepModel: MLModel,
        decoderModel: MLModel,
        initialHState: MLMultiArray,
        initialCState: MLMultiArray
    ) throws -> (tokens: [Int], timestamps: [Int]) {

        let encoderShape = encoderOutput.shape.map { $0.intValue }
        guard encoderShape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(encoderShape)")
        }

        let numFrames = encoderShape[2]
        guard numFrames > 0 else {
            return ([], [])
        }

        // Build encoder frame view for efficient access
        let encoderFrames = try EncoderFrameView(encoderOutput: encoderOutput, validLength: numFrames)

        // Prime decoder with SOS (blank) token to get initial decoder output and state
        let (initialDecoderOut, primedH, primedC) = try runDecoder(
            model: decoderModel,
            lastToken: config.blankId,
            hState: initialHState,
            cState: initialCState
        )

        // Initialize beam with single hypothesis
        var beam: [Hypothesis] = [
            Hypothesis(
                tokens: [],
                acousticScore: 0.0,
                vocabBoost: 0.0,
                hState: try copyMLMultiArray(primedH),
                cState: try copyMLMultiArray(primedC),
                decoderOutput: initialDecoderOut,
                lastToken: config.blankId,
                timestamps: [],
                frameIndex: 0,
                symbolsAtCurrentFrame: 0,
                trieCursor: vocabularyTrie?.makeCursor()
            )
        ]

        // Preallocate arrays for joint model input
        let encoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1024, 1],
            dataType: .float32
        )
        let decoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 640, 1],
            dataType: .float32
        )

        // Get actual strides for ANE-aligned arrays (critical for correct data layout)
        let encDestStride = encoderStep.strides.map { $0.intValue }[1]
        let encDestPtr = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: 1024)

        // Process frames using TDT-style decoding with beam search
        while !beam.isEmpty {
            // Find minimum frame index across all hypotheses
            let minFrameIdx = beam.map { $0.frameIndex }.min() ?? numFrames
            guard minFrameIdx < numFrames else { break }

            var allCandidates: [Hypothesis] = []

            for hypothesis in beam {
                // Skip if this hypothesis is ahead of the minimum frame
                guard hypothesis.frameIndex == minFrameIdx else {
                    allCandidates.append(hypothesis)
                    continue
                }

                // Get or compute decoder output
                var baseHypothesis = hypothesis
                let decoderOut: MLMultiArray

                if let cached = hypothesis.decoderOutput {
                    decoderOut = cached
                } else {
                    let result = try runDecoder(
                        model: decoderModel,
                        lastToken: hypothesis.lastToken,
                        hState: hypothesis.hState,
                        cState: hypothesis.cState
                    )
                    decoderOut = result.output
                    baseHypothesis.hState = try copyMLMultiArray(result.hOut)
                    baseHypothesis.cState = try copyMLMultiArray(result.cOut)
                    baseHypothesis.decoderOutput = decoderOut
                }

                // Prepare encoder step for current frame (use ANE-aligned stride)
                try encoderFrames.copyFrame(
                    at: hypothesis.frameIndex, into: encDestPtr, destinationStride: encDestStride)

                // Prepare decoder step
                try populateDecoderStep(decoderOut, into: decoderStep)

                // Run joint single-step model to get top-K candidates
                let jointResult = try runJointSingleStep(
                    model: jointSingleStepModel,
                    encoderStep: encoderStep,
                    decoderStep: decoderStep
                )

                // Get the model's predicted duration for this frame
                // Duration is a property of the acoustic signal at this frame, not the specific token
                let frameDuration = max(jointResult.duration, 1)

                // Get greedy log probability for acoustic gap comparison
                let greedyLogProb = jointResult.topKLogits[0] - jointResult.logSumExp

                // Process top-K candidates
                for i in 0..<min(config.topK, jointResult.topKIds.count) {
                    let tokenId = jointResult.topKIds[i]
                    let logit = jointResult.topKLogits[i]

                    // Convert logit to log probability using the truncated logSumExp
                    var logProb = logit - jointResult.logSumExp

                    // Filter out tokens that are too acoustically unlikely compared to greedy
                    // This prevents vocabulary boost from hallucinating words during clear speech/silence
                    if i > 0 {
                        let acousticGap = greedyLogProb - logProb
                        guard acousticGap <= config.maxAcousticGap else { continue }

                        // Apply penalty for non-greedy tokens
                        logProb -= config.nonGreedyPenalty
                    }

                    guard logProb >= config.minLogProb else { continue }

                    var newHypothesis = baseHypothesis
                    newHypothesis.acousticScore += logProb

                    // All candidates at this frame use the same duration (from model prediction)
                    // Duration is determined by the acoustic signal, not by which token is chosen
                    var duration = frameDuration

                    if tokenId == config.blankId {
                        // Blank token - advance frame, keep same decoder state
                        newHypothesis.frameIndex = hypothesis.frameIndex + duration
                        newHypothesis.symbolsAtCurrentFrame = 0
                        // Keep cached decoder output for blank
                    } else {
                        // Non-blank token - emit token, invalidate decoder cache
                        newHypothesis.tokens.append(tokenId)
                        newHypothesis.lastToken = tokenId
                        newHypothesis.decoderOutput = nil  // Will recompute next iteration

                        // Apply vocabulary boost using cursor
                        if var cursor = newHypothesis.trieCursor {
                            var match = cursor.advance(tokenId)

                            // Check if this token starts a NEW match from root
                            if !match.isMatch {
                                let rootMatch = cursor.startsMatch(tokenId)
                                if rootMatch.isMatch {
                                    cursor.reset()
                                    cursor.forceAdvanceFromRoot(tokenId)
                                    match = rootMatch
                                } else {
                                    cursor.reset()
                                }
                            }

                            newHypothesis.trieCursor = cursor

                            switch match {
                            case .partial(let depth, _):
                                newHypothesis.vocabBoost += config.partialMatchBoost * Float(depth)
                            case .complete(let term, _):
                                let boost = config.completeMatchBoost * (term.weight ?? 1.0)
                                newHypothesis.vocabBoost += boost
                                logger.debug("Vocab boost: '\(term.text)' +\(boost)")
                            case .noMatch:
                                break
                            }
                        }

                        // Record timestamp
                        newHypothesis.timestamps.append(hypothesis.frameIndex)

                        // Track symbols at current frame for maxSymbolsPerStep guard
                        if duration == 0 {
                            let nextSymbols = newHypothesis.symbolsAtCurrentFrame + 1
                            if nextSymbols >= config.maxSymbolsPerStep {
                                duration = 1
                                newHypothesis.symbolsAtCurrentFrame = 0
                            } else {
                                newHypothesis.symbolsAtCurrentFrame = nextSymbols
                            }
                        } else {
                            newHypothesis.symbolsAtCurrentFrame = 0
                        }

                        // Advance frame by duration
                        newHypothesis.frameIndex = hypothesis.frameIndex + duration
                    }

                    allCandidates.append(newHypothesis)
                }
            }

            // Prune to top beamWidth hypotheses
            allCandidates.sort { $0.combinedScore > $1.combinedScore }
            beam = Array(allCandidates.prefix(config.beamWidth))
        }

        // Select best hypothesis with length normalization for final ranking
        guard !beam.isEmpty else {
            return ([], [])
        }

        // Apply length normalization for final selection
        let best = beam.max { a, b in
            let lenA = Float(max(a.tokens.count, 1))
            let lenB = Float(max(b.tokens.count, 1))
            let scoreA = (a.acousticScore + a.vocabBoost) / lenA
            let scoreB = (b.acousticScore + b.vocabBoost) / lenB
            return scoreA < scoreB
        }!

        return (best.tokens, best.timestamps)
    }

    // MARK: - Private Helper Methods

    private func runDecoder(
        model: MLModel,
        lastToken: Int,
        hState: MLMultiArray,
        cState: MLMultiArray
    ) throws -> (output: MLMultiArray, hOut: MLMultiArray, cOut: MLMultiArray) {

        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[0] = NSNumber(value: Int32(lastToken))

        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_length": MLFeatureValue(multiArray: targetLength),
            "h_in": MLFeatureValue(multiArray: hState),
            "c_in": MLFeatureValue(multiArray: cState),
        ])

        let output = try model.prediction(from: input)

        guard let decoderOut = output.featureValue(for: "decoder")?.multiArrayValue,
            let hOut = output.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = output.featureValue(for: "c_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Decoder output missing required features")
        }

        return (decoderOut, hOut, cOut)
    }

    /// Result from JointDecisionSingleStep model
    private struct JointSingleStepResult {
        let tokenId: Int
        let tokenProb: Float
        let duration: Int
        let topKIds: [Int]
        let topKLogits: [Float]
        let logSumExp: Float  // For converting logits to log probs
    }

    /// Run JointDecisionSingleStep model
    private func runJointSingleStep(
        model: MLModel,
        encoderStep: MLMultiArray,
        decoderStep: MLMultiArray
    ) throws -> JointSingleStepResult {

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: decoderStep),
        ])

        let output = try model.prediction(from: input)

        guard let tokenIdArray = output.featureValue(for: "token_id")?.multiArrayValue,
            let tokenProbArray = output.featureValue(for: "token_prob")?.multiArrayValue,
            let durationArray = output.featureValue(for: "duration")?.multiArrayValue,
            let topKIdsArray = output.featureValue(for: "top_k_ids")?.multiArrayValue,
            let topKLogitsArray = output.featureValue(for: "top_k_logits")?.multiArrayValue
        else {
            throw ASRError.processingFailed("JointSingleStep output missing required features")
        }

        // Extract scalar values
        let tokenIdPtr = tokenIdArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
        let tokenId = Int(tokenIdPtr[0])

        let tokenProbPtr = tokenProbArray.dataPointer.bindMemory(to: Float.self, capacity: 1)
        let tokenProb = tokenProbPtr[0]

        let durationPtr = durationArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
        let duration = Int(durationPtr[0])

        // Extract top-K arrays [1, 1, 1, K] -> [K]
        let k = topKIdsArray.shape.last!.intValue
        let topKIdsPtr = topKIdsArray.dataPointer.bindMemory(to: Int32.self, capacity: k)
        let topKLogitsPtr = topKLogitsArray.dataPointer.bindMemory(to: Float.self, capacity: k)

        var topKIds = [Int](repeating: 0, count: k)
        var topKLogits = [Float](repeating: 0, count: k)

        for i in 0..<k {
            topKIds[i] = Int(topKIdsPtr[i])
            topKLogits[i] = topKLogitsPtr[i]
        }

        // Compute log-sum-exp for normalizing logits to log probabilities
        let maxLogit = topKLogits.max() ?? 0
        let sumExp = topKLogits.reduce(0.0) { $0 + exp($1 - maxLogit) }
        let logSumExp = maxLogit + log(sumExp)

        return JointSingleStepResult(
            tokenId: tokenId,
            tokenProb: tokenProb,
            duration: duration,
            topKIds: topKIds,
            topKLogits: topKLogits,
            logSumExp: logSumExp
        )
    }

    /// Populate decoder step array from decoder output
    /// Handles stride conversion between source and destination arrays
    private func populateDecoderStep(_ decoderOutput: MLMultiArray, into decoderStep: MLMultiArray) throws {
        let hiddenSize = 640
        let shape = decoderOutput.shape.map { $0.intValue }

        guard shape.count == 3, decoderOutput.dataType == .float32 else {
            throw ASRError.processingFailed("Invalid decoder output shape: \(shape)")
        }

        // Find hidden dimension axis in source
        let hiddenAxis: Int
        if shape[2] == hiddenSize {
            hiddenAxis = 2
        } else if shape[1] == hiddenSize {
            hiddenAxis = 1
        } else {
            throw ASRError.processingFailed("Decoder output hidden size mismatch: \(shape)")
        }

        let srcPtr = decoderOutput.dataPointer.bindMemory(to: Float.self, capacity: decoderOutput.count)
        let dstPtr = decoderStep.dataPointer.bindMemory(to: Float.self, capacity: decoderStep.count)

        let srcStrides = decoderOutput.strides.map { $0.intValue }
        let srcHiddenStride = srcStrides[hiddenAxis]

        // Get destination stride (ANE-aligned arrays have stride[1] for hidden dimension)
        let dstStrides = decoderStep.strides.map { $0.intValue }
        let dstHiddenStride = dstStrides[1]

        // Copy with proper stride handling for both source and destination
        for i in 0..<hiddenSize {
            dstPtr[i * dstHiddenStride] = srcPtr[i * srcHiddenStride]
        }
    }

    /// Deep copy an MLMultiArray
    private func copyMLMultiArray(_ source: MLMultiArray) throws -> MLMultiArray {
        let copy = try MLMultiArray(shape: source.shape, dataType: source.dataType)
        let srcPtr = source.dataPointer.bindMemory(to: Float.self, capacity: source.count)
        let dstPtr = copy.dataPointer.bindMemory(to: Float.self, capacity: copy.count)
        for i in 0..<source.count {
            dstPtr[i] = srcPtr[i]
        }
        return copy
    }
}
