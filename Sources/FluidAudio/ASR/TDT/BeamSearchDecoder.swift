import Accelerate
import CoreML
import Foundation
import OSLog

/// Beam search decoder with vocabulary biasing for TDT models.
///
/// Uses the raw logits from the Joint model to maintain multiple hypotheses
/// and apply vocabulary boosting during decoding rather than post-processing.
///
/// The joint model takes:
/// - encoder: [1, 1024, T] - full encoder output
/// - decoder: [1, 640, 1] - single decoder step output
/// And produces:
/// - logits: [1, T, 1, 8198] - logits for all frames given decoder state
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

        /// Minimum probability to consider a token (log scale)
        public let minLogProb: Float

        /// Number of duration bins in the model
        public let numDurationBins: Int

        /// Vocabulary size (excluding blank)
        public let vocabSize: Int

        /// Blank token ID
        public let blankId: Int

        /// Maximum symbols per step (prevents infinite loops)
        public let maxSymbolsPerStep: Int

        public static let `default` = Config(
            beamWidth: 10,  // Increased from 4 to capture lower probability prefixes
            partialMatchBoost: 5.0,  // Increased from 0.5 to keep prefixes alive
            completeMatchBoost: 2.0,
            minLogProb: -15.0,
            numDurationBins: 5,
            vocabSize: 8192,
            blankId: 8192,
            maxSymbolsPerStep: 10
        )

        public init(
            beamWidth: Int = 10,
            partialMatchBoost: Float = 5.0,
            completeMatchBoost: Float = 2.0,
            minLogProb: Float = -15.0,
            numDurationBins: Int = 5,
            vocabSize: Int = 8192,
            blankId: Int = 8192,
            maxSymbolsPerStep: Int = 10
        ) {
            self.beamWidth = beamWidth
            self.partialMatchBoost = partialMatchBoost
            self.completeMatchBoost = completeMatchBoost
            self.minLogProb = minLogProb
            self.numDurationBins = numDurationBins
            self.vocabSize = vocabSize
            self.blankId = blankId
            self.maxSymbolsPerStep = maxSymbolsPerStep
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

        /// Combined score for ranking (length-normalized)
        var combinedScore: Float {
            // Simple length normalization to prevent bias against longer paths
            let len = Float(tokens.count + 1)
            return (acousticScore + vocabBoost) / len
        }

        /// Current decoder hidden state (h)
        var hState: MLMultiArray

        /// Current decoder cell state (c)
        var cState: MLMultiArray

        /// Cached decoder output for reuse in inner loop
        var decoderOutput: MLMultiArray?

        /// Last emitted token (for decoder input)
        var lastToken: Int

        /// Token-level timing information (frame indices)
        var timestamps: [Int]

        /// Current frame index
        var frameIndex: Int

        /// Number of non-blank symbols emitted without advancing frames.
        /// Used to guard against duration=0 infinite loops (mirrors TDT maxSymbolsPerStep).
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
    /// The joint model outputs logits for all encoder frames at once given a decoder state.
    /// We use this to efficiently score all frames, then apply TDT-style duration jumps.
    ///
    /// - Parameters:
    ///   - encoderOutput: Encoder output [1, D, T] where D=1024, T=num_frames
    ///   - jointModel: Joint model that outputs raw logits [1, T, 1, 8198]
    ///   - decoderModel: Decoder model for prediction network
    ///   - initialHState: Initial decoder hidden state
    ///   - initialCState: Initial decoder cell state
    /// - Returns: Best hypothesis tokens and frame timestamps
    public func decode(
        encoderOutput: MLMultiArray,
        jointModel: MLModel,
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

        // Prime decoder with SOS (blank) token to get initial decoder output and state.
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

                // Ensure we have decoder output/state for the current lastToken.
                // If missing, run the decoder once and treat the resulting state as "after lastToken".
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

                // Run joint to get logits for all frames given this decoder state
                // Note: The Joint model requires the full encoder sequence [1, 1024, T]
                // It computes logits for all frames in parallel against the current decoder state.
                let logits = try runJoint(
                    model: jointModel,
                    encoderOutput: encoderOutput,
                    decoderOutput: decoderOut
                )

                // Extract logits at current frame
                let (tokenLogProbs, durationLogits) = try extractLogitsAtFrame(
                    logits, frameIdx: hypothesis.frameIndex, numFrames: numFrames)

                // Get top-k tokens
                let topK = getTopKTokens(tokenLogProbs, k: config.beamWidth * 2)

                for (tokenId, logProb) in topK {
                    guard logProb >= config.minLogProb else { continue }

                    var newHypothesis = baseHypothesis
                    newHypothesis.acousticScore += logProb

                    // Get duration for frame advancement
                    let durationBin = argmax(durationLogits)
                    var duration = durationBin

                    // In TDT, blank duration=0 can cause stalls; always advance at least 1 frame.
                    if tokenId == config.blankId && duration == 0 {
                        duration = 1
                    }

                    if tokenId == config.blankId {
                        // Blank token - advance frame, keep same decoder state
                        newHypothesis.frameIndex = hypothesis.frameIndex + duration
                        // Keep cached decoder output for blank
                        // Trie cursor doesn't change on blank
                        newHypothesis.symbolsAtCurrentFrame = 0
                    } else {
                        // Non-blank token - emit token, need to update decoder
                        newHypothesis.tokens.append(tokenId)
                        newHypothesis.lastToken = tokenId
                        // Keep state after previous token; the new token will be incorporated
                        // the next time this hypothesis is expanded.
                        newHypothesis.decoderOutput = nil  // Invalidate cache

                        // Apply vocabulary boost using cursor
                        if var cursor = newHypothesis.trieCursor {
                            // 1. Try advancing current path
                            var match = cursor.advance(tokenId)

                            // 2. If no match, check if this token starts a NEW match from root
                            // (e.g. "The doc" -> "tor" (match) vs "The" -> "zebra" (no match, but "zebra" might be in vocab))
                            if !match.isMatch {
                                let rootMatch = cursor.startsMatch(tokenId)
                                if rootMatch.isMatch {
                                    cursor.reset()
                                    cursor.forceAdvanceFromRoot(tokenId)
                                    match = rootMatch
                                } else {
                                    cursor.reset()  // Lost track, reset to root
                                }
                            }

                            newHypothesis.trieCursor = cursor

                            switch match {
                            case .partial(let depth, _):
                                newHypothesis.vocabBoost += config.partialMatchBoost * Float(depth)
                            case .complete(let term, _):
                                let boost = self.config.completeMatchBoost * (term.weight ?? 1.0)
                                newHypothesis.vocabBoost += boost
                                self.logger.debug("Vocab boost: '\(term.text)' +\(boost)")
                            case .noMatch:
                                break
                            }
                        }

                        // Record timestamp
                        newHypothesis.timestamps.append(hypothesis.frameIndex)

                        // Track symbols emitted without advancing frames (duration=0 case)
                        if duration == 0 {
                            let nextSymbols = newHypothesis.symbolsAtCurrentFrame + 1
                            let exceedsLimit = nextSymbols >= config.maxSymbolsPerStep
                            // Force advancement to avoid infinite loops at a single frame.
                            duration = exceedsLimit ? 1 : duration
                            newHypothesis.symbolsAtCurrentFrame = exceedsLimit ? 0 : nextSymbols
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

        // Return best hypothesis
        guard let best = beam.first else {
            return ([], [])
        }

        return (best.tokens, best.timestamps)
    }

    private func runDecoder(
        model: MLModel,
        lastToken: Int,
        hState: MLMultiArray,
        cState: MLMultiArray
    ) throws -> (output: MLMultiArray, hOut: MLMultiArray, cOut: MLMultiArray) {

        // Create target input [1, 1] with last token
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

    /// Run joint model with full encoder output and single decoder step
    /// Joint model expects:
    /// - encoder: [1, 1024, T]
    /// - decoder: [1, 640, 1]
    /// Returns logits: [1, T, 1, 8198]
    private func runJoint(
        model: MLModel,
        encoderOutput: MLMultiArray,
        decoderOutput: MLMultiArray
    ) throws -> MLMultiArray {

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder": MLFeatureValue(multiArray: encoderOutput),
            "decoder": MLFeatureValue(multiArray: decoderOutput),
        ])

        let output = try model.prediction(from: input)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Joint output missing logits")
        }

        return logits
    }

    /// Extract logits at a specific frame from batch output [1, T, 1, 8198]
    private func extractLogitsAtFrame(
        _ logits: MLMultiArray,
        frameIdx: Int,
        numFrames: Int
    ) throws -> (tokenLogProbs: [Float], durationLogits: [Float]) {
        let shape = logits.shape.map { $0.intValue }
        let vocabWithBlank = config.vocabSize + 1  // 8193
        let totalSize = vocabWithBlank + config.numDurationBins  // 8198

        // Expected shape: [1, T, 1, 8198]
        guard shape.count == 4, shape[3] == totalSize else {
            throw ASRError.processingFailed("Unexpected logits shape: \(shape)")
        }

        let T = shape[1]
        let safeFrameIdx = min(frameIdx, T - 1)

        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)

        // Calculate offset for this frame: [1, frameIdx, 1, :]
        // Strides: shape is [1, T, 1, 8198]
        let strides = logits.strides.map { $0.intValue }
        let offset = safeFrameIdx * strides[1]

        // Extract token logits and apply log softmax
        var tokenLogits = [Float](repeating: 0, count: vocabWithBlank)
        for i in 0..<vocabWithBlank {
            tokenLogits[i] = ptr[offset + i]
        }
        let tokenLogProbs = logSoftmax(tokenLogits)

        // Extract duration logits
        var durationLogits = [Float](repeating: 0, count: config.numDurationBins)
        for i in 0..<config.numDurationBins {
            durationLogits[i] = ptr[offset + vocabWithBlank + i]
        }

        return (tokenLogProbs, durationLogits)
    }

    private func getTopKTokens(_ logProbs: [Float], k: Int) -> [(tokenId: Int, logProb: Float)] {
        let indexed = logProbs.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k)).map { (tokenId: $0.0, logProb: $0.1) }
    }

    private func logSoftmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        let maxVal = logits.max() ?? 0
        let shifted = logits.map { $0 - maxVal }
        let expSum = shifted.map { exp($0) }.reduce(0, +)
        guard expSum > 0 else { return [Float](repeating: -Float.infinity, count: logits.count) }
        let logExpSum = log(expSum)
        return shifted.map { $0 - logExpSum }
    }

    private func argmax(_ values: [Float]) -> Int {
        guard !values.isEmpty else { return 0 }
        var maxIdx = 0
        var maxVal = values[0]
        for (idx, val) in values.enumerated() {
            if val > maxVal {
                maxVal = val
                maxIdx = idx
            }
        }
        return maxIdx
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
