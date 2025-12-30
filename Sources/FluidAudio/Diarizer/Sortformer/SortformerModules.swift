import Accelerate
import Foundation
import OSLog

/// Core streaming logic for Sortformer diarization.
///
/// This mirrors NeMo's SortformerModules class, ported from Gradient Descent's implementation.
/// Reference: NeMo nemo/collections/asr/modules/sortformer_modules.py
public struct SortformerModules {

    private let logger = AppLogger(category: "SortformerModules")
    private let config: SortformerConfig

    public init(config: SortformerConfig) {
        self.config = config
    }

    // MARK: - State Initialization

    /// Initialize empty streaming state.
    public func initStreamingState() -> SortformerStreamingState {
        return SortformerStreamingState(config: config)
    }

    // MARK: - Streaming Update

    /// Update streaming state with new chunk.
    ///
    /// This is the core streaming logic from NeMo's streaming_update(),
    /// ported from Gradient Descent's MLTensor implementation.
    ///
    /// - Parameters:
    ///   - state: Current streaming state (mutated in place)
    ///   - chunk: Chunk embeddings from encoder [chunkLen, fcDModel] flattened (already trimmed, no context)
    ///   - preds: Full predictions [spkcache+fifo+chunkTotalFrames, numSpeakers] flattened
    ///   - leftContext: Left context frames to skip in predictions
    ///   - rightContext: Right context frames (for info only)
    /// - Returns: Confirmed predictions for this chunk [chunkLen * numSpeakers]
    public func streamUpdate(
        state: inout SortformerStreamingState,
        chunk: [Float],
        preds: [Float],
        leftContext: Int,
        rightContext: Int
    ) -> [Float] {
        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let spkcacheCapacity = config.spkcacheLen
        let fifoCapacity = config.fifoLen

        let currentSpkcacheLength = state.spkcacheLength
        let currentFifoLength = state.fifoLength
        let chunkStart = currentSpkcacheLength + currentFifoLength

        state.chunkCount += 1

        // Extract FIFO predictions if FIFO exists
        if currentFifoLength > 0 {
            let fifoPredsStart = currentSpkcacheLength * numSpeakers
            let fifoPredsEnd = (currentSpkcacheLength + currentFifoLength) * numSpeakers
            if fifoPredsEnd <= preds.count {
                state.fifoPreds = Array(preds[fifoPredsStart..<fifoPredsEnd])
            }
        }

        // Extract only CORE frames from chunk embeddings (skip left context, take chunkLen frames)
        // This matches Gradient Descent's: chunk[0..., lc..<chunkLen+lc, 0...]
        // Use ACTUAL leftContext (varies by chunk position), not fixed config.chunkLeftContext
        let lc = leftContext
        let coreFrames = config.chunkLen

        // Extract core embeddings only (frames lc..<lc+coreFrames)
        let embStartIdx = lc * fcDModel
        let embEndIdx = (lc + coreFrames) * fcDModel
        var chunkEmbs: [Float]
        if embEndIdx <= chunk.count {
            chunkEmbs = Array(chunk[embStartIdx..<embEndIdx])
        } else {
            // Fallback for short chunks: use what we have after left context
            let availableFrames = max(0, (chunk.count / fcDModel) - lc)
            if availableFrames > 0 {
                chunkEmbs = Array(chunk[embStartIdx..<(embStartIdx + availableFrames * fcDModel)])
            } else {
                chunkEmbs = []
            }
        }
        let actualCoreFrames = chunkEmbs.count / fcDModel

        // Extract chunk predictions for CORE frames only
        // This matches Gradient Descent's: preds[0..., chunkStart+lc..<chunkStart+chunkLen+lc, 0...]
        let chunkPredsStart = (chunkStart + lc) * numSpeakers
        let chunkPredsEnd = (chunkStart + lc + actualCoreFrames) * numSpeakers

        var chunkPreds: [Float]
        if chunkPredsEnd <= preds.count {
            chunkPreds = Array(preds[chunkPredsStart..<chunkPredsEnd])
        } else {
            // Fallback: just use zeros if we can't extract properly
            chunkPreds = [Float](repeating: 0.0, count: actualCoreFrames * numSpeakers)
        }

        // Append CORE chunk embeddings to FIFO (not full chunk with context)
        state.fifo.append(contentsOf: chunkEmbs)
        state.fifoLength += actualCoreFrames

        // Append CORE chunk predictions to FIFO preds
        if state.fifoPreds != nil {
            state.fifoPreds?.append(contentsOf: chunkPreds)
        } else {
            state.fifoPreds = chunkPreds
        }

        // Update speaker cache if FIFO overflows
        // Use actualCoreFrames (not full chunk), matching Gradient Descent's: chunkLen + currentFifoLength
        let contextLength = actualCoreFrames + currentFifoLength
        if contextLength > fifoCapacity {
            guard let currentFifoPreds = state.fifoPreds else {
                return chunkPreds
            }

            // Calculate how many frames to pop
            var popOutLength = config.spkcacheUpdatePeriod
            popOutLength = max(popOutLength, contextLength - fifoCapacity)
            popOutLength = min(popOutLength, state.fifoLength)

            // Extract frames to pop from FIFO
            let popOutEmbs = Array(state.fifo.prefix(popOutLength * fcDModel))
            let popOutPreds = Array(currentFifoPreds.prefix(popOutLength * numSpeakers))

            // Update silence profile
            updateSilenceProfile(
                state: &state,
                embs: popOutEmbs,
                preds: popOutPreds,
                frameCount: popOutLength
            )

            // Remove popped frames from FIFO
            state.fifo.removeFirst(popOutLength * fcDModel)
            state.fifoLength -= popOutLength
            state.fifoPreds?.removeFirst(popOutLength * numSpeakers)

            // Append popped embeddings to speaker cache
            state.spkcache.append(contentsOf: popOutEmbs)
            state.spkcacheLength += popOutLength

            // Update speaker cache predictions
            if state.spkcachePreds != nil {
                state.spkcachePreds?.append(contentsOf: popOutPreds)
            }

            // Compress speaker cache if it overflows
            if state.spkcacheLength > spkcacheCapacity {
                if state.spkcachePreds == nil {
                    // First time spkcache overflows - initialize predictions
                    let prevSpkcachePreds = Array(preds.prefix(currentSpkcacheLength * numSpeakers))
                    if currentSpkcacheLength > 0 {
                        state.spkcachePreds = prevSpkcachePreds + popOutPreds
                    } else {
                        state.spkcachePreds = popOutPreds
                    }
                }

                compressSpkcache(state: &state)
            }
        }

        return chunkPreds
    }

    // MARK: - Legacy Streaming Update (for backwards compatibility)

    /// Legacy streaming update that also returns tentative predictions.
    ///
    /// - Parameters:
    ///   - state: Current streaming state (mutated in place)
    ///   - chunkEmbeddings: Chunk embeddings from encoder [chunkLen, fcDModel]
    ///   - predictions: Full predictions [spkcacheLen + fifoLen + chunkLen, numSpeakers]
    ///   - leftContext: Left context frames to skip
    ///   - rightContext: Right context frames to skip (used for tentative predictions)
    ///   - modelSpkcacheLen: Spkcache length passed to model
    ///   - modelFifoLen: Fifo length passed to model
    /// - Returns: StreamingUpdateResult with confirmed and tentative predictions
    public func streamingUpdate(
        state: inout SortformerStreamingState,
        chunkEmbeddings: [Float],
        predictions: [Float],
        leftContext: Int,
        rightContext: Int,
        modelSpkcacheLen: Int? = nil,
        modelFifoLen: Int? = nil
    ) -> StreamingUpdateResult {
        // Use new streamUpdate logic
        let confirmed = streamUpdate(
            state: &state, chunk: chunkEmbeddings, preds: predictions, leftContext: leftContext,
            rightContext: rightContext)

        // Extract tentative predictions for right context frames
        let numSpeakers = config.numSpeakers
        let chunkStart = (modelSpkcacheLen ?? state.spkcacheLength) + (modelFifoLen ?? state.fifoLength)

        // Tentative predictions are after the confirmed CORE frames
        // Use config.chunkLen (6) as the core frame count, not full chunk with context
        let coreFrames = config.chunkLen
        let tentativeStart = (chunkStart + leftContext + coreFrames) * numSpeakers
        let tentativeCount = rightContext * numSpeakers
        var tentative: [Float] = []

        if tentativeCount > 0 && tentativeStart + tentativeCount <= predictions.count {
            tentative = Array(predictions[tentativeStart..<(tentativeStart + tentativeCount)])
        }

        return StreamingUpdateResult(confirmed: confirmed, tentative: tentative)
    }

    // MARK: - Silence Profile

    /// Update running mean of silence embeddings.
    private func updateSilenceProfile(
        state: inout SortformerStreamingState,
        embs: [Float],
        preds: [Float],
        frameCount: Int
    ) {
        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let silThreshold = config.silenceThreshold

        for frame in 0..<frameCount {
            // Check if frame is silence (sum of probs < threshold)
            var probSum: Float = 0.0
            for spk in 0..<numSpeakers {
                let idx = frame * numSpeakers + spk
                if idx < preds.count {
                    probSum += preds[idx]
                }
            }

            if probSum < silThreshold {
                // Update running mean
                let n = Float(state.silenceFrameCount)
                let newN = n + 1.0

                for d in 0..<fcDModel {
                    let embIdx = frame * fcDModel + d
                    if embIdx < embs.count {
                        let oldMean = state.meanSilenceEmbedding[d]
                        let newVal = embs[embIdx]
                        state.meanSilenceEmbedding[d] = (oldMean * n + newVal) / newN
                    }
                }

                state.silenceFrameCount += 1
            }
        }
    }

    // MARK: - Speaker Cache Compression

    /// Compress speaker cache to keep most important frames.
    ///
    /// This mirrors NeMo's _compress_spkcache() function,
    /// ported from Gradient Descent's implementation.
    private func compressSpkcache(state: inout SortformerStreamingState) {
        guard let spkcachePreds = state.spkcachePreds else { return }

        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let spkcacheCapacity = config.spkcacheLen
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk
        let currentLength = state.spkcacheLength

        let spkcacheLenPerSpk = spkcacheCapacity / numSpeakers - silFramesPerSpk
        let strongBoostPerSpk = Int(Float(spkcacheLenPerSpk) * config.strongBoostRate)
        let weakBoostPerSpk = Int(Float(spkcacheLenPerSpk) * config.weakBoostRate)
        let minPosScoresPerSpk = Int(Float(spkcacheLenPerSpk) * config.minPosScoresRate)

        // Compute log-based prediction scores
        var scores = getLogPredScores(preds: spkcachePreds, frameCount: currentLength)

        // Disable low scores
        scores = disableLowScores(
            preds: spkcachePreds,
            scores: scores,
            frameCount: currentLength,
            minPosScores: minPosScoresPerSpk
        )

        // Boost recent scores (frames beyond spkcacheCapacity)
        if currentLength > spkcacheCapacity {
            for frame in spkcacheCapacity..<currentLength {
                for spk in 0..<numSpeakers {
                    scores[frame * numSpeakers + spk] += config.scoresBoostLatest
                }
            }
        }

        // Strong boost to top-k scores
        scores = boostTopKScores(scores: scores, frameCount: currentLength, k: strongBoostPerSpk, scaleFactor: 2.0)

        // Weak boost to top-k scores
        scores = boostTopKScores(scores: scores, frameCount: currentLength, k: weakBoostPerSpk, scaleFactor: 1.0)

        // Add silence frame placeholders (infinity score to ensure selection)
        let totalFrames = currentLength + silFramesPerSpk
        for _ in 0..<(silFramesPerSpk * numSpeakers) {
            scores.append(Float.infinity)
        }

        // Get top-k indices
        let (topKIndices, isDisabled) = getTopKIndices(
            scores: scores,
            frameCount: totalFrames,
            k: spkcacheCapacity
        )

        // Gather compressed embeddings and predictions
        var newSpkcache = [Float](repeating: 0.0, count: spkcacheCapacity * fcDModel)
        var newSpkcachePreds = [Float](repeating: 0.0, count: spkcacheCapacity * numSpeakers)

        for (i, frameIdx) in topKIndices.enumerated() {
            if isDisabled[i] {
                // Use mean silence embedding
                for d in 0..<fcDModel {
                    newSpkcache[i * fcDModel + d] = state.meanSilenceEmbedding[d]
                }
                // Zero predictions for silence (already initialized to 0)
            } else if frameIdx < currentLength {
                // Copy embedding
                for d in 0..<fcDModel {
                    let srcIdx = frameIdx * fcDModel + d
                    if srcIdx < state.spkcache.count {
                        newSpkcache[i * fcDModel + d] = state.spkcache[srcIdx]
                    }
                }
                // Copy predictions
                for s in 0..<numSpeakers {
                    let srcIdx = frameIdx * numSpeakers + s
                    if srcIdx < spkcachePreds.count {
                        newSpkcachePreds[i * numSpeakers + s] = spkcachePreds[srcIdx]
                    }
                }
            }
        }

        state.spkcache = newSpkcache
        state.spkcacheLength = spkcacheCapacity
        state.spkcachePreds = newSpkcachePreds
    }

    // MARK: - Score Computation

    /// Compute log-based prediction scores.
    /// Score = log(p) - log(1-p) + sum(log(1-p_others)) - log(0.5)
    private func getLogPredScores(preds: [Float], frameCount: Int) -> [Float] {
        let numSpeakers = config.numSpeakers
        let threshold = config.predScoreThreshold
        var scores = [Float](repeating: 0.0, count: frameCount * numSpeakers)

        for frame in 0..<frameCount {
            // Compute sum of log(1-p) for all speakers
            var log1ProbsSum: Float = 0.0
            for spk in 0..<numSpeakers {
                let p = preds[frame * numSpeakers + spk]
                log1ProbsSum += log(max(1.0 - p, threshold))
            }

            for spk in 0..<numSpeakers {
                let p = preds[frame * numSpeakers + spk]
                let logP = log(max(p, threshold))
                let log1P = log(max(1.0 - p, threshold))

                // Score: log(p) - log(1-p) + sum(log(1-p_all)) - log(0.5)
                scores[frame * numSpeakers + spk] = logP - log1P + log1ProbsSum - log(0.5)
            }
        }

        return scores
    }

    /// Disable low scores for non-speech and overlapped speech.
    private func disableLowScores(
        preds: [Float],
        scores: [Float],
        frameCount: Int,
        minPosScores: Int
    ) -> [Float] {
        let numSpeakers = config.numSpeakers
        var result = scores

        // Count positive scores per speaker
        var posScoreCounts = [Int](repeating: 0, count: numSpeakers)
        for frame in 0..<frameCount {
            for spk in 0..<numSpeakers {
                if scores[frame * numSpeakers + spk] > 0 {
                    posScoreCounts[spk] += 1
                }
            }
        }

        for frame in 0..<frameCount {
            for spk in 0..<numSpeakers {
                let idx = frame * numSpeakers + spk
                let p = preds[idx]

                // Disable non-speech (p < 0.5)
                if p < 0.5 {
                    result[idx] = -.infinity
                    continue
                }

                // Disable non-positive scores if speaker has enough positive scores
                if result[idx] <= 0 && posScoreCounts[spk] >= minPosScores {
                    result[idx] = -.infinity
                }
            }
        }

        return result
    }

    /// Boost top-k scores for each speaker.
    private func boostTopKScores(
        scores: [Float],
        frameCount: Int,
        k: Int,
        scaleFactor: Float
    ) -> [Float] {
        let numSpeakers = config.numSpeakers
        let boostValue = scaleFactor * log(0.5)
        var result = scores

        for spk in 0..<numSpeakers {
            // Get scores for this speaker (excluding -infinity)
            var speakerScores: [(Int, Float)] = []
            for frame in 0..<frameCount {
                let idx = frame * numSpeakers + spk
                if result[idx] != -.infinity {
                    speakerScores.append((frame, result[idx]))
                }
            }

            // Sort by score descending
            speakerScores.sort { $0.1 > $1.1 }

            // Boost top-k
            for i in 0..<min(k, speakerScores.count) {
                let frame = speakerScores[i].0
                result[frame * numSpeakers + spk] -= boostValue
            }
        }

        return result
    }

    /// Get top-k frame indices based on scores.
    ///
    /// This mirrors NeMo's _get_topk_indices() exactly:
    /// - Permutes scores from (frames, speakers) to (speakers, frames)
    /// - Flattens and takes top-k indices
    /// - Uses modulo to convert back to frame indices
    private func getTopKIndices(
        scores: [Float],
        frameCount: Int,
        k: Int
    ) -> (indices: [Int], isDisabled: [Bool]) {
        let numSpeakers = config.numSpeakers
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk
        let nFramesNoSil = frameCount - silFramesPerSpk
        let maxIndex = config.maxIndex

        // Permute scores: (frames, speakers) -> (speakers, frames), then flatten
        var scoresFlattened = [Float](repeating: 0.0, count: numSpeakers * frameCount)
        for spk in 0..<numSpeakers {
            for frame in 0..<frameCount {
                let srcIdx = frame * numSpeakers + spk
                let dstIdx = spk * frameCount + frame
                scoresFlattened[dstIdx] = scores[srcIdx]
            }
        }

        // Get indices sorted by score (descending)
        let indexedScores = scoresFlattened.enumerated().map { ($0.offset, $0.element) }
        let sortedByScore = indexedScores.sorted { $0.1 > $1.1 }

        // Take top-k indices
        var topKIndices = [Int](repeating: 0, count: k)
        var topKValues = [Float](repeating: 0.0, count: k)

        for i in 0..<k {
            if i < sortedByScore.count {
                topKIndices[i] = sortedByScore[i].0
                topKValues[i] = sortedByScore[i].1
            } else {
                topKIndices[i] = maxIndex
                topKValues[i] = -.infinity
            }
        }

        // Replace -inf indices with maxIndex placeholder
        for i in 0..<k {
            if topKValues[i] == -.infinity {
                topKIndices[i] = maxIndex
            }
        }

        // Sort indices to preserve original order
        let sortedPairs = topKIndices.enumerated().sorted { $0.element < $1.element }
        var topKIndicesSorted = sortedPairs.map { $0.element }

        // Compute isDisabled BEFORE converting to frame indices
        var isDisabled = [Bool](repeating: false, count: k)
        for i in 0..<k {
            if topKIndicesSorted[i] == maxIndex {
                isDisabled[i] = true
            }
        }

        // Convert flattened index to frame index using modulo
        for i in 0..<k {
            if !isDisabled[i] {
                topKIndicesSorted[i] = topKIndicesSorted[i] % frameCount
            }
        }

        // Mark frames beyond actual content as disabled (silence padding frames)
        for i in 0..<k {
            if !isDisabled[i] && topKIndicesSorted[i] >= nFramesNoSil {
                isDisabled[i] = true
            }
        }

        // Set placeholder index for disabled frames
        for i in 0..<k where isDisabled[i] {
            topKIndicesSorted[i] = 0
        }

        return (topKIndicesSorted, isDisabled)
    }

    // MARK: - Sigmoid

    /// Apply sigmoid to convert logits to probabilities.
    public func applySigmoid(_ logits: [Float]) -> [Float] {
        return logits.map { 1.0 / (1.0 + exp(-$0)) }
    }
}
