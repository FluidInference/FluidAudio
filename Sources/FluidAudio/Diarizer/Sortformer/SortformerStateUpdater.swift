import Accelerate
import Foundation

/// Core streaming logic for Sortformer diarization.
///
/// This mirrors NeMo's StateUpdater class, ported from the default implementation.
/// Reference: NeMo nemo/collections/asr/modules/sortformer_modules.py
public struct SortformerStateUpdater {

    private let logger = AppLogger(category: "SortformerStateUpdater")
    private let config: SortformerConfig

    public init(config: SortformerConfig) {
        self.config = config
    }

    // MARK: - Streaming Update

    /// Update streaming state with new chunk.
    ///
    /// This is the core streaming logic from NeMo's streaming_update(),
    /// ported from the default MLTensor implementation.
    ///
    /// - Parameters:
    ///   - state: Current streaming state (mutated in place)
    ///   - chunk: Chunk embeddings from encoder [leftContext + chunkLen + rightContext, fcDModel] flattened
    ///   - preds: Full predictions [spkcache + fifo + chunkTotalFrames, numSpeakers] flattened
    ///   - leftContext: Left context frames to skip in predictions
    ///   - rightContext: Right context frames (for info only)
    /// - Returns: `StreamingUpdateResult` with confirmed and tentative predictions for this chunk [chunkLen * numSpeakers]
    public func streamingUpdate(
        state: inout SortformerStreamingState,
        chunk: [Float],
        preds: [Float],
        leftContext: Int,
        rightContext: Int
    ) throws -> StreamingUpdateResult {
        let fcDModel = config.preEncoderDims
        let numSpeakers = config.numSpeakers
        let spkcacheCapacity = config.spkcacheLen
        let fifoCapacity = config.fifoLen

        let currentSpkcacheLength = state.spkcacheLength
        let currentFifoLength = state.fifoLength

        // Extract FIFO predictions if FIFO exists
        if currentFifoLength > 0 {
            let fifoPredsStart = currentSpkcacheLength * numSpeakers
            let fifoPredsEnd = (currentSpkcacheLength + currentFifoLength) * numSpeakers
            guard fifoPredsEnd <= preds.count else {
                throw SortformerError.insufficientPredsLength(
                    "Not enough predictions for FIFO in streaming update: \(fifoPredsEnd) > \(preds.count)")
            }
            state.fifoPreds = Array(preds[fifoPredsStart..<fifoPredsEnd])
        }

        // Extract only CORE frames from chunk embeddings (skip left context, take chunkLen frames)
        // This matches the default impl: chunk[0..., lc..<chunkLen+lc, 0...]
        // Use ACTUAL leftContext (varies by chunk position), not fixed config.chunkLeftContext
        let lc = leftContext
        let rc = rightContext
        let coreFrames = (chunk.count / fcDModel) - lc - rc

        // Extract core embeddings only (frames lc..<lc+coreFrames)
        let embsStartIdx = lc * fcDModel
        let embsEndIdx = (lc + coreFrames) * fcDModel
        guard embsEndIdx <= chunk.count else {
            throw SortformerError.insufficientChunkLength(
                "Not enough chunk embeddings for streaming update: \(embsEndIdx) > \(chunk.count)")
        }
        let chunkEmbs = Array(chunk[embsStartIdx..<embsEndIdx])

        // Extract chunk predictions for CORE frames only
        // This matches the default impl: preds[0..., chunkStart+lc..<chunkStart+chunkLen+lc, 0...]
        let chunkStart = currentSpkcacheLength + currentFifoLength + lc
        let chunkEnd = chunkStart + coreFrames

        let chunkPredsStart = chunkStart * numSpeakers
        let chunkPredsEnd = chunkEnd * numSpeakers

        let tentativePredsStart = chunkPredsEnd
        let tentativePredsEnd = (chunkEnd + rc) * numSpeakers

        guard tentativePredsEnd <= preds.count else {
            if chunkPredsEnd > preds.count {
                throw SortformerError.insufficientPredsLength(
                    "Not enough predictions for chunk in streaming update: \(chunkPredsEnd) > \(preds.count)")
            }
            throw SortformerError.insufficientPredsLength(
                "Not enough predictions for tentative predictions in streaming update: \(tentativePredsEnd) > \(preds.count)"
            )
        }
        let chunkPreds: [Float] = Array(preds[chunkPredsStart..<chunkPredsEnd])
        let tentativePreds: [Float] = Array(preds[tentativePredsStart..<tentativePredsEnd])

        // Append chunk core to FIFO
        state.fifo.append(contentsOf: chunkEmbs)
        state.fifoLength += coreFrames

        if state.fifoPreds != nil {
            state.fifoPreds?.append(contentsOf: chunkPreds)
        } else {
            state.fifoPreds = chunkPreds
        }

        // Update speaker cache if FIFO overflows
        // Use actualCoreFrames (not full chunk), matching the default impl: chunkLen + currentFifoLength
        let contextLength = coreFrames + currentFifoLength
        if contextLength > fifoCapacity {
            guard let currentFifoPreds = state.fifoPreds else {
                logger.error(
                    "FIFO predictions are nil immediately after updating them during streaming update. THIS SHOULD NEVER HAPPEN!"
                )
                return StreamingUpdateResult(
                    confirmed: chunkPreds, tentative: tentativePreds, numSpeakers: config.numSpeakers)
            }

            // Calculate how many frames to pop
            var popOutLength = config.spkcacheUpdatePeriod
            popOutLength = max(popOutLength, contextLength - fifoCapacity)
            popOutLength = min(popOutLength, contextLength)

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
                    if currentSpkcacheLength > 0 {
                        state.spkcachePreds = Array(preds.prefix(currentSpkcacheLength * numSpeakers)) + popOutPreds
                    } else {
                        state.spkcachePreds = popOutPreds
                    }
                }

                compressSpkcache(state: &state)
            }
        }

        return StreamingUpdateResult(confirmed: chunkPreds, tentative: tentativePreds, numSpeakers: config.numSpeakers)
    }

    // MARK: - Silence Profile

    /// Update running mean of silence embeddings.
    /// - Parameters:
    ///   - state: Streaming state
    ///   - embs: Frame-wise speaker embeddings  [frameCount, fcDModel] flattened
    ///   - preds: Frame-wise speaker activity predictions  [frameCount, numSpeakers] flattened
    ///   - frameCount: Number of frames
    private func updateSilenceProfile(
        state: inout SortformerStreamingState,
        embs: [Float],
        preds: [Float],
        frameCount: Int
    ) {
        let fcDModel = config.preEncoderDims
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
    /// ported from the default implementation.
    private func compressSpkcache(state: inout SortformerStreamingState) {
        guard let spkcachePreds = state.spkcachePreds else { return }

        let fcDModel = config.preEncoderDims
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

        var tmp = [Float](repeating: 0, count: preds.count)
        var log1P = [Float](repeating: 0, count: preds.count)

        // Scores -> log(p)
        vDSP.clip(preds, to: threshold...Float.greatestFiniteMagnitude, result: &tmp)
        vForce.log(tmp, result: &scores)

        // Scores -> log(p) - log(1-p)
        vDSP.clip(preds, to: 0...(1 - threshold), result: &tmp)
        vDSP.negative(tmp, result: &tmp)
        vForce.log1p(tmp, result: &log1P)
        vDSP.subtract(scores, log1P, result: &scores)

        // Scores -> log(p) - log(1-p) - log(0.5)
        vDSP.add(logf(2), scores, result: &scores)

        // Scores -> log(p) - log(1-p) + sum(log(1-p_others)) - log(0.5)
        scores.withUnsafeMutableBufferPointer { sBuf in
            log1P.withUnsafeBufferPointer { lBuf in
                guard let s = sBuf.baseAddress, let l = lBuf.baseAddress else { return }
                let S = numSpeakers

                for frame in 0..<frameCount {
                    let base = frame &* S
                    var sum: Float = 0
                    for spk in 0..<S { sum += l[base + spk] }
                    for spk in 0..<S { s[base + spk] += sum }
                }
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
                let index = frame * numSpeakers + spk
                if preds[index] > 0.5 && scores[index] > 0 {
                    posScoreCounts[spk] += 1
                }
            }
        }

        for spk in 0..<numSpeakers {
            for frame in 0..<frameCount {
                let idx = frame * numSpeakers + spk
                let p = preds[idx]

                // Disable non-speech (p < 0.5)
                if p <= 0.5 {
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
        let S = config.numSpeakers
        guard frameCount > 0, S > 0, k > 0 else { return scores }

        let boostDelta: Float = -scaleFactor * logf(0.5)  // positive

        var result = scores
        let kEff = min(k, frameCount)

        result.withUnsafeMutableBufferPointer { resBuf in
            guard let base = resBuf.baseAddress else { return }

            for spk in 0..<S {
                // Keep arrays sorted DESC by score: [0] is largest, [count-1] is smallest among kept.
                var topFrames = [Int](repeating: 0, count: kEff)
                var topScores = [Float](repeating: -Float.greatestFiniteMagnitude, count: kEff)
                var count = 0

                for frame in 0..<frameCount {
                    let idx = frame &* S &+ spk
                    let v = base[idx]
                    if v == -.infinity { continue }

                    if count < kEff {
                        // Insert into [0..<count] maintaining DESC order.
                        var pos = count
                        while pos > 0 && v > topScores[pos - 1] {
                            topScores[pos] = topScores[pos - 1]
                            topFrames[pos] = topFrames[pos - 1]
                            pos -= 1
                        }
                        topScores[pos] = v
                        topFrames[pos] = frame
                        count += 1
                    } else {
                        // If v isn't better than the smallest kept, skip.
                        if v <= topScores[count - 1] { continue }

                        // Insert v into the correct position, dropping the last element.
                        var pos = count - 1
                        while pos > 0 && v > topScores[pos - 1] {
                            topScores[pos] = topScores[pos - 1]
                            topFrames[pos] = topFrames[pos - 1]
                            pos -= 1
                        }
                        topScores[pos] = v
                        topFrames[pos] = frame
                    }
                }

                // Apply boost to the top frames we found.
                for i in 0..<count {
                    let idx = topFrames[i] &* S &+ spk
                    base[idx] += boostDelta
                }
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
        let S = config.numSpeakers
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk
        let nFramesNoSil = frameCount - silFramesPerSpk
        let maxIndex = config.maxIndex

        precondition(scores.count >= frameCount * S)
        precondition(frameCount >= 0 && S > 0)

        let N = frameCount * S
        if k <= 0 {
            return ([], [])
        }

        // We'll compute topK over at most N real elements, then pad to k with maxIndex.
        let kEff = min(k, N)

        // Top-k buffers (kept sorted by score DESC; tie-break by smaller index).
        var bestIdx = [Int](repeating: 0, count: kEff)
        var bestVal = [Float](repeating: -.infinity, count: kEff)
        var count = 0

        // Iterate over "permuted-flattened" indices without building the permuted array.
        // permutedIdx = spk*frameCount + frame
        // scoreAt(permutedIdx) = scores[frame*S + spk]
        for spk in 0..<S {
            for frame in 0..<frameCount {
                let permutedIdx = spk * frameCount + frame
                let v = scores[frame * S + spk]

                if count < kEff {
                    // Insert into [0..<count] in descending order (val, then smaller index).
                    var pos = count
                    while pos > 0 {
                        let pv = bestVal[pos - 1]
                        let pi = bestIdx[pos - 1]
                        if v > pv || (v == pv && permutedIdx < pi) {
                            bestVal[pos] = pv
                            bestIdx[pos] = pi
                            pos -= 1
                        } else {
                            break
                        }
                    }
                    bestVal[pos] = v
                    bestIdx[pos] = permutedIdx
                    count += 1
                } else {
                    // Compare against the current worst kept (last element).
                    let worstV = bestVal[kEff - 1]
                    let worstI = bestIdx[kEff - 1]
                    if v < worstV || (v == worstV && permutedIdx >= worstI) {
                        continue
                    }

                    // Insert v, drop the last.
                    var pos = kEff - 1
                    while pos > 0 {
                        let pv = bestVal[pos - 1]
                        let pi = bestIdx[pos - 1]
                        if v > pv || (v == pv && permutedIdx < pi) {
                            bestVal[pos] = pv
                            bestIdx[pos] = pi
                            pos -= 1
                        } else {
                            break
                        }
                    }
                    bestVal[pos] = v
                    bestIdx[pos] = permutedIdx
                }
            }
        }

        // Build topKIndices (length k), padding with maxIndex if k > N
        var topKIndices = [Int](repeating: maxIndex, count: k)
        for i in 0..<kEff {
            topKIndices[i] = (bestVal[i] == -.infinity) ? maxIndex : bestIdx[i]
        }

        // Sort indices ascending (matches your "preserve original order" step)
        topKIndices.sort()

        // Compute isDisabled BEFORE modulo conversion
        var isDisabled = [Bool](repeating: false, count: k)
        for i in 0..<k {
            if topKIndices[i] == maxIndex {
                isDisabled[i] = true
            }
        }

        // Convert flattened permuted idx -> frame idx via modulo
        for i in 0..<k where !isDisabled[i] {
            topKIndices[i] = topKIndices[i] % frameCount
        }

        // Disable frames beyond actual content
        for i in 0..<k where !isDisabled[i] {
            if topKIndices[i] >= nFramesNoSil {
                isDisabled[i] = true
            }
        }

        // Set placeholder index for disabled frames
        for i in 0..<k where isDisabled[i] {
            topKIndices[i] = 0
        }

        return (topKIndices, isDisabled)
    }

    // MARK: - Sigmoid

    /// Apply sigmoid to convert logits to probabilities.
    public func applySigmoid(_ logits: [Float]) -> [Float] {
        return logits.map { 1.0 / (1.0 + exp(-$0)) }
    }
}
