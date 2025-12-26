import Accelerate
import Foundation
import OSLog

/// Core streaming logic for Sortformer diarization.
///
/// This mirrors NeMo's SortformerModules class.
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

    // MARK: - Streaming Update (Synchronous)

    /// Update streaming state with new chunk.
    ///
    /// This is the core streaming logic from NeMo's streaming_update().
    ///
    /// - Parameters:
    ///   - state: Current streaming state (mutated in place)
    ///   - chunkEmbeddings: Chunk embeddings from encoder [chunkLen, fcDModel]
    ///   - predictions: Full predictions [spkcacheLen + fifoLen + chunkLen, numSpeakers]
    ///   - leftContext: Left context frames to skip
    ///   - rightContext: Right context frames to skip
    ///   - modelSpkcacheLen: Spkcache length passed to model (may differ from state due to padding)
    ///   - modelFifoLen: Fifo length passed to model (may differ from state due to padding)
    /// - Returns: Updated state and chunk predictions [chunkLen, numSpeakers]
    public func streamingUpdate(
        state: inout SortformerStreamingState,
        chunkEmbeddings: [Float],
        predictions: [Float],
        leftContext: Int,
        rightContext: Int,
        modelSpkcacheLen: Int? = nil,
        modelFifoLen: Int? = nil
    ) -> [Float] {
        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let maxFifoLen = config.fifoLen
        let maxSpkcacheLen = config.spkcacheLen

        // Actual content lengths for state management
        let spkcacheLen = modelSpkcacheLen ?? state.spkcacheLength
        let fifoLen = modelFifoLen ?? state.fifoLength

        // Derive chunk_len from embedding tensor shape, exactly like NeMo:
        // chunk_len = chunk.shape[1] - lc - rc
        // This is crucial for correct prediction extraction, especially for chunk 0
        // where the embedding count differs due to padding.
        let embeddingCount = chunkEmbeddings.count / fcDModel
        let chunkLen = embeddingCount - leftContext - rightContext

        // Use ACTUAL state content lengths for prediction offset (matching NeMo's streaming_update)
        // NOTE: spkcacheLen/fifoLen may be 1+ due to max(1,...) for CoreML inputs,
        // but we should use the TRUE state lengths here.
        // The model outputs valid predictions only for the first (actualSpkcache + actualFifo + chunk) frames
        let actualSpkcacheLen = state.spkcacheLength
        let actualFifoLen = state.fifoLength
        let predOffset = (actualSpkcacheLen + actualFifoLen + leftContext) * numSpeakers
        let chunkPredCount = chunkLen * numSpeakers
        var chunkPreds = [Float](repeating: 0.0, count: chunkPredCount)

        for i in 0..<chunkPredCount {
            if predOffset + i < predictions.count {
                chunkPreds[i] = predictions[predOffset + i]
            }
        }

        // Extract chunk embeddings (skip left context)
        let chunkStartIdx = leftContext * fcDModel
        var chunk = [Float](repeating: 0.0, count: chunkLen * fcDModel)

        for i in 0..<(chunkLen * fcDModel) {
            if chunkStartIdx + i < chunkEmbeddings.count {
                chunk[i] = chunkEmbeddings[chunkStartIdx + i]
            }
        }

        // Update FIFO predictions (using ACTUAL state length for offset, matching NeMo)
        // NeMo: streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        let fifoPredsOffset = actualSpkcacheLen * numSpeakers
        var newFifoPreds = [Float](repeating: 0.0, count: actualFifoLen * numSpeakers)
        for i in 0..<(actualFifoLen * numSpeakers) {
            if fifoPredsOffset + i < predictions.count {
                newFifoPreds[i] = predictions[fifoPredsOffset + i]
            }
        }
        state.fifoPreds = newFifoPreds

        // Append chunk to FIFO
        state.fifo.append(contentsOf: chunk)
        state.fifoLength += chunkLen

        if var fifoPreds = state.fifoPreds {
            fifoPreds.append(contentsOf: chunkPreds)
            state.fifoPreds = fifoPreds
        } else {
            state.fifoPreds = chunkPreds
        }

        // Check if FIFO overflows
        if state.fifoLength > maxFifoLen {
            // Debug: trace overflow handling
            if config.debugMode {
                print("[State] FIFO overflow: fifoLen=\(state.fifoLength) > max=\(maxFifoLen)")
            }

            if config.useSimpleStateUpdate {
                // Simple sliding window: keep only most recent MAX_FIFO frames
                // This matches Python's simple state behavior exactly:
                // - No spkcache updates (spkcache stays empty/at 0)
                // - FIFO acts as a sliding window of recent context
                let overflow = state.fifoLength - maxFifoLen
                state.fifo.removeFirst(overflow * fcDModel)
                state.fifoLength = maxFifoLen

                if var fifoPreds = state.fifoPreds {
                    fifoPreds.removeFirst(overflow * numSpeakers)
                    state.fifoPreds = fifoPreds
                }

                // No spkcache updates - keep it empty for simple mode
                // This avoids the complexity of compression and matches
                // the Python baseline that achieves good DER
            } else {
                // Full NeMo state update logic
                // Calculate how many frames to pop
                var popOutLen = config.spkcacheUpdatePeriod
                popOutLen = max(popOutLen, chunkLen - maxFifoLen + fifoLen)
                popOutLen = min(popOutLen, state.fifoLength)

                // Pop embeddings from FIFO
                let popOutEmbs = Array(state.fifo.prefix(popOutLen * fcDModel))
                let popOutPreds = state.fifoPreds.map { Array($0.prefix(popOutLen * numSpeakers)) } ?? []

                // Update silence profile
                updateSilenceProfile(
                    state: &state,
                    embeddings: popOutEmbs,
                    predictions: popOutPreds,
                    frameCount: popOutLen
                )

                // Remove popped frames from FIFO
                state.fifo.removeFirst(popOutLen * fcDModel)
                state.fifoLength -= popOutLen

                if var fifoPreds = state.fifoPreds {
                    fifoPreds.removeFirst(popOutLen * numSpeakers)
                    state.fifoPreds = fifoPreds
                }

                // Append popped embeddings to speaker cache
                state.spkcache.append(contentsOf: popOutEmbs)
                state.spkcacheLength += popOutLen

                // Update speaker cache predictions
                if state.spkcachePreds != nil {
                    state.spkcachePreds?.append(contentsOf: popOutPreds)
                } else if state.spkcacheLength > maxSpkcacheLen {
                    // First time spkcache overflows - initialize predictions
                    let spkcachePredCount = spkcacheLen * numSpeakers
                    var spkcachePreds = [Float](repeating: 0.0, count: spkcachePredCount)
                    for i in 0..<spkcachePredCount {
                        if i < predictions.count {
                            spkcachePreds[i] = predictions[i]
                        }
                    }
                    spkcachePreds.append(contentsOf: popOutPreds)
                    state.spkcachePreds = spkcachePreds
                }

                // Compress speaker cache if it overflows
                if state.spkcacheLength > maxSpkcacheLen {
                    compressSpkcache(state: &state)
                }
            }
        }

        return chunkPreds
    }

    // MARK: - Speaker Cache Compression

    /// Compress speaker cache to keep most important frames.
    ///
    /// This mirrors NeMo's _compress_spkcache() function.
    private func compressSpkcache(state: inout SortformerStreamingState) {
        guard let spkcachePreds = state.spkcachePreds else { return }

        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let maxSpkcacheLen = config.spkcacheLen
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk

        let currentLen = state.spkcacheLength
        let spkcacheLenPerSpk = maxSpkcacheLen / numSpeakers - silFramesPerSpk

        // Calculate scores for each frame
        var scores = computeLogPredScores(
            predictions: spkcachePreds,
            frameCount: currentLen,
            numSpeakers: numSpeakers
        )

        // Disable low scores (non-speech, overlapped)
        disableLowScores(
            predictions: spkcachePreds,
            scores: &scores,
            frameCount: currentLen,
            numSpeakers: numSpeakers,
            minPosScoresPerSpk: Int(Float(spkcacheLenPerSpk) * config.minPosScoresRate)
        )

        // Boost latest frames
        if config.scoresBoostLatest > 0 {
            let boostStart = maxSpkcacheLen
            for frame in boostStart..<currentLen {
                for spk in 0..<numSpeakers {
                    scores[frame * numSpeakers + spk] += config.scoresBoostLatest
                }
            }
        }

        // Strong boost to ensure each speaker has K frames
        let strongBoostPerSpk = Int(Float(spkcacheLenPerSpk) * config.strongBoostRate)
        boostTopKScores(
            scores: &scores,
            frameCount: currentLen,
            numSpeakers: numSpeakers,
            k: strongBoostPerSpk,
            scaleFactor: 2.0
        )

        // Weak boost to prevent dominance
        let weakBoostPerSpk = Int(Float(spkcacheLenPerSpk) * config.weakBoostRate)
        boostTopKScores(
            scores: &scores,
            frameCount: currentLen,
            numSpeakers: numSpeakers,
            k: weakBoostPerSpk,
            scaleFactor: 1.0
        )

        // Add silence frames placeholder
        let totalFramesWithSil = currentLen + silFramesPerSpk
        for _ in 0..<(silFramesPerSpk * numSpeakers) {
            scores.append(Float.infinity)
        }

        // Get top-k indices
        let (topkIndices, isDisabled) = getTopKIndices(
            scores: scores,
            frameCount: totalFramesWithSil,
            numSpeakers: numSpeakers,
            k: maxSpkcacheLen
        )

        // Gather compressed embeddings and predictions
        var newSpkcache = [Float](repeating: 0.0, count: maxSpkcacheLen * fcDModel)
        var newSpkcachePreds = [Float](repeating: 0.0, count: maxSpkcacheLen * numSpeakers)

        for (i, frameIdx) in topkIndices.enumerated() {
            if isDisabled[i] {
                // Use mean silence embedding
                for d in 0..<fcDModel {
                    newSpkcache[i * fcDModel + d] = state.meanSilenceEmbedding[d]
                }
                // Zero predictions for silence
            } else if frameIdx < currentLen {
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
        state.spkcacheLength = maxSpkcacheLen
        state.spkcachePreds = newSpkcachePreds
    }

    // MARK: - Score Computation

    /// Compute log-based prediction scores.
    ///
    /// Score = log(p) - log(1-p) + sum(log(1-p_others)) - log(0.5)
    private func computeLogPredScores(
        predictions: [Float],
        frameCount: Int,
        numSpeakers: Int
    ) -> [Float] {
        let threshold = config.predScoreThreshold
        var scores = [Float](repeating: 0.0, count: frameCount * numSpeakers)

        for frame in 0..<frameCount {
            // Compute sum of log(1-p) for all speakers
            var log1ProbsSum: Float = 0.0
            for spk in 0..<numSpeakers {
                let p = predictions[frame * numSpeakers + spk]
                log1ProbsSum += log(max(1.0 - p, threshold))
            }

            for spk in 0..<numSpeakers {
                let p = predictions[frame * numSpeakers + spk]
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
        predictions: [Float],
        scores: inout [Float],
        frameCount: Int,
        numSpeakers: Int,
        minPosScoresPerSpk: Int
    ) {
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
                let p = predictions[idx]

                // Disable non-speech (p <= 0.5)
                if p <= 0.5 {
                    scores[idx] = -.infinity
                    continue
                }

                // Disable non-positive scores if speaker has enough positive scores
                if scores[idx] <= 0 && posScoreCounts[spk] >= minPosScoresPerSpk {
                    scores[idx] = -.infinity
                }
            }
        }
    }

    /// Boost top-k scores for each speaker.
    private func boostTopKScores(
        scores: inout [Float],
        frameCount: Int,
        numSpeakers: Int,
        k: Int,
        scaleFactor: Float
    ) {
        let boostValue = scaleFactor * log(0.5)

        for spk in 0..<numSpeakers {
            // Get scores for this speaker
            var speakerScores: [(Int, Float)] = []
            for frame in 0..<frameCount {
                let idx = frame * numSpeakers + spk
                if scores[idx] != -.infinity {
                    speakerScores.append((frame, scores[idx]))
                }
            }

            // Sort by score descending
            speakerScores.sort { $0.1 > $1.1 }

            // Boost top-k
            for i in 0..<min(k, speakerScores.count) {
                let frame = speakerScores[i].0
                scores[frame * numSpeakers + spk] -= boostValue
            }
        }
    }

    /// Get top-k frame indices based on scores.
    ///
    /// This mirrors NeMo's _get_topk_indices() exactly:
    /// - Permutes scores from (frames, speakers) to (speakers, frames)
    /// - Flattens and takes top-k indices
    /// - Allows the same frame to appear multiple times (for different speakers)
    /// - Uses modulo to convert back to frame indices
    private func getTopKIndices(
        scores: [Float],
        frameCount: Int,
        numSpeakers: Int,
        k: Int
    ) -> (indices: [Int], isDisabled: [Bool]) {
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk
        let nFramesNoSil = frameCount - silFramesPerSpk
        let maxIndex = config.maxIndex

        // NeMo: scores.permute(0, 2, 1).reshape(batch_size, -1)
        // scores is (frames, speakers), permute to (speakers, frames), then flatten
        // Input: scores[frame * numSpeakers + spk]
        // After permute: (spk, frame) order in flattened array
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
        var topkIndices = [Int](repeating: 0, count: k)
        var topkValues = [Float](repeating: 0.0, count: k)

        for i in 0..<k {
            if i < sortedByScore.count {
                topkIndices[i] = sortedByScore[i].0
                topkValues[i] = sortedByScore[i].1
            } else {
                topkIndices[i] = maxIndex
                topkValues[i] = -.infinity
            }
        }

        // Replace -inf indices with maxIndex placeholder
        for i in 0..<k {
            if topkValues[i] == -.infinity {
                topkIndices[i] = maxIndex
            }
        }

        // Sort indices to preserve original order
        let sortedPairs = topkIndices.enumerated().sorted { $0.element < $1.element }
        var topkIndicesSorted = sortedPairs.map { $0.element }

        // Compute isDisabled BEFORE converting to frame indices
        var isDisabled = [Bool](repeating: false, count: k)
        for i in 0..<k {
            if topkIndicesSorted[i] == maxIndex {
                isDisabled[i] = true
            }
        }

        // NeMo: topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        // Convert flattened index to frame index
        for i in 0..<k {
            if !isDisabled[i] {
                topkIndicesSorted[i] = topkIndicesSorted[i] % frameCount
            }
        }

        // Mark frames beyond actual content as disabled (silence padding frames)
        for i in 0..<k {
            if !isDisabled[i] && topkIndicesSorted[i] >= nFramesNoSil {
                isDisabled[i] = true
            }
        }

        // Set placeholder index for disabled frames
        for i in 0..<k where isDisabled[i] {
            topkIndicesSorted[i] = 0
        }

        return (topkIndicesSorted, isDisabled)
    }

    // MARK: - Silence Profile

    /// Update running mean of silence embeddings.
    private func updateSilenceProfile(
        state: inout SortformerStreamingState,
        embeddings: [Float],
        predictions: [Float],
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
                if idx < predictions.count {
                    probSum += predictions[idx]
                }
            }

            if probSum < silThreshold {
                // Update running mean
                let n = Float(state.silenceFrameCount)
                let newN = n + 1.0

                for d in 0..<fcDModel {
                    let embIdx = frame * fcDModel + d
                    if embIdx < embeddings.count {
                        let oldMean = state.meanSilenceEmbedding[d]
                        let newVal = embeddings[embIdx]
                        state.meanSilenceEmbedding[d] = (oldMean * n + newVal) / newN
                    }
                }

                state.silenceFrameCount += 1
            }
        }
    }

    // MARK: - Sigmoid

    /// Apply sigmoid to convert logits to probabilities.
    public func applySigmoid(_ logits: [Float]) -> [Float] {
        return logits.map { 1.0 / (1.0 + exp(-$0)) }
    }
}
