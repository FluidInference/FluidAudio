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
            if config.useSimpleStateUpdate {
                // Simple state update matching Python test
                // Just keep the last maxFifoLen frames
                let overflow = state.fifoLength - maxFifoLen
                state.fifo.removeFirst(overflow * fcDModel)
                state.fifoLength = maxFifoLen

                if var fifoPreds = state.fifoPreds {
                    fifoPreds.removeFirst(overflow * numSpeakers)
                    state.fifoPreds = fifoPreds
                }

                // Simple spkcache update every spkcacheUpdatePeriod chunks
                state.chunkCount += 1
                if state.chunkCount % 5 == 0 {
                    // Append some frames to spkcache
                    let framesToAdd = min(chunkLen, maxSpkcacheLen - state.spkcacheLength)
                    if framesToAdd > 0 {
                        let newEmbs = Array(chunk.prefix(framesToAdd * fcDModel))
                        state.spkcache.append(contentsOf: newEmbs)
                        state.spkcacheLength += framesToAdd
                    }

                    // Trim spkcache if over limit
                    if state.spkcacheLength > maxSpkcacheLen {
                        let overflow = state.spkcacheLength - maxSpkcacheLen
                        state.spkcache.removeFirst(overflow * fcDModel)
                        state.spkcacheLength = maxSpkcacheLen
                    }
                }
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
    private func getTopKIndices(
        scores: [Float],
        frameCount: Int,
        numSpeakers: Int,
        k: Int
    ) -> (indices: [Int], isDisabled: [Bool]) {
        let silFramesPerSpk = config.spkcacheSilFramesPerSpk
        let actualFrameCount = frameCount - silFramesPerSpk

        // Flatten scores with frame indices
        var allScores: [(frameIdx: Int, speaker: Int, score: Float)] = []

        for frame in 0..<frameCount {
            for spk in 0..<numSpeakers {
                let score = scores[frame * numSpeakers + spk]
                allScores.append((frame, spk, score))
            }
        }

        // Sort by score descending
        allScores.sort { $0.score > $1.score }

        // Take top-k unique frame indices
        var selectedFrames: [Int] = []
        var isDisabled: [Bool] = []
        var usedFrames = Set<Int>()

        for entry in allScores {
            if selectedFrames.count >= k { break }

            // Skip if already used (but we want all speakers, so just use frame)
            let frameIdx = entry.frameIdx
            if usedFrames.contains(frameIdx) { continue }

            if entry.score == -.infinity || entry.score == .infinity || frameIdx >= actualFrameCount {
                selectedFrames.append(0)  // Placeholder
                isDisabled.append(true)
            } else {
                selectedFrames.append(frameIdx)
                isDisabled.append(false)
                usedFrames.insert(frameIdx)
            }
        }

        // Fill remaining with placeholders
        while selectedFrames.count < k {
            selectedFrames.append(0)
            isDisabled.append(true)
        }

        // Sort to preserve original order
        let sortedWithDisabled = zip(selectedFrames, isDisabled).sorted { $0.0 < $1.0 }
        return (sortedWithDisabled.map { $0.0 }, sortedWithDisabled.map { $0.1 })
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
