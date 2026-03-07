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
        leftContext lc: Int,
        rightContext rc: Int
    ) throws -> SortformerStateUpdateResult {
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
        let coreFrames = (chunk.count / fcDModel) - lc - rc

        // Extract core embeddings only (frames lc..<lc+coreFrames)
        let embsStartIdx = lc * fcDModel
        let embsEndIdx = (lc + coreFrames) * fcDModel
        guard embsEndIdx <= chunk.count else {
            throw SortformerError.insufficientChunkLength(
                "Not enough chunk embeddings for streaming update: \(embsEndIdx) > \(chunk.count)")
        }
        let coreEmbs = Array(chunk[embsStartIdx..<embsEndIdx])

        // Extract chunk predictions for CORE frames only
        // This matches the default impl: preds[0..., chunkStart+lc..<chunkStart+chunkLen+lc, 0...]
        let coreStart = currentSpkcacheLength + currentFifoLength + lc
        let coreEnd = coreStart + coreFrames
        let newStart = coreStart + state.lastRightContext
        let newEnd = coreEnd + rc

        let corePredsStart = coreStart * numSpeakers
        let corePredsEnd = coreEnd * numSpeakers
        let newPredsStart = newStart * numSpeakers
        let newPredsEnd = newEnd * numSpeakers

        guard newPredsEnd <= preds.count else {
            throw SortformerError.insufficientPredsLength(
                "Not enough predictions for chunk + right context in streaming update: \(newPredsEnd) > \(preds.count)")
        }
        let corePreds: [Float] = Array(preds[corePredsStart..<corePredsEnd])

        let newPreds: [Float] = Array(preds[newPredsStart..<newPredsEnd])
        let oldPreds: [Float] = state.fifoPreds + preds[corePredsStart..<newPredsStart]
        
        // Append chunk core to FIFO
        state.fifo.append(contentsOf: coreEmbs)
        state.fifoPreds.append(contentsOf: corePreds)
        state.fifoLength += coreFrames

        // Update speaker cache if FIFO overflows
        // Use actualCoreFrames (not full chunk), matching the default impl: chunkLen + currentFifoLength
        var popOutLength: Int = 0
        let contextLength = coreFrames + currentFifoLength
        
        if contextLength > fifoCapacity {
            // Calculate how many frames to pop
            popOutLength = config.spkcacheUpdatePeriod
            popOutLength = max(popOutLength, contextLength - fifoCapacity)
            popOutLength = min(popOutLength, contextLength)

            // Extract frames to pop from FIFO
            let popOutEmbs = Array(state.fifo.prefix(popOutLength * fcDModel))
            let popOutPreds = Array(state.fifoPreds.prefix(popOutLength * numSpeakers))

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
            state.fifoPreds.removeFirst(popOutLength * numSpeakers)

            // Append popped embeddings to speaker cache
            state.spkcache.append(contentsOf: popOutEmbs)
            state.spkcacheLength += popOutLength

            // Update speaker cache predictions
            if !state.spkcachePreds.isEmpty {
                state.spkcachePreds.append(contentsOf: popOutPreds)
            }

            // Compress speaker cache if it overflows
            if state.spkcacheLength > spkcacheCapacity {
                if state.spkcachePreds.isEmpty {
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

        defer {
            state.lastRightContext = rc
            state.nextNewFrame += rc
        }
        
        return SortformerStateUpdateResult(
            firstNewFrame: state.nextNewFrame,
            newPredictions: newPreds,
            newFrameCount: newEnd - newStart,
            oldPredictions: oldPreds,
            oldFrameCount: currentFifoLength + state.lastRightContext,
            finalizedFrameCount: popOutLength
        )
    }

    // MARK: - Remove Speaker

    /// Remove a speaker from the FIFO and speaker cache.
    ///
    /// This erases all trace of a speaker slot from the streaming state:
    /// - **Solo frames** (speaker is the only one active above `silenceThreshold`):
    ///   The embedding is replaced with `meanSilenceEmbedding` and all speaker
    ///   predictions for that frame are zeroed, effectively converting it to silence.
    /// - **Overlap frames** (speaker is active alongside one other speaker):
    ///   The target speaker's prediction is zeroed, and the embedding is
    ///   replaced by cycling through the other speaker's solo-frame embeddings,
    ///   preserving the acoustic temporal structure. Falls back to leaving the
    ///   embedding unchanged if the other speaker has no solo frames.
    /// - **Multi-overlap frames** (speaker is active alongside 2+ others):
    ///   Only the target prediction is zeroed. No clean substitute embedding
    ///   exists for this case; the entangled embedding still serves the others.
    /// - **Inactive frames**: The target speaker's prediction column is zeroed everywhere
    ///   to ensure full suppression even for sub-threshold activations.
    ///
    /// - Parameters:
    ///   - speakerIndex: The speaker slot to remove (0..<numSpeakers)
    ///   - state: Streaming state (mutated in place)
//    public func removeSpeaker(
//        at speakerIndex: Int,
//        from state: inout SortformerStreamingState
//    ) {
//        let D = config.preEncoderDims
//        let S = config.numSpeakers
//        let threshold = config.speechThreshold
//
//        guard speakerIndex >= 0, speakerIndex < S else { return }
//
//        // Concatenate spkcache + fifo into a single buffer so solo frames in
//        // either region can serve as sources for overlap frames in the other.
//        let spkLen = state.spkcacheLength
//        let fifoLen = state.fifoLength
//        let hasSpkcachePreds = !state.spkcachePreds.isEmpty && spkLen > 0
//
//        let totalFrames: Int
//        var embs: [Float]
//        var preds: [Float]
//
//        if hasSpkcachePreds {
//            totalFrames = spkLen + fifoLen
//            embs = state.spkcache + state.fifo
//            preds = state.spkcachePreds + state.fifoPreds
//        } else {
//            totalFrames = fifoLen
//            embs = state.fifo
//            preds = state.fifoPreds
//        }
//
//        guard totalFrames > 0, !preds.isEmpty else { return }
//
//        // --- Pass 1: Collect solo frame indices per speaker ---
//        var soloFrames = [[Int]](repeating: [], count: S)
//
//        preds.withUnsafeBufferPointer { predsBuf in
//            let p = predsBuf.baseAddress!
//            for frame in 0..<totalFrames {
//                let base = frame &* S
//                var activeCount = 0
//                var activeSpeaker = -1
//                for spk in 0..<S {
//                    if p[base &+ spk] >= threshold {
//                        activeCount &+= 1
//                        activeSpeaker = spk
//                    }
//                }
//                if activeCount == 1 {
//                    soloFrames[activeSpeaker].append(frame)
//                }
//            }
//        }
//
//        var loopPos = [Int](repeating: 0, count: S)
//
//        // --- Pass 2: Scrub frames ---
//        // Solo frames of other speakers only have their target pred zeroed
//        // (embedding untouched), so reading from them for overlap copy is safe.
//        embs.withUnsafeMutableBufferPointer { embsBuf in
//            preds.withUnsafeMutableBufferPointer { predsBuf in
//                state.meanSilenceEmbedding.withUnsafeBufferPointer { silBuf in
//                    let e = embsBuf.baseAddress!
//                    let p = predsBuf.baseAddress!
//                    let sil = silBuf.baseAddress!
//
//                    for frame in 0..<totalFrames {
//                        let predBase = frame &* S
//                        let embBase = frame &* D
//                        let targetActive = p[predBase &+ speakerIndex] >= threshold
//
//                        // Count other active speakers without heap allocation
//                        var otherCount = 0
//                        var firstOtherSpk = 0
//                        if targetActive {
//                            for spk in 0..<S where spk != speakerIndex {
//                                if p[predBase &+ spk] >= threshold {
//                                    otherCount &+= 1
//                                    if otherCount == 1 { firstOtherSpk = spk }
//                                }
//                            }
//                        }
//
//                        if targetActive && otherCount == 0 {
//                            // Solo target → bulk-copy silence embedding, zero preds
//                            (e + embBase).update(from: sil, count: D)
//                            vDSP_vclr(p + predBase, 1, vDSP_Length(S))
//
//                        } else if targetActive && otherCount == 1 {
//                            // Overlap with one speaker → cycle their solo frames
//                            let soloCount = soloFrames[firstOtherSpk].count
//                            if soloCount > 0 {
//                                let srcFrame = soloFrames[firstOtherSpk][loopPos[firstOtherSpk] % soloCount]
//                                loopPos[firstOtherSpk] &+= 1
//                                let srcBase = srcFrame &* D
//                                (e + embBase).update(from: e + srcBase, count: D)
//                            }
//                            p[predBase &+ speakerIndex] = 0
//
//                        } else {
//                            // Multi-overlap or inactive → zero target pred only
//                            p[predBase &+ speakerIndex] = 0
//                        }
//                    }
//                }
//            }
//        }
//
//        // --- Pass 3: Compact prediction columns (vectorized) ---
//        // Each speaker slot is a strided column (stride = S).  Use cblas_scopy
//        // to shift entire columns in one BLAS call, then vDSP_vclr with stride
//        // to zero the vacated last column.
//        // e.g. remove slot 1 → [spk0, spk2, spk3, 0]
//        let slotsToMove = S - speakerIndex - 1
//        if slotsToMove > 0 {
//            preds.withUnsafeMutableBufferPointer { predsBuf in
//                let p = predsBuf.baseAddress!
//                let n = Int32(totalFrames)
//                let stride = Int32(S)
//
//                // Shift each column left by one (process in ascending order so
//                // column k-1 is already saved before column k overwrites it).
//                for k in speakerIndex + 1..<S {
//                    cblas_scopy(n, p + k, stride, p + k - 1, stride)
//                }
//
//                // Zero the vacated last column
//                vDSP_vclr(p + (S - 1), vDSP_Stride(S), vDSP_Length(totalFrames))
//            }
//        }
//
//        // --- Split back into spkcache + fifo ---
//        if hasSpkcachePreds {
//            let spkEmbEnd = spkLen &* D
//            let spkPredEnd = spkLen &* S
//            state.spkcache = Array(embs[..<spkEmbEnd])
//            state.spkcachePreds = Array(preds[..<spkPredEnd])
//            state.fifo = Array(embs[spkEmbEnd...])
//            state.fifoPreds = Array(preds[spkPredEnd...])
//        } else {
//            state.fifo = embs
//            state.fifoPreds = preds
//        }
//    }
    
    /// Remove a speaker from the FIFO and speaker cache.
    ///
    /// This erases all trace of a speaker slot from the streaming state:
    /// - **Solo frames** (speaker is the only one active above `silenceThreshold`):
    ///   The embedding is replaced with `meanSilenceEmbedding` and all speaker
    ///   predictions for that frame are zeroed, effectively converting it to silence.
    /// - **Overlap frames** (speaker is active alongside one other speaker):
    ///   The target speaker's prediction is zeroed, and the embedding is
    ///   replaced by cycling through the other speaker's solo-frame embeddings,
    ///   preserving the acoustic temporal structure. Falls back to leaving the
    ///   embedding unchanged if the other speaker has no solo frames.
    /// - **Multi-overlap frames** (speaker is active alongside 2+ others):
    ///   Only the target prediction is zeroed. No clean substitute embedding
    ///   exists for this case; the entangled embedding still serves the others.
    /// - **Inactive frames**: The target speaker's prediction column is zeroed everywhere
    ///   to ensure full suppression even for sub-threshold activations.
    ///
    /// - Parameters:
    ///   - speakerIndex: The speaker slot to remove (0..<numSpeakers)
    ///   - state: Streaming state (mutated in place)
    /// - Returns: The removed solo embeddings
    @discardableResult
    public func removeSpeaker(
        at speakerIndex: Int,
        from state: inout SortformerStreamingState
    ) -> [Float] {
        let D = config.preEncoderDims
        let S = config.numSpeakers
        let threshold = config.speechThreshold

        guard speakerIndex >= 0, speakerIndex < S else { return [] }

        // Concatenate spkcache + fifo into a single buffer so solo frames in
        // either region can serve as sources for overlap frames in the other.
        let spkLen = state.spkcacheLength
        let fifoLen = state.fifoLength
        let hasSpkcachePreds = !state.spkcachePreds.isEmpty && spkLen > 0

        let totalFrames: Int
        var embs: [Float]
        var preds: [Float]

        if hasSpkcachePreds {
            totalFrames = spkLen + fifoLen
            embs = state.spkcache + state.fifo
            preds = state.spkcachePreds + state.fifoPreds
        } else {
            totalFrames = fifoLen
            embs = state.fifo
            preds = state.fifoPreds
        }

        guard totalFrames > 0, !preds.isEmpty else { return [] }

        // Pass 1: Build powerset
        var powersetFrames = [[Int]](repeating: [], count: 1 << S)

        preds.withUnsafeBufferPointer { predsBuf in
            let p = predsBuf.baseAddress!
            for frame in 0..<totalFrames {
                let base = frame &* S
                var powersetIndex = 0
                
                for spk in 0..<S {
                    if p[base &+ spk] >= threshold {
                        powersetIndex |= 1 << spk
                    }
                }
                powersetFrames[powersetIndex].append(frame)
            }
        }

        var removedEmbeddings: [Float] = []
        embs.withUnsafeMutableBufferPointer { embsBuf in
            guard let embsBase = embsBuf.baseAddress else { return }
            
            // Replace the removed speaker with silence embeddings
            let removedIndex: Int = 1 << speakerIndex
            removedEmbeddings.reserveCapacity(D * powersetFrames[removedIndex].count)
            
            for frame in powersetFrames[removedIndex] {
                let startIndex = frame * D
                removedEmbeddings.append(contentsOf: embsBuf[startIndex..<(startIndex + D)])
                
                memcpy(embsBase + startIndex,
                       &state.meanSilenceEmbedding,
                       MemoryLayout<Float>.stride * D)
            }
            
            // Fill the remainder of frames that intersected with the removed speaker using the powerset embeddings where possible. Loop-pad them if needed
            for powersetIndex in powersetFrames.indices where powersetIndex & removedIndex != 0 {
                let dstFrames = powersetFrames[powersetIndex]
                guard !dstFrames.isEmpty else { continue }
                guard powersetIndex != removedIndex else { continue }
                
                let srcIndex = powersetIndex ^ removedIndex
                let srcFrames = powersetFrames[srcIndex]
                let srcCount = srcFrames.count
                
                if !srcFrames.isEmpty {
                    for (i, frame) in dstFrames.enumerated() {
                        memcpy(embsBase + frame * D,
                               embsBase + srcFrames[i % srcCount] * D,
                               MemoryLayout<Float>.stride * D)
                    }
                } else {
                    // 1. Pick the smallest subset of the powerset whose union makes the source
                    var availableMasks: [Int] = []
                    for m in 1..<powersetFrames.count {
                        if (m & srcIndex) == m && !powersetFrames[m].isEmpty {
                            availableMasks.append(m)
                        }
                    }
                    
                    // DP to find the minimum components required to form each subset
                    var dp: [Int: [Int]] = [0: []]
                    for mask in availableMasks {
                        var updates: [Int: [Int]] = [:]
                        for (existingMask, components) in dp {
                            let newMask = existingMask | mask
                            let newComponents = components + [mask]
                            
                            if let current = dp[newMask] ?? updates[newMask] {
                                if newComponents.count < current.count {
                                    updates[newMask] = newComponents
                                }
                            } else {
                                updates[newMask] = newComponents
                            }
                        }
                        for (k, v) in updates {
                            if let current = dp[k] {
                                if v.count < current.count { dp[k] = v }
                            } else {
                                dp[k] = v
                            }
                        }
                    }
                    
                    var bestCombo = dp[srcIndex]
                    
                    // 2. If no such subset is found, then just try to get as many of the speakers as possible
                    if bestCombo == nil {
                        var bestMask = 0
                        for (mask, components) in dp {
                            // Maximize coverage (number of bits set). If tied, minimize the number of subsets used.
                            if mask.nonzeroBitCount > bestMask.nonzeroBitCount {
                                bestMask = mask
                                bestCombo = components
                            } else if mask.nonzeroBitCount == bestMask.nonzeroBitCount {
                                if components.count < (dp[bestMask]?.count ?? Int.max) {
                                    bestMask = mask
                                    bestCombo = components
                                }
                            }
                        }
                    }
                    
                    // 3. Use the sum of the embeddings to replace the embeddings
                    guard let combo = bestCombo, !combo.isEmpty else {
                        // Absolute fallback: Silence if no overlap source components exist whatsoever
                        for frame in dstFrames {
                            memcpy(embsBase + frame * D,
                                   &state.meanSilenceEmbedding,
                                   MemoryLayout<Float>.stride * D)
                        }
                        continue
                    }
                    
                    var interpBuffer = [Float](repeating: 0, count: D)
                    interpBuffer.withUnsafeMutableBufferPointer { interpBuffer in
                        guard let interpBase = interpBuffer.baseAddress else {
                            return
                        }
                        
                        for (i, frame) in dstFrames.enumerated() {
                            vDSP_vclr(interpBase, 1, vDSP_Length(D))
                            
                            // Generate random weights from a uniform distribution [0, 1]
                            // Yes, this is apparently the best way to blend it.
                            var weights = combo.map { _ in Float.random(in: 0...1) }
                            let totalWeight = weights.reduce(0, +)
                            
                            // Normalize weights so they sum to exactly 1.0
                            if totalWeight > 0 {
                                weights = weights.map { $0 / totalWeight }
                            } else {
                                let equalWeight = 1.0 / Float(combo.count)
                                weights = weights.map { _ in equalWeight }
                            }
                            
                            for (j, srcMask) in combo.enumerated() {
                                let targetSrcFrames = powersetFrames[srcMask]
                                let srcFrame = targetSrcFrames[i % targetSrcFrames.count]
                                var weight = weights[j]
                                
                                // Vectorized scalar multiply and add: interpBuffer += srcEmbedding * weight
                                vDSP_vsma(embsBase + srcFrame * D, 1,
                                          &weight,
                                          interpBase, 1,
                                          interpBase, 1,
                                          vDSP_Length(D))
                            }
                            
                            memcpy(embsBase + frame * D,
                                   interpBase,
                                   MemoryLayout<Float>.stride * D)
                        }
                    }
                }
            }
        }
        
        // --- Pass 3: Compact prediction columns (vectorized) ---
        // Each speaker slot is a strided column (stride = S).  Use cblas_scopy
        // to shift entire columns in one BLAS call, then vDSP_vclr with stride
        // to zero the vacated last column.
        // e.g. remove slot 1 → [spk0, spk2, spk3, 0]
        let slotsToMove = S - speakerIndex - 1
        if slotsToMove > 0 {
            preds.withUnsafeMutableBufferPointer { predsBuf in
                let p = predsBuf.baseAddress!
                let n = Int32(totalFrames)
                let stride = Int32(S)

                // Shift each column left by one (process in ascending order so
                // column k-1 is already saved before column k overwrites it).
                for k in speakerIndex + 1..<S {
                    cblas_scopy(n, p + k, stride, p + k - 1, stride)
                }

                // Zero the vacated last column
                vDSP_vclr(p + (S - 1), vDSP_Stride(S), vDSP_Length(totalFrames))
            }
        }

        // --- Split back into spkcache + fifo ---
        if hasSpkcachePreds {
            let spkEmbEnd = spkLen &* D
            let spkPredEnd = spkLen &* S
            state.spkcache = Array(embs[..<spkEmbEnd])
            state.spkcachePreds = Array(preds[..<spkPredEnd])
            state.fifo = Array(embs[spkEmbEnd...])
            state.fifoPreds = Array(preds[spkPredEnd...])
        } else {
            state.fifo = embs
            state.fifoPreds = preds
        }
        
        return removedEmbeddings
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
        var scores = getLogPredScores(preds: state.spkcachePreds, frameCount: currentLength)

        // Disable low scores
        scores = disableLowScores(
            preds: state.spkcachePreds,
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
                    if srcIdx < state.spkcachePreds.count {
                        newSpkcachePreds[i * numSpeakers + s] = state.spkcachePreds[srcIdx]
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
