import Accelerate
import Foundation

/// Host-side streaming state update for Nemotron 3 Diarization.
///
/// Port of NeMo `SortformerModules.streaming_update_async` (batch size 1) for the
/// 8-speaker preview checkpoint: fixed-capacity spkcache/FIFO, learned silence embedding
/// (`use_learnable_sil_emb: true`, so no running silence profile), score-based cache
/// compression with latest-frame boosting.
public struct Nemotron3StateUpdater {

    private let config: Nemotron3Config
    private let silenceEmbedding: [Float]

    public init(config: Nemotron3Config, silenceEmbedding: [Float]) {
        self.config = config
        self.silenceEmbedding = silenceEmbedding
    }

    /// Apply one streaming update.
    ///
    /// - Parameters:
    ///   - state: Streaming state, mutated in place.
    ///   - chunkEmbeddings: Pre-encode embeddings for the chunk incl. contexts,
    ///     [chunkEncFrames * 512] (only the first `chunkEncLength` frames valid).
    ///   - chunkEncLength: Valid encoder frames in `chunkEmbeddings` (incl. lc and rc).
    ///   - predictions: Packed 80 ms predictions [packedFrames * 8], valid frames left-packed
    ///     as [spkcache | fifo | chunk].
    ///   - highResPredictions: Packed 10 ms predictions [packedFrames * 8 * 8].
    ///   - lc: Encoded left-context frames of this chunk.
    ///   - rc: Encoded right-context frames of this chunk.
    /// - Returns: This chunk's core-frame probabilities at 10 ms resolution.
    public func update(
        state: inout Nemotron3StreamingState,
        chunkEmbeddings: [Float],
        chunkEncLength: Int,
        predictions: [Float],
        highResPredictions: [Float],
        lc: Int,
        rc: Int
    ) throws -> Nemotron3ChunkResult {
        let d = config.preEncoderDims
        let s = config.numSpeakers
        let up = config.upsampleFactor
        let maxChunk = config.chunkEncFrames - lc - rc
        let fifoCap = config.fifoLen
        let scCap = config.spkcacheLen

        let scLen = state.spkcacheLength
        let fifoLen = state.fifoLength
        let chunkLen = min(max(chunkEncLength - lc, 0), maxChunk)

        // Region slices of the packed predictions (valid frames are left-packed).
        let spkcachePredsCur = Array(predictions[0..<(scLen * s)])
        let fifoPredsCur = Array(predictions[(scLen * s)..<((scLen + fifoLen) * s)])
        let chunkPredStart = (scLen + fifoLen + lc) * s
        let chunkPredsCur = Array(predictions[chunkPredStart..<(chunkPredStart + chunkLen * s)])

        // High-resolution output for this chunk's core region (NeMo
        // `_extract_async_high_resolution_chunk_preds`), taken before the state mutates.
        let hiStart = (scLen + fifoLen + lc) * up * s
        let hiCount = chunkLen * up * s
        let chunkResult = Nemotron3ChunkResult(
            probabilities: Array(highResPredictions[hiStart..<(hiStart + hiCount)]),
            frameCount: chunkLen * up,
            numSpeakers: s
        )

        // FIFO pop lengths (NeMo `_compute_async_fifo_pop_lengths`).
        let combined = fifoLen + chunkLen
        var pop = 0
        if combined > fifoCap {
            pop = min(combined, max(config.spkcacheUpdatePeriod, combined - fifoCap))
        }
        if chunkLen == 0 {
            pop = fifoLen  // finalized stream: flush remaining FIFO into the cache
        }
        let newFifoLen = combined - pop

        // Logical [FIFO | chunk core] concatenation.
        var logicalEmbs = [Float]()
        logicalEmbs.reserveCapacity(combined * d)
        logicalEmbs.append(contentsOf: state.fifo[0..<(fifoLen * d)])
        logicalEmbs.append(contentsOf: chunkEmbeddings[(lc * d)..<((lc + chunkLen) * d)])
        var logicalPreds = [Float]()
        logicalPreds.reserveCapacity(combined * s)
        logicalPreds.append(contentsOf: fifoPredsCur)
        logicalPreds.append(contentsOf: chunkPredsCur)

        let popEmbs = Array(logicalEmbs[0..<(pop * d)])
        let popPreds = Array(logicalPreds[0..<(pop * s)])

        // Retained frames become the new FIFO (zero-padded to capacity).
        replaceRegion(&state.fifo, with: logicalEmbs[(pop * d)..<(combined * d)], capacity: fifoCap * d)
        replaceRegion(&state.fifoPreds, with: logicalPreds[(pop * s)..<(combined * s)], capacity: fifoCap * s)
        state.fifoLength = newFifoLen

        // No running silence profile: the checkpoint uses a learned silence embedding.

        // Speaker cache append + compression (NeMo `_update_async_spkcache`).
        let updatedLen = scLen + pop
        if updatedLen > scCap {
            // Candidate preds: fresh predictions for the cache region on the FIRST compression,
            // stored (frozen) predictions afterwards.
            var candidateEmbs = Array(state.spkcache[0..<(scLen * d)])
            candidateEmbs.append(contentsOf: popEmbs)
            var candidatePreds =
                state.spkcacheCompressed
                ? Array(state.spkcachePreds[0..<(scLen * s)])
                : spkcachePredsCur
            candidatePreds.append(contentsOf: popPreds)

            let (newCache, newCachePreds) = compressSpkcache(
                embs: candidateEmbs, preds: candidatePreds, frameCount: updatedLen)
            replaceRegion(&state.spkcache, with: newCache[...], capacity: scCap * d)
            replaceRegion(&state.spkcachePreds, with: newCachePreds[...], capacity: scCap * s)
            state.spkcacheLength = scCap
            state.spkcacheCompressed = true
        } else if pop > 0 {
            state.spkcache.replaceSubrange((scLen * d)..<(updatedLen * d), with: popEmbs)
            state.spkcachePreds.replaceSubrange((scLen * s)..<(updatedLen * s), with: popPreds)
            state.spkcacheLength = updatedLen
        }

        return chunkResult
    }

    /// Overwrite a fixed-capacity flattened buffer with new content, zero-padding the tail.
    private func replaceRegion(_ buffer: inout [Float], with content: ArraySlice<Float>, capacity: Int) {
        buffer.replaceSubrange(0..<content.count, with: content)
        if content.count < capacity {
            buffer.replaceSubrange(
                content.count..<capacity, with: repeatElement(0, count: capacity - content.count))
        }
    }

    // MARK: - Speaker cache compression (NeMo `_compress_spkcache`)

    private func compressSpkcache(
        embs: [Float], preds: [Float], frameCount: Int
    ) -> (cache: [Float], cachePreds: [Float]) {
        let d = config.preEncoderDims
        let s = config.numSpeakers
        let scCap = config.spkcacheLen
        let silFrames = config.spkcacheSilFramesPerSpk

        let perSpk = scCap / s - silFrames
        let strongBoost = Int(Float(perSpk) * config.strongBoostRate)
        let weakBoost = Int(Float(perSpk) * config.weakBoostRate)
        let minPosScores = Int(Float(perSpk) * config.minPosScoresRate)

        var scores = logPredScores(preds: preds, frameCount: frameCount)
        disableLowScores(preds: preds, scores: &scores, frameCount: frameCount, minPosScores: minPosScores)

        // Boost newly added frames (indices beyond the previous cache capacity).
        if config.scoresBoostLatest > 0 && frameCount > scCap {
            for frame in scCap..<frameCount {
                for spk in 0..<s where scores[frame * s + spk] != -.infinity {
                    scores[frame * s + spk] += config.scoresBoostLatest
                }
            }
        }

        boostTopKScores(scores: &scores, frameCount: frameCount, k: strongBoost, scaleFactor: 2.0)
        boostTopKScores(scores: &scores, frameCount: frameCount, k: weakBoost, scaleFactor: 1.0)

        // Append silence placeholder frames with +inf scores so they are always selected.
        let totalFrames = frameCount + silFrames
        scores.append(contentsOf: repeatElement(.infinity, count: silFrames * s))

        let (topK, isDisabled) = topKIndices(scores: scores, frameCount: totalFrames, k: scCap)

        var cache = [Float](repeating: 0, count: scCap * d)
        var cachePreds = [Float](repeating: 0, count: scCap * s)
        for (i, frameIdx) in topK.enumerated() {
            if isDisabled[i] {
                for dim in 0..<d { cache[i * d + dim] = silenceEmbedding[dim] }
            } else {
                let src = frameIdx * d
                cache.replaceSubrange((i * d)..<((i + 1) * d), with: embs[src..<(src + d)])
                let psrc = frameIdx * s
                cachePreds.replaceSubrange((i * s)..<((i + 1) * s), with: preds[psrc..<(psrc + s)])
            }
        }
        return (cache, cachePreds)
    }

    /// score = log(p) - log(1-p) + sum_spk(log(1-p)) - log(0.5), with clamping at
    /// `predScoreThreshold` (NeMo `_get_log_pred_scores`).
    private func logPredScores(preds: [Float], frameCount: Int) -> [Float] {
        let s = config.numSpeakers
        let threshold = config.predScoreThreshold
        let count = frameCount * s
        var scores = [Float](repeating: 0, count: count)
        var logP = [Float](repeating: 0, count: count)
        var log1P = [Float](repeating: 0, count: count)
        var tmp = [Float](repeating: 0, count: count)

        let p = Array(preds[0..<count])
        vDSP.clip(p, to: threshold...Float.greatestFiniteMagnitude, result: &tmp)
        vForce.log(tmp, result: &logP)

        // log(1-p) clamped at threshold: log(max(1-p, threshold))
        vDSP.negative(p, result: &tmp)
        vDSP.add(1, tmp, result: &tmp)
        vDSP.clip(tmp, to: threshold...Float.greatestFiniteMagnitude, result: &tmp)
        vForce.log(tmp, result: &log1P)

        for frame in 0..<frameCount {
            let base = frame * s
            var sum: Float = 0
            for spk in 0..<s { sum += log1P[base + spk] }
            for spk in 0..<s {
                scores[base + spk] = logP[base + spk] - log1P[base + spk] + sum + logf(2)
            }
        }
        return scores
    }

    /// NeMo `_disable_low_scores`: -inf for non-speech; -inf for non-positive scores when the
    /// speaker already has >= minPosScores positive-scored frames.
    private func disableLowScores(
        preds: [Float], scores: inout [Float], frameCount: Int, minPosScores: Int
    ) {
        let s = config.numSpeakers
        var posCounts = [Int](repeating: 0, count: s)
        for frame in 0..<frameCount {
            for spk in 0..<s {
                let i = frame * s + spk
                if preds[i] > 0.5 && scores[i] > 0 { posCounts[spk] += 1 }
            }
        }
        for frame in 0..<frameCount {
            for spk in 0..<s {
                let i = frame * s + spk
                if preds[i] <= 0.5 {
                    scores[i] = -.infinity
                } else if scores[i] <= 0 && posCounts[spk] >= minPosScores {
                    scores[i] = -.infinity
                }
            }
        }
    }

    /// NeMo `_boost_topk_scores`: add scaleFactor * log(2) to each speaker's top-k finite scores.
    private func boostTopKScores(
        scores: inout [Float], frameCount: Int, k: Int, scaleFactor: Float
    ) {
        let s = config.numSpeakers
        guard k > 0, frameCount > 0 else { return }
        let delta = scaleFactor * logf(2)
        let kEff = min(k, frameCount)

        var topFrames = [Int](repeating: 0, count: kEff)
        var topScores = [Float](repeating: -.greatestFiniteMagnitude, count: kEff)

        for spk in 0..<s {
            var count = 0
            for frame in 0..<frameCount {
                let v = scores[frame * s + spk]
                if v == -.infinity { continue }
                if count < kEff {
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
                    if v <= topScores[count - 1] { continue }
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
            for i in 0..<count {
                scores[topFrames[i] * s + spk] += delta
            }
        }
    }

    /// NeMo `_get_topk_indices`: top-k over speaker-major flattened scores, order-preserving
    /// sort, modulo back to frame indices, silence-pad frames flagged as disabled.
    private func topKIndices(
        scores: [Float], frameCount: Int, k: Int
    ) -> (indices: [Int], isDisabled: [Bool]) {
        let s = config.numSpeakers
        let silFrames = config.spkcacheSilFramesPerSpk
        let nFramesNoSil = frameCount - silFrames
        let maxIndex = config.maxIndex
        let n = frameCount * s
        let kEff = min(k, n)

        // Top-k over permuted index space (spk * frameCount + frame), kept DESC by score with
        // smaller-index tie-break (matches torch.topk + sort behavior).
        var bestIdx = [Int](repeating: 0, count: kEff)
        var bestVal = [Float](repeating: -.infinity, count: kEff)
        var count = 0

        for spk in 0..<s {
            for frame in 0..<frameCount {
                let permutedIdx = spk * frameCount + frame
                let v = scores[frame * s + spk]
                if count < kEff {
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
                    let worstV = bestVal[kEff - 1]
                    let worstI = bestIdx[kEff - 1]
                    if v < worstV || (v == worstV && permutedIdx >= worstI) { continue }
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

        var topK = [Int](repeating: maxIndex, count: k)
        for i in 0..<kEff where bestVal[i] != -.infinity {
            topK[i] = bestIdx[i]
        }
        topK.sort()

        var isDisabled = [Bool](repeating: false, count: k)
        for i in 0..<k {
            if topK[i] == maxIndex {
                isDisabled[i] = true
                topK[i] = 0
                continue
            }
            topK[i] = topK[i] % frameCount
            if topK[i] >= nFramesNoSil {
                isDisabled[i] = true
                topK[i] = 0
            }
        }
        return (topK, isDisabled)
    }
}
