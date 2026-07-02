import Accelerate
@preconcurrency import CoreML
import Foundation

/// Split out of `StreamingNemotronMultilingualAsrManager+Pipeline.swift`
/// — same extension, grouped by concern. No logic change.
extension StreamingNemotronMultilingualAsrManager {

    // MARK: - Tensor Utilities (duplicated from the English pipeline so the
    // two managers stay independent; the math is small and self-contained).

    /// Nonisolated helper for async pipelining — computes the native-Swift
    /// log-mel for a chunk without touching actor state. The caller passes a
    /// dedicated `melExtractor` instance (distinct from the on-actor one) so
    /// the extractor's non-thread-safe FFT scratch buffers are never shared
    /// across the concurrent prefetch boundary.
    nonisolated internal static func runPreprocessorPure(
        samples: [Float],
        melExtractor: NemotronMelExtractor
    ) throws -> MLMultiArray? {
        return try melExtractor.melSpectrogram(samples: samples)
    }

    /// Triple-stage pipeline helper: runs preprocessor[t+1] + encoder[t+1] in
    /// one async task. Captures all required state by value so the closure
    /// can run concurrent with the current chunk's decode loop.
    /// Returns: (encoded, encoderProj, cacheChannelOut, cacheTimeOut, cacheLenOut).
    /// Uses no output backings (passes nil prediction options) so its output
    /// buffers don't race with the current chunk's `encoded` reads.
    /// Cached env-var read: FLUIDAUDIO_VAD_RMS_THRESHOLD ([0,1] linear PCM).
    /// 0 (default) = VAD disabled. Recommended starting values:
    ///   0.003 = very conservative (only dead silence)
    ///   0.005 = conservative (silence + faint background)
    ///   0.010 = aggressive (background room noise included; WER risk)
    nonisolated internal static let vadRmsThreshold: Float = {
        guard let s = ProcessInfo.processInfo.environment["FLUIDAUDIO_VAD_RMS_THRESHOLD"],
            let v = Float(s), v > 0, v < 1.0
        else { return 0 }
        return v
    }()

    /// Smarter-VAD hangover: number of consecutive low-RMS chunks required
    /// before triggering a skip. Default 2 — first low chunk after speech
    /// is always processed (consonant-tail edge preserve). Set to 1 to
    /// match the old per-chunk-only behavior.
    nonisolated internal static let vadHangoverChunks: Int = {
        guard let s = ProcessInfo.processInfo.environment["FLUIDAUDIO_VAD_HANGOVER_CHUNKS"],
            let v = Int(s), v >= 1
        else { return 2 }
        return v
    }()

    /// Energy-based silence detector. Returns true iff the RMS of `samples`
    /// is below `threshold`. Uses vDSP_rmsqv for one-pass single-instruction
    /// RMS — cost is dwarfed by even one MLMultiArray alloc.
    nonisolated internal static func isAudioSilent(samples: [Float], threshold: Float) -> Bool {
        guard !samples.isEmpty else { return true }
        var rms: Float = 0
        samples.withUnsafeBufferPointer { ptr in
            vDSP_rmsqv(ptr.baseAddress!, 1, &rms, vDSP_Length(samples.count))
        }
        return rms < threshold
    }

    nonisolated internal static func runPrepAndEncoderPure(
        samples: [Float],
        melCacheForPrepend: MLMultiArray?,
        cacheChannel: MLMultiArray,
        cacheTime: MLMultiArray,
        cacheLen: MLMultiArray,
        promptId: Int32,
        totalMelFrames: Int,
        melFeatures: Int,
        preEncodeCache: Int,
        melExtractor: NemotronMelExtractor,
        encoder: MLModel
    ) async throws -> (
        encoded: MLMultiArray,
        encoderProj: MLMultiArray?,
        cacheChannel: MLMultiArray,
        cacheTime: MLMultiArray,
        cacheLen: MLMultiArray,
        newMelCache: MLMultiArray
    )? {
        // VAD short-circuit in the triple-stage prefetch helper: same rules
        // as in processChunk — skip the entire encoder call for silent
        // chunks. Returning nil makes processChunk(t+1) fall through to its
        // own serial preprocessor+encoder path, where the VAD check fires
        // again (and skips if still silent).
        if vadRmsThreshold > 0, isAudioSilent(samples: samples, threshold: vadRmsThreshold) {
            return nil
        }
        guard
            let chunkMel = try runPreprocessorPure(
                samples: samples,
                melExtractor: melExtractor
            )
        else {
            return nil
        }
        let inputMel = try prependMelCachePure(
            melCache: melCacheForPrepend,
            chunkMel: chunkMel,
            totalMelFrames: totalMelFrames,
            melFeatures: melFeatures,
            preEncodeCache: preEncodeCache
        )

        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: totalMelFrames)

        let promptIdArray = try MLMultiArray(shape: [1], dataType: .int32)
        promptIdArray[0] = NSNumber(value: promptId)

        let encInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: inputMel),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
            "prompt_id": MLFeatureValue(multiArray: promptIdArray),
        ])
        let encOutput = try await encoder.prediction(from: encInput)
        guard let encoded = encOutput.featureValue(for: "encoded")?.multiArrayValue,
            let cacheChOut = encOutput.featureValue(for: "cache_channel_out")?.multiArrayValue,
            let cacheTOut = encOutput.featureValue(for: "cache_time_out")?.multiArrayValue,
            let cacheLenOut = encOutput.featureValue(for: "cache_len_out")?.multiArrayValue
        else {
            return nil
        }
        let encoderProj = encOutput.featureValue(for: "encoder_proj")?.multiArrayValue
        let newMelCache = try extractMelCachePure(
            chunkMel: chunkMel,
            melFeatures: melFeatures,
            preEncodeCache: preEncodeCache
        )
        return (encoded, encoderProj, cacheChOut, cacheTOut, cacheLenOut, newMelCache)
    }

    nonisolated internal static func extractMelCachePure(
        chunkMel: MLMultiArray,
        melFeatures: Int,
        preEncodeCache: Int
    ) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(preEncodeCache, chunkFrames)
        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )
        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)
        let srcStride1 = chunkMel.strides[1].intValue
        let srcStride2 = chunkMel.strides[2].intValue
        let dstStride1 = cache.strides[1].intValue
        let dstStride2 = cache.strides[2].intValue
        let startT = chunkFrames - cacheFrames
        for mel in 0..<melFeatures {
            for t in 0..<cacheFrames {
                dstPtr[mel * dstStride1 + t * dstStride2] =
                    srcPtr[mel * srcStride1 + (startT + t) * srcStride2]
            }
        }
        return cache
    }

    /// Nonisolated version of prependMelCache for use in the async encoder
    /// task. Performs the same mel-cache prepending as the instance method
    /// but takes all required state explicitly.
    nonisolated internal static func prependMelCachePure(
        melCache: MLMultiArray?,
        chunkMel: MLMultiArray,
        totalMelFrames: Int,
        melFeatures: Int,
        preEncodeCache: Int
    ) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: totalMelFrames)],
            dataType: .float32
        )
        result.reset(to: 0)
        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let chunkPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue
        let chunkStride1 = chunkMel.strides[1].intValue
        let chunkStride2 = chunkMel.strides[2].intValue

        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue
            for mel in 0..<melFeatures {
                for t in 0..<cacheFrames {
                    resultPtr[mel * resultStride1 + t * resultStride2] =
                        cachePtr[mel * cacheStride1 + t * cacheStride2]
                }
            }
        }
        let copyFrames = min(chunkFrames, totalMelFrames - preEncodeCache)
        for mel in 0..<melFeatures {
            for t in 0..<copyFrames {
                resultPtr[mel * resultStride1 + (preEncodeCache + t) * resultStride2] =
                    chunkPtr[mel * chunkStride1 + t * chunkStride2]
            }
        }
        return result
    }

    internal func prependMelCache(to chunkMel: MLMultiArray) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let totalFrames = config.totalMelFrames

        let result = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: totalFrames)],
            dataType: .float32
        )
        result.reset(to: 0)

        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let chunkPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)

        let resultStride0 = result.strides[0].intValue
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue
        let chunkStride0 = chunkMel.strides[0].intValue
        let chunkStride1 = chunkMel.strides[1].intValue
        let chunkStride2 = chunkMel.strides[2].intValue

        // Copy mel cache (or zeros if first chunk)
        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue
            let cacheStride0 = melCache.strides[0].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue

            for mel in 0..<config.melFeatures {
                for t in 0..<cacheFrames {
                    let srcIdx = 0 * cacheStride0 + mel * cacheStride1 + t * cacheStride2
                    let dstIdx = 0 * resultStride0 + mel * resultStride1 + t * resultStride2
                    resultPtr[dstIdx] = cachePtr[srcIdx]
                }
            }
        }

        // Copy chunk mel (after cache position)
        let copyFrames = min(chunkFrames, totalFrames - config.preEncodeCache)
        for mel in 0..<config.melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = 0 * chunkStride0 + mel * chunkStride1 + t * chunkStride2
                let dstIdx = 0 * resultStride0 + mel * resultStride1 + (config.preEncodeCache + t) * resultStride2
                resultPtr[dstIdx] = chunkPtr[srcIdx]
            }
        }

        return result
    }

    internal func extractMelCache(from chunkMel: MLMultiArray) throws -> MLMultiArray {
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(config.preEncodeCache, chunkFrames)

        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )

        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)

        let srcStride0 = chunkMel.strides[0].intValue
        let srcStride1 = chunkMel.strides[1].intValue
        let srcStride2 = chunkMel.strides[2].intValue
        let dstStride0 = cache.strides[0].intValue
        let dstStride1 = cache.strides[1].intValue
        let dstStride2 = cache.strides[2].intValue

        let startT = chunkFrames - cacheFrames

        for mel in 0..<config.melFeatures {
            for t in 0..<cacheFrames {
                let srcIdx = 0 * srcStride0 + mel * srcStride1 + (startT + t) * srcStride2
                let dstIdx = 0 * dstStride0 + mel * dstStride1 + t * dstStride2
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return cache
    }

    /// Fill a pre-allocated [1, dim, 1] buffer with one time-step from
    /// encoded [1, dim, T]. Zero allocations per call. Used inside the
    /// inner RNN-T greedy loop.
    internal func fillEncoderStep(into dest: MLMultiArray, from encoded: MLMultiArray, timeIndex: Int) {
        let dim = encoded.shape[1].intValue
        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = dest.dataPointer.bindMemory(to: Float.self, capacity: dest.count)
        let stride0 = encoded.strides[0].intValue
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue
        for c in 0..<dim {
            let srcIdx = c * stride1 + timeIndex * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }
        _ = stride0  // suppress unused
    }

    /// Fill a pre-allocated [1, 1, joint_dim] buffer with one time-step from
    /// encoder_proj [1, T, joint_dim]. Mirrors fillEncoderStep for the B3
    /// path.
    internal func fillEncoderProjStep(into dest: MLMultiArray, from encoderProj: MLMultiArray, timeIndex: Int) {
        let jointDim = encoderProj.shape[2].intValue
        let srcPtr = encoderProj.dataPointer.bindMemory(to: Float.self, capacity: encoderProj.count)
        let dstPtr = dest.dataPointer.bindMemory(to: Float.self, capacity: dest.count)
        let stride1 = encoderProj.strides[1].intValue
        let stride2 = encoderProj.strides[2].intValue
        for c in 0..<jointDim {
            let srcIdx = timeIndex * stride1 + c * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }
    }

    internal func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        // encoded: [1, 1024, T] -> step: [1, 1024, 1]
        let dim = encoded.shape[1].intValue
        let step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)

        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)

        let stride0 = encoded.strides[0].intValue
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        for c in 0..<dim {
            let srcIdx = 0 * stride0 + c * stride1 + timeIndex * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return step
    }

    /// B3 helper: extract one frame from the per-chunk pre-projected encoder
    /// output. Layout is [1, T, 640] (B=1 batch, T=time, 640=joint_dim).
    internal func extractEncoderProjStep(from encoderProj: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let jointDim = encoderProj.shape[2].intValue
        let step = try MLMultiArray(shape: [1, 1, NSNumber(value: jointDim)], dataType: .float32)

        let srcPtr = encoderProj.dataPointer.bindMemory(to: Float.self, capacity: encoderProj.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)

        let stride0 = encoderProj.strides[0].intValue
        let stride1 = encoderProj.strides[1].intValue
        let stride2 = encoderProj.strides[2].intValue

        for c in 0..<jointDim {
            let srcIdx = 0 * stride0 + timeIndex * stride1 + c * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return step
    }

    internal func sliceDecoderOutput(_ decoderOut: MLMultiArray) throws -> MLMultiArray {
        // decoder_out: [1, hidden, T] -> [1, hidden, 1] (first frame, index 0)
        let hidden = decoderOut.shape[1].intValue

        let result = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1], dataType: .float32)

        let srcPtr = decoderOut.dataPointer.bindMemory(to: Float.self, capacity: decoderOut.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)

        let stride0 = decoderOut.strides[0].intValue
        let stride1 = decoderOut.strides[1].intValue
        let stride2 = decoderOut.strides[2].intValue

        let firstT = 0
        for c in 0..<hidden {
            let srcIdx = 0 * stride0 + c * stride1 + firstT * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return result
    }

    internal func findMaxIndex(_ logits: MLMultiArray) -> Int {
        // Use actual logits count to prevent out-of-bounds when config is incorrect
        let count = logits.count
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)

        var maxVal: Float = -Float.infinity
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))

        return Int(maxIdx)
    }
}
