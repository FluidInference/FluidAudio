import Accelerate
@preconcurrency import CoreML
import Foundation

/// Internal processing pipeline for Nemotron multilingual streaming ASR.
///
/// Mirrors the English-only pipeline (`StreamingNemotronAsrManager+Pipeline`)
/// with two additions:
///   1. The encoder feature dict carries an extra `prompt_id` int32 [1] input
///      with the currently selected language hint.
///   2. The greedy RNNT decode loop tracks the first language-tag token id
///      it sees and forwards the corresponding piece text to the manager
///      via `recordDetectedLanguage(_:)`.
extension StreamingNemotronMultilingualAsrManager {

    /// Process a single audio chunk through the full pipeline.
    /// If `nextChunkSamples` is non-nil, runs preprocessor[t+1] in parallel
    /// with encoder[t] (CPU preprocessor while ANE encoder runs).
    internal func processChunk(_ samples: [Float], nextChunkSamples: [Float]? = nil) async throws {
        guard let preprocessor = preprocessor,
            let encoder = encoder,
            let decoder = decoder,
            let joint = joint,
            let cacheChannel = cacheChannel,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            var currentH = hState,
            var currentC = cState,
            let tokenizer = tokenizer
        else {
            throw ASRError.notInitialized
        }

        // Track decoder state locally to ensure atomicity
        var currentToken = lastToken

        self.chunkCount += 1
        let prepStart = DispatchTime.now().uptimeNanoseconds

        // TRIPLE-STAGE: if encoder[t] was prefetched by the previous chunk's
        // async helper, skip preprocessor + encoder entirely. The helper
        // already updated self.cacheChannel/Time/Len and self.melCache when
        // we awaited it at the end of processChunk(t-1).
        let encoded: MLMultiArray
        let encoderProj: MLMultiArray?
        if let pre = self.prefetchedEncoded {
            encoded = pre
            encoderProj = self.prefetchedEncoderProj
            self.prefetchedEncoded = nil
            self.prefetchedEncoderProj = nil
            // Clear any stale prefetchedMel — we don't need chunkMel since
            // melCache was already advanced by the helper.
            self.prefetchedMel = nil
            self.prepNanos &+= DispatchTime.now().uptimeNanoseconds &- prepStart
            // encNanos: not counted in this path; the encoder time was
            // hidden under the previous chunk's decode loop.
        } else {
            // Original path: preprocessor[t] + encoder[t] (first call or
            // when triple-stage couldn't pre-dispatch).
            let chunkMel: MLMultiArray
            if let prefetched = self.prefetchedMel {
                chunkMel = prefetched
                self.prefetchedMel = nil
            } else {
                let audioArray = try createAudioArray(samples)
                let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
                audioLen[0] = NSNumber(value: samples.count)
                let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
                    "audio": MLFeatureValue(multiArray: audioArray),
                    "audio_length": MLFeatureValue(multiArray: audioLen),
                ])
                let preprocOutput = try await preprocessor.prediction(from: preprocInput)
                guard let mel = preprocOutput.featureValue(for: "mel")?.multiArrayValue else {
                    throw ASRError.processingFailed("Preprocessor failed to produce mel output")
                }
                chunkMel = mel
            }
            self.prepNanos &+= DispatchTime.now().uptimeNanoseconds &- prepStart

            let encStart = DispatchTime.now().uptimeNanoseconds
            let inputMel = try prependMelCache(to: chunkMel)
            let melLen = try MLMultiArray(shape: [1], dataType: .int32)
            melLen[0] = NSNumber(value: config.totalMelFrames)

            let promptIdArray = try MLMultiArray(shape: [1], dataType: .int32)
            promptIdArray[0] = NSNumber(value: currentPromptIdValue())

            let encoderOutput: MLFeatureProvider
            if #available(macOS 15, iOS 18, *), let state = encoderState as? MLState {
                let statefulInput = try MLDictionaryFeatureProvider(dictionary: [
                    "mel": MLFeatureValue(multiArray: inputMel),
                    "mel_length": MLFeatureValue(multiArray: melLen),
                    "cache_len": MLFeatureValue(multiArray: cacheLen),
                    "prompt_id": MLFeatureValue(multiArray: promptIdArray),
                ])
                if let opts = encoderPredictionOptions {
                    encoderOutput = try await encoder.prediction(from: statefulInput, using: state, options: opts)
                } else {
                    encoderOutput = try await encoder.prediction(from: statefulInput, using: state)
                }
                if let newLen = encoderOutput.featureValue(for: "cache_len_out")?.multiArrayValue {
                    self.cacheLen = newLen
                }
            } else {
                let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
                    "mel": MLFeatureValue(multiArray: inputMel),
                    "mel_length": MLFeatureValue(multiArray: melLen),
                    "cache_channel": MLFeatureValue(multiArray: cacheChannel),
                    "cache_time": MLFeatureValue(multiArray: cacheTime),
                    "cache_len": MLFeatureValue(multiArray: cacheLen),
                    "prompt_id": MLFeatureValue(multiArray: promptIdArray),
                ])
                if let opts = encoderPredictionOptions {
                    encoderOutput = try await encoder.prediction(from: encoderInput, options: opts)
                } else {
                    encoderOutput = try await encoder.prediction(from: encoderInput)
                }
                let updatedCaches = EncoderCacheManager.extractCachesFromOutput(encoderOutput)
                if let newChannel = updatedCaches.channel {
                    self.cacheChannel = newChannel
                }
                if let newTime = updatedCaches.time {
                    self.cacheTime = newTime
                }
                if let newLen = updatedCaches.len {
                    self.cacheLen = newLen
                }
            }

            guard let e = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
                throw ASRError.processingFailed("Encoder failed to produce output")
            }
            encoded = e
            encoderProj = encoderOutput.featureValue(for: "encoder_proj")?.multiArrayValue
            self.encNanos &+= DispatchTime.now().uptimeNanoseconds &- encStart
            // Save mel cache for next chunk (last 9 frames). Only needed on
            // the non-prefetched path — the triple-stage helper already
            // advanced melCache when we awaited it at the previous chunk's end.
            melCache = try extractMelCache(from: chunkMel)
        }

        // TRIPLE-STAGE PIPELINE: dispatch preprocessor[t+1] + encoder[t+1]
        // concurrent with this chunk's decode loop. The async task reads
        // the just-extracted melCache + the just-updated caches by value,
        // runs preproc + encoder on its own, and returns the new state.
        // We await it after the decode loop and save outputs as `prefetched*`
        // so the next processChunk call can skip the encoder entirely.
        //
        // Stateful (MLState) encoder path is excluded from triple-stage —
        // MLState's session ownership doesn't cross task boundaries cleanly.
        let mlStateActive: Bool
        if #available(macOS 15, iOS 18, *) {
            mlStateActive = (encoderState as? MLState) != nil
        } else {
            mlStateActive = false
        }
        nonisolated(unsafe) let snapshotCacheChannel = self.cacheChannel
        nonisolated(unsafe) let snapshotCacheTime = self.cacheTime
        nonisolated(unsafe) let snapshotCacheLen = self.cacheLen
        nonisolated(unsafe) let snapshotMelCache = self.melCache
        let snapshotPromptId = currentPromptIdValue()
        let snapshotTotalMelFrames = config.totalMelFrames
        let snapshotMelFeatures = config.melFeatures
        let snapshotPreEncodeCache = config.preEncodeCache
        async let nextEncFuture: (
            encoded: MLMultiArray,
            encoderProj: MLMultiArray?,
            cacheChannel: MLMultiArray,
            cacheTime: MLMultiArray,
            cacheLen: MLMultiArray,
            newMelCache: MLMultiArray
        )? = {
            guard let next = nextChunkSamples,
                !mlStateActive,
                let ch = snapshotCacheChannel,
                let ti = snapshotCacheTime,
                let ln = snapshotCacheLen
            else { return nil }
            return try await Self.runPrepAndEncoderPure(
                samples: next,
                melCacheForPrepend: snapshotMelCache,
                cacheChannel: ch,
                cacheTime: ti,
                cacheLen: ln,
                promptId: snapshotPromptId,
                totalMelFrames: snapshotTotalMelFrames,
                melFeatures: snapshotMelFeatures,
                preEncodeCache: snapshotPreEncodeCache,
                preprocessor: preprocessor,
                encoder: encoder
            )
        }()

        // 4. RNNT decode loop for each encoder frame
        let decStart = DispatchTime.now().uptimeNanoseconds
        let numEncoderFrames = encoded.shape[2].intValue
        var newTokens: [Int] = []

        for t in 0..<numEncoderFrames {
            let encStep: MLMultiArray
            if let buf = encoderStepBuf {
                fillEncoderStep(into: buf, from: encoded, timeIndex: t)
                encStep = buf
            } else {
                encStep = try extractEncoderStep(from: encoded, timeIndex: t)
            }
            let encStepProj: MLMultiArray?
            if encoderProj != nil && decoderJointNoEncProj != nil {
                if let projBuf = encoderProjStepBuf {
                    fillEncoderProjStep(into: projBuf, from: encoderProj!, timeIndex: t)
                    encStepProj = projBuf
                } else {
                    encStepProj = try extractEncoderProjStep(from: encoderProj!, timeIndex: t)
                }
            } else {
                encStepProj = nil
            }

            // Greedy decode loop (max 10 symbols per frame)
            for _ in 0..<10 {
                let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                tokenInput[0] = NSNumber(value: currentToken)

                let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
                tokenLen[0] = 1

                // Inner-loop call: priority is (1) triple-fused (B2) → returns int32
                // token_id directly, (2) dec+joint fusion (B1) → returns logits +
                // states in one call, (3) separate decoder + joint calls (fallback).
                let predToken: Int
                let hOut: MLMultiArray
                let cOut: MLMultiArray
                if let djne = decoderJointNoEncProj, let encProjStep = encStepProj {
                    // B3+B1 fused path: dec+joint-without-encproj uses
                    // pre-projected encoder features. Saves a 1024->640
                    // matmul per token.
                    let b3Input = try MLDictionaryFeatureProvider(dictionary: [
                        "token": MLFeatureValue(multiArray: tokenInput),
                        "token_length": MLFeatureValue(multiArray: tokenLen),
                        "h_in": MLFeatureValue(multiArray: currentH),
                        "c_in": MLFeatureValue(multiArray: currentC),
                        "encoder_proj": MLFeatureValue(multiArray: encProjStep),
                    ])
                    let b3Output: MLFeatureProvider
                    if let opts = decoderJointNoEncProjPredictionOptions {
                        b3Output = try await djne.prediction(from: b3Input, options: opts)
                    } else {
                        b3Output = try await djne.prediction(from: b3Input)
                    }
                    guard let fl = b3Output.featureValue(for: "logits")?.multiArrayValue,
                        let fh = b3Output.featureValue(for: "h_out")?.multiArrayValue,
                        let fc = b3Output.featureValue(for: "c_out")?.multiArrayValue
                    else {
                        throw ASRError.processingFailed("B3+B1 fused decoder_joint_noencproj failed")
                    }
                    predToken = findMaxIndex(fl)
                    hOut = fh
                    cOut = fc
                } else if let dja = decoderJointArgmax {
                    // Triple-fused path: token + h + c + encoder → token_id (int32) + h + c
                    let tripleInput = try MLDictionaryFeatureProvider(dictionary: [
                        "token": MLFeatureValue(multiArray: tokenInput),
                        "token_length": MLFeatureValue(multiArray: tokenLen),
                        "h_in": MLFeatureValue(multiArray: currentH),
                        "c_in": MLFeatureValue(multiArray: currentC),
                        "encoder": MLFeatureValue(multiArray: encStep),
                    ])
                    let tripleOutput: MLFeatureProvider
                    if let opts = decoderJointArgmaxPredictionOptions {
                        tripleOutput = try await dja.prediction(from: tripleInput, options: opts)
                    } else {
                        tripleOutput = try await dja.prediction(from: tripleInput)
                    }
                    guard let tokenIdArr = tripleOutput.featureValue(for: "token_id")?.multiArrayValue,
                        let fh = tripleOutput.featureValue(for: "h_out")?.multiArrayValue,
                        let fc = tripleOutput.featureValue(for: "c_out")?.multiArrayValue
                    else {
                        throw ASRError.processingFailed("Triple-fused decoder_joint_argmax failed")
                    }
                    predToken = Int(tokenIdArr[0].int32Value)
                    hOut = fh
                    cOut = fc
                } else if let dj = decoderJoint {
                    let fusedInput = try MLDictionaryFeatureProvider(dictionary: [
                        "token": MLFeatureValue(multiArray: tokenInput),
                        "token_length": MLFeatureValue(multiArray: tokenLen),
                        "h_in": MLFeatureValue(multiArray: currentH),
                        "c_in": MLFeatureValue(multiArray: currentC),
                        "encoder": MLFeatureValue(multiArray: encStep),
                    ])
                    let fusedOutput: MLFeatureProvider
                    if let opts = decoderJointPredictionOptions {
                        fusedOutput = try await dj.prediction(from: fusedInput, options: opts)
                    } else {
                        fusedOutput = try await dj.prediction(from: fusedInput)
                    }
                    guard let fl = fusedOutput.featureValue(for: "logits")?.multiArrayValue,
                        let fh = fusedOutput.featureValue(for: "h_out")?.multiArrayValue,
                        let fc = fusedOutput.featureValue(for: "c_out")?.multiArrayValue
                    else {
                        throw ASRError.processingFailed("Fused decoder_joint failed")
                    }
                    predToken = findMaxIndex(fl)
                    hOut = fh
                    cOut = fc
                } else {
                    let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                        "token": MLFeatureValue(multiArray: tokenInput),
                        "token_length": MLFeatureValue(multiArray: tokenLen),
                        "h_in": MLFeatureValue(multiArray: currentH),
                        "c_in": MLFeatureValue(multiArray: currentC),
                    ])
                    let decoderOutput: MLFeatureProvider
                    if let opts = decoderPredictionOptions {
                        decoderOutput = try await decoder.prediction(from: decoderInput, options: opts)
                    } else {
                        decoderOutput = try await decoder.prediction(from: decoderInput)
                    }
                    guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                        let dh = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                        let dc = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
                    else {
                        throw ASRError.processingFailed("Decoder failed")
                    }
                    let decoderStep = try sliceDecoderOutput(decoderOut)
                    let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                        "encoder": MLFeatureValue(multiArray: encStep),
                        "decoder": MLFeatureValue(multiArray: decoderStep),
                    ])
                    let jointOutput: MLFeatureProvider
                    if let opts = jointPredictionOptions {
                        jointOutput = try await joint.prediction(from: jointInput, options: opts)
                    } else {
                        jointOutput = try await joint.prediction(from: jointInput)
                    }
                    guard let jl = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                        throw ASRError.processingFailed("Joint failed")
                    }
                    predToken = findMaxIndex(jl)
                    hOut = dh
                    cOut = dc
                }

                if predToken == config.blankIdx {
                    // Blank token - move to next encoder frame
                    break
                } else {
                    // Non-blank token - emit and update local state
                    newTokens.append(predToken)
                    accumulatedTokenIds.append(predToken)
                    currentToken = Int32(predToken)
                    currentH = hOut
                    currentC = cOut

                    // Surface the first language-tag piece we encounter so
                    // callers can observe `detectedLanguage()` without waiting
                    // for the final decode pass.
                    if config.langTagTokenIds.contains(predToken),
                        let piece = tokenizerPiece(forId: predToken, tokenizer: tokenizer)
                    {
                        let lang = NemotronMultilingualTokenizer.stripAngleBrackets(piece)
                        recordDetectedLanguage(lang)
                    }
                }
            }
        }

        self.decNanos &+= DispatchTime.now().uptimeNanoseconds &- decStart

        // Save final decoder state back to actor properties atomically
        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        // Invoke partial callback if new tokens were decoded
        if !newTokens.isEmpty, let callback = partialCallback {
            let decoded = tokenizer.decode(ids: accumulatedTokenIds)
            callback(decoded.text)
        }

        // TRIPLE-STAGE PIPELINE: collect the next chunk's encoder output
        // (computed concurrently with this chunk's decode loop). Save the
        // encoded[t+1] tensor + the encoder caches it produced + the new
        // melCache extracted from chunkMel[t+1]; the next processChunk call
        // will skip preprocessor and encoder entirely.
        if let nextEnc = try await nextEncFuture {
            self.prefetchedEncoded = nextEnc.encoded
            self.prefetchedEncoderProj = nextEnc.encoderProj
            self.cacheChannel = nextEnc.cacheChannel
            self.cacheTime = nextEnc.cacheTime
            self.cacheLen = nextEnc.cacheLen
            self.melCache = nextEnc.newMelCache
        }

        processedChunks += 1
    }

    /// Read a piece from the underlying base tokenizer through the multilingual
    /// wrapper. Kept as a separate helper so the pipeline doesn't need to
    /// reach inside `NemotronMultilingualTokenizer`.
    private func tokenizerPiece(forId id: Int, tokenizer: NemotronMultilingualTokenizer) -> String? {
        // The wrapper doesn't expose the underlying piece map directly. We
        // round-trip through `decode(ids:)` on a single-id list: if the id
        // is itself a lang-tag we get the detected language back; otherwise
        // we get the raw piece text (minus the SentencePiece marker, which
        // is harmless for the `<xx-XX>` tag check).
        let decoded = tokenizer.decode(ids: [id])
        if let lang = decoded.detectedLanguage {
            return "<\(lang)>"
        }
        return decoded.text.isEmpty ? nil : decoded.text
    }

    // MARK: - Tensor Utilities (duplicated from the English pipeline so the
    // two managers stay independent; the math is small and self-contained).

    internal func createAudioArray(_ samples: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.update(from: samples, count: samples.count)
        return array
    }

    /// Nonisolated helper for async pipelining — runs the preprocessor on a
    /// chunk of samples without touching actor state. Sendable inputs only.
    nonisolated internal static func runPreprocessorPure(samples: [Float], preprocessor: MLModel) async throws -> MLMultiArray? {
        let array = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.update(from: samples, count: samples.count)

        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: array),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        let output = try await preprocessor.prediction(from: input)
        return output.featureValue(for: "mel")?.multiArrayValue
    }

    /// Triple-stage pipeline helper: runs preprocessor[t+1] + encoder[t+1] in
    /// one async task. Captures all required state by value so the closure
    /// can run concurrent with the current chunk's decode loop.
    /// Returns: (encoded, encoderProj, cacheChannelOut, cacheTimeOut, cacheLenOut).
    /// Uses no output backings (passes nil prediction options) so its output
    /// buffers don't race with the current chunk's `encoded` reads.
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
        preprocessor: MLModel,
        encoder: MLModel
    ) async throws -> (
        encoded: MLMultiArray,
        encoderProj: MLMultiArray?,
        cacheChannel: MLMultiArray,
        cacheTime: MLMultiArray,
        cacheLen: MLMultiArray,
        newMelCache: MLMultiArray
    )? {
        guard let chunkMel = try await runPreprocessorPure(samples: samples, preprocessor: preprocessor) else {
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
