import Accelerate
@preconcurrency import CoreML
import Foundation

/// IEEE-754 binary16 (half) bit pattern → `Float`.
///
/// On arm64 this uses the native `Float16` hardware conversion. On x86_64
/// (Intel) macOS, Swift's `Float16` lacks the `bitPattern:` initializer, the
/// `Float(_:)` conversion, and `Comparable` conformance (no hardware half
/// precision), so a Universal build's x86_64 slice fails to compile. The
/// portable integer path below produces bit-identical results, keeping this
/// file arch-agnostic with no behavior change on arm64.
@inline(__always)
internal func nemotronHalfBitsToFloat(_ bits: UInt16) -> Float {
    #if arch(arm64)
    return Float(Float16(bitPattern: bits))
    #else
    let h = UInt32(bits)
    let sign = (h & 0x8000) << 16
    var exp = Int32((h & 0x7C00) >> 10)
    var mant = h & 0x03FF
    let f: UInt32
    if exp == 0 {
        if mant == 0 {
            f = sign  // ±0
        } else {
            exp = 1  // subnormal → normalized
            while (mant & 0x0400) == 0 {
                mant <<= 1
                exp -= 1
            }
            mant &= 0x03FF
            f = sign | (UInt32(exp + 112) << 23) | (mant << 13)
        }
    } else if exp == 0x1F {
        f = sign | 0x7F80_0000 | (mant << 13)  // Inf / NaN
    } else {
        f = sign | (UInt32(exp + 112) << 23) | (mant << 13)  // normalized
    }
    return Float(bitPattern: f)
    #endif
}

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
        // decoder/joint are optional (lean B1 ships omit them) — bound
        // locally at the sites that need them (smart-spec, unfused fallback).
        guard let preprocessor = preprocessor,
            let encoder = encoder,
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

        // VAD-GATED SKIP with hangover: skip only after N consecutive
        // low-RMS chunks (default N=2). The first low chunk after speech
        // is ALWAYS processed — preserves consonant tails / quiet word
        // endings on dense speech. On podcast/silence-heavy audio, true
        // sustained silence still gets skipped after the hangover window.
        //
        // FLUIDAUDIO_VAD_RMS_THRESHOLD: 0 (disabled) / 0.003-0.010 (workload-tuned)
        // FLUIDAUDIO_VAD_HANGOVER_CHUNKS: required consecutive low-RMS chunks (default 2)
        if Self.vadRmsThreshold > 0 {
            if Self.isAudioSilent(samples: samples, threshold: Self.vadRmsThreshold) {
                self.vadConsecutiveLowChunks &+= 1
                if self.vadConsecutiveLowChunks >= Self.vadHangoverChunks {
                    self.vadSkipCount &+= 1
                    return
                }
                // Within hangover window — process normally (edge preserve).
            } else {
                self.vadConsecutiveLowChunks = 0
            }
        }

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
                // Reuse pre-allocated audio buffer when sized correctly
                // (chunkSamples == config.chunkSamples for normal chunks).
                // Final chunk may be shorter (padded) so falls back to fresh
                // alloc to match the actual sample count.
                let audioArray: MLMultiArray
                let audioLen: MLMultiArray
                if let buf = audioInputBuf,
                    buf.shape[1].intValue == samples.count,
                    let lenBuf = audioLenBuf
                {
                    let ptr = buf.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
                    ptr.update(from: samples, count: samples.count)
                    lenBuf[0] = NSNumber(value: samples.count)
                    audioArray = buf
                    audioLen = lenBuf
                } else {
                    audioArray = try createAudioArray(samples)
                    audioLen = try MLMultiArray(shape: [1], dataType: .int32)
                    audioLen[0] = NSNumber(value: samples.count)
                }
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
        nonisolated(unsafe) let snapshotAudioBuf = self.audioInputBuf
        nonisolated(unsafe) let snapshotAudioLenBuf = self.audioLenBuf
        let snapshotPromptId = currentPromptIdValue()
        let snapshotTotalMelFrames = config.totalMelFrames
        let snapshotMelFeatures = config.melFeatures
        let snapshotPreEncodeCache = config.preEncodeCache
        async let nextEncFuture:
            (
                encoded: MLMultiArray,
                encoderProj: MLMultiArray?,
                cacheChannel: MLMultiArray,
                cacheTime: MLMultiArray,
                cacheLen: MLMultiArray,
                newMelCache: MLMultiArray
            )? = {
                // TEMP env-var disable used during the session-9 A/B bench that
                // measures baseline vs +triple-stage. Remove after the doc table
                // is finalized.
                let tripleStageDisabled = ProcessInfo.processInfo.environment["FLUIDAUDIO_DISABLE_TRIPLE_STAGE"] != nil
                guard let next = nextChunkSamples,
                    !mlStateActive,
                    !tripleStageDisabled,
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
                    encoder: encoder,
                    audioInputBuf: snapshotAudioBuf,
                    audioLenBuf: snapshotAudioLenBuf
                )
            }()

        // 4. RNNT decode loop for each encoder frame
        let decStart = DispatchTime.now().uptimeNanoseconds
        let numEncoderFrames = encoded.shape[2].intValue
        var newTokens: [Int] = []

        // SMART SPECULATIVE BLANK path — speculative scan over K=8 frames
        // per batched joint call. Blank streaks consume 1 joint call per
        // K frames vs K calls in the standard per-token loop.
        //
        // Two ways to get encoder_proj:
        // (A) Multi-output encoder (was tried; breaks ANE compilation
        //     — stalls in ANECompilerService for 30+ min)
        // (B) Compute encoder_proj IN SWIFT via cblas_sgemm using
        //     joint.enc weights from native_weights/ (this path).
        //     Standard encoder stays ANE-resident; encoder_proj is a
        //     ~4ms CPU matmul per chunk.
        //
        // Activates when joint_noencproj_batched.mlpackage is loaded
        // AND either: (A) encoder emits encoder_proj OR (B) native
        // weights are available for the Swift-side projection.
        //
        // Default-on as of May 2026 (T3 confirmed K=4 at 1120ms is
        // +2.0% non-overlapping, K=8 at 4480ms is +1.7% non-overlapping,
        // both WER-neutral). Opt-out via FLUIDAUDIO_ENABLE_SMART_SPECULATIVE=0
        // (or "false"). When the required assets aren't shipped, the path
        // falls back transparently to the legacy inner loop regardless.
        let smartSpecEnabled: Bool
        if let v = ProcessInfo.processInfo.environment["FLUIDAUDIO_ENABLE_SMART_SPECULATIVE"] {
            let lowered = v.lowercased()
            smartSpecEnabled = !(lowered == "0" || lowered == "false" || lowered == "no")
        } else {
            smartSpecEnabled = true
        }
        if smartSpecEnabled,
            let jointBatched = self.jointNoEncProjBatched,
            let decoder = self.decoder,
            let encProjResolved: MLMultiArray = try await {
                if let direct = encoderProj { return direct }
                return nil
            }()
        {
            let encProj = encProjResolved
            try await runSpeculativeBlankDecodeV2(
                encoded: encoded,
                encoderProj: encProj,
                numEncoderFrames: numEncoderFrames,
                decoder: decoder,
                jointBatched: jointBatched,
                currentH: &currentH,
                currentC: &currentC,
                currentToken: &currentToken,
                newTokens: &newTokens,
                tokenizer: tokenizer
            )
            self.decNanos &+= DispatchTime.now().uptimeNanoseconds &- decStart
            self.lastToken = currentToken
            self.hState = currentH
            self.cState = currentC
            if !newTokens.isEmpty, let callback = partialCallback {
                let decoded = tokenizer.decode(ids: accumulatedTokenIds)
                callback(decoded.text)
            }
            if let nextEnc = try await nextEncFuture {
                self.prefetchedEncoded = nextEnc.encoded
                self.prefetchedEncoderProj = nextEnc.encoderProj
                self.cacheChannel = nextEnc.cacheChannel
                self.cacheTime = nextEnc.cacheTime
                self.cacheLen = nextEnc.cacheLen
                self.melCache = nextEnc.newMelCache
            }
            processedChunks += 1
            return
        }

        // SPECULATIVE BATCHED path — KNOWN-SLOW (Session 9 finding).
        // The naive implementation issues 1 decoder call + 1 joint_batched
        // call per emission. joint_batched is ~10× the cost of a single
        // joint call, and savings from blank-streak skipping don't make up
        // for it (measured 8× slowdown vs B1+triple-stage on 1h test-clean
        // continuous). Gated behind FLUIDAUDIO_ENABLE_SPECULATIVE_BATCHED
        // env var so we can keep the code for future redesigns (e.g. cached
        // enc_proj + single-frame joint walk) without affecting default
        // routing. See what_failed_rtfx_table.md for full analysis.
        let speculativeEnabled = ProcessInfo.processInfo.environment["FLUIDAUDIO_ENABLE_SPECULATIVE_BATCHED"] != nil
        if speculativeEnabled, let jb = self.jointBatched, let decoder = self.decoder {
            try await runSpeculativeBatchedDecode(
                encoded: encoded,
                numEncoderFrames: numEncoderFrames,
                decoder: decoder,
                jointBatched: jb,
                currentH: &currentH,
                currentC: &currentC,
                currentToken: &currentToken,
                newTokens: &newTokens,
                tokenizer: tokenizer
            )
            // Skip the per-token loop below.
            self.decNanos &+= DispatchTime.now().uptimeNanoseconds &- decStart
            self.lastToken = currentToken
            self.hState = currentH
            self.cState = currentC
            if !newTokens.isEmpty, let callback = partialCallback {
                let decoded = tokenizer.decode(ids: accumulatedTokenIds)
                callback(decoded.text)
            }
            if let nextEnc = try await nextEncFuture {
                self.prefetchedEncoded = nextEnc.encoded
                self.prefetchedEncoderProj = nextEnc.encoderProj
                self.cacheChannel = nextEnc.cacheChannel
                self.cacheTime = nextEnc.cacheTime
                self.cacheLen = nextEnc.cacheLen
                self.melCache = nextEnc.newMelCache
            }
            processedChunks += 1
            return
        }

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
            let disablePrealloc = ProcessInfo.processInfo.environment["FLUIDAUDIO_DISABLE_TOKEN_PREALLOC"] != nil
            for _ in 0..<10 {
                // Reuse pre-allocated buffers from the manager when present;
                // tokenInput just needs slot 0 refilled with currentToken
                // (tokenLen is a constant 1 already set at loadModels).
                // Set FLUIDAUDIO_DISABLE_TOKEN_PREALLOC=1 to A/B the old
                // alloc-per-iter path.
                let tokenInput: MLMultiArray
                if let buf = tokenInputBuf, !disablePrealloc {
                    buf[0] = NSNumber(value: currentToken)
                    tokenInput = buf
                } else {
                    tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                    tokenInput[0] = NSNumber(value: currentToken)
                }

                let tokenLen: MLMultiArray
                if let buf = tokenLenBuf, !disablePrealloc {
                    tokenLen = buf
                } else {
                    tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
                    tokenLen[0] = 1
                }

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
                } else if let decoder = self.decoder, let joint = self.joint {
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
                } else {
                    throw ASRError.processingFailed(
                        "No decode path: need a fused decoder_joint (B1/B3/B2) or bare decoder+joint")
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

    /// Smart speculative-blank decode (V2). Uses B3 encoder-proj split +
    /// batched joint_noencproj to fast-skip blank streaks K-at-a-time.
    ///
    /// Algorithm per chunk:
    ///   t = 0
    ///   while t < T:
    ///     dec_out, h_out, c_out = decoder(currentToken, currentH, currentC)
    ///     enc_proj_batch = encoder_proj[t : t+K]
    ///     logits[1, K, 1, V] = jointBatched(enc_proj_batch, dec_out)
    ///     // Scan for first non-blank in K frames
    ///     for k in 0..<K:
    ///       pred = argmax(logits[k])
    ///       if pred != blank:
    ///         emit pred
    ///         currentToken, currentH, currentC = pred, h_out, c_out
    ///         // Optional per-frame multi-emission at THIS frame:
    ///         //   re-run decoder + (could use single-frame joint_noencproj
    ///         //   or just slice joint_noencproj_batched output) until blank
    ///         //   Cap at max 10 per frame
    ///         t = t + k + 1
    ///         break
    ///     else:
    ///       // All-blank streak — fast skip
    ///       t = t + K
    ///
    ///   Semantic parity argument:
    ///   - Blank emissions in RNN-T greedy do NOT advance state. So K
    ///     consecutive blank predictions made with the same dec_out are
    ///     valid — the state would have been the same regardless.
    ///   - First non-blank at frame t+k used the (correct) state-at-t
    ///     dec_out. Emit and update state. Subsequent frames have new state.
    ///   - Multi-emission per frame: capped at standard 10 via per-frame
    ///     fallback after speculative scan finds a non-blank.
    ///
    /// K is a fixed structural choice (matches joint_noencproj_batched batch
    /// dim). Not benchmark-tuned.
    internal func runSpeculativeBlankDecodeV2(
        encoded: MLMultiArray,
        encoderProj: MLMultiArray,
        numEncoderFrames: Int,
        decoder: MLModel,
        jointBatched: MLModel,
        currentH: inout MLMultiArray,
        currentC: inout MLMultiArray,
        currentToken: inout Int32,
        newTokens: inout [Int],
        tokenizer: NemotronMultilingualTokenizer
    ) async throws {
        let K = self.jointNoEncProjBatchedK
        let blankIdx = config.blankIdx

        // L8: reuse the batched encoder_proj slice buffer across chunks
        // (was allocated fresh per chunk before; ~30 KB × 7860 chunks ≈
        // 235 MB allocation churn over a full test-clean run).
        if self.encProjBatchReusable == nil
            || self.encProjBatchReusable?.shape[1].intValue != K
        {
            self.encProjBatchReusable = try MLMultiArray(
                shape: [1, NSNumber(value: K), 640], dataType: .float32
            )
        }
        let encProjBatchBuf = self.encProjBatchReusable!

        // Stride/dim info for slicing encoderProj into the batched buf.
        // encoderProj shape: [1, T_enc, 640]
        let projDim = encoderProj.shape[2].intValue
        precondition(projDim == 640, "encoder_proj last dim must be 640")
        let encProjStride0 = encoderProj.strides[0].intValue
        let encProjStride1 = encoderProj.strides[1].intValue
        let encProjStride2 = encoderProj.strides[2].intValue

        // encProj may be fp16 on ANE; convert per-element when copying.
        let encProjIsF16 = (encoderProj.dataType == .float16)
        let srcF16Ptr: UnsafeMutablePointer<UInt16>? =
            encProjIsF16
            ? encoderProj.dataPointer.bindMemory(to: UInt16.self, capacity: encoderProj.count)
            : nil
        let srcF32Ptr: UnsafeMutablePointer<Float>? =
            encProjIsF16
            ? nil
            : encoderProj.dataPointer.bindMemory(to: Float.self, capacity: encoderProj.count)
        let dstPtr = encProjBatchBuf.dataPointer.bindMemory(to: Float.self, capacity: encProjBatchBuf.count)

        var t = 0
        while t < numEncoderFrames {
            let kActual = min(K, numEncoderFrames - t)

            // ── Step 1: run decoder once with current state to get dec_out.
            let tokInput: MLMultiArray
            if let buf = tokenInputBuf {
                buf[0] = NSNumber(value: currentToken)
                tokInput = buf
            } else {
                tokInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                tokInput[0] = NSNumber(value: currentToken)
            }
            let tokLen: MLMultiArray
            if let buf = tokenLenBuf {
                tokLen = buf
            } else {
                tokLen = try MLMultiArray(shape: [1], dataType: .int32)
                tokLen[0] = 1
            }

            let decInput = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokInput),
                "token_length": MLFeatureValue(multiArray: tokLen),
                "h_in": MLFeatureValue(multiArray: currentH),
                "c_in": MLFeatureValue(multiArray: currentC),
            ])
            let decOutput: MLFeatureProvider
            if let opts = decoderPredictionOptions {
                decOutput = try await decoder.prediction(from: decInput, options: opts)
            } else {
                decOutput = try await decoder.prediction(from: decInput)
            }
            guard let decOutRaw = decOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                let candidateH = decOutput.featureValue(for: "h_out")?.multiArrayValue,
                let candidateC = decOutput.featureValue(for: "c_out")?.multiArrayValue
            else {
                throw ASRError.processingFailed("Speculative decoder failed")
            }
            // Slice decoder_out [B, D, U] → [B, D, 1] (handles export-shape variance)
            let decOut = try sliceDecoderOutput(decOutRaw)

            // ── Step 2: copy encoder_proj[:, t:t+kActual, :] into batched buf.
            // (pad the unused [kActual..K) slots with zeros for safe joint call)
            for k in 0..<K {
                let srcT = t + k
                let dstBase = k * projDim
                if k < kActual {
                    if encProjIsF16 {
                        for d in 0..<projDim {
                            let srcIdx = 0 * encProjStride0 + srcT * encProjStride1 + d * encProjStride2
                            dstPtr[dstBase + d] = nemotronHalfBitsToFloat(srcF16Ptr![srcIdx])
                        }
                    } else {
                        for d in 0..<projDim {
                            let srcIdx = 0 * encProjStride0 + srcT * encProjStride1 + d * encProjStride2
                            dstPtr[dstBase + d] = srcF32Ptr![srcIdx]
                        }
                    }
                } else {
                    // Padding zone — zero out (won't be read in the for-loop below either)
                    for d in 0..<projDim {
                        dstPtr[dstBase + d] = 0
                    }
                }
            }

            // ── Step 3: batched joint over K encoder_proj frames.
            let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_proj": MLFeatureValue(multiArray: encProjBatchBuf),
                "decoder": MLFeatureValue(multiArray: decOut),
            ])
            let jointOutput: MLFeatureProvider
            if let opts = jointNoEncProjBatchedPredictionOptions {
                jointOutput = try await jointBatched.prediction(from: jointInput, options: opts)
            } else {
                jointOutput = try await jointBatched.prediction(from: jointInput)
            }
            guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("Speculative joint_noencproj_batched failed")
            }
            // logits shape: [1, K, 1, V]
            let logitsStride0 = logits.strides[0].intValue
            let logitsStride1 = logits.strides[1].intValue
            let logitsStride2 = logits.strides[2].intValue
            let logitsStride3 = logits.strides[3].intValue
            let vocabSize = logits.shape[3].intValue
            let logitsIsF16 = (logits.dataType == .float16)
            let logitsF16Ptr: UnsafeMutablePointer<UInt16>? =
                logitsIsF16
                ? logits.dataPointer.bindMemory(to: UInt16.self, capacity: logits.count)
                : nil
            let logitsF32Ptr: UnsafeMutablePointer<Float>? =
                logitsIsF16
                ? nil
                : logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)

            // ── Step 4: scan for first non-blank in kActual frames.
            // Fast path: Float32 logits with vocab-stride=1 → vDSP_maxvi
            // (SIMD argmax). Falls back to scalar loop for FP16 or strided.
            let vocabContiguous = (logitsStride3 == 1)
            var firstNonBlankAt = -1
            var emittedToken = blankIdx
            for kk in 0..<kActual {
                var bestIdx = blankIdx
                let frameBase = 0 * logitsStride0 + kk * logitsStride1 + 0 * logitsStride2
                if !logitsIsF16, vocabContiguous, let f32 = logitsF32Ptr {
                    var maxVal: Float = 0
                    var maxIdx: vDSP_Length = 0
                    vDSP_maxvi(f32.advanced(by: frameBase), 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
                    bestIdx = Int(maxIdx)
                } else {
                    var bestVal: Float = -.greatestFiniteMagnitude
                    if logitsIsF16 {
                        for v in 0..<vocabSize {
                            let val = nemotronHalfBitsToFloat(logitsF16Ptr![frameBase + v * logitsStride3])
                            if val > bestVal {
                                bestVal = val
                                bestIdx = v
                            }
                        }
                    } else {
                        for v in 0..<vocabSize {
                            let val = logitsF32Ptr![frameBase + v * logitsStride3]
                            if val > bestVal {
                                bestVal = val
                                bestIdx = v
                            }
                        }
                    }
                }
                if bestIdx != blankIdx {
                    firstNonBlankAt = kk
                    emittedToken = bestIdx
                    break
                }
            }

            // E4 instrumentation: count this speculation window.
            self.specWindowsTotal &+= 1
            if firstNonBlankAt == -1 {
                // All-blank streak — fast skip
                self.specWindowsAllBlank &+= 1
                t += kActual
            } else {
                self.specWindowsHitNonBlank &+= 1
                // Emit first non-blank from speculative scan.
                newTokens.append(emittedToken)
                accumulatedTokenIds.append(emittedToken)
                currentToken = Int32(emittedToken)
                currentH = candidateH
                currentC = candidateC

                if config.langTagTokenIds.contains(emittedToken),
                    let piece = tokenizerPiece(forId: emittedToken, tokenizer: tokenizer)
                {
                    let lang = NemotronMultilingualTokenizer.stripAngleBrackets(piece)
                    recordDetectedLanguage(lang)
                }

                // MULTI-EMISSION DRAIN: standard RNN-T allows up to 10
                // emissions per encoder frame. After the speculative scan
                // finds the FIRST non-blank, fall back to per-frame loop
                // AT THIS FRAME until blank (max 9 more).
                //
                // OPTIMIZED: use B1 fusion (decoder_joint.mlpackage,
                // single CoreML call per emission) for the drain rather
                // than re-using joint_noencproj_batched at K=8 (8×
                // wasteful — model computes K logits we read 1 of).
                // B1 path takes encoder [1, 1024, 1] + token + state.
                // We extract encoded[:, :, drainFrameT:drainFrameT+1].
                let drainFrameT = t + firstNonBlankAt
                let drainEncStep: MLMultiArray
                if let buf = encoderStepBuf {
                    fillEncoderStep(into: buf, from: encoded, timeIndex: drainFrameT)
                    drainEncStep = buf
                } else {
                    drainEncStep = try extractEncoderStep(from: encoded, timeIndex: drainFrameT)
                }
                // E7: also slice encoder_proj[:, drainFrameT, :] for B3+B1
                // drain path. Only populated if B3+B1 asset is loaded
                // (`decoderJointNoEncProj != nil`); otherwise drainEncStep
                // alone is sufficient for B2/B1 drain.
                let drainEncProjStep: MLMultiArray?
                if self.decoderJointNoEncProj != nil {
                    if let projBuf = encoderProjStepBuf {
                        fillEncoderProjStep(into: projBuf, from: encoderProj, timeIndex: drainFrameT)
                        drainEncProjStep = projBuf
                    } else {
                        drainEncProjStep = try extractEncoderProjStep(from: encoderProj, timeIndex: drainFrameT)
                    }
                } else {
                    drainEncProjStep = nil
                }

                for _ in 0..<9 {
                    let tokInput2: MLMultiArray
                    if let buf = tokenInputBuf {
                        tokInput2 = buf
                    } else {
                        tokInput2 = try MLMultiArray(shape: [1, 1], dataType: .int32)
                    }
                    tokInput2[0] = NSNumber(value: currentToken)
                    let tokLen2: MLMultiArray
                    if let buf = tokenLenBuf {
                        tokLen2 = buf
                    } else {
                        tokLen2 = try MLMultiArray(shape: [1], dataType: .int32)
                        tokLen2[0] = 1
                    }

                    // Drain priority (May 26 update): B3+B1 (decoder_joint_noencproj)
                    // > B2 (decoder_joint_argmax) > B1 (decoder_joint) > slow
                    // decoder+joint fallback. B3+B1 saves the 1024→640 encoder
                    // projection matmul per drain emission by taking the
                    // pre-projected encoder_proj directly. Earnings22-1h
                    // decoder is 85% of wall time, so per-token drain savings
                    // visibly move the headline.
                    var dBestIdx = blankIdx
                    var newH: MLMultiArray = currentH
                    var newC: MLMultiArray = currentC

                    if let djne = self.decoderJointNoEncProj,
                        let drainProjStep = drainEncProjStep
                    {
                        let djneInput = try MLDictionaryFeatureProvider(dictionary: [
                            "token": MLFeatureValue(multiArray: tokInput2),
                            "token_length": MLFeatureValue(multiArray: tokLen2),
                            "h_in": MLFeatureValue(multiArray: currentH),
                            "c_in": MLFeatureValue(multiArray: currentC),
                            "encoder_proj": MLFeatureValue(multiArray: drainProjStep),
                        ])
                        let djneOutput: MLFeatureProvider
                        if let opts = decoderJointNoEncProjPredictionOptions {
                            djneOutput = try await djne.prediction(from: djneInput, options: opts)
                        } else {
                            djneOutput = try await djne.prediction(from: djneInput)
                        }
                        guard let djneLogits = djneOutput.featureValue(for: "logits")?.multiArrayValue,
                            let djneH = djneOutput.featureValue(for: "h_out")?.multiArrayValue,
                            let djneC = djneOutput.featureValue(for: "c_out")?.multiArrayValue
                        else { throw ASRError.processingFailed("Drain B3+B1 failed") }
                        dBestIdx = findMaxIndex(djneLogits)
                        newH = djneH
                        newC = djneC
                    } else if let dja = self.decoderJointArgmax {
                        let djaInput = try MLDictionaryFeatureProvider(dictionary: [
                            "token": MLFeatureValue(multiArray: tokInput2),
                            "token_length": MLFeatureValue(multiArray: tokLen2),
                            "h_in": MLFeatureValue(multiArray: currentH),
                            "c_in": MLFeatureValue(multiArray: currentC),
                            "encoder": MLFeatureValue(multiArray: drainEncStep),
                        ])
                        let djaOutput: MLFeatureProvider
                        if let opts = decoderJointArgmaxPredictionOptions {
                            djaOutput = try await dja.prediction(from: djaInput, options: opts)
                        } else {
                            djaOutput = try await dja.prediction(from: djaInput)
                        }
                        guard let tokenIdArr = djaOutput.featureValue(for: "token_id")?.multiArrayValue,
                            let djaH = djaOutput.featureValue(for: "h_out")?.multiArrayValue,
                            let djaC = djaOutput.featureValue(for: "c_out")?.multiArrayValue
                        else { throw ASRError.processingFailed("Drain B2 failed") }
                        dBestIdx = Int(tokenIdArr[0].int32Value)
                        newH = djaH
                        newC = djaC
                    } else if let dj = self.decoderJoint {
                        let djInput = try MLDictionaryFeatureProvider(dictionary: [
                            "token": MLFeatureValue(multiArray: tokInput2),
                            "token_length": MLFeatureValue(multiArray: tokLen2),
                            "h_in": MLFeatureValue(multiArray: currentH),
                            "c_in": MLFeatureValue(multiArray: currentC),
                            "encoder": MLFeatureValue(multiArray: drainEncStep),
                        ])
                        let djOutput: MLFeatureProvider
                        if let opts = decoderJointPredictionOptions {
                            djOutput = try await dj.prediction(from: djInput, options: opts)
                        } else {
                            djOutput = try await dj.prediction(from: djInput)
                        }
                        guard let djLogits = djOutput.featureValue(for: "logits")?.multiArrayValue,
                            let djH = djOutput.featureValue(for: "h_out")?.multiArrayValue,
                            let djC = djOutput.featureValue(for: "c_out")?.multiArrayValue
                        else { throw ASRError.processingFailed("Drain B1 failed") }
                        dBestIdx = findMaxIndex(djLogits)
                        newH = djH
                        newC = djC
                    } else {
                        // Fallback: decoder + separate joint
                        let decIn2 = try MLDictionaryFeatureProvider(dictionary: [
                            "token": MLFeatureValue(multiArray: tokInput2),
                            "token_length": MLFeatureValue(multiArray: tokLen2),
                            "h_in": MLFeatureValue(multiArray: currentH),
                            "c_in": MLFeatureValue(multiArray: currentC),
                        ])
                        let dec2 = try await decoder.prediction(from: decIn2)
                        guard let do2Raw = dec2.featureValue(for: "decoder_out")?.multiArrayValue,
                            let h2 = dec2.featureValue(for: "h_out")?.multiArrayValue,
                            let c2 = dec2.featureValue(for: "c_out")?.multiArrayValue
                        else { throw ASRError.processingFailed("Drain decoder failed") }
                        let decOut2 = try sliceDecoderOutput(do2Raw)
                        let jIn = try MLDictionaryFeatureProvider(dictionary: [
                            "encoder": MLFeatureValue(multiArray: drainEncStep),
                            "decoder": MLFeatureValue(multiArray: decOut2),
                        ])
                        let jOut = try await self.joint!.prediction(from: jIn)
                        guard let jLogits = jOut.featureValue(for: "logits")?.multiArrayValue
                        else { throw ASRError.processingFailed("Drain joint failed") }
                        dBestIdx = findMaxIndex(jLogits)
                        newH = h2
                        newC = c2
                    }

                    if dBestIdx == blankIdx { break }
                    // Non-blank → emit + commit state
                    newTokens.append(dBestIdx)
                    accumulatedTokenIds.append(dBestIdx)
                    currentToken = Int32(dBestIdx)
                    currentH = newH
                    currentC = newC
                    if config.langTagTokenIds.contains(dBestIdx),
                        let piece = tokenizerPiece(forId: dBestIdx, tokenizer: tokenizer)
                    {
                        let lang = NemotronMultilingualTokenizer.stripAngleBrackets(piece)
                        recordDetectedLanguage(lang)
                    }
                }

                t = t + firstNonBlankAt + 1
            }
        }
    }

    /// Speculative-blank RNN-T inner loop. Replaces the per-token loop when
    /// `joint_batched.mlpackage` is loaded.
    ///
    /// Standard greedy RNN-T per chunk does:
    ///     for t in 0..<numEncoderFrames:
    ///         while not blank: decoder + joint + argmax → token
    /// → roughly (numEncoderFrames + num_non_blank_emissions) CoreML calls
    /// per chunk. On clean speech that's ~56-87 calls/chunk.
    ///
    /// This speculative variant computes joint over ALL remaining frames in
    /// one ANE call (assuming dec_out doesn't change — true while emitting
    /// blanks, which is ~70-80% of the time). Then walks the predictions
    /// linearly to find the first non-blank. When found: emit it, recompute
    /// dec_out from the new LSTM state, re-batch joint for the remaining
    /// frames. All-blanks streaks consume zero per-frame CoreML calls.
    ///
    /// Semantic parity argument:
    /// - Blank predictions made with stale dec_out are STILL valid, because
    ///   blank emission doesn't advance LSTM state in RNN-T. So if frames
    ///   [t..i-1] all predicted blank under dec_out_t, they would also have
    ///   predicted blank under any intermediate state (none was advanced).
    /// - The first non-blank prediction at frame i is correct because the
    ///   model has not yet emitted anything for frames [t..i-1], so dec_out
    ///   at frame i in the standard loop would be IDENTICAL to dec_out_t.
    /// - After emitting at frame i, predictions for frames [i+1..end] used
    ///   the now-stale dec_out_t — we discard them and recompute.
    ///
    /// Bottom line: emissions are bit-identical to the standard greedy loop.
    /// WER is unchanged. CoreML call count drops ~3-10x.
    internal func runSpeculativeBatchedDecode(
        encoded: MLMultiArray,
        numEncoderFrames: Int,
        decoder: MLModel,
        jointBatched: MLModel,
        currentH: inout MLMultiArray,
        currentC: inout MLMultiArray,
        currentToken: inout Int32,
        newTokens: inout [Int],
        tokenizer: NemotronMultilingualTokenizer
    ) async throws {
        var t = 0
        var symbolsAtCurrentFrame = 0
        let maxSymbolsPerFrame = 10
        while t < numEncoderFrames {
            // Safety: cap emissions at a single encoder frame (mirrors the
            // 0..<10 loop in the standard per-token path). If a frame keeps
            // emitting non-blank tokens without hitting blank, force-advance
            // to prevent infinite loop.
            if symbolsAtCurrentFrame >= maxSymbolsPerFrame {
                t += 1
                symbolsAtCurrentFrame = 0
                continue
            }
            // 1. decoder.predict(currentToken, h, c) → dec_out, h_out, c_out
            let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenInput[0] = NSNumber(value: currentToken)
            let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
            tokenLen[0] = 1
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
                throw ASRError.processingFailed("Speculative decoder failed")
            }
            let decoderStep = try sliceDecoderOutput(decoderOut)  // [1, 640, 1]

            // 2. jointBatched.predict(encoded[all frames], decoderStep) →
            //    logits [1, numEncoderFrames, 1, vocab]
            //    We always pass the full encoded tensor; we ignore predictions
            //    for frames < t. Small wasted compute on the unused tail, but
            //    keeps the input shape fixed at the converter-baked size.
            let jbInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder": MLFeatureValue(multiArray: encoded),
                "decoder": MLFeatureValue(multiArray: decoderStep),
            ])
            let jbOutput = try await jointBatched.prediction(from: jbInput)
            guard let logits = jbOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("joint_batched failed")
            }

            // 3. Walk frames [t..numEncoderFrames]; find first non-blank.
            let blankIdx = config.blankIdx
            var firstNonBlankIdx: Int? = nil
            var firstNonBlankToken: Int = -1
            for i in t..<numEncoderFrames {
                let token = argmaxFrame(logits: logits, frame: i)
                if token != blankIdx {
                    firstNonBlankIdx = i
                    firstNonBlankToken = token
                    break
                }
            }

            if let nbIdx = firstNonBlankIdx {
                // Emit token at frame nbIdx. LSTM state advances.
                newTokens.append(firstNonBlankToken)
                accumulatedTokenIds.append(firstNonBlankToken)
                currentToken = Int32(firstNonBlankToken)
                currentH = dh
                currentC = dc

                // Surface lang-tag piece for detectedLanguage() observers.
                if config.langTagTokenIds.contains(firstNonBlankToken),
                    let piece = tokenizerPiece(forId: firstNonBlankToken, tokenizer: tokenizer)
                {
                    let lang = NemotronMultilingualTokenizer.stripAngleBrackets(piece)
                    recordDetectedLanguage(lang)
                }

                // Reset symbol counter when advancing to a new encoder frame
                // (nbIdx > t means we skipped past some blanks).
                if nbIdx > t {
                    symbolsAtCurrentFrame = 0
                }
                // Stay at this frame — same frame may emit more tokens with
                // the new dec_out. (RNN-T greedy: keep emitting non-blanks
                // at frame t until blank, then advance.)
                t = nbIdx
                symbolsAtCurrentFrame += 1
            } else {
                // All remaining frames predicted blank — done with chunk.
                break
            }
        }
    }

    /// Argmax over the `vocab` axis at a specific frame of the batched joint
    /// output. logits shape: [1, numEncoderFrames, 1, vocab].
    @inline(__always)
    internal func argmaxFrame(logits: MLMultiArray, frame: Int) -> Int {
        let vocab = logits.shape[3].intValue
        let stride0 = logits.strides[0].intValue
        let stride1 = logits.strides[1].intValue
        let stride2 = logits.strides[2].intValue
        let stride3 = logits.strides[3].intValue
        let base = 0 * stride0 + frame * stride1 + 0 * stride2

        // Read with the model's actual element type. Hardcoding Float16 here
        // reinterprets Float32 logits as garbage (→ wrong tokens / ~100% WER)
        // whenever the joint emits Float32. Mirror runSpeculativeBlankDecodeV2's
        // dtype-aware handling.
        func scan<T: Comparable>(_ ptr: UnsafeMutablePointer<T>) -> Int {
            var bestIdx = 0
            var bestVal = ptr[base]
            for v in 1..<vocab {
                let val = ptr[base + v * stride3]
                if val > bestVal {
                    bestVal = val
                    bestIdx = v
                }
            }
            return bestIdx
        }

        if logits.dataType == .float16 {
            // x86_64 has no `Float16: Comparable`; scan the raw UInt16 bits and
            // convert each candidate to Float for the max. (arm64 path is
            // unchanged numerically — same native conversion under the hood.)
            let ptr = logits.dataPointer.bindMemory(to: UInt16.self, capacity: logits.count)
            var bestIdx = 0
            var bestVal = nemotronHalfBitsToFloat(ptr[base])
            for v in 1..<vocab {
                let val = nemotronHalfBitsToFloat(ptr[base + v * stride3])
                if val > bestVal {
                    bestVal = val
                    bestIdx = v
                }
            }
            return bestIdx
        } else {
            return scan(logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count))
        }
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
    /// Reuses caller-provided `audioInputBuf` / `audioLenBuf` when supplied
    /// and shape-compatible; otherwise falls back to fresh allocation.
    nonisolated internal static func runPreprocessorPure(
        samples: [Float],
        preprocessor: MLModel,
        audioInputBuf: MLMultiArray? = nil,
        audioLenBuf: MLMultiArray? = nil
    ) async throws -> MLMultiArray? {
        let array: MLMultiArray
        let audioLen: MLMultiArray
        if let buf = audioInputBuf,
            buf.shape[1].intValue == samples.count,
            let lenBuf = audioLenBuf
        {
            let ptr = buf.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
            ptr.update(from: samples, count: samples.count)
            lenBuf[0] = NSNumber(value: samples.count)
            array = buf
            audioLen = lenBuf
        } else {
            array = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
            ptr.update(from: samples, count: samples.count)
            audioLen = try MLMultiArray(shape: [1], dataType: .int32)
            audioLen[0] = NSNumber(value: samples.count)
        }

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
        preprocessor: MLModel,
        encoder: MLModel,
        audioInputBuf: MLMultiArray? = nil,
        audioLenBuf: MLMultiArray? = nil
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
            let chunkMel = try await runPreprocessorPure(
                samples: samples,
                preprocessor: preprocessor,
                audioInputBuf: audioInputBuf,
                audioLenBuf: audioLenBuf
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
