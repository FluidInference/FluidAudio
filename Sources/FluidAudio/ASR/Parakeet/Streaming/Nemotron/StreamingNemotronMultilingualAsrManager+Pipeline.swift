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
                    // The skipped chunk's audio still elapses on the file
                    // timeline. No encoder ran, so advance the frame base by the
                    // chunk's nominal encoder-frame count (derived from sample
                    // count) to keep later token timings aligned. The processed
                    // path asserts this nominal count equals the encoder's actual
                    // output, so skipped and processed chunks stay on one timeline.
                    absoluteFrameBase += samples.count / ASRConstants.samplesPerEncoderFrame
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
        // AIDEV-NOTE: timing-invariant; processChunk always receives exactly
        // config.chunkSamples (process() slices full chunks; finish() zero-pads
        // the tail), so the encoder's actual frame count must equal the nominal
        // sample-derived count the VAD-skip branch above uses to advance
        // absoluteFrameBase. Assert it so a future model/tier whose CoreML
        // encoder emits a different frame count is caught in debug/CI instead of
        // silently drifting skipped-chunk token timings.
        assert(
            numEncoderFrames == samples.count / ASRConstants.samplesPerEncoderFrame,
            "Nemotron encoder emitted \(numEncoderFrames) frames for \(samples.count) "
                + "samples; expected \(samples.count / ASRConstants.samplesPerEncoderFrame). "
                + "VAD-skip frame accounting assumes encoder frames == nominal."
        )
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
            // Advance the encoder-frame base by this chunk's frame count so the
            // next chunk's token timings continue on the same timeline.
            absoluteFrameBase += numEncoderFrames
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
                    // Legacy per-frame loop: this token was emitted at frame t.
                    appendTokenTiming(predToken, frameInChunk: t, tokenizer: tokenizer)
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

        // Advance the encoder-frame base by this chunk's frame count so the next
        // chunk's token timings continue on the same timeline.
        absoluteFrameBase += numEncoderFrames
        processedChunks += 1
    }
}
