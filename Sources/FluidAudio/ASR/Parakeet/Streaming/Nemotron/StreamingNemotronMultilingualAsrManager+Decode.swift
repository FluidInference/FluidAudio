import Accelerate
@preconcurrency import CoreML
import Foundation

/// Split out of `StreamingNemotronMultilingualAsrManager+Pipeline.swift`
/// — same extension, grouped by concern. No logic change.
extension StreamingNemotronMultilingualAsrManager {

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
                // This token was found at encoder frame t + firstNonBlankAt.
                appendTokenTiming(
                    emittedToken, frameInChunk: t + firstNonBlankAt, tokenizer: tokenizer)
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
                    // Multi-emission drain stays on the same frame: drainFrameT.
                    appendTokenTiming(dBestIdx, frameInChunk: drainFrameT, tokenizer: tokenizer)
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
    internal func tokenizerPiece(forId id: Int, tokenizer: NemotronMultilingualTokenizer) -> String? {
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

    /// Append a per-token timing for a token emitted at encoder frame
    /// `frameInChunk` within the current chunk. `startTime` is absolute seconds
    /// from the start of the fed audio (`absoluteFrameBase` carries prior chunks).
    /// Lang-tag tokens are skipped: they are stripped from the decoded transcript,
    /// so excluding them keeps the timing stream aligned 1:1 with the user-visible
    /// tokens. Uses `rawToken(for:)` (NOT `tokenizerPiece`, which strips the `▁`
    /// word-boundary marker) so callers can reconstruct word boundaries.
    internal func appendTokenTiming(
        _ tokenId: Int, frameInChunk: Int, tokenizer: NemotronMultilingualTokenizer
    ) {
        guard !config.langTagTokenIds.contains(tokenId) else { return }
        let startTime =
            Double(absoluteFrameBase + frameInChunk) * ASRConstants.secondsPerEncoderFrame
        accumulatedTokenTimings.append(
            TokenTiming(
                token: tokenizer.rawToken(for: tokenId) ?? "",
                tokenId: tokenId,
                startTime: startTime,
                endTime: startTime + ASRConstants.secondsPerEncoderFrame,
                confidence: 1.0
            )
        )
    }
}
