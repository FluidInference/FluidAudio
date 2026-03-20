import AVFoundation
import Accelerate
@preconcurrency import CoreML
import Foundation

/// VoxCPM 1.5 diffusion autoregressive TTS synthesizer.
///
/// Generates 44.1kHz audio from text using 6 CoreML models:
/// audio_vae_encoder, audio_vae_decoder, feat_encoder,
/// base_lm_step, residual_lm_step, locdit_step.
///
/// Pipeline: tokenize → encode prompt → prefill KV → generate loop → decode
public struct VoxCpmSynthesizer {

    static let logger = AppLogger(category: "VoxCpmSynthesizer")

    private enum Context {
        @TaskLocal static var modelStore: VoxCpmModelStore?
    }

    static func withModelStore<T>(
        _ store: VoxCpmModelStore,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelStore.withValue(store) {
            try await operation()
        }
    }

    static func currentModelStore() throws -> VoxCpmModelStore {
        guard let store = Context.modelStore else {
            throw VoxCpmError.processingFailed(
                "VoxCpmSynthesizer requires a model store context.")
        }
        return store
    }

    // MARK: - Public API

    /// Synthesize speech from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - promptAudioURL: Optional URL to prompt audio for voice cloning (WAV, 44.1kHz preferred).
    ///   - promptText: Transcript of prompt audio (required when using prompt audio).
    ///   - maxLen: Maximum generation steps (default: 200).
    ///   - minLen: Minimum steps before stop head can trigger (default: 5).
    ///   - seed: Optional random seed for reproducible voice generation.
    /// - Returns: A synthesis result containing WAV audio data at 44.1kHz.
    public static func synthesize(
        text: String,
        promptAudioURL: URL? = nil,
        promptText: String? = nil,
        maxLen: Int = VoxCpmConstants.defaultMaxLen,
        minLen: Int = VoxCpmConstants.defaultMinLen,
        seed: UInt64? = nil
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()

        let effectiveSeed = seed ?? UInt64.random(in: 0...UInt64.max)
        var rng = SeededRandomNumberGenerator(seed: effectiveSeed)
        logger.info("VoxCPM synthesizing: '\(text)' (seed=\(effectiveSeed))")

        let constants = try await store.constants()

        // Load models
        let vaeEnc = try await store.audioVaeEncoder()
        let vaeDec = try await store.audioVaeDecoder()
        let featEnc = try await store.featEncoder()
        let baseLm = try await store.baseLmStep()
        let resLm = try await store.residualLmStep()
        let locdit = try await store.locditStep()

        // Tokenize
        let textIds: [Int]
        let promptTextIds: [Int]?
        if let promptText = promptText {
            let pIds = constants.tokenizer.encode(promptText)
            promptTextIds = pIds
            textIds = constants.tokenizer.encode(text)
            logger.info("  Prompt text: \(pIds.count) tokens, text: \(textIds.count) tokens")
        } else {
            promptTextIds = nil
            textIds = constants.tokenizer.encode(text)
            logger.info("  Text: \(textIds.count) tokens")
        }

        // Encode prompt audio (if provided)
        let promptLatent: MLMultiArray?
        if let audioURL = promptAudioURL {
            let samples = try loadAudioSamples(from: audioURL)
            promptLatent = try encodePromptAudio(samples: samples, model: vaeEnc)
            logger.info("  Prompt latent: \(promptLatent!.shape)")
        } else {
            promptLatent = nil
            logger.info("  No prompt audio (unconditioned)")
        }

        // Encode features and build embeddings
        let (featLmEmb, nPatches, prefixCond) = try encodePromptFeatures(
            latent: promptLatent,
            constants: constants,
            featEncoder: featEnc
        )

        // Build text embeddings
        var allTextIds: [Int]
        if let pIds = promptTextIds {
            allTextIds = pIds + textIds + [VoxCpmConstants.audioStartToken]
        } else {
            allTextIds = textIds + [VoxCpmConstants.audioStartToken]
        }
        let textLen = allTextIds.count
        let textEmb = embedTokens(allTextIds, constants: constants)

        // Combine sequence: [text_emb, feat_lm_emb]
        let seqLen = textLen + nPatches
        logger.info("  Combined sequence: \(seqLen) tokens (text=\(textLen), audio=\(nPatches))")

        guard seqLen < VoxCpmConstants.maxSeqLen else {
            throw VoxCpmError.processingFailed(
                "Sequence length \(seqLen) exceeds max \(VoxCpmConstants.maxSeqLen)")
        }

        // Sequential prefill: process tokens one at a time through base_lm_step.
        // This uses the same model for both prefill and generation, ensuring
        // KV cache compatibility and correct stop head behavior.
        logger.info("  Prefilling...")
        var caches = try createCacheState()
        var lmHiddenFsq: MLMultiArray!
        var stopLogit: MLMultiArray!
        var resHidden: MLMultiArray!

        let prefillStart = Date()
        for pos in 0..<seqLen {
            let tokenEmb: MLMultiArray
            if pos < textLen {
                tokenEmb = try floatArrayToMLMultiArray(textEmb[pos], shape: [1, 1024])
            } else {
                let audioIdx = pos - textLen
                tokenEmb = try floatArrayToMLMultiArray(
                    featLmEmb[audioIdx], shape: [1, 1024])
            }

            let baseResult = try runBaseLmStep(
                embed: tokenEmb, position: pos, caches: &caches, model: baseLm)
            let lmHidden = baseResult.lmHidden
            lmHiddenFsq = baseResult.lmHiddenFsq
            stopLogit = baseResult.stopLogit

            // Residual LM input differs for text vs audio positions
            let resInput: MLMultiArray
            let isAudioPos = pos >= textLen
            if isAudioPos {
                let audioIdx = pos - textLen
                resInput = try addArrays(
                    lmHiddenFsq,
                    try floatArrayToMLMultiArray(featLmEmb[audioIdx], shape: [1, 1024]))
            } else {
                resInput = lmHidden
            }

            resHidden = try runResidualLmStep(
                embed: resInput, position: pos, caches: &caches, model: resLm)
        }
        let prefillTime = Date().timeIntervalSince(prefillStart)
        logger.info(
            "  Prefill: \(seqLen) tokens in \(String(format: "%.1f", prefillTime))s (\(String(format: "%.1f", Double(seqLen) / prefillTime)) tok/s)"
        )

        // Effective minimum steps: use the caller's minLen but ensure it's at least
        // proportional to text length. Each text token produces ~2 audio patches on average.
        let textBasedMinLen = max(minLen, textLen * 2)
        logger.info(
            "  minLen=\(textBasedMinLen) (text-based=\(textLen * 2), caller=\(minLen))")

        // Generate
        logger.info("  Generating...")
        var generatedLatents: [MLMultiArray] = []
        var pos = seqLen
        var currentPrefixCond = prefixCond
        var currentLmHiddenFsq = lmHiddenFsq!
        var currentResHidden = resHidden!
        var currentStopLogit = stopLogit!

        let genStart = Date()
        for step in 0..<maxLen {
            if pos >= VoxCpmConstants.maxSeqLen {
                logger.info("  Max KV cache position reached at step \(step)")
                break
            }

            // dit_hidden = lm_to_dit_proj(fsq) + res_to_dit_proj(res_hidden)
            let ditHidden = try computeDitHidden(
                lmHiddenFsq: currentLmHiddenFsq, resHidden: currentResHidden,
                constants: constants)

            // Run LocDiT diffusion
            let predFeat = try runLocDiT(mu: ditHidden, cond: currentPrefixCond, model: locdit, rng: &rng)
            generatedLatents.append(predFeat)

            // Check stop
            let stopPtr = currentStopLogit.dataPointer.bindMemory(to: Float.self, capacity: 2)
            if step >= textBasedMinLen && stopPtr[1] > stopPtr[0] {
                logger.info("  Stop at step \(step + 1)")
                break
            }

            // Update prefix conditioning
            currentPrefixCond = predFeat

            // Encode predicted feature for next step
            let predPatch4d = try transposeToPatch4d(predFeat)
            let currEmb = try encodeSingleFeature(predPatch4d, model: featEnc)
            let currLmEmb = linear(currEmb, w: constants.encToLmProjW, b: constants.encToLmProjB)
            let currLmEmbML = try floatArrayToMLMultiArray(currLmEmb, shape: [1, 1024])

            // Run base LM step
            let baseResult = try runBaseLmStep(
                embed: currLmEmbML, position: pos, caches: &caches, model: baseLm)
            currentLmHiddenFsq = baseResult.lmHiddenFsq
            currentStopLogit = baseResult.stopLogit

            // Run residual LM step: input = fsq(lm_hidden) + feat_embed
            let resInput = try addArrays(currentLmHiddenFsq, currLmEmbML)
            currentResHidden = try runResidualLmStep(
                embed: resInput, position: pos, caches: &caches, model: resLm)

            pos += 1

            if (step + 1) % 10 == 0 {
                let elapsed = Date().timeIntervalSince(genStart)
                logger.info(
                    "  Step \(step + 1): \(String(format: "%.1f", elapsed))s (\(String(format: "%.2f", Double(step + 1) / elapsed)) steps/s)"
                )
            }
        }

        let genTime = Date().timeIntervalSince(genStart)
        let nSteps = generatedLatents.count
        logger.info(
            "  Generated \(nSteps) steps in \(String(format: "%.1f", genTime))s (\(String(format: "%.2f", Double(nSteps) / genTime)) steps/s)"
        )

        // Decode
        logger.info("  Decoding audio...")
        let allLatents = try concatenateLatents(generatedLatents)
        let audioSamples = try decodeAudio(latent: allLatents, model: vaeDec)

        let duration = Double(audioSamples.count) / Double(VoxCpmConstants.audioSampleRate)
        logger.info("  Audio: \(audioSamples.count) samples (\(String(format: "%.2f", duration))s)")

        let audioData = try AudioWAV.data(
            from: audioSamples,
            sampleRate: Double(VoxCpmConstants.audioSampleRate)
        )

        return SynthesisResult(
            audio: audioData,
            samples: audioSamples,
            patchCount: nSteps,
            duration: duration
        )
    }

    // MARK: - Audio Encoding

    /// Load audio samples from a URL and resample to 44.1kHz mono.
    ///
    /// Uses AudioConverter for high-quality resampling via AVAudioConverter.
    private static func loadAudioSamples(from url: URL) throws -> [Float] {
        guard
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(VoxCpmConstants.audioSampleRate),
                channels: 1,
                interleaved: false
            )
        else {
            throw VoxCpmError.processingFailed("Failed to create target audio format")
        }

        let converter = AudioConverter(targetFormat: targetFormat)
        let samples = try converter.resampleAudioFile(url)

        guard !samples.isEmpty else {
            throw VoxCpmError.processingFailed("Audio file contains no samples")
        }

        return samples
    }

    /// Encode prompt audio through audio_vae_encoder.
    ///
    /// Pads/truncates to fixed 5s input, returns latent [1, 64, T].
    private static func encodePromptAudio(
        samples: [Float],
        model: MLModel
    ) throws -> MLMultiArray {
        let encoderSamples = VoxCpmConstants.encoderSamples
        var audio = samples

        // Truncate to last 5s
        if audio.count > encoderSamples {
            audio = Array(audio.suffix(encoderSamples))
        }

        let validSamples = audio.count
        // Pad to encoder size
        if audio.count < encoderSamples {
            audio.append(contentsOf: [Float](repeating: 0, count: encoderSamples - audio.count))
        }

        // Create input [1, 1, 220500]
        let shape: [NSNumber] = [1, 1, NSNumber(value: encoderSamples)]
        let input = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: encoderSamples)
        audio.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            ptr.update(from: base, count: encoderSamples)
        }

        let inputDict = try MLDictionaryFeatureProvider(dictionary: [
            VaeEncoderKeys.audio: MLFeatureValue(multiArray: input)
        ])

        let pred = try model.prediction(from: inputDict)
        let latent = try ensureFloat32(
            pred.featureValue(for: VaeEncoderKeys.latent)!.multiArrayValue!)

        // Trim to valid frames (remove padding artifacts)
        let nValidFrames = validSamples / VoxCpmConstants.hopLength
        if nValidFrames > 0 && nValidFrames < latent.shape[2].intValue {
            return try trimLatent(latent, toFrames: nValidFrames)
        }

        return latent
    }

    /// Trim latent to first `n` frames along dimension 2.
    private static func trimLatent(_ latent: MLMultiArray, toFrames n: Int) throws -> MLMultiArray {
        let featDim = latent.shape[1].intValue
        let shape: [NSNumber] = [1, NSNumber(value: featDim), NSNumber(value: n)]
        let result = try MLMultiArray(shape: shape, dataType: .float32)
        let srcPtr = latent.dataPointer.bindMemory(
            to: Float.self, capacity: featDim * latent.shape[2].intValue)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: featDim * n)

        let srcStride = latent.shape[2].intValue
        for c in 0..<featDim {
            for t in 0..<n {
                dstPtr[c * n + t] = srcPtr[c * srcStride + t]
            }
        }

        return result
    }

    // MARK: - Feature Encoding

    /// Encode prompt latent into feature embeddings and project to LM space.
    ///
    /// Returns (feat_lm_emb: [[Float]], n_patches: Int, prefix_cond: MLMultiArray).
    private static func encodePromptFeatures(
        latent: MLMultiArray?,
        constants: VoxCpmConstantsBundle,
        featEncoder: MLModel
    ) throws -> ([[Float]], Int, MLMultiArray) {
        let patchSize = VoxCpmConstants.patchSize
        let featDim = VoxCpmConstants.featDim

        guard let latent = latent else {
            // No prompt: zero conditioning
            let zeroShape: [NSNumber] = [1, NSNumber(value: featDim), NSNumber(value: patchSize)]
            let zeroCond = try MLMultiArray(shape: zeroShape, dataType: .float32)
            let ptr = zeroCond.dataPointer.bindMemory(
                to: Float.self, capacity: featDim * patchSize)
            memset(ptr, 0, featDim * patchSize * MemoryLayout<Float>.size)
            return ([], 0, zeroCond)
        }

        let totalFrames = latent.shape[2].intValue
        let nPatches = totalFrames / patchSize
        let latentPtr = latent.dataPointer.bindMemory(
            to: Float.self, capacity: featDim * totalFrames)

        var featEmbeddings: [[Float]] = []
        for p in 0..<nPatches {
            // Extract patch [1, 1, 4, 64] from latent [1, 64, T]
            let patch = try extractPatch(latentPtr, totalFrames: totalFrames, patchIndex: p)
            let emb = try encodeSingleFeature(patch, model: featEncoder)
            featEmbeddings.append(emb)
        }

        // Project: feat_lm = enc_to_lm_proj(feat_emb)
        let featLmEmb = featEmbeddings.map { emb in
            linear(emb, w: constants.encToLmProjW, b: constants.encToLmProjB)
        }

        // Prefix conditioning: last patch of the latent [1, 64, patchSize]
        let condShape: [NSNumber] = [1, NSNumber(value: featDim), NSNumber(value: patchSize)]
        let prefixCond = try MLMultiArray(shape: condShape, dataType: .float32)
        let condPtr = prefixCond.dataPointer.bindMemory(
            to: Float.self, capacity: featDim * patchSize)
        let startFrame = (nPatches - 1) * patchSize
        for c in 0..<featDim {
            for t in 0..<patchSize {
                condPtr[c * patchSize + t] = latentPtr[c * totalFrames + startFrame + t]
            }
        }

        return (featLmEmb, nPatches, prefixCond)
    }

    /// Extract a single patch [1, 1, 4, 64] from latent [1, 64, T].
    ///
    /// The latent is [1, 64, T] (channels × time). We need [1, 1, patchSize, 64]
    /// which is [batch, 1, time_slice, channels] — a transpose of the slice.
    private static func extractPatch(
        _ latentPtr: UnsafePointer<Float>,
        totalFrames: Int,
        patchIndex: Int
    ) throws -> MLMultiArray {
        let patchSize = VoxCpmConstants.patchSize
        let featDim = VoxCpmConstants.featDim

        let shape: [NSNumber] = [1, 1, NSNumber(value: patchSize), NSNumber(value: featDim)]
        let patch = try MLMultiArray(shape: shape, dataType: .float32)
        let patchPtr = patch.dataPointer.bindMemory(
            to: Float.self, capacity: patchSize * featDim)

        let startFrame = patchIndex * patchSize
        // Transpose: latent[c, t] → patch[t, c]
        for t in 0..<patchSize {
            for c in 0..<featDim {
                patchPtr[t * featDim + c] = latentPtr[c * totalFrames + startFrame + t]
            }
        }

        return patch
    }

    /// Encode a single patch [1, 1, 4, 64] through feat_encoder → [1024].
    private static func encodeSingleFeature(
        _ patch: MLMultiArray, model: MLModel
    ) throws -> [Float] {
        let inputDict = try MLDictionaryFeatureProvider(dictionary: [
            FeatEncoderKeys.feat: MLFeatureValue(multiArray: patch)
        ])
        let pred = try model.prediction(from: inputDict)
        let emb = try ensureFloat32(
            pred.featureValue(for: FeatEncoderKeys.embedding)!.multiArrayValue!)

        // emb is [1, 1, 1024] — extract the [1024] vector
        let dim = VoxCpmConstants.hiddenSize
        let embPtr = emb.dataPointer.bindMemory(to: Float.self, capacity: dim)
        return Array(UnsafeBufferPointer(start: embPtr, count: dim))
    }

    /// Transpose predFeat [1, 64, 4] → patch4d [1, 1, 4, 64].
    private static func transposeToPatch4d(_ predFeat: MLMultiArray) throws -> MLMultiArray {
        let patchSize = VoxCpmConstants.patchSize
        let featDim = VoxCpmConstants.featDim

        let shape: [NSNumber] = [1, 1, NSNumber(value: patchSize), NSNumber(value: featDim)]
        let patch = try MLMultiArray(shape: shape, dataType: .float32)
        let srcPtr = predFeat.dataPointer.bindMemory(
            to: Float.self, capacity: featDim * patchSize)
        let dstPtr = patch.dataPointer.bindMemory(
            to: Float.self, capacity: patchSize * featDim)

        // [64, 4] → [4, 64]: transpose
        for c in 0..<featDim {
            for t in 0..<patchSize {
                dstPtr[t * featDim + c] = srcPtr[c * patchSize + t]
            }
        }

        return patch
    }

    // MARK: - Audio Decoding

    /// Concatenate generated latent patches along the time dimension.
    ///
    /// Each patch is [1, 64, 4], result is [1, 64, N*4].
    private static func concatenateLatents(_ latents: [MLMultiArray]) throws -> MLMultiArray {
        let featDim = VoxCpmConstants.featDim
        let patchSize = VoxCpmConstants.patchSize
        let totalFrames = latents.count * patchSize

        let shape: [NSNumber] = [1, NSNumber(value: featDim), NSNumber(value: totalFrames)]
        let result = try MLMultiArray(shape: shape, dataType: .float32)
        let dstPtr = result.dataPointer.bindMemory(
            to: Float.self, capacity: featDim * totalFrames)

        for (patchIdx, latent) in latents.enumerated() {
            let srcPtr = latent.dataPointer.bindMemory(
                to: Float.self, capacity: featDim * patchSize)
            let offset = patchIdx * patchSize
            for c in 0..<featDim {
                for t in 0..<patchSize {
                    dstPtr[c * totalFrames + offset + t] = srcPtr[c * patchSize + t]
                }
            }
        }

        return result
    }

    /// Decode latents to audio through audio_vae_decoder.
    private static func decodeAudio(
        latent: MLMultiArray,
        model: MLModel
    ) throws -> [Float] {
        let inputDict = try MLDictionaryFeatureProvider(dictionary: [
            VaeDecoderKeys.latent: MLFeatureValue(multiArray: latent)
        ])

        let pred = try model.prediction(from: inputDict)
        let audio = try ensureFloat32(
            pred.featureValue(for: VaeDecoderKeys.audio)!.multiArrayValue!)

        // audio is [1, 1, M] — extract samples
        let totalElements = audio.count
        let ptr = audio.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
        return Array(UnsafeBufferPointer(start: ptr, count: totalElements))
    }

    // MARK: - Linear Algebra Helpers

    /// Apply linear projection: y = x @ w.T + b
    ///
    /// x: [dim], w: [dim × dim] row-major, b: [dim]
    static func linear(_ x: [Float], w: [Float], b: [Float]) -> [Float] {
        let dim = VoxCpmConstants.hiddenSize
        var result = [Float](repeating: 0, count: dim)

        // y = x @ w.T + b
        // w is [outDim, inDim] row-major, so w.T[j,i] = w[i*outDim + j]
        // y[j] = sum_i(x[i] * w[j*dim + i]) + b[j]
        x.withUnsafeBufferPointer { xBuf in
            w.withUnsafeBufferPointer { wBuf in
                b.withUnsafeBufferPointer { bBuf in
                    result.withUnsafeMutableBufferPointer { rBuf in
                        // Use vDSP for matrix-vector multiply
                        // cblas_sgemv: y = alpha * A * x + beta * y
                        // A is w [dim × dim] in row-major, treating as transposed
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            Int32(dim), Int32(dim),
                            1.0, wBuf.baseAddress!, Int32(dim),
                            xBuf.baseAddress!, 1,
                            0.0, rBuf.baseAddress!, 1
                        )
                        // Add bias
                        vDSP_vadd(
                            rBuf.baseAddress!, 1, bBuf.baseAddress!, 1,
                            rBuf.baseAddress!, 1, vDSP_Length(dim))
                    }
                }
            }
        }

        return result
    }

    /// Compute dit_hidden = lm_to_dit_proj(fsq) + res_to_dit_proj(res_hidden).
    private static func computeDitHidden(
        lmHiddenFsq: MLMultiArray,
        resHidden: MLMultiArray,
        constants: VoxCpmConstantsBundle
    ) throws -> MLMultiArray {
        let dim = VoxCpmConstants.hiddenSize

        let fsq32 = try ensureFloat32(lmHiddenFsq)
        let res32 = try ensureFloat32(resHidden)

        let fsqPtr = fsq32.dataPointer.bindMemory(to: Float.self, capacity: dim)
        let resPtr = res32.dataPointer.bindMemory(to: Float.self, capacity: dim)

        let fsqArr = Array(UnsafeBufferPointer(start: fsqPtr, count: dim))
        let resArr = Array(UnsafeBufferPointer(start: resPtr, count: dim))

        let lmProj = linear(fsqArr, w: constants.lmToDitProjW, b: constants.lmToDitProjB)
        let resProj = linear(resArr, w: constants.resToDitProjW, b: constants.resToDitProjB)

        // Sum
        var ditArr = [Float](repeating: 0, count: dim)
        vDSP_vadd(lmProj, 1, resProj, 1, &ditArr, 1, vDSP_Length(dim))

        return try floatArrayToMLMultiArray(ditArr, shape: [1, NSNumber(value: dim)])
    }

    // MARK: - Embedding

    /// Look up text token embeddings from the embedding table, scaled by scaleEmb.
    static func embedTokens(
        _ tokenIds: [Int], constants: VoxCpmConstantsBundle
    ) -> [[Float]] {
        let dim = VoxCpmConstants.hiddenSize
        let scale = VoxCpmConstants.scaleEmb
        let vocabSize = VoxCpmConstants.vocabSize

        return tokenIds.map { id in
            let clampedId = min(max(id, 0), vocabSize - 1)
            let offset = clampedId * dim
            var emb = Array(constants.embedTokens[offset..<(offset + dim)])
            // Scale embeddings
            var s = scale
            vDSP_vsmul(emb, 1, &s, &emb, 1, vDSP_Length(dim))
            return emb
        }
    }

    // MARK: - MLMultiArray Helpers

    /// Convert a [Float] array to an MLMultiArray with the given shape.
    static func floatArrayToMLMultiArray(
        _ array: [Float], shape: [NSNumber]
    ) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = result.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        array.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            ptr.update(from: base, count: array.count)
        }
        return result
    }

    /// Element-wise add two MLMultiArrays of the same shape.
    private static func addArrays(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let a32 = try ensureFloat32(a)
        let b32 = try ensureFloat32(b)
        let count = a32.count
        let result = try MLMultiArray(shape: a32.shape, dataType: .float32)
        let aPtr = a32.dataPointer.bindMemory(to: Float.self, capacity: count)
        let bPtr = b32.dataPointer.bindMemory(to: Float.self, capacity: count)
        let rPtr = result.dataPointer.bindMemory(to: Float.self, capacity: count)
        vDSP_vadd(aPtr, 1, bPtr, 1, rPtr, 1, vDSP_Length(count))
        return result
    }

    /// Convert an MLMultiArray to Float32 if it's Float16. Returns as-is if already Float32.
    ///
    /// CoreML models often output Float16 tensors with non-contiguous strides
    /// (padding between dimensions). This helper correctly handles strided layouts
    /// by checking whether the data is contiguous before choosing a conversion path.
    static func ensureFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        if array.dataType == .float32 {
            return array
        }
        if array.dataType == .float16 {
            let count = array.count
            let result = try MLMultiArray(shape: array.shape, dataType: .float32)
            let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: count)

            // Check if strides are contiguous (row-major with no padding)
            let isContiguous = Self.isContiguousRowMajor(array)

            if isContiguous {
                // Fast path: flat buffer copy via vImage
                let srcPtr = array.dataPointer
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size
                )
                var dst = vImage_Buffer(
                    data: dstPtr,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            } else {
                // Strided path: gather Float16 values respecting strides, then batch convert
                let srcRaw = array.dataPointer.assumingMemoryBound(to: UInt16.self)
                let shape = array.shape.map { $0.intValue }
                let strides = array.strides.map { $0.intValue }
                let ndim = shape.count

                // Gather strided Float16 values into a contiguous buffer
                var gathered = [UInt16](repeating: 0, count: count)
                var indices = [Int](repeating: 0, count: ndim)
                for flatIdx in 0..<count {
                    var srcOffset = 0
                    for d in 0..<ndim {
                        srcOffset += indices[d] * strides[d]
                    }
                    gathered[flatIdx] = srcRaw[srcOffset]

                    // Increment indices (row-major order)
                    for d in stride(from: ndim - 1, through: 0, by: -1) {
                        indices[d] += 1
                        if indices[d] < shape[d] { break }
                        indices[d] = 0
                    }
                }

                // Batch convert the contiguous Float16 buffer to Float32
                gathered.withUnsafeMutableBufferPointer { srcBuf in
                    var src = vImage_Buffer(
                        data: srcBuf.baseAddress!,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.size
                    )
                    var dst = vImage_Buffer(
                        data: dstPtr,
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
                }
            }
            return result
        }
        // For other types (double, int32), do element-by-element conversion
        let count = array.count
        let result = try MLMultiArray(shape: array.shape, dataType: .float32)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            dstPtr[i] = array[i].floatValue
        }
        return result
    }

    /// Check if an MLMultiArray has contiguous row-major strides.
    private static func isContiguousRowMajor(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        let ndim = shape.count
        if ndim == 0 { return true }
        // Expected row-major strides: last dim stride=1, each preceding dim = product of following dims
        var expectedStride = 1
        for d in stride(from: ndim - 1, through: 0, by: -1) {
            if strides[d] != expectedStride {
                return false
            }
            expectedStride *= shape[d]
        }
        return true
    }
}
