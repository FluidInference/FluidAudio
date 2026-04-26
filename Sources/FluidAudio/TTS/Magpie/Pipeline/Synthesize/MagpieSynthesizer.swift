import Accelerate
@preconcurrency import CoreML
import Foundation

/// Orchestrates one Magpie synthesis call end-to-end.
///
/// Pipeline (mirroring `generate_coreml.generate`):
///   1. Tokenize text → padded ids (256) + mask.
///   2. `text_encoder.predict` → encoderOutput (1, 256, 768).
///   3. (CFG) make zero-context encoder pair.
///   4. Prefill: 110 step-by-step `decoder_step` calls with speaker embedding rows.
///   5. AR loop (≤ 500 steps):
///        embed current 8 codes → `decoder_step` → LT sample → new codes.
///   6. NanoCodec decode → fp32 PCM 22 kHz.
///   7. Peak-normalize to 0.9 when `options.peakNormalize`.
public actor MagpieSynthesizer {

    private let logger = AppLogger(category: "MagpieSynthesizer")

    private let store: MagpieModelStore
    private let tokenizer: MagpieTokenizer

    public init(store: MagpieModelStore, tokenizer: MagpieTokenizer) {
        self.store = store
        self.tokenizer = tokenizer
    }

    /// Synthesize from plain text (honors `|...|` IPA override per `options`).
    public func synthesize(
        text: String, speaker: MagpieSpeaker, language: MagpieLanguage,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let tokenized = try await tokenizer.tokenize(text, language: language, options: options)
        return try await synthesize(tokenized: tokenized, speaker: speaker, options: options)
    }

    /// Synthesize from pre-tokenized phoneme ids.
    public func synthesize(
        phonemes: MagpiePhonemeTokens, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let tokenized = try await tokenizer.pad(phonemes: phonemes)
        return try await synthesize(tokenized: tokenized, speaker: speaker, options: options)
    }

    // MARK: - Core

    private func synthesize(
        tokenized: MagpieTokenizedText, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let constants = try await store.constants()
        let ltWeights = try await store.localTransformer()
        let textEncoder = try await store.textEncoder()
        let decoderStep = try await store.decoderStep()
        let nanocodecModel = try await store.nanocodecDecoder()

        let dModel = constants.config.dModel
        let maxTextLen = MagpieConstants.maxTextLength
        let numCodebooks = constants.config.numCodebooks
        let audioBosId = constants.config.audioBosId
        let audioEosId = constants.config.audioEosId
        let speakerContextLength = constants.config.speakerContextLength

        let speakerIndex = speaker.rawValue
        guard speakerIndex >= 0 && speakerIndex < constants.speakerEmbeddings.count else {
            throw MagpieError.invalidSpeakerIndex(speakerIndex)
        }

        // 1. text_encoder
        let (encoderOutput, encoderMask) = try runTextEncoder(
            tokenized: tokenized, maxTextLen: maxTextLen, model: textEncoder)

        let useCfg = options.cfgScale != 1.0
        let uncond: (encoderOutput: MLMultiArray, encoderMask: MLMultiArray)?
        if useCfg {
            uncond = try MagpiePrefill.makeUnconditional(
                encoderOutputShape: encoderOutput.shape, maxTextLen: maxTextLen)
        } else {
            uncond = nil
        }

        // 2. KV caches (conditional + optional unconditional).
        let condCache = try MagpieKvCache(
            numLayers: constants.config.numDecoderLayers,
            maxCacheLength: constants.config.maxCacheLength,
            numHeads: constants.config.numHeads,
            headDim: constants.config.headDim)
        let uncondCache: MagpieKvCache? =
            useCfg
            ? try MagpieKvCache(
                numLayers: constants.config.numDecoderLayers,
                maxCacheLength: constants.config.maxCacheLength,
                numHeads: constants.config.numHeads,
                headDim: constants.config.headDim)
            : nil

        // 3. Prefill (fast batched path when decoder_prefill is available).
        let prefill = MagpiePrefill(decoderStep: decoderStep)
        let hasPrefill = await store.hasDecoderPrefill()
        let prefillStart = Date()
        if hasPrefill {
            let decoderPrefill = try await store.decoderPrefill()
            try prefill.prefillFast(
                decoderPrefill: decoderPrefill,
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefillFast(
                    decoderPrefill: decoderPrefill,
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        } else {
            try prefill.prefill(
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefill(
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        }
        let prefillElapsed = Date().timeIntervalSince(prefillStart)
        logger.info(
            "Prefill done in \(String(format: "%.2f", prefillElapsed))s "
                + "(\(hasPrefill ? "fast batched" : "slow loop"))")

        // 4. AR loop.
        let sampler = MagpieLocalSampler(
            localTransformer: MagpieLocalTransformer(weights: ltWeights),
            audioEmbeddings: constants.audioEmbeddings)

        var currentCodes = Swift.Array<Int32>(repeating: audioBosId, count: numCodebooks)
        var allFrames: [[Int32]] = []
        var finishedOnEos = false

        let rng = MagpieSamplerRng(seed: options.seed)

        for step in 0..<options.maxSteps {
            let audioEmbed = try embedAudioCodes(
                currentCodes, tables: constants.audioEmbeddings, dModel: dModel)

            let condHidden = try runDecoderStep(
                audioEmbed: audioEmbed,
                encoderOutput: encoderOutput, encoderMask: encoderMask,
                cache: condCache, model: decoderStep)

            var uncondHidden: [Float]? = nil
            if useCfg, let uncondTensors = uncond, let uncondCache = uncondCache {
                let h = try runDecoderStep(
                    audioEmbed: audioEmbed,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache, model: decoderStep)
                uncondHidden = h
            }

            let forbidEos = step < options.minFrames
            let next = sampler.sample(
                decoderHidden: condHidden,
                uncondDecoderHidden: uncondHidden,
                forbidEos: forbidEos,
                options: options,
                rng: rng)

            let isEos = next.contains(audioEosId)
            if isEos && step >= options.minFrames {
                finishedOnEos = true
                logger.info("EOS at step \(step)")
                break
            }
            allFrames.append(next)
            currentCodes = next
        }

        let numFrames = allFrames.count
        guard numFrames > 0 else {
            throw MagpieError.inferenceFailed(
                stage: "synthesize", underlying: "no audio frames generated")
        }

        // 5. NanoCodec decode: reshape (numFrames × numCodebooks) into
        //    per-codebook rows.
        var codebookRows = Swift.Array(
            repeating: Swift.Array<Int32>(repeating: 0, count: numFrames),
            count: numCodebooks)
        for t in 0..<numFrames {
            let row = allFrames[t]
            for cb in 0..<numCodebooks {
                codebookRows[cb][t] = row[cb]
            }
        }
        let nanocodec = MagpieNanocodec(
            model: nanocodecModel, numCodebooks: numCodebooks)
        var samples = try nanocodec.decode(frames: codebookRows)

        // 6. Peak normalize to 0.9.
        if options.peakNormalize {
            var peak: Float = 0
            for s in samples where abs(s) > peak { peak = abs(s) }
            if peak > 0 {
                let scale = MagpieConstants.peakTarget / peak
                for i in 0..<samples.count { samples[i] *= scale }
            }
        }

        return MagpieSynthesisResult(
            samples: samples,
            sampleRate: MagpieConstants.audioSampleRate,
            codeCount: numFrames,
            finishedOnEos: finishedOnEos)
    }

    // MARK: - Model runners

    private func runTextEncoder(
        tokenized: MagpieTokenizedText, maxTextLen: Int, model: MLModel
    ) throws -> (encoderOutput: MLMultiArray, encoderMask: MLMultiArray) {
        let tokenArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .int32)
        tokenArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Int32.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.paddedIds[i] }
        }
        let maskArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .float32)
        maskArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Float.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.mask[i] }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokenArr),
            "text_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let out = try model.prediction(from: provider)
        guard let encoderOutput = out.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw MagpieError.inferenceFailed(
                stage: "text_encoder", underlying: "missing encoder_output key")
        }
        return (encoderOutput, maskArr)
    }

    private func runDecoderStep(
        audioEmbed: MLMultiArray,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        cache: MagpieKvCache,
        model: MLModel
    ) throws -> [Float] {
        var inputs: [String: MLMultiArray] = [
            "audio_embed": audioEmbed,
            "encoder_output": encoderOutput,
            "encoder_mask": encoderMask,
        ]
        cache.addInputs(to: &inputs)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let out = try model.prediction(from: provider)
        try cache.absorbOutputs(out)
        guard let hidden = out.featureValue(for: MagpieKvCache.decoderHiddenKey)?.multiArrayValue
        else {
            throw MagpieError.inferenceFailed(
                stage: "decoder_step", underlying: "missing hidden key")
        }
        let dim = hidden.count
        var result = Swift.Array<Float>(repeating: 0, count: dim)
        switch hidden.dataType {
        case .float16:
            // Convert fp16 → fp32 via vImage (Accelerate) for speed.
            hidden.withUnsafeBytes { raw in
                guard let src = raw.baseAddress else { return }
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
                result.withUnsafeMutableBufferPointer { dst in
                    var dstBuffer = vImage_Buffer(
                        data: dst.baseAddress, height: 1,
                        width: vImagePixelCount(dim), rowBytes: dim * 4)
                    _ = vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
                }
            }
        case .float32:
            hidden.withUnsafeBytes { raw in
                let ptr = raw.bindMemory(to: Float.self)
                for i in 0..<dim { result[i] = ptr[i] }
            }
        case .double:
            hidden.withUnsafeBytes { raw in
                let ptr = raw.bindMemory(to: Double.self)
                for i in 0..<dim { result[i] = Float(ptr[i]) }
            }
        default:
            for i in 0..<dim { result[i] = hidden[i].floatValue }
        }
        return result
    }

    private func embedAudioCodes(
        _ codes: [Int32], tables: [[Float]], dModel: Int
    ) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)
        arr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Float.self).baseAddress!
            for i in 0..<dModel { base[i] = 0 }
            let numCodebooks = codes.count
            for cb in 0..<numCodebooks {
                let row = Int(codes[cb])
                let table = tables[cb]
                let start = row * dModel
                for i in 0..<dModel {
                    base[i] += table[start + i]
                }
            }
            let inv = 1.0 / Float(numCodebooks)
            for i in 0..<dModel { base[i] *= inv }
        }
        return arr
    }

}
