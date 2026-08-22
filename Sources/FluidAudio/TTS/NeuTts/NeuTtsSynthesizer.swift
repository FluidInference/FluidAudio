@preconcurrency import CoreML
import Foundation

/// NeuTTS-2E synthesis: prefill → stateful autoregressive decode (top-k
/// sampling over speech tokens) → NeuCodec vocoding.
@available(macOS 15.0, iOS 18.0, *)
struct NeuTtsSynthesizer {

    private static let logger = AppLogger(category: "NeuTtsSynthesizer")

    enum SynthesisError: Error, LocalizedError {
        case missingOutput(String)
        case noSpeechGenerated

        var errorDescription: String? {
            switch self {
            case .missingOutput(let name):
                return "Model output '\(name)' missing or has unexpected type"
            case .noSpeechGenerated:
                return "The language model produced no speech tokens"
            }
        }
    }

    struct Result {
        let samples: [Float]
        let codeCount: Int
        let prefillSeconds: Double
        let decodeSeconds: Double
        let decodedTokens: Int
        let codecSeconds: Double
    }

    let models: NeuTtsModels

    /// Runs on the caller's actor (`#isolation`) so the non-Sendable CoreML
    /// state never crosses an isolation boundary.
    func synthesize(
        text: String,
        speaker: String,
        emotion: String,
        temperature: Float,
        topK: Int,
        seed: UInt64,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Result {
        let reference = try models.speakerReference(speaker)
        let promptIds = try NeuTtsPrompt.buildIds(
            tokenizer: models.tokenizer, text: text, reference: reference, emotion: emotion)
        guard let eosId = models.tokenizer.tokenId("<|SPEECH_GENERATION_END|>") else {
            throw NeuTtsPrompt.PromptError.missingSpecialToken("<|SPEECH_GENERATION_END|>")
        }

        // ---- Prefill ----
        let prefillStart = Date()
        let inputIds = try MLMultiArray(
            shape: [1, NSNumber(value: NeuTtsConstants.prefillLength)], dataType: .int32)
        let idsPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<NeuTtsConstants.prefillLength {
            idsPtr[i] = i < promptIds.count ? Int32(promptIds[i]) : 0
        }
        let inputLen = try MLMultiArray(shape: [1], dataType: .int32)
        inputLen[0] = NSNumber(value: promptIds.count)

        let prefillOut = try await models.prefill.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "input_len": MLFeatureValue(multiArray: inputLen),
            ]))
        guard let logitsLast = prefillOut.featureValue(for: "logits_last")?.multiArrayValue,
            let kvK = prefillOut.featureValue(for: "kv_k")?.multiArrayValue,
            let kvV = prefillOut.featureValue(for: "kv_v")?.multiArrayValue
        else {
            throw SynthesisError.missingOutput("logits_last/kv_k/kv_v")
        }
        let prefillSeconds = -prefillStart.timeIntervalSinceNow

        // ---- Seed decode state from prefill KV ----
        let state = models.decode.makeState()
        try seedState(state, kvK: kvK, kvV: kvV)

        // ---- Autoregressive decode ----
        let decodeStart = Date()
        var rng = SplitMix64(seed: seed)
        var logits = try floatBuffer(logitsLast)
        var generated: [Int] = []
        var curLen = promptIds.count

        let stepIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let curLenArr = try MLMultiArray(shape: [1], dataType: .int32)

        while curLen < NeuTtsConstants.maxContext - 1 {
            let suppressEos = generated.count < NeuTtsConstants.minNewTokens
            let token = Self.sampleTopK(
                logits: logits, k: topK, temperature: temperature,
                excluding: suppressEos ? eosId : nil, rng: &rng)
            generated.append(token)
            if token == eosId { break }

            stepIds[0] = NSNumber(value: token)
            curLenArr[0] = NSNumber(value: curLen)
            let out = try await models.decode.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: stepIds),
                    "cur_len": MLFeatureValue(multiArray: curLenArr),
                ]),
                using: state,
                options: MLPredictionOptions())
            guard let stepLogits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw SynthesisError.missingOutput("logits")
            }
            logits = try floatBuffer(stepLogits)
            curLen += 1
        }
        let decodeSeconds = -decodeStart.timeIntervalSinceNow

        let codes = try NeuTtsPrompt.extractCodes(tokenizer: models.tokenizer, ids: generated)
        guard !codes.isEmpty else { throw SynthesisError.noSpeechGenerated }

        // ---- NeuCodec vocoding ----
        let codecStart = Date()
        let samples = try await decodeCodes(codes)
        let codecSeconds = -codecStart.timeIntervalSinceNow

        return Result(
            samples: samples,
            codeCount: codes.count,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodedTokens: generated.count,
            codecSeconds: codecSeconds)
    }

    // MARK: - Codec

    private func decodeCodes(
        _ codes: [Int], isolation: isolated (any Actor)? = #isolation
    ) async throws -> [Float] {
        let count = min(codes.count, NeuTtsConstants.maxCodecCodes)
        let codesArr = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .int32)
        let ptr = codesArr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<count { ptr[i] = Int32(codes[i]) }

        let out = try await models.codec.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "codes": MLFeatureValue(multiArray: codesArr)
            ]))
        guard let audio = out.featureValue(for: "audio")?.multiArrayValue else {
            throw SynthesisError.missingOutput("audio")
        }
        return try floatBuffer(audio)
    }

    // MARK: - KV state seeding

    /// Copy prefill KV (fp32, [layers, 1, heads, maxContext, headDim]) into
    /// the decode model's per-layer fp16 state buffers.
    private func seedState(_ state: MLState, kvK: MLMultiArray, kvV: MLMultiArray) throws {
        let layerElements =
            NeuTtsConstants.kvHeads * NeuTtsConstants.maxContext * NeuTtsConstants.headDim
        let kPtr = kvK.dataPointer.assumingMemoryBound(to: Float.self)
        let vPtr = kvV.dataPointer.assumingMemoryBound(to: Float.self)

        for layer in 0..<NeuTtsConstants.layerCount {
            for (name, src) in [("kv_k_\(layer)", kPtr), ("kv_v_\(layer)", vPtr)] {
                let base = src.advanced(by: layer * layerElements)
                state.withMultiArray(for: name) { array in
                    array.withUnsafeMutableBytes { rawBuffer, _ in
                        guard let dstBase = rawBuffer.baseAddress else { return }
                        // Address fp16 storage as raw bit patterns: Swift's
                        // `Float16` is unavailable on macOS x86_64.
                        let dst = dstBase.assumingMemoryBound(to: UInt16.self)
                        Float16Conversion.fromFloat32(src: base, dst: dst, count: layerElements)
                    }
                }
            }
        }
    }

    // MARK: - Sampling

    /// Top-k + temperature sampling over the full logits vector, matching the
    /// upstream generation settings. `excluding` masks EOS while the
    /// min-new-tokens constraint is active.
    static func sampleTopK(
        logits: [Float], k: Int, temperature: Float, excluding: Int?, rng: inout SplitMix64
    ) -> Int {
        // Single pass with a fixed-size bottom-anchored selection.
        var topIds = [Int](repeating: -1, count: k)
        var topVals = [Float](repeating: -.infinity, count: k)
        var minIndex = 0
        for (id, value) in logits.enumerated() {
            if id == excluding { continue }
            if value > topVals[minIndex] {
                topVals[minIndex] = value
                topIds[minIndex] = id
                // Recompute the smallest slot.
                minIndex = 0
                for j in 1..<k where topVals[j] < topVals[minIndex] { minIndex = j }
            }
        }

        let invTemp = 1.0 / max(temperature, 1e-6)
        let maxVal = topVals.max() ?? 0
        var probs = topVals.map { expf(($0 - maxVal) * invTemp) }
        let total = probs.reduce(0, +)
        guard total > 0 else { return topIds[0] }
        for i in 0..<probs.count { probs[i] /= total }

        var draw = Float(rng.nextUniform())
        for i in 0..<probs.count {
            draw -= probs[i]
            if draw <= 0 { return topIds[i] }
        }
        return topIds[probs.count - 1]
    }

    // MARK: - Helpers

    private func floatBuffer(_ array: MLMultiArray) throws -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            out.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            out.withUnsafeMutableBufferPointer { dst in
                Float16Conversion.toFloat32(src: ptr, dst: dst.baseAddress!, count: count)
            }
        default:
            throw SynthesisError.missingOutput("unexpected dataType \(array.dataType.rawValue)")
        }
        return out
    }
}

/// Seedable RNG for reproducible sampling (`--seed`).
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    mutating func nextUniform() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}
