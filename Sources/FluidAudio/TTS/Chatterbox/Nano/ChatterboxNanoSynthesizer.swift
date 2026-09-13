@preconcurrency import CoreML
import Foundation

/// Chatterbox Nano synthesis: T3 prefill → stateful autoregressive decode
/// (turbo sampling over S3 speech tokens — no CFG, no alignment analyzer) →
/// S3Gen meanflow (mel) → HiFT vocoding.
///
/// Host responsibilities mirror the mobius reference driver
/// (`models/tts/chatterbox/coreml/verify/e2e_nano_coreml.py`): embedding
/// prep from tables (bare lookups — GPT2's `wpe` is applied in-graph; the
/// prefill context is cond ++ text ++ a single BOS), turbo sampling
/// (temperature → top-k → top-p → repetition penalty), SineGen randomness,
/// and bucket padding/cropping.
@available(macOS 15.0, iOS 18.0, *)
struct ChatterboxNanoSynthesizer {

    private static let logger = AppLogger(category: "ChatterboxNanoSynthesizer")

    struct Result {
        let samples: [Float]
        let speechTokens: Int
        let prefillSeconds: Double
        let decodeSeconds: Double
        let decodedTokens: Int
        let flowSeconds: Double
        let vocoderSeconds: Double
    }

    let models: ChatterboxNanoModels

    /// Runs on the caller's actor (`#isolation`) so the non-Sendable CoreML
    /// state never crosses an isolation boundary.
    func synthesize(
        text: String,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        seed: UInt64,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Result {
        // ---- Tokenize + prefill embeds ----
        let normalized = ChatterboxNanoTokenizer.puncNorm(text)
        let textIds = models.tokenizer.encode(normalized)
        let condLen = models.voice.condEmb.rows
        let contextLen = condLen + textIds.count + 1  // single BOS embed
        guard contextLen <= ChatterboxNanoConstants.prefillLength else {
            throw ChatterboxError.textTooLong(
                tokens: contextLen, max: ChatterboxNanoConstants.prefillLength)
        }

        let prefillEmbeds = try buildPrefillEmbeds(textIds: textIds)
        let inputLen = try MLMultiArray(shape: [1], dataType: .int32)
        inputLen[0] = NSNumber(value: contextLen)

        let prefillStart = Date()
        let prefillOut = try await models.prefill.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "inputs_embeds": MLFeatureValue(multiArray: prefillEmbeds),
                "input_len": MLFeatureValue(multiArray: inputLen),
            ]))
        guard let logitsArr = prefillOut.featureValue(for: "logits")?.multiArrayValue,
            let kvK = prefillOut.featureValue(for: "kv_k")?.multiArrayValue,
            let kvV = prefillOut.featureValue(for: "kv_v")?.multiArrayValue
        else {
            throw ChatterboxError.processingFailed("prefill outputs missing")
        }
        let prefillSeconds = -prefillStart.timeIntervalSinceNow

        // ---- Seed decode state from prefill KV ----
        let state = models.decode.makeState()
        let layerElements =
            ChatterboxNanoConstants.kvHeads * ChatterboxNanoConstants.maxContext
            * ChatterboxNanoConstants.headDim
        try ChatterboxMLSupport.seedState(
            state, kvK: kvK, kvV: kvV,
            layerCount: ChatterboxNanoConstants.layerCount, layerElements: layerElements)

        // ---- Autoregressive decode ----
        let decodeStart = Date()
        var rng = SplitMix64(seed: seed)
        var logits = try ChatterboxMLSupport.floatBuffer(logitsArr)  // [V]
        var generatedIds: [Int] = []
        var speechTokens: [Int] = []
        var decodedTokens = 0

        let stepEmbeds = try MLMultiArray(
            shape: [1, 1, NSNumber(value: ChatterboxNanoConstants.hiddenSize)],
            dataType: .float32)
        let curLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        let maxSteps = min(
            ChatterboxNanoConstants.maxNewTokens,
            ChatterboxNanoConstants.maxContext - contextLen - 1)

        for step in 0..<maxSteps {
            // Upstream's first sample penalizes the BOS id (its input_ids
            // start as [BOS]); every later step penalizes generated ids only.
            let penalized =
                generatedIds.isEmpty
                ? [ChatterboxNanoConstants.startSpeechToken] : generatedIds
            let token = Self.sample(
                logits: &logits, generated: penalized,
                temperature: temperature, topK: topK, topP: topP,
                repetitionPenalty: repetitionPenalty, rng: &rng)
            decodedTokens += 1
            generatedIds.append(token)
            if token == ChatterboxNanoConstants.stopSpeechToken { break }
            if token < ChatterboxNanoConstants.speechVocabSize { speechTokens.append(token) }

            try fillStepEmbeds(stepEmbeds, token: token)
            curLenArr[0] = NSNumber(value: contextLen + step)
            let out = try await models.decode.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "inputs_embeds": MLFeatureValue(multiArray: stepEmbeds),
                    "cur_len": MLFeatureValue(multiArray: curLenArr),
                ]),
                using: state,
                options: MLPredictionOptions())
            guard let stepLogits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw ChatterboxError.processingFailed("decode outputs missing")
            }
            logits = try ChatterboxMLSupport.floatBuffer(stepLogits)
        }
        let decodeSeconds = -decodeStart.timeIntervalSinceNow

        guard !speechTokens.isEmpty else {
            throw ChatterboxError.processingFailed("no speech tokens generated")
        }
        // Upstream appends three silence tokens before vocoding.
        speechTokens.append(
            contentsOf: [Int](
                repeating: ChatterboxNanoConstants.silenceToken,
                count: ChatterboxNanoConstants.silenceTokenCount))
        let promptLen = models.voice.promptTokens.count
        let totalTokens = promptLen + speechTokens.count
        guard totalTokens <= ChatterboxNanoConstants.flowTokenBucket else {
            throw ChatterboxError.generationTooLong(
                tokens: totalTokens, max: ChatterboxNanoConstants.flowTokenBucket)
        }

        // ---- S3Gen: meanflow (mel) + HiFT (waveform) ----
        let flowStart = Date()
        let mel = try await runFlow(speechTokens: speechTokens, rng: &rng)
        let flowSeconds = -flowStart.timeIntervalSinceNow

        let vocoderStart = Date()
        let samples = try await runVocoder(
            mel: mel, melFrames: 2 * speechTokens.count, rng: &rng)
        let vocoderSeconds = -vocoderStart.timeIntervalSinceNow

        return Result(
            samples: samples,
            speechTokens: speechTokens.count,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodedTokens: decodedTokens,
            flowSeconds: flowSeconds,
            vocoderSeconds: vocoderSeconds)
    }

    // MARK: - Embedding prep

    /// `[1, prefillLength, hidden]` fp32: `[cond, text, BOS]` — bare table
    /// lookups, no positional adds (GPT2 `wpe` is applied in-graph).
    private func buildPrefillEmbeds(textIds: [Int]) throws -> MLMultiArray {
        let hidden = ChatterboxNanoConstants.hiddenSize
        let tPrefill = ChatterboxNanoConstants.prefillLength
        let tables = models.tables
        let condEmb = models.voice.condEmb

        let array = try MLMultiArray(
            shape: [1, NSNumber(value: tPrefill), NSNumber(value: hidden)],
            dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        ptr.update(repeating: 0, count: array.count)

        var offset = 0
        for value in condEmb.values {
            ptr[offset] = value
            offset += 1
        }
        for id in textIds {
            for e in tables.textEmb.row(id) {
                ptr[offset] = e
                offset += 1
            }
        }
        for e in tables.speechEmb.row(ChatterboxNanoConstants.startSpeechToken) {
            ptr[offset] = e
            offset += 1
        }
        return array
    }

    private func fillStepEmbeds(_ array: MLMultiArray, token: Int) throws {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for (i, e) in models.tables.speechEmb.row(token).enumerated() {
            ptr[i] = e
        }
    }

    // MARK: - Sampling

    /// Upstream `inference_turbo` order: temperature → top-k → top-p →
    /// repetition penalty → softmax → multinomial. Note the repetition
    /// penalty runs LAST, on the already-filtered logits — the reverse of
    /// the multilingual sampler.
    static func sample(
        logits: inout [Float], generated: [Int],
        temperature: Float, topK: Int, topP: Float, repetitionPenalty: Float,
        rng: inout SplitMix64
    ) -> Int {
        if temperature > 0 && temperature != 1.0 {
            let invTemp = 1.0 / temperature
            for i in 0..<logits.count { logits[i] *= invTemp }
        }

        // Top-k: mask everything below the k-th largest logit.
        if topK > 0 && topK < logits.count {
            var sorted = logits
            sorted.sort(by: >)
            let threshold = sorted[topK - 1]
            for i in 0..<logits.count where logits[i] < threshold {
                logits[i] = -Float.infinity
            }
        }

        // Top-p nucleus over the softmax of the surviving logits.
        if topP < 1.0 {
            var maxLogit = -Float.infinity
            for v in logits where v > maxLogit { maxLogit = v }
            var probs = [Float](repeating: 0, count: logits.count)
            var total: Float = 0
            for i in 0..<logits.count where logits[i] > -Float.infinity {
                let p = expf(logits[i] - maxLogit)
                probs[i] = p
                total += p
            }
            let order = probs.indices.sorted { probs[$0] > probs[$1] }
            var cumulative: Float = 0
            var cut = false
            for idx in order {
                if cut {
                    logits[idx] = -Float.infinity
                    continue
                }
                cumulative += probs[idx] / total
                if cumulative >= topP { cut = true }
            }
        }

        // HF RepetitionPenaltyLogitsProcessor on the filtered logits.
        if repetitionPenalty != 1.0 {
            for id in Set(generated) {
                let score = logits[id]
                if score.isFinite {
                    logits[id] = score < 0 ? score * repetitionPenalty : score / repetitionPenalty
                }
            }
        }

        // Softmax + multinomial.
        var maxLogit = -Float.infinity
        for v in logits where v > maxLogit { maxLogit = v }
        var probs = [Float](repeating: 0, count: logits.count)
        var total: Float = 0
        for i in 0..<logits.count where logits[i] > -Float.infinity {
            let p = expf(logits[i] - maxLogit)
            probs[i] = p
            total += p
        }
        var draw = Float(rng.nextUniform()) * total
        var last = 0
        for i in 0..<probs.count where probs[i] > 0 {
            draw -= probs[i]
            last = i
            if draw <= 0 { return i }
        }
        return last
    }

    // MARK: - S3Gen

    private func runFlow(
        speechTokens: [Int], rng: inout SplitMix64,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> MLMultiArray {
        let bucket = ChatterboxNanoConstants.flowTokenBucket
        let melBucket = ChatterboxNanoConstants.melFrameBucket
        let voice = models.voice
        let promptLen = voice.promptTokens.count
        let totalLen = promptLen + speechTokens.count

        let tokens = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
        let tokensPtr = tokens.dataPointer.assumingMemoryBound(to: Int32.self)
        for (i, t) in voice.promptTokens.enumerated() { tokensPtr[i] = t }
        for (i, t) in speechTokens.enumerated() { tokensPtr[promptLen + i] = Int32(t) }
        for i in totalLen..<bucket { tokensPtr[i] = 0 }

        let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
        tokenLen[0] = NSNumber(value: totalLen)
        let promptLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        promptLenArr[0] = NSNumber(value: promptLen)

        let promptFeat = try MLMultiArray(
            shape: [1, NSNumber(value: melBucket), 80], dataType: .float32)
        let pfPtr = promptFeat.dataPointer.assumingMemoryBound(to: Float.self)
        pfPtr.update(repeating: 0, count: promptFeat.count)
        voice.promptFeat.values.withUnsafeBufferPointer { src in
            pfPtr.update(from: src.baseAddress!, count: src.count)
        }

        let embedding = try MLMultiArray(shape: [1, 192], dataType: .float32)
        let embPtr = embedding.dataPointer.assumingMemoryBound(to: Float.self)
        voice.embedding.withUnsafeBufferPointer { src in
            embPtr.update(from: src.baseAddress!, count: src.count)
        }

        // Meanflow initial noise: gaussian over the live frames, zeros beyond.
        let z = try MLMultiArray(
            shape: [1, 80, NSNumber(value: melBucket)], dataType: .float32)
        let zPtr = z.dataPointer.assumingMemoryBound(to: Float.self)
        zPtr.update(repeating: 0, count: z.count)
        let liveFrames = 2 * totalLen
        for c in 0..<80 {
            for f in 0..<liveFrames {
                zPtr[c * melBucket + f] = rng.nextGaussian()
            }
        }

        let out = try await models.flow.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "tokens": MLFeatureValue(multiArray: tokens),
                "token_len": MLFeatureValue(multiArray: tokenLen),
                "prompt_len": MLFeatureValue(multiArray: promptLenArr),
                "prompt_feat": MLFeatureValue(multiArray: promptFeat),
                "embedding": MLFeatureValue(multiArray: embedding),
                "z": MLFeatureValue(multiArray: z),
            ]))
        guard let mel = out.featureValue(for: "mel")?.multiArrayValue else {
            throw ChatterboxError.processingFailed("flow output missing")
        }
        return mel
    }

    private func runVocoder(
        mel: MLMultiArray, melFrames: Int, rng: inout SplitMix64,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> [Float] {
        let melBucket = ChatterboxNanoConstants.melFrameBucket
        let promptFrames = 2 * models.voice.promptTokens.count
        let melValues = try ChatterboxMLSupport.floatBuffer(mel)  // [80 * melBucket]

        // Crop the generated frames ([2*promptLen, 2*totalLen)) to the front
        // of the vocoder bucket, zero-padded beyond.
        let melPad = try MLMultiArray(
            shape: [1, 80, NSNumber(value: melBucket)], dataType: .float32)
        let melPtr = melPad.dataPointer.assumingMemoryBound(to: Float.self)
        melPtr.update(repeating: 0, count: melPad.count)
        for c in 0..<80 {
            for f in 0..<melFrames {
                melPtr[c * melBucket + f] = melValues[c * melBucket + promptFrames + f]
            }
        }

        // SineGen randomness: per-harmonic phase (row 0 fixed at 0) + noise.
        let harmonics = ChatterboxNanoConstants.hiftHarmonics
        let phase = try MLMultiArray(
            shape: [1, NSNumber(value: harmonics), 1], dataType: .float32)
        let phasePtr = phase.dataPointer.assumingMemoryBound(to: Float.self)
        phasePtr[0] = 0
        for h in 1..<harmonics {
            phasePtr[h] = Float(rng.nextUniform()) * 2 * .pi - .pi
        }
        let sampleCount = melBucket * ChatterboxNanoConstants.samplesPerMelFrame
        let noise = try MLMultiArray(
            shape: [1, NSNumber(value: harmonics), NSNumber(value: sampleCount)],
            dataType: .float32)
        let noisePtr = noise.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(harmonics * sampleCount) { noisePtr[i] = rng.nextGaussian() }

        let out = try await models.vocoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: melPad),
                "phase_vec": MLFeatureValue(multiArray: phase),
                "noise": MLFeatureValue(multiArray: noise),
            ]))
        guard let audio = out.featureValue(for: "audio")?.multiArrayValue else {
            throw ChatterboxError.processingFailed("vocoder output missing")
        }

        var samples = try ChatterboxMLSupport.floatBuffer(audio)
        let validSamples = melFrames * ChatterboxNanoConstants.samplesPerMelFrame
        if samples.count > validSamples {
            samples.removeLast(samples.count - validSamples)
        }

        // Upstream trim fade: 20 ms silence + 20 ms half-cosine ramp-in to
        // reduce reference-clip spillover.
        let nTrim = ChatterboxNanoConstants.sampleRate / 50
        for i in 0..<min(nTrim, samples.count) { samples[i] = 0 }
        for i in 0..<nTrim where nTrim + i < samples.count {
            let ramp = (cosf(.pi - .pi * Float(i) / Float(nTrim - 1)) + 1) / 2
            samples[nTrim + i] *= ramp
        }
        return samples
    }
}
