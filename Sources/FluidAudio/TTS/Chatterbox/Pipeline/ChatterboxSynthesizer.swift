@preconcurrency import CoreML
import Foundation

/// Chatterbox Multilingual synthesis: T3 prefill → stateful autoregressive
/// decode (CFG + alignment analyzer + repetition penalty / min-p sampling
/// over S3 speech tokens) → S3Gen flow (mel) → HiFT vocoding.
///
/// Host responsibilities mirror the mobius reference driver
/// (`models/tts/chatterbox/coreml/verify/e2e_coreml.py`): embedding prep from
/// tables (CFG zeroes the text embedding BEFORE the positional add; the
/// prefill context ends with two BOS embeds — both are upstream-faithful),
/// sampling, SineGen randomness, and bucket padding/cropping.
@available(macOS 15.0, iOS 18.0, *)
struct ChatterboxSynthesizer {

    private static let logger = AppLogger(category: "ChatterboxSynthesizer")

    struct Result {
        let samples: [Float]
        let speechTokens: Int
        let prefillSeconds: Double
        let decodeSeconds: Double
        let decodedTokens: Int
        let flowSeconds: Double
        let vocoderSeconds: Double
    }

    let models: ChatterboxModels

    /// Runs on the caller's actor (`#isolation`) so the non-Sendable CoreML
    /// state never crosses an isolation boundary.
    func synthesize(
        text: String,
        language: String,
        cfgWeight: Float,
        temperature: Float,
        repetitionPenalty: Float,
        minP: Float,
        topP: Float,
        seed: UInt64,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Result {
        let lang = language.lowercased()
        guard ChatterboxConstants.supportedLanguages.contains(lang) else {
            throw ChatterboxError.unsupportedLanguage(lang)
        }


        // ---- Tokenize + prefill embeds ----
        let normalized = ChatterboxTokenizer.puncNorm(text)
        var textIds = models.tokenizer.encode(normalized, languageId: lang)
        textIds.insert(ChatterboxConstants.startTextToken, at: 0)
        textIds.append(ChatterboxConstants.stopTextToken)
        let condLen = models.voice.condEmb.rows
        let contextLen = condLen + textIds.count + 2  // two BOS embeds
        guard contextLen <= ChatterboxConstants.prefillLength else {
            throw ChatterboxError.textTooLong(
                tokens: contextLen, max: ChatterboxConstants.prefillLength)
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
            let alignArr = prefillOut.featureValue(for: "align_attn")?.multiArrayValue,
            let kvK = prefillOut.featureValue(for: "kv_k")?.multiArrayValue,
            let kvV = prefillOut.featureValue(for: "kv_v")?.multiArrayValue
        else {
            throw ChatterboxError.processingFailed("prefill outputs missing")
        }
        let prefillSeconds = -prefillStart.timeIntervalSinceNow

        // ---- Seed decode state from prefill KV ----
        let state = models.decode.makeState()
        try seedState(state, kvK: kvK, kvV: kvV)

        // ---- Autoregressive decode ----
        let decodeStart = Date()
        var rng = SplitMix64(seed: seed)
        var analyzer = ChatterboxAlignmentAnalyzer(
            textStart: condLen, textEnd: condLen + textIds.count,
            eosIndex: ChatterboxConstants.stopSpeechToken)
        var logits2 = try floatBuffer(logitsArr)  // [2 * V]
        var alignRows = try prefillAlignRows(alignArr, contextLen: contextLen)
        var generatedIds: [Int] = [ChatterboxConstants.startSpeechToken]
        var speechTokens: [Int] = []
        var decodedTokens = 0

        let stepEmbeds = try MLMultiArray(
            shape: [2, 1, NSNumber(value: ChatterboxConstants.hiddenSize)], dataType: .float32)
        let curLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        let vocab = ChatterboxConstants.outputVocabSize
        let maxSteps = min(
            ChatterboxConstants.maxNewTokens,
            ChatterboxConstants.maxContext - contextLen - 1)

        for step in 0..<maxSteps {
            // CFG combine over the batch-2 logits.
            var logits = [Float](repeating: 0, count: vocab)
            for i in 0..<vocab {
                let cond = logits2[i]
                let uncond = logits2[vocab + i]
                logits[i] = cond + cfgWeight * (cond - uncond)
            }
            analyzer.step(
                logits: &logits, alignRows: alignRows,
                nextToken: generatedIds.last)
            let token = Self.sample(
                logits: &logits, generated: generatedIds,
                temperature: temperature, repetitionPenalty: repetitionPenalty,
                minP: minP, topP: topP, rng: &rng)
            decodedTokens += 1
            generatedIds.append(token)
            if token == ChatterboxConstants.stopSpeechToken { break }
            if token < ChatterboxConstants.speechVocabSize { speechTokens.append(token) }

            try fillStepEmbeds(stepEmbeds, token: token, step: step)
            curLenArr[0] = NSNumber(value: contextLen + step)
            let out = try await models.decode.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "inputs_embeds": MLFeatureValue(multiArray: stepEmbeds),
                    "cur_len": MLFeatureValue(multiArray: curLenArr),
                ]),
                using: state,
                options: MLPredictionOptions())
            guard let stepLogits = out.featureValue(for: "logits")?.multiArrayValue,
                let stepAlign = out.featureValue(for: "align_attn")?.multiArrayValue
            else {
                throw ChatterboxError.processingFailed("decode outputs missing")
            }
            logits2 = try floatBuffer(stepLogits)
            alignRows = [decodeAlignRow(try floatBuffer(stepAlign), contextLen: contextLen + step + 1)]
        }
        let decodeSeconds = -decodeStart.timeIntervalSinceNow

        guard !speechTokens.isEmpty else {
            throw ChatterboxError.processingFailed("no speech tokens generated")
        }
        let promptLen = models.voice.promptTokens.count
        let totalTokens = promptLen + speechTokens.count
        guard totalTokens <= ChatterboxConstants.flowTokenBucket else {
            throw ChatterboxError.generationTooLong(
                tokens: totalTokens, max: ChatterboxConstants.flowTokenBucket)
        }

        // ---- S3Gen: flow (mel) + HiFT (waveform) ----
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

    /// `[2, prefillLength, hidden]` fp32: `[cond, text(+pos), BOS, BOS]` per
    /// CFG row; row 1 zeroes the text embedding but keeps the positional add.
    private func buildPrefillEmbeds(textIds: [Int]) throws -> MLMultiArray {
        let hidden = ChatterboxConstants.hiddenSize
        let tPrefill = ChatterboxConstants.prefillLength
        let tables = models.tables
        let condEmb = models.voice.condEmb
        let condLen = condEmb.rows

        let array = try MLMultiArray(
            shape: [2, NSNumber(value: tPrefill), NSNumber(value: hidden)],
            dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        ptr.update(repeating: 0, count: array.count)

        var bos = [Float](repeating: 0, count: hidden)
        let bosEmb = tables.speechEmb.row(ChatterboxConstants.startSpeechToken)
        let bosPos = tables.speechPos.row(0)
        for (i, (e, p)) in zip(bosEmb, bosPos).enumerated() { bos[i] = e + p }

        for row in 0..<2 {
            var offset = row * tPrefill * hidden
            for value in condEmb.values {
                ptr[offset] = value
                offset += 1
            }
            for (pos, id) in textIds.enumerated() {
                let posEmb = tables.textPos.row(pos)
                if row == 0 {
                    let tokEmb = tables.textEmb.row(id)
                    for (e, p) in zip(tokEmb, posEmb) {
                        ptr[offset] = e + p
                        offset += 1
                    }
                } else {
                    for p in posEmb {
                        ptr[offset] = p
                        offset += 1
                    }
                }
            }
            for _ in 0..<2 {
                for v in bos {
                    ptr[offset] = v
                    offset += 1
                }
            }
            _ = condLen  // context layout: cond + text + 2×BOS
        }
        return array
    }

    private func fillStepEmbeds(_ array: MLMultiArray, token: Int, step: Int) throws {
        let hidden = ChatterboxConstants.hiddenSize
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let tokEmb = models.tables.speechEmb.row(token)
        let posEmb = models.tables.speechPos.row(step + 1)
        for (i, (e, p)) in zip(tokEmb, posEmb).enumerated() {
            let v = e + p
            ptr[i] = v
            ptr[hidden + i] = v
        }
    }

    // MARK: - Alignment rows

    /// Prefill `align_attn` is `[3, 2, maxContext]` (three heads × two BOS
    /// queries); average heads → two `[ctx]` rows.
    private func prefillAlignRows(_ array: MLMultiArray, contextLen: Int) throws -> [[Float]] {
        let maxLen = ChatterboxConstants.maxContext
        let values = try floatBuffer(array)
        var rows = [[Float]](repeating: [Float](repeating: 0, count: contextLen), count: 2)
        for q in 0..<2 {
            for c in 0..<contextLen {
                var sum: Float = 0
                for h in 0..<3 { sum += values[(h * 2 + q) * maxLen + c] }
                rows[q][c] = sum / 3
            }
        }
        return rows
    }

    /// Decode `align_attn` is `[3, maxContext]`; average heads → one row.
    private func decodeAlignRow(_ values: [Float], contextLen: Int) -> [Float] {
        let maxLen = ChatterboxConstants.maxContext
        var row = [Float](repeating: 0, count: contextLen)
        for c in 0..<contextLen {
            row[c] = (values[c] + values[maxLen + c] + values[2 * maxLen + c]) / 3
        }
        return row
    }

    // MARK: - KV state seeding

    /// Copy prefill KV (`[layers, 2, heads, maxContext, headDim]`, possibly
    /// strided fp16 — see `floatBuffer`) into the decode model's per-layer
    /// fp16 state buffers.
    private func seedState(_ state: MLState, kvK: MLMultiArray, kvV: MLMultiArray) throws {
        let layerElements =
            2 * ChatterboxConstants.kvHeads * ChatterboxConstants.maxContext
            * ChatterboxConstants.headDim
        let kFloats = try floatBuffer(kvK)
        let vFloats = try floatBuffer(kvV)

        for layer in 0..<ChatterboxConstants.layerCount {
            for (name, src) in [("kv_k_\(layer)", kFloats), ("kv_v_\(layer)", vFloats)] {
                src.withUnsafeBufferPointer { srcBuf in
                    let base = srcBuf.baseAddress!.advanced(by: layer * layerElements)
                    state.withMultiArray(for: name) { array in
                        array.withUnsafeMutableBytes { rawBuffer, _ in
                            guard let dstBase = rawBuffer.baseAddress else { return }
                            // Address fp16 storage as raw bit patterns: Swift's
                            // `Float16` is unavailable on macOS x86_64.
                            let dst = dstBase.assumingMemoryBound(to: UInt16.self)
                            Float16Conversion.fromFloat32(
                                src: base, dst: dst, count: layerElements)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Sampling

    /// Upstream order: repetition penalty → temperature → min-p → top-p →
    /// softmax → multinomial. `logits` arrive CFG-combined and
    /// analyzer-adjusted.
    static func sample(
        logits: inout [Float], generated: [Int],
        temperature: Float, repetitionPenalty: Float, minP: Float, topP: Float,
        rng: inout SplitMix64
    ) -> Int {
        if repetitionPenalty != 1.0 {
            for id in Set(generated) {
                let score = logits[id]
                logits[id] = score < 0 ? score * repetitionPenalty : score / repetitionPenalty
            }
        }
        if temperature != 1.0 {
            let invTemp = 1.0 / max(temperature, 1e-6)
            for i in 0..<logits.count { logits[i] *= invTemp }
        }

        // Softmax (stable).
        var maxLogit = -Float.infinity
        for v in logits where v > maxLogit { maxLogit = v }
        var probs = [Float](repeating: 0, count: logits.count)
        var total: Float = 0
        for i in 0..<logits.count {
            let p = expf(logits[i] - maxLogit)
            probs[i] = p
            total += p
        }
        for i in 0..<probs.count { probs[i] /= total }

        // Min-p: drop tokens below minP × top probability.
        if minP > 0 {
            var topProb: Float = 0
            for p in probs where p > topProb { topProb = p }
            let threshold = minP * topProb
            var kept: Float = 0
            for i in 0..<probs.count {
                if probs[i] < threshold {
                    probs[i] = 0
                } else {
                    kept += probs[i]
                }
            }
            for i in 0..<probs.count { probs[i] /= kept }
        }

        // Top-p nucleus (no-op at the upstream default of 1.0).
        if topP < 1.0 {
            let order = probs.indices.sorted { probs[$0] > probs[$1] }
            var cumulative: Float = 0
            var kept: Float = 0
            var cut = false
            for idx in order {
                if cut {
                    probs[idx] = 0
                    continue
                }
                cumulative += probs[idx]
                kept += probs[idx]
                if cumulative >= topP { cut = true }
            }
            for i in 0..<probs.count { probs[i] /= kept }
        }

        var draw = Float(rng.nextUniform())
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
        let bucket = ChatterboxConstants.flowTokenBucket
        let melBucket = ChatterboxConstants.melFrameBucket
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

        // CFM initial noise: gaussian over the live frames, zeros beyond.
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
        let melBucket = ChatterboxConstants.melFrameBucket
        let promptFrames = 2 * models.voice.promptTokens.count
        let melValues = try floatBuffer(mel)  // [80 * melBucket]

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
        let harmonics = ChatterboxConstants.hiftHarmonics
        let phase = try MLMultiArray(
            shape: [1, NSNumber(value: harmonics), 1], dataType: .float32)
        let phasePtr = phase.dataPointer.assumingMemoryBound(to: Float.self)
        phasePtr[0] = 0
        for h in 1..<harmonics {
            phasePtr[h] = Float(rng.nextUniform()) * 2 * .pi - .pi
        }
        let sampleCount = melBucket * ChatterboxConstants.samplesPerMelFrame
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

        var samples = try floatBuffer(audio)
        let validSamples = melFrames * ChatterboxConstants.samplesPerMelFrame
        if samples.count > validSamples {
            samples.removeLast(samples.count - validSamples)
        }

        // Upstream trim fade: 20 ms silence + 20 ms half-cosine ramp-in to
        // reduce reference-clip spillover.
        let nTrim = ChatterboxConstants.sampleRate / 50
        for i in 0..<min(nTrim, samples.count) { samples[i] = 0 }
        for i in 0..<nTrim where nTrim + i < samples.count {
            let ramp = (cosf(.pi - .pi * Float(i) / Float(nTrim - 1)) + 1) / 2
            samples[nTrim + i] *= ramp
        }
        return samples
    }

    // MARK: - Helpers

    /// Copy an MLMultiArray into a dense row-major `[Float]`, honoring the
    /// array's strides and dtype. GPU-backed CoreML outputs routinely arrive
    /// as fp16 IOSurfaces with padded row strides (e.g. a [1, 80, 1000] mel
    /// with strides [80640, 1008, 1]) even when the model declares fp32
    /// outputs — a naive contiguous read smears every row.
    private func floatBuffer(_ array: MLMultiArray) throws -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)
        let rowLen = shape.last ?? 1
        guard strides.last == 1 else {
            throw ChatterboxError.processingFailed("innermost stride \(strides) unsupported")
        }
        let rows = count / max(rowLen, 1)
        let outerShape = shape.dropLast()
        let outerStrides = strides.dropLast()

        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { dst in
            let dstBase = dst.baseAddress!
            for row in 0..<rows {
                // Mixed-radix decode of the outer index → source offset.
                var remainder = row
                var srcOffset = 0
                for (dim, stride) in zip(outerShape, outerStrides).reversed() {
                    srcOffset += (remainder % dim) * stride
                    remainder /= dim
                }
                switch array.dataType {
                case .float32:
                    let src = array.dataPointer.assumingMemoryBound(to: Float.self)
                    dstBase.advanced(by: row * rowLen)
                        .update(from: src.advanced(by: srcOffset), count: rowLen)
                case .float16:
                    let src = array.dataPointer.assumingMemoryBound(to: UInt16.self)
                    Float16Conversion.toFloat32(
                        src: src.advanced(by: srcOffset),
                        dst: dstBase.advanced(by: row * rowLen), count: rowLen)
                default:
                    break
                }
            }
        }
        switch array.dataType {
        case .float32, .float16:
            return out
        default:
            throw ChatterboxError.processingFailed(
                "unexpected MLMultiArray dataType \(array.dataType.rawValue)")
        }
    }
}

extension SplitMix64 {
    /// Standard normal via Box–Muller.
    mutating func nextGaussian() -> Float {
        let u1 = max(nextUniform(), 1e-12)
        let u2 = nextUniform()
        return Float((-2.0 * Foundation.log(u1)).squareRoot() * Foundation.cos(2.0 * .pi * u2))
    }
}
