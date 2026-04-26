#if os(macOS)
import Accelerate
@preconcurrency import CoreML
import FluidAudio
import Foundation

/// Stage-by-stage parity probe for diagnosing where Swift diverges from the
/// Python+CoreML reference. Operates on an `.npz` fixture emitted by
/// `mobius/.../emit_parity_fixture.py`.
///
/// Stages:
///   1. `text_encoder` → `encoderOutput`
///   2. speaker prefill → `prefillCacheK{i}` / `prefillCacheV{i}` / `prefillPosition{i}`
///   3. AR `decoder_step` replay → `perStepDecoderHidden` (skips Swift LT/sampler)
///
/// Each stage prints MAE / max|Δ| / SNR. Whichever stage first shows non-trivial
/// drift is the layer that broke parity; everything upstream is provably correct.
public enum MagpieProbeCommand {

    private static let logger = AppLogger(category: "MagpieProbe")

    public static func run(arguments: [String]) async {
        var fixturePath: String? = nil
        var text: String? = nil
        var languageCode = "en"
        var speakerIdx = 0
        var stagesArg = "1,2,3"

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--fixture":
                if i + 1 < arguments.count {
                    fixturePath = arguments[i + 1]
                    i += 1
                }
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    speakerIdx = v
                    i += 1
                }
            case "--stages":
                if i + 1 < arguments.count {
                    stagesArg = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        guard let fixturePath = fixturePath else {
            logger.error("--fixture <npz> is required")
            exit(1)
        }
        guard let text = text else {
            logger.error("--text \"…\" is required")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Unknown language code \(languageCode)")
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker \(speakerIdx)")
            exit(1)
        }
        let stages: Set<Int> = Set(stagesArg.split(separator: ",").compactMap { Int($0) })

        do {
            stderr("Loading fixture \(fixturePath)…")
            let fixture = try MagpieNpzReader.read(from: URL(fileURLWithPath: fixturePath))
            stderr(
                "  keys: \(fixture.keys.sorted().joined(separator: ", "))")

            stderr("Initialising Magpie store…")
            let store = MagpieModelStore(preferredLanguages: [language])
            try await store.loadIfNeeded()
            let bundle = try await store.constants()
            let repoDir = try await store.repoDir()
            let tokenizerDir = MagpieResourceDownloader.tokenizerDirectory(in: repoDir)
            let tokenizer = MagpieTokenizer(
                tokenizerDir: tokenizerDir, eosId: bundle.textEosId)

            // Tokenise once — used by all stages.
            let opts = MagpieSynthesisOptions()
            let tokenized = try await tokenizer.tokenize(
                text, language: language, options: opts)
            stderr("Tokenised: realLength=\(tokenized.realLength) (eos=\(bundle.textEosId))")

            // ---------- Stage 1 ----------
            var encoderOutputArray: MLMultiArray? = nil
            var encoderMaskArray: MLMultiArray? = nil
            if stages.contains(1) {
                let result = try await runStage1(
                    tokenized: tokenized, store: store, fixture: fixture,
                    maxTextLen: bundle.config.maxTextLength)
                encoderOutputArray = result.encoderOutput
                encoderMaskArray = result.encoderMask
            }

            // ---------- Stage 2 ----------
            var condCache: MagpieKvCache? = nil
            if stages.contains(2) || stages.contains(3) {
                guard let encOut = encoderOutputArray, let encMask = encoderMaskArray else {
                    stderr("Stage 2 requires Stage 1; re-run with --stages 1,2 or 1,2,3")
                    exit(1)
                }
                condCache = try await runStage2(
                    speaker: speaker, store: store, fixture: fixture,
                    encoderOutput: encOut, encoderMask: encMask,
                    config: bundle.config,
                    speakerEmbedding: bundle.speakerEmbeddings[speakerIdx])
            }

            // ---------- Stage 3 ----------
            if stages.contains(3) {
                guard let cache = condCache,
                    let encOut = encoderOutputArray, let encMask = encoderMaskArray
                else {
                    stderr("Stage 3 requires Stage 1 + 2")
                    exit(1)
                }
                try await runStage3(
                    store: store, fixture: fixture, cache: cache,
                    encoderOutput: encOut, encoderMask: encMask,
                    audioEmbeddings: bundle.audioEmbeddings,
                    config: bundle.config)
            }
        } catch {
            logger.error("Probe failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - Stage 1: text_encoder

    private static func runStage1(
        tokenized: MagpieTokenizedText,
        store: MagpieModelStore,
        fixture: [String: NpyReader.Array],
        maxTextLen: Int
    ) async throws -> (encoderOutput: MLMultiArray, encoderMask: MLMultiArray) {
        stderr("\n=== Stage 1: text_encoder ===")
        let model = try await store.textEncoder()

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

        // Compare paddedIds + mask against fixture before running encoder.
        if let padded = fixture["textTokensPadded"] {
            var matches = 0
            for i in 0..<min(padded.data.count, tokenized.paddedIds.count)
            where Int32(padded.data[i]) == tokenized.paddedIds[i] {
                matches += 1
            }
            stderr(
                "  textTokensPadded: \(matches)/\(min(padded.data.count, tokenized.paddedIds.count)) match"
            )
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokenArr),
            "text_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let out = try await model.prediction(from: provider)
        guard let encOut = out.featureValue(for: "encoder_output")?.multiArrayValue else {
            stderr("  encoder_output key missing!")
            exit(1)
        }

        let actual = mlArrayToFloat(encOut)
        if let expected = fixture["encoderOutput"] {
            let stat = compare(actual: actual, expected: expected.data)
            stderr(
                "  encoderOutput \(expected.shape): MAE=\(fmt(stat.mae)) max|Δ|=\(fmt(stat.maxAbs)) SNR=\(snrFmt(stat.snrDb)) dB"
            )
        } else {
            stderr("  encoderOutput key missing in fixture")
        }
        return (encOut, maskArr)
    }

    // MARK: - Stage 2: speaker prefill

    private static func runStage2(
        speaker: MagpieSpeaker,
        store: MagpieModelStore,
        fixture: [String: NpyReader.Array],
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        config: MagpieModelConfig,
        speakerEmbedding: [Float]
    ) async throws -> MagpieKvCache {
        stderr("\n=== Stage 2: speaker prefill ===")
        let cache = try MagpieKvCache(
            numLayers: config.numDecoderLayers,
            maxCacheLength: config.maxCacheLength,
            numHeads: config.numHeads,
            headDim: config.headDim)
        let prefill = MagpiePrefill(decoderStep: try await store.decoderStep())
        try prefill.prefill(
            speakerEmbedding: speakerEmbedding,
            speakerContextLength: config.speakerContextLength,
            dModel: config.dModel,
            encoderOutput: encoderOutput,
            encoderMask: encoderMask,
            cache: cache)

        // Compare each layer's K, V, position against fixture.
        var worstK = 0.0
        var worstV = 0.0
        for layer in 0..<config.numDecoderLayers {
            let actK = mlArrayToFloat(cache.cachesK[layer])
            let actV = mlArrayToFloat(cache.cachesV[layer])
            let actPos = mlArrayToFloat(cache.positions[layer])

            if let exp = fixture["prefillCacheK\(layer)"] {
                let s = compare(actual: actK, expected: exp.data)
                worstK = max(worstK, s.mae)
                if layer == 0 || layer == config.numDecoderLayers - 1 {
                    stderr(
                        "  L\(layer) K shape=\(exp.shape) MAE=\(fmt(s.mae)) max|Δ|=\(fmt(s.maxAbs)) SNR=\(snrFmt(s.snrDb))"
                    )
                }
            }
            if let exp = fixture["prefillCacheV\(layer)"] {
                let s = compare(actual: actV, expected: exp.data)
                worstV = max(worstV, s.mae)
                if layer == 0 || layer == config.numDecoderLayers - 1 {
                    stderr(
                        "  L\(layer) V shape=\(exp.shape) MAE=\(fmt(s.mae)) max|Δ|=\(fmt(s.maxAbs)) SNR=\(snrFmt(s.snrDb))"
                    )
                }
            }
            if let exp = fixture["prefillPosition\(layer)"] {
                let py = exp.data.first ?? -1
                let sw = actPos.first ?? -1
                if layer == 0 {
                    stderr("  L\(layer) position: swift=\(sw) python=\(py)")
                }
            }
        }
        stderr(
            "  worst-layer MAE: K=\(fmt(worstK)) V=\(fmt(worstV)) (across \(config.numDecoderLayers) layers)"
        )
        return cache
    }

    // MARK: - Stage 3: AR decoder_step replay

    private static func runStage3(
        store: MagpieModelStore,
        fixture: [String: NpyReader.Array],
        cache: MagpieKvCache,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        audioEmbeddings: [[Float]],
        config: MagpieModelConfig
    ) async throws {
        stderr("\n=== Stage 3: decoder_step AR replay (Python codes) ===")
        guard let codesArr = fixture["perStepCodes"],
            codesArr.shape.count == 2
        else {
            stderr("  perStepCodes missing or wrong shape")
            return
        }
        guard let hiddenArr = fixture["perStepDecoderHidden"],
            hiddenArr.shape.count == 2
        else {
            stderr("  perStepDecoderHidden missing or wrong shape")
            return
        }

        let numSteps = codesArr.shape[0]
        let numCodebooks = codesArr.shape[1]
        let dModel = hiddenArr.shape[1]
        precondition(numCodebooks == config.numCodebooks)
        precondition(dModel == config.dModel)

        let decoderStep = try await store.decoderStep()

        // BOS frame: same as MagpieSynthesizer — at step 0, codes are all audio_bos_id;
        // at step k>0, codes are perStepCodes[k-1] (the codes sampled to produce step k).
        var prevCodes: [Int32] = Swift.Array(
            repeating: config.audioBosId, count: numCodebooks)

        var totalMae = 0.0
        var worstMae = 0.0
        var worstStep = 0
        for step in 0..<numSteps {
            let codes: [Int32]
            if step == 0 {
                codes = prevCodes
            } else {
                let row = step - 1
                codes = (0..<numCodebooks).map {
                    Int32(codesArr.data[row * numCodebooks + $0])
                }
            }
            prevCodes = codes

            // Embed (mean of 8 codebook rows).
            let audioEmbed = try MLMultiArray(
                shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)
            audioEmbed.withUnsafeMutableBytes { ptr, _ in
                let base = ptr.bindMemory(to: Float.self).baseAddress!
                for j in 0..<dModel { base[j] = 0 }
                for cb in 0..<numCodebooks {
                    let row = Int(codes[cb])
                    let table = audioEmbeddings[cb]
                    let start = row * dModel
                    for j in 0..<dModel { base[j] += table[start + j] }
                }
                let inv = 1.0 / Float(numCodebooks)
                for j in 0..<dModel { base[j] *= inv }
            }

            var inputs: [String: MLMultiArray] = [
                "audio_embed": audioEmbed,
                "encoder_output": encoderOutput,
                "encoder_mask": encoderMask,
            ]
            cache.addInputs(to: &inputs)
            let provider = try MLDictionaryFeatureProvider(
                dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            let out = try await decoderStep.prediction(from: provider)
            try cache.absorbOutputs(out)

            guard
                let h = out.featureValue(for: MagpieKvCache.decoderHiddenKey)?
                    .multiArrayValue
            else {
                stderr("  step \(step): missing hidden output")
                return
            }
            let swiftHidden = mlArrayToFloat(h)
            let pyHidden = Swift.Array(
                hiddenArr.data[(step * dModel)..<((step + 1) * dModel)])
            let s = compare(actual: swiftHidden, expected: pyHidden)
            totalMae += s.mae
            if s.mae > worstMae {
                worstMae = s.mae
                worstStep = step
            }
            if step < 3 || step == numSteps - 1 {
                stderr(
                    "  step \(step): MAE=\(fmt(s.mae)) max|Δ|=\(fmt(s.maxAbs)) SNR=\(snrFmt(s.snrDb))"
                )
            }
        }
        let avgMae = totalMae / Double(numSteps)
        stderr(
            "  summary: avgMAE=\(fmt(avgMae)) worstMAE=\(fmt(worstMae)) at step \(worstStep) (over \(numSteps) steps)"
        )
    }

    // MARK: - Helpers

    private struct Stat {
        let mae: Double
        let maxAbs: Double
        let snrDb: Double
    }

    private static func compare(actual: [Float], expected: [Float]) -> Stat {
        let n = min(actual.count, expected.count)
        var sumAbs: Double = 0
        var sumSq: Double = 0
        var sumRefSq: Double = 0
        var maxAbs: Double = 0
        for i in 0..<n {
            let d = Double(actual[i] - expected[i])
            let ad = abs(d)
            sumAbs += ad
            sumSq += d * d
            sumRefSq += Double(expected[i]) * Double(expected[i])
            if ad > maxAbs { maxAbs = ad }
        }
        let mae = sumAbs / Double(n)
        let mse = sumSq / Double(n)
        let refPower = sumRefSq / Double(n)
        let snrDb: Double
        if mse > 0 && refPower > 0 {
            snrDb = 10 * log10(refPower / mse)
        } else if mse == 0 {
            snrDb = .infinity
        } else {
            snrDb = -.infinity
        }
        return Stat(mae: mae, maxAbs: maxAbs, snrDb: snrDb)
    }

    private static func mlArrayToFloat(_ arr: MLMultiArray) -> [Float] {
        var out = Swift.Array<Float>(repeating: 0, count: arr.count)
        switch arr.dataType {
        case .float32:
            arr.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float.self)
                for i in 0..<arr.count { out[i] = p[i] }
            }
        case .float16:
            arr.withUnsafeBytes { raw in
                guard let src = raw.baseAddress else { return }
                var s = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: vImagePixelCount(arr.count), rowBytes: arr.count * 2)
                out.withUnsafeMutableBufferPointer { dst in
                    var d = vImage_Buffer(
                        data: dst.baseAddress, height: 1,
                        width: vImagePixelCount(arr.count), rowBytes: arr.count * 4)
                    _ = vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
                }
            }
        case .double:
            arr.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Double.self)
                for i in 0..<arr.count { out[i] = Float(p[i]) }
            }
        default:
            for i in 0..<arr.count { out[i] = arr[i].floatValue }
        }
        return out
    }

    private static func fmt(_ v: Double) -> String { String(format: "%.6e", v) }
    private static func snrFmt(_ v: Double) -> String {
        if v.isFinite { return String(format: "%.2f", v) }
        return v > 0 ? "+inf" : "-inf"
    }

    private static func stderr(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }
}
#endif
