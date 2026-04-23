import Accelerate
@preconcurrency import CoreML
import Foundation

// Self-contained Cohere Transcribe pipeline (cache-external decoder):
//   1. Mel spectrogram matches FilterbankFeatures (n_fft=512, preemph,
//      slaney mel, natural-log + 2^-24 guard, per-feature CMVN ddof=1).
//   2. Cross-attention mask: additive, -1e4 for padded encoder frames
//      (valid = ceil(feature_length * 438 / 3500)).
//   3. Decode: repetition penalty + no-repeat n-gram + SentencePiece
//      byte-fallback detokenization so CJK comes out as real characters.
//
// Supports mixed-precision: load the encoder and decoder from different
// directories (e.g. INT8 encoder for speed, FP16 decoder for quality).

private let pipelineLogger = AppLogger(category: "CoherePipeline")

// MARK: - Error type

public enum CohereAsrError: Error, LocalizedError {
    case modelNotFound(String)
    case encodingFailed(String)
    case decodingFailed(String)
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return "Model not found: \(msg)"
        case .encodingFailed(let msg): return "Encoding failed: \(msg)"
        case .decodingFailed(let msg): return "Decoding failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        }
    }
}

// MARK: - Mel spectrogram (FilterbankFeatures-compatible)

@available(macOS 14, iOS 17, *)
public final class CohereMelSpectrogram {

    public struct Config: Sendable {
        public let sampleRate: Int
        public let winLength: Int
        public let hopLength: Int
        public let nMels: Int
        public let fMin: Float
        public let fMax: Float
        public let preemph: Float
        public let magPower: Float
        public let logZeroGuard: Float
        public let cmvnEpsilon: Float

        public init(
            sampleRate: Int = 16_000,
            winLength: Int = 400,
            hopLength: Int = 160,
            nMels: Int = 128,
            fMin: Float = 0.0,
            fMax: Float = 8_000.0,
            preemph: Float = 0.97,
            magPower: Float = 2.0,
            logZeroGuard: Float = 5.960_464_5e-08,  // 2^-24
            cmvnEpsilon: Float = 1.0e-5
        ) {
            self.sampleRate = sampleRate
            self.winLength = winLength
            self.hopLength = hopLength
            self.nMels = nMels
            self.fMin = fMin
            self.fMax = fMax
            self.preemph = preemph
            self.magPower = magPower
            self.logZeroGuard = logZeroGuard
            self.cmvnEpsilon = cmvnEpsilon
        }
    }

    public let config: Config
    public let nFFT: Int
    private let paddedWindow: [Float]
    private let melFilter: [[Float]]
    private let fftSetup: vDSP.FFT<DSPSplitComplex>

    public init(config: Config = Config()) {
        self.config = config
        self.nFFT = Self.nextPowerOfTwo(atLeast: config.winLength)

        // Symmetric hann window (periodic=false): 0.5 * (1 - cos(2π n / (N-1))).
        var hann = [Float](repeating: 0, count: config.winLength)
        if config.winLength > 1 {
            let denom = Float(config.winLength - 1)
            for n in 0..<config.winLength {
                hann[n] = 0.5 * (1.0 - cos(2.0 * .pi * Float(n) / denom))
            }
        }

        // torch.stft zero-pads the window symmetrically to n_fft.
        var padded = [Float](repeating: 0, count: self.nFFT)
        if config.winLength < self.nFFT {
            let padLeft = (self.nFFT - config.winLength) / 2
            for n in 0..<config.winLength { padded[padLeft + n] = hann[n] }
        } else {
            padded = hann
        }
        self.paddedWindow = padded

        self.melFilter = Self.slaneyMelFilter(
            sampleRate: config.sampleRate, nFFT: self.nFFT,
            nMels: config.nMels, fMin: config.fMin, fMax: config.fMax)

        let log2n = vDSP_Length(Int(log2(Double(self.nFFT))))
        guard let setup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self) else {
            fatalError("vDSP.FFT setup failed for n=\(self.nFFT)")
        }
        self.fftSetup = setup
    }

    public func validFrameCount(forSamples n: Int) -> Int { max(0, n) / config.hopLength }

    public struct Output {
        public let mel: [[Float]]  // (nMels, nFrames)
        public let validFrames: Int
    }

    public func compute(audio: [Float]) -> Output {
        var samples = audio
        let validFrames = validFrameCount(forSamples: samples.count)

        // Preemphasis over valid samples only.
        if config.preemph != 0, samples.count > 1 {
            var filtered = [Float](repeating: 0, count: samples.count)
            filtered[0] = samples[0]
            for i in 1..<samples.count {
                filtered[i] = samples[i] - config.preemph * samples[i - 1]
            }
            samples = filtered
        }

        // center=True, constant padding.
        let padSize = nFFT / 2
        var padded = [Float](repeating: 0, count: samples.count + 2 * padSize)
        for i in 0..<samples.count { padded[padSize + i] = samples[i] }

        let nFrames = 1 + (padded.count - nFFT) / config.hopLength
        let nBins = nFFT / 2 + 1

        var power = [Float](repeating: 0, count: nBins * nFrames)

        let halfN = nFFT / 2
        var realp = [Float](repeating: 0, count: halfN)
        var imagp = [Float](repeating: 0, count: halfN)
        var windowed = [Float](repeating: 0, count: nFFT)

        for frame in 0..<nFrames {
            let start = frame * config.hopLength
            vDSP_vmul(
                UnsafePointer(padded).advanced(by: start), 1,
                paddedWindow, 1,
                &windowed, 1,
                vDSP_Length(nFFT))

            // Pack to split-complex.
            windowed.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                base.withMemoryRebound(to: DSPComplex.self, capacity: halfN) { cbuf in
                    realp.withUnsafeMutableBufferPointer { rBuf in
                        imagp.withUnsafeMutableBufferPointer { iBuf in
                            var split = DSPSplitComplex(
                                realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                            vDSP_ctoz(cbuf, 2, &split, 1, vDSP_Length(halfN))
                        }
                    }
                }
            }

            realp.withUnsafeMutableBufferPointer { rBuf in
                imagp.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(
                        realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    fftSetup.forward(input: split, output: &split)
                }
            }

            // vDSP packed: realp[0] = DC real, imagp[0] = Nyquist real.
            // Interior bins (1..<halfN): (realp[k], imagp[k]). Scale by 0.5 to
            // match numpy/torch rfft.
            let scale: Float = 0.5
            let dcMag = abs(realp[0] * scale)
            power[0 * nFrames + frame] = pow(dcMag, config.magPower)
            let nyMag = abs(imagp[0] * scale)
            power[(nBins - 1) * nFrames + frame] = pow(nyMag, config.magPower)
            for k in 1..<halfN {
                let r = realp[k] * scale
                let im = imagp[k] * scale
                let mag = sqrt(r * r + im * im)
                power[k * nFrames + frame] = pow(mag, config.magPower)
            }
        }

        // Mel filterbank.
        var mel = [[Float]](repeating: [Float](repeating: 0, count: nFrames), count: config.nMels)
        for m in 0..<config.nMels {
            let filt = melFilter[m]
            for f in 0..<nFrames {
                var sum: Float = 0
                for k in 0..<nBins { sum += filt[k] * power[k * nFrames + f] }
                mel[m][f] = sum
            }
        }

        // Natural log with additive guard.
        for m in 0..<config.nMels {
            for f in 0..<nFrames {
                mel[m][f] = log(mel[m][f] + config.logZeroGuard)
            }
        }

        // Per-feature CMVN (ddof=1) over valid frames.
        if validFrames > 1 {
            for m in 0..<config.nMels {
                var mean: Float = 0
                for f in 0..<validFrames { mean += mel[m][f] }
                mean /= Float(validFrames)
                var ssq: Float = 0
                for f in 0..<validFrames {
                    let d = mel[m][f] - mean
                    ssq += d * d
                }
                let variance = ssq / Float(validFrames - 1)
                var std = sqrt(variance)
                if !std.isFinite { std = 0.0 }
                let denom = std + config.cmvnEpsilon
                for f in 0..<validFrames { mel[m][f] = (mel[m][f] - mean) / denom }
            }
        }

        // Zero invalid trailing frames.
        if validFrames < nFrames {
            for m in 0..<config.nMels {
                for f in validFrames..<nFrames { mel[m][f] = 0 }
            }
        }

        return Output(mel: mel, validFrames: validFrames)
    }

    /// Pad or truncate mel to fixed frames for the encoder's fixed input.
    public static func padOrTruncate(
        mel: [[Float]], validFrames: Int, fixedFrames: Int = 3_500
    ) -> (mel: [[Float]], featureLength: Int) {
        let nMels = mel.count
        guard nMels > 0 else { return (mel, 0) }
        let cur = mel[0].count
        if cur == fixedFrames { return (mel, min(validFrames, fixedFrames)) }
        if cur > fixedFrames {
            return (mel.map { Array($0.prefix(fixedFrames)) }, min(validFrames, fixedFrames))
        }
        let padLen = fixedFrames - cur
        let padded: [[Float]] = mel.map { $0 + [Float](repeating: 0, count: padLen) }
        return (padded, min(validFrames, fixedFrames))
    }

    private static func nextPowerOfTwo(atLeast n: Int) -> Int {
        var x = 1
        while x < n { x <<= 1 }
        return x
    }

    // MARK: Slaney mel filterbank

    private static func slaneyMelFilter(
        sampleRate: Int, nFFT: Int, nMels: Int, fMin: Float, fMax: Float
    ) -> [[Float]] {
        let nBins = nFFT / 2 + 1
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins { fftFreqs[k] = Float(sampleRate) * Float(k) / Float(nFFT) }

        let melMin = hzToMelSlaney(fMin)
        let melMax = hzToMelSlaney(fMax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        let step = (melMax - melMin) / Float(nMels + 1)
        for i in 0..<(nMels + 2) { melPoints[i] = melMin + Float(i) * step }
        let hzPoints = melPoints.map { melToHzSlaney($0) }

        var fb = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nMels)
        for m in 0..<nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            let leftDen = max(center - lower, 1e-10)
            let rightDen = max(upper - center, 1e-10)
            for k in 0..<nBins {
                let f = fftFreqs[k]
                if f < lower || f > upper { continue }
                fb[m][k] = (f <= center) ? (f - lower) / leftDen : (upper - f) / rightDen
            }
            let enorm: Float = 2.0 / max(upper - lower, 1e-10)
            for k in 0..<nBins { fb[m][k] *= enorm }
        }
        return fb
    }

    private static let slaneyMinLogHz: Float = 1000.0
    private static let slaneyMinLogMel: Float = 15.0
    private static let slaneyLogStep: Float = 0.06875177742  // log(6.4) / 27

    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        if hz >= slaneyMinLogHz {
            return slaneyMinLogMel + log(hz / slaneyMinLogHz) / slaneyLogStep
        }
        return hz / fSp
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        if mel >= slaneyMinLogMel {
            return slaneyMinLogHz * exp(slaneyLogStep * (mel - slaneyMinLogMel))
        }
        return fSp * mel
    }
}

// MARK: - Pipeline

@available(macOS 14, iOS 17, *)
public actor CoherePipeline {

    public struct LoadedModels: Sendable {
        public let encoder: MLModel
        public let decoder: MLModel
        public let vocabulary: [Int: String]
    }

    private let melExtractor: CohereMelSpectrogram

    public init(mel: CohereMelSpectrogram = CohereMelSpectrogram()) {
        self.melExtractor = mel
    }

    /// Load encoder, decoder and vocabulary from (potentially different) directories.
    ///
    /// This supports mixed-precision — e.g. INT8 encoder + FP16 decoder. Pass
    /// the same URL to both for a single-precision setup. Each directory must
    /// contain `cohere_encoder.mlmodelc` and/or
    /// `cohere_decoder_cache_external.mlmodelc`; the vocab is loaded from
    /// `vocabDir`.
    public static func loadModels(
        encoderDir: URL,
        decoderDir: URL,
        vocabDir: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> LoadedModels {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let encoderURL = encoderDir.appendingPathComponent("cohere_encoder.mlmodelc")
        let decoderURL = decoderDir.appendingPathComponent("cohere_decoder_cache_external.mlmodelc")

        pipelineLogger.info("Loading encoder from \(encoderURL.path)")
        let encoder = try await MLModel.load(contentsOf: encoderURL, configuration: config)
        pipelineLogger.info("Loading decoder from \(decoderURL.path)")
        let decoder = try await MLModel.load(contentsOf: decoderURL, configuration: config)

        // Sanity: decoder must expose k_cache_0 (cache-external variant).
        let decoderInputs = decoder.modelDescription.inputDescriptionsByName.keys
        guard decoderInputs.contains("k_cache_0") else {
            throw CohereAsrError.decodingFailed(
                "Decoder at \(decoderURL.path) is not cache-external (missing k_cache_0).")
        }

        let vocab = try loadVocabulary(vocabDir: vocabDir)
        return LoadedModels(encoder: encoder, decoder: decoder, vocabulary: vocab)
    }

    private static func loadVocabulary(vocabDir: URL) throws -> [Int: String] {
        let vocabURL = vocabDir.appendingPathComponent("vocab.json")
        let data = try Data(contentsOf: vocabURL)
        let json = try JSONSerialization.jsonObject(with: data)
        guard let dict = json as? [String: String] else {
            throw CohereAsrError.modelNotFound("Invalid vocab.json format")
        }
        var vocab: [Int: String] = [:]
        vocab.reserveCapacity(dict.count)
        for (key, value) in dict {
            if let tokenId = Int(key) { vocab[tokenId] = value }
        }
        pipelineLogger.info("Loaded vocabulary with \(vocab.count) tokens")
        return vocab
    }

    // MARK: Transcribe

    public struct TranscriptionResult: Sendable {
        public let text: String
        public let tokenIds: [Int]
        public let encoderSeconds: Double
        public let decoderSeconds: Double
        public let totalSeconds: Double
    }

    public func transcribe(
        audio: [Float],
        models: LoadedModels,
        language: CohereAsrConfig.Language = .english,
        maxNewTokens: Int = 108,
        repetitionPenalty: Float = 1.1,
        noRepeatNgram: Int = 3
    ) async throws -> TranscriptionResult {
        let total0 = CFAbsoluteTimeGetCurrent()

        // 1) Mel spectrogram + pad to 3500
        let melOut = melExtractor.compute(audio: audio)
        let (paddedMel, featureLength) = CohereMelSpectrogram.padOrTruncate(
            mel: melOut.mel, validFrames: melOut.validFrames, fixedFrames: 3_500)
        guard featureLength > 0 else {
            throw CohereAsrError.invalidInput("Audio too short to extract mel features")
        }

        // 2) Encoder
        let enc0 = CFAbsoluteTimeGetCurrent()
        let (encoderHidden, encoderValid) = try await runEncoder(
            paddedMel: paddedMel, featureLength: featureLength, encoder: models.encoder)
        let encoderSecs = CFAbsoluteTimeGetCurrent() - enc0

        // 3) Cache-external decode
        let dec0 = CFAbsoluteTimeGetCurrent()
        let outputTokens = try await decodeCacheExternal(
            encoderHidden: encoderHidden,
            encoderValidFrames: encoderValid,
            language: language,
            maxNewTokens: maxNewTokens,
            repetitionPenalty: repetitionPenalty,
            noRepeatNgram: noRepeatNgram,
            decoder: models.decoder)
        let decoderSecs = CFAbsoluteTimeGetCurrent() - dec0

        // 4) Detokenize
        let text = Self.convertTokensToText(outputTokens, vocabulary: models.vocabulary)

        return TranscriptionResult(
            text: text,
            tokenIds: outputTokens,
            encoderSeconds: encoderSecs,
            decoderSeconds: decoderSecs,
            totalSeconds: CFAbsoluteTimeGetCurrent() - total0)
    }

    // MARK: Encoder

    private func runEncoder(
        paddedMel: [[Float]], featureLength: Int, encoder: MLModel
    ) async throws -> (hidden: MLMultiArray, encoderValidFrames: Int) {
        let nMels = CohereAsrConfig.numMelBins
        let nFrames = 3_500

        let mel = try MLMultiArray(
            shape: [1, NSNumber(value: nMels), NSNumber(value: nFrames)], dataType: .float32)
        let melPtr = mel.dataPointer.bindMemory(to: Float.self, capacity: nMels * nFrames)
        for m in 0..<nMels {
            let row = paddedMel[m]
            for f in 0..<nFrames { melPtr[m * nFrames + f] = row[f] }
        }

        let featLen = try MLMultiArray(shape: [1], dataType: .int32)
        featLen[0] = NSNumber(value: featureLength)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: mel),
            "feature_length": MLFeatureValue(multiArray: featLen),
        ])
        let out = try await encoder.prediction(from: provider)
        guard let hidden = out.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw CohereAsrError.encodingFailed("Encoder output missing 'hidden_states'")
        }
        let encoderSeqLen = hidden.shape[1].intValue
        let encoderValid = Self.encoderValidFrames(
            featureLength: featureLength, encoderSeqLen: encoderSeqLen)
        return (hidden, encoderValid)
    }

    /// ceil(feature_length * 438 / 3500), clamped to [1, encoderSeqLen].
    public static func encoderValidFrames(
        featureLength: Int, encoderSeqLen: Int,
        melFrames: Int = 3_500, encoderFrames: Int = 438
    ) -> Int {
        let raw = Int((Double(featureLength * encoderFrames) / Double(melFrames)).rounded(.up))
        return max(1, min(raw, encoderSeqLen))
    }

    // MARK: Decoder (cache-external with corrected mask + penalties)

    private func decodeCacheExternal(
        encoderHidden: MLMultiArray,
        encoderValidFrames: Int,
        language: CohereAsrConfig.Language,
        maxNewTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        decoder: MLModel
    ) async throws -> [Int] {
        let cacheShape: [NSNumber] = [
            1,
            NSNumber(value: CohereAsrConfig.numDecoderHeads),
            NSNumber(value: CohereAsrConfig.maxSeqLen),
            NSNumber(value: CohereAsrConfig.headDim),
        ]

        // Probe the expected cache dtype from the model description to match
        // whatever precision the .mlmodelc was compiled with.
        let cacheDType: MLMultiArrayDataType = {
            if let desc = decoder.modelDescription.inputDescriptionsByName["k_cache_0"],
                desc.type == .multiArray,
                let arrCon = desc.multiArrayConstraint
            {
                return arrCon.dataType
            }
            return .float16
        }()

        var kCaches: [MLMultiArray] = []
        var vCaches: [MLMultiArray] = []
        kCaches.reserveCapacity(CohereAsrConfig.numDecoderLayers)
        vCaches.reserveCapacity(CohereAsrConfig.numDecoderLayers)
        for _ in 0..<CohereAsrConfig.numDecoderLayers {
            let k = try MLMultiArray(shape: cacheShape, dataType: cacheDType)
            let v = try MLMultiArray(shape: cacheShape, dataType: cacheDType)
            Self.zeroFill(k)
            Self.zeroFill(v)
            kCaches.append(k)
            vCaches.append(v)
        }

        let encoderSeqLen = encoderHidden.shape[1].intValue
        let crossMask = try Self.buildCrossAttentionMask(
            encoderSeqLen: encoderSeqLen,
            encoderValid: encoderValidFrames,
            dtype: cacheDType)

        let prompt = language.promptSequence
        var allTokens: [Int] = []
        var outputTokens: [Int] = []
        var currentToken = prompt[0]
        let effectiveMax = min(maxNewTokens + prompt.count, CohereAsrConfig.maxSeqLen)

        for step in 0..<effectiveMax {
            if step < prompt.count { currentToken = prompt[step] }

            let inputId = try MLMultiArray(shape: [1, 1], dataType: .int32)
            inputId[0] = NSNumber(value: currentToken)

            let positionId = try MLMultiArray(shape: [1, 1], dataType: .int32)
            positionId[0] = NSNumber(value: step)

            let selfMask = try MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: step + 1)], dataType: cacheDType)
            Self.zeroFill(selfMask)

            var inputs: [String: MLFeatureValue] = [
                "input_id": MLFeatureValue(multiArray: inputId),
                "position_id": MLFeatureValue(multiArray: positionId),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "cross_attention_mask": MLFeatureValue(multiArray: crossMask),
                "attention_mask": MLFeatureValue(multiArray: selfMask),
            ]
            for i in 0..<CohereAsrConfig.numDecoderLayers {
                inputs["k_cache_\(i)"] = MLFeatureValue(multiArray: kCaches[i])
                inputs["v_cache_\(i)"] = MLFeatureValue(multiArray: vCaches[i])
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let out = try await decoder.prediction(from: provider)

            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.decodingFailed("Decoder output missing 'logits'")
            }

            var logitBuf = Self.copyLogitsFloat32(logits)
            Self.applyRepetitionPenalty(&logitBuf, history: allTokens, penalty: repetitionPenalty)
            Self.applyNoRepeatNgram(&logitBuf, history: allTokens, n: noRepeatNgram)

            let nextToken = Self.argmax(logitBuf)
            // Record the token that was actually consumed at this step:
            // a prompt token during prompt-feeding (model prediction discarded,
            // input is forced) and the previously-generated token afterwards.
            // Recording the phantom prompt-phase prediction would cause
            // `applyNoRepeatNgram` to suppress valid output tokens.
            // Note: at the prompt/output boundary (step == prompt.count - 1),
            // `currentToken` is still the last prompt token and `nextToken` is
            // the first real output. The first output is recorded on the next
            // iteration once it has rotated into `currentToken`.
            allTokens.append(currentToken)

            if step >= prompt.count - 1, nextToken == CohereAsrConfig.SpecialTokens.eosToken {
                break
            }
            if step >= prompt.count - 1 {
                outputTokens.append(nextToken)
            }

            for i in 0..<CohereAsrConfig.numDecoderLayers {
                guard
                    let kOut = out.featureValue(for: "k_cache_\(i)_out")?.multiArrayValue,
                    let vOut = out.featureValue(for: "v_cache_\(i)_out")?.multiArrayValue
                else {
                    throw CohereAsrError.decodingFailed(
                        "Missing updated cache for layer \(i)")
                }
                kCaches[i] = kOut
                vCaches[i] = vOut
            }

            currentToken = (step < prompt.count - 1) ? prompt[step + 1] : nextToken
        }

        return outputTokens
    }

    // MARK: Helpers

    static func buildCrossAttentionMask(
        encoderSeqLen: Int, encoderValid: Int, dtype: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: encoderSeqLen)], dataType: dtype)
        zeroFill(mask)
        let valid = max(0, min(encoderValid, encoderSeqLen))
        guard valid < encoderSeqLen else { return mask }

        // Fill invalid positions with -1e4 (additive mask).
        let invalidValue: Float = -1.0e4
        switch dtype {
        case .float32:
            let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: encoderSeqLen)
            for i in valid..<encoderSeqLen { ptr[i] = invalidValue }
        case .float16:
            let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: encoderSeqLen)
            let bits = float16Bits(invalidValue)
            for i in valid..<encoderSeqLen { ptr[i] = bits }
        default:
            for i in valid..<encoderSeqLen {
                mask[[0, 0, 0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: invalidValue)
            }
        }
        return mask
    }

    /// Convert a Float32 value to IEEE 754 binary16 bit pattern.
    private static func float16Bits(_ value: Float) -> UInt16 {
        var v = value
        var dst: UInt16 = 0
        withUnsafeMutablePointer(to: &v) { srcPtr in
            withUnsafeMutablePointer(to: &dst) { dstPtr in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(srcPtr),
                    height: 1, width: 1, rowBytes: MemoryLayout<Float>.size)
                var dstBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dstPtr),
                    height: 1, width: 1, rowBytes: MemoryLayout<UInt16>.size)
                _ = vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
            }
        }
        return dst
    }

    static func copyLogitsFloat32(_ logits: MLMultiArray) -> [Float] {
        let count = logits.count
        var out = [Float](repeating: 0, count: count)
        switch logits.dataType {
        case .float32:
            let src = logits.dataPointer.bindMemory(to: Float.self, capacity: count)
            out.withUnsafeMutableBufferPointer { buf in
                _ = memcpy(buf.baseAddress, src, count * MemoryLayout<Float>.size)
            }
        case .float16:
            let src = logits.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            out.withUnsafeMutableBufferPointer { buf in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src),
                    height: 1, width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size)
                var dstBuf = vImage_Buffer(
                    data: buf.baseAddress, height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size)
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        default:
            for i in 0..<count { out[i] = logits[i].floatValue }
        }
        return out
    }

    static func applyRepetitionPenalty(
        _ logits: inout [Float], history: [Int], penalty: Float
    ) {
        guard penalty != 1.0, !history.isEmpty else { return }
        var seen = Set<Int>()
        seen.reserveCapacity(history.count)
        for t in history where t >= 0 && t < logits.count { seen.insert(t) }
        for t in seen {
            let v = logits[t]
            logits[t] = (v >= 0) ? v / penalty : v * penalty
        }
    }

    static func applyNoRepeatNgram(_ logits: inout [Float], history: [Int], n: Int) {
        guard n > 0, history.count >= n - 1 else { return }
        if n == 1 {
            for t in history where t >= 0 && t < logits.count { logits[t] = -1e9 }
            return
        }
        let prefix = Array(history.suffix(n - 1))
        var forbidden = Set<Int>()
        let upper = history.count - (n - 1)
        if upper <= 0 { return }
        for i in 0..<upper {
            var match = true
            for j in 0..<(n - 1) where history[i + j] != prefix[j] {
                match = false
                break
            }
            if match {
                let idx = i + (n - 1)
                if idx < history.count { forbidden.insert(history[idx]) }
            }
        }
        for t in forbidden where t >= 0 && t < logits.count { logits[t] = -1e9 }
    }

    static func argmax(_ logits: [Float]) -> Int {
        var maxIdx: vDSP_Length = 0
        var maxVal: Float = 0
        logits.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            vDSP_maxvi(base, 1, &maxVal, &maxIdx, vDSP_Length(logits.count))
        }
        return Int(maxIdx)
    }

    static func zeroFill(_ array: MLMultiArray) {
        let count = array.count
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            memset(ptr, 0, count * MemoryLayout<Float>.size)
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            memset(ptr, 0, count * MemoryLayout<UInt16>.size)
        case .int32:
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            memset(ptr, 0, count * MemoryLayout<Int32>.size)
        default:
            for i in 0..<count {
                array[i] = NSNumber(value: 0)
            }
        }
    }

    // MARK: Detokenize with SentencePiece byte-fallback

    static func convertTokensToText(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        guard !tokenIds.isEmpty else { return "" }
        var out = ""
        var byteBuf: [UInt8] = []

        @inline(__always) func flushBytes() {
            if !byteBuf.isEmpty {
                out.append(String(decoding: byteBuf, as: UTF8.self))
                byteBuf.removeAll(keepingCapacity: true)
            }
        }

        for tokenId in tokenIds {
            if tokenId <= 4 || tokenId == CohereAsrConfig.SpecialTokens.eosToken { continue }
            guard let piece = vocabulary[tokenId], !piece.isEmpty else { continue }
            if piece.hasPrefix("<|") { continue }
            if let b = parseByteFallback(piece) {
                byteBuf.append(b)
                continue
            }
            flushBytes()
            out.append(piece)
        }
        flushBytes()
        return out.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    static func parseByteFallback(_ piece: String) -> UInt8? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        let start = piece.index(piece.startIndex, offsetBy: 3)
        let end = piece.index(piece.endIndex, offsetBy: -1)
        return UInt8(piece[start..<end], radix: 16)
    }
}
