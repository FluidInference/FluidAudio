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
        /// Which cache-external decoder variant was loaded. Drives the
        /// `attention_mask` build at decode time without re-probing the
        /// model's input shape.
        public let decoderVariant: DecoderVariant
        /// KV-cache slots in the loaded decoder (`k_cache_0` shape `[1, H, slots, D]`).
        /// Shared by the language prompt and the generated tokens; see
        /// `resolveMaxNewTokens` for the resulting output budget.
        public let cacheLength: Int
        /// Element type the decoder was compiled with for its caches and masks.
        public let cacheDataType: MLMultiArrayDataType
        /// Compute units the encoder was loaded with.
        public let encoderComputeUnits: MLComputeUnits
        /// Compute units the decoder was loaded with.
        public let decoderComputeUnits: MLComputeUnits
    }

    private let melExtractor: CohereMelSpectrogram

    public init(mel: CohereMelSpectrogram = CohereMelSpectrogram()) {
        self.melExtractor = mel
    }

    /// Cache-external decoder variant selector.
    ///
    /// All variants share the same input/output contract except for the
    /// `attention_mask` shape, the KV-cache length and how position is
    /// sourced.
    public enum DecoderVariant: Sendable {
        /// `cohere_decoder_cache_external_v3.mlmodelc` — static shapes like
        /// v2 with a 256-slot KV cache. Default.
        case v3
        /// `cohere_decoder_cache_external_v2.mlmodelc` — fixed
        /// `attention_mask` shape `[1, 1, 1, 108]`, ANE-friendly.
        case v2
        /// `cohere_decoder_cache_external.mlmodelc` — dynamic
        /// `attention_mask` (`RangeDim(1, 108)`), CPU/GPU only.
        case v1

        public var compiledFileName: String {
            switch self {
            case .v3: return ModelNames.CohereTranscribe.decoderCacheExternalV3CompiledFile
            case .v2: return ModelNames.CohereTranscribe.decoderCacheExternalV2CompiledFile
            case .v1: return ModelNames.CohereTranscribe.decoderCacheExternalCompiledFile
            }
        }

        /// Whether the variant expects a full-length `[1, 1, 1, maxSeqLen]`
        /// additive causal `attention_mask`. The dynamic v1 decoder instead
        /// takes a `[1, 1, 1, step+1]` zero-filled mask.
        public var usesStaticSelfMask: Bool {
            switch self {
            case .v3, .v2: return true
            case .v1: return false
            }
        }
    }

    /// Compute units the encoder loads with unless the caller picks some.
    ///
    /// The INT8 encoder stays INT8 on the Neural Engine. Under `.all`,
    /// Core ML may route it to the GPU instead, where the 1.8 GB of INT8
    /// weights are materialised at full precision: the process keeps ~10 GB
    /// more memory resident after the first `transcribe` (12.4 GB footprint
    /// in #1045, M5 Pro, macOS 26.6) and that first call takes over a
    /// minute. On the ANE the footprint stays near 130 MB for the whole
    /// run, at the cost of a slower warm encoder (~1.2 s vs ~0.5 s per 35 s
    /// window on M5 Pro). Pass `encoderComputeUnits: .all` or `.cpuAndGPU`
    /// to opt into the GPU on machines that can afford the memory. See
    /// Documentation/ASR/Cohere.md, "Compute units and memory".
    public static let defaultEncoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    /// Compute units the decoder loads with unless the caller picks some.
    ///
    /// The FP16 cache-external decoder is 291 MB and ANE-resident with its
    /// static shapes. Moving it to `.all` changes its time by a few percent
    /// and adds ~100 MB, so it shares the encoder's default.
    public static let defaultDecoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    /// Load encoder, decoder and vocabulary from (potentially different) directories.
    ///
    /// This supports mixed-precision — e.g. INT8 encoder + FP16 decoder. Pass
    /// the same URL to both for a single-precision setup.
    ///
    /// `decoderVariant` selects which cache-external decoder file to load
    /// from `decoderDir`. The default (`.v3`) is the ANE-friendly
    /// static-shape build with a 256-slot cache; `.v2` is the 108-slot
    /// static build and `.v1` the legacy FP16 dynamic-shape decoder. The
    /// runtime reads the cache length from the loaded model and adapts the
    /// `attention_mask` build to whichever variant is loaded.
    ///
    /// Compute units are resolved per model: `encoderComputeUnits` /
    /// `decoderComputeUnits` win, then the shared `computeUnits`, then
    /// `defaultEncoderComputeUnits` / `defaultDecoderComputeUnits` (both
    /// `.cpuAndNeuralEngine`). Keep the encoder off the GPU unless you can
    /// spare ~10 GB; see `defaultEncoderComputeUnits`.
    public static func loadModels(
        encoderDir: URL,
        decoderDir: URL,
        vocabDir: URL,
        decoderVariant: DecoderVariant = .v3,
        computeUnits: MLComputeUnits? = nil,
        encoderComputeUnits: MLComputeUnits? = nil,
        decoderComputeUnits: MLComputeUnits? = nil
    ) async throws -> LoadedModels {
        let units = resolveComputeUnits(
            shared: computeUnits, encoder: encoderComputeUnits, decoder: decoderComputeUnits)
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = units.encoder
        let decoderConfig = MLModelConfiguration()
        decoderConfig.computeUnits = units.decoder

        let encoderURL = encoderDir.appendingPathComponent(
            ModelNames.CohereTranscribe.encoderCompiledFile)
        let decoderURL = decoderDir.appendingPathComponent(decoderVariant.compiledFileName)

        guard FileManager.default.fileExists(atPath: decoderURL.path) else {
            throw CohereAsrError.modelNotFound(
                "Decoder not found at \(decoderURL.path) (variant: \(decoderVariant)).")
        }

        pipelineLogger.info(
            "Loading encoder from \(encoderURL.path) (\(describe(units.encoder)))")
        let encoder = try await MLModel.load(contentsOf: encoderURL, configuration: encoderConfig)
        pipelineLogger.info(
            "Loading decoder from \(decoderURL.path) (\(describe(units.decoder)))")
        let decoder = try await MLModel.load(contentsOf: decoderURL, configuration: decoderConfig)

        let cacheLength = try Self.probeCacheLength(
            decoder: decoder, variant: decoderVariant, path: decoderURL.path)
        let cacheDataType =
            decoder.modelDescription.inputDescriptionsByName["k_cache_0"]?
            .multiArrayConstraint?.dataType ?? .float16
        pipelineLogger.info("Decoder \(decoderVariant): \(cacheLength)-slot KV cache")

        let vocab = try loadVocabulary(vocabDir: vocabDir)
        return LoadedModels(
            encoder: encoder,
            decoder: decoder,
            vocabulary: vocab,
            decoderVariant: decoderVariant,
            cacheLength: cacheLength,
            cacheDataType: cacheDataType,
            encoderComputeUnits: units.encoder,
            decoderComputeUnits: units.decoder)
    }

    /// Per-model compute units for `loadModels`: the per-model value wins,
    /// then the shared value, then the model's default.
    static func resolveComputeUnits(
        shared: MLComputeUnits?, encoder: MLComputeUnits?, decoder: MLComputeUnits?
    ) -> (encoder: MLComputeUnits, decoder: MLComputeUnits) {
        return (
            encoder: encoder ?? shared ?? defaultEncoderComputeUnits,
            decoder: decoder ?? shared ?? defaultDecoderComputeUnits
        )
    }

    /// Log label for a compute-unit choice.
    public static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpu-only"
        case .cpuAndGPU: return "cpu-gpu"
        case .cpuAndNeuralEngine: return "cpu-ane"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    /// Read the KV-cache length from the decoder's `k_cache_0` input shape
    /// (`[1, heads, slots, headDim]`) instead of assuming a constant, so the
    /// 108-slot and 256-slot exports load through the same path. Static
    /// variants must declare a matching `[1, 1, 1, slots]` `attention_mask`.
    static func probeCacheLength(
        decoder: MLModel, variant: DecoderVariant, path: String
    ) throws -> Int {
        let inputs = decoder.modelDescription.inputDescriptionsByName
        guard let kDesc = inputs["k_cache_0"], kDesc.type == .multiArray,
            let kConstraint = kDesc.multiArrayConstraint, kConstraint.shape.count == 4
        else {
            throw CohereAsrError.decodingFailed(
                "Decoder at \(path) is not cache-external (missing rank-4 k_cache_0).")
        }
        let slots = Self.maxExtent(of: kConstraint, dimension: 2)
        guard slots > 0 else {
            throw CohereAsrError.decodingFailed(
                "Decoder at \(path) declares an empty KV cache (k_cache_0 shape \(kConstraint.shape)).")
        }
        if variant.usesStaticSelfMask {
            guard let mDesc = inputs["attention_mask"], mDesc.type == .multiArray,
                let mConstraint = mDesc.multiArrayConstraint, !mConstraint.shape.isEmpty,
                Self.maxExtent(of: mConstraint, dimension: mConstraint.shape.count - 1) == slots
            else {
                throw CohereAsrError.decodingFailed(
                    "Decoder at \(path): static attention_mask length does not match the "
                        + "\(slots)-slot KV cache (variant \(variant)).")
            }
        }
        return slots
    }

    /// Largest size an input can take along `dimension`: the declared size for
    /// a fixed shape, the upper bound of a range, or the widest enumerated
    /// shape. The shipped exports are fixed-shape; the fallback keeps a
    /// flexible export from reporting its (small) default shape as capacity.
    static func maxExtent(of constraint: MLMultiArrayConstraint, dimension: Int) -> Int {
        return maxExtent(
            shape: constraint.shape,
            shapeConstraint: constraint.shapeConstraint,
            dimension: dimension)
    }

    static func maxExtent(
        shape: [NSNumber], shapeConstraint: MLMultiArrayShapeConstraint?, dimension: Int
    ) -> Int {
        guard dimension >= 0, dimension < shape.count else { return 0 }
        guard let shapeConstraint else { return shape[dimension].intValue }
        switch shapeConstraint.type {
        case .range:
            let ranges = shapeConstraint.sizeRangeForDimension
            guard dimension < ranges.count else { return shape[dimension].intValue }
            let range = ranges[dimension].rangeValue
            return range.location + range.length - 1
        case .enumerated:
            let widest = shapeConstraint.enumeratedShapes.compactMap { shape -> Int? in
                dimension < shape.count ? shape[dimension].intValue : nil
            }.max()
            return widest ?? shape[dimension].intValue
        case .unspecified:
            return shape[dimension].intValue
        @unknown default:
            return shape[dimension].intValue
        }
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
        /// Number of 35 s windows whose decode loop exhausted the token
        /// budget without emitting EOS. Non-zero means the tail of those
        /// windows was dropped; the budget is `maxNewTokens` or, by default,
        /// the decoder's cache length minus the prompt.
        public let cappedWindows: Int
    }

    /// Output-token budget for one window: the caller's `maxNewTokens` if
    /// given, otherwise everything the KV cache can hold. The decode loop
    /// runs at most `cacheLength` steps (one cache slot each); the last
    /// prompt step already yields the first output, so the prompt costs
    /// `promptCount - 1` steps and the capacity is `cacheLength - promptCount + 1`
    /// (99 with a 108-slot cache, 247 with 256). Either way the budget never
    /// exceeds the cache.
    static func resolveMaxNewTokens(
        requested: Int?, cacheLength: Int, promptCount: Int
    ) -> Int {
        let capacity = max(0, cacheLength - promptCount + 1)
        guard let requested = requested else { return capacity }
        return max(0, min(requested, capacity))
    }

    /// Decode steps needed to emit `maxNewTokens` outputs after a
    /// `promptCount`-token prompt, bounded by the cache. Outputs start at step
    /// `promptCount - 1`, so `steps - (promptCount - 1)` tokens come out.
    static func decodeStepCount(maxNewTokens: Int, promptCount: Int, cacheLength: Int) -> Int {
        return max(0, min(maxNewTokens + promptCount - 1, cacheLength))
    }

    public func transcribe(
        audio: [Float],
        models: LoadedModels,
        language: CohereAsrConfig.Language = .english,
        maxNewTokens: Int? = nil,
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
        let budget = Self.resolveMaxNewTokens(
            requested: maxNewTokens,
            cacheLength: models.cacheLength,
            promptCount: language.promptSequence.count)
        let decoded = try await decodeCacheExternal(
            encoderHidden: encoderHidden,
            encoderValidFrames: encoderValid,
            language: language,
            maxNewTokens: budget,
            repetitionPenalty: repetitionPenalty,
            noRepeatNgram: noRepeatNgram,
            decoder: models.decoder,
            cacheLength: models.cacheLength,
            cacheDType: models.cacheDataType,
            useStaticSelfMask: models.decoderVariant.usesStaticSelfMask)
        let decoderSecs = CFAbsoluteTimeGetCurrent() - dec0
        if decoded.hitCap {
            pipelineLogger.warning(
                "Window hit the \(budget)-token decode cap without EOS "
                    + "(\(models.cacheLength)-slot cache); trailing speech in this window was dropped.")
        }

        // 4) Detokenize
        let text = Self.convertTokensToText(decoded.tokens, vocabulary: models.vocabulary)

        return TranscriptionResult(
            text: text,
            tokenIds: decoded.tokens,
            encoderSeconds: encoderSecs,
            decoderSeconds: decoderSecs,
            totalSeconds: CFAbsoluteTimeGetCurrent() - total0,
            cappedWindows: decoded.hitCap ? 1 : 0)
    }

    // MARK: Long-form (sliding-window over the 35 s single-chunk encoder)

    /// Transcribe arbitrary-length audio by sliding the existing 35 s encoder
    /// window with a 5 s overlap and merging adjacent chunks via token-level
    /// longest-common-substring.
    ///
    /// Audio ≤ `CohereAsrConfig.maxAudioSeconds` short-circuits to the
    /// single-chunk `transcribe()` path and is byte-identical to it. Longer
    /// audio is split at `chunkHopSeconds = maxAudioSeconds − chunkOverlapSeconds`
    /// hops; chunk windows of `maxAudioSeconds` are decoded independently and
    /// stitched. Encoder/decoder/total seconds are summed across chunks.
    ///
    /// Mirrors the upstream Python pipeline's `overlap_chunk_second: 5`. No
    /// model changes — the encoder still sees a fixed `[1, 128, 3500]` mel and
    /// the decoder cache is reset per chunk.
    public func transcribeLong(
        audio: [Float],
        models: LoadedModels,
        language: CohereAsrConfig.Language = .english,
        maxNewTokens: Int? = nil,
        repetitionPenalty: Float = 1.1,
        noRepeatNgram: Int = 3
    ) async throws -> TranscriptionResult {
        let total0 = CFAbsoluteTimeGetCurrent()

        let sr = CohereAsrConfig.sampleRate
        let chunkSamples = Int(CohereAsrConfig.maxAudioSeconds) * sr
        let hopSamples = Int(CohereAsrConfig.chunkHopSeconds) * sr

        // Short audio: pass through unchanged.
        if audio.count <= chunkSamples {
            return try await transcribe(
                audio: audio,
                models: models,
                language: language,
                maxNewTokens: maxNewTokens,
                repetitionPenalty: repetitionPenalty,
                noRepeatNgram: noRepeatNgram)
        }

        var mergedTokens: [Int] = []
        var encoderSecs: Double = 0
        var decoderSecs: Double = 0
        var cappedWindows = 0
        var start = 0
        var chunkIndex = 0

        while start < audio.count {
            let end = min(start + chunkSamples, audio.count)
            let chunk = Array(audio[start..<end])

            // Don't bother decoding a final tail of pure overlap — the previous
            // chunk already covered it.
            if chunkIndex > 0, (end - start) <= (chunkSamples - hopSamples) {
                break
            }

            let result = try await transcribe(
                audio: chunk,
                models: models,
                language: language,
                maxNewTokens: maxNewTokens,
                repetitionPenalty: repetitionPenalty,
                noRepeatNgram: noRepeatNgram)

            encoderSecs += result.encoderSeconds
            decoderSecs += result.decoderSeconds
            cappedWindows += result.cappedWindows
            pipelineLogger.debug(
                "Window \(chunkIndex) [\(start / sr)s, \(end / sr)s): \(result.tokenIds.count) tokens, "
                    + "encoder \(String(format: "%.2f", result.encoderSeconds))s, "
                    + "decoder \(String(format: "%.2f", result.decoderSeconds))s"
                    + (result.cappedWindows > 0 ? ", hit cap" : ""))

            mergedTokens = Self.mergeTokenStreams(prefix: mergedTokens, suffix: result.tokenIds)

            chunkIndex += 1
            if end >= audio.count { break }
            start += hopSamples
        }

        let text = Self.convertTokensToText(mergedTokens, vocabulary: models.vocabulary)
        return TranscriptionResult(
            text: text,
            tokenIds: mergedTokens,
            encoderSeconds: encoderSecs,
            decoderSeconds: decoderSecs,
            totalSeconds: CFAbsoluteTimeGetCurrent() - total0,
            cappedWindows: cappedWindows)
    }

    /// Merge two adjacent chunk token streams using longest-common-substring.
    ///
    /// Both chunks transcribe ~5 s of identical audio at their seam, so their
    /// token IDs share a common run near the prefix's tail / the suffix's
    /// head. We search a bounded window (`windowTokens` tokens at the
    /// boundary) for the longest common substring of length ≥ `minMatch`. On
    /// a hit the seam is cut at the end of the matched run: the prefix is
    /// kept up to and including the run, the suffix from just after it. Both
    /// sides' copies of the overlap audio beyond the run are therefore
    /// dropped exactly once — keeping the whole prefix would emit the audio
    /// after the run twice (once per window). On a miss we concatenate
    /// plainly — better to duplicate a few tokens than to lose content.
    ///
    /// Ties between equally long runs go to the one closest to the prefix's
    /// end, so a phrase repeated inside the prefix window does not cut the
    /// seam early and drop the text between its two occurrences.
    ///
    /// For a genuine seam the prefix tokens after the run plus the suffix
    /// tokens up to the run's end together cover the overlap audio once, so
    /// their sum is about the overlap's token count (~30 for dense speech). A
    /// match whose cut would drop more than `windowTokens` tokens in total is
    /// a chance repeat of a common phrase, not the seam, and is ignored.
    static func mergeTokenStreams(
        prefix: [Int],
        suffix: [Int],
        windowTokens: Int = 48,
        minMatch: Int = 4
    ) -> [Int] {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }

        let pTail = Array(prefix.suffix(windowTokens))
        let sHead = Array(suffix.prefix(windowTokens))
        let m = pTail.count
        let n = sHead.count
        if m == 0 || n == 0 { return prefix + suffix }

        // Classic LCS-substring DP (O(m·n), m,n ≤ windowTokens).
        var dp = [Int](repeating: 0, count: n + 1)
        var bestLen = 0
        var bestPEnd = 0  // index in pTail (exclusive) where the match ends
        var bestSEnd = 0  // index in sHead (exclusive) where the match ends
        for i in 1...m {
            var prev = 0
            for j in 1...n {
                let temp = dp[j]
                if pTail[i - 1] == sHead[j - 1] {
                    dp[j] = prev + 1
                    // Ties go to the later prefix position (larger i); for the
                    // same prefix position the earliest suffix position wins.
                    if dp[j] > bestLen || (dp[j] == bestLen && i > bestPEnd) {
                        bestLen = dp[j]
                        bestPEnd = i
                        bestSEnd = j
                    }
                } else {
                    dp[j] = 0
                }
                prev = temp
            }
        }

        guard bestLen >= minMatch else { return prefix + suffix }
        let prefixDrop = m - bestPEnd
        guard prefixDrop + bestSEnd <= windowTokens else { return prefix + suffix }

        // Cut both streams at the end of the matched run.
        return Array(prefix.prefix(prefix.count - prefixDrop)) + Array(suffix.dropFirst(bestSEnd))
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

    private struct DecodeOutput {
        let tokens: [Int]
        /// The loop ran out of budget before the model emitted EOS.
        let hitCap: Bool
    }

    private func decodeCacheExternal(
        encoderHidden: MLMultiArray,
        encoderValidFrames: Int,
        language: CohereAsrConfig.Language,
        maxNewTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        decoder: MLModel,
        cacheLength: Int,
        cacheDType: MLMultiArrayDataType,
        useStaticSelfMask: Bool
    ) async throws -> DecodeOutput {
        let cacheShape: [NSNumber] = [
            1,
            NSNumber(value: CohereAsrConfig.numDecoderHeads),
            NSNumber(value: cacheLength),
            NSNumber(value: CohereAsrConfig.headDim),
        ]

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
        var sawEos = false
        let effectiveMax = Self.decodeStepCount(
            maxNewTokens: maxNewTokens, promptCount: prompt.count, cacheLength: cacheLength)

        for step in 0..<effectiveMax {
            if step < prompt.count { currentToken = prompt[step] }

            let inputId = try MLMultiArray(shape: [1, 1], dataType: .int32)
            inputId[0] = NSNumber(value: currentToken)

            let positionId = try MLMultiArray(shape: [1, 1], dataType: .int32)
            positionId[0] = NSNumber(value: step)

            let selfMask = try Self.buildSelfAttentionMask(
                step: step, length: cacheLength, useStatic: useStaticSelfMask, dtype: cacheDType)

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
                sawEos = true
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

        return DecodeOutput(tokens: outputTokens, hitCap: !sawEos)
    }

    // MARK: Helpers

    /// Build the self-attention additive mask for one decoding step.
    ///
    /// Dynamic path (RangeDim decoder): shape `[1, 1, 1, step+1]`, all zeros.
    /// The model's attended K/V slice is exactly `step+1` long.
    ///
    /// Static path (fixed-shape decoder): shape `[1, 1, 1, length]` where
    /// `length` is the decoder's cache length. The decoder attends over the
    /// full cache; positions `[0..=step]` get `0` (attend) and
    /// `[step+1..length-1]` get `-1e4` (masked — the cache slots aren't
    /// written yet).
    static func buildSelfAttentionMask(
        step: Int, length: Int, useStatic: Bool, dtype: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        if !useStatic {
            let mask = try MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: step + 1)], dataType: dtype)
            zeroFill(mask)
            return mask
        }

        let mask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: length)], dataType: dtype)
        zeroFill(mask)
        let firstBlocked = step + 1
        guard firstBlocked < length else { return mask }

        let invalidValue: Float = -1.0e4
        switch dtype {
        case .float32:
            let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
            for i in firstBlocked..<length { ptr[i] = invalidValue }
        case .float16:
            let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: length)
            let bits = float16Bits(invalidValue)
            for i in firstBlocked..<length { ptr[i] = bits }
        default:
            for i in firstBlocked..<length {
                mask[[0, 0, 0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: invalidValue)
            }
        }
        return mask
    }

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
