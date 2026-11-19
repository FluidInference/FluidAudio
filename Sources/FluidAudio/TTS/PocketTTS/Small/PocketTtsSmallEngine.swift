#if TTS
@preconcurrency import CoreML
import Foundation

/// The `MLModel` handles of a small-model pack, loaded once per manager and
/// shared by every engine (sessions each own an `MLState`). Which functions are
/// loaded depends on `mimiPrecision`: `.fp32` loads `prefill` + `generate` from
/// `pocket_state.mlmodelc` and the fp32 `mimi_decoder.mlmodelc` (CPU);
/// `.fp16Fused` loads `prefill` + `generate_decode` only.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
struct PocketTtsSmallModels: Sendable {
    let mimiPrecision: PocketTtsSmallMimiPrecision
    let prefill: MLModel
    /// FlowLM step + flow head (`generate`); `.fp32` mode only.
    let generate: MLModel?
    /// Fused step + head + fp16 Mimi (`generate_decode`); `.fp16Fused` mode only.
    let generateDecode: MLModel?
    /// fp32 I/O Mimi decoder on the CPU; `.fp32` mode only.
    let mimiDecoder: MLModel?
    /// The function whose declared state is the union a session needs.
    let stateSource: MLModel

    init(packDirectory: URL, computeUnits: MLComputeUnits, mimiPrecision: PocketTtsSmallMimiPrecision) throws {
        let package = packDirectory.appendingPathComponent(ModelNames.PocketTTS.pocketStateFile)
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw PocketTTSError.modelNotFound(
                "\(ModelNames.PocketTTS.pocketStateFile) missing in \(packDirectory.path)")
        }
        func config(_ function: String) -> MLModelConfiguration {
            let c = MLModelConfiguration()
            c.computeUnits = computeUnits
            c.functionName = function
            return c
        }
        self.mimiPrecision = mimiPrecision
        // Check every file this mode needs before paying the first ANE compile.
        let mimiURL = packDirectory.appendingPathComponent(ModelNames.PocketTTS.mimiDecoderFile)
        if mimiPrecision == .fp32, !FileManager.default.fileExists(atPath: mimiURL.path) {
            throw PocketTTSError.modelNotFound(
                "\(ModelNames.PocketTTS.mimiDecoderFile) missing in \(packDirectory.path) "
                    + "(needed for mimiPrecision .fp32; re-download the pack or use .fp16Fused)")
        }
        prefill = try MLModel(
            contentsOf: package, configuration: config(ModelNames.PocketTTS.StateFunction.prefill))
        switch mimiPrecision {
        case .fp16Fused:
            let fused = try MLModel(
                contentsOf: package, configuration: config(ModelNames.PocketTTS.StateFunction.generateDecode))
            generateDecode = fused
            generate = nil
            mimiDecoder = nil
            stateSource = fused
        case .fp32:
            let gen = try MLModel(
                contentsOf: package, configuration: config(ModelNames.PocketTTS.StateFunction.generate))
            let mimiConfig = MLModelConfiguration()
            mimiConfig.computeUnits = .cpuOnly
            generate = gen
            generateDecode = nil
            mimiDecoder = try MLModel(contentsOf: mimiURL, configuration: mimiConfig)
            stateSource = gen
        }
    }
}

/// Drives a small-model pack: one `MLState` per engine shared by `prefill` and
/// the per-frame function. One engine per session: the state is mutable and
/// every prediction writes it.
///
/// Per utterance: zero the Mimi state (fp16 KV buffers in fused mode, the fp32
/// I/O arrays otherwise) and the conv histories, reset the frame counter. Per
/// chunk: write the voice KV snapshot over the FlowLM buffers (zero-padded, so
/// it doubles as the cache reset), one `prefill` call for the text, then per
/// 80 ms frame either one `generate_decode` call (`.fp16Fused`: next latent,
/// EOS logit and audio together) or `generate` on the ANE followed by the fp32
/// `mimi_decoder` on the CPU (`.fp32`, the default: no fp16 noise floor).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
actor PocketTtsSmallEngine {

    /// A voice snapshot in the state's at-rest format (fp16 bit patterns,
    /// zero-padded to full `[1, L, Hkv, D]` buffers) — the 100M engine's type.
    typealias Fp16Snapshot = PocketTtsStateEngine.Fp16Snapshot

    struct Frame: Sendable {
        let latent: [Float]
        let eosLogit: Float
        let samples: [Float]
    }

    enum EngineError: Error, LocalizedError {
        case missingOutput(String)
        case stateBufferMismatch(name: String, expected: Int, actual: Int)
        case unsupportedStateLayout(name: String)
        case fusedFunctionMissing
        case perFrameModelMissing(PocketTtsSmallMimiPrecision)
        case unexpectedInput(name: String)
        case outputShapeMismatch(name: String, expected: Int, actual: Int)
        case phaseOutOfRange(Int, phases: Int)

        var errorDescription: String? {
            switch self {
            case .missingOutput(let name): return "PocketTTS small: model output '\(name)' missing"
            case .stateBufferMismatch(let name, let expected, let actual):
                return "PocketTTS small: state buffer '\(name)' has \(actual) elements, expected \(expected)"
            case .unsupportedStateLayout(let name): return "PocketTTS small: state buffer '\(name)' is not fp16"
            case .fusedFunctionMissing:
                return "PocketTTS small: pocket_state.mlmodelc has no generate_decode function"
            case .perFrameModelMissing(let precision):
                return "PocketTTS small: the per-frame models for mimiPrecision .\(precision) are not loaded"
            case .unexpectedInput(let name):
                return "PocketTTS small: mimi_decoder input '\(name)' is not a multi-array"
            case .outputShapeMismatch(let name, let expected, let actual):
                return "PocketTTS small: output '\(name)' has \(actual) elements, expected \(expected)"
            case .phaseOutOfRange(let p, let phases): return "PocketTTS small: Mimi phase \(p) outside [0, \(phases))"
            }
        }
    }

    let constants: PocketTtsSmallConstants
    private let models: PocketTtsSmallModels
    private let mlState: MLState
    /// Next FlowLM KV write slot.
    private(set) var position: Float = 0
    /// Mimi frames decoded since the last `beginUtterance()`.
    private(set) var mimiFrame: Int = 0
    /// Mimi state that travels as inputs/outputs and is fed back each call:
    /// the conv histories / overlap-add partials in fused mode, plus the Mimi
    /// KV caches in `.fp32` mode (the I/O model has no MLState).
    private var convState: [String: MLMultiArray] = [:]
    private let convSpec: [String: [Int]]
    private let convNames: [String]

    var mimiPrecision: PocketTtsSmallMimiPrecision { models.mimiPrecision }

    var arch: PocketTtsSmallArch { constants.arch }
    var kvBufferElementCount: Int {
        constants.state.kvCacheLen * arch.numKvHeads * arch.headDim
    }

    init(models: PocketTtsSmallModels, constants: PocketTtsSmallConstants) throws {
        self.constants = constants
        self.models = models
        switch models.mimiPrecision {
        case .fp16Fused:
            guard models.generateDecode != nil else { throw EngineError.perFrameModelMissing(.fp16Fused) }
            convSpec = constants.state.mimiConvIO
        case .fp32:
            // Every input of the fp32 I/O model except the latent and the phase is
            // state to feed back; shapes come from the compiled model, so the host
            // needs none of the converter's JSON for it. Checked once here so a
            // pack/arch mismatch fails at load, not as an out-of-bounds read.
            guard let mimi = models.mimiDecoder, models.generate != nil else {
                throw EngineError.perFrameModelMissing(.fp32)
            }
            let inputs = mimi.modelDescription.inputDescriptionsByName
            let outputs = mimi.modelDescription.outputDescriptionsByName
            func count(_ desc: MLFeatureDescription?, _ name: String) throws -> Int {
                guard let shape = desc?.multiArrayConstraint?.shape else {
                    throw EngineError.unexpectedInput(name: name)
                }
                return shape.map(\.intValue).reduce(1, *)
            }
            for (name, expected) in [("latent", constants.arch.ldim), ("phase", constants.state.mimiPhases)] {
                let actual = try count(inputs[name], name)
                guard actual == expected else {
                    throw EngineError.outputShapeMismatch(name: name, expected: expected, actual: actual)
                }
            }
            let audioCount = try count(outputs["audio"], "audio")
            guard audioCount == constants.arch.frameSize else {
                throw EngineError.outputShapeMismatch(
                    name: "audio", expected: constants.arch.frameSize, actual: audioCount)
            }
            var spec: [String: [Int]] = [:]
            for (name, desc) in inputs where name != "latent" && name != "phase" {
                guard let shape = desc.multiArrayConstraint?.shape else {
                    throw EngineError.unexpectedInput(name: name)
                }
                let n = shape.map(\.intValue).reduce(1, *)
                let outCount = try count(outputs["\(name)_out"], "\(name)_out")
                guard outCount == n else {
                    throw EngineError.outputShapeMismatch(name: "\(name)_out", expected: n, actual: outCount)
                }
                spec[name] = shape.map(\.intValue)
            }
            convSpec = spec
        }
        // The state `prefill` and the frame function share (FlowLM buffers, plus
        // the Mimi KV buffers in fused mode).
        self.mlState = models.stateSource.makeState()
        self.convNames = convSpec.keys.sorted()
        self.convState = try Self.makeZeroConvState(convSpec)
    }

    // MARK: - Utterance / chunk lifecycle

    /// Fresh Mimi decoder: zero its KV buffers and conv state, frame counter 0.
    func beginUtterance() throws {
        if models.mimiPrecision == .fp16Fused {
            for (name, shape) in constants.state.mimiState {
                try writeStateBuffer([UInt16](repeating: 0, count: shape.reduce(1, *)), name: name)
            }
        }
        try zeroConvState()
        mimiFrame = 0
    }

    /// Inject a voice snapshot over the whole FlowLM KV state (= chunk reset).
    func reset(with snapshot: Fp16Snapshot) throws {
        for i in 0..<arch.numLayers {
            try writeStateBuffer(snapshot.kBuffers[i], name: "k_cache\(i)")
            try writeStateBuffer(snapshot.vBuffers[i], name: "v_cache\(i)")
        }
        position = snapshot.position
    }

    func resetToZero() throws {
        let zeros = [UInt16](repeating: 0, count: kvBufferElementCount)
        for i in 0..<arch.numLayers {
            try writeStateBuffer(zeros, name: "k_cache\(i)")
            try writeStateBuffer(zeros, name: "v_cache\(i)")
        }
        position = 0
    }

    func captureSnapshot() throws -> Fp16Snapshot {
        var k: [[UInt16]] = []
        var v: [[UInt16]] = []
        for i in 0..<arch.numLayers {
            k.append(try readStateBuffer(name: "k_cache\(i)"))
            v.append(try readStateBuffer(name: "v_cache\(i)"))
        }
        return Fp16Snapshot(kBuffers: k, vBuffers: v, position: position)
    }

    // MARK: - Prefill

    /// Write `tokenCount` pre-embedded conditioning rows (`[tokens, dModel]`
    /// flat) into the FlowLM state, in windows of the compiled block length.
    func prefill(flatConditioning: [Float], tokenCount: Int) async throws {
        guard tokenCount > 0 else { return }
        let dim = arch.dModel
        let tMax = constants.state.condPrefillTMax
        guard Int(position) + tokenCount <= constants.state.kvCacheLen else {
            throw PocketTTSError.processingFailed(
                "PocketTTS small: conditioning (\(Int(position)) + \(tokenCount)) exceeds the "
                    + "\(constants.state.kvCacheLen)-slot KV cache")
        }
        var processed = 0
        while processed < tokenCount {
            let n = min(tMax, tokenCount - processed)
            let conditioning = try MLMultiArray(
                shape: [1, NSNumber(value: tMax), NSNumber(value: dim)], dataType: .float32)
            let ptr = conditioning.dataPointer.bindMemory(to: Float.self, capacity: tMax * dim)
            ptr.initialize(repeating: 0, count: tMax * dim)
            flatConditioning.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                ptr.update(from: base.advanced(by: processed * dim), count: n * dim)
            }
            let validLen = try MLMultiArray(shape: [1], dataType: .float32)
            validLen[0] = NSNumber(value: Float(n))
            let pos = try MLMultiArray(shape: [1], dataType: .float32)
            pos[0] = NSNumber(value: position)
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "conditioning": conditioning, "valid_len": validLen, "position": pos,
            ])
            let output = try await models.prefill.prediction(
                from: input, using: mlState, options: MLPredictionOptions())
            if let newPosition = output.featureValue(for: "new_position")?.multiArrayValue {
                position = newPosition[0].floatValue
            } else {
                position += Float(n)
            }
            processed += n
        }
    }

    // MARK: - Frame

    /// One frame: FlowLM step + flow head + Mimi decode. `sequence` is the
    /// previous latent (the BOS latent on the first step); `noise` is z_0
    /// already scaled by sqrt(temperature). One dispatch in `.fp16Fused` mode,
    /// `generate` (ANE) + `mimi_decoder` (CPU, fp32) in `.fp32` mode.
    func generateDecodeFrame(sequence: [Float], noise: [Float]) async throws -> Frame {
        guard Int(position) + 1 <= constants.state.kvCacheLen else {
            throw PocketTTSError.processingFailed(
                "PocketTTS small: KV cache exhausted at position \(Int(position))")
        }
        let ldim = arch.ldim
        let seq = try MLMultiArray(shape: [1, 1, NSNumber(value: ldim)], dataType: .float32)
        let latentInit = try MLMultiArray(shape: [1, NSNumber(value: ldim)], dataType: .float32)
        for i in 0..<ldim {
            seq[i] = NSNumber(value: i < sequence.count ? sequence[i] : 0)
            latentInit[i] = NSNumber(value: i < noise.count ? noise[i] : 0)
        }
        let pos = try MLMultiArray(shape: [1], dataType: .float32)
        pos[0] = NSNumber(value: position)
        let phases = constants.state.mimiPhases
        let phaseIndex = constants.state.mimiPhase(frame: mimiFrame)
        guard phaseIndex >= 0, phaseIndex < phases else {
            throw EngineError.phaseOutOfRange(phaseIndex, phases: phases)
        }
        let phase = try MLMultiArray(shape: [1, NSNumber(value: phases)], dataType: .float32)
        let phasePtr = phase.dataPointer.bindMemory(to: Float.self, capacity: phases)
        phasePtr.initialize(repeating: 0, count: phases)
        phasePtr[phaseIndex] = 1

        let latentValues: [Float]
        let eos: MLMultiArray
        let mimiOutput: MLFeatureProvider
        if let fused = models.generateDecode {
            var features: [String: MLFeatureValue] = [
                "sequence": MLFeatureValue(multiArray: seq),
                "latent_init": MLFeatureValue(multiArray: latentInit),
                "position": MLFeatureValue(multiArray: pos),
                "phase": MLFeatureValue(multiArray: phase),
            ]
            for (name, array) in convState {
                features[name] = MLFeatureValue(multiArray: array)
            }
            let output = try await fused.prediction(
                from: try MLDictionaryFeatureProvider(dictionary: features),
                using: mlState, options: MLPredictionOptions())
            guard let l = output.featureValue(for: "latent_final")?.multiArrayValue else {
                throw EngineError.missingOutput("latent_final")
            }
            guard let e = output.featureValue(for: "is_eos")?.multiArrayValue else {
                throw EngineError.missingOutput("is_eos")
            }
            latentValues = try Self.readFloats(l, count: ldim, name: "latent_final")
            eos = e
            mimiOutput = output
        } else {
            guard let generate = models.generate, let mimi = models.mimiDecoder else {
                throw EngineError.perFrameModelMissing(.fp32)
            }
            let output = try await generate.prediction(
                from: try MLDictionaryFeatureProvider(dictionary: [
                    "sequence": seq, "latent_init": latentInit, "position": pos,
                ]),
                using: mlState, options: MLPredictionOptions())
            guard let l = output.featureValue(for: "latent_final")?.multiArrayValue else {
                throw EngineError.missingOutput("latent_final")
            }
            guard let e = output.featureValue(for: "is_eos")?.multiArrayValue else {
                throw EngineError.missingOutput("is_eos")
            }
            // The I/O model wants a contiguous fp32 [1, ldim]; copy rather than
            // trust the function output's layout and dtype.
            let latentIn = try MLMultiArray(shape: [1, NSNumber(value: ldim)], dataType: .float32)
            let values = try Self.readFloats(l, count: ldim, name: "latent_final")
            for i in 0..<ldim { latentIn[i] = NSNumber(value: values[i]) }
            var features: [String: MLFeatureValue] = [
                "latent": MLFeatureValue(multiArray: latentIn),
                "phase": MLFeatureValue(multiArray: phase),
            ]
            for (name, array) in convState {
                features[name] = MLFeatureValue(multiArray: array)
            }
            latentValues = values
            eos = e
            mimiOutput = try await mimi.prediction(from: try MLDictionaryFeatureProvider(dictionary: features))
        }

        guard let audio = mimiOutput.featureValue(for: "audio")?.multiArrayValue else {
            throw EngineError.missingOutput("audio")
        }
        for name in convNames {
            guard let updated = mimiOutput.featureValue(for: "\(name)_out")?.multiArrayValue else {
                throw EngineError.missingOutput("\(name)_out")
            }
            convState[name] = updated
        }
        position += 1
        mimiFrame += 1
        return Frame(
            latent: latentValues,
            eosLogit: eos[0].floatValue,
            samples: try Self.readFloats(audio, count: arch.frameSize, name: "audio"))
    }

    // MARK: - Snapshot conversion

    /// A shipped voice as fp16 state buffers, zero-padded to the full cache.
    func fp16Snapshot(from voice: PocketTtsSmallVoice) throws -> Fp16Snapshot {
        guard voice.kLayers.count == arch.numLayers else {
            throw PocketTTSError.processingFailed(
                "voice \(voice.name) has \(voice.kLayers.count) layers, model has \(arch.numLayers)")
        }
        guard voice.promptLength <= constants.state.kvCacheLen else {
            throw PocketTTSError.processingFailed(
                "voice \(voice.name) prompt length \(voice.promptLength) exceeds the KV cache")
        }
        let total = kvBufferElementCount
        return Fp16Snapshot(
            kBuffers: voice.kLayers.map { Self.fp16Buffer(from: $0, totalCount: total) },
            vBuffers: voice.vLayers.map { Self.fp16Buffer(from: $0, totalCount: total) },
            position: Float(voice.promptLength))
    }

    /// fp32 -> fp16 (IEEE round-to-nearest-even via vImage), zero-padded to
    /// `totalCount`; the 100M engine's converter with this pack's buffer size.
    static func fp16Buffer(from source: [Float], totalCount: Int) -> [UInt16] {
        PocketTtsStateEngine.fp16Buffer(
            from: source, offset: 0, count: min(source.count, totalCount), totalCount: totalCount)
    }

    // MARK: - State buffer access

    private func zeroConvState() throws {
        convState = try Self.makeZeroConvState(convSpec)
    }

    private static func makeZeroConvState(_ spec: [String: [Int]]) throws -> [String: MLMultiArray] {
        var out: [String: MLMultiArray] = [:]
        for (name, shape) in spec {
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            let n = shape.reduce(1, *)
            array.dataPointer.bindMemory(to: Float.self, capacity: n).initialize(repeating: 0, count: n)
            out[name] = array
        }
        return out
    }

    private func writeStateBuffer(_ bits: [UInt16], name: String) throws {
        try mlState.withMultiArray(for: name) { array in
            guard array.dataType == .float16 else { throw EngineError.unsupportedStateLayout(name: name) }
            guard array.count == bits.count else {
                throw EngineError.stateBufferMismatch(name: name, expected: bits.count, actual: array.count)
            }
            array.withUnsafeMutableBytes { raw, strides in
                guard let base = raw.baseAddress else { return }
                let dst = base.assumingMemoryBound(to: UInt16.self)
                precondition(
                    PocketTtsStateEngine.isContiguous(shape: array.shape.map(\.intValue), strides: strides),
                    "state buffer \(name) is not contiguous")
                bits.withUnsafeBufferPointer { src in
                    guard let s = src.baseAddress else { return }
                    dst.update(from: s, count: bits.count)
                }
            }
        }
    }

    private func readStateBuffer(name: String) throws -> [UInt16] {
        try mlState.withMultiArray(for: name) { array in
            guard array.dataType == .float16 else { throw EngineError.unsupportedStateLayout(name: name) }
            var bits = [UInt16](repeating: 0, count: array.count)
            array.withUnsafeMutableBytes { raw, strides in
                guard let base = raw.baseAddress else { return }
                precondition(
                    PocketTtsStateEngine.isContiguous(shape: array.shape.map(\.intValue), strides: strides),
                    "state buffer \(name) is not contiguous")
                let src = base.assumingMemoryBound(to: UInt16.self)
                bits.withUnsafeMutableBufferPointer { dst in
                    guard let d = dst.baseAddress else { return }
                    d.update(from: src, count: array.count)
                }
            }
            return bits
        }
    }

    /// First `count` elements of a model output as Float; bounds-checked, with
    /// the raw-pointer fast path only for float32.
    private static func readFloats(_ array: MLMultiArray, count: Int, name: String) throws -> [Float] {
        guard array.count >= count else {
            throw EngineError.outputShapeMismatch(name: name, expected: count, actual: array.count)
        }
        guard array.dataType == .float32 else {
            return (0..<count).map { array[$0].floatValue }
        }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
#endif
