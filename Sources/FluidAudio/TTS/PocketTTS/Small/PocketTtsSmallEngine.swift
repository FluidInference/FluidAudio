#if TTS
@preconcurrency import CoreML
import Foundation

/// The two `MLModel` handles of a small-model `pocket_state.mlmodelc`, loaded
/// once per manager and shared by every engine (sessions each own an `MLState`).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
struct PocketTtsSmallModels: Sendable {
    let prefill: MLModel
    let generateDecode: MLModel

    init(packageURL: URL, computeUnits: MLComputeUnits) throws {
        func config(_ function: String) -> MLModelConfiguration {
            let c = MLModelConfiguration()
            c.computeUnits = computeUnits
            c.functionName = function
            return c
        }
        prefill = try MLModel(
            contentsOf: packageURL, configuration: config(ModelNames.PocketTTS.StateFunction.prefill))
        generateDecode = try MLModel(
            contentsOf: packageURL, configuration: config(ModelNames.PocketTTS.StateFunction.generateDecode))
    }
}

/// Drives a small-model `pocket_state.mlmodelc`: one `MLState` (made from
/// `generate_decode`, the union of the FlowLM and Mimi buffers) shared by
/// `prefill` and `generate_decode`. One engine per session: the state is
/// mutable and every prediction writes it.
///
/// Per utterance: zero the Mimi buffers and conv state, reset the frame
/// counter. Per chunk: write the voice KV snapshot over the FlowLM buffers
/// (zero-padded, so it doubles as the cache reset), one `prefill` call for the
/// text, then one `generate_decode` call per 80 ms frame which returns the
/// next latent, the EOS logit and the decoded audio together.
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
        case phaseOutOfRange(Int, phases: Int)

        var errorDescription: String? {
            switch self {
            case .missingOutput(let name): return "PocketTTS small: model output '\(name)' missing"
            case .stateBufferMismatch(let name, let expected, let actual):
                return "PocketTTS small: state buffer '\(name)' has \(actual) elements, expected \(expected)"
            case .unsupportedStateLayout(let name): return "PocketTTS small: state buffer '\(name)' is not fp16"
            case .fusedFunctionMissing:
                return "PocketTTS small: pocket_state.mlmodelc has no generate_decode function"
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
    /// Mimi conv histories / overlap-add partials, fed back each call.
    private var convState: [String: MLMultiArray] = [:]
    private let convNames: [String]

    var arch: PocketTtsSmallArch { constants.arch }
    var kvBufferElementCount: Int {
        constants.state.kvCacheLen * arch.numKvHeads * arch.headDim
    }

    init(models: PocketTtsSmallModels, constants: PocketTtsSmallConstants) throws {
        guard constants.state.hasFusedGenerateDecode else {
            throw EngineError.fusedFunctionMissing
        }
        self.constants = constants
        self.models = models
        // The union state (FlowLM + Mimi buffers); `prefill` accepts it too.
        self.mlState = models.generateDecode.makeState()
        self.convNames = constants.state.mimiConvIO.keys.sorted()
        self.convState = try Self.makeZeroConvState(constants.state.mimiConvIO)
    }

    // MARK: - Utterance / chunk lifecycle

    /// Fresh Mimi decoder: zero its KV buffers and conv state, frame counter 0.
    func beginUtterance() throws {
        for (name, shape) in constants.state.mimiState {
            try writeStateBuffer([UInt16](repeating: 0, count: shape.reduce(1, *)), name: name)
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

    // MARK: - Fused frame

    /// One frame: FlowLM step + flow head + Mimi decode in a single dispatch.
    /// `sequence` is the previous latent (the BOS latent on the first step);
    /// `noise` is z_0 already scaled by sqrt(temperature).
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

        var features: [String: MLFeatureValue] = [
            "sequence": MLFeatureValue(multiArray: seq),
            "latent_init": MLFeatureValue(multiArray: latentInit),
            "position": MLFeatureValue(multiArray: pos),
            "phase": MLFeatureValue(multiArray: phase),
        ]
        for (name, array) in convState {
            features[name] = MLFeatureValue(multiArray: array)
        }
        let output = try await models.generateDecode.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: features),
            using: mlState, options: MLPredictionOptions())

        guard let latent = output.featureValue(for: "latent_final")?.multiArrayValue else {
            throw EngineError.missingOutput("latent_final")
        }
        guard let eos = output.featureValue(for: "is_eos")?.multiArrayValue else {
            throw EngineError.missingOutput("is_eos")
        }
        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw EngineError.missingOutput("audio")
        }
        for name in convNames {
            guard let updated = output.featureValue(for: "\(name)_out")?.multiArrayValue else {
                throw EngineError.missingOutput("\(name)_out")
            }
            convState[name] = updated
        }
        position += 1
        mimiFrame += 1
        return Frame(
            latent: Self.readFloats(latent, count: ldim),
            eosLogit: eos[0].floatValue,
            samples: Self.readFloats(audio, count: arch.frameSize))
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
        convState = try Self.makeZeroConvState(constants.state.mimiConvIO)
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

    private static func readFloats(_ array: MLMultiArray, count: Int) -> [Float] {
        if array.dataType == .float16 {
            return (0..<count).map { array[$0].floatValue }
        }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
#endif
