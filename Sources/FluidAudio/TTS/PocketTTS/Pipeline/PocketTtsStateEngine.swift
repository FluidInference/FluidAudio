import Accelerate
@preconcurrency import CoreML
import Foundation

/// The `MLModel` handles loaded from the ONE compiled `pocket_state.mlmodelc`
/// multifunction package (mobius Trial 23), one instance per CoreML function
/// (`MLModelConfiguration.functionName`).
///
/// The package also ships a third `write_state` function (12 fp16 inputs that
/// overwrite the whole state) as the contract-explicit voice-injection
/// fallback. The host does NOT load it: Trial 23 measured the host-side
/// `MLState.withMultiArray(for:)` write as both bit-exact and faster
/// (1.56 ms vs 2.18 ms for the function call), so `PocketTtsStateEngine`
/// writes the state directly.
struct PocketTtsStateModels: Sendable {
    /// `prefill` function: `conditioning` `[1, 256, 1024]` + `valid_len` +
    /// `position` → `new_position`, writing the KV state in place.
    let prefill: MLModel
    /// `generate` function: `sequence` `[1, 1, 32]` + `latent_init` `[1, 32]`
    /// + `position` → `latent_final` + `is_eos`, fusing flowlm_step and the
    /// 8-step LSD flow decoder into one dispatch over the shared KV state.
    let generate: MLModel
}

/// Drives the Trial 23 MLState pipeline: one shared 12-buffer fp16 KV state
/// (`k_cache0..5` / `v_cache0..5`, each `[1, 512, 16, 64]`) that stays
/// resident across `prefill` and `generate` calls, replacing the 24-tensor
/// cache I/O of the `.gpu`/`.ane` placements.
///
/// An actor: the `MLState` is mutable shared state that every prediction
/// writes, so access must be serialized. One engine per session; the state
/// is reset per chunk by re-injecting the fp16 voice snapshot (which
/// overwrites every slot, so no `makeState()` per utterance is needed).
@available(macOS 15.0, iOS 18.0, *)
actor PocketTtsStateEngine {

    /// A voice KV snapshot converted to the state's at-rest format: fp16 bit
    /// patterns, zero-padded to the full `[1, 512, 16, 64]` buffer so that
    /// injecting it doubles as the utterance reset.
    struct Fp16Snapshot: Sendable {
        /// Per-layer K buffers, each `kvCacheMaxLen * 16 * 64` fp16 values.
        let kBuffers: [[UInt16]]
        /// Per-layer V buffers, same element count as `kBuffers`.
        let vBuffers: [[UInt16]]
        /// KV position right after the snapshot (number of valid slots).
        let position: Float
    }

    enum StateError: Error {
        case stateBufferMismatch(name: String, expected: Int, actual: Int)
        case unsupportedStateLayout(name: String)
        case missingOutput(String)
    }

    private let models: PocketTtsStateModels
    private let mlState: MLState
    private let layers: Int
    /// Next KV write slot. Advanced by `prefill` (via the model's
    /// `new_position` output) and by 1 per `generateFrame` call.
    private(set) var position: Float = 0

    /// Elements per state buffer: `1 * kvCacheMaxLen * 16 * 64`.
    private static var bufferElementCount: Int {
        PocketTtsConstants.kvCacheMaxLen * 16 * 64
    }

    init(models: PocketTtsStateModels, layers: Int) {
        self.models = models
        self.layers = layers
        // Trial 23: `makeState()` from ANY function instance of the
        // multifunction package is accepted by the others.
        self.mlState = models.generate.makeState()
    }

    // MARK: - Voice injection / reset

    /// Overwrite the ENTIRE state with the fp16 voice snapshot. Because the
    /// snapshot buffers are full-length (zero-padded), one call is both the
    /// per-chunk cache reset and the voice injection.
    func reset(with snapshot: Fp16Snapshot) throws {
        for i in 0..<layers {
            try writeStateBuffer(snapshot.kBuffers[i], name: "k_cache\(i)")
            try writeStateBuffer(snapshot.vBuffers[i], name: "v_cache\(i)")
        }
        position = snapshot.position
    }

    /// Zero the whole state (cloned-voice prefill starts from scratch).
    func resetToZero() throws {
        let zeros = [UInt16](repeating: 0, count: Self.bufferElementCount)
        for i in 0..<layers {
            try writeStateBuffer(zeros, name: "k_cache\(i)")
            try writeStateBuffer(zeros, name: "v_cache\(i)")
        }
        position = 0
    }

    /// Read the current state back into an `Fp16Snapshot`. Used to capture a
    /// cloned voice's prefilled KV once so later chunks re-inject it instead
    /// of re-running the ~126-token voice prefill.
    func captureSnapshot() throws -> Fp16Snapshot {
        var kBuffers: [[UInt16]] = []
        var vBuffers: [[UInt16]] = []
        kBuffers.reserveCapacity(layers)
        vBuffers.reserveCapacity(layers)
        for i in 0..<layers {
            kBuffers.append(try readStateBuffer(name: "k_cache\(i)"))
            vBuffers.append(try readStateBuffer(name: "v_cache\(i)"))
        }
        return Fp16Snapshot(kBuffers: kBuffers, vBuffers: vBuffers, position: position)
    }

    // MARK: - Prefill

    /// One-shot conditioning prefill over the shared state. Mirrors
    /// `runCondPrefill`'s ≤ T_max windowing (the KV position carries across
    /// windows inside the state), but with a single shared `position` scalar
    /// instead of 6 per-layer ones and NO cache tensors marshalled.
    func prefill(flatConditioning: [Float], tokenCount: Int) async throws {
        guard tokenCount > 0 else { return }
        let dim = PocketTtsConstants.embeddingDim
        let tMax = PocketTtsConstants.condPrefillMaxTokens

        var processed = 0
        while processed < tokenCount {
            let n = min(tMax, tokenCount - processed)

            let conditioning = try MLMultiArray(
                shape: [1, NSNumber(value: tMax), NSNumber(value: dim)], dataType: .float32)
            let condPtr = conditioning.dataPointer.bindMemory(
                to: Float.self, capacity: tMax * dim)
            condPtr.initialize(repeating: 0, count: tMax * dim)
            let srcOffset = processed * dim
            let copyCount = min(n * dim, max(0, flatConditioning.count - srcOffset))
            if copyCount > 0 {
                flatConditioning.withUnsafeBufferPointer { buffer in
                    guard let base = buffer.baseAddress else { return }
                    condPtr.update(from: base.advanced(by: srcOffset), count: copyCount)
                }
            }

            let validLen = try MLMultiArray(shape: [1], dataType: .float32)
            validLen[0] = NSNumber(value: Float(n))
            let positionArray = try MLMultiArray(shape: [1], dataType: .float32)
            positionArray[0] = NSNumber(value: position)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "conditioning": conditioning,
                "valid_len": validLen,
                "position": positionArray,
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

    // MARK: - Generation

    /// One fused generation step: flowlm_step + flow decode in a single
    /// dispatch. `sequence` is the previous frame's latent (or the BOS
    /// latent on the first step — the fused graph has no NaN-BOS protocol).
    /// `noise` is the host-provided z_0, already scaled by sqrt(temperature)
    /// from the seeded RNG so `--seed` reproducibility holds. The package's
    /// `transformer_out` debug output is ignored.
    func generateFrame(
        sequence: [Float], noise: [Float]
    ) async throws -> (latent: [Float], eosLogit: Float) {
        let latentDim = PocketTtsConstants.latentDim

        let sequenceArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: latentDim)], dataType: .float32)
        let sequencePtr = sequenceArray.dataPointer.bindMemory(
            to: Float.self, capacity: latentDim)
        sequence.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            sequencePtr.update(from: base, count: min(sequence.count, latentDim))
        }

        let latentInit = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim)], dataType: .float32)
        let latentPtr = latentInit.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        noise.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            latentPtr.update(from: base, count: min(noise.count, latentDim))
        }

        let positionArray = try MLMultiArray(shape: [1], dataType: .float32)
        positionArray[0] = NSNumber(value: position)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "sequence": sequenceArray,
            "latent_init": latentInit,
            "position": positionArray,
        ])
        let output = try await models.generate.prediction(
            from: input, using: mlState, options: MLPredictionOptions())

        guard let latentArray = output.featureValue(for: "latent_final")?.multiArrayValue else {
            throw StateError.missingOutput("latent_final")
        }
        guard let eosArray = output.featureValue(for: "is_eos")?.multiArrayValue else {
            throw StateError.missingOutput("is_eos")
        }

        position += 1
        return (
            latent: Self.readFloats(latentArray, count: latentDim),
            eosLogit: eosArray[0].floatValue
        )
    }

    // MARK: - Snapshot conversion

    /// Convert a pre-baked v2 voice snapshot (fp32, K-then-V per layer) to
    /// the state's fp16 at-rest format, zero-padded to full buffers.
    ///
    /// fp16 rounding here (vImage, IEEE round-to-nearest-even) matches the
    /// in-state representation exactly, so re-injection is a pure bit copy —
    /// the same property Trial 23 proved for the python host-side write.
    static func fp16Snapshot(
        from snapshot: PocketTtsVoiceCacheSnapshot, layers: Int
    ) throws -> Fp16Snapshot {
        guard snapshot.layers.count == layers else {
            throw PocketTTSError.processingFailed(
                "voice snapshot layer count \(snapshot.layers.count) != model layer count \(layers)"
            )
        }
        let srcSeq = snapshot.cacheSeqLen
        guard srcSeq <= PocketTtsConstants.kvCacheMaxLen else {
            throw PocketTTSError.processingFailed(
                "voice snapshot seqLen \(srcSeq) exceeds capacity \(PocketTtsConstants.kvCacheMaxLen)"
            )
        }
        let perKV = srcSeq * 16 * 64

        var kBuffers: [[UInt16]] = []
        var vBuffers: [[UInt16]] = []
        kBuffers.reserveCapacity(layers)
        vBuffers.reserveCapacity(layers)
        for layerIdx in 0..<layers {
            let source = snapshot.layers[layerIdx]
            guard source.cache.count == 2 * perKV else {
                throw PocketTTSError.processingFailed(
                    "voice snapshot layer \(layerIdx) has \(source.cache.count) floats, expected \(2 * perKV)"
                )
            }
            kBuffers.append(fp16Buffer(from: source.cache, offset: 0, count: perKV))
            vBuffers.append(fp16Buffer(from: source.cache, offset: perKV, count: perKV))
        }
        return Fp16Snapshot(
            kBuffers: kBuffers,
            vBuffers: vBuffers,
            position: Float(snapshot.layers[0].offset)
        )
    }

    /// Convert `count` floats starting at `offset` into a full-length,
    /// zero-padded fp16 state buffer (IEEE round-to-nearest-even via vImage).
    static func fp16Buffer(from source: [Float], offset: Int, count: Int) -> [UInt16] {
        var dst = [UInt16](repeating: 0, count: bufferElementCount)
        guard count > 0 else { return dst }
        source.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            dst.withUnsafeMutableBufferPointer { dstBuf in
                guard let dstBase = dstBuf.baseAddress else { return }
                var srcVImage = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcBase.advanced(by: offset)),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                var dstVImage = vImage_Buffer(
                    data: dstBase,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size
                )
                vImageConvert_PlanarFtoPlanar16F(
                    &srcVImage, &dstVImage, vImage_Flags(kvImageNoFlags))
            }
        }
        return dst
    }

    // MARK: - State buffer access

    /// Write fp16 bit patterns over an entire state buffer via the mutable
    /// `MLState.withMultiArray(for:)` accessor (read-write on iOS 18+).
    private func writeStateBuffer(_ bits: [UInt16], name: String) throws {
        try mlState.withMultiArray(for: name) { array in
            try Self.validateBuffer(array, name: name, expectedCount: bits.count)
            array.withUnsafeMutableBytes { rawBuffer, strides in
                guard let base = rawBuffer.baseAddress else { return }
                let dst = base.assumingMemoryBound(to: UInt16.self)
                if Self.isContiguous(shape: array.shape.map(\.intValue), strides: strides) {
                    bits.withUnsafeBufferPointer { src in
                        guard let srcBase = src.baseAddress else { return }
                        dst.update(from: srcBase, count: bits.count)
                    }
                } else {
                    Self.stridedVisit(
                        shape: array.shape.map(\.intValue), strides: strides
                    ) { flatIndex, storageIndex in
                        dst[storageIndex] = bits[flatIndex]
                    }
                }
            }
        }
    }

    /// Read an entire state buffer's fp16 bit patterns.
    private func readStateBuffer(name: String) throws -> [UInt16] {
        try mlState.withMultiArray(for: name) { array in
            let count = array.count
            try Self.validateBuffer(array, name: name, expectedCount: count)
            var bits = [UInt16](repeating: 0, count: count)
            array.withUnsafeMutableBytes { rawBuffer, strides in
                guard let base = rawBuffer.baseAddress else { return }
                let src = base.assumingMemoryBound(to: UInt16.self)
                if Self.isContiguous(shape: array.shape.map(\.intValue), strides: strides) {
                    bits.withUnsafeMutableBufferPointer { dst in
                        guard let dstBase = dst.baseAddress else { return }
                        dstBase.update(from: src, count: count)
                    }
                } else {
                    Self.stridedVisit(
                        shape: array.shape.map(\.intValue), strides: strides
                    ) { flatIndex, storageIndex in
                        bits[flatIndex] = src[storageIndex]
                    }
                }
            }
            return bits
        }
    }

    private static func validateBuffer(
        _ array: MLMultiArray, name: String, expectedCount: Int
    ) throws {
        guard array.dataType == .float16 else {
            throw StateError.unsupportedStateLayout(name: name)
        }
        guard array.count == expectedCount else {
            throw StateError.stateBufferMismatch(
                name: name, expected: expectedCount, actual: array.count)
        }
    }

    static func isContiguous(shape: [Int], strides: [Int]) -> Bool {
        guard shape.count == strides.count else { return false }
        var expected = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[axis] != expected { return false }
            expected *= shape[axis]
        }
        return true
    }

    /// Visit every element of a rank-4 buffer in flat row-major order,
    /// yielding the matching strided storage index. Fallback path for
    /// non-contiguous state buffers (not observed in practice).
    private static func stridedVisit(
        shape: [Int], strides: [Int], _ visit: (_ flatIndex: Int, _ storageIndex: Int) -> Void
    ) {
        precondition(shape.count == 4 && strides.count == 4, "state buffers are rank 4")
        var flat = 0
        for a in 0..<shape[0] {
            for b in 0..<shape[1] {
                for c in 0..<shape[2] {
                    let rowBase = a * strides[0] + b * strides[1] + c * strides[2]
                    for d in 0..<shape[3] {
                        visit(flat, rowBase + d * strides[3])
                        flat += 1
                    }
                }
            }
        }
    }

    /// Read `count` floats from a (fp32 or fp16) output array.
    private static func readFloats(_ array: MLMultiArray, count: Int) -> [Float] {
        if array.dataType == .float16 {
            return (0..<count).map { array[$0].floatValue }
        }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
