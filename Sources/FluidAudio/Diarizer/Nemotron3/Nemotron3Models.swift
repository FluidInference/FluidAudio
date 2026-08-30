@preconcurrency import CoreML
import Accelerate
import Foundation

/// Container for the Nemotron 3 Diarization CoreML model and the learned silence embedding.
///
/// Loaded from a local directory only (NVIDIA evaluation license — the converted models are
/// not distributed). The directory must contain the variant's `.mlmodelc` (or `.mlpackage`)
/// and `learnable_sil_emb.bin` (512 float32, little-endian).
public struct Nemotron3Models {
    public let model: MLModel
    /// Learned silence embedding, [512]. Used for disabled frames during cache compression.
    public let silenceEmbedding: [Float]
    /// FeatureStacking projection, [1024, 512] row-major (W^T). Split-graph mode only.
    public let preEncodeProjection: [Float]?
    public let compilationDuration: TimeInterval

    private let chunkArray: MLMultiArray
    private let chunkLengthArray: MLMultiArray
    private let spkcacheArray: MLMultiArray
    private let spkcacheLengthArray: MLMultiArray
    private let fifoArray: MLMultiArray
    private let fifoLengthArray: MLMultiArray
    private let packedArray: MLMultiArray?
    private let attnBiasArray: MLMultiArray?
    private let outputMaskArray: MLMultiArray?
    private let memoryOptimizer: ANEMemoryOptimizer

    private static let logger = AppLogger(category: "Nemotron3Models")

    public init(
        config: Nemotron3Config,
        model: MLModel,
        silenceEmbedding: [Float],
        preEncodeProjection: [Float]? = nil,
        compilationDuration: TimeInterval = 0
    ) throws {
        self.model = model
        self.silenceEmbedding = silenceEmbedding
        self.preEncodeProjection = preEncodeProjection
        self.compilationDuration = compilationDuration

        self.memoryOptimizer = .init()
        self.chunkArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.chunkMelFrames), NSNumber(value: config.melFeatures)],
            dataType: .float32)
        self.spkcacheArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.spkcacheLen), NSNumber(value: config.preEncoderDims)],
            dataType: .float32)
        self.fifoArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.fifoLen), NSNumber(value: config.preEncoderDims)],
            dataType: .float32)
        self.chunkLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.spkcacheLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.fifoLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        if config.splitGraph {
            let t = config.packedFrames
            // packed's innermost dim (512) is tile-aligned, so the aligned array is
            // contiguous. The mask inputs are NOT: the aligned helper pads innermost
            // dims to tile boundaries (output_mask [1,T,1] would get row stride 16,
            // scrambling linear writes) — use plain contiguous MLMultiArrays for them.
            self.packedArray = try memoryOptimizer.createAlignedArray(
                shape: [1, NSNumber(value: t), NSNumber(value: config.preEncoderDims)],
                dataType: .float32)
            self.attnBiasArray = try MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: t)], dataType: .float32)
            self.outputMaskArray = try MLMultiArray(
                shape: [1, NSNumber(value: t), 1], dataType: .float32)
        } else {
            self.packedArray = nil
            self.attnBiasArray = nil
            self.outputMaskArray = nil
        }
    }

    /// Load from a local models directory.
    public static func load(
        config: Nemotron3Config,
        directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> Nemotron3Models {
        let start = Date()

        var modelURL = directory.appendingPathComponent(config.modelFileName)
        if !FileManager.default.fileExists(atPath: modelURL.path) {
            // Fall back to the uncompiled mlpackage next to the expected mlmodelc.
            let packageURL = directory.appendingPathComponent(
                config.modelFileName.replacingOccurrences(of: ".mlmodelc", with: ".mlpackage"))
            guard FileManager.default.fileExists(atPath: packageURL.path) else {
                throw Nemotron3Error.modelLoadFailed(
                    "Neither \(config.modelFileName) nor its .mlpackage found in \(directory.path)")
            }
            modelURL = try await MLModel.compileModel(at: packageURL)
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

        let silURL = directory.appendingPathComponent("learnable_sil_emb.bin")
        guard let silData = try? Data(contentsOf: silURL) else {
            throw Nemotron3Error.modelLoadFailed("Missing learnable_sil_emb.bin in \(directory.path)")
        }
        let silCount = silData.count / MemoryLayout<Float>.size
        guard silCount == config.preEncoderDims else {
            throw Nemotron3Error.modelLoadFailed(
                "learnable_sil_emb.bin has \(silCount) floats, expected \(config.preEncoderDims)")
        }
        let silenceEmbedding = silData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        var projection: [Float]? = nil
        if config.splitGraph {
            let projURL = directory.appendingPathComponent("pre_encode_proj_t.bin")
            guard let projData = try? Data(contentsOf: projURL),
                projData.count == 1024 * 512 * MemoryLayout<Float>.size
            else {
                throw Nemotron3Error.modelLoadFailed(
                    "Split-graph mode requires pre_encode_proj_t.bin ([1024,512] fp32) in \(directory.path)")
            }
            projection = projData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        let duration = Date().timeIntervalSince(start)
        logger.info("Loaded Nemotron 3 diarization model in \(String(format: "%.2f", duration))s")
        return try Nemotron3Models(
            config: config, model: model, silenceEmbedding: silenceEmbedding,
            preEncodeProjection: projection, compilationDuration: duration)
    }

    // MARK: - Split-graph inference

    /// Run one streaming step through the split graph: host does feature stacking, the
    /// 1024->512 projection, state packing, and mask construction; the model is the pure
    /// transformer+head. Returns the same `Output` contract (chunk embeddings host-computed).
    public func runSplit(
        chunk: [Float],
        chunkLength: Int,
        state: Nemotron3StreamingState,
        config: Nemotron3Config
    ) throws -> Output {
        guard let packedArray, let attnBiasArray, let outputMaskArray,
            let projection = preEncodeProjection
        else {
            throw Nemotron3Error.invalidState("runSplit called on a non-split configuration")
        }
        let d = config.preEncoderDims
        let sub = config.subsamplingFactor
        let t = config.packedFrames

        var tStage = Date()
        // Feature stacking is a pure reshape of the zero-padded fixed-size mel buffer:
        // [mel, 128] row-major == [mel/8, 1024]. Project with one sgemm.
        let encCapacity = config.chunkMelFrames / sub
        var chunkEmbs = [Float](repeating: 0, count: encCapacity * d)
        chunk.withUnsafeBufferPointer { src in
            chunkEmbs.withUnsafeMutableBufferPointer { dst in
                projection.withUnsafeBufferPointer { proj in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(encCapacity), Int32(d), Int32(1024),
                        1.0, src.baseAddress, Int32(1024),
                        proj.baseAddress, Int32(d),
                        0.0, dst.baseAddress, Int32(d))
                }
            }
        }
        let encLen = (chunkLength + sub - 1) / sub

        // Pack [spkcache | fifo | chunk] valid frames, zero-pad, build masks.
        let packedPtr = packedArray.dataPointer.bindMemory(to: Float.self, capacity: t * d)
        var pos = 0
        for (buffer, n) in [
            (state.spkcache, state.spkcacheLength), (state.fifo, state.fifoLength),
            (chunkEmbs, encLen),
        ] {
            buffer.withUnsafeBufferPointer { src in
                packedPtr.advanced(by: pos * d).update(from: src.baseAddress!, count: n * d)
            }
            pos += n
        }
        if pos < t {
            packedPtr.advanced(by: pos * d).update(repeating: 0, count: (t - pos) * d)
        }
        let biasPtr = attnBiasArray.dataPointer.bindMemory(to: Float.self, capacity: t)
        let maskPtr = outputMaskArray.dataPointer.bindMemory(to: Float.self, capacity: t)
        biasPtr.update(repeating: 0, count: pos)
        biasPtr.advanced(by: pos).update(repeating: -30000.0, count: t - pos)
        maskPtr.update(repeating: 1, count: pos)
        maskPtr.advanced(by: pos).update(repeating: 0, count: t - pos)

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "packed": MLFeatureValue(multiArray: packedArray),
            "attn_bias": MLFeatureValue(multiArray: attnBiasArray),
            "output_mask": MLFeatureValue(multiArray: outputMaskArray),
        ])
        let inputPrepSeconds = Date().timeIntervalSince(tStage)

        tStage = Date()
        let output = try model.prediction(from: inputs)
        let predictSeconds = Date().timeIntervalSince(tStage)
        tStage = Date()

        guard let predsArray = output.featureValue(for: "speaker_preds")?.multiArrayValue,
            let hiresArray = output.featureValue(for: "speaker_preds_10ms")?.multiArrayValue
        else {
            throw Nemotron3Error.inferenceFailed("Missing split model outputs")
        }
        return Output(
            predictions: Self.floats(from: predsArray),
            highResPredictions: Self.floats(from: hiresArray),
            chunkEmbeddings: chunkEmbs,
            chunkLength: encLen,
            inputPrepSeconds: inputPrepSeconds,
            predictSeconds: predictSeconds,
            readbackSeconds: Date().timeIntervalSince(tStage)
        )
    }

    // MARK: - Inference

    public struct Output {
        /// 80 ms packed predictions [spkcacheLen + fifoLen + chunkEncFrames, 8] flattened.
        public let predictions: [Float]
        /// 10 ms packed predictions [(spkcacheLen + fifoLen + chunkEncFrames) * 8, 8] flattened.
        public let highResPredictions: [Float]
        /// Chunk pre-encode embeddings [chunkEncFrames, 512] flattened.
        public let chunkEmbeddings: [Float]
        /// Valid encoder frames in `chunkEmbeddings`.
        public let chunkLength: Int
        /// Per-call wall time split: input tensor copies, CoreML predict, output readback.
        public let inputPrepSeconds: Double
        public let predictSeconds: Double
        public let readbackSeconds: Double
    }

    /// Run one streaming step.
    ///
    /// - Parameters:
    ///   - chunk: Mel features [chunkMelFrames * 128] flattened (zero-padded to capacity).
    ///   - chunkLength: Valid mel frames.
    ///   - state: Current streaming state (read-only here).
    public func run(
        chunk: [Float],
        chunkLength: Int,
        state: Nemotron3StreamingState,
        config: Nemotron3Config
    ) throws -> Output {
        var tStage = Date()
        memoryOptimizer.optimizedCopy(from: chunk, to: chunkArray, pad: true)
        memoryOptimizer.optimizedCopy(from: state.spkcache, to: spkcacheArray, pad: true)
        memoryOptimizer.optimizedCopy(from: state.fifo, to: fifoArray, pad: true)
        chunkLengthArray[0] = NSNumber(value: Int32(chunkLength))
        spkcacheLengthArray[0] = NSNumber(value: Int32(state.spkcacheLength))
        fifoLengthArray[0] = NSNumber(value: Int32(state.fifoLength))

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunkArray),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLengthArray),
            "spkcache": MLFeatureValue(multiArray: spkcacheArray),
            "spkcache_lengths": MLFeatureValue(multiArray: spkcacheLengthArray),
            "fifo": MLFeatureValue(multiArray: fifoArray),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLengthArray),
        ])
        let inputPrepSeconds = Date().timeIntervalSince(tStage)

        tStage = Date()
        let output = try model.prediction(from: inputs)
        let predictSeconds = Date().timeIntervalSince(tStage)
        tStage = Date()

        guard let predsArray = output.featureValue(for: "speaker_preds")?.multiArrayValue,
            let hiresArray = output.featureValue(for: "speaker_preds_10ms")?.multiArrayValue,
            let embsArray = output.featureValue(for: "chunk_pre_encode_embs")?.multiArrayValue
        else {
            throw Nemotron3Error.inferenceFailed("Missing model outputs")
        }
        let preds = Self.floats(from: predsArray)
        let hires = Self.floats(from: hiresArray)
        let embs = Self.floats(from: embsArray)
        // Advisory only: on GPU-scheduled graphs its fp16 floor_div can be off by one for
        // large offline chunks, so derive the valid length host-side instead.
        let chunkEncLength = (chunkLength + config.subsamplingFactor - 1) / config.subsamplingFactor

        return Output(
            predictions: preds,
            highResPredictions: hires,
            chunkEmbeddings: embs,
            chunkLength: chunkEncLength,
            inputPrepSeconds: inputPrepSeconds,
            predictSeconds: predictSeconds,
            readbackSeconds: Date().timeIntervalSince(tStage)
        )
    }

    /// Direct-pointer MLMultiArray -> [Float] copy, honoring strides (reading a strided
    /// array through the contiguous fast path scrambles element order — FluidAudio #612).
    ///
    /// The model's outputs are fp16 with padded rows (e.g. shape [1, T, 8] with row
    /// stride 16), so this does one bulk fp16->fp32 conversion over the padded extent
    /// followed by a single `vDSP_mmov` 2D compaction, instead of per-element NSNumber
    /// reads or `shapedArrayValue` (~1.6 ms/chunk at fast32).
    static func floats(from array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let count = shape.reduce(1, *)

        // Collapse leading singleton dims to a rows x cols view with unit column stride.
        // All model outputs are [1, R, C]; also handle fully contiguous arrays as one row.
        var rows = 1
        var cols = count
        var rowStride = count
        if let last = strides.last, last == 1 {
            if shape.count >= 2, shape.dropLast(2).allSatisfy({ $0 == 1 }) {
                rows = shape[shape.count - 2]
                cols = shape[shape.count - 1]
                rowStride = strides[strides.count - 2]
            } else if strides == (0..<shape.count).map({ d in shape[(d + 1)...].reduce(1, *) }) {
                rows = 1
                cols = count
                rowStride = count
            } else {
                return floatsViaSubscript(array, count: count)
            }
        } else {
            return floatsViaSubscript(array, count: count)
        }

        let paddedCount = (rows - 1) * rowStride + cols
        var result = [Float](repeating: 0, count: count)

        switch array.dataType {
        case .float32:
            let src = array.dataPointer.bindMemory(to: Float.self, capacity: paddedCount)
            result.withUnsafeMutableBufferPointer { dst in
                vDSP_mmov(
                    src, dst.baseAddress!, vDSP_Length(cols), vDSP_Length(rows),
                    vDSP_Length(rowStride), vDSP_Length(cols))
            }
        #if arch(arm64)
        case .float16:
            let src = array.dataPointer.bindMemory(to: Float16.self, capacity: paddedCount)
            var scratch = [Float](repeating: 0, count: paddedCount)
            let srcBuffer = UnsafeBufferPointer(start: src, count: paddedCount)
            scratch.withUnsafeMutableBufferPointer { dst in
                vDSP.convertElements(of: srcBuffer, to: &dst)
            }
            scratch.withUnsafeBufferPointer { s in
                result.withUnsafeMutableBufferPointer { dst in
                    vDSP_mmov(
                        s.baseAddress!, dst.baseAddress!, vDSP_Length(cols), vDSP_Length(rows),
                        vDSP_Length(rowStride), vDSP_Length(cols))
                }
            }
        #endif
        default:
            return floatsViaSubscript(array, count: count)
        }
        return result
    }

    private static func floatsViaSubscript(_ array: MLMultiArray, count: Int) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = array[i].floatValue
        }
        return result
    }
}
