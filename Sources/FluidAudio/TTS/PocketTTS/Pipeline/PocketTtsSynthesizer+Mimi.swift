@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Mutable streaming state for the Mimi neural audio codec decoder.
    ///
    /// Contains 26 tensors that track convolutional history, attention caches,
    /// and partial upsampling buffers. Unlike the KV cache (which resets per
    /// text chunk), Mimi state persists across all chunks to produce seamless
    /// audio — the decoder needs prior frame context for smooth waveform continuity.
    struct MimiState {
        var tensors: [String: MLMultiArray]
    }

    /// Create the initial Mimi decoder state by zero-initialization driven
    /// purely from the discovered model schema.
    ///
    /// Mimi's streaming state is "all-zero past" with two semantic exceptions:
    /// the `*_first` boolean scalars must be 1.0 on the very first frame so
    /// the decoder takes the cold-start convolution path. All other state
    /// tensors (caches, partials, offsets) are zero-initialized.
    ///
    /// This replaces the previous manifest-driven `.bin` loader. The shapes
    /// come from the CoreML model description itself, so v1 and v2 packs
    /// (which differ in mimi attention cache layout) work uniformly.
    static func loadMimiInitialState(schema: PocketTtsMimiSchema) throws -> MimiState {
        var tensors: [String: MLMultiArray] = [:]
        let dtype = schema.stateInputDataType
        let elementSize = mimiElementSize(for: dtype)

        for (inputName, _) in schema.stateMapping {
            guard let shape = schema.stateInputShapes[inputName], !shape.isEmpty else {
                throw PocketTTSError.processingFailed(
                    "Mimi state input `\(inputName)` has no shape in schema")
            }

            let totalCount = shape.reduce(1, *)
            let nsShape = shape.map { NSNumber(value: $0) }
            let array: MLMultiArray
            if totalCount == 0 {
                // CoreML's MLE5 input binder rejects buffers with NULL data
                // pointers (which is what MLMultiArray returns for zero-element
                // shapes like `[1, 128, 0]`). Allocate a 1-byte sentinel buffer
                // and hand it to MLMultiArray via the dataPointer initializer
                // so the model gets a valid non-NULL pointer.
                let sentinel = UnsafeMutableRawPointer.allocate(
                    byteCount: max(elementSize, 1), alignment: 64)
                memset(sentinel, 0, max(elementSize, 1))
                let strides = mimiContiguousStrides(for: shape).map { NSNumber(value: $0) }
                array = try MLMultiArray(
                    dataPointer: sentinel,
                    shape: nsShape,
                    dataType: dtype,
                    strides: strides,
                    deallocator: { ptr in ptr.deallocate() })
            } else {
                array = try MLMultiArray(shape: nsShape, dataType: dtype)
                memset(array.dataPointer, 0, totalCount * elementSize)
                // `*_first` scalars signal the first-frame cold-start path.
                if inputName.hasSuffix("_first") {
                    array[0] = NSNumber(value: Float(1))
                }
            }

            tensors[inputName] = array
        }

        return MimiState(tensors: tensors)
    }

    /// Byte size of one element for the given MLMultiArray dtype.
    private static func mimiElementSize(for dtype: MLMultiArrayDataType) -> Int {
        switch dtype {
        case .float16: return MemoryLayout<UInt16>.size
        case .float32: return MemoryLayout<Float>.size
        case .double: return MemoryLayout<Double>.size
        case .int32: return MemoryLayout<Int32>.size
        @unknown default: return MemoryLayout<Float>.size
        }
    }

    /// Row-major (C-contiguous) strides for the given shape, in elements.
    private static func mimiContiguousStrides(for shape: [Int]) -> [Int] {
        var strides = Array(repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * max(shape[i + 1], 1)
        }
        return strides
    }

    /// Clone a Mimi state for independent use.
    static func cloneMimiState(_ state: MimiState) throws -> MimiState {
        var newTensors: [String: MLMultiArray] = [:]
        for (key, array) in state.tensors {
            let copy = try MLMultiArray(shape: array.shape, dataType: array.dataType)
            let byteSize: Int
            switch array.dataType {
            case .float16:
                byteSize = array.count * MemoryLayout<UInt16>.size
            default:
                byteSize = array.count * MemoryLayout<Float>.size
            }
            if byteSize > 0 {
                copy.dataPointer.copyMemory(from: array.dataPointer, byteCount: byteSize)
            }
            newTensors[key] = copy
        }
        return MimiState(tensors: newTensors)
    }

    /// Run the Mimi decoder for a single latent frame.
    ///
    /// The model internally denormalizes and quantizes the 32-dim latent
    /// before decoding to audio.
    ///
    /// - Parameters:
    ///   - latent: The raw latent vector, shape [32].
    ///   - state: The streaming state (24 or 26 tensors depending on
    ///     conversion vintage), modified in place.
    ///   - model: The Mimi CoreML model.
    ///   - schema: I/O schema discovered at model-load time (audio output
    ///     name + state input ↔ output mapping).
    /// - Returns: Audio samples for this frame (1920 samples = 80ms at 24kHz).
    static func runMimiDecoder(
        latent: [Float],
        state: inout MimiState,
        model: MLModel,
        schema: PocketTtsMimiSchema
    ) async throws -> [Float] {
        // Create latent input: [1, 32] at the dtype the model expects.
        let latentDim = PocketTtsConstants.latentDim
        let latentArray = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim)], dataType: schema.latentDataType)
        switch schema.latentDataType {
        case .float32:
            let dst = latentArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
            latent.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                dst.update(from: base, count: latentDim)
            }
        case .float16:
            let dst = latentArray.dataPointer.bindMemory(to: Float16.self, capacity: latentDim)
            for i in 0..<latentDim { dst[i] = Float16(latent[i]) }
        default:
            for i in 0..<latentDim {
                latentArray[i] = NSNumber(value: latent[i])
            }
        }

        // Build input dictionary
        var inputDict: [String: Any] = ["latent": latentArray]
        for (key, array) in state.tensors {
            inputDict[key] = array
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract audio output [1, 1, 1920]
        guard let audioArray = output.featureValue(for: schema.audioOutputName)?.multiArrayValue
        else {
            throw PocketTTSError.processingFailed(
                "Missing Mimi audio output: \(schema.audioOutputName)")
        }

        let sampleCount = PocketTtsConstants.samplesPerFrame
        let samples = readFloatArray(from: audioArray, count: sampleCount)

        // Update streaming state
        for (inputName, outputName) in schema.stateMapping {
            guard let updated = output.featureValue(for: outputName)?.multiArrayValue else {
                throw PocketTTSError.processingFailed(
                    "Missing Mimi state output: \(outputName) (for \(inputName))")
            }
            state.tensors[inputName] = updated
        }

        return samples
    }

    /// Read Float values from an MLMultiArray, handling both float32 and float16 data types.
    ///
    /// The Mimi decoder CoreML model outputs float16 tensors. Using `dataPointer` with
    /// `Float.self` binding on float16 data produces garbage values. This method
    /// uses the type-safe subscript accessor which handles conversion automatically.
    private static func readFloatArray(from array: MLMultiArray, count: Int) -> [Float] {
        if array.dataType == .float16 {
            // Use subscript for correct float16 → float32 conversion
            return (0..<count).map { array[$0].floatValue }
        }
        // Fast path for float32: direct memory access
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
