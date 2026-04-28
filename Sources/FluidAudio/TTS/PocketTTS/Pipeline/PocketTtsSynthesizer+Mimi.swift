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

    /// Create the initial Mimi decoder state from the constants directory.
    ///
    /// Tensor shapes come from the loaded model's input descriptions, not a
    /// manifest. `.bin` files must be Float32 with element count matching
    /// the model's declared shape; missing files mean a zero-initialized
    /// tensor.
    static func loadMimiInitialState(
        from repoDirectory: URL,
        mimiKeys: PocketTtsMimiKeys
    ) throws -> MimiState {
        let constantsDir = repoDirectory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let stateDir = constantsDir.appendingPathComponent("mimi_init_state")

        var tensors: [String: MLMultiArray] = [:]

        for (name, shapeInts) in mimiKeys.stateShapes {
            let shape = shapeInts.map { NSNumber(value: $0) }
            let array = try MLMultiArray(shape: shape, dataType: .float32)

            // Zero-length tensors (e.g. `res*_conv1_prev: [1, 128, 0]`) are
            // empty pass-throughs — nothing to load.
            if shapeInts.contains(0) {
                tensors[name] = array
                continue
            }

            let elementCount = shapeInts.reduce(1, *)
            let dstPtr = array.dataPointer.bindMemory(to: Float.self, capacity: elementCount)
            // Default to zero in case the .bin file is absent (offset scalars,
            // attention caches in zero-init packs, etc.).
            dstPtr.initialize(repeating: 0, count: elementCount)

            let binURL = stateDir.appendingPathComponent("\(name).bin")
            if FileManager.default.fileExists(atPath: binURL.path) {
                let data = try Data(contentsOf: binURL)
                let expectedBytes = elementCount * MemoryLayout<Float>.size
                if data.count == expectedBytes {
                    data.withUnsafeBytes { rawBuffer in
                        let srcPtr = rawBuffer.bindMemory(to: Float.self)
                        dstPtr.update(from: srcPtr.baseAddress!, count: elementCount)
                    }
                }
                // Mismatched bin sizes fall back to zero-init, which is
                // the correct empty-cache initial value anyway.
            }

            tensors[name] = array
        }

        return MimiState(tensors: tensors)
    }

    /// Clone a Mimi state for independent use.
    static func cloneMimiState(_ state: MimiState) throws -> MimiState {
        var newTensors: [String: MLMultiArray] = [:]
        newTensors.reserveCapacity(state.tensors.count)
        for (key, array) in state.tensors {
            newTensors[key] = try deepCopy(array)
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
    ///   - state: The streaming state (26 tensors), modified in place.
    ///   - model: The Mimi CoreML model.
    /// - Returns: Audio samples for this frame (1920 samples = 80ms at 24kHz).
    static func runMimiDecoder(
        latent: [Float],
        state: inout MimiState,
        model: MLModel,
        mimiKeys: PocketTtsMimiKeys
    ) async throws -> [Float] {
        // Create latent input: [1, 32]
        let latentDim = PocketTtsConstants.latentDim
        let latentArray = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim)], dataType: .float32)
        let latentPtr = latentArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        latent.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            latentPtr.update(from: base, count: latentDim)
        }

        // Build input dictionary from discovered state mapping.
        var inputDict: [String: Any] = ["latent": latentArray]
        for (inputName, _) in mimiKeys.stateMapping {
            guard let array = state.tensors[inputName] else {
                throw PocketTTSError.processingFailed(
                    "Mimi state missing tensor '\(inputName)'")
            }
            inputDict[inputName] = array
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract audio output [1, 1, 1920]
        guard let audioArray = output.featureValue(for: mimiKeys.audioOutput)?.multiArrayValue else {
            throw PocketTTSError.processingFailed("Missing Mimi audio output")
        }

        let sampleCount = PocketTtsConstants.samplesPerFrame
        let samples = readFloatArray(from: audioArray, count: sampleCount)

        // Update streaming state
        for (inputName, outputName) in mimiKeys.stateMapping {
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
