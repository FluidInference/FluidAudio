@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Mutable streaming state for the Mimi audio decoder.
    ///
    /// Contains 26 tensors that track convolutional history,
    /// attention caches, and partial upsampling buffers.
    struct MimiState {
        var tensors: [String: MLMultiArray]
    }

    /// Create the initial Mimi decoder state from the constants directory.
    ///
    /// Loads pre-computed initial state tensors from `.bin` files,
    /// using `manifest.json` for shape metadata.
    static func loadMimiInitialState(from repoDirectory: URL) throws -> MimiState {
        let constantsDir = repoDirectory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let stateDir = constantsDir.appendingPathComponent("mimi_init_state")
        let manifestURL = constantsDir.appendingPathComponent("manifest.json")

        // Parse manifest for mimi_init_state shapes
        let manifestData = try Data(contentsOf: manifestURL)
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
            let mimiManifest = manifest["mimi_init_state"] as? [String: Any]
        else {
            throw PocketTTSError.processingFailed("Failed to parse mimi_init_state from manifest.json")
        }

        var tensors: [String: MLMultiArray] = [:]

        for (name, info) in mimiManifest {
            guard let infoDict = info as? [String: Any],
                let shapeArray = infoDict["shape"] as? [Int],
                let byteCount = infoDict["bytes"] as? Int
            else {
                continue
            }

            let shape = shapeArray.map { NSNumber(value: $0) }
            let array = try MLMultiArray(shape: shape, dataType: .float32)

            if byteCount > 0 && !shapeArray.contains(0) {
                let binURL = stateDir.appendingPathComponent("\(name).bin")
                let data = try Data(contentsOf: binURL)
                let floatCount = byteCount / MemoryLayout<Float>.size
                let dstPtr = array.dataPointer.bindMemory(to: Float.self, capacity: floatCount)
                data.withUnsafeBytes { rawBuffer in
                    let srcPtr = rawBuffer.bindMemory(to: Float.self)
                    dstPtr.update(from: srcPtr.baseAddress!, count: floatCount)
                }
            }

            tensors[name] = array
        }

        // Ensure offset scalars exist
        for key in ["attn0_offset", "attn0_end_offset", "attn1_offset", "attn1_end_offset"] {
            if tensors[key] == nil {
                let scalar = try MLMultiArray(shape: [1], dataType: .float32)
                scalar[0] = NSNumber(value: Float(0))
                tensors[key] = scalar
            }
        }

        return MimiState(tensors: tensors)
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
    ///   - state: The streaming state (26 tensors), modified in place.
    ///   - model: The Mimi CoreML model.
    /// - Returns: Audio samples for this frame (1920 samples = 80ms at 24kHz).
    static func runMimiDecoder(
        latent: [Float],
        state: inout MimiState,
        model: MLModel
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

        // Build input dictionary
        var inputDict: [String: Any] = ["latent": latentArray]
        for (key, array) in state.tensors {
            inputDict[key] = array
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract audio output [1, 1, 1920]
        guard let audioArray = output.featureValue(for: MimiKeys.audioOutput)?.multiArrayValue else {
            throw PocketTTSError.processingFailed("Missing Mimi audio output")
        }

        let sampleCount = PocketTtsConstants.samplesPerFrame
        let samples = readFloatArray(from: audioArray, count: sampleCount)

        // Update streaming state
        for (inputName, outputName) in mimiStateMapping {
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
