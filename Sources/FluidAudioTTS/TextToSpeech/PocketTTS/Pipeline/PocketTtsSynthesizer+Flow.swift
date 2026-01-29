@preconcurrency import CoreML
import FluidAudio
import Foundation

extension PocketTtsSynthesizer {

    /// Run the flow decoder using Euler integration (LSD steps).
    ///
    /// Converts transformer output to a 32-dimensional audio latent
    /// via `numSteps` iterative denoising steps.
    static func flowDecode(
        transformerOut: MLMultiArray,
        numSteps: Int,
        temperature: Float,
        model: MLModel
    ) async throws -> [Float] {
        let latentDim = PocketTtsConstants.latentDim
        let dt: Float = 1.0 / Float(numSteps)

        // Initialize latent with scaled random noise: randn * sqrt(temperature)
        var latent = [Float](repeating: 0, count: latentDim)
        let scale = sqrtf(temperature)
        for i in 0..<latentDim {
            latent[i] = Float.gaussianRandom() * scale
        }

        // Flatten transformer_out from [1, 1, 1024] to [1, 1024]
        let transformerFlat = try reshapeToFlat(transformerOut, dim: PocketTtsConstants.transformerDim)

        // Euler integration: 8 steps from t=0 to t=1
        for step in 0..<numSteps {
            let sValue = Float(step) * dt
            let tValue = Float(step + 1) * dt

            let velocity = try await runFlowDecoderStep(
                transformerOut: transformerFlat,
                latent: latent,
                s: sValue,
                t: tValue,
                model: model
            )

            // Euler step: latent += velocity * dt
            for i in 0..<latentDim {
                latent[i] += velocity[i] * dt
            }
        }

        return latent
    }

    /// Denormalize a latent vector: result = latent * std + mean.
    static func denormalize(
        _ latent: [Float], mean: [Float], std: [Float]
    ) -> [Float] {
        var result = [Float](repeating: 0, count: latent.count)
        for i in 0..<latent.count {
            result[i] = latent[i] * std[i] + mean[i]
        }
        return result
    }

    /// Quantize a latent vector using the quantizer weight matrix.
    ///
    /// Computes `dot(latent, weight.T)` where weight is [512, 32] (stored as flat array).
    /// Result shape: [512].
    static func quantize(_ latent: [Float], weight: [Float]) -> [Float] {
        let outDim = 512
        let inDim = PocketTtsConstants.latentDim
        var result = [Float](repeating: 0, count: outDim)

        // weight is [512, 32] row-major: weight[i * 32 + j]
        for i in 0..<outDim {
            var sum: Float = 0
            let rowOffset = i * inDim
            for j in 0..<inDim {
                sum += weight[rowOffset + j] * latent[j]
            }
            result[i] = sum
        }

        return result
    }

    // MARK: - Private

    /// Run a single flow decoder step.
    private static func runFlowDecoderStep(
        transformerOut: MLMultiArray,
        latent: [Float],
        s: Float,
        t: Float,
        model: MLModel
    ) async throws -> [Float] {
        let latentDim = PocketTtsConstants.latentDim

        // Create latent MLMultiArray [1, 32]
        let latentArray = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim)], dataType: .float32)
        let latentPtr = latentArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        latent.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            latentPtr.update(from: base, count: latentDim)
        }

        // Create s and t MLMultiArrays [1, 1]
        let sArray = try MLMultiArray(shape: [1, 1], dataType: .float32)
        sArray[0] = NSNumber(value: s)

        let tArray = try MLMultiArray(shape: [1, 1], dataType: .float32)
        tArray[0] = NSNumber(value: t)

        let inputDict: [String: Any] = [
            "transformer_out": transformerOut,
            "latent": latentArray,
            "s": sArray,
            "t": tArray,
        ]

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract velocity — take the first (and likely only) output
        let outputNames = Array(output.featureNames)
        guard let velocityArray = output.featureValue(for: outputNames[0])?.multiArrayValue else {
            throw TTSError.processingFailed("Missing flow decoder velocity output")
        }

        let velocityPtr = velocityArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        return Array(UnsafeBufferPointer(start: velocityPtr, count: latentDim))
    }

    /// Reshape a [1, 1, D] MLMultiArray to [1, D].
    private static func reshapeToFlat(_ array: MLMultiArray, dim: Int) throws -> MLMultiArray {
        let flat = try MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)
        let srcPtr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        let dstPtr = flat.dataPointer.bindMemory(to: Float.self, capacity: dim)
        dstPtr.update(from: srcPtr, count: dim)
        return flat
    }
}

// MARK: - Gaussian Random

extension Float {
    /// Generate a single sample from the standard normal distribution (Box-Muller transform).
    static func gaussianRandom() -> Float {
        let u1 = Float.random(in: Float.leastNonzeroMagnitude...1.0)
        let u2 = Float.random(in: 0.0...1.0)
        return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
    }
}
