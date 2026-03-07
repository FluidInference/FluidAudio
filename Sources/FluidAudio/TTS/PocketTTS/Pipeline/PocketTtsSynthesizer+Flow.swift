@preconcurrency import CoreML
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
        model: MLModel,
        rng: inout some RandomNumberGenerator
    ) async throws -> [Float] {
        let latentDim = PocketTtsConstants.latentDim
        let dt: Float = 1.0 / Float(numSteps)

        // Initialize latent with scaled random noise: randn * sqrt(temperature)
        var latent = [Float](repeating: 0, count: latentDim)
        let scale = sqrtf(temperature)
        for i in 0..<latentDim {
            latent[i] = Float.gaussianRandom(using: &rng) * scale
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
            throw PocketTTSError.processingFailed("Missing flow decoder velocity output")
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

// MARK: - Seeded Random

/// Simple seeded random number generator (xoshiro256**).
///
/// Provides reproducible random sequences when a seed is set,
/// and falls back to system entropy when unseeded.
struct SeededRNG: RandomNumberGenerator {
    private var state: (UInt64, UInt64, UInt64, UInt64)

    init(seed: UInt64) {
        // SplitMix64 to expand seed into 4-part state
        var s = seed
        func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        state = (next(), next(), next(), next())
    }

    mutating func next() -> UInt64 {
        let result = rotl(state.1 &* 5, 7) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = rotl(state.3, 45)
        return result
    }

    private func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }
}

extension Float {
    /// Generate a single sample from the standard normal distribution (Box-Muller transform).
    static func gaussianRandom(using rng: inout some RandomNumberGenerator) -> Float {
        let u1 = Float.random(in: Float.leastNonzeroMagnitude...1.0, using: &rng)
        let u2 = Float.random(in: 0.0...1.0, using: &rng)
        return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
    }
}
