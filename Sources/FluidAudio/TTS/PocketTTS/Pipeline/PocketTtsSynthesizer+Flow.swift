@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Run the flow decoder (LSD Euler integration), fused into one predict.
    ///
    /// Converts the 1024-d transformer hidden state into a 32-d audio latent code.
    /// Flow matching starts from random Gaussian noise and moves it toward a
    /// valid audio code over `numSteps` Euler steps. The whole integration loop
    /// is fused into the CoreML graph (`flow_decoder_fused.mlpackage`), so this
    /// makes a SINGLE `predict()` per frame instead of `numSteps` calls — the
    /// host previously dispatched 8×/frame (~336 calls/utterance), which paid
    /// dispatch + fp32↔fp16 cast 8× on a kernel too small to amortize ANE
    /// residency. The fused model takes `latent_init` (z_0) and returns
    /// `latent_final` (z_N); the `s`/`t` time endpoints are baked in at
    /// conversion for the chosen step count, so `numSteps` here MUST match the
    /// value passed to `convert_flow_decoder_fused.py --num-steps`.
    static func flowDecode(
        transformerOut: MLMultiArray,
        numSteps: Int,
        temperature: Float,
        model: MLModel,
        rng: inout some RandomNumberGenerator
    ) async throws -> [Float] {
        let latentDim = PocketTtsConstants.latentDim

        // Initialize z_0 with scaled random noise (host owns the seed/RNG).
        // sqrt(temperature) because variance scales quadratically with the multiplier.
        var latent = [Float](repeating: 0, count: latentDim)
        let scale = sqrtf(temperature)
        for i in 0..<latentDim {
            latent[i] = Float.gaussianRandom(using: &rng) * scale
        }

        // Flatten transformer_out from [1, 1, 1024] to [1, 1024].
        let transformerFlat = try reshapeToFlat(transformerOut, dim: PocketTtsConstants.transformerDim)

        // Build latent_init [1, 32].
        let latentArray = try MLMultiArray(
            shape: [1, NSNumber(value: latentDim)], dataType: .float32)
        let latentPtr = latentArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        latent.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            latentPtr.update(from: base, count: latentDim)
        }

        // One fused predict: the model runs all `numSteps` Euler steps internally.
        let inputDict: [String: Any] = [
            "transformer_out": transformerFlat,
            "latent_init": latentArray,
        ]
        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Prefer the named output; fall back to the sole output for robustness
        // against CoreML auto-generated names.
        let finalName =
            output.featureNames.contains("latent_final")
            ? "latent_final" : (output.featureNames.first ?? "")
        guard let finalArray = output.featureValue(for: finalName)?.multiArrayValue else {
            throw PocketTTSError.processingFailed("Missing fused flow decoder latent_final output")
        }

        let finalPtr = finalArray.dataPointer.bindMemory(to: Float.self, capacity: latentDim)
        return Array(UnsafeBufferPointer(start: finalPtr, count: latentDim))
    }

    // MARK: - Private

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
