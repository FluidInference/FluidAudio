import Accelerate
import CoreML
import Foundation

/// Lightweight helpers that exercise Core ML models with zero-valued inputs to
/// prime memory allocations before running the offline diarization pipeline.
enum ModelWarmup {

    /// Performs a warmup loop for a model with a single MLMultiArray input.
    ///
    /// - Parameters:
    ///   - model: Model to warm up.
    ///   - inputName: Feature name expected by the model.
    ///   - inputShape: Shape of the MLMultiArray (e.g. `[1, 1, 160_000]`).
    ///   - iterations: Number of times to execute `prediction`.
    /// - Returns: Total elapsed duration in seconds.
    static func warmup(
        model: MLModel,
        inputName: String,
        inputShape: [Int],
        iterations: Int = 1
    ) throws -> TimeInterval {
        precondition(iterations > 0, "Warmup iterations must be positive")
        precondition(!inputShape.isEmpty, "Input shape must not be empty")

        let array = try MLMultiArray(
            shape: inputShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        array.resetToZeros()

        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: array)
        ])

        let start = Date()
        for _ in 0..<iterations {
            _ = try model.prediction(from: features)
        }
        return Date().timeIntervalSince(start)
    }

    /// Warm up the embedding extractor with representative audio + weight inputs.
    ///
    /// We reproduce the exact shapes used during inference to make sure Core ML
    /// allocates and caches buffers on the correct compute units (ANE/GPU).
    static func warmupEmbeddingModel(
        _ model: MLModel,
        weightFrames: Int,
        audioSamples: Int = 160_000
    ) throws {
        precondition(weightFrames > 0, "weightFrames must be positive")

        do {
            let inputs = model.modelDescription.inputDescriptionsByName
            let featureShape: [Int]
            if let fbank = inputs.first(where: { $0.key.caseInsensitiveCompare("fbank_features") == .orderedSame })?
                .value.multiArrayConstraint?.shape
            {
                let mapped = fbank.map { $0.intValue }
                if !mapped.isEmpty, mapped.allSatisfy({ $0 > 0 }) {
                    featureShape = mapped
                } else {
                    featureShape = [1, 1, 80, 998]
                }
            } else {
                featureShape = [1, 1, 80, 998]
            }

            let weightsShape: [Int]
            if let weights = inputs.first(where: { $0.key.caseInsensitiveCompare("weights") == .orderedSame })?
                .value.multiArrayConstraint?.shape
            {
                let mapped = weights.map { $0.intValue }
                if !mapped.isEmpty, mapped.allSatisfy({ $0 > 0 }) {
                    weightsShape = mapped
                } else {
                    weightsShape = [1, weightFrames]
                }
            } else {
                weightsShape = [1, weightFrames]
            }

            let featureArray = try MLMultiArray(
                shape: featureShape.map { NSNumber(value: $0) },
                dataType: .float32
            )
            featureArray.resetToZeros()

            let weightArray = try MLMultiArray(
                shape: weightsShape.map { NSNumber(value: $0) },
                dataType: .float32
            )
            weightArray.resetToZeros()

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "fbank_features": MLFeatureValue(multiArray: featureArray),
                "weights": MLFeatureValue(multiArray: weightArray),
            ])

            _ = try model.prediction(from: provider)
            return
        } catch {
            // Fall back to combined legacy interface.
        }

        let totalElements = audioSamples + weightFrames
        do {
            let combinedArray = try MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: totalElements)],
                dataType: .float32
            )
            combinedArray.resetToZeros()

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "audio_and_weights": MLFeatureValue(multiArray: combinedArray)
            ])

            _ = try model.prediction(from: provider)
            return
        } catch {
            // Fall through to legacy dual-input warmup for older embedding models.
        }

        let audioArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: audioSamples)],
            dataType: .float32
        )
        audioArray.resetToZeros()

        let weightArray = try MLMultiArray(
            shape: [1, NSNumber(value: weightFrames)],
            dataType: .float32
        )
        weightArray.resetToZeros()

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "weights": MLFeatureValue(multiArray: weightArray),
        ])

        _ = try model.prediction(from: provider)
    }
}

extension MLMultiArray {
    fileprivate func resetToZeros() {
        let pointer = dataPointer.assumingMemoryBound(to: Float.self)
        let count = self.count
        var zero: Float = 0
        vDSP_vfill(&zero, pointer, 1, vDSP_Length(count))
    }
}
