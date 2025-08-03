import Accelerate
import CoreML
import Foundation

/// Batch processor for efficient multi-chunk ASR processing
@available(macOS 13.0, iOS 16.0, *)
internal struct BatchProcessor {
    let batchSize: Int
    let enableParallel: Bool

    init(batchSize: Int = 4, enableParallel: Bool = true) {
        self.batchSize = batchSize
        self.enableParallel = enableParallel
    }

    /// Process multiple audio chunks in parallel
    func processBatch(
        chunks: [[Float]],
        using manager: AsrManager
    ) async throws -> [ASRResult] {
        guard enableParallel else {
            // Sequential processing
            var results: [ASRResult] = []
            for chunk in chunks {
                let result = try await manager.transcribe(chunk)
                results.append(result)
            }
            return results
        }

        // Parallel processing with controlled concurrency
        return try await withThrowingTaskGroup(of: (Int, ASRResult).self) { group in
            var results: [ASRResult?] = Array(repeating: nil, count: chunks.count)

            // Process in batches to avoid overwhelming the system
            for (index, chunk) in chunks.enumerated() {
                group.addTask {
                    let result = try await manager.transcribe(chunk)
                    return (index, result)
                }
            }

            // Collect results in order
            for try await (index, result) in group {
                results[index] = result
            }

            return results.compactMap { $0 }
        }
    }

    /// Optimized mel-spectrogram batch processing
    func batchMelSpectrogram(
        audioChunks: [[Float]],
        model: MLModel,
        options: MLPredictionOptions
    ) async throws -> [MLFeatureProvider] {
        // Process multiple mel-spectrograms in parallel
        return try await withThrowingTaskGroup(of: MLFeatureProvider.self) { group in
            var results: [MLFeatureProvider] = []

            for chunk in audioChunks {
                group.addTask {
                    let input = try prepareMelSpectrogramInput(chunk)
                    return try model.prediction(from: input, options: options)
                }
            }

            for try await result in group {
                results.append(result)
            }

            return results
        }
    }

    private func prepareMelSpectrogramInput(_ audioSamples: [Float]) throws -> MLFeatureProvider {
        let audioLength = audioSamples.count

        let audioArray = try MLMultiArray(
            shape: [1, audioLength] as [NSNumber], dataType: .float32)

        // Use Accelerate for optimized copy
        audioArray.dataPointer.withMemoryRebound(to: Float.self, capacity: audioLength) { ptr in
            vDSP_mmov(audioSamples, ptr, 1, 1, vDSP_Length(audioLength), 1)
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: audioLength)

        return try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])
    }
}
