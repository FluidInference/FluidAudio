@preconcurrency import CoreML
import Foundation

extension MLModel {
    /// Compatibly call Core ML prediction using async API.
    public func compatPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions
    ) async throws -> MLFeatureProvider {
        try await prediction(from: input, options: options)
    }
}
