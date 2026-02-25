import CoreML
import Foundation

// MLModel is thread-safe by design but lacks a formal Sendable conformance.
// This retroactive conformance makes TaskGroup closures that capture MLModel
// compile cleanly under Swift 6 strict concurrency.
extension MLModel: @unchecked @retroactive Sendable {}

extension MLModel {
    /// Compatibly call Core ML prediction using async API.
    public func compatPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions
    ) async throws -> MLFeatureProvider {
        try await prediction(from: input, options: options)
    }
}
