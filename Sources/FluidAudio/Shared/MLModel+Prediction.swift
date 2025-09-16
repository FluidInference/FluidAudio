import CoreML
import Foundation

extension MLModel {
    /// Compatibly call Core ML prediction across Swift compiler versions.
    internal func compatPrediction(
        from input: MLFeatureProvider,
        options: MLPredictionOptions
    ) async throws -> MLFeatureProvider {
        #if compiler(>=6.0)
        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *) {
            return try await prediction(from: input, options: options)
        } else {
            return try await withCheckedThrowingContinuation { continuation in
                do {
                    let result = try prediction(from: input, options: options)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        #else
        return try prediction(from: input, options: options)
        #endif
    }
}
