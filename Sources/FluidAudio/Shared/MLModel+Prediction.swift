import CoreML
import Foundation

extension MLModel {
    /// Executes a Core ML prediction using the async API only on platforms where
    /// the runtime is stable. macOS 14/iOS 17 can still expose the async method
    /// but the internal E5RT executor times out on large programs (e.g. Kokoro),
    /// so we fall back to the synchronous call there.
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
