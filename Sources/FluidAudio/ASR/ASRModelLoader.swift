import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public enum ASRModelLoader {

    private static let logger = Logger(
        subsystem: "com.fluidinfluence.asr", category: "ASRModelLoader")

    /// Loads ASR optimization models from downloaded models
    public static func loadOptimizationModels() async throws -> ASROptimizationModels {
        logger.info("Loading ASR optimization models from downloaded models")

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        // Use DownloadUtils to load the TokenDurationPrediction model
        let models = try await DownloadUtils.loadModels(
            .parakeet,
            modelNames: [AsrModels.ModelNames.tokenDuration],
            directory: AsrModels.defaultCacheDirectory().deletingLastPathComponent(),
            computeUnits: configuration.computeUnits
        )

        guard let tokenDurationModel = models[AsrModels.ModelNames.tokenDuration] else {
            throw ASRError.modelLoadFailed
        }

        logger.info("Successfully loaded TokenDurationPrediction model")

        return ASROptimizationModels(
            tokenDuration: tokenDurationModel
        )
    }

}

/// Container for ASR optimization models (Token Duration Prediction)
public final class ASROptimizationModels {
    public let tokenDuration: MLModel

    public init(
        tokenDuration: MLModel
    ) {
        self.tokenDuration = tokenDuration
    }
}
