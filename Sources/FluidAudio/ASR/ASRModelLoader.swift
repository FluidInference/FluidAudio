import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public enum ASRModelLoader {

    private static let logger = Logger(
        subsystem: "com.fluidinfluence.asr", category: "ASRModelLoader")

    /// Loads pre-generated ASR optimization models from bundle resources
    public static func loadOptimizationModels() -> ASROptimizationModels {
        // Get the bundle containing the resources
        let bundle = Bundle.module

        // Look for .mlpackage files in Resources/OptimizationModels
        guard let resourcePath = bundle.path(forResource: "OptimizationModels", ofType: nil) else {
            logger.error("OptimizationModels resource directory not found")
            fatalError("Failed to find OptimizationModels resource directory")
        }

        let modelsDirectory = URL(fileURLWithPath: resourcePath)

        // Model URLs - using .mlmodelc extension (compiled models)
        let transposeURL = modelsDirectory.appendingPathComponent("TransposeEncoder.mlmodelc")
        let tokenDurationURL = modelsDirectory.appendingPathComponent(
            "TokenDurationPrediction.mlmodelc")

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        // Load models with error handling
        var loadedModels = 0

        do {
            let transposeModel = try loadModel(
                at: transposeURL, name: "TransposeEncoder", configuration: configuration,
                logger: logger, loadedCount: &loadedModels)
            let tokenDurationModel = try loadModel(
                at: tokenDurationURL, name: "TokenDurationPrediction", configuration: configuration,
                logger: logger, loadedCount: &loadedModels)

            logger.info("Loaded \(loadedModels) optimization models")

            return ASROptimizationModels(
                transpose: transposeModel,
                tokenDuration: tokenDurationModel
            )
        } catch {
            logger.error("Failed to load optimization models: \(error)")
            fatalError("Failed to load CoreML optimization models: \(error)")
        }
    }

    private static func loadModel(
        at url: URL, name: String, configuration: MLModelConfiguration, logger: Logger,
        loadedCount: inout Int
    ) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("\(name) model not found at: \(url.path)")
            throw ASRError.modelLoadFailed
        }

        let model = try MLModel(contentsOf: url, configuration: configuration)
        logger.info("✅ Loaded \(name) model")
        loadedCount += 1
        return model
    }
}

/// Container for ASR optimization models (Transpose, Token Duration Prediction)
public final class ASROptimizationModels {
    public let transpose: MLModel
    public let tokenDuration: MLModel

    public init(
        transpose: MLModel, tokenDuration: MLModel
    ) {
        self.transpose = transpose
        self.tokenDuration = tokenDuration
    }
}
