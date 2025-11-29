import CoreML
import Foundation
import OSLog

public struct CanaryModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let projectionWeights: [Float]
    public let projectionBias: [Float]
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "CanaryModels")

    public init(
        preprocessor: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        projectionWeights: [Float],
        projectionBias: [Float],
        vocabulary: [Int: String]
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.projectionWeights = projectionWeights
        self.projectionBias = projectionBias
        self.vocabulary = vocabulary
    }
}

extension CanaryModels {

    private struct ModelSpec {
        let fileName: String
        let computeUnits: MLComputeUnits
    }

    private static func createModelSpecs(using config: MLModelConfiguration) -> [ModelSpec] {
        return [
            ModelSpec(fileName: ModelNames.Canary.preprocessorFile, computeUnits: .cpuOnly),
            ModelSpec(fileName: ModelNames.Canary.encoderFile, computeUnits: config.computeUnits),
            ModelSpec(fileName: ModelNames.Canary.decoderFile, computeUnits: config.computeUnits),
        ]
    }

    private static func repoPath(from modelsDirectory: URL) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(Repo.canary.folderName)
    }

    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> CanaryModels {
        logger.info("Loading Canary models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()
        let parentDirectory = directory.deletingLastPathComponent()
        let specs = createModelSpecs(using: config)

        // Ensure all model files are downloaded/present
        _ = try await DownloadUtils.loadModels(
            .canary,
            modelNames: specs.map { $0.fileName },
            directory: parentDirectory,
            computeUnits: config.computeUnits // Use config compute units for download, actual loading will use specific ones
        )

        let repoPath = repoPath(from: directory)

        let preprocessorUrl = repoPath.appendingPathComponent(ModelNames.Canary.preprocessorFile)
        let encoderUrl = repoPath.appendingPathComponent(ModelNames.Canary.encoderFile)
        let decoderUrl = repoPath.appendingPathComponent(ModelNames.Canary.decoderFile)

        let preprocessor = try MLModel(contentsOf: preprocessorUrl, configuration: {
            let preprocessorConfig = MLModelConfiguration()
            preprocessorConfig.computeUnits = .cpuOnly // Preprocessor always on CPU
            return preprocessorConfig
        }())
        let encoder = try MLModel(contentsOf: encoderUrl, configuration: config)
        let decoder = try MLModel(contentsOf: decoderUrl, configuration: config)

        // Load projection weights
        let weightsUrl = repoPath.appendingPathComponent("projection_weights.bin")
        let biasUrl = repoPath.appendingPathComponent("projection_bias.bin")
        
        let weightsData = try Data(contentsOf: weightsUrl)
        let biasData = try Data(contentsOf: biasUrl)
        
        let weights = weightsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let bias = biasData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        let vocabulary = try loadVocabulary(from: directory)

        return CanaryModels(
            preprocessor: preprocessor,
            encoder: encoder,
            decoder: decoder,
            projectionWeights: weights,
            projectionBias: bias,
            vocabulary: vocabulary
        )
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = repoPath(from: directory).appendingPathComponent(
            ModelNames.Canary.vocabularyFile)

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            // Fallback or error
             logger.warning("Vocabulary file not found at \(vocabPath.path)")
             // For now, return empty or throw. 
             // If it's a tokenizer.json, we need to parse it differently.
             // Let's assume for now we can just load it if it exists, but we might need to implement a proper tokenizer loader.
             throw AsrModelsError.modelNotFound(ModelNames.Canary.vocabularyFile, vocabPath)
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            // TODO: Implement proper tokenizer.json parsing
            // For now, assuming a simple map for compatibility or placeholder
            // If it's a standard HF tokenizer.json, it has a "model" -> "vocab" structure.
            
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let model = json["model"] as? [String: Any] {
                
                if let vocab = model["vocab"] as? [String: Int] {
                    var vocabulary: [Int: String] = [:]
                    for (token, id) in vocab {
                        vocabulary[id] = token
                    }
                    return vocabulary
                } else if let vocabList = model["vocab"] as? [String] {
                    var vocabulary: [Int: String] = [:]
                    for (index, token) in vocabList.enumerated() {
                        vocabulary[index] = token
                    }
                    return vocabulary
                }
            }
            
            // Fallback to simple map check (if not nested under model)
                 // Try simple map
                 let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]
                 var vocabulary: [Int: String] = [:]
                 for (key, value) in jsonDict {
                     if let tokenId = Int(key) {
                         vocabulary[tokenId] = value
                     }
                 }
                 return vocabulary
        } catch {
            logger.error("Failed to load vocabulary: \(error)")
            throw AsrModelsError.loadingFailed("Vocabulary parsing failed")
        }
    }

    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return config
    }
    
    public static func download(
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? AsrModels.defaultCacheDirectory(for: .v3).deletingLastPathComponent().appendingPathComponent(Repo.canary.folderName)
        
        // Reuse AsrModels download logic or call DownloadUtils directly
        // ...
        // For brevity, I'll implement a simple download wrapper similar to AsrModels
        
        logger.info("Downloading Canary models to: \(targetDir.path)")
        let parentDir = targetDir.deletingLastPathComponent()

        let specs = [
            ModelNames.Canary.preprocessorFile,
            ModelNames.Canary.encoderFile,
            ModelNames.Canary.decoderFile
        ]
        
        _ = try await DownloadUtils.loadModels(
            .canary,
            modelNames: specs,
            directory: parentDir,
            computeUnits: .cpuAndNeuralEngine
        )
        
        return targetDir
    }
}
