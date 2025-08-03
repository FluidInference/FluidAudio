@preconcurrency import CoreML
import Foundation
import OSLog

public enum CoreMLDiarizer {
    public typealias SegmentationModel = MLModel
    public typealias EmbeddingModel = MLModel
}

@available(macOS 13.0, iOS 16.0, *)
public struct DiarizerModels: Sendable {

    public let segmentationModel: CoreMLDiarizer.SegmentationModel
    public let embeddingModel: CoreMLDiarizer.EmbeddingModel
    public let downloadDuration: TimeInterval
    public let compilationDuration: TimeInterval

    init(segmentation: MLModel, embedding: MLModel, 
         downloadDuration: TimeInterval = 0, 
         compilationDuration: TimeInterval = 0) {
        self.segmentationModel = segmentation
        self.embeddingModel = embedding
        self.downloadDuration = downloadDuration
        self.compilationDuration = compilationDuration
    }
}

// -----------------------------
// MARK: - Download from Hugging Face.
// -----------------------------

extension DiarizerModels {

    private static let SegmentationModelFileName = "pyannote_segmentation"
    private static let EmbeddingModelFileName = "wespeaker"

    public static func download(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        let logger = Logger(subsystem: "FluidAudio", category: "DiarizerModels")
        logger.info("Checking for diarizer models...")

        let startTime = Date()
        let directory = directory ?? defaultModelsDirectory()
        let config = configuration ?? defaultConfiguration()

        let modelNames = [
            SegmentationModelFileName + ".mlmodelc",
            "wespeaker_int8.mlmodelc"  // INT8 is now the primary embedding model
        ]

        let models = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: modelNames,
            directory: directory.deletingLastPathComponent(),
            computeUnits: config.computeUnits
        )

        guard let segmentationModel = models[SegmentationModelFileName + ".mlmodelc"] else {
            throw DiarizerError.modelDownloadFailed
        }
        
        // Priority order for embedding models:
        // 1. INT8 quantized model (DEFAULT - best performance/accuracy tradeoff)
        // 2. Optimized model without SliceByIndex operations
        // 3. Float16 optimized version
        // 4. Regular wespeaker model (fallback)
        var embeddingModel: MLModel?
        var embeddingModelType = "Standard Float32"
        
        // Always try INT8 model first (it's now the default)
        let int8Path = directory.appendingPathComponent("wespeaker_int8.mlmodelc")
        if FileManager.default.fileExists(atPath: int8Path.path) {
            do {
                logger.info("🚀 Loading INT8 quantized embedding model (default)")
                embeddingModel = try MLModel(contentsOf: int8Path, configuration: config)
                embeddingModelType = "🔥 INT8 Quantized (100x+ RTF)"
                logger.info("✅ Loaded INT8 embedding model - optimal performance enabled!")
            } catch {
                logger.warning("Failed to load INT8 model: \(error.localizedDescription)")
            }
        }
        
        // Check for optimized model without SliceByIndex
        let optimizedNoSlicePath = directory.appendingPathComponent("wespeaker_optimized_no_slice.mlpackage")
        let float16Path = directory.appendingPathComponent("wespeaker_float16.mlpackage")
        var isDirectory: ObjCBool = false
        
        if embeddingModel == nil && FileManager.default.fileExists(atPath: optimizedNoSlicePath.path, isDirectory: &isDirectory) && isDirectory.boolValue {
            do {
                logger.info("🚀 Found optimized embedding model WITHOUT SliceByIndex operations!")
                // Check if we need to compile it first
                let compiledPath = optimizedNoSlicePath.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledPath.path) {
                    logger.info("   Compiling optimized model...")
                    let compiledURL = try await MLModel.compileModel(at: optimizedNoSlicePath)
                    embeddingModel = try MLModel(contentsOf: compiledURL, configuration: config)
                } else {
                    embeddingModel = try MLModel(contentsOf: compiledPath, configuration: config)
                }
                embeddingModelType = "✅ Optimized (No SliceByIndex!)"
                logger.info("✅ Loaded optimized embedding model - NO SliceByIndex operations!")
            } catch {
                logger.warning("Failed to load optimized no-slice model: \(error.localizedDescription)")
            }
        }
        
        // Try Float16 if optimized not available
        if embeddingModel == nil && FileManager.default.fileExists(atPath: float16Path.path, isDirectory: &isDirectory) {
            do {
                logger.info("🚀 Found Float16 optimized embedding model")
                // Check if we need to compile it first
                let compiledPath = float16Path.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledPath.path) {
                    logger.info("   Compiling Float16 model...")
                    let compiledURL = try await MLModel.compileModel(at: float16Path)
                    embeddingModel = try MLModel(contentsOf: compiledURL, configuration: config)
                } else {
                    embeddingModel = try MLModel(contentsOf: compiledPath, configuration: config)
                }
                embeddingModelType = "✅ Float16 Optimized"
                logger.info("✅ Loaded Float16 optimized embedding model")
            } catch {
                logger.warning("Failed to load Float16 model: \(error.localizedDescription)")
            }
        }
        
        // Fallback to regular model if optimized versions not available
        if embeddingModel == nil {
            guard let regularModel = models[EmbeddingModelFileName + ".mlmodelc"] else {
                throw DiarizerError.modelDownloadFailed
            }
            embeddingModel = regularModel
            embeddingModelType = "📦 Standard Float32"
        }


        let endTime = Date()
        let totalDuration = endTime.timeIntervalSince(startTime)
        // For now, we don't have separate download vs compilation times, so we'll estimate
        // In reality, if models are cached, download time is 0
        let downloadDuration: TimeInterval = 0 // Models are typically cached
        let compilationDuration = totalDuration // Most time is spent on compilation
        
        // Debug print to verify models are loaded
        print("🔍 Model Loading Status:")
        print("   Embedding Model: \(embeddingModelType)")
        
        return DiarizerModels(
            segmentation: segmentationModel, 
            embedding: embeddingModel!, // Force unwrap safe - we ensure it's set above
            downloadDuration: downloadDuration, 
            compilationDuration: compilationDuration
        )
    }

    public static func load(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        let directory = directory ?? defaultModelsDirectory()
        return try await download(to: directory, configuration: configuration)
    }

    public static func downloadIfNeeded(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        return try await download(to: directory, configuration: configuration)
    }

    static func defaultModelsDirectory() -> URL {
        let applicationSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return applicationSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(DownloadUtils.Repo.diarizer.folderName, isDirectory: true)
    }

    static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        // Enable Float16 optimization for ~2x speedup
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
        return config
    }
}

// -----------------------------
// MARK: - Predownloaded models.
// -----------------------------

extension DiarizerModels {

    /// Load the models from the given local files.
    ///
    /// If the models fail to load, no recovery will be attempted. No models are downloaded.
    ///
    public static func load(
        localSegmentationModel: URL,
        localEmbeddingModel: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {

        let logger = Logger(subsystem: "FluidAudio", category: "DiarizerModels")
        logger.info("Loading predownloaded models")

        let configuration = configuration ?? defaultConfiguration()

        let startTime = Date()
        let segmentationModel = try MLModel(contentsOf: localSegmentationModel, configuration: configuration)
        let embeddingModel = try MLModel(contentsOf: localEmbeddingModel, configuration: configuration)

        let endTime = Date()
        let loadDuration = endTime.timeIntervalSince(startTime)
        return DiarizerModels(
            segmentation: segmentationModel, 
            embedding: embeddingModel,
            downloadDuration: 0, 
            compilationDuration: loadDuration
        )
    }
}