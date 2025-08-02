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
    public let embeddingPreprocessor: MLModel?
    public let batchFrameExtractor: MLModel?
    public let unifiedPostEmbeddingModel: MLModel?
    public let downloadDuration: TimeInterval
    public let compilationDuration: TimeInterval

    init(segmentation: MLModel, embedding: MLModel, 
         embeddingPreprocessor: MLModel? = nil,
         batchFrameExtractor: MLModel? = nil,
         unifiedPostEmbeddingModel: MLModel? = nil,
         downloadDuration: TimeInterval = 0, 
         compilationDuration: TimeInterval = 0) {
        self.segmentationModel = segmentation
        self.embeddingModel = embedding
        self.embeddingPreprocessor = embeddingPreprocessor
        self.batchFrameExtractor = batchFrameExtractor
        self.unifiedPostEmbeddingModel = unifiedPostEmbeddingModel
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
            EmbeddingModelFileName + ".mlmodelc"
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
        // 1. Optimized model without SliceByIndex operations
        // 2. Float16 optimized version
        // 3. Regular wespeaker model
        var embeddingModel: MLModel?
        var embeddingModelType = "Standard Float32"
        
        // Check for optimized model without SliceByIndex
        let optimizedNoSlicePath = directory.appendingPathComponent("wespeaker_optimized_no_slice.mlpackage")
        let float16Path = directory.appendingPathComponent("wespeaker_float16.mlpackage")
        var isDirectory: ObjCBool = false
        
        if FileManager.default.fileExists(atPath: optimizedNoSlicePath.path, isDirectory: &isDirectory) && isDirectory.boolValue {
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

        // Look for optional optimization models
        print("🔍 Looking for models in directory: \(directory.path)")
        
        // List contents of directory for debugging
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: directory.path) {
            print("   Directory contents: \(contents.filter { $0.contains(".ml") })")
        }
        
        let embeddingPreprocessorPath = directory.appendingPathComponent("embedding_preprocessor.mlpackage")
        
        var embeddingPreprocessor: MLModel?
        
        // Try to load embedding preprocessor
        logger.info("🔍 Looking for embedding preprocessor at: \(embeddingPreprocessorPath.path)")
        if FileManager.default.fileExists(atPath: embeddingPreprocessorPath.path, isDirectory: &isDirectory) {
            do {
                // Check if we need to compile the model first
                let compiledPath = embeddingPreprocessorPath.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledPath.path) {
                    print("   Compiling embedding preprocessor...")
                    let compiledURL = try await MLModel.compileModel(at: embeddingPreprocessorPath)
                    print("   ✅ Compiled to: \(compiledURL.lastPathComponent)")
                    embeddingPreprocessor = try MLModel(contentsOf: compiledURL, configuration: config)
                } else {
                    embeddingPreprocessor = try MLModel(contentsOf: compiledPath, configuration: config)
                }
                logger.info("✅ Successfully loaded embedding preprocessor model - GPU acceleration enabled!")
                print("   ✅ Embedding preprocessor loaded successfully!")
            } catch {
                logger.warning("Failed to load embedding preprocessor: \(error.localizedDescription)")
                print("   ❌ Failed to load embedding preprocessor: \(error)")
            }
        } else {
            logger.info("❌ Embedding preprocessor not found at: \(embeddingPreprocessorPath.path)")
        }
        
        // Load batch frame extractor model
        var batchFrameExtractor: MLModel?
        
        let batchExtractorPath = directory.appendingPathComponent("batch_frame_extractor.mlpackage")
        if FileManager.default.fileExists(atPath: batchExtractorPath.path, isDirectory: &isDirectory) {
            do {
                logger.info("🚀 Found batch frame extractor - eliminates 1001 SliceByIndex operations!")
                let compiledPath = batchExtractorPath.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledPath.path) {
                    print("   Compiling batch frame extractor...")
                    let compiledURL = try await MLModel.compileModel(at: batchExtractorPath)
                    print("   ✅ Compiled to: \(compiledURL.lastPathComponent)")
                    batchFrameExtractor = try MLModel(contentsOf: compiledURL, configuration: config)
                } else {
                    batchFrameExtractor = try MLModel(contentsOf: compiledPath, configuration: config)
                }
                logger.info("✅ Batch frame extractor loaded - 3-5x speedup enabled!")
                print("   ✅ Batch frame extractor loaded successfully!")
            } catch {
                logger.warning("Failed to load batch frame extractor: \(error.localizedDescription)")
                print("   ❌ Failed to load batch frame extractor: \(error)")
            }
        }
        
        // Load unified post-embedding model
        var unifiedPostEmbeddingModel: MLModel?
        
        let unifiedModelPath = directory.appendingPathComponent("unified_post_embedding.mlpackage")
        isDirectory = false
        if FileManager.default.fileExists(atPath: unifiedModelPath.path, isDirectory: &isDirectory) {
            do {
                // Check if we need to compile the model first
                let compiledPath = unifiedModelPath.deletingPathExtension().appendingPathExtension("mlmodelc")
                if !FileManager.default.fileExists(atPath: compiledPath.path) {
                    print("   Compiling unified post-embedding model...")
                    let compiledURL = try await MLModel.compileModel(at: unifiedModelPath)
                    print("   ✅ Compiled to: \(compiledURL.lastPathComponent)")
                    unifiedPostEmbeddingModel = try MLModel(contentsOf: compiledURL, configuration: config)
                } else {
                    unifiedPostEmbeddingModel = try MLModel(contentsOf: compiledPath, configuration: config)
                }
                logger.info("✅ Successfully loaded unified post-embedding model - GPU acceleration enabled!")
                print("   ✅ Unified post-embedding model loaded successfully!")
            } catch {
                logger.warning("Failed to load unified post-embedding model: \(error.localizedDescription)")
                print("   ❌ Failed to load unified post-embedding model: \(error)")
            }
        } else {
            logger.info("🔍 Looking for unified post-embedding model at: \(unifiedModelPath.path)")
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
        print("   Batch Frame Extractor: \(batchFrameExtractor != nil ? "✅ Loaded (No SliceByIndex!)" : "❌ Not Found")")
        print("   Embedding Preprocessor: \(embeddingPreprocessor != nil ? "✅ Loaded" : "❌ Not Found")")
        print("   Unified Post-Embedding: \(unifiedPostEmbeddingModel != nil ? "✅ Loaded" : "❌ Not Found")")
        
        return DiarizerModels(
            segmentation: segmentationModel, 
            embedding: embeddingModel!, // Force unwrap safe - we ensure it's set above
            embeddingPreprocessor: embeddingPreprocessor,
            batchFrameExtractor: batchFrameExtractor,
            unifiedPostEmbeddingModel: unifiedPostEmbeddingModel,
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
            embeddingPreprocessor: nil,
            unifiedPostEmbeddingModel: nil,
            downloadDuration: 0, 
            compilationDuration: loadDuration
        )
    }
}
