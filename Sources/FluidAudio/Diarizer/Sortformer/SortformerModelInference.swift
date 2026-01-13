@preconcurrency import CoreML
import Foundation
import OSLog

// MARK: - Models Container

/// Container for Sortformer CoreML models.
///
/// Sortformer uses three models:
/// - Preprocessor: Audio → Mel features
/// - PreEncoder: Mel features + State → Concatenated embeddings
/// - Head: Concatenated embeddings → Predictions + Chunk embeddings
public struct SortformerModels {
    /// Main Sortformer model for diarization (combined pipeline, deprecated)
    public let mainModel: MLModel

    /// Time taken to compile/load models
    public let compilationDuration: TimeInterval

    /// Cached buffers
    private let memoryOptimizer: ANEMemoryOptimizer
    private let chunkArray: MLMultiArray
    private let chunkLengthArray: MLMultiArray
    private let fifoArray: MLMultiArray
    private let fifoLengthArray: MLMultiArray
    private let spkcacheArray: MLMultiArray
    private let spkcacheLengthArray: MLMultiArray

    public init(
        config: SortformerConfig,
        main: MLModel,
        compilationDuration: TimeInterval = 0
    ) throws {
        self.mainModel = main
        self.compilationDuration = compilationDuration

        self.memoryOptimizer = .init()
        self.chunkArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.chunkMelFrames), NSNumber(value: config.melFeatures)], dataType: .float32)
        self.fifoArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.fifoLen), NSNumber(value: config.preEncoderDims)], dataType: .float32)
        self.spkcacheArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.spkcacheLen), NSNumber(value: config.preEncoderDims)], dataType: .float32)
        self.chunkLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.fifoLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.spkcacheLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
    }
}

// MARK: - Model Loading

extension SortformerModels {

    private static let logger = AppLogger(category: "SortformerModels")

    /// Load models from local file paths (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - preprocessorPath: Path to SortformerPreprocessor.mlpackage
    ///   - mainModelPath: Path to Sortformer.mlpackage
    ///   - configuration: Optional MLModel configuration
    /// - Returns: Loaded SortformerModels
    public static func load(
        config: SortformerConfig,
        mainModelPath: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from local paths (combined pipeline mode)")

        let startTime = Date()

        // Compile mlpackage to mlmodelc first
        logger.info("Compiling main model...")
        let compiledMainModelURL = try await MLModel.compileModel(at: mainModelPath)

        // Load main model - .all lets CoreML pick optimal compute units
        let mainConfig = MLModelConfiguration()
        mainConfig.computeUnits = .all
        let mainModel = try MLModel(contentsOf: compiledMainModelURL, configuration: mainConfig)
        logger.info("Loaded main Sortformer model")

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Models loaded in \(String(format: "%.2f", duration))s")

        return try SortformerModels(
            config: config,
            main: mainModel,
            compilationDuration: duration
        )
    }

    /// Default MLModel configuration
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
        return config
    }

    /// Load Sortformer models from HuggingFace.
    ///
    /// Downloads models from FluidInference/diar-streaming-sortformer-coreml if not cached.
    ///
    /// - Parameters:
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: CoreML compute units to use (default: cpuOnly for consistency)
    /// - Returns: Loaded SortformerModels
    public static func loadFromHuggingFace(
        config: SortformerConfig,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .all
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from HuggingFace...")

        let startTime = Date()

        // Determine cache directory
        let directory: URL
        if let cache = cacheDirectory {
            directory = cache
        } else {
            directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FluidAudio/Models")
        }

        // Determine which file to retrieve
        guard let bundle = ModelNames.Sortformer.bundle(for: config) else {
            throw SortformerError.modelLoadFailed("Unsupported Sortformer configuration")
        }

        logger.info("Downloading Sortformer models from HuggingFace from bundle: \(bundle)...")

        // Download models if needed

        let models = try await DownloadUtils.loadModels(
            .sortformer,
            modelNames: [bundle],
            directory: directory,
            computeUnits: computeUnits
        )

        guard let sortformer = models[bundle]
        else {
            throw SortformerError.modelLoadFailed("Failed to load Sortformer models from HuggingFace")
        }

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Sortformer models loaded from HuggingFace in \(String(format: "%.2f", duration))s")

        return try SortformerModels(
            config: config,
            main: sortformer,
            compilationDuration: duration
        )
    }
}

// MARK: - Main Model Inference

extension SortformerModels {

    /// Main model output structure
    public struct MainModelOutput {
        /// Raw predictions (logits) [spkcache_len + fifo_len + chunk_len, num_speakers]
        public let predictions: [Float]

        /// Chunk embeddings [chunk_len, fc_d_model]
        public let chunkEmbeddings: [Float]

        /// Actual chunk embedding length
        public let chunkLength: Int
    }

    /// Run main Sortformer model.
    ///
    /// - Parameters:
    ///   - chunk: Feature chunk [T, 128] transposed from mel
    ///   - chunkLength: Actual chunk length
    ///   - spkcache: Speaker cache embeddings [spkcache_len, 512]
    ///   - spkcacheLength: Actual speaker cache length
    ///   - fifo: FIFO queue embeddings [fifo_len, 512]
    ///   - fifoLength: Actual FIFO length
    ///   - config: Sortformer configuration
    /// - Returns: MainModelOutput with predictions and embeddings
    public func runMainModel(
        chunk: [Float],
        chunkLength: Int,
        spkcache: [Float],
        spkcacheLength: Int,
        fifo: [Float],
        fifoLength: Int,
        config: SortformerConfig
    ) throws -> MainModelOutput {
        // Copy chunk features
        memoryOptimizer.optimizedCopy(
            from: chunk,
            to: chunkArray,
            pad: true
        )

        // Copy FIFO queue
        memoryOptimizer.optimizedCopy(
            from: fifo,
            to: fifoArray,
            pad: true
        )

        // Copy speaker cache
        memoryOptimizer.optimizedCopy(
            from: spkcache,
            to: spkcacheArray,
            pad: true
        )

        // Create chunk length input
        chunkLengthArray[0] = NSNumber(value: Int32(chunkLength))

        // Create FIFO length input
        fifoLengthArray[0] = NSNumber(value: Int32(fifoLength))

        // Create speaker cache length input
        spkcacheLengthArray[0] = NSNumber(value: Int32(spkcacheLength))

        // Run inference
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunkArray),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLengthArray),
            "spkcache": MLFeatureValue(multiArray: spkcacheArray),
            "spkcache_lengths": MLFeatureValue(multiArray: spkcacheLengthArray),
            "fifo": MLFeatureValue(multiArray: fifoArray),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLengthArray),
        ])

        let output = try mainModel.prediction(from: inputFeatures)

        // Extract outputs (names must match CoreML Sortformer model)
        guard let predictions = output.featureValue(for: "speaker_preds")?.shapedArrayValue(of: Float32.self)?.scalars,
            let chunkEmbeddings = output.featureValue(for: "chunk_pre_encoder_embs")?.shapedArrayValue(
                of: Float32.self)?.scalars,
            let chunkEmbeddingsLength = output.featureValue(for: "chunk_pre_encoder_lengths")?.shapedArrayValue(
                of: Int32.self)?.scalars.first
        else {
            throw SortformerError.inferenceFailed("Missing model outputs")
        }

        return MainModelOutput(
            predictions: predictions,
            chunkEmbeddings: chunkEmbeddings,
            chunkLength: Int(chunkEmbeddingsLength)
        )
    }
}
