@preconcurrency import CoreML
import Foundation
import OSLog

// MARK: - Model Type Aliases

public enum CoreMLSortformer {
    public typealias PreprocessorModel = MLModel
    public typealias MainModel = MLModel
}

// MARK: - Models Container

/// Container for Sortformer CoreML models.
///
/// Sortformer uses three models:
/// - Preprocessor: Audio → Mel features
/// - PreEncoder: Mel features + State → Concatenated embeddings
/// - Head: Concatenated embeddings → Predictions + Chunk embeddings
public struct SortformerModels: Sendable {

    /// Preprocessor model for mel spectrogram extraction
    public let preprocessorModel: CoreMLSortformer.PreprocessorModel

    /// Main Sortformer model for diarization (combined pipeline, deprecated)
    public let mainModel: CoreMLSortformer.MainModel?

    /// Time taken to compile/load models
    public let compilationDuration: TimeInterval

    /// Whether to use separate PreEncoder + Head models (recommended)
    public let useSeparateModels: Bool
    
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
        preprocessor: MLModel,
        main: MLModel,
        compilationDuration: TimeInterval = 0
    ) throws {
        self.preprocessorModel = preprocessor
        self.mainModel = main
        self.useSeparateModels = false
        self.compilationDuration = compilationDuration
        
        self.memoryOptimizer = .init()
        self.chunkArray = try memoryOptimizer.createAlignedArray(shape: [1, NSNumber(value: config.chunkMelFrames), NSNumber(value: config.melFeatures)], dataType: .float32)
        self.fifoArray = try memoryOptimizer.createAlignedArray(shape: [1, NSNumber(value: config.fifoLen), NSNumber(value: config.fcDModel)], dataType: .float32)
        self.spkcacheArray = try memoryOptimizer.createAlignedArray(shape: [1, NSNumber(value: config.spkcacheLen), NSNumber(value: config.fcDModel)], dataType: .float32)
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
        preprocessorPath: URL,
        mainModelPath: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from local paths (combined pipeline mode)")

        let startTime = Date()

        // Compile mlpackage to mlmodelc first
        logger.info("Compiling preprocessor model...")
        let compiledPreprocessorURL = try await MLModel.compileModel(at: preprocessorPath)
        logger.info("Compiling main model...")
        let compiledMainModelURL = try await MLModel.compileModel(at: mainModelPath)

        // Load preprocessor
        let preprocessorConfig = MLModelConfiguration()
        preprocessorConfig.computeUnits = .cpuOnly

        let preprocessor = try MLModel(contentsOf: compiledPreprocessorURL, configuration: preprocessorConfig)
        logger.info("Loaded preprocessor model")

        // Load main model - .all lets CoreML pick optimal compute units
        let mainConfig = MLModelConfiguration()
        mainConfig.computeUnits = .all
        let mainModel = try MLModel(contentsOf: compiledMainModelURL, configuration: mainConfig)
        logger.info("Loaded main Sortformer model")

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Models loaded in \(String(format: "%.2f", duration))s")

        return try SortformerModels(
            config: config,
            preprocessor: preprocessor,
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
    /// Downloads models from alexwengg/diar-streaming-sortformer-coreml if not cached.
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
        
        // Download models if needed
        let modelNames = [
            ModelNames.Sortformer.preprocessorFile,
            ModelNames.Sortformer.unifiedFile,
        ]
        
        let models = try await DownloadUtils.loadModels(
            .sortformer,
            modelNames: modelNames,
            directory: directory,
            computeUnits: computeUnits
        )
        
        guard let preprocessor = models[ModelNames.Sortformer.preprocessorFile],
              let sortformer = models[ModelNames.Sortformer.unifiedFile]
        else {
            throw SortformerError.modelLoadFailed("Failed to load Sortformer models from HuggingFace")
        }
        
        let duration = Date().timeIntervalSince(startTime)
        logger.info("Sortformer models loaded from HuggingFace in \(String(format: "%.2f", duration))s")
        
        return try SortformerModels(
            config: config,
            preprocessor: preprocessor,
            main: sortformer,
            compilationDuration: duration
        )
    }
}

// MARK: - Preprocessor Inference

extension SortformerModels {

    /// Run preprocessor to extract mel features from audio.
    ///
    /// - Parameters:
    ///   - audioSamples: Audio samples (16kHz mono)
    ///   - config: Sortformer configuration
    /// - Returns: Mel features [1, 128, T] flattened and feature length
    public func runPreprocessor(
        audioSamples: [Float],
        config: SortformerConfig
    ) throws -> (features: [Float], featureLength: Int) {

        let expectedSamples = config.preprocessorAudioSamples

        // Create input array with padding if needed
        var paddedAudio = audioSamples
        if paddedAudio.count < expectedSamples {
            paddedAudio.append(contentsOf: [Float](repeating: 0.0, count: expectedSamples - paddedAudio.count))
        }

        // Create MLMultiArray for audio input
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: expectedSamples)], dataType: .float32)
        for i in 0..<expectedSamples {
            audioArray[i] = NSNumber(value: paddedAudio[i])
        }

        // Create length input
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: Int32(expectedSamples))

        // Run inference
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "length": MLFeatureValue(multiArray: lengthArray),
        ])

        let output = try preprocessorModel.prediction(from: inputFeatures)

        // Extract features
        guard let featuresArray = output.featureValue(for: "features")?.multiArrayValue,
            let featureLengthArray = output.featureValue(for: "feature_lengths")?.multiArrayValue
        else {
            throw SortformerError.preprocessorFailed("Missing output features")
        }

        let featureLength = featureLengthArray[0].intValue

        // Convert to flat array [1, 128, T] -> [T * 128] row-major
        var features: [Float] = []
        let shape = featuresArray.shape.map { $0.intValue }
        let melBins = shape[1]
        let timeSteps = shape[2]

        for t in 0..<timeSteps {
            for m in 0..<melBins {
                let idx = m * timeSteps + t
                features.append(featuresArray[idx].floatValue)
            }
        }

        return (features, featureLength)
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
        public let chunkEmbeddingLength: Int
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
        guard let mainModel = mainModel else {
            throw SortformerError.inferenceFailed("Combined model not loaded")
        }

        let chunkFrames = config.chunkMelFrames
        let spkcacheLen = config.spkcacheLen
        let fifoLen = config.fifoLen
        let fcDModel = config.fcDModel
        let melFeatures = config.melFeatures

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

        // Extract outputs (names must match CoreML SortformerPipeline model)
        guard let predictions = output.featureValue(for: "speaker_preds")?.shapedArrayValue(of: Float32.self)?.scalars,
              let chunkEmbeddings = output.featureValue(for: "chunk_pre_encoder_embs")?.shapedArrayValue(of: Float32.self)?.scalars,
              let chunkEmbeddingsLength = output.featureValue(for: "chunk_pre_encoder_lengths")?.shapedArrayValue(of: Int32.self)?.scalars.first
        else {
            throw SortformerError.inferenceFailed("Missing model outputs")
        }

        return MainModelOutput(
            predictions: predictions,
            chunkEmbeddings: chunkEmbeddings,
            chunkEmbeddingLength: Int(chunkEmbeddingsLength)
        )
    }
}
