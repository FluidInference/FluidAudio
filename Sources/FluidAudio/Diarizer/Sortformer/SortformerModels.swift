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

    /// PreEncoder model (features → embeddings)
    public let preEncoderModel: MLModel

    /// Head model (embeddings → predictions)
    public let headModel: MLModel

    /// Main Sortformer model for diarization (combined pipeline, deprecated)
    public let mainModel: CoreMLSortformer.MainModel?

    /// Time taken to compile/load models
    public let compilationDuration: TimeInterval

    /// Whether to use separate PreEncoder + Head models (recommended)
    public let useSeparateModels: Bool

    public init(
        preprocessor: MLModel,
        main: MLModel,
        compilationDuration: TimeInterval = 0
    ) {
        self.preprocessorModel = preprocessor
        self.mainModel = main
        self.preEncoderModel = main  // Fallback
        self.headModel = main  // Fallback
        self.useSeparateModels = false
        self.compilationDuration = compilationDuration
    }

    public init(
        preprocessor: MLModel,
        preEncoder: MLModel,
        head: MLModel,
        compilationDuration: TimeInterval = 0
    ) {
        self.preprocessorModel = preprocessor
        self.preEncoderModel = preEncoder
        self.headModel = head
        self.mainModel = nil
        self.useSeparateModels = true
        self.compilationDuration = compilationDuration
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
        preprocessorConfig.computeUnits = .all

        let preprocessor = try MLModel(contentsOf: compiledPreprocessorURL, configuration: preprocessorConfig)
        logger.info("Loaded preprocessor model")

        // Load main model
        let mainConfig = MLModelConfiguration()
        mainConfig.computeUnits = .all
        let mainModel = try MLModel(contentsOf: compiledMainModelURL, configuration: mainConfig)
        logger.info("Loaded main Sortformer model")

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Models loaded in \(String(format: "%.2f", duration))s")

        return SortformerModels(
            preprocessor: preprocessor,
            main: mainModel,
            compilationDuration: duration
        )
    }

    /// Load models from local file paths (separate PreEncoder + Head mode).
    /// This matches Python's behavior and avoids combined pipeline issues.
    ///
    /// - Parameters:
    ///   - preprocessorPath: Path to Pipeline_Preprocessor.mlpackage
    ///   - preEncoderPath: Path to Pipeline_PreEncoder.mlpackage
    ///   - headPath: Path to Pipeline_Head_Fixed.mlpackage
    /// - Returns: Loaded SortformerModels
    public static func loadSeparate(
        preprocessorPath: URL,
        preEncoderPath: URL,
        headPath: URL
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from local paths (separate models mode)")

        let startTime = Date()

        // Compile all models
        logger.info("Compiling preprocessor model...")
        let compiledPreprocessorURL = try await MLModel.compileModel(at: preprocessorPath)
        logger.info("Compiling PreEncoder model...")
        let compiledPreEncoderURL = try await MLModel.compileModel(at: preEncoderPath)
        logger.info("Compiling Head model...")
        let compiledHeadURL = try await MLModel.compileModel(at: headPath)

        // Load preprocessor
        // Use cpuOnly to match Python's CPU_ONLY for numerical consistency
        let preprocessorConfig = MLModelConfiguration()
        preprocessorConfig.computeUnits = .cpuOnly
        let preprocessor = try MLModel(contentsOf: compiledPreprocessorURL, configuration: preprocessorConfig)
        logger.info("Loaded preprocessor model")

        // Load PreEncoder
        let preEncoderConfig = MLModelConfiguration()
        preEncoderConfig.computeUnits = .cpuOnly
        let preEncoder = try MLModel(contentsOf: compiledPreEncoderURL, configuration: preEncoderConfig)
        logger.info("Loaded PreEncoder model")

        // Load Head
        let headConfig = MLModelConfiguration()
        headConfig.computeUnits = .cpuOnly
        let head = try MLModel(contentsOf: compiledHeadURL, configuration: headConfig)
        logger.info("Loaded Head model")

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Models loaded in \(String(format: "%.2f", duration))s (separate mode)")

        return SortformerModels(
            preprocessor: preprocessor,
            preEncoder: preEncoder,
            head: head,
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
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly
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
            ModelNames.Sortformer.preEncoderFile,
            ModelNames.Sortformer.headFile,
        ]

        let models = try await DownloadUtils.loadModels(
            .sortformer,
            modelNames: modelNames,
            directory: directory,
            computeUnits: computeUnits
        )

        guard let preprocessor = models[ModelNames.Sortformer.preprocessorFile],
            let preEncoder = models[ModelNames.Sortformer.preEncoderFile],
            let head = models[ModelNames.Sortformer.headFile]
        else {
            throw SortformerError.modelLoadFailed("Failed to load Sortformer models from HuggingFace")
        }

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Sortformer models loaded from HuggingFace in \(String(format: "%.2f", duration))s")

        return SortformerModels(
            preprocessor: preprocessor,
            preEncoder: preEncoder,
            head: head,
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
    /// - Returns: Mel features [1, 80, T] flattened and feature length
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

        // Convert to flat array [1, 80, T] -> [T * 80] row-major
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
    ///   - chunk: Feature chunk [T, 80] transposed from mel
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
        if useSeparateModels {
            return try runSeparateModels(
                chunk: chunk,
                chunkLength: chunkLength,
                spkcache: spkcache,
                spkcacheLength: spkcacheLength,
                fifo: fifo,
                fifoLength: fifoLength,
                config: config
            )
        } else {
            return try runCombinedModel(
                chunk: chunk,
                chunkLength: chunkLength,
                spkcache: spkcache,
                spkcacheLength: spkcacheLength,
                fifo: fifo,
                fifoLength: fifoLength,
                config: config
            )
        }
    }

    /// Run combined SortformerPipeline model (legacy mode).
    private func runCombinedModel(
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

        let chunkFrames = config.chunkFrames
        let spkcacheLen = config.spkcacheLen
        let fifoLen = config.fifoLen
        let fcDModel = config.fcDModel
        let melFeatures = config.melFeatures

        // Create chunk input [1, chunkFrames, melFeatures] - FP32 for pipeline mode
        let chunkArray = try MLMultiArray(
            shape: [1, NSNumber(value: chunkFrames), NSNumber(value: melFeatures)],
            dataType: .float32
        )

        // Copy chunk data (pad if needed)
        let chunkPtr = chunkArray.dataPointer.bindMemory(to: Float32.self, capacity: chunkFrames * melFeatures)
        for t in 0..<chunkFrames {
            for f in 0..<melFeatures {
                let srcIdx = t * melFeatures + f
                let dstIdx = t * melFeatures + f
                if srcIdx < chunk.count {
                    chunkPtr[dstIdx] = Float32(chunk[srcIdx])
                } else {
                    chunkPtr[dstIdx] = Float32(0.0)
                }
            }
        }

        // Create chunk length input
        let chunkLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        chunkLengthArray[0] = NSNumber(value: Int32(chunkLength))

        // Create spkcache input [1, spkcacheLen, fcDModel] - FP32 for pipeline mode
        let spkcacheArray = try MLMultiArray(
            shape: [1, NSNumber(value: spkcacheLen), NSNumber(value: fcDModel)],
            dataType: .float32
        )
        let spkcachePtr = spkcacheArray.dataPointer.bindMemory(to: Float32.self, capacity: spkcacheLen * fcDModel)
        for i in 0..<min(spkcache.count, spkcacheLen * fcDModel) {
            spkcachePtr[i] = Float32(spkcache[i])
        }

        let spkcacheLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        spkcacheLengthArray[0] = NSNumber(value: Int32(spkcacheLength))

        // Create fifo input [1, fifoLen, fcDModel] - FP32 for pipeline mode
        let fifoArray = try MLMultiArray(
            shape: [1, NSNumber(value: fifoLen), NSNumber(value: fcDModel)],
            dataType: .float32
        )
        let fifoPtr = fifoArray.dataPointer.bindMemory(to: Float32.self, capacity: fifoLen * fcDModel)
        for i in 0..<min(fifo.count, fifoLen * fcDModel) {
            fifoPtr[i] = Float32(fifo[i])
        }

        let fifoLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        fifoLengthArray[0] = NSNumber(value: Int32(fifoLength))

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
        guard let predsArray = output.featureValue(for: "speaker_preds")?.multiArrayValue,
            let chunkEmbsArray = output.featureValue(for: "chunk_pre_encoder_embs")?.multiArrayValue,
            let chunkEmbLenArray = output.featureValue(for: "chunk_pre_encoder_lengths")?.multiArrayValue
        else {
            throw SortformerError.inferenceFailed("Missing model outputs")
        }

        // Convert predictions to flat array
        var predictions: [Float] = []
        for i in 0..<predsArray.count {
            predictions.append(predsArray[i].floatValue)
        }

        // Convert chunk embeddings to flat array
        var chunkEmbeddings: [Float] = []
        for i in 0..<chunkEmbsArray.count {
            chunkEmbeddings.append(chunkEmbsArray[i].floatValue)
        }

        let chunkEmbLength = chunkEmbLenArray[0].intValue

        return MainModelOutput(
            predictions: predictions,
            chunkEmbeddings: chunkEmbeddings,
            chunkEmbeddingLength: chunkEmbLength
        )
    }

    /// Run separate PreEncoder + Head models (matches Python behavior).
    ///
    /// This approach avoids combined pipeline issues where embeddings might
    /// get corrupted during the pipeline merge process.
    private func runSeparateModels(
        chunk: [Float],
        chunkLength: Int,
        spkcache: [Float],
        spkcacheLength: Int,
        fifo: [Float],
        fifoLength: Int,
        config: SortformerConfig
    ) throws -> MainModelOutput {
        let chunkFrames = config.chunkFrames
        let spkcacheLen = config.spkcacheLen
        let fifoLen = config.fifoLen
        let fcDModel = config.fcDModel
        let melFeatures = config.melFeatures

        // Create chunk input [1, chunkFrames, melFeatures]
        // IMPORTANT: Zero-fill first to ensure padding frames are zeros
        let chunkArray = try MLMultiArray(
            shape: [1, NSNumber(value: chunkFrames), NSNumber(value: melFeatures)],
            dataType: .float32
        )
        let chunkPtr = chunkArray.dataPointer.bindMemory(to: Float32.self, capacity: chunkFrames * melFeatures)
        // Zero-fill first
        for i in 0..<(chunkFrames * melFeatures) {
            chunkPtr[i] = 0.0
        }
        // Then copy actual data
        for i in 0..<min(chunk.count, chunkFrames * melFeatures) {
            chunkPtr[i] = Float32(chunk[i])
        }

        let chunkLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        chunkLengthArray[0] = NSNumber(value: Int32(chunkLength))

        // Create spkcache input [1, spkcacheLen, fcDModel]
        // IMPORTANT: Initialize to zeros first (MLMultiArray doesn't zero-fill by default)
        let spkcacheArray = try MLMultiArray(
            shape: [1, NSNumber(value: spkcacheLen), NSNumber(value: fcDModel)],
            dataType: .float32
        )
        let spkcachePtr = spkcacheArray.dataPointer.bindMemory(to: Float32.self, capacity: spkcacheLen * fcDModel)
        // Zero-fill first
        for i in 0..<(spkcacheLen * fcDModel) {
            spkcachePtr[i] = 0.0
        }
        // Then copy actual data
        for i in 0..<min(spkcache.count, spkcacheLen * fcDModel) {
            spkcachePtr[i] = Float32(spkcache[i])
        }

        let spkcacheLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        spkcacheLengthArray[0] = NSNumber(value: Int32(spkcacheLength))

        // Create fifo input [1, fifoLen, fcDModel]
        // IMPORTANT: Initialize to zeros first
        let fifoArray = try MLMultiArray(
            shape: [1, NSNumber(value: fifoLen), NSNumber(value: fcDModel)],
            dataType: .float32
        )
        let fifoPtr = fifoArray.dataPointer.bindMemory(to: Float32.self, capacity: fifoLen * fcDModel)
        // Zero-fill first
        for i in 0..<(fifoLen * fcDModel) {
            fifoPtr[i] = 0.0
        }
        // Then copy actual data
        for i in 0..<min(fifo.count, fifoLen * fcDModel) {
            fifoPtr[i] = Float32(fifo[i])
        }

        let fifoLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        fifoLengthArray[0] = NSNumber(value: Int32(fifoLength))

        // Step 1: Run PreEncoder
        let preEncoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunkArray),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLengthArray),
            "spkcache": MLFeatureValue(multiArray: spkcacheArray),
            "spkcache_lengths": MLFeatureValue(multiArray: spkcacheLengthArray),
            "fifo": MLFeatureValue(multiArray: fifoArray),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLengthArray),
        ])

        let preEncoderOutput = try preEncoderModel.prediction(from: preEncoderInput)

        // Extract PreEncoder outputs
        // Note: NVIDIA model uses chunk_embs_in/chunk_lens_in, low-latency uses chunk_pre_encoder_embs/chunk_pre_encoder_lengths
        guard let preEncoderEmbs = preEncoderOutput.featureValue(for: "pre_encoder_embs")?.multiArrayValue,
            let preEncoderLengths = preEncoderOutput.featureValue(for: "pre_encoder_lengths")?.multiArrayValue
        else {
            throw SortformerError.inferenceFailed("Missing PreEncoder embs/lengths outputs")
        }

        // Try both naming conventions for chunk embeddings
        let chunkEmbsIn: MLMultiArray
        let chunkLensIn: MLMultiArray
        if let nvidia = preEncoderOutput.featureValue(for: "chunk_embs_in")?.multiArrayValue,
            let nvidiaLens = preEncoderOutput.featureValue(for: "chunk_lens_in")?.multiArrayValue
        {
            chunkEmbsIn = nvidia
            chunkLensIn = nvidiaLens
        } else if let lowLatency = preEncoderOutput.featureValue(for: "chunk_pre_encoder_embs")?.multiArrayValue,
            let lowLatencyLens = preEncoderOutput.featureValue(for: "chunk_pre_encoder_lengths")?.multiArrayValue
        {
            chunkEmbsIn = lowLatency
            chunkLensIn = lowLatencyLens
        } else {
            throw SortformerError.inferenceFailed("Missing PreEncoder chunk outputs")
        }

        // Step 2: Run Head
        // NVIDIA model uses pre_encoder_embs/pre_encoder_lengths/chunk_embs_in/chunk_lens_in
        // Low-latency model uses concat_embs/concat_lens/chunk_embs/chunk_lens
        let headInput: MLDictionaryFeatureProvider

        // Check which naming convention to use by seeing if NVIDIA names work
        // NVIDIA model has pre_encoder_embs input, low-latency has concat_embs
        let headInputNames = headModel.modelDescription.inputDescriptionsByName.keys
        if headInputNames.contains("pre_encoder_embs") {
            // NVIDIA naming
            headInput = try MLDictionaryFeatureProvider(dictionary: [
                "pre_encoder_embs": MLFeatureValue(multiArray: preEncoderEmbs),
                "pre_encoder_lengths": MLFeatureValue(multiArray: preEncoderLengths),
                "chunk_embs_in": MLFeatureValue(multiArray: chunkEmbsIn),
                "chunk_lens_in": MLFeatureValue(multiArray: chunkLensIn),
            ])
        } else {
            // Low-latency naming
            headInput = try MLDictionaryFeatureProvider(dictionary: [
                "concat_embs": MLFeatureValue(multiArray: preEncoderEmbs),
                "concat_lens": MLFeatureValue(multiArray: preEncoderLengths),
                "chunk_embs": MLFeatureValue(multiArray: chunkEmbsIn),
                "chunk_lens": MLFeatureValue(multiArray: chunkLensIn),
            ])
        }

        let headOutput = try headModel.prediction(from: headInput)

        // Extract Head outputs - speaker_preds is common
        guard let predsArray = headOutput.featureValue(for: "speaker_preds")?.multiArrayValue else {
            throw SortformerError.inferenceFailed("Missing speaker_preds output")
        }

        // Try different naming conventions for chunk embeddings output
        let chunkEmbsArray: MLMultiArray
        let chunkEmbLenArray: MLMultiArray
        if let nvidia = headOutput.featureValue(for: "chunk_pre_encoder_embs")?.multiArrayValue,
            let nvidiaLen = headOutput.featureValue(for: "chunk_pre_encoder_lengths")?.multiArrayValue
        {
            chunkEmbsArray = nvidia
            chunkEmbLenArray = nvidiaLen
        } else if let lowLatency = headOutput.featureValue(for: "output_chunk_embs")?.multiArrayValue,
            let lowLatencyLen = headOutput.featureValue(for: "output_chunk_lens")?.multiArrayValue
        {
            chunkEmbsArray = lowLatency
            chunkEmbLenArray = lowLatencyLen
        } else {
            throw SortformerError.inferenceFailed("Missing Head chunk embedding outputs")
        }

        // Convert predictions to flat array
        var predictions: [Float] = []
        for i in 0..<predsArray.count {
            predictions.append(predsArray[i].floatValue)
        }

        // Convert chunk embeddings to flat array
        var chunkEmbeddings: [Float] = []
        for i in 0..<chunkEmbsArray.count {
            chunkEmbeddings.append(chunkEmbsArray[i].floatValue)
        }

        let chunkEmbLength = chunkEmbLenArray[0].intValue

        return MainModelOutput(
            predictions: predictions,
            chunkEmbeddings: chunkEmbeddings,
            chunkEmbeddingLength: chunkEmbLength
        )
    }
}
