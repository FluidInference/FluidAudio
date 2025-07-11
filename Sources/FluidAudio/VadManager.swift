import CoreML
import Foundation
import OSLog
import Accelerate

/// VAD model types
public enum VADModelType: Sendable {
    case coreML          // CoreML-based pipeline (STFT + Encoder + RNN + Enhanced Fallback)
}

/// Configuration for VAD processing
public struct VADConfig: Sendable {
    public var threshold: Float = 0.3  // Voice activity threshold (0.0-1.0) - lowered for better sensitivity
    public var chunkSize: Int = 512   // Audio chunk size for processing
    public var sampleRate: Int = 16000 // Sample rate for audio processing
    public var modelCacheDirectory: URL?
    public var debugMode: Bool = true
    public var adaptiveThreshold: Bool = false  // Disable adaptive thresholding temporarily
    public var minThreshold: Float = 0.1       // Minimum threshold for adaptive mode
    public var maxThreshold: Float = 0.7       // Maximum threshold for adaptive mode
    public var modelType: VADModelType = .coreML  // Use CoreML for Mac
    public var useGPU: Bool = true             // Use Metal Performance Shaders on Mac

    // SNR and noise detection parameters
    public var enableSNRFiltering: Bool = true      // Enable SNR-based filtering for better noise rejection
    public var minSNRThreshold: Float = 6.0         // Minimum SNR for speech detection (dB) - more aggressive
    public var noiseFloorWindow: Int = 100          // Window size for noise floor estimation
    public var spectralRolloffThreshold: Float = 0.85  // Threshold for spectral rolloff
    public var spectralCentroidRange: (min: Float, max: Float) = (200.0, 8000.0)  // Expected speech range (Hz)

    public static let `default` = VADConfig()

    public init(
        threshold: Float = 0.3,
        chunkSize: Int = 512,
        sampleRate: Int = 16000,
        modelCacheDirectory: URL? = nil,
        debugMode: Bool = false,
        adaptiveThreshold: Bool = false,
        minThreshold: Float = 0.1,
        maxThreshold: Float = 0.7,
        modelType: VADModelType = .coreML,
        useGPU: Bool = true,
        enableSNRFiltering: Bool = true,
        minSNRThreshold: Float = 6.0,
        noiseFloorWindow: Int = 100,
        spectralRolloffThreshold: Float = 0.85,
        spectralCentroidRange: (min: Float, max: Float) = (200.0, 8000.0)
    ) {
        self.threshold = threshold
        self.chunkSize = chunkSize
        self.sampleRate = sampleRate
        self.modelCacheDirectory = modelCacheDirectory
        self.debugMode = debugMode
        self.adaptiveThreshold = adaptiveThreshold
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.modelType = modelType
        self.useGPU = useGPU
        self.enableSNRFiltering = enableSNRFiltering
        self.minSNRThreshold = minSNRThreshold
        self.noiseFloorWindow = noiseFloorWindow
        self.spectralRolloffThreshold = spectralRolloffThreshold
        self.spectralCentroidRange = spectralCentroidRange
    }
}

/// VAD processing result
public struct VADResult: Sendable {
    public let probability: Float  // Voice activity probability (0.0-1.0)
    public let isVoiceActive: Bool // Whether voice is detected
    public let processingTime: TimeInterval
    public let snrValue: Float?    // Signal-to-Noise Ratio (dB) if calculated
    public let spectralFeatures: SpectralFeatures?  // Spectral analysis results

    public init(probability: Float, isVoiceActive: Bool, processingTime: TimeInterval, snrValue: Float? = nil, spectralFeatures: SpectralFeatures? = nil) {
        self.probability = probability
        self.isVoiceActive = isVoiceActive
        self.processingTime = processingTime
        self.snrValue = snrValue
        self.spectralFeatures = spectralFeatures
    }
}

/// Spectral features for enhanced VAD
public struct SpectralFeatures: Sendable {
    public let spectralCentroid: Float      // Center frequency of the spectrum
    public let spectralRolloff: Float       // Frequency below which 85% of energy is contained
    public let spectralFlux: Float          // Measure of spectral change
    public let mfccFeatures: [Float]        // MFCC coefficients (first 13)
    public let zeroCrossingRate: Float      // Zero crossing rate
    public let spectralEntropy: Float       // Measure of spectral complexity

    public init(spectralCentroid: Float, spectralRolloff: Float, spectralFlux: Float, mfccFeatures: [Float], zeroCrossingRate: Float, spectralEntropy: Float) {
        self.spectralCentroid = spectralCentroid
        self.spectralRolloff = spectralRolloff
        self.spectralFlux = spectralFlux
        self.mfccFeatures = mfccFeatures
        self.zeroCrossingRate = zeroCrossingRate
        self.spectralEntropy = spectralEntropy
    }
}

/// VAD error types
public enum VADError: Error, LocalizedError {
    case notInitialized
    case modelLoadingFailed
    case modelProcessingFailed(String)
    case invalidAudioData
    case invalidModelPath
    case modelDownloadFailed
    case modelCompilationFailed

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "VAD system not initialized. Call initialize() first."
        case .modelLoadingFailed:
            return "Failed to load VAD models."
        case .modelProcessingFailed(let message):
            return "Model processing failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidModelPath:
            return "Invalid model path provided."
        case .modelDownloadFailed:
            return "Failed to download VAD models from Hugging Face."
        case .modelCompilationFailed:
            return "Failed to compile VAD models after multiple attempts."
        }
    }
}

/// Voice Activity Detection Manager using CoreML models
@available(macOS 13.0, iOS 16.0, *)
public actor VADManager {

    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "VAD")
    private let config: VADConfig

    // CoreML models for VAD pipeline (isolated to actor)
    private var stftModel: MLModel?
    private var encoderModel: MLModel?
    private var rnnModel: MLModel?

    // RNN state management (isolated to actor)
    private var hState: MLMultiArray?
    private var cState: MLMultiArray?
    private var featureBuffer: [MLMultiArray] = []

    // Audio processing handler
    private var audioProcessor: VADAudioProcessor

    // Timing tracking
    private var modelLoadTime: TimeInterval = 0

    public init(config: VADConfig = .default) {
        self.config = config
        self.audioProcessor = VADAudioProcessor(config: config)
    }

    public var isAvailable: Bool {
        return stftModel != nil && encoderModel != nil && rnnModel != nil
    }

    /// Initialize VAD system with selected model type
    public func initialize() async throws {
        let initStartTime = Date()
        logger.info("Initializing VAD system with CoreML")

        let loadStartTime = Date()

        try await loadCoreMLModels()

        self.modelLoadTime = Date().timeIntervalSince(loadStartTime)

        // Initialize states
        resetState()

        let totalInitTime = Date().timeIntervalSince(initStartTime)
        logger.info(
            "VAD system initialized successfully in \(String(format: "%.2f", totalInitTime))s (model loading: \(String(format: "%.2f", self.modelLoadTime))s)"
        )
    }

    /// Enhanced model management: check compiled models exist, auto-download if missing
    private func loadCoreMLModels() async throws {
        let modelsDirectory = getModelsDirectory()
        let compiledModelNames = ["silero_stft.mlmodelc", "silero_encoder.mlmodelc", "silero_rnn_decoder.mlmodelc"]

        // Check if we need to download any compiled models using utility class
        let modelChecker = ModelFileChecker()
        let missingModels = try modelChecker.checkModelFiles(in: modelsDirectory, modelNames: compiledModelNames)

        // Auto-download missing models
        if !missingModels.isEmpty {
            logger.info("🔄 Missing VAD models: \(missingModels.joined(separator: ", "))")
            logger.info("🔄 Auto-downloading VAD models from Hugging Face...")
            try await downloadMissingVADModels()
        }

        // Verify all models are now present
        for modelName in compiledModelNames {
            let modelPath = modelsDirectory.appendingPathComponent(modelName)
            if !FileManager.default.fileExists(atPath: modelPath.path) {
                logger.error("❌ Missing compiled model after download: \(modelName)")
                throw VADError.modelLoadingFailed
            }
        }

        // Get paths for compiled models
        logger.info("🔍 Loading pre-compiled VAD models...")
        let pathStart = Date()
        let stftPath = modelsDirectory.appendingPathComponent("silero_stft.mlmodelc")
        let encoderPath = modelsDirectory.appendingPathComponent("silero_encoder.mlmodelc")
        let rnnPath = modelsDirectory.appendingPathComponent("silero_rnn_decoder.mlmodelc")
        logger.info("✓ Model paths resolved in \(String(format: "%.2f", Date().timeIntervalSince(pathStart)))s")

        // Load models with auto-recovery mechanism (similar to DiarizerManager)
        try await loadModelsWithAutoRecovery(
            stftPath: stftPath,
            encoderPath: encoderPath,
            rnnPath: rnnPath
        )
    }

    /// Load models with automatic recovery on compilation failures (based on DiarizerManager pattern)
    private func loadModelsWithAutoRecovery(
        stftPath: URL, encoderPath: URL, rnnPath: URL, maxRetries: Int = 2
    ) async throws {
        var attempt = 0

        while attempt <= maxRetries {
            do {
                logger.info("Attempting to load VAD models (attempt \(attempt + 1)/\(maxRetries + 1))")

                // Try GPU first, then CPU fallback
                for useGPU in [true, false] {
                    let config = MLModelConfiguration()
                    config.computeUnits = useGPU ? .all : .cpuOnly
                    let computeType = useGPU ? "All (CPU+GPU+Neural Engine)" : "CPU-only"

                    do {
                        logger.info("🔄 Loading with \(computeType)...")

                        let stftStart = Date()
                        let stftModel = try MLModel(contentsOf: stftPath, configuration: config)
                        logger.info("✓ STFT model loaded (\(String(format: "%.2f", Date().timeIntervalSince(stftStart)))s)")

                        let encoderStart = Date()
                        let encoderModel = try MLModel(contentsOf: encoderPath, configuration: config)
                        logger.info("✓ Encoder model loaded (\(String(format: "%.2f", Date().timeIntervalSince(encoderStart)))s)")

                        let rnnStart = Date()
                        let rnnModel = try MLModel(contentsOf: rnnPath, configuration: config)
                        logger.info("✓ RNN model loaded (\(String(format: "%.2f", Date().timeIntervalSince(rnnStart)))s)")

                        // If we get here, all models loaded successfully
                        self.stftModel = stftModel
                        self.encoderModel = encoderModel
                        self.rnnModel = rnnModel

                        if attempt > 0 {
                            logger.info("Models loaded successfully after \(attempt) recovery attempt(s)")
                        }
                        logger.info("🎉 VAD models loaded successfully with \(computeType)")
                        return

                    } catch {
                        if !useGPU {
                            throw error // Rethrow CPU failure - will trigger recovery
                        }
                        logger.warning("⚠️ \(computeType) loading failed, trying CPU-only...")
                    }
                }

            } catch {
                logger.warning("Model loading failed (attempt \(attempt + 1)): \(error.localizedDescription)")

                // If this is our last attempt, throw the error
                if attempt >= maxRetries {
                    logger.error("Model loading failed after \(maxRetries + 1) attempts, giving up")
                    throw VADError.modelCompilationFailed
                }

                // Auto-recovery: Delete corrupted models and re-download
                logger.info("Initiating auto-recovery: removing corrupted models and re-downloading...")
                try await performVADModelRecovery(stftPath: stftPath, encoderPath: encoderPath, rnnPath: rnnPath)

                attempt += 1
            }
        }
    }

    /// Perform model recovery by deleting corrupted compiled models and re-downloading
    private func performVADModelRecovery(stftPath: URL, encoderPath: URL, rnnPath: URL) async throws {
        let modelPaths = [stftPath, encoderPath, rnnPath]

        // Remove potentially corrupted compiled model files
        for modelPath in modelPaths {
            if FileManager.default.fileExists(atPath: modelPath.path) {
                logger.info("Removing corrupted compiled model at \(modelPath.lastPathComponent)")
                try FileManager.default.removeItem(at: modelPath)
            }
        }

        // Re-download models from Hugging Face
        logger.info("🔄 Re-downloading VAD models from Hugging Face...")
        try await downloadMissingVADModels()
    }

    /// Download missing VAD models from Hugging Face
    private func downloadMissingVADModels() async throws {
        let modelsDirectory = getModelsDirectory()

        // Download .mlmodelc folders from Hugging Face
        let modelFolders = ["silero_stft.mlmodelc", "silero_encoder.mlmodelc", "silero_rnn_decoder.mlmodelc"]

        for folderName in modelFolders {
            let folderPath = modelsDirectory.appendingPathComponent(folderName)

            if !FileManager.default.fileExists(atPath: folderPath.path) {
                logger.info("📥 Downloading \(folderName)...")
                print("📥 Downloading \(folderName)...")
                try await downloadVADModelFolder(folderName: folderName, to: folderPath)
            }
        }

        logger.info("✅ VAD models downloaded successfully")
    }

    /// Download a .mlmodelc folder from Hugging Face
    private func downloadVADModelFolder(folderName: String, to folderPath: URL) async throws {
        // Create the folder
        try FileManager.default.createDirectory(at: folderPath, withIntermediateDirectories: true)

        // Download the main files inside the .mlmodelc folder
        let modelFiles = [
            "coremldata.bin",
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights"
        ]

        let baseURL = "https://huggingface.co/alexwengg/coreml-silero-vad/resolve/main/\(folderName)"

        for fileName in modelFiles {
            let fileURL = "\(baseURL)/\(fileName)"
            let destinationPath = folderPath.appendingPathComponent(fileName)

            try await downloadVADModelFile(from: fileURL, to: destinationPath)
        }

        // Download subdirectory files required by CoreML
        let subDirs = ["analytics", "model", "neural_network_optionals"]
        for subDir in subDirs {
            let subDirPath = folderPath.appendingPathComponent(subDir)
            try FileManager.default.createDirectory(at: subDirPath, withIntermediateDirectories: true)

            // Download the coremldata.bin file in each subdirectory
            let subFileURL = "\(baseURL)/\(subDir)/coremldata.bin"
            let subDestinationPath = subDirPath.appendingPathComponent("coremldata.bin")

            do {
                try await downloadVADModelFile(from: subFileURL, to: subDestinationPath)
            } catch {
                logger.warning("⚠️ Failed to download \(subDir)/coremldata.bin for \(folderName): \(error)")
                // Some subdirectory files might be optional
            }
        }
    }

    /// Download a model file from URL
    private func downloadVADModelFile(from urlString: String, to destinationPath: URL) async throws {
        guard let url = URL(string: urlString) else {
            logger.error("Invalid URL: \(urlString)")
            throw VADError.modelDownloadFailed
        }

        logger.info("🔄 Downloading from: \(urlString)")

        do {
            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                logger.error("Invalid response type for \(urlString)")
                throw VADError.modelDownloadFailed
            }

            guard httpResponse.statusCode == 200 else {
                logger.error("HTTP Error \(httpResponse.statusCode) for \(urlString)")
                throw VADError.modelDownloadFailed
            }

            try data.write(to: destinationPath)
            logger.info("✅ Downloaded \(destinationPath.lastPathComponent) (\(data.count) bytes)")
        } catch {
            logger.error("Download failed for \(urlString): \(error.localizedDescription)")
            throw VADError.modelDownloadFailed
        }
    }


    /// Reset RNN state and feature buffer
    public func resetState() {
        do {
            // Initialize CoreML RNN states with shape (1, 1, 128) - explicitly set to zero
            self.hState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)
            self.cState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)

            // Explicitly initialize to zero (MLMultiArray may contain random values)
            if let hState = self.hState, let cState = self.cState {
                for i in 0..<hState.count {
                    hState[i] = 0.0
                }
                for i in 0..<cState.count {
                    cState[i] = 0.0
                }
            }

            // Clear feature buffer
            self.featureBuffer.removeAll()

        } catch {
            logger.error("Failed to reset CoreML VAD state: \(error.localizedDescription)")
        }

        // Reset audio processor state
        audioProcessor.reset()

        if config.debugMode {
            logger.debug("VAD state reset successfully for CoreML")
        }
    }

    /// Process audio chunk and return VAD probability
    public func processChunk(_ audioChunk: [Float]) async throws -> VADResult {
        guard isAvailable else {
            throw VADError.notInitialized
        }

        let processingStartTime = Date()

        let rawProbability: Float

        rawProbability = try await processCoreMLChunk(audioChunk)

        // Post-process raw ML probability
        let (enhancedProbability, snrValue, spectralFeatures) = audioProcessor.processRawProbability(
            rawProbability,
            audioChunk: audioChunk
        )

        // Apply temporal smoothing
        let smoothedProbability = audioProcessor.applySmoothingFilter(enhancedProbability)

        // Apply fixed threshold
        let isVoiceActive = smoothedProbability >= config.threshold

        let processingTime = Date().timeIntervalSince(processingStartTime)

        if config.debugMode {
            let snrString = snrValue.map { String(format: "%.1f", $0) } ?? "N/A"
            let debugMessage = "VAD processing (CoreML): raw=\(String(format: "%.3f", rawProbability)), enhanced=\(String(format: "%.3f", enhancedProbability)), smoothed=\(String(format: "%.3f", smoothedProbability)), threshold=\(String(format: "%.3f", config.threshold)), snr=\(snrString)dB, active=\(isVoiceActive), time=\(String(format: "%.3f", processingTime))s"
            logger.debug("\(debugMessage)")
        }

        return VADResult(
            probability: smoothedProbability,
            isVoiceActive: isVoiceActive,
            processingTime: processingTime,
            snrValue: snrValue,
            spectralFeatures: spectralFeatures
        )
    }

    /// Process audio chunk using CoreML pipeline (STFT + Encoder + RNN + Fallback)
    private func processCoreMLChunk(_ audioChunk: [Float]) async throws -> Float {
        // Ensure correct audio shape
        var processedChunk = audioChunk
        if processedChunk.count != config.chunkSize {
            // Pad or truncate to expected size
            if processedChunk.count < config.chunkSize {
                let paddingSize = config.chunkSize - processedChunk.count
                if config.debugMode {
                    logger.debug("Padding audio chunk with \(paddingSize) zeros (original: \(processedChunk.count) samples)")
                }
                processedChunk.append(contentsOf: Array(repeating: 0.0, count: paddingSize))
            } else {
                if config.debugMode {
                    logger.debug("Truncating audio chunk from \(processedChunk.count) to \(self.config.chunkSize) samples")
                }
                processedChunk = Array(processedChunk.prefix(config.chunkSize))
            }
        }

        // Step 1: STFT processing
        let stftFeatures = try processSTFT(processedChunk)

        // Step 2: Manage temporal context
        manageTemporalContext(stftFeatures)

        // Step 3: Encoder processing
        let encoderFeatures = try processEncoder()

        // Step 4: RNN processing
        let rnnFeatures = try processRNN(encoderFeatures)

        // Step 5: Calculate VAD probability from RNN features
        return audioProcessor.calculateVADProbability(from: rnnFeatures)
    }

    /// Process audio through STFT model
    /// Input: audio_input - shape (1, 512) - raw audio samples
    /// Output: stft_features - shape (1, 201) - STFT features
    private func processSTFT(_ audioChunk: [Float]) throws -> MLMultiArray {
        guard let stftModel = self.stftModel else {
            throw VADError.notInitialized
        }

        // Create input array with shape (1, chunkSize=512)
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: config.chunkSize)], dataType: .float32)

        for i in 0..<audioChunk.count {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio_input": audioArray])
        let output = try stftModel.prediction(from: input)

        // Try to use specific named output first
        let preferredOutputNames = ["stft_features", "features", "output"]

        for outputName in preferredOutputNames {
            if let stftOutput = output.featureValue(for: outputName)?.multiArrayValue {
                if config.debugMode {
                    let shape = stftOutput.shape.map { $0.intValue }
                    logger.debug("STFT output '\(outputName)' shape: \(shape)")
                }
                return stftOutput
            }
        }

        // Fallback: use first available output
        guard let outputName = output.featureNames.first,
              let stftOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VADError.modelProcessingFailed("No STFT output found")
        }

        if config.debugMode {
            let shape = stftOutput.shape.map { $0.intValue }
            logger.debug("STFT fallback output '\(outputName)' shape: \(shape)")
        }

        return stftOutput
    }

    /// Manage temporal context buffer
    private func manageTemporalContext(_ stftFeatures: MLMultiArray) {
        // Add current features to buffer
        featureBuffer.append(stftFeatures)

        // Keep only the last 4 frames for temporal context
        if featureBuffer.count > 4 {
            featureBuffer = Array(featureBuffer.suffix(4))
        }

        // Pad with zeros if we have less than 4 frames
        while featureBuffer.count < 4 {
            do {
                let zeroFeatures = try MLMultiArray(shape: stftFeatures.shape, dataType: .float32)
                // MLMultiArray is initialized with zeros by default
                featureBuffer.insert(zeroFeatures, at: 0)
            } catch {
                logger.error("Failed to create zero features for temporal context: \(error)")
                break
            }
        }
    }

    /// Process features through encoder
    /// Input: stft_features - shape (1, 201, 4) - concatenated temporal STFT features
    /// Output: encoder_features - shape (1, 128) - encoded features
    private func processEncoder() throws -> MLMultiArray {
        guard let encoderModel = self.encoderModel else {
            throw VADError.notInitialized
        }

        guard !featureBuffer.isEmpty else {
            throw VADError.modelProcessingFailed("No features in buffer")
        }

        // Concatenate all 4 frames in the buffer to create shape (1, 201, 4)
        let concatenatedFeatures = try concatenateTemporalFeatures()

        if config.debugMode {
            let shape = concatenatedFeatures.shape.map { $0.intValue }
            logger.debug("Encoder input shape: \(shape)")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["stft_features": concatenatedFeatures])
        let output = try encoderModel.prediction(from: input)

        // Try to use specific named output first (more robust than shape-based detection)
        let preferredOutputNames = ["encoder_output", "encoded_features", "output"]

        for outputName in preferredOutputNames {
            if let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue {
                if config.debugMode {
                    let shape = encoderOutput.shape.map { $0.intValue }
                    logger.debug("Encoder output '\(outputName)' shape: \(shape)")
                }
                return encoderOutput
            }
        }

        // Fallback: use first available output with warning
        if config.debugMode {
            let availableOutputs = output.featureNames.joined(separator: ", ")
            logger.warning("None of preferred output names found. Available outputs: [\(availableOutputs)]. Using first available.")
        }

        guard let outputName = output.featureNames.first,
              let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VADError.modelProcessingFailed("No encoder output found")
        }

        if config.debugMode {
            let shape = encoderOutput.shape.map { $0.intValue }
            logger.debug("Encoder fallback output '\(outputName)' shape: \(shape)")
        }

        return encoderOutput
    }

    /// Process features through RNN decoder
    /// Inputs:
    ///   - encoder_features: shape (1, 4, 128) - encoded features
    ///   - h_in: shape (1, 1, 128) - LSTM hidden state
    ///   - c_in: shape (1, 1, 128) - LSTM cell state
    /// Outputs:
    ///   - rnn_features: shape (1, 4, 128) - RNN output features
    ///   - h_out: shape (1, 1, 128) - updated LSTM hidden state
    ///   - c_out: shape (1, 1, 128) - updated LSTM cell state
    private func processRNN(_ encoderFeatures: MLMultiArray) throws -> MLMultiArray {
        guard let rnnModel = self.rnnModel else {
            throw VADError.notInitialized
        }

        // Initialize states if they don't exist (more robust than throwing)
        if self.hState == nil || self.cState == nil {
            logger.warning("RNN states not initialized, initializing to zero")
            resetState()
        }

        guard let hState = self.hState, let cState = self.cState else {
            throw VADError.modelProcessingFailed("Failed to initialize RNN states")
        }

        // Prepare encoder features for RNN (reshape if needed)
        let processedFeatures = try prepareEncoderFeaturesForRNN(encoderFeatures)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_features": processedFeatures,
            "h_in": hState,
            "c_in": cState
        ])

        let output = try rnnModel.prediction(from: input)

        // Extract RNN outputs and update states
        var rnnFeatures: MLMultiArray?

        for featureName in output.featureNames {
            if let featureValue = output.featureValue(for: featureName)?.multiArrayValue {
                let shape = featureValue.shape.map { $0.intValue }

                if shape.count == 3 {
                    if shape[1] > 1 {
                        // This is likely the sequence output
                        rnnFeatures = featureValue
                    } else if shape == [1, 1, 128] {
                        // This is a state output - update our states
                        if featureName.contains("h") || self.hState == nil {
                            self.hState = featureValue
                        } else if featureName.contains("c") || self.cState == nil {
                            self.cState = featureValue
                        }
                    }
                }
            }
        }

        guard let finalRnnFeatures = rnnFeatures else {
            throw VADError.modelProcessingFailed("No RNN sequence output found")
        }

        return finalRnnFeatures
    }

    /// Prepare encoder features for RNN input
    private func prepareEncoderFeaturesForRNN(_ encoderFeatures: MLMultiArray) throws -> MLMultiArray {
        let shape = encoderFeatures.shape.map { $0.intValue }

        // If already in correct shape (1, 4, 128), return as is
        if shape.count == 3 && shape[0] == 1 && shape[1] == 4 && shape[2] == 128 {
            return encoderFeatures
        }

        // Log unexpected shape for debugging
        if config.debugMode {
            logger.debug("Encoder features have unexpected shape: \(shape), reshaping to (1, 4, 128)")
        }

        // Create target shape (1, 4, 128)
        let targetArray = try MLMultiArray(shape: [1, 4, 128], dataType: .float32)

        // Handle different input shapes - this suggests a model architecture mismatch
        if shape.count == 2 && shape[0] == 1 {
            // Encoder output is (1, N) - expand to (1, 4, 128) by replicating
            let featureSize = min(shape[1], 128)
            for timeStep in 0..<4 {
                for feature in 0..<featureSize {
                    let sourceIndex = feature
                    let targetIndex = timeStep * 128 + feature
                    if sourceIndex < encoderFeatures.count && targetIndex < targetArray.count {
                        targetArray[targetIndex] = encoderFeatures[sourceIndex]
                    }
                }
            }
        } else {
            // Fallback: linear copy (may not be semantically correct)
            logger.warning("Unexpected encoder shape \(shape), using linear copy fallback")
            let sourceElements = min(encoderFeatures.count, targetArray.count)
            for i in 0..<sourceElements {
                targetArray[i] = encoderFeatures[i]
            }
        }

        return targetArray
    }
    /// Concatenate temporal features from buffer to create shape (1, 201, 4)
    private func concatenateTemporalFeatures() throws -> MLMultiArray {
        guard featureBuffer.count == 4 else {
            throw VADError.modelProcessingFailed("Feature buffer must contain exactly 4 frames")
        }

        // Get the shape of individual features (should be [1, 201])
        let singleShape = featureBuffer[0].shape.map { $0.intValue }
        guard singleShape.count >= 2 else {
            throw VADError.modelProcessingFailed("Invalid feature shape")
        }

        let batchSize = singleShape[0]
        let featureSize = singleShape[1]
        let temporalFrames = 4

        // Create concatenated array with shape (1, 201, 4)
        let concatenatedArray = try MLMultiArray(
            shape: [NSNumber(value: batchSize), NSNumber(value: featureSize), NSNumber(value: temporalFrames)],
            dataType: .float32
        )

        // Copy each frame's features into the corresponding slice
        for frameIndex in 0..<temporalFrames {
            let frameFeatures = featureBuffer[frameIndex]

            for i in 0..<batchSize {
                for j in 0..<featureSize {
                    let sourceIndex = i * featureSize + j
                    let targetIndex = i * (featureSize * temporalFrames) + j * temporalFrames + frameIndex
                    concatenatedArray[targetIndex] = frameFeatures[sourceIndex]
                }
            }
        }

        return concatenatedArray
    }

    /// Get models directory
    private func getModelsDirectory() -> URL {
        let directory: URL

        if let customDirectory = config.modelCacheDirectory {
            directory = customDirectory.appendingPathComponent("vad", isDirectory: true)
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            directory = appSupport.appendingPathComponent("FluidAudio/vad", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    /// Process audio file and return VAD results
    public func processAudioFile(_ audioData: [Float]) async throws -> [VADResult] {
        guard isAvailable else {
            throw VADError.notInitialized
        }

        resetState()

        var results: [VADResult] = []
        let chunkSize = config.chunkSize

        // Process in chunks
        for chunkStart in stride(from: 0, to: audioData.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, audioData.count)
            let chunk = Array(audioData[chunkStart..<chunkEnd])

            let result = try await processChunk(chunk)
            results.append(result)
        }
        return results
    }

    /// Cleanup resources
    public func cleanup() {
        stftModel = nil
        encoderModel = nil
        rnnModel = nil
        hState = nil
        cState = nil
        featureBuffer.removeAll()

        // Reset audio processor
        audioProcessor.reset()

        logger.info("VAD resources cleaned up")
    }
}
