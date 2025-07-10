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
    public var debugMode: Bool = false
    public var adaptiveThreshold: Bool = true  // Enable adaptive thresholding
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
        adaptiveThreshold: Bool = true,
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
    private var classifierModel: MLModel?

    // RNN state management (isolated to actor)
    private var hState: MLMultiArray?
    private var cState: MLMultiArray?
    private var featureBuffer: [MLMultiArray] = []

    // Adaptive thresholding (isolated to actor)
    private var probabilityHistory: [Float] = []
    private var adaptiveThreshold: Float = 0.3

    // Probability smoothing (isolated to actor)
    private var probabilityWindow: [Float] = []
    private let windowSize = 5

    // SNR and noise floor estimation (isolated to actor)
    private var noiseFloorBuffer: [Float] = []
    private var currentNoiseFloor: Float = -60.0  // Initial noise floor estimate (dB)
    private var previousSpectrum: [Float] = []     // For spectral flux calculation

    // Timing tracking
    private var modelLoadTime: TimeInterval = 0

    public init(config: VADConfig = .default) {
        self.config = config
    }

    public var isAvailable: Bool {
        return stftModel != nil && encoderModel != nil && rnnModel != nil // Classifier is optional - using fallback
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

    /// Load CoreML models from the model directory
    private func loadCoreMLModels() async throws {
        let modelsDirectory = getModelsDirectory()
        logger.info("Looking for VAD models in: \(modelsDirectory.path)")

        let stftPath = modelsDirectory.appendingPathComponent("silero_stft.mlmodel")
        let encoderPath = modelsDirectory.appendingPathComponent("silero_encoder.mlmodel")
        let rnnPath = modelsDirectory.appendingPathComponent("silero_rnn_decoder.mlmodel")

        // Load models with auto-recovery mechanism
        try await loadModelsWithAutoRecovery(
            stftPath: stftPath,
            encoderPath: encoderPath,
            rnnPath: rnnPath
        )
    }



    /// Reset RNN state and feature buffer
    public func resetState() {
        do {
            // Initialize CoreML RNN states with shape (1, 1, 128)
            self.hState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)
            self.cState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)

            // Clear feature buffer
            self.featureBuffer.removeAll()

        } catch {
            logger.error("Failed to reset CoreML VAD state: \(error.localizedDescription)")
        }

        // Reset adaptive thresholding state (common to all models)
        self.probabilityHistory.removeAll()
        self.probabilityWindow.removeAll()
        self.adaptiveThreshold = config.threshold

        // Reset SNR and spectral analysis state
        self.noiseFloorBuffer.removeAll()
        self.currentNoiseFloor = -60.0
        self.previousSpectrum.removeAll()

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

        // Calculate SNR and spectral features if enabled
        var snrValue: Float?
        var spectralFeatures: SpectralFeatures?
        var enhancedProbability = rawProbability

        if config.enableSNRFiltering {
            // Calculate spectral features
            spectralFeatures = calculateSpectralFeatures(audioChunk)

            // Calculate SNR
            snrValue = calculateSNR(audioChunk)

            // Apply SNR-based filtering
            enhancedProbability = applyAudioQualityFiltering(
                rawProbability: rawProbability,
                snr: snrValue,
                spectralFeatures: spectralFeatures
            )
        }

        // Apply probability smoothing (common to all models)
        let smoothedProbability = applySmoothingFilter(enhancedProbability)

        // Apply adaptive thresholding (common to all models)
        let effectiveThreshold = updateAdaptiveThreshold(smoothedProbability)
        let isVoiceActive = smoothedProbability >= effectiveThreshold

        let processingTime = Date().timeIntervalSince(processingStartTime)

        if config.debugMode {
            let snrString = snrValue.map { String(format: "%.1f", $0) } ?? "N/A"
            let debugMessage = "VAD processing (CoreML): raw=\(String(format: "%.3f", rawProbability)), enhanced=\(String(format: "%.3f", enhancedProbability)), smoothed=\(String(format: "%.3f", smoothedProbability)), threshold=\(String(format: "%.3f", effectiveThreshold)), snr=\(snrString)dB, active=\(isVoiceActive), time=\(String(format: "%.3f", processingTime))s"
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
                processedChunk.append(contentsOf: Array(repeating: 0.0, count: config.chunkSize - processedChunk.count))
            } else {
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

        // Step 5: Classification using the classifier model or fallback
        return try processClassifier(rnnFeatures)
    }




    /// Fallback VAD processing method (temporary)
    private func processFallbackVAD(_ audioChunk: [Float]) async throws -> Float {
        // Simple fallback based on energy and statistical features
        let energy = audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count)
        let logEnergy = log(max(energy, 1e-10))

        // Normalize energy to probability (0.0 - 1.0)
        // This is a very basic fallback method
        let normalizedEnergy = max(0.0, min(1.0, (logEnergy + 10.0) / 10.0))

        return normalizedEnergy
    }

    /// Process audio through STFT model
    private func processSTFT(_ audioChunk: [Float]) throws -> MLMultiArray {
        guard let stftModel = self.stftModel else {
            throw VADError.notInitialized
        }

        // Create input array with shape (1, chunkSize)
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: config.chunkSize)], dataType: .float32)

        for i in 0..<audioChunk.count {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio_input": audioArray])
        let output = try stftModel.prediction(from: input)

        // Get the first (and likely only) output
        guard let outputName = output.featureNames.first,
              let stftOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VADError.modelProcessingFailed("No STFT output found")
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
    private func processEncoder() throws -> MLMultiArray {
        guard let encoderModel = self.encoderModel else {
            throw VADError.notInitialized
        }

        guard !featureBuffer.isEmpty else {
            throw VADError.modelProcessingFailed("No features in buffer")
        }

        // Concatenate all 4 frames in the buffer to create shape (1, 201, 4)
        let concatenatedFeatures = try concatenateTemporalFeatures()

        let input = try MLDictionaryFeatureProvider(dictionary: ["stft_features": concatenatedFeatures])
        let output = try encoderModel.prediction(from: input)

        guard let outputName = output.featureNames.first,
              let encoderOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VADError.modelProcessingFailed("No encoder output found")
        }

        return encoderOutput
    }

    /// Process features through RNN decoder
    private func processRNN(_ encoderFeatures: MLMultiArray) throws -> MLMultiArray {
        guard let rnnModel = self.rnnModel else {
            throw VADError.notInitialized
        }

        guard let hState = self.hState, let cState = self.cState else {
            throw VADError.modelProcessingFailed("RNN states not initialized")
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

        // Create target shape (1, 4, 128)
        let targetArray = try MLMultiArray(shape: [1, 4, 128], dataType: .float32)

        // Copy data with appropriate reshaping/padding
        let _ = encoderFeatures.strides.map { $0.intValue }
        let _ = targetArray.strides.map { $0.intValue }

        let sourceElements = min(encoderFeatures.count, targetArray.count)

        for i in 0..<sourceElements {
            targetArray[i] = encoderFeatures[i]
        }

        return targetArray
    }

    /// Process features through classifier model
    private func processClassifier(_ rnnFeatures: MLMultiArray) throws -> Float {
        guard let classifierModel = self.classifierModel else {
            // Use fallback calculation when classifier model is not available
            if config.debugMode {
                logger.debug("Using fallback VAD calculation (classifier model not loaded)")
            }
            return calculateVADProbability(from: rnnFeatures)
        }

        // Ensure correct shape for classifier (1, 4, 128)
        let processedFeatures = try prepareRNNFeaturesForClassifier(rnnFeatures)

        let input = try MLDictionaryFeatureProvider(dictionary: ["rnn_features": processedFeatures])
        let output = try classifierModel.prediction(from: input)

        guard let outputName = output.featureNames.first,
              let classifierOutput = output.featureValue(for: outputName)?.multiArrayValue else {
            throw VADError.modelProcessingFailed("No classifier output found")
        }

        // Get raw probability and apply calibration
        let rawProbability = classifierOutput[0].floatValue
        let calibratedProbability = calibrateClassifierOutput(rawProbability)

        if config.debugMode {
            logger.debug("Classifier: raw=\(String(format: "%.4f", rawProbability)), calibrated=\(String(format: "%.4f", calibratedProbability))")
        }

        return calibratedProbability
    }

    /// Prepare RNN features for classifier input
    private func prepareRNNFeaturesForClassifier(_ rnnFeatures: MLMultiArray) throws -> MLMultiArray {
        let shape = rnnFeatures.shape.map { $0.intValue }

        // If already in correct shape (1, 4, 128), return as is
        if shape.count == 3 && shape[0] == 1 && shape[1] == 4 && shape[2] == 128 {
            return rnnFeatures
        }

        // Create target shape (1, 4, 128)
        let targetArray = try MLMultiArray(shape: [1, 4, 128], dataType: .float32)

        // Handle different input shapes
        if shape.count == 3 {
            let batchSize = min(shape[0], 1)
            let timeSteps = min(shape[1], 4)
            let featureSize = min(shape[2], 128)

            for b in 0..<batchSize {
                for t in 0..<timeSteps {
                    for f in 0..<featureSize {
                        let sourceIndex = b * (shape[1] * shape[2]) + t * shape[2] + f
                        let targetIndex = b * (4 * 128) + t * 128 + f

                        if sourceIndex < rnnFeatures.count && targetIndex < targetArray.count {
                            targetArray[targetIndex] = rnnFeatures[sourceIndex]
                        }
                    }
                }
            }
        } else {
            // Fallback: copy as much as possible
            let sourceElements = min(rnnFeatures.count, targetArray.count)
            for i in 0..<sourceElements {
                targetArray[i] = rnnFeatures[i]
            }
        }

        return targetArray
    }

    /// Calibrate classifier output to match expected probability distribution
    private func calibrateClassifierOutput(_ rawProbability: Float) -> Float {
        // Based on the Python reference implementation
        let scale: Float = 3.0
        let bias: Float = -0.25
        let sharpness: Float = 4.0

        // Apply scaling and bias
        let scaled = (rawProbability + bias) * scale

        // Apply sharper sigmoid for better separation
        let calibrated = 1.0 / (1.0 + exp(-sharpness * (scaled - 0.5)))

        // Ensure valid probability range
        return max(0.0, min(1.0, calibrated))
    }

    /// Calculate VAD probability from RNN features (improved fallback method)
    private func calculateVADProbability(from rnnFeatures: MLMultiArray) -> Float {
        let shape = rnnFeatures.shape.map { $0.intValue }
        guard shape.count >= 2 else {
            return 0.0
        }

        let totalElements = rnnFeatures.count
        guard totalElements > 0 else { return 0.0 }

        // Extract values from MLMultiArray
        var values: [Float] = []
        for i in 0..<totalElements {
            values.append(rnnFeatures[i].floatValue)
        }

        // Advanced feature extraction optimized for speech vs non-speech discrimination
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Float(values.count)
        let std = sqrt(variance)

        // Energy-based features (fundamental for VAD)
        let energy = values.map { $0 * $0 }.reduce(0, +) / Float(values.count)
        let logEnergy = log(max(energy, 1e-10))

        // Temporal features
        let zeroCrossingRate = calculateZeroCrossingRate(values)
        let peakCount = calculatePeakCount(values)
        let _ = Float(peakCount) / Float(values.count)  // peakRatio for future use

        // Speech pattern analysis
        let speechIndicator = calculateSpeechIndicator(values)

        // Noise detection features
        let isHighFrequencyNoise = zeroCrossingRate > 0.3  // Too many zero crossings = noise
        let isVeryHighEnergy = logEnergy > -0.5           // Extremely high energy = often noise
        let isTooManyPeaks = peakCount > 200              // Too many peaks = noise

        // Balanced feature weighting for better speech/noise discrimination
        let energyScore = max(0.0, tanh(logEnergy + 5.0) * 0.3)   // Moderate energy threshold
        let varianceScore = tanh(std * 2.0) * 0.15                // Moderate variance importance
        let speechScore = speechIndicator * 0.4                   // Strong weight for speech patterns
        let crossingScore = tanh(zeroCrossingRate * 4.0) * 0.15   // Moderate crossing rate

        // Refined noise detection with moderate penalties
        var penalties: Float = 0.0
        if isHighFrequencyNoise { penalties -= 0.2 }         // Moderate penalty for high-frequency noise
        if isVeryHighEnergy { penalties -= 0.15 }            // Moderate penalty for very high energy
        if isTooManyPeaks { penalties -= 0.15 }              // Moderate penalty for too many peaks
        if speechIndicator < 0.3 { penalties -= 0.2 }       // Moderate penalty for non-speech patterns

        // Moderate additional noise pattern detection
        if std < 0.005 { penalties -= 0.15 }                 // Very little variation = likely noise
        if logEnergy > 3.0 { penalties -= 0.1 }              // Very high energy = potentially noise
        if zeroCrossingRate > 0.5 { penalties -= 0.2 }      // Excessive zero crossings = noise

        let baseScore = energyScore + varianceScore + speechScore + crossingScore
        let combinedScore = max(0.0, baseScore + penalties)  // Ensure non-negative

        // Balanced sigmoid for reasonable speech detection
        let probability = 1.0 / (1.0 + exp(-8.0 * (combinedScore - 0.4)))

        if config.debugMode {
            // print("VAD Debug: energy=\(String(format: "%.4f", logEnergy)), std=\(String(format: "%.4f", std)), peaks=\(peakCount), speech=\(String(format: "%.4f", speechIndicator)), combined=\(String(format: "%.4f", combinedScore)), prob=\(String(format: "%.4f", probability))")
        }

        return max(0.0, min(1.0, probability))
    }

    /// Calculate zero crossing rate for VAD
    private func calculateZeroCrossingRate(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0.0 }

        var crossings = 0
        for i in 1..<values.count {
            if (values[i] >= 0) != (values[i-1] >= 0) {
                crossings += 1
            }
        }
        return Float(crossings) / Float(values.count - 1)
    }

    /// Calculate number of peaks (local maxima)
    private func calculatePeakCount(_ values: [Float]) -> Int {
        guard values.count > 2 else { return 0 }

        var peaks = 0
        for i in 1..<(values.count - 1) {
            if values[i] > values[i-1] && values[i] > values[i+1] && abs(values[i]) > 0.1 {
                peaks += 1
            }
        }
        return peaks
    }

    /// Calculate skewness
    private func calculateSkewness(_ values: [Float], mean: Float, std: Float) -> Float {
        guard std > 0 else { return 0.0 }

        let skew = values.map { pow(($0 - mean) / std, 3) }.reduce(0, +) / Float(values.count)
        return skew
    }

    /// Calculate kurtosis
    private func calculateKurtosis(_ values: [Float], mean: Float, std: Float) -> Float {
        guard std > 0 else { return 0.0 }

        let kurt = values.map { pow(($0 - mean) / std, 4) }.reduce(0, +) / Float(values.count) - 3.0
        return kurt
    }

    /// Calculate speech-like pattern indicator
    private func calculateSpeechIndicator(_ values: [Float]) -> Float {
        // Look for patterns that are characteristic of speech vs noise/music
        let absValues = values.map { abs($0) }
        let sortedValues = absValues.sorted(by: >)

        // Speech typically has moderate dynamic range (not too flat, not too extreme)
        let topQuartile = Array(sortedValues.prefix(max(1, sortedValues.count / 4)))
        let bottomQuartile = Array(sortedValues.suffix(max(1, sortedValues.count / 4)))
        let middleHalf = Array(sortedValues[sortedValues.count/4..<3*sortedValues.count/4])

        let topMean = topQuartile.reduce(0, +) / Float(topQuartile.count)
        let bottomMean = bottomQuartile.reduce(0, +) / Float(bottomQuartile.count)
        let middleMean = middleHalf.isEmpty ? 0 : middleHalf.reduce(0, +) / Float(middleHalf.count)

        // Speech has balanced distribution (strong middle component)
        let dynamicRange = topMean / max(bottomMean, 1e-10)
        let middleRatio = middleMean / max(topMean, 1e-10)

        // Speech-like patterns: moderate dynamic range + good middle energy
        let rangeScore = 1.0 / (1.0 + exp(-2.0 * (log(max(dynamicRange, 1.0)) - 3.0))) // Peak around 20:1 ratio
        let middleScore = middleRatio  // Higher middle energy is speech-like

        // Temporal consistency check (speech has structured patterns)
        let consistency = calculateTemporalConsistency(absValues)

        // Combine indicators (speech needs all three)
        let speechLikeness = (rangeScore * 0.4 + middleScore * 0.4 + consistency * 0.2)

        return max(0.0, min(1.0, speechLikeness))
    }

    /// Calculate temporal consistency for speech patterns
    private func calculateTemporalConsistency(_ absValues: [Float]) -> Float {
        guard absValues.count > 10 else { return 0.0 }

        // Look for structured patterns (not random noise)
        let windowSize = 5
        var consistencyScore: Float = 0.0
        var windowCount = 0

        for i in stride(from: 0, to: absValues.count - windowSize, by: windowSize) {
            let window = Array(absValues[i..<min(i + windowSize, absValues.count)])
            let windowMean = window.reduce(0, +) / Float(window.count)
            let windowVar = window.map { pow($0 - windowMean, 2) }.reduce(0, +) / Float(window.count)

            // Speech has moderate variance within short windows
            let normalizedVar = tanh(sqrt(windowVar) * 10.0)
            consistencyScore += normalizedVar
            windowCount += 1
        }

        return windowCount > 0 ? consistencyScore / Float(windowCount) : 0.0
    }

    /// Calculate Signal-to-Noise Ratio for audio quality assessment
    private func calculateSNR(_ audioChunk: [Float]) -> Float {
        guard audioChunk.count > 0 else { return -Float.infinity }

        // Calculate signal energy
        let signalEnergy = audioChunk.map { $0 * $0 }.reduce(0, +) / Float(audioChunk.count)
        let signalPower = max(signalEnergy, 1e-10)

        // Update noise floor estimate
        updateNoiseFloor(signalPower)

        // Calculate SNR in dB
        let snrLinear = signalPower / pow(10, currentNoiseFloor / 10.0)
        let snrDB = 10.0 * log10(max(snrLinear, 1e-10))

        return snrDB
    }

    /// Update noise floor estimation using minimum statistics
    private func updateNoiseFloor(_ currentPower: Float) {
        let powerDB = 10.0 * log10(max(currentPower, 1e-10))

        // Add to noise floor buffer
        noiseFloorBuffer.append(powerDB)

        // Keep only recent samples for noise floor estimation
        if noiseFloorBuffer.count > config.noiseFloorWindow {
            noiseFloorBuffer.removeFirst()
        }

        // Update noise floor using minimum statistics (conservative approach)
        if noiseFloorBuffer.count >= 10 {
            let sortedPowers = noiseFloorBuffer.sorted()
            let percentile10 = sortedPowers[sortedPowers.count / 10]  // 10th percentile

            // Smooth the noise floor update
            let alpha: Float = 0.1
            currentNoiseFloor = currentNoiseFloor * (1 - alpha) + percentile10 * alpha
        }
    }

    /// Calculate spectral features for enhanced VAD (optimized version)
    private func calculateSpectralFeatures(_ audioChunk: [Float]) -> SpectralFeatures {
        let fftSize = min(256, audioChunk.count)  // Reduced FFT size for better performance
        let fftInput = Array(audioChunk.prefix(fftSize))

        // Compute FFT magnitude spectrum
        let spectrum = computeFFTMagnitude(fftInput)

        // Calculate spectral features (optimized calculations)
        let spectralCentroid = calculateSpectralCentroid(spectrum)
        let spectralRolloff = calculateSpectralRolloff(spectrum)
        let spectralFlux = calculateSpectralFlux(spectrum)
        let mfccFeatures = calculateSimplifiedMFCC(spectrum)  // Simplified MFCC
        let zeroCrossingRate = calculateZeroCrossingRate(fftInput)
        let spectralEntropy = calculateSpectralEntropy(spectrum)

        return SpectralFeatures(
            spectralCentroid: spectralCentroid,
            spectralRolloff: spectralRolloff,
            spectralFlux: spectralFlux,
            mfccFeatures: mfccFeatures,
            zeroCrossingRate: zeroCrossingRate,
            spectralEntropy: spectralEntropy
        )
    }

    /// Apply audio quality filtering based on SNR and spectral features
    private func applyAudioQualityFiltering(
        rawProbability: Float,
        snr: Float?,
        spectralFeatures: SpectralFeatures?
    ) -> Float {
        var filteredProbability = rawProbability

        // SNR-based filtering - more aggressive
        if let snr = snr, snr < config.minSNRThreshold {
            let snrPenalty = max(0.0, (config.minSNRThreshold - snr) / config.minSNRThreshold)
            filteredProbability *= (1.0 - snrPenalty * 0.8)  // Reduce probability by up to 80%
        }

        // Spectral feature-based filtering - more aggressive
        if let features = spectralFeatures {
            // Check if spectral centroid is in expected speech range
            let centroidInRange = features.spectralCentroid >= config.spectralCentroidRange.min &&
                                 features.spectralCentroid <= config.spectralCentroidRange.max

            if !centroidInRange {
                filteredProbability *= 0.5  // Reduce probability by 50%
            }

            // Check spectral rolloff (speech should have energy distributed across frequencies)
            if features.spectralRolloff > config.spectralRolloffThreshold {
                filteredProbability *= 0.6  // Reduce probability by 40%
            }

            // Excessive zero crossings indicate noise
            if features.zeroCrossingRate > 0.3 {
                filteredProbability *= 0.4  // Reduce probability by 60%
            }

            // Low spectral entropy indicates tonal/musical content (not speech)
            if features.spectralEntropy < 0.3 {
                filteredProbability *= 0.3  // Reduce probability by 70%
            }
        }

        return max(0.0, min(1.0, filteredProbability))
    }

    /// Compute FFT magnitude spectrum using Accelerate framework
    private func computeFFTMagnitude(_ input: [Float]) -> [Float] {
        let n = input.count
        guard n > 0 else { return [] }

        // Find next power of 2 for FFT
        let log2n = Int(log2(Float(n)).rounded(.up))
        let fftSize = 1 << log2n

        // Prepare input with zero padding
        var paddedInput = input
        paddedInput.append(contentsOf: Array(repeating: 0.0, count: fftSize - n))

        // Setup FFT
        guard let fftSetup = vDSP_create_fftsetup(vDSP_Length(log2n), FFTRadix(kFFTRadix2)) else {
            return []
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Prepare complex buffer using proper pointer management
        var realInput = paddedInput
        var imagInput = Array(repeating: Float(0.0), count: fftSize)

        // Use withUnsafeMutableBufferPointer for proper pointer management
        return realInput.withUnsafeMutableBufferPointer { realPtr in
            imagInput.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                // Perform FFT
                vDSP_fft_zip(fftSetup, &splitComplex, 1, vDSP_Length(log2n), FFTDirection(FFT_FORWARD))

                // Compute magnitude spectrum
                var magnitudes = Array(repeating: Float(0.0), count: fftSize / 2)
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))

                // Take square root to get magnitude (not power)
                for i in 0..<magnitudes.count {
                    magnitudes[i] = sqrt(magnitudes[i])
                }

                return magnitudes
            }
        }
    }

    /// Calculate spectral centroid (center of mass of spectrum)
    private func calculateSpectralCentroid(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        var weightedSum: Float = 0.0
        for (i, magnitude) in spectrum.enumerated() {
            let frequency = Float(i) * Float(config.sampleRate) / Float(spectrum.count * 2)
            weightedSum += frequency * magnitude
        }

        return weightedSum / totalEnergy
    }

    /// Calculate spectral rolloff (frequency below which X% of energy is contained)
    private func calculateSpectralRolloff(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.map { $0 * $0 }.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        let rolloffThreshold = totalEnergy * config.spectralRolloffThreshold
        var cumulativeEnergy: Float = 0.0

        for (i, magnitude) in spectrum.enumerated() {
            cumulativeEnergy += magnitude * magnitude
            if cumulativeEnergy >= rolloffThreshold {
                return Float(i) * Float(config.sampleRate) / Float(spectrum.count * 2)
            }
        }

        return Float(spectrum.count - 1) * Float(config.sampleRate) / Float(spectrum.count * 2)
    }

    /// Calculate spectral flux (measure of spectral change)
    private func calculateSpectralFlux(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        if previousSpectrum.isEmpty {
            previousSpectrum = spectrum
            return 0.0
        }

        guard previousSpectrum.count == spectrum.count else {
            previousSpectrum = spectrum
            return 0.0
        }

        var flux: Float = 0.0
        for i in 0..<spectrum.count {
            let diff = spectrum[i] - previousSpectrum[i]
            flux += max(0.0, diff)  // Only positive changes
        }

        previousSpectrum = spectrum
        return flux / Float(spectrum.count)
    }

    /// Calculate MFCC features (simplified version)
    private func calculateMFCC(_ spectrum: [Float]) -> [Float] {
        guard !spectrum.isEmpty else { return Array(repeating: 0.0, count: 13) }

        // Simplified MFCC calculation (for production, use proper mel-scale filtering)
        let numCoefficients = 13
        let logSpectrum = spectrum.map { log(max($0, 1e-10)) }

        // DCT (Discrete Cosine Transform) approximation
        var mfcc: [Float] = []
        let n = logSpectrum.count

        for i in 0..<numCoefficients {
            var sum: Float = 0.0
            for j in 0..<n {
                let angle = Float(Double.pi * Double(i) * (Double(j) + 0.5) / Double(n))
                sum += logSpectrum[j] * cos(angle)
            }
            mfcc.append(sum)
        }

        return mfcc
    }

    /// Calculate simplified MFCC features (faster version)
    private func calculateSimplifiedMFCC(_ spectrum: [Float]) -> [Float] {
        guard !spectrum.isEmpty else { return Array(repeating: 0.0, count: 13) }

        // Very simplified MFCC - just use log spectrum with decimation
        let numCoefficients = 13
        let logSpectrum = spectrum.map { log(max($0, 1e-10)) }

        // Simple decimation approach (much faster than full DCT)
        var mfcc: [Float] = []
        let step = max(1, logSpectrum.count / numCoefficients)

        for i in 0..<numCoefficients {
            let index = min(i * step, logSpectrum.count - 1)
            mfcc.append(logSpectrum[index])
        }

        return mfcc
    }

    /// Calculate spectral entropy (measure of spectral complexity)
    private func calculateSpectralEntropy(_ spectrum: [Float]) -> Float {
        guard !spectrum.isEmpty else { return 0.0 }

        let totalEnergy = spectrum.map { $0 * $0 }.reduce(0, +)
        guard totalEnergy > 0 else { return 0.0 }

        // Normalize to probability distribution
        let probabilities = spectrum.map { ($0 * $0) / totalEnergy }

        // Calculate entropy
        var entropy: Float = 0.0
        for p in probabilities {
            if p > 0 {
                entropy -= p * log(p)
            }
        }

        // Normalize entropy
        return entropy / log(Float(spectrum.count))
    }

    /// Apply temporal smoothing filter to reduce noise
    private func applySmoothingFilter(_ probability: Float) -> Float {
        // Add to sliding window
        probabilityWindow.append(probability)
        if probabilityWindow.count > windowSize {
            probabilityWindow.removeFirst()
        }

        // Apply weighted moving average (more weight to recent values)
        guard !probabilityWindow.isEmpty else { return probability }

        let weights: [Float] = [0.1, 0.15, 0.2, 0.25, 0.3] // Most recent gets highest weight
        var weightedSum: Float = 0.0
        var totalWeight: Float = 0.0

        let startIndex = max(0, weights.count - probabilityWindow.count)
        for (i, prob) in probabilityWindow.enumerated() {
            let weightIndex = startIndex + i
            if weightIndex < weights.count {
                let weight = weights[weightIndex]
                weightedSum += prob * weight
                totalWeight += weight
            }
        }

        return totalWeight > 0 ? weightedSum / totalWeight : probability
    }

    /// Update adaptive threshold based on recent probability distribution
    private func updateAdaptiveThreshold(_ probability: Float) -> Float {
        guard config.adaptiveThreshold else {
            return config.threshold
        }

        // Update probability history
        probabilityHistory.append(probability)
        let historyLimit = 50  // Keep last 50 values for analysis
        if probabilityHistory.count > historyLimit {
            probabilityHistory.removeFirst()
        }

        // Calculate adaptive threshold based on recent distribution
        guard probabilityHistory.count >= 10 else {
            return config.threshold  // Use default until we have enough data
        }

        let sortedProbs = probabilityHistory.sorted()
        let median = sortedProbs[sortedProbs.count / 2]
        let q75 = sortedProbs[Int(Float(sortedProbs.count) * 0.75)]
        let q25 = sortedProbs[Int(Float(sortedProbs.count) * 0.25)]

        // Adaptive threshold based on distribution statistics
        let iqr = q75 - q25
        let adaptiveBase = median + (iqr * 0.5)  // Above median + half IQR

        // Constrain to configured bounds
        let newThreshold = max(config.minThreshold, min(config.maxThreshold, adaptiveBase))

        // Smooth threshold changes to avoid sudden jumps
        let smoothingFactor: Float = 0.1
        adaptiveThreshold = adaptiveThreshold * (1 - smoothingFactor) + newThreshold * smoothingFactor

        return adaptiveThreshold
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

    /// Copy models from bundle to cache directory
    public func copyModelsFromBundle() throws {
        let modelsDirectory = getModelsDirectory()

        let modelNames = ["silero_stft.mlmodel", "silero_encoder.mlmodel", "silero_rnn_decoder.mlmodel"]

        for modelName in modelNames {
            let destinationPath = modelsDirectory.appendingPathComponent(modelName)

            // Skip if already exists
            if FileManager.default.fileExists(atPath: destinationPath.path) {
                continue
            }

            // Try to find in bundle
            if modelName.hasSuffix(".mlmodel") {
                // Handle .mlmodel files
                let baseName = modelName.replacingOccurrences(of: ".mlmodel", with: "")
                if let bundlePath = Bundle.main.path(forResource: baseName, ofType: "mlmodel") {
                    try FileManager.default.copyItem(atPath: bundlePath, toPath: destinationPath.path)
                    logger.info("Copied \(modelName) from bundle to cache")
                } else {
                    logger.warning("Model \(modelName) not found in bundle")
                }
            } else {
                logger.warning("Unknown model format: \(modelName)")
            }
        }
    }

    /// Copy models from a specific directory (deprecated - use Hugging Face download instead)
    @available(*, deprecated, message: "Use automatic Hugging Face download instead")
    public func copyModelsFromDirectory(_ sourceDirectory: URL) throws {
        let modelsDirectory = getModelsDirectory()

        // Define source models and their destination names
        let modelMappings = [
            ("silero_stft.mlmodel", "silero_stft.mlmodel"),
            ("silero_encoder.mlmodel", "silero_encoder.mlmodel"),
            ("silero_rnn_decoder.mlmodel", "silero_rnn_decoder.mlmodel")
        ]

        for (sourceModelName, destinationModelName) in modelMappings {
            let sourcePath = sourceDirectory.appendingPathComponent(sourceModelName)
            let destinationPath = modelsDirectory.appendingPathComponent(destinationModelName)

            // Skip if source doesn't exist
            guard FileManager.default.fileExists(atPath: sourcePath.path) else {
                logger.warning("Source model \(sourceModelName) not found at: \(sourcePath.path)")
                continue
            }

            // Remove existing destination if it exists
            if FileManager.default.fileExists(atPath: destinationPath.path) {
                try FileManager.default.removeItem(at: destinationPath)
            }

            // Copy the model
            try FileManager.default.copyItem(at: sourcePath, to: destinationPath)
            logger.info("Copied \(sourceModelName) from \(sourcePath.path) to cache")
        }
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

    /// Check if all required VAD models exist
    private func allModelsExist() -> Bool {
        let modelsDirectory = getModelsDirectory()
        let modelNames = ["silero_stft.mlmodel", "silero_encoder.mlmodel", "silero_rnn_decoder.mlmodel"]

        for modelName in modelNames {
            let modelPath = modelsDirectory.appendingPathComponent(modelName)
            if !FileManager.default.fileExists(atPath: modelPath.path) {
                return false
            }
        }
        return true
    }

    /// Download VAD models from Hugging Face
    private func downloadVADModels() async throws {
        logger.info("Downloading VAD models...")

        // Download from Hugging Face (primary method)
        do {
            try await downloadVADModelsFromHuggingFace()
            logger.info("Successfully downloaded VAD models from Hugging Face")
            return
        } catch {
            logger.warning("Failed to download from Hugging Face: \(error)")
        }

        // Try copying from bundle as fallback only
        do {
            try copyModelsFromBundle()
            logger.info("Successfully copied VAD models from bundle")
            return
        } catch {
            logger.warning("Failed to copy from bundle: \(error)")
        }

        // If we get here, we couldn't download or find the models
        logger.error("Could not download VAD models from any source")
        logger.error("  - Failed: Hugging Face repository download")
        logger.error("  - Failed: application bundle fallback")
        logger.error("  - Note: vadCoreml/ directory is no longer used")

        throw VADError.modelLoadingFailed
    }

    /// Load VAD models with automatic recovery on compilation failures
    private func loadModelsWithAutoRecovery(
        stftPath: URL, encoderPath: URL, rnnPath: URL, maxRetries: Int = 2
    ) async throws {
        var attempt = 0

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        while attempt <= maxRetries {
            do {
                // Check if models exist, download if not found
                let modelsExist = allModelsExist()
                if !modelsExist {
                    logger.info("VAD models not found, attempting to download...")
                    try await downloadVADModelsFromHuggingFace()
                }

                // Compile models if needed and get compiled paths
                let compiledStftPath = try await compileModelIfNeeded(stftPath)
                let compiledEncoderPath = try await compileModelIfNeeded(encoderPath)
                let compiledRnnPath = try await compileModelIfNeeded(rnnPath)
                // Try to load all models
                logger.info("Attempting to load VAD models (attempt \(attempt + 1)/\(maxRetries + 1))")

                let stftModel = try MLModel(contentsOf: compiledStftPath, configuration: config)
                let encoderModel = try MLModel(contentsOf: compiledEncoderPath, configuration: config)
                let rnnModel = try MLModel(contentsOf: compiledRnnPath, configuration: config)

                // If we get here, all models loaded successfully
                self.stftModel = stftModel
                self.encoderModel = encoderModel
                self.rnnModel = rnnModel
                self.classifierModel = nil  // Always use fallback

                if attempt > 0 {
                    logger.info("VAD models loaded successfully after \(attempt) recovery attempt(s)")
                } else {
                    logger.info("All VAD models loaded successfully")
                }
                return

            } catch {
                logger.warning("VAD model loading failed (attempt \(attempt + 1)): \(error.localizedDescription)")

                // If this is our last attempt, try CPU fallback
                if attempt >= maxRetries {
                    logger.info("Final attempt with CPU-only configuration...")

                    do {
                        let cpuConfig = MLModelConfiguration()
                        cpuConfig.computeUnits = .cpuOnly

                        // Try CPU-only loading as last resort
                        let compiledStftPath = try await compileModelIfNeeded(stftPath)
                        let compiledEncoderPath = try await compileModelIfNeeded(encoderPath)
                        let compiledRnnPath = try await compileModelIfNeeded(rnnPath)

                        self.stftModel = try MLModel(contentsOf: compiledStftPath, configuration: cpuConfig)
                        self.encoderModel = try MLModel(contentsOf: compiledEncoderPath, configuration: cpuConfig)
                        self.rnnModel = try MLModel(contentsOf: compiledRnnPath, configuration: cpuConfig)

                        // No classifier model - always use fallback
                        self.classifierModel = nil

                        logger.info("VAD models loaded successfully with CPU-only configuration")
                        return

                    } catch {
                        logger.error("VAD model loading failed after \(maxRetries + 1) attempts, giving up")
                        throw VADError.modelCompilationFailed
                    }
                }

                // Auto-recovery: Delete corrupted models and re-download
                logger.info("Initiating auto-recovery: removing corrupted models and re-downloading...")

                        try await performModelRecovery(
            stftPath: stftPath,
            encoderPath: encoderPath,
            rnnPath: rnnPath
        )

                attempt += 1
            }
        }
    }

    /// Perform model recovery by removing corrupted models and re-downloading
    private func performModelRecovery(
        stftPath: URL, encoderPath: URL, rnnPath: URL
    ) async throws {
        // Remove potentially corrupted model files
        let modelPaths = [stftPath, encoderPath, rnnPath]

        for modelPath in modelPaths {
            if FileManager.default.fileExists(atPath: modelPath.path) {
                logger.info("Removing corrupted VAD model at \(modelPath.path)")
                try FileManager.default.removeItem(at: modelPath)
            }

            // Also remove compiled versions (for .mlmodel files)
            if modelPath.lastPathComponent.hasSuffix(".mlmodel") {
                let compiledPath = modelPath.appendingPathExtension("mlmodelc")
                if FileManager.default.fileExists(atPath: compiledPath.path) {
                    logger.info("Removing corrupted compiled VAD model at \(compiledPath.path)")
                    try FileManager.default.removeItem(at: compiledPath)
                }
            }
        }

        // Re-download the models from Hugging Face
        logger.info("Re-downloading VAD models from Hugging Face...")
        try await downloadVADModelsFromHuggingFace()

        logger.info("VAD model recovery completed - models re-downloaded")
    }

    /// Download VAD models from Hugging Face repository
    private func downloadVADModelsFromHuggingFace() async throws {
        logger.info("Downloading VAD models from Hugging Face...")

        let modelsDirectory = getModelsDirectory()
        let repoPath = "alexwengg/coreml-silero-vad"

        // Model files to download (source models)
        let modelFiles = [
            "silero_stft.mlmodel",
            "silero_encoder.mlmodel",
            "silero_rnn_decoder.mlmodel",
        ]

        // Download each source model file
        for modelFile in modelFiles {
            let modelURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelFile)")!
            let destinationPath = modelsDirectory.appendingPathComponent(modelFile)

            do {
                logger.info("Downloading \(modelFile)...")
                let (data, response) = try await URLSession.shared.data(from: modelURL)

                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: destinationPath)
                        logger.info("✅ Downloaded \(modelFile) (\(data.count) bytes)")
                    } else {
                        logger.error("Failed to download \(modelFile): HTTP \(httpResponse.statusCode)")
                        throw VADError.modelDownloadFailed
                    }
                } else {
                    // No HTTP response, but we have data - write it anyway
                    try data.write(to: destinationPath)
                    logger.info("✅ Downloaded \(modelFile) (\(data.count) bytes)")
                }

            } catch {
                logger.error("Failed to download \(modelFile): \(error)")
                throw VADError.modelDownloadFailed
            }
        }

        logger.info("✅ All VAD models downloaded successfully from Hugging Face")
    }

    /// Download a complete .mlmodelc bundle from Hugging Face (for compiled models)
    private func downloadMLModelCBundle(repoPath: String, modelName: String, outputPath: URL) async throws {
        logger.info("Downloading \(modelName) bundle from Hugging Face")

        // Create output directory
        try FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)

        let bundleFiles = [
            "model.mil",
            "coremldata.bin",
            "metadata.json",
        ]

        let weightFiles = [
            "weights/weight.bin"
        ]

        // Download each file in the bundle
        for fileName in bundleFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(fileName)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(fileName)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(fileName) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(fileName) for \(modelName) - file may not exist")
                    // Create empty file if it doesn't exist (some files are optional)
                    if fileName == "metadata.json" {
                        let destinationPath = outputPath.appendingPathComponent(fileName)
                        try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                    }
                }
            } catch {
                logger.warning("Error downloading \(fileName): \(error.localizedDescription)")
                // For critical files, create minimal versions
                if fileName == "coremldata.bin" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try Data().write(to: destinationPath)
                } else if fileName == "metadata.json" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                }
            }
        }

        // Download weight files
        for weightFile in weightFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(weightFile)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(weightFile)

                    // Create weights directory if it doesn't exist
                    let weightsDir = destinationPath.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(weightFile) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(weightFile) for \(modelName)")
                    throw VADError.modelDownloadFailed
                }
            } catch {
                logger.error("Critical error downloading \(weightFile): \(error.localizedDescription)")
                throw VADError.modelDownloadFailed
            }
        }

        logger.info("Completed downloading \(modelName) bundle")
    }

    /// Cleanup resources
    public func cleanup() {
        stftModel = nil
        encoderModel = nil
        rnnModel = nil
        classifierModel = nil
        hState = nil
        cState = nil
        featureBuffer.removeAll()
        noiseFloorBuffer.removeAll()
        previousSpectrum.removeAll()
        probabilityHistory.removeAll()
        probabilityWindow.removeAll()

        logger.info("VAD resources cleaned up")
    }

    /// Compile model if needed and return path to compiled model
    private func compileModelIfNeeded(_ modelPath: URL) async throws -> URL {
        let modelName = modelPath.lastPathComponent

        // If the model is already compiled (.mlmodelc), return it directly
        if modelName.hasSuffix(".mlmodelc") {
            logger.info("Model is already compiled: \(modelName)")
            return modelPath
        }

        // Handle .mlmodel files that need compilation
        if modelName.hasSuffix(".mlmodel") {
            let compiledModelName = modelName.replacingOccurrences(of: ".mlmodel", with: ".mlmodelc")
            let compiledPath = modelPath.deletingLastPathComponent().appendingPathComponent(compiledModelName)

            // Check if compiled model already exists and is newer than source
            if FileManager.default.fileExists(atPath: compiledPath.path) {
                do {
                    let sourceAttributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
                    let compiledAttributes = try FileManager.default.attributesOfItem(atPath: compiledPath.path)

                    if let sourceDate = sourceAttributes[.modificationDate] as? Date,
                       let compiledDate = compiledAttributes[.modificationDate] as? Date,
                       compiledDate >= sourceDate {
                        // Compiled model is up to date
                        logger.info("Using existing compiled model: \(compiledModelName)")
                        return compiledPath
                    }
                } catch {
                    // If we can't check dates, just recompile
                    logger.warning("Could not check model dates, recompiling: \(error)")
                }
            }

            // Compile the model
            logger.info("Compiling model: \(modelName)")
            let startTime = Date()

            do {
                let newCompiledPath = try await MLModel.compileModel(at: modelPath)
                let compileTime = Date().timeIntervalSince(startTime)
                logger.info("Model compiled successfully in \(String(format: "%.2f", compileTime))s: \(compiledModelName)")

                // Move compiled model to expected location if needed
                if newCompiledPath != compiledPath {
                    // Remove existing if present
                    if FileManager.default.fileExists(atPath: compiledPath.path) {
                        try FileManager.default.removeItem(at: compiledPath)
                    }
                    try FileManager.default.moveItem(at: newCompiledPath, to: compiledPath)
                    logger.info("Moved compiled model to: \(compiledPath.path)")
                }

                return compiledPath

            } catch {
                logger.error("Failed to compile model \(modelName): \(error)")
                throw VADError.modelLoadingFailed
            }
        }

        // If we get here, the model is neither .mlmodel nor .mlmodelc
        logger.error("Unknown model format: \(modelName)")
        throw VADError.modelLoadingFailed
    }
}
