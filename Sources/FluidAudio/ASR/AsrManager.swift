@preconcurrency import CoreML
import Foundation
import OSLog

// MARK: - AsrManager

/// AsrManager provides automatic speech recognition using Parakeet TDT (Token-and-Duration Transducer) models.
///
/// This class implements high-performance speech-to-text transcription optimized for Apple Neural Engine,
/// achieving real-time factors (RTFx) of 11-50x depending on audio duration and hardware.
///
/// ## Features
/// - **Parakeet TDT-0.6b** model implementation with duration-aware decoding
/// - **Apple Neural Engine** optimization for maximum performance
/// - **Chunked processing** for long audio files (automatic 10-second chunks)
/// - **Text normalization** compatible with HuggingFace ASR evaluation standards
/// - **Auto-recovery** from CoreML model loading failures
///
/// ## Performance
/// - LibriSpeech test-clean: Average WER 5.9%, Median WER 1.9%
/// - LibriSpeech test-other: Average WER 4-6%
/// - RTFx: ~11.8x (includes full end-to-end file I/O overhead)
///
/// ## Usage
/// ```swift
/// let asrManager = AsrManager()
/// try await asrManager.initialize()
/// let result = try await asrManager.transcribe(audioSamples)
/// print("Transcription: \(result.text)")
/// ```
///
/// ## Technical Details
/// The TDT model uses a novel architecture that jointly predicts both tokens and their durations,
/// enabling more accurate transcription of speech with varying speaking rates. The implementation
/// includes advanced features like duration-based token repetition and confidence-weighted decoding.
///
/// - Note: Requires macOS 13.0+ or iOS 16.0+ for CoreML 6 features
@available(macOS 13.0, iOS 16.0, *)
public final class AsrManager {

    internal let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ASR")
    internal let config: ASRConfig
    private let modelsDirectory: URL

    internal var melSpectrogramModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?
    var decoderState: DecoderState = DecoderState()

    let blankId = 1024  // Verified: this works correctly (not actually "warming" token)
    let sosId = 1024    // Start of sequence token (same as blank for this model)

    /// Creates a new ASR manager with the specified configuration
    /// - Parameters:
    ///   - config: ASR configuration settings
    ///   - modelsDirectory: Custom directory for CoreML models. If nil, uses default Application Support location
    public init(config: ASRConfig = .default, modelsDirectory: URL? = nil) {
        self.config = config
        self.modelsDirectory = modelsDirectory ?? Self.getDefaultModelsDirectory()

        logger.info("TDT enabled with durations: \(config.tdtConfig.durations)")
    }

    /// Indicates whether all required models are loaded and ready for transcription
    public var isAvailable: Bool {
        return melSpectrogramModel != nil && encoderModel != nil && decoderModel != nil && jointModel != nil
    }

    /// Initializes the ASR system by loading Parakeet TDT models.
    ///
    /// This method downloads models if needed and compiles them for Apple Neural Engine.
    /// Models are cached for subsequent runs.
    ///
    /// - Throws: `ASRError.modelLoadFailed` if models cannot be loaded
    /// - Note: This is an async operation that may take several seconds on first run
    public func initialize() async throws {
        logger.info("Initializing AsrManager with Parakeet models")

        logger.info("Models directory: \(self.modelsDirectory.path)")

        let melSpectrogramPath = modelsDirectory.appendingPathComponent("Melspectogram.mlmodelc")
        let encoderPath = modelsDirectory.appendingPathComponent("ParakeetEncoder.mlmodelc")
        let decoderPath = modelsDirectory.appendingPathComponent("ParakeetDecoder.mlmodelc")
        let jointPath = modelsDirectory.appendingPathComponent("RNNTJoint.mlmodelc")

        do {
            try await DownloadUtils.downloadParakeetModelsIfNeeded(to: modelsDirectory)

            let modelConfig = MLModelConfiguration()

            modelConfig.allowLowPrecisionAccumulationOnGPU = true

            if ProcessInfo.processInfo.environment["CI"] != nil {
                // Force CPU and Neural Engine only (no GPU) in CI environments
                // GPU can cause issues in virtualized environments like GitHub Actions
                modelConfig.computeUnits = .cpuAndNeuralEngine
                logger.info("🔧 ASR: Using compute units: cpuAndNeuralEngine (CI environment)")

            } else {
                // Use all available compute units for best performance on real hardware
                modelConfig.computeUnits = .all
            }

            logger.info("Loading Parakeet models from \(self.modelsDirectory.path)")

            let modelPaths = [
                ("Mel-spectrogram", melSpectrogramPath),
                ("Encoder", encoderPath),
                ("Decoder", decoderPath),
                ("Joint", jointPath)
            ]

            for (name, path) in modelPaths {
                if !FileManager.default.fileExists(atPath: path.path) {
                    logger.error("\(name) model not found at: \(path.path)")
                    throw ASRError.modelLoadFailed
                }
                logger.info("\(name) model found at: \(path.path)")
            }

            let models = try await loadAllModels(
                melSpectrogramPath: melSpectrogramPath,
                encoderPath: encoderPath,
                decoderPath: decoderPath,
                jointPath: jointPath,
                configuration: modelConfig
            )

            melSpectrogramModel = models.melSpectrogram
            encoderModel = models.encoder
            decoderModel = models.decoder
            jointModel = models.joint

            logger.info("AsrManager initialized successfully")

        } catch {
            logger.error("Failed to initialize AsrManager: \(error.localizedDescription)")
            if let mlError = error as? MLModelError {
                logger.error("MLModel error details: \(mlError)")
            }
            throw ASRError.modelLoadFailed
        }
    }

    /// Generic helper to create MLFeatureProvider with multiple arrays
    private func createFeatureProvider(features: [(name: String, array: MLMultiArray)]) throws -> MLFeatureProvider {
        var featureDict: [String: MLFeatureValue] = [:]
        for (name, array) in features {
            featureDict[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: featureDict)
    }

    /// Helper to create single-value MLMultiArray
    private func createScalarArray(value: Int, shape: [NSNumber] = [1], dataType: MLMultiArrayDataType = .int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }

    func prepareMelSpectrogramInput(_ audioSamples: [Float]) throws -> MLFeatureProvider {
        let audioLength = audioSamples.count

        let audioArray = try MLMultiArray(shape: [1, audioLength] as [NSNumber], dataType: .float32)
        for i in 0..<audioLength {
            audioArray[i] = NSNumber(value: audioSamples[i])
        }

        let lengthArray = try createScalarArray(value: audioLength)

        return try createFeatureProvider(features: [
            ("audio_signal", audioArray),
            ("audio_length", lengthArray)
        ])
    }

    func prepareEncoderInput(_ melSpectrogramOutput: MLFeatureProvider) throws -> MLFeatureProvider {
        let melSpectrogram = try extractFeatureValue(from: melSpectrogramOutput, key: "melspectogram", errorMessage: "Invalid mel-spectrogram output")
        let melSpectrogramLength = try extractFeatureValue(from: melSpectrogramOutput, key: "melspectogram_length", errorMessage: "Invalid mel-spectrogram length output")

        return try createFeatureProvider(features: [
            ("audio_signal", melSpectrogram),
            ("length", melSpectrogramLength)
        ])
    }

    func transposeEncoderOutput(_ encoderOutput: MLMultiArray) throws -> MLMultiArray {
        // Transpose encoder output from (1, 1024, T) to (1, T, 1024)
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let featSize = shape[1].intValue
        let timeSteps = shape[2].intValue

        let transposedArray = try MLMultiArray(shape: [batchSize, timeSteps, featSize] as [NSNumber], dataType: .float32)

        for t in 0..<timeSteps {
            for f in 0..<featSize {
                let sourceIndex = f * timeSteps + t
                let targetIndex = t * featSize + f
                transposedArray[targetIndex] = encoderOutput[sourceIndex]
            }
        }

        return transposedArray
    }

    func prepareDecoderInput(
        targetToken: Int,
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try createScalarArray(value: targetToken, shape: [1, 1])
        let targetLengthArray = try createScalarArray(value: 1)  // Always 1 for single token

        return try createFeatureProvider(features: [
            ("targets", targetArray),
            ("target_lengths", targetLengthArray),
            ("h_in", hiddenState),
            ("c_in", cellState)
        ])
    }

    // MARK: - TDT Helper Functions

    /// Initialize decoder state with a clean blank token pass
    internal func initializeDecoderState() async throws {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        var freshState = DecoderState()

        let initDecoderInput = try prepareDecoderInput(
            targetToken: blankId,
            hiddenState: freshState.hiddenState,
            cellState: freshState.cellState
        )

        let initDecoderOutput = try decoderModel.prediction(from: initDecoderInput, options: MLPredictionOptions())

        freshState.update(from: initDecoderOutput)

        if config.enableDebug {
            logger.info("Decoder state initialized cleanly")
        }

        decoderState = freshState
    }
    private func loadVocabulary() -> [Int: String] {
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        let vocabPath = appDirectory.appendingPathComponent("parakeet_vocab.json")

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning("Vocabulary file not found at \(vocabPath.path). Please ensure parakeet_vocab.json is downloaded with the models.")
            return [:]
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]

            var vocabulary: [Int: String] = [:]

            for (key, value) in jsonDict {
                if let tokenId = Int(key) {
                    vocabulary[tokenId] = value
                }
            }

            logger.info("✅ Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error("Failed to load or parse vocabulary file at \(vocabPath.path): \(error.localizedDescription)")
            return [:]
        }
    }

    private func loadModel(
        path: URL,
        name: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        do {
            return try MLModel(contentsOf: path, configuration: configuration)
        } catch {
            logger.error("Failed to load \(name) model: \(error)")

            throw ASRError.modelLoadFailed
        }
    }

    /// Load all required models in a single operation
    private func loadAllModels(
        melSpectrogramPath: URL,
        encoderPath: URL,
        decoderPath: URL,
        jointPath: URL,
        configuration: MLModelConfiguration
    ) async throws -> (melSpectrogram: MLModel, encoder: MLModel, decoder: MLModel, joint: MLModel) {
        async let melSpectrogram = loadModel(path: melSpectrogramPath, name: "mel-spectrogram", configuration: configuration)
        async let encoder = loadModel(path: encoderPath, name: "encoder", configuration: configuration)
        async let decoder = loadModel(path: decoderPath, name: "decoder", configuration: configuration)
        async let joint = loadModel(path: jointPath, name: "joint", configuration: configuration)

        return try await (melSpectrogram, encoder, decoder, joint)
    }

    private static func getDefaultModelsDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        let directory = appDirectory.appendingPathComponent("Models/Parakeet", isDirectory: true)

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    public func cleanup() {
        melSpectrogramModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        decoderState = DecoderState()
        logger.info("AsrManager resources cleaned up")
    }

    // MARK: - TDT Decoding

    /// TDT decoding using the dedicated TdtDecoder
    internal func tdtDecode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        originalAudioSamples: [Float]
    ) async throws -> [Int] {
        try await initializeDecoderState()

        let decoder = TdtDecoder(config: config)
        return try await decoder.decode(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            decoderModel: decoderModel!,
            jointModel: jointModel!,
            decoderState: &decoderState
        )
    }

    /// Transcribes audio samples to text using Parakeet TDT models.
    ///
    /// This method automatically handles long audio files by splitting them into 10-second chunks
    /// and concatenating the results. For optimal performance, audio should be:
    /// - 16kHz sample rate
    /// - Mono channel
    /// - Float32 samples normalized to [-1.0, 1.0]
    ///
    /// - Parameter audioSamples: Array of audio samples at 16kHz
    /// - Returns: `ASRResult` containing transcribed text, confidence, and timing information
    /// - Audio ≤10 seconds: Processed in a single pass
    /// - Audio >10 seconds: Automatically chunked with 0.5s overlap
    /// - RTFx includes full end-to-end time (I/O + inference)
    /// moved to asrtranscription.swift for better organization
    public func transcribe(_ audioSamples: [Float]) async throws -> ASRResult {
        return try await transcribeUnified(audioSamples)
    }

    /// Convert tokens to text using existing timing information from TDT
    internal func convertTokensWithExistingTimings(_ tokenIds: [Int], timings: [TokenTiming]) -> (text: String, timings: [TokenTiming]) {
        guard !tokenIds.isEmpty else { return ("", []) }

        let vocabulary = loadVocabulary()
        var result = ""
        var lastWasSpace = false
        var adjustedTimings: [TokenTiming] = []

        for (index, tokenId) in tokenIds.enumerated() {
            guard let token = vocabulary[tokenId], !token.isEmpty else { continue }

            let timing = index < timings.count ? timings[index] : nil

            if token.hasPrefix("▁") {
                let cleanToken = String(token.dropFirst())
                if !cleanToken.isEmpty {
                    if !result.isEmpty && !lastWasSpace { result += " " }
                    result += cleanToken
                    lastWasSpace = false

                    if let timing = timing {
                        adjustedTimings.append(TokenTiming(
                            token: cleanToken, tokenId: tokenId,
                            startTime: timing.startTime, endTime: timing.endTime,
                            confidence: timing.confidence
                        ))
                    }
                }
            } else {
                result += token
                lastWasSpace = false

                if let timing = timing {
                    adjustedTimings.append(TokenTiming(
                        token: token, tokenId: tokenId,
                        startTime: timing.startTime, endTime: timing.endTime,
                        confidence: timing.confidence
                    ))
                }
            }
        }

        return (result, adjustedTimings)
    }
    
    // MARK: - Error Handling Helpers
    
    /// Validates and extracts a required feature value from MLFeatureProvider
    internal func extractFeatureValue(from provider: MLFeatureProvider, key: String, errorMessage: String) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }
    
    /// Validates and extracts multiple feature values from MLFeatureProvider
    internal func extractFeatureValues(from provider: MLFeatureProvider, keys: [(key: String, errorSuffix: String)]) throws -> [String: MLMultiArray] {
        var results: [String: MLMultiArray] = [:]
        for (key, errorSuffix) in keys {
            results[key] = try extractFeatureValue(from: provider, key: key, errorMessage: "Invalid \(errorSuffix)")
        }
        return results
    }
}

