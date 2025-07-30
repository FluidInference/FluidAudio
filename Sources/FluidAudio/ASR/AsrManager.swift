import CoreML
import Foundation
import OSLog

public enum AudioSource: Sendable {
    case microphone
    case system
}

@available(macOS 13.0, *)
public final class AsrManager {

    internal let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ASR")
    internal let config: ASRConfig

    internal var melspectrogramModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?

    /// The AsrModels instance if initialized with models
    private var asrModels: AsrModels?

    /// Token duration optimization model  

    /// Cached vocabulary loaded once during initialization
    internal var vocabulary: [Int: String] = [:]
    #if DEBUG
        // Test-only setter
        internal func setVocabularyForTesting(_ vocab: [Int: String]) {
            vocabulary = vocab
        }
    #endif

    private var microphoneDecoderState = DecoderState()
    private var systemDecoderState = DecoderState()

    let blankId = 1024
    let sosId = 1024

    public init(config: ASRConfig = .default) {
        self.config = config
        logger.info("TDT enabled with durations: \(config.tdtConfig.durations)")

        // Optimization models will be loaded during initialize()
        
        // Load vocabulary once during initialization
        self.vocabulary = loadVocabulary()
    }

    public var isAvailable: Bool {
        return melspectrogramModel != nil && encoderModel != nil && decoderModel != nil
            && jointModel != nil
    }

    /// Initialize ASR Manager with pre-loaded models
    /// - Parameter models: Pre-loaded ASR models
    public func initialize(models: AsrModels) async throws {
        logger.info("Initializing AsrManager with provided models")

        self.asrModels = models
        self.melspectrogramModel = models.melspectrogram
        self.encoderModel = models.encoder
        self.decoderModel = models.decoder
        self.jointModel = models.joint

        logger.info("Token duration optimization model loaded successfully")

        logger.info("AsrManager initialized successfully with provided models")
    }

    /// Initialize ASR Manager by downloading and loading models from default location
    /// - Note: This method is deprecated. Use AsrModels.downloadAndLoad() followed by initialize(models:) instead
    @available(
        *, deprecated,
        message:
            "Use AsrModels.downloadAndLoad() followed by initialize(models:) for more control over model loading"
    )
    public func initialize() async throws {
        logger.info("Initializing AsrManager with automatic model download (deprecated)")

        do {
            // Download and load models using the new AsrModels API
            let models = try await AsrModels.downloadAndLoad()
            try await initialize(models: models)
        } catch {
            logger.error("Failed to initialize AsrManager: \(error.localizedDescription)")
            throw ASRError.modelLoadFailed
        }
    }

    private func createFeatureProvider(features: [(name: String, array: MLMultiArray)]) throws
        -> MLFeatureProvider
    {
        var featureDict: [String: MLFeatureValue] = [:]
        for (name, array) in features {
            featureDict[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: featureDict)
    }

    internal func createScalarArray(
        value: Int, shape: [NSNumber] = [1], dataType: MLMultiArrayDataType = .int32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }

    func prepareMelSpectrogramInput(_ audioSamples: [Float]) throws -> MLFeatureProvider {
        let audioLength = audioSamples.count

        let audioArray = try MLMultiArray(
            shape: [1, audioLength] as [NSNumber], dataType: .float32)
        for i in 0..<audioLength {
            audioArray[i] = NSNumber(value: audioSamples[i])
        }

        let lengthArray = try createScalarArray(value: audioLength)

        return try createFeatureProvider(features: [
            ("audio_signal", audioArray),
            ("audio_length", lengthArray),
        ])
    }

    func prepareEncoderInput(_ melspectrogramOutput: MLFeatureProvider) throws -> MLFeatureProvider
    {
        let melspectrogram = try extractFeatureValue(
            from: melspectrogramOutput, key: "melspectogram",
            errorMessage: "Invalid mel-spectrogram output")
        let melspectrogramLength = try extractFeatureValue(
            from: melspectrogramOutput, key: "melspectogram_length",
            errorMessage: "Invalid mel-spectrogram length output")

        return try createFeatureProvider(features: [
            ("audio_signal", melspectrogram),
            ("length", melspectrogramLength),
        ])
    }


    func prepareDecoderInput(
        targetToken: Int,
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try createScalarArray(value: targetToken, shape: [1, 1])
        let targetLengthArray = try createScalarArray(value: 1)

        return try createFeatureProvider(features: [
            ("targets", targetArray),
            ("target_lengths", targetLengthArray),
            ("h_in", hiddenState),
            ("c_in", cellState),
        ])
    }

    internal func initializeDecoderState(decoderState: inout DecoderState) async throws {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        var freshState = DecoderState()

        let initDecoderInput = try prepareDecoderInput(
            targetToken: blankId,
            hiddenState: freshState.hiddenState,
            cellState: freshState.cellState
        )

        let initDecoderOutput = try decoderModel.prediction(
            from: initDecoderInput, options: MLPredictionOptions())

        freshState.update(from: initDecoderOutput)

        if config.enableDebug {
            logger.info("Decoder state initialized cleanly")
        }

        decoderState = freshState
    }
    private func loadVocabulary() -> [Int: String] {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent(
            "FluidAudio", isDirectory: true
        )
        .appendingPathComponent("Models", isDirectory: true)
        .appendingPathComponent("parakeet-tdt-0.6b-v2-coreml", isDirectory: true)
        let vocabPath = appDirectory.appendingPathComponent("parakeet_vocab.json")

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "Vocabulary file not found at \(vocabPath.path). Please ensure parakeet_vocab.json is downloaded with the models."
            )
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

            logger.info(
                "✅ Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error(
                "Failed to load or parse vocabulary file at \(vocabPath.path): \(error.localizedDescription)"
            )
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

    private func loadAllModels(
        melspectrogramPath: URL,
        encoderPath: URL,
        decoderPath: URL,
        jointPath: URL,
        configuration: MLModelConfiguration
    ) async throws -> (melspectrogram: MLModel, encoder: MLModel, decoder: MLModel, joint: MLModel)
    {
        async let melspectrogram = loadModel(
            path: melspectrogramPath, name: "mel-spectrogram", configuration: configuration)
        async let encoder = loadModel(
            path: encoderPath, name: "encoder", configuration: configuration)
        async let decoder = loadModel(
            path: decoderPath, name: "decoder", configuration: configuration)
        async let joint = loadModel(path: jointPath, name: "joint", configuration: configuration)

        return try await (melspectrogram, encoder, decoder, joint)
    }

    private static func getDefaultModelsDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent(
            "FluidAudio", isDirectory: true)
        let directory = appDirectory.appendingPathComponent("Models/Parakeet", isDirectory: true)

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    public func cleanup() {
        melspectrogramModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        microphoneDecoderState = DecoderState()
        systemDecoderState = DecoderState()
        logger.info("AsrManager resources cleaned up")
    }

    internal func tdtDecode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        originalAudioSamples: [Float],
        decoderState: inout DecoderState
    ) async throws -> [Int] {
        try await initializeDecoderState(decoderState: &decoderState)

        let decoder = TdtDecoder(config: config)
        return try await decoder.decode(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            decoderModel: decoderModel!,
            jointModel: jointModel!,
            decoderState: &decoderState
        )
    }

    public func transcribe(_ audioSamples: [Float]) async throws -> ASRResult {
        return try await transcribe(audioSamples, source: .microphone)
    }

    public func transcribe(_ audioSamples: [Float], source: AudioSource) async throws -> ASRResult {
        switch source {
        case .microphone:
            return try await transcribeWithState(
                audioSamples, decoderState: &microphoneDecoderState)
        case .system:
            return try await transcribeWithState(audioSamples, decoderState: &systemDecoderState)
        }
    }

    internal func transcribeWithState(_ audioSamples: [Float], decoderState: inout DecoderState)
        async throws -> ASRResult
    {
        return try await transcribeUnifiedWithState(audioSamples, decoderState: &decoderState)
    }

    internal func convertTokensWithExistingTimings(_ tokenIds: [Int], timings: [TokenTiming]) -> (
        text: String, timings: [TokenTiming]
    ) {
        guard !tokenIds.isEmpty else { return ("", []) }

        // Fallback: if vocabulary is empty (failed to load during init), try loading it now
        if vocabulary.isEmpty {
            vocabulary = loadVocabulary()
        }

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
                        adjustedTimings.append(
                            TokenTiming(
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
                    adjustedTimings.append(
                        TokenTiming(
                            token: token, tokenId: tokenId,
                            startTime: timing.startTime, endTime: timing.endTime,
                            confidence: timing.confidence
                        ))
                }
            }
        }

        return (result, adjustedTimings)
    }

    internal func extractFeatureValue(
        from provider: MLFeatureProvider, key: String, errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }

    internal func extractFeatureValues(
        from provider: MLFeatureProvider, keys: [(key: String, errorSuffix: String)]
    ) throws -> [String: MLMultiArray] {
        var results: [String: MLMultiArray] = [:]
        for (key, errorSuffix) in keys {
            results[key] = try extractFeatureValue(
                from: provider, key: key, errorMessage: "Invalid \(errorSuffix)")
        }
        return results
    }
}
