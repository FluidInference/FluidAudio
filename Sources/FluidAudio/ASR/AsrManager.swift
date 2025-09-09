import CoreML
import Foundation
import OSLog

public enum AudioSource: Sendable {
    case microphone
    case system
}

@available(macOS 13.0, *)
public final class AsrManager {

    internal let logger = AppLogger(category: "ASR")
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

    internal var microphoneDecoderState: TdtDecoderState
    internal var systemDecoderState: TdtDecoderState

    // Cached prediction options for reuse
    internal lazy var predictionOptions: MLPredictionOptions = {
        AsrModels.optimizedPredictionOptions()
    }()

    // Persistent feature providers for zero-copy model chaining
    private var zeroCopyProviders: [String: ZeroCopyFeatureProvider] = [:]

    public init(config: ASRConfig = .default) {
        self.config = config

        // Initialize decoder states with fallback
        do {
            self.microphoneDecoderState = try TdtDecoderState()
            self.systemDecoderState = try TdtDecoderState()
        } catch {
            logger.warning("Failed to create ANE-aligned decoder states, using standard allocation")
            // This should rarely happen, but if it does, we'll create them during first use
            self.microphoneDecoderState = TdtDecoderState(fallback: true)
            self.systemDecoderState = TdtDecoderState(fallback: true)
        }

        // Pre-warm caches if possible
        Task {
            await sharedMLArrayCache.prewarm(shapes: [
                ([1, 240000], .float32),
                ([1], .int32),
                ([2, 1, 640], .float32),
            ])
        }
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
        self.vocabulary = models.vocabulary

        logger.info("Token duration optimization model loaded successfully")

        logger.info("AsrManager initialized successfully with provided models")
    }

    private func createFeatureProvider(
        features: [(name: String, array: MLMultiArray)]
    ) throws
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

    func prepareMelSpectrogramInput(
        _ audioSamples: [Float], actualLength: Int? = nil
    ) async throws
        -> MLFeatureProvider
    {
        let audioLength = audioSamples.count
        let actualAudioLength = actualLength ?? audioLength  // Use provided actual length or default to sample count

        // Use ANE-aligned array from cache
        let audioArray = try await sharedMLArrayCache.getArray(
            shape: [1, audioLength] as [NSNumber],
            dataType: .float32
        )

        // Use optimized memory copy
        audioSamples.withUnsafeBufferPointer { buffer in
            let destPtr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: audioLength)
            memcpy(destPtr, buffer.baseAddress!, audioLength * MemoryLayout<Float>.stride)
        }

        // Pass the actual audio length, not the padded length
        let lengthArray = try createScalarArray(value: actualAudioLength)

        return try createFeatureProvider(features: [
            ("audio_signal", audioArray),
            ("audio_length", lengthArray),
        ])
    }

    func prepareEncoderInput(_ melspectrogramOutput: MLFeatureProvider) throws -> MLFeatureProvider {
        // Zero-copy: chain mel-spectrogram outputs directly to encoder inputs
        if let provider = ZeroCopyFeatureProvider.chain(
            from: melspectrogramOutput,
            outputName: "melspectrogram",
            to: "audio_signal"
        ) {
            // Also need to chain the length
            if let melLength = melspectrogramOutput.featureValue(for: "melspectrogram_length") {
                let features = [
                    "audio_signal": provider.featureValue(for: "audio_signal")!,
                    "length": melLength,
                ]
                return ZeroCopyFeatureProvider(features: features)
            }
        }

        // Fallback to copying if zero-copy fails
        let melspectrogram = try extractFeatureValue(
            from: melspectrogramOutput, key: "melspectrogram",
            errorMessage: "Invalid mel-spectrogram output")
        let melspectrogramLength = try extractFeatureValue(
            from: melspectrogramOutput, key: "melspectrogram_length",
            errorMessage: "Invalid mel-spectrogram length output")

        return try createFeatureProvider(features: [
            ("audio_signal", melspectrogram),
            ("length", melspectrogramLength),
        ])
    }

    private func prepareDecoderInput(
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try createScalarArray(value: 0, shape: [1, 1])
        let targetLengthArray = try createScalarArray(value: 1)

        return try createFeatureProvider(features: [
            ("targets", targetArray),
            ("target_lengths", targetLengthArray),
            ("h_in", hiddenState),
            ("c_in", cellState),
        ])
    }

    internal func initializeDecoderState(decoderState: inout TdtDecoderState) async throws {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        // Reset the existing decoder state to clear all cached values including predictorOutput
        decoderState.reset()

        let initDecoderInput = try prepareDecoderInput(
            hiddenState: decoderState.hiddenState,
            cellState: decoderState.cellState
        )

        let initDecoderOutput = try decoderModel.prediction(
            from: initDecoderInput, options: predictionOptions)

        decoderState.update(from: initDecoderOutput)

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
    ) async throws -> (melspectrogram: MLModel, encoder: MLModel, decoder: MLModel, joint: MLModel) {
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

    public func resetState() {
        microphoneDecoderState = TdtDecoderState(fallback: true)
        systemDecoderState = TdtDecoderState(fallback: true)
    }

    public func cleanup() {
        melspectrogramModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        // Reset decoder states - use fallback initializer that won't throw
        microphoneDecoderState = TdtDecoderState(fallback: true)
        systemDecoderState = TdtDecoderState(fallback: true)
        logger.info("AsrManager resources cleaned up")
    }

    internal func tdtDecodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        originalAudioSamples: [Float],
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> TdtHypothesis {
        let decoder = TdtDecoder(config: config)
        return try await decoder.decodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames,
            decoderModel: decoderModel!,
            jointModel: jointModel!,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )
    }

    public func transcribe(_ audioSamples: [Float]) async throws -> ASRResult {
        return try await transcribe(audioSamples, source: .microphone)
    }

    public func transcribe(_ audioSamples: [Float], source: AudioSource) async throws -> ASRResult {
        var result: ASRResult
        switch source {
        case .microphone:
            result = try await transcribeWithState(
                audioSamples, decoderState: &microphoneDecoderState)
        case .system:
            result = try await transcribeWithState(audioSamples, decoderState: &systemDecoderState)
        }

        // When batching audio, assume that the state needs to be reset comepletely between calls
        try await self.resetDecoderState()

        return result
    }

    // Reset both decoder states
    public func resetDecoderState() async throws {
        try await resetDecoderState(for: .microphone)
        try await resetDecoderState(for: .system)
    }

    /// Reset the decoder state for a specific audio source
    /// This should be called when starting a new transcription session or switching between different audio files
    public func resetDecoderState(for source: AudioSource) async throws {
        switch source {
        case .microphone:
            try await initializeDecoderState(decoderState: &microphoneDecoderState)
        case .system:
            try await initializeDecoderState(decoderState: &systemDecoderState)
        }
    }

    internal func convertTokensWithExistingTimings(
        _ tokenIds: [Int], timings: [TokenTiming]
    ) -> (
        text: String, timings: [TokenTiming]
    ) {
        guard !tokenIds.isEmpty else { return ("", []) }

        // SentencePiece-compatible decoding algorithm:
        // 1. Convert token IDs to token strings
        var tokens: [String] = []
        var tokenInfos: [(token: String, tokenId: Int, timing: TokenTiming?)] = []

        for (index, tokenId) in tokenIds.enumerated() {
            if let token = vocabulary[tokenId], !token.isEmpty {
                tokens.append(token)
                let timing = index < timings.count ? timings[index] : nil
                tokenInfos.append((token: token, tokenId: tokenId, timing: timing))
            }
        }

        // 2. Concatenate all tokens (this is how SentencePiece works)
        let concatenated = tokens.joined()

        // 3. Replace ▁ with space (SentencePiece standard)
        let text = concatenated.replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)

        // 4. For now, return original timings as-is
        // Note: Proper timing alignment would require tracking character positions
        // through the concatenation and replacement process
        let adjustedTimings = tokenInfos.compactMap { info in
            info.timing.map { timing in
                TokenTiming(
                    token: info.token.replacingOccurrences(of: "▁", with: ""),
                    tokenId: info.tokenId,
                    startTime: timing.startTime,
                    endTime: timing.endTime,
                    confidence: timing.confidence
                )
            }
        }

        return (text, adjustedTimings)
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
