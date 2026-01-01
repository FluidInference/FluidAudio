import AVFoundation
@preconcurrency import CoreML
import Foundation
import OSLog

public enum AudioSource: Sendable {
    case microphone
    case system
}

public final class AsrManager {

    internal let logger = AppLogger(category: "ASR")
    internal let config: ASRConfig
    private let audioConverter: AudioConverter = AudioConverter()

    internal var preprocessorModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?

    /// The AsrModels instance if initialized with models
    private var asrModels: AsrModels?

    internal let progressEmitter = ProgressEmitter()

    /// Token duration optimization model

    /// Cached vocabulary loaded once during initialization
    internal var vocabulary: [Int: String] = [:]
    #if DEBUG
    // Test-only setter
    internal func setVocabularyForTesting(_ vocab: [Int: String]) {
        vocabulary = vocab
    }
    #endif

    // TODO:: the decoder state should be moved higher up in the API interface
    internal var microphoneDecoderState: TdtDecoderState
    internal var systemDecoderState: TdtDecoderState

    // Cached prediction options for reuse
    internal lazy var predictionOptions: MLPredictionOptions = {
        AsrModels.optimizedPredictionOptions()
    }()

    public init(config: ASRConfig = .default) {
        self.config = config

        self.microphoneDecoderState = TdtDecoderState.make()
        self.systemDecoderState = TdtDecoderState.make()

        // Pre-warm caches if possible
        Task {
            await sharedMLArrayCache.prewarm(shapes: [
                ([NSNumber(value: 1), NSNumber(value: 240_000)], .float32),
                ([NSNumber(value: 1)], .int32),
                (
                    [
                        NSNumber(value: 2),
                        NSNumber(value: 1),
                        NSNumber(value: ASRConstants.decoderHiddenSize),
                    ], .float32
                ),
            ])
        }

        let emitter = progressEmitter
        Task {
            await emitter.ensureSession()
        }
    }

    /// Returns the current transcription progress stream for offline long audio (>\(240_000) samples).
    /// Only one session is supported at a time.
    public var transcriptionProgressStream: AsyncThrowingStream<Double, Error> {
        get async {
            await progressEmitter.currentStream()
        }
    }

    public var isAvailable: Bool {
        let baseModelsReady = encoderModel != nil && decoderModel != nil && jointModel != nil
        guard baseModelsReady else { return false }

        if asrModels?.usesSplitFrontend == true {
            return preprocessorModel != nil
        }

        return true
    }

    /// Initialize ASR Manager with pre-loaded models
    /// - Parameter models: Pre-loaded ASR models
    public func initialize(models: AsrModels) async throws {
        logger.info("Initializing AsrManager with provided models")

        self.asrModels = models
        self.preprocessorModel = models.preprocessor
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

    func preparePreprocessorInput(
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

    private func prepareDecoderInput(
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try createScalarArray(value: 0, shape: [1, 1])
        let targetLengthArray = try createScalarArray(value: 1)

        return try createFeatureProvider(features: [
            ("targets", targetArray),
            ("target_length", targetLengthArray),
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

        let initDecoderOutput = try await decoderModel.compatPrediction(
            from: initDecoderInput,
            options: predictionOptions
        )

        decoderState.update(from: initDecoderOutput)

    }

    private func loadModel(
        path: URL,
        name: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        do {
            let model = try MLModel(contentsOf: path, configuration: configuration)
            return model
        } catch {
            logger.error("Failed to load \(name) model: \(error)")

            throw ASRError.modelLoadFailed
        }
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
        microphoneDecoderState = TdtDecoderState.make()
        systemDecoderState = TdtDecoderState.make()
    }

    public func cleanup() {
        preprocessorModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        // Reset decoder states using fresh allocations for deterministic behavior
        microphoneDecoderState = TdtDecoderState.make()
        systemDecoderState = TdtDecoderState.make()
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
        // Route to appropriate decoder based on model version
        switch asrModels!.version {
        case .v2:
            let decoder = TdtDecoderV2(config: config)
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
        case .v3:
            let decoder = TdtDecoderV3(config: config)
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
    }

    /// Transcribe audio from an AVAudioPCMBuffer.
    ///
    /// Performs speech-to-text transcription on the provided audio buffer. The decoder state is automatically
    /// reset after transcription completes, ensuring each transcription call is independent. This enables
    /// efficient batch processing where multiple files are transcribed without state carryover.
    ///
    /// - Parameters:
    ///   - audioBuffer: The audio buffer to transcribe
    ///   - source: The audio source type (microphone or system audio)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(_ audioBuffer: AVAudioPCMBuffer, source: AudioSource = .microphone) async throws -> ASRResult
    {
        let audioFloatArray = try audioConverter.resampleBuffer(audioBuffer)

        let result = try await transcribe(audioFloatArray, source: source)

        return result
    }

    /// Transcribe audio from a file URL.
    ///
    /// Performs speech-to-text transcription on the audio file at the provided URL. The decoder state is
    /// automatically reset after transcription completes, ensuring each transcription call is independent.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - source: The audio source type (defaults to .system)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribe(_ url: URL, source: AudioSource = .system) async throws -> ASRResult {
        let audioFloatArray = try audioConverter.resampleAudioFile(url)

        let result = try await transcribe(audioFloatArray, source: source)

        return result
    }

    /// Transcribe audio from raw float samples.
    ///
    /// Performs speech-to-text transcription on raw audio samples at 16kHz. The decoder state is
    /// automatically reset after transcription completes, ensuring each transcription call is independent
    /// and enabling efficient batch processing of multiple audio files.
    ///
    /// - Parameters:
    ///   - audioSamples: Array of 16-bit audio samples at 16kHz
    ///   - source: The audio source type (microphone or system audio)
    /// - Note: Progress stream is emitted only when `audioSamples.count > 240_000` (~15s).
    ///         Use `transcriptionProgressStream` before calling this method to observe progress.
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioSamples: [Float],
        source: AudioSource = .microphone
    ) async throws -> ASRResult {
        let shouldEmitProgress = audioSamples.count > 240_000
        if shouldEmitProgress {
            _ = await progressEmitter.ensureSession()
        }
        do {
            let result: ASRResult
            switch source {
            case .microphone:
                result = try await transcribeWithState(
                    audioSamples, decoderState: &microphoneDecoderState)
            case .system:
                result = try await transcribeWithState(
                    audioSamples, decoderState: &systemDecoderState)
            }

            // Stateless architecture: reset decoder state after each transcription to ensure
            // independent processing for batch operations without state carryover
            try await self.resetDecoderState()
            if shouldEmitProgress {
                await progressEmitter.finishSession()
            }

            return result
        } catch {
            if shouldEmitProgress {
                await progressEmitter.failSession(error)
            }
            throw error
        }
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

    internal func normalizedTimingToken(_ token: String) -> String {
        token.replacingOccurrences(of: "▁", with: " ")
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
                    token: normalizedTimingToken(info.token),
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
