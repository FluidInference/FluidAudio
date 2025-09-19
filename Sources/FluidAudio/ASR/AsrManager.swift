import AVFoundation
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
    private let audioConverter: AudioConverter = AudioConverter()

    internal var melEncoderModel: MLModel?
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

    // TODO:: the decoder state should be moved higher up in the API interface
    internal var microphoneDecoderState: TdtDecoderState
    internal var systemDecoderState: TdtDecoderState

    // Cached prediction options for reuse
    internal lazy var predictionOptions: MLPredictionOptions = {
        AsrModels.optimizedPredictionOptions()
    }()

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
        return melEncoderModel != nil && decoderModel != nil && jointModel != nil
    }

    /// Initialize ASR Manager with pre-loaded models
    /// - Parameter models: Pre-loaded ASR models
    public func initialize(models: AsrModels) async throws {
        logger.info("Initializing AsrManager with provided models")

        self.asrModels = models
        self.melEncoderModel = models.melEncoder
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

    func prepareMelEncoderInput(
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

    func normalizeEncoderOutput(_ encoderOutput: MLMultiArray) throws -> MLMultiArray {
        let shape = encoderOutput.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output rank: \(shape)")
        }

        let expectedHiddenSize = ASRConstants.encoderHiddenSize

        // If the hidden dimension is already the last axis, nothing to do.
        if shape[2] == expectedHiddenSize {
            return encoderOutput
        }

        // Handle models that emit [batch, hidden, time] by transposing to [batch, time, hidden].
        if shape[1] == expectedHiddenSize {
            return try transposeHiddenAndTimeAxes(encoderOutput)
        }

        // Fallback: if shape mismatch but last dimension still looks like time, return as-is.
        return encoderOutput
    }

    private func transposeHiddenAndTimeAxes(_ encoderOutput: MLMultiArray) throws -> MLMultiArray {
        let shape = encoderOutput.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output rank: \(shape)")
        }

        let batch = shape[0]
        let hidden = shape[1]
        let time = shape[2]

        let transposedShape = [NSNumber(value: batch), NSNumber(value: time), NSNumber(value: hidden)]
        let result = try MLMultiArray(shape: transposedShape, dataType: encoderOutput.dataType)

        let srcStrides = encoderOutput.strides.map { $0.intValue }
        let dstStrides = result.strides.map { $0.intValue }

        switch encoderOutput.dataType {
        case .float32:
            let srcPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
            let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)

            for b in 0..<batch {
                for t in 0..<time {
                    for h in 0..<hidden {
                        let srcIndex = b * srcStrides[0] + h * srcStrides[1] + t * srcStrides[2]
                        let dstIndex = b * dstStrides[0] + t * dstStrides[1] + h * dstStrides[2]
                        dstPtr[dstIndex] = srcPtr[srcIndex]
                    }
                }
            }

        default:
            for b in 0..<batch {
                for t in 0..<time {
                    for h in 0..<hidden {
                        let srcIndex = b * srcStrides[0] + h * srcStrides[1] + t * srcStrides[2]
                        let dstIndex = b * dstStrides[0] + t * dstStrides[1] + h * dstStrides[2]
                        result[dstIndex] = encoderOutput[srcIndex]
                    }
                }
            }
        }

        return result
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

        // Compat helper awaits async prediction when available without breaking older SDKs.
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
            return try MLModel(contentsOf: path, configuration: configuration)
        } catch {
            logger.error("Failed to load \(name) model: \(error)")

            throw ASRError.modelLoadFailed
        }
    }

    private func loadAllModels(
        melEncoderPath: URL,
        decoderPath: URL,
        jointPath: URL,
        configuration: MLModelConfiguration
    ) async throws -> (melEncoder: MLModel, decoder: MLModel, joint: MLModel) {
        async let melEncoder = loadModel(
            path: melEncoderPath, name: "mel-encoder", configuration: configuration)
        async let decoder = loadModel(
            path: decoderPath, name: "decoder", configuration: configuration)
        async let joint = loadModel(path: jointPath, name: "joint", configuration: configuration)

        return try await (melEncoder, decoder, joint)
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
        melEncoderModel = nil
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

    public func transcribe(_ audioBuffer: AVAudioPCMBuffer, source: AudioSource = .microphone) async throws -> ASRResult
    {
        let audioFloatArray = try audioConverter.resampleBuffer(audioBuffer)

        let result = try await transcribe(audioFloatArray, source: source)

        return result
    }

    public func transcribe(_ url: URL, source: AudioSource = .system) async throws -> ASRResult {
        let audioFloatArray = try audioConverter.resampleAudioFile(url)

        let result = try await transcribe(audioFloatArray, source: source)

        return result
    }

    public func transcribe(
        _ audioSamples: [Float],
        source: AudioSource = .microphone
    ) async throws -> ASRResult {
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
