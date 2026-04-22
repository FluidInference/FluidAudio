import AVFoundation
@preconcurrency import CoreML
import Foundation
import OSLog

public actor AsrManager {

    internal let logger = AppLogger(category: "ASR")
    internal let config: ASRConfig
    private let audioConverter: AudioConverter = AudioConverter()

    internal var preprocessorModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?

    /// The AsrModels instance if initialized with models
    internal var asrModels: AsrModels?

    internal let progressEmitter = ProgressEmitter()

    /// Number of decoder layers for the current model.
    /// Returns 2 if models not loaded (v2/v3 default, tdtCtc110m uses 1).
    public var decoderLayerCount: Int {
        asrModels?.version.decoderLayers ?? 2
    }

    internal var parallelChunkConcurrency: Int {
        config.parallelChunkConcurrency
    }

    /// Cached vocabulary loaded once during initialization
    internal var vocabulary: [Int: String] = [:]
    #if DEBUG
    // Test-only setter
    internal func setVocabularyForTesting(_ vocab: [Int: String]) {
        vocabulary = vocab
    }
    #endif

    // Cached prediction options for reuse
    internal lazy var predictionOptions: MLPredictionOptions = {
        AsrModels.optimizedPredictionOptions()
    }()

    public init(config: ASRConfig = .default, models: AsrModels? = nil) {
        self.config = config

        if let models {
            self.asrModels = models
            self.preprocessorModel = models.preprocessor
            self.encoderModel = models.encoder
            self.decoderModel = models.decoder
            self.jointModel = models.joint
            self.vocabulary = models.vocabulary
        }

        // Pre-warm caches if possible
        Task {
            await sharedMLArrayCache.prewarm(shapes: [
                ([NSNumber(value: 1), NSNumber(value: ASRConstants.maxModelSamples)], .float32),
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

    internal func makeWorkerClone() -> AsrManager? {
        guard let models = asrModels else { return nil }
        return AsrManager(config: config, models: models)
    }

    /// Returns the current transcription progress stream for offline long audio (>240,000 samples / ~15s).
    /// Only one session is supported at a time.
    public var transcriptionProgressStream: AsyncThrowingStream<Double, Error> {
        get async {
            await progressEmitter.ensureSession()
        }
    }

    public var isAvailable: Bool {
        let decoderReady = decoderModel != nil && jointModel != nil
        guard decoderReady else { return false }

        if asrModels?.usesSplitFrontend == true {
            // Split frontend: need both preprocessor and encoder
            return preprocessorModel != nil && encoderModel != nil
        } else {
            // Fused frontend: preprocessor contains encoder
            return preprocessorModel != nil
        }
    }

    /// Load pre-loaded ASR models into this manager.
    /// - Parameter models: Pre-loaded ASR models
    public func loadModels(_ models: AsrModels) async throws {
        logger.info("Loading AsrManager with provided models")

        self.asrModels = models
        self.preprocessorModel = models.preprocessor
        self.encoderModel = models.encoder
        self.decoderModel = models.decoder
        self.jointModel = models.joint
        self.vocabulary = models.vocabulary

        logger.info("AsrManager loaded successfully with provided models")
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

    public func reset() {
        Task { await sharedMLArrayCache.clear() }
    }

    public func cleanup() {
        asrModels = nil
        preprocessorModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        Task { await sharedMLArrayCache.clear() }
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
        guard let models = asrModels, let decoder_ = decoderModel, let joint = jointModel else {
            throw ASRError.notInitialized
        }

        // Adapt config to match the loaded model version
        // Step 1: Adapt blankId if needed (e.g. default 8192 but tdtJa needs 3072)
        var workingConfig = config
        if config.tdtConfig.blankId != models.version.blankId {
            let adaptedTdtConfig = TdtConfig(
                includeTokenDuration: config.tdtConfig.includeTokenDuration,
                maxSymbolsPerStep: config.tdtConfig.maxSymbolsPerStep,
                durationBins: config.tdtConfig.durationBins,
                blankId: models.version.blankId,
                boundarySearchFrames: config.tdtConfig.boundarySearchFrames,
                maxTokensPerChunk: config.tdtConfig.maxTokensPerChunk,
                consecutiveBlankLimit: config.tdtConfig.consecutiveBlankLimit
            )

            workingConfig = ASRConfig(
                sampleRate: workingConfig.sampleRate,
                tdtConfig: adaptedTdtConfig,
                encoderHiddenSize: workingConfig.encoderHiddenSize,
                parallelChunkConcurrency: workingConfig.parallelChunkConcurrency,
                streamingEnabled: workingConfig.streamingEnabled,
                streamingThreshold: workingConfig.streamingThreshold
            )
        }

        // Step 2: Adapt encoderHiddenSize if needed (e.g. default 1024 but tdtCtc110m needs 512)
        let adaptedConfig: ASRConfig
        if workingConfig.encoderHiddenSize != models.version.encoderHiddenSize {
            adaptedConfig = ASRConfig(
                sampleRate: workingConfig.sampleRate,
                tdtConfig: workingConfig.tdtConfig,
                encoderHiddenSize: models.version.encoderHiddenSize,
                parallelChunkConcurrency: workingConfig.parallelChunkConcurrency,
                streamingEnabled: workingConfig.streamingEnabled,
                streamingThreshold: workingConfig.streamingThreshold
            )
        } else {
            adaptedConfig = workingConfig
        }

        switch models.version {
        case .v2, .tdtCtc110m:
            let decoder = TdtDecoderV2(config: adaptedConfig)
            return try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualAudioFrames,
                decoderModel: decoder_,
                jointModel: joint,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )
        case .v3, .tdtJa:
            let decoder = TdtDecoderV3(config: adaptedConfig)
            return try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualAudioFrames,
                decoderModel: decoder_,
                jointModel: joint,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )
        case .ctcZhCn:
            throw ASRError.processingFailed(
                "CTC-only model .ctcZhCn does not support TDT decoding. Use CtcZhCnManager instead."
            )
        }
    }

    /// Transcribe audio from an AVAudioPCMBuffer.
    ///
    /// Performs speech-to-text transcription on the provided audio buffer.
    ///
    /// - Parameters:
    ///   - audioBuffer: The audio buffer to transcribe
    ///   - decoderState: The TDT decoder state to use and update during transcription
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioBuffer: AVAudioPCMBuffer, decoderState: inout TdtDecoderState
    ) async throws -> ASRResult {
        let audioFloatArray = try audioConverter.resampleBuffer(audioBuffer)

        let result = try await transcribe(audioFloatArray, decoderState: &decoderState)

        return result
    }

    /// Transcribe audio from a file URL.
    ///
    /// Performs speech-to-text transcription on the audio file at the provided URL.
    ///
    /// For large files (exceeding `config.streamingThreshold`), automatically uses disk-backed processing
    /// to maintain constant memory usage regardless of file size.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - decoderState: The TDT decoder state to use and update during transcription
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribe(_ url: URL, decoderState: inout TdtDecoderState) async throws -> ASRResult {
        // Check file size to decide streaming vs memory loading
        if config.streamingEnabled {
            let audioFile = try AVAudioFile(forReading: url)
            let inputFormat = audioFile.processingFormat
            let sampleRateRatio = Double(config.sampleRate) / inputFormat.sampleRate
            let estimatedSamples = Int((Double(audioFile.length) * sampleRateRatio).rounded(.up))

            if estimatedSamples > config.streamingThreshold {
                return try await transcribeDiskBacked(url, decoderState: &decoderState)
            }
        }

        let audioFloatArray = try audioConverter.resampleAudioFile(url)
        let result = try await transcribe(audioFloatArray, decoderState: &decoderState)
        return result
    }

    /// Transcribe audio from a file URL using disk-backed chunked processing.
    ///
    /// Memory-efficient transcription that memory-maps the file and processes audio in chunks,
    /// maintaining constant memory usage (~1.2MB) regardless of file size. Ideal for long audio files.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - decoderState: The TDT decoder state to use and update during transcription
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribeDiskBacked(_ url: URL, decoderState: inout TdtDecoderState) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }

        let startTime = Date()

        // Create a disk-backed source for memory-efficient access
        let factory = AudioSourceFactory()
        let (sampleSource, _) = try factory.makeDiskBackedSource(
            from: url,
            targetSampleRate: config.sampleRate
        )

        let totalSamples = sampleSource.sampleCount
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard totalSamples >= minimumRequiredSamples else {
            sampleSource.cleanup()
            throw ASRError.invalidAudioData
        }

        let shouldEmitProgress = totalSamples > 240_000
        if shouldEmitProgress {
            _ = await progressEmitter.ensureSession()
        }

        do {
            let processor = ChunkProcessor(sampleSource: sampleSource)
            let result = try await processor.process(
                using: self,
                startTime: startTime,
                progressHandler: { [weak self] progress in
                    guard let self else { return }
                    await self.progressEmitter.report(progress: progress)
                }
            )

            sampleSource.cleanup()

            if shouldEmitProgress {
                await progressEmitter.finishSession()
            }

            return result
        } catch {
            sampleSource.cleanup()
            if shouldEmitProgress {
                await progressEmitter.failSession(error)
            }
            throw error
        }
    }

    /// Transcribe audio from raw float samples.
    ///
    /// Performs speech-to-text transcription on raw audio samples at 16kHz.
    ///
    /// - Parameters:
    ///   - audioSamples: Array of 16-bit audio samples at 16kHz
    ///   - decoderState: The TDT decoder state to use and update during transcription
    /// - Note: Progress stream is emitted only when `audioSamples.count > ASRConstants.maxModelSamples` (~15s).
    ///         Use `transcriptionProgressStream` before calling this method to observe progress.
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioSamples: [Float],
        decoderState: inout TdtDecoderState
    ) async throws -> ASRResult {
        let shouldEmitProgress = audioSamples.count > ASRConstants.maxModelSamples
        if shouldEmitProgress {
            _ = await progressEmitter.ensureSession()
        }
        do {
            let result = try await transcribeWithState(audioSamples, decoderState: &decoderState)

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

    nonisolated internal func normalizedTimingToken(_ token: String) -> String {
        token.replacingOccurrences(of: "▁", with: " ")
    }

    /// Decode token IDs to text using SentencePiece conventions.
    internal func convertTokensToText(_ tokenIds: [Int]) -> String {
        guard !tokenIds.isEmpty else { return "" }

        let tokens = tokenIds.compactMap { vocabulary[$0] }.filter { !$0.isEmpty }
        return tokens.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    nonisolated internal func extractFeatureValue(
        from provider: MLFeatureProvider, key: String, errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }

    nonisolated internal func extractFeatureValues(
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
