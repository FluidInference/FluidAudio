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
    internal var decoderLayerCount: Int {
        asrModels?.version.decoderLayers ?? 2
    }

    /// Cached vocabulary loaded once during initialization
    internal var vocabulary: [Int: String] = [:]
    #if DEBUG
    // Test-only setter
    internal func setVocabularyForTesting(_ vocab: [Int: String]) {
        vocabulary = vocab
    }
    #endif

    // Per-source decoder states are actor-internal; callers reset via resetDecoderState().
    internal var microphoneDecoderState: TdtDecoderState
    internal var systemDecoderState: TdtDecoderState

    /// Get decoder state for a given audio source.
    internal func decoderState(for source: AudioSource) -> TdtDecoderState {
        switch source {
        case .microphone: return microphoneDecoderState
        case .system: return systemDecoderState
        }
    }

    /// Set decoder state for a given audio source.
    internal func setDecoderState(_ state: TdtDecoderState, for source: AudioSource) {
        switch source {
        case .microphone: microphoneDecoderState = state
        case .system: systemDecoderState = state
        }
    }

    // Cached CTC logits from fused Preprocessor (unified custom vocabulary)
    internal var cachedCtcLogits: MLMultiArray?
    internal var cachedCtcFrameDuration: Double?
    internal var cachedCtcValidFrames: Int?

    /// Whether the Preprocessor outputs CTC logits (unified custom vocabulary model).
    public var hasCachedCtcLogits: Bool { cachedCtcLogits != nil }

    /// Clear all cached CTC data (logits, frame duration, valid frames).
    internal func clearCachedCtcData() {
        cachedCtcLogits = nil
        cachedCtcFrameDuration = nil
        cachedCtcValidFrames = nil
    }

    /// Get cached CTC raw logits as [[Float]] for external use (e.g. benchmarks).
    /// These are raw logits — callers must apply `CtcKeywordSpotter.applyLogSoftmax()`
    /// to convert to log-probabilities before use in keyword detection.
    /// Returns nil if the CTC head model is not available or audio was multi-chunk.
    public func getCachedCtcRawLogits() -> (rawLogits: [[Float]], frameDuration: Double)? {
        guard let logits = cachedCtcLogits, let duration = cachedCtcFrameDuration else { return nil }
        let shape = logits.shape
        guard shape.count == 3 else { return nil }
        let numFrames = min(shape[1].intValue, cachedCtcValidFrames ?? shape[1].intValue)
        let vocabSize = shape[2].intValue
        var result: [[Float]] = []
        result.reserveCapacity(numFrames)
        for t in 0..<numFrames {
            var frame: [Float] = []
            frame.reserveCapacity(vocabSize)
            for v in 0..<vocabSize {
                frame.append(logits[[0, t, v] as [NSNumber]].floatValue)
            }
            result.append(frame)
        }
        return (rawLogits: result, frameDuration: duration)
    }

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

        // Recreate decoder states with the correct layer count for this model version
        let layers = models.version.decoderLayers
        self.microphoneDecoderState = TdtDecoderState.make(decoderLayers: layers)
        self.systemDecoderState = TdtDecoderState.make(decoderLayers: layers)

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

    internal func initializeDecoderState(for source: AudioSource) async throws {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        var state = decoderState(for: source)
        state.reset()

        let initDecoderInput = try prepareDecoderInput(
            hiddenState: state.hiddenState,
            cellState: state.cellState
        )

        let initDecoderOutput = try await decoderModel.compatPrediction(
            from: initDecoderInput,
            options: predictionOptions
        )

        state.update(from: initDecoderOutput)
        setDecoderState(state, for: source)
    }

    public func resetState() {
        // Use model's decoder layer count, or 2 if models not loaded (v2/v3 default)
        let layers = asrModels?.version.decoderLayers ?? 2
        microphoneDecoderState = TdtDecoderState.make(decoderLayers: layers)
        systemDecoderState = TdtDecoderState.make(decoderLayers: layers)
        clearCachedCtcData()
        Task { await sharedMLArrayCache.clear() }
    }

    public func cleanup() {
        // Capture layer count before releasing models, fallback to 2 (v2/v3 default)
        let layers = asrModels?.version.decoderLayers ?? 2
        asrModels = nil
        preprocessorModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        // Reset decoder states using fresh allocations for deterministic behavior
        microphoneDecoderState = TdtDecoderState.make(decoderLayers: layers)
        systemDecoderState = TdtDecoderState.make(decoderLayers: layers)
        clearCachedCtcData()
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

        // Adapt config's encoderHiddenSize to match the loaded model version
        // (e.g. default config uses 1024 but tdtCtc110m needs 512)
        let adaptedConfig: ASRConfig
        if config.encoderHiddenSize != models.version.encoderHiddenSize {
            adaptedConfig = ASRConfig(
                sampleRate: config.sampleRate,
                tdtConfig: config.tdtConfig,
                encoderHiddenSize: models.version.encoderHiddenSize,
                streamingEnabled: config.streamingEnabled,
                streamingThreshold: config.streamingThreshold
            )
        } else {
            adaptedConfig = config
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
        case .v3:
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
    /// For large files (exceeding `config.streamingThreshold`), automatically uses streaming mode
    /// to maintain constant memory usage regardless of file size.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - source: The audio source type (defaults to .system)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribe(_ url: URL, source: AudioSource = .system) async throws -> ASRResult {
        // Check file size to decide streaming vs memory loading
        if config.streamingEnabled {
            let audioFile = try AVAudioFile(forReading: url)
            let inputFormat = audioFile.processingFormat
            let sampleRateRatio = Double(config.sampleRate) / inputFormat.sampleRate
            let estimatedSamples = Int((Double(audioFile.length) * sampleRateRatio).rounded(.up))

            if estimatedSamples > config.streamingThreshold {
                return try await transcribeDiskBacked(url, source: source)
            }
        }

        let audioFloatArray = try audioConverter.resampleAudioFile(url)
        let result = try await transcribe(audioFloatArray, source: source)
        return result
    }

    /// Transcribe audio from a file URL using disk-backed chunked processing.
    ///
    /// Memory-efficient transcription that memory-maps the file and processes audio in chunks,
    /// maintaining constant memory usage (~1.2MB) regardless of file size. Ideal for long audio files.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - source: The audio source type (defaults to .system)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribeDiskBacked(_ url: URL, source: AudioSource = .system) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }

        let startTime = Date()

        // Create a disk-backed source for memory-efficient access
        let factory = AudioSourceFactory()
        let (sampleSource, _) = try factory.makeDiskBackedSource(
            from: url,
            targetSampleRate: config.sampleRate
        )

        let totalSamples = sampleSource.sampleCount
        guard totalSamples >= config.sampleRate else {
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

            try await self.resetDecoderState()
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
    /// Performs speech-to-text transcription on raw audio samples at 16kHz. The decoder state is
    /// automatically reset after transcription completes, ensuring each transcription call is independent
    /// and enabling efficient batch processing of multiple audio files.
    ///
    /// - Parameters:
    ///   - audioSamples: Array of 16-bit audio samples at 16kHz
    ///   - source: The audio source type (microphone or system audio)
    /// - Note: Progress stream is emitted only when `audioSamples.count > ASRConstants.maxModelSamples` (~15s).
    ///         Use `transcriptionProgressStream` before calling this method to observe progress.
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioSamples: [Float],
        source: AudioSource = .microphone
    ) async throws -> ASRResult {
        let shouldEmitProgress = audioSamples.count > ASRConstants.maxModelSamples
        if shouldEmitProgress {
            _ = await progressEmitter.ensureSession()
        }
        do {
            let result = try await transcribeWithState(audioSamples, source: source)

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
        try await initializeDecoderState(for: source)
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
