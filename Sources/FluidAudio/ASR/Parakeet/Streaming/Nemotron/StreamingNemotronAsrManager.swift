import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates)
public typealias NemotronPartialCallback = @Sendable (String) -> Void

/// High-level manager for Nemotron Speech Streaming 0.6B pipeline.
/// Implements true streaming with encoder cache states.
public actor StreamingNemotronAsrManager {
    private let logger = AppLogger(category: "NemotronStreaming")

    // Models
    /// Native-Swift log-mel front-end, replacing the CoreML `preprocessor`
    /// model. See `NemotronMelExtractor` (and issue #739) for why the CoreML
    /// preprocessor was removed.
    internal var melExtractor: NemotronMelExtractor?
    internal var encoder: MLModel?
    internal var decoder: MLModel?
    internal var joint: MLModel?
    /// Optional fused decoder+joint model (B1). When present, the RNN-T inner
    /// loop makes one CoreML call per step instead of two (decoder then joint),
    /// halving per-step dispatch. Argmax stays in Swift. Mirrors the multilingual
    /// manager's `decoderJoint` path. Falls back to separate decoder+joint when nil.
    internal var decoderJoint: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    internal var tokenizer: Tokenizer?

    // Configuration (loaded from metadata.json)
    public private(set) var config: NemotronStreamingConfig

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs
    internal var accumulatedTokenIds: [Int] = []

    // Per-token absolute timings captured during the RNNT decode loop. Each
    // token's startTime is its absolute encoder-frame index multiplied by
    // ASRConstants.secondsPerEncoderFrame. Exposed via finishWithTokenTimings()
    // so callers can derive word-level timestamps (e.g. for speaker attribution).
    internal var accumulatedTokenTimings: [TokenTiming] = []
    // Running encoder-frame base across processed chunks (advances by the chunk's
    // encoder-frame count after each decode loop). Reset with the session.
    internal var absoluteFrameBase: Int = 0
    // Snapshot of token timings taken in finish() before the working buffers are
    // cleared, so finishWithTokenTimings() can return them after finish() runs.
    internal var lastFinishTokenTimings: [TokenTiming] = []

    // Encoder cache states
    internal var cacheChannel: MLMultiArray?
    internal var cacheTime: MLMultiArray?
    internal var cacheLen: MLMultiArray?

    // Mel cache (last 9 frames from previous chunk)
    internal var melCache: MLMultiArray?

    // Decoder LSTM states
    internal var hState: MLMultiArray?
    internal var cState: MLMultiArray?
    internal var lastToken: Int32

    // Callbacks
    internal var partialCallback: NemotronPartialCallback?

    /// Chunk size for auto-download. Set by `StreamingModelVariant.createManager()`
    /// to determine which HuggingFace repo to download from in `loadModels()`.
    internal var requestedChunkSize: NemotronChunkSize?

    // Stats
    internal var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(
        configuration: MLModelConfiguration? = nil,
        requestedChunkSize: NemotronChunkSize? = nil
    ) {
        // Default to `.cpuAndNeuralEngine`: the int8 encoder is ANE-targeted.
        // Under the bare `MLModelConfiguration()` default (which is `.all`),
        // CoreML routes int8 ops to GPU and runs ~10× slower than the ANE path.
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.requestedChunkSize = requestedChunkSize
        self.config = NemotronStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) {
        self.partialCallback = callback
    }

    /// Load models from a directory containing encoder, decoder, joint, and tokenizer
    /// (the mel front-end is computed natively in Swift; no CoreML preprocessor)
    /// - Parameter directory: Directory containing the model files
    public func loadModels(from directory: URL) async throws {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron int8 streaming models require Apple Silicon (ANE). Intel Macs are not supported."
            )
        }
        logger.info("Loading Nemotron CoreML models from \(directory.path)...")

        // Load config from metadata.json
        let metadataPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.metadata)
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            self.config = try NemotronStreamingConfig(from: metadataPath)
            logger.info("Loaded config: \(config.chunkMs)ms chunks, \(config.chunkMelFrames) mel frames")
        }

        // Native-Swift log-mel front-end (replaces the CoreML preprocessor).
        self.melExtractor = NemotronMelExtractor(nMels: config.melFeatures)

        // Load encoder (int8 quantized)
        let encoderPath = directory.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        self.encoder = try await MLModel.load(contentsOf: encoderPath, configuration: mlConfiguration)

        // Load decoder
        let decoderPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.decoderFile)
        self.decoder = try await MLModel.load(contentsOf: decoderPath, configuration: mlConfiguration)

        // Load joint
        let jointPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.jointFile)
        self.joint = try await MLModel.load(contentsOf: jointPath, configuration: mlConfiguration)

        // Optional fused decoder+joint (B1). When the tier folder ships a
        // `decoder_joint.mlmodelc`, prefer it in the inner loop (one call/step).
        let fusedPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.decoderJointFile)
        if FileManager.default.fileExists(atPath: fusedPath.path) {
            self.decoderJoint = try await MLModel.load(contentsOf: fusedPath, configuration: mlConfiguration)
            logger.info("Loaded fused decoder_joint (B1) — merged inner-loop path enabled")
        }

        // Load tokenizer
        let tokenizerUrl = directory.appendingPathComponent(ModelNames.NemotronStreaming.tokenizer)
        self.tokenizer = try Tokenizer(vocabPath: tokenizerUrl)

        // Initialize states
        try resetStates()

        // Fail loudly if the encoder's ANE program failed to instantiate (issue
        // #739): on iPadOS cold starts the int8 encoder can silently return an
        // all-zero buffer, yielding an empty transcript with no error thrown.
        // Probe once with non-zero input; throw a clear error instead of letting
        // every transcript come back empty. The probe also warms the encoder.
        let encoderHealthy = try await encoderProducesNonZeroOutput()
        guard encoderHealthy else {
            throw ASRError.encoderInstantiationFailed(
                "Nemotron int8 encoder returned all-zero output on a load-time probe — the ANE "
                    + "program did not instantiate (see CoreML ANEProgramProcessRequestDirect "
                    + "status=0x12). This is the iPadOS cold-start failure in issue #739; "
                    + "re-download or update the encoder model."
            )
        }
        // Probe used throwaway inputs; restore clean session state.
        try resetStates()

        logger.info("Nemotron models loaded successfully (\(config.chunkMs)ms chunks).")
    }

    /// Downloads and loads Nemotron streaming models from Hugging Face if not cached locally.
    ///
    /// - Parameters:
    ///   - directory: Root directory for model cache (default: Application Support)
    ///   - configuration: Optional model configuration override
    ///   - progressHandler: Optional callback for download progress updates
    public func loadModels(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        if let configuration {
            self.mlConfiguration = configuration
        }

        let chunkSize = requestedChunkSize ?? .ms2240
        let repo = chunkSize.repo

        let modelsBaseDir =
            directory
            ?? FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)

        let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)
        let encoderInt8Path = cacheDir.appendingPathComponent("encoder/\(NemotronEncoder.fileName)")

        if !FileManager.default.fileExists(atPath: encoderInt8Path.path) {
            logger.info("Downloading Nemotron models to \(modelsBaseDir.path)...")
            try await ModelHub.download(repo, to: modelsBaseDir, progressHandler: progressHandler)
        } else {
            logger.info("Using cached Nemotron models at \(cacheDir.path)")
        }

        try await loadModels(from: cacheDir)
    }

    /// Reset all states for a new transcription session
    public func reset() async {
        StreamingAsrUtils.resetSharedState(
            audioBuffer: &audioBuffer,
            accumulatedTokenIds: &accumulatedTokenIds,
            processedChunks: &processedChunks
        )
        accumulatedTokenTimings.removeAll()
        absoluteFrameBase = 0
        lastFinishTokenTimings.removeAll()
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
    }

    public func cleanup() async {
        await reset()
        melExtractor = nil
        encoder = nil
        decoder = nil
        joint = nil
        decoderJoint = nil
        tokenizer = nil
        cacheChannel = nil
        cacheTime = nil
        cacheLen = nil
        melCache = nil
        hState = nil
        cState = nil
        logger.info("StreamingNemotronAsrManager resources cleaned up")
    }

    private func resetStates() throws {
        // Encoder cache states using EncoderCacheManager
        let cacheConfig = EncoderCacheManager.CacheConfig(
            channelShape: config.cacheChannelShape,
            timeShape: config.cacheTimeShape,
            lenShape: [1]
        )
        let caches = try EncoderCacheManager.createInitialCaches(config: cacheConfig)
        cacheChannel = caches.channel
        cacheTime = caches.time
        cacheLen = caches.len
        // Seed cache_len with 1 instead of 0 so the encoder's
        // `ios17.slice_by_index` op never sees a zero-length slice, which would
        // fail CoreML shape inference and skip MPSGraph caching on every
        // session start. The cache buffers are zero, so this is equivalent to
        // 1 frame of silence preamble. See issue #607.
        cacheLen?[0] = 1

        // Mel cache (will be initialized on first chunk)
        melCache = nil

        // Decoder LSTM states
        hState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        cState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        lastToken = Int32(config.blankIdx)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        try StreamingAsrUtils.appendAudio(buffer, using: audioConverter, to: &audioBuffer)
    }

    /// Process audio and return partial transcript
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // Check if models are loaded
        guard melExtractor != nil, encoder != nil, decoder != nil, joint != nil else {
            throw ASRError.notInitialized
        }

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)

        // Process complete chunks
        while self.audioBuffer.count >= config.chunkSamples {
            let chunk = Array(self.audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            // Recheck buffer count after await to handle actor reentrancy
            let samplesToRemove = min(config.chunkSamples, self.audioBuffer.count)
            self.audioBuffer.removeFirst(samplesToRemove)
        }

        return ""
    }

    /// Finish processing and return final transcript
    public func finish() async throws -> String {
        // Check if models are loaded
        guard let tokenizer = tokenizer,
            melExtractor != nil,
            encoder != nil,
            decoder != nil,
            joint != nil
        else {
            throw ASRError.notInitialized
        }

        // Process remaining audio (padded if needed)
        if !audioBuffer.isEmpty {
            let paddingNeeded = config.chunkSamples - audioBuffer.count
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            audioBuffer.removeAll()
        }

        // Decode accumulated tokens
        let transcript = tokenizer.decode(ids: accumulatedTokenIds)
        // Snapshot timings before clearing so finishWithTokenTimings() can return
        // them; finish() must clear the working buffers atomically with the ids.
        lastFinishTokenTimings = accumulatedTokenTimings
        accumulatedTokenIds.removeAll()
        accumulatedTokenTimings.removeAll()

        return transcript
    }

    /// Finish processing and return the final transcript together with per-token
    /// timings (absolute seconds from the start of the fed audio). The timings
    /// are aligned 1:1 with the decoded token stream; group them by the
    /// SentencePiece word-boundary marker to obtain word-level timestamps.
    public func finishWithTokenTimings() async throws -> (text: String, timings: [TokenTiming]) {
        let text = try await finish()
        return (text, lastFinishTokenTimings)
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(ids: accumulatedTokenIds)
    }

    /// Get per-token timings accumulated so far without finishing. Aligned 1:1
    /// with the tokens behind getPartialTranscript(). Use this when a caller must
    /// salvage a partially-processed session that cannot safely call finish()
    /// (e.g. after a mid-stream decode failure).
    public func getTokenTimings() -> [TokenTiming] {
        return accumulatedTokenTimings
    }
}

// MARK: - StreamingAsrManager Conformance

extension StreamingNemotronAsrManager: StreamingAsrManager {
    public var displayName: String {
        "Nemotron 0.6B (\(config.chunkMs)ms)"
    }

    public func loadModels() async throws {
        try await loadModels(to: nil, configuration: nil, progressHandler: nil)
    }

    public func processBufferedAudio() async throws {
        guard melExtractor != nil, encoder != nil, decoder != nil, joint != nil else {
            throw ASRError.notInitialized
        }

        while audioBuffer.count >= config.chunkSamples {
            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            let samplesToRemove = min(config.chunkSamples, audioBuffer.count)
            audioBuffer.removeFirst(samplesToRemove)
        }
    }

    public func setPartialTranscriptCallback(_ callback: @escaping @Sendable (String) -> Void) {
        self.partialCallback = callback
    }
}
