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
    internal var preprocessor: MLModel?
    internal var encoder: MLModel?
    internal var decoder: MLModel?
    internal var joint: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    internal var tokenizer: Tokenizer?

    // Configuration (loaded from metadata.json)
    public private(set) var config: NemotronStreamingConfig

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs
    internal var accumulatedTokenIds: [Int] = []

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
        configuration: MLModelConfiguration = MLModelConfiguration(),
        requestedChunkSize: NemotronChunkSize? = nil
    ) {
        self.mlConfiguration = configuration
        self.requestedChunkSize = requestedChunkSize
        self.config = NemotronStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) {
        self.partialCallback = callback
    }

    /// Load models from a directory containing preprocessor, encoder, decoder, joint, and tokenizer
    /// - Parameter directory: Directory containing the model files
    public func loadModels(from directory: URL) async throws {
        logger.info("Loading Nemotron CoreML models from \(directory.path)...")

        // Load config from metadata.json
        let metadataPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.metadata)
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            self.config = try NemotronStreamingConfig(from: metadataPath)
            logger.info("Loaded config: \(config.chunkMs)ms chunks, \(config.chunkMelFrames) mel frames")
        }

        // Load preprocessor
        let preprocessorPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.preprocessorFile)
        self.preprocessor = try await MLModel.load(contentsOf: preprocessorPath, configuration: mlConfiguration)

        // Load encoder (int8 quantized)
        let encoderPath = directory.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        self.encoder = try await MLModel.load(contentsOf: encoderPath, configuration: mlConfiguration)

        // Load decoder
        let decoderPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.decoderFile)
        self.decoder = try await MLModel.load(contentsOf: decoderPath, configuration: mlConfiguration)

        // Load joint
        let jointPath = directory.appendingPathComponent(ModelNames.NemotronStreaming.jointFile)
        self.joint = try await MLModel.load(contentsOf: jointPath, configuration: mlConfiguration)

        // Load tokenizer
        let tokenizerUrl = directory.appendingPathComponent(ModelNames.NemotronStreaming.tokenizer)
        self.tokenizer = try Tokenizer(vocabPath: tokenizerUrl)

        // Initialize states
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
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        if let configuration {
            self.mlConfiguration = configuration
        }

        let chunkSize = requestedChunkSize ?? .ms1120
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
            try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir, progressHandler: progressHandler)
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
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
    }

    public func cleanup() async {
        await reset()
        preprocessor = nil
        encoder = nil
        decoder = nil
        joint = nil
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
        guard preprocessor != nil, encoder != nil, decoder != nil, joint != nil else {
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
            preprocessor != nil,
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
        accumulatedTokenIds.removeAll()

        return transcript
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(ids: accumulatedTokenIds)
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
        guard preprocessor != nil, encoder != nil, decoder != nil, joint != nil else {
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
