import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates)
public typealias NemotronPartialCallback = @Sendable (String) -> Void

/// High-level manager for Nemotron Speech Streaming 0.6B pipeline.
/// Implements true streaming with encoder cache states.
public actor NemotronStreamingAsrManager {
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

    // Stats
    internal var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(configuration: MLModelConfiguration = MLModelConfiguration()) {
        self.mlConfiguration = configuration
        self.config = NemotronStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) {
        self.partialCallback = callback
    }

    /// Load models from a directory containing preprocessor, encoder, decoder, joint, and tokenizer
    /// - Parameters:
    ///   - modelDir: Directory containing the model files
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading Nemotron CoreML models from \(modelDir.path)...")

        // Load config from metadata.json
        let metadataPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.metadata)
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            self.config = try NemotronStreamingConfig(from: metadataPath)
            logger.info("Loaded config: \(config.chunkMs)ms chunks, \(config.chunkMelFrames) mel frames")
        }

        // Load preprocessor
        let preprocessorPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.preprocessorFile)
        self.preprocessor = try await MLModel.load(contentsOf: preprocessorPath, configuration: mlConfiguration)

        // Load encoder (int8 quantized)
        let encoderPath = modelDir.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        self.encoder = try await MLModel.load(contentsOf: encoderPath, configuration: mlConfiguration)

        // Load decoder
        let decoderPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.decoderFile)
        self.decoder = try await MLModel.load(contentsOf: decoderPath, configuration: mlConfiguration)

        // Load joint
        let jointPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.jointFile)
        self.joint = try await MLModel.load(contentsOf: jointPath, configuration: mlConfiguration)

        // Load tokenizer
        let tokenizerUrl = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.tokenizer)
        self.tokenizer = try Tokenizer(vocabPath: tokenizerUrl)

        // Initialize states
        try resetStates()

        logger.info("Nemotron models loaded successfully (\(config.chunkMs)ms chunks).")
    }

    /// Reset all states for a new transcription session
    public func reset() async {
        audioBuffer.removeAll()
        accumulatedTokenIds.removeAll()
        processedChunks = 0
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
    }

    private func resetStates() throws {
        // Encoder cache states
        cacheChannel = try MLMultiArray(
            shape: config.cacheChannelShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheChannel?.reset(to: 0)

        cacheTime = try MLMultiArray(
            shape: config.cacheTimeShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheTime?.reset(to: 0)

        cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen?[0] = 0

        // Mel cache (will be initialized on first chunk)
        melCache = nil

        // Decoder LSTM states
        hState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        hState?.reset(to: 0)

        cState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        cState?.reset(to: 0)

        lastToken = Int32(config.blankIdx)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        audioBuffer.append(contentsOf: samples)
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
