import AVFoundation
import Foundation
import OSLog

/// A high-level streaming ASR manager that provides a simple API for real-time transcription
/// Similar to Apple's SpeechAnalyzer, it handles audio conversion and buffering automatically
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingAsrManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "StreamingASR")
    private let audioConverter = AudioConverter()
    private let config: StreamingAsrConfig

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Transcription output stream
    private var updateContinuation: AsyncStream<StreamingTranscriptionUpdate>.Continuation?

    // ASR components
    private var asrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone

    // Two-tier transcription state (like Apple's Speech API)
    public private(set) var volatileTranscript: String = ""
    public private(set) var confirmedTranscript: String = ""

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics
    private var startTime: Date?
    private var processedChunks: Int = 0

    /// Initialize the streaming ASR manager
    /// - Parameter config: Configuration for streaming behavior
    public init(config: StreamingAsrConfig = .default) {
        self.config = config

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info(
            "Initialized StreamingAsrManager with config: chunkDuration=\(config.chunkDuration)s, confirmationThreshold=\(config.confirmationThreshold)"
        )
    }

    /// Start the streaming ASR engine
    /// This will download models if needed and begin processing
    /// - Parameter source: The audio source to use (default: microphone)
    public func start(source: AudioSource = .microphone) async throws {
        logger.info("Starting streaming ASR engine for source: \(String(describing: source))...")

        // Initialize ASR models
        let models = try await AsrModels.downloadAndLoad()
        try await start(models: models, source: source)
    }

    /// Start the streaming ASR engine with pre-loaded models
    /// - Parameters:
    ///   - models: Pre-loaded ASR models to use
    ///   - source: The audio source to use (default: microphone)
    public func start(models: AsrModels, source: AudioSource = .microphone) async throws {
        logger.info(
            "Starting streaming ASR engine with pre-loaded models for source: \(String(describing: source))..."
        )

        self.audioSource = source

        // Initialize ASR manager with provided models
        asrManager = AsrManager(config: config.asrConfig)
        try await asrManager?.initialize(models: models)

        // Reset decoder state for the specific source
        try await asrManager?.resetDecoderState(for: source)

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {

            let audioBuffer = AudioBuffer(capacity: config.bufferCapacity)

            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in await self.inputSequence {
                do {
                    // Convert to 16kHz mono
                    let samples = try await audioConverter.convertToAsrFormat(pcmBuffer)

                    // Add to buffer
                    try await audioBuffer.append(samples)

                    // Process when we have enough samples
                    let availableSamples = await audioBuffer.availableSamples()
                    if availableSamples >= config.chunkSizeInSamples {
                        if let chunk = await audioBuffer.getChunk(size: config.chunkSizeInSamples) {
                            await self.processChunk(chunk)
                        }
                    }
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Process any remaining audio when stream ends
            let remainingSamples = await audioBuffer.availableSamples()
            if remainingSamples > 0 {
                logger.info("Processing \(remainingSamples) remaining samples...")
                let remaining = await audioBuffer.peekAvailable()
                if !remaining.isEmpty {
                    await self.processChunk(remaining)
                }
            }

            logger.info("Recognition task completed")
        }

        logger.info("Streaming ASR engine started successfully")
    }

    /// Stream audio data for transcription
    /// - Parameter buffer: Audio buffer in any format (will be converted to 16kHz mono)
    public func streamAudio(_ buffer: AVAudioPCMBuffer) {
        inputBuilder.yield(buffer)
    }

    /// Get an async stream of transcription updates
    public var transcriptionUpdates: AsyncStream<StreamingTranscriptionUpdate> {
        AsyncStream { continuation in
            self.updateContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearUpdateContinuation()
                }
            }
        }
    }

    /// Finish streaming and get the final transcription
    /// - Returns: The complete transcription text
    public func finish() async throws -> String {
        logger.info("Finishing streaming ASR...")

        // Signal end of input
        inputBuilder.finish()

        // Wait for recognition task to complete
        do {
            try await recognizerTask?.value
        } catch {
            logger.error("Recognition task failed: \(error)")
            throw error
        }

        // Return complete transcription
        let finalText = confirmedTranscript + volatileTranscript
        logger.info("Final transcription: \(finalText.count) characters")

        return finalText
    }

    /// Reset the transcriber for a new session
    public func reset() async throws {
        volatileTranscript = ""
        confirmedTranscript = ""
        processedChunks = 0
        startTime = Date()

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }

        logger.info("StreamingAsrManager reset for source: \(String(describing: self.audioSource))")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()
        updateContinuation?.finish()

        // Cleanup audio converter state
        await audioConverter.cleanup()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear the update continuation
    private func clearUpdateContinuation() {
        updateContinuation = nil
    }

    // MARK: - Private Methods

    /// Process an audio chunk
    private func processChunk(_ chunk: [Float]) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            // Transcribe the chunk using the configured audio source
            let result = try await asrManager.transcribe(chunk, source: audioSource)

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            logger.debug(
                "Chunk \(self.processedChunks): '\(result.text)' (confidence: \(result.confidence), time: \(String(format: "%.3f", processingTime))s)"
            )

            // Apply confidence-based confirmation logic
            await updateTranscriptionState(with: result)

            // Emit update
            let update = StreamingTranscriptionUpdate(
                text: result.text,
                isConfirmed: result.confidence >= config.confirmationThreshold,
                confidence: result.confidence,
                timestamp: Date()
            )

            updateContinuation?.yield(update)

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Update transcription state based on confidence
    private func updateTranscriptionState(with result: ASRResult) async {
        if result.confidence >= config.confirmationThreshold {
            // High confidence: move volatile to confirmed and update volatile
            if !volatileTranscript.isEmpty {
                var components: [String] = []
                if !confirmedTranscript.isEmpty {
                    components.append(confirmedTranscript)
                }
                components.append(volatileTranscript)

                // Join with spaces, avoiding double spaces
                confirmedTranscript = components.joined(separator: " ")
            }
            volatileTranscript = result.text

            logger.debug(
                "High confidence (\(result.confidence)): confirmed '\(self.volatileTranscript)' -> volatile '\(result.text)'"
            )
        } else {
            // Low confidence: just update volatile
            volatileTranscript = result.text
            logger.debug(
                "Low confidence (\(result.confidence)): volatile updated to '\(result.text)'")
        }
    }

    /// Attempt to recover from processing errors
    private func attemptErrorRecovery(error: Error) async {
        logger.warning("Attempting error recovery for: \(error)")

        // Handle specific error types with targeted recovery
        if let streamingError = error as? StreamingAsrError {
            switch streamingError {
            case .modelsNotLoaded:
                logger.error("Models not loaded - cannot recover automatically")

            case .streamAlreadyExists:
                logger.error("Stream already exists - cannot recover automatically")

            case .audioBufferProcessingFailed:
                logger.info("Recovering from audio buffer error - resetting audio converter")
                await audioConverter.reset()

            case .audioConversionFailed:
                logger.info("Recovering from audio conversion error - resetting converter")
                await audioConverter.reset()

            case .modelProcessingFailed:
                logger.info("Recovering from model processing error - resetting decoder state")
                await resetDecoderForRecovery()

            case .bufferOverflow:
                logger.info("Buffer overflow handled automatically")

            case .invalidConfiguration:
                logger.error("Configuration error cannot be recovered automatically")
            }
        } else {
            // Generic recovery for non-streaming errors
            await resetDecoderForRecovery()
        }
    }

    /// Reset decoder state for error recovery
    private func resetDecoderForRecovery() async {
        if let asrManager = asrManager {
            do {
                try await asrManager.resetDecoderState(for: audioSource)
                logger.info("Successfully reset decoder state during error recovery")
            } catch {
                logger.error("Failed to reset decoder state during recovery: \(error)")

                // Last resort: try to reinitialize the ASR manager
                do {
                    let models = try await AsrModels.downloadAndLoad()
                    let newAsrManager = AsrManager(config: config.asrConfig)
                    try await newAsrManager.initialize(models: models)
                    self.asrManager = newAsrManager
                    logger.info("Successfully reinitialized ASR manager during error recovery")
                } catch {
                    logger.error("Failed to reinitialize ASR manager during recovery: \(error)")
                }
            }
        }
    }
}

/// Configuration for StreamingAsrManager
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingAsrConfig: Sendable {
    /// Confidence threshold for confirming transcriptions (0.0 - 1.0)
    public let confirmationThreshold: Float

    /// Duration of each audio chunk in seconds
    public let chunkDuration: TimeInterval

    /// Enable debug logging
    public let enableDebug: Bool

    /// Default configuration with balanced settings
    public static let `default` = StreamingAsrConfig(
        confirmationThreshold: 0.85,
        chunkDuration: 15.0,  // 15 second chunks
        enableDebug: false
    )

    /// Low latency configuration with faster updates
    public static let lowLatency = StreamingAsrConfig(
        confirmationThreshold: 0.75,
        chunkDuration: 15.0,  // 15 second chunks
        enableDebug: false
    )

    /// High accuracy configuration with conservative confirmation
    public static let highAccuracy = StreamingAsrConfig(
        confirmationThreshold: 0.9,
        chunkDuration: 15.0,  // 15 second chunks
        enableDebug: false
    )

    public init(
        confirmationThreshold: Float = 0.85,
        chunkDuration: TimeInterval = 15.0,
        enableDebug: Bool = false
    ) {
        self.confirmationThreshold = confirmationThreshold
        self.chunkDuration = chunkDuration
        self.enableDebug = enableDebug
    }

    /// Custom configuration with specified chunk duration
    /// - Note: The underlying model currently works best with 15s chunks
    /// - Shorter chunk support is still being validated
    public static func custom(
        chunkDuration: TimeInterval,
        confirmationThreshold: Float = 0.85,
        enableDebug: Bool = false
    ) -> StreamingAsrConfig {
        StreamingAsrConfig(
            confirmationThreshold: confirmationThreshold,
            chunkDuration: chunkDuration,
            enableDebug: enableDebug
        )
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            enableDebug: enableDebug,
            realtimeMode: true,
            chunkSizeMs: Int(chunkDuration * 1000),
            tdtConfig: TdtConfig(
                includeTokenDuration: true,
                maxSymbolsPerStep: 3
            )
        )
    }

    var bufferCapacity: Int {
        Int(15.0 * 16000)  // 15 seconds at 16kHz
    }

    var chunkSizeInSamples: Int {
        Int(chunkDuration * 16000)
    }
}

/// Transcription update from streaming ASR
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptionUpdate: Sendable {
    /// The transcribed text
    public let text: String

    /// Whether this text is confirmed (high confidence) or volatile (may change)
    public let isConfirmed: Bool

    /// Confidence score (0.0 - 1.0)
    public let confidence: Float

    /// Timestamp of this update
    public let timestamp: Date

    public init(
        text: String,
        isConfirmed: Bool,
        confidence: Float,
        timestamp: Date
    ) {
        self.text = text
        self.isConfirmed = isConfirmed
        self.confidence = confidence
        self.timestamp = timestamp
    }
}
