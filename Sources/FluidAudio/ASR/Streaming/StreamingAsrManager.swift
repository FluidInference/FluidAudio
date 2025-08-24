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

    // Sliding window state
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var accumulatedTokens: [Int] = []

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]
    private var nextWindowCenterStart: Int = 0  // absolute index where next chunk (center) begins

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
            "Initialized StreamingAsrManager with config: chunk=\(config.chunkSeconds)s left=\(config.leftContextSeconds)s right=\(config.rightContextSeconds)s"
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

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokens.removeAll()

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in await self.inputSequence {
                do {
                    // Convert to 16kHz mono
                    let samples = try await audioConverter.convertToAsrFormat(pcmBuffer)

                    // Append to raw sample buffer and attempt windowed processing
                    await self.appendSamplesAndProcess(samples)
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Stream ended: flush remaining audio (without requiring full right context)
            await self.flushRemaining()

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
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        nextWindowCenterStart = 0

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokens.removeAll()

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

    /// Append new samples and process as many windows as available
    private func appendSamplesAndProcess(_ samples: [Float]) async {
        // Append samples to buffer
        sampleBuffer.append(contentsOf: samples)

        // Process while we have at least chunk + right ahead of the current center start
        let chunk = config.chunkSamples
        let right = config.rightContextSamples
        let left = config.leftContextSamples
        let sampleRate = config.asrConfig.sampleRate

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd >= (nextWindowCenterStart + chunk + right) {
            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + chunk + right
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = rightEndAbs - bufferStartIndex
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx {
                break
            }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            let actualLeftSecs = Double(nextWindowCenterStart - leftStartAbs) / Double(sampleRate)
            await processWindow(window, actualLeftSeconds: actualLeftSecs)

            // Advance by chunk size
            nextWindowCenterStart += chunk

            // Trim buffer to keep only what's needed for left context
            let trimToAbs = max(0, nextWindowCenterStart - left)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Flush any remaining audio at end of stream (no right-context requirement)
    private func flushRemaining() async {
        let chunk = config.chunkSamples
        let left = config.leftContextSamples
        let sampleRate = config.asrConfig.sampleRate

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd > nextWindowCenterStart {  // process until we exhaust
            // If we have less than a chunk ahead, process the final partial chunk
            let availableAhead = currentAbsEnd - nextWindowCenterStart
            if availableAhead <= 0 { break }
            let effectiveChunk = min(chunk, availableAhead)

            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + effectiveChunk
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx { break }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            let actualLeftSecs = Double(nextWindowCenterStart - leftStartAbs) / Double(sampleRate)
            await processWindow(window, actualLeftSeconds: actualLeftSecs)

            nextWindowCenterStart += effectiveChunk

            // Trim
            let trimToAbs = max(0, nextWindowCenterStart - left)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Process a single assembled window: [left, chunk, right]
    private func processWindow(_ windowSamples: [Float], actualLeftSeconds: Double) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            // Calculate start frame offset
            let startOffset = asrManager.calculateStartFrameOffset(
                segmentIndex: segmentIndex,
                leftContextSeconds: actualLeftSeconds
            )

            // Call AsrManager directly with deduplication
            let (tokens, timestamps, _) = try await asrManager.transcribeStreamingChunk(
                windowSamples,
                source: audioSource,
                startFrameOffset: startOffset,
                lastProcessedFrame: lastProcessedFrame,
                previousTokens: accumulatedTokens,
                enableDebug: config.enableDebug
            )

            // Update state
            accumulatedTokens.append(contentsOf: tokens)
            lastProcessedFrame = max(lastProcessedFrame, timestamps.max() ?? 0)
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Convert tokens to text + timings via shared manager utility
            let interim = asrManager.processTranscriptionResult(
                tokenIds: tokens,
                timestamps: timestamps,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            logger.debug(
                "Chunk \(self.processedChunks): '\(interim.text)' (confidence: \(interim.confidence), time: \(String(format: "%.3f", processingTime))s)"
            )

            // Apply confidence-based confirmation logic
            await updateTranscriptionState(with: interim)

            // Emit update
            let update = StreamingTranscriptionUpdate(
                text: interim.text,
                isConfirmed: interim.confidence >= 0.5,
                confidence: interim.confidence,
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
    /// New chunk size per update (seconds). Typical streaming: 2.0s
    public let chunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s
    public let leftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency)
    public let rightContextSeconds: TimeInterval

    /// Enable debug logging
    public let enableDebug: Bool

    /// Default configuration aligned with NeMo streaming: 10-2-2 (L-C-R), ~4s latency
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 10.0,
        leftContextSeconds: 2.0,
        rightContextSeconds: 2.0,
        enableDebug: false
    )

    public init(
        chunkSeconds: TimeInterval = 10.0,
        leftContextSeconds: TimeInterval = 2.0,
        rightContextSeconds: TimeInterval = 2.0,
        enableDebug: Bool = false
    ) {
        self.chunkSeconds = chunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.enableDebug = enableDebug
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            enableDebug: enableDebug,
            tdtConfig: TdtConfig()
        )
    }

    // Sample counts at 16 kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }

    // Backward-compat convenience for existing call-sites/tests
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
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
