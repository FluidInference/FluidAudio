import AVFoundation
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct StreamingAsrEngineMetrics: Sendable {
    public let chunkCount: Int
    public let totalChunkProcessingTime: TimeInterval
    public let averageChunkProcessingTime: TimeInterval?
    public let maxChunkProcessingTime: TimeInterval?
    public let minChunkProcessingTime: TimeInterval?
    public let firstTokenLatency: TimeInterval?
    public let firstConfirmedTokenLatency: TimeInterval?
}

/// A high-level streaming ASR manager that provides a simple API for real-time transcription
/// Similar to Apple's SpeechAnalyzer, it handles audio conversion and buffering automatically
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingAsrManager {
    private let logger = AppLogger(category: "StreamingASR")
    private let audioConverter: AudioConverter = AudioConverter()
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

    // VAD components
    private let injectedVadManager: VadManager?
    private var vadPipeline: StreamingVadPipeline

    // Sliding window state
    private var windowProcessor: StreamingWindowProcessor
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var tokenAccumulator = StreamingTokenAccumulator()
    private var stabilizerSink: StreamingStabilizerSink
    private var cumulativeVadDroppedSamples: Int = 0

    // Two-tier transcription state (like Apple's Speech API)
    public var volatileTranscript: String {
        stabilizerSink.volatileTranscript
    }

    public var confirmedTranscript: String {
        stabilizerSink.confirmedTranscript
    }

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics
    private var startTime: Date?
    private var processedChunks: Int = 0
    private var totalChunkProcessingTime: TimeInterval = 0
    private var maxChunkProcessingTime: TimeInterval = 0
    private var minChunkProcessingTime: TimeInterval = .infinity
    private var firstTokenLatencySeconds: TimeInterval?

    /// Initialize the streaming ASR manager
    /// - Parameter config: Configuration for streaming behavior
    /// Initialize a streaming ASR manager.
    /// - Parameters:
    ///   - config: Streaming configuration including stabilizer and VAD policies.
    ///   - vadManager: Optional voice activity detector to reuse. When `nil` and
    ///     `config.vad.isEnabled` is true, the manager will auto-initialize a VAD instance.
    public init(config: StreamingAsrConfig = .default, vadManager: VadManager? = nil) {
        self.config = config
        self.injectedVadManager = vadManager
        self.vadPipeline = StreamingVadPipeline(config: config, injectedManager: vadManager)
        self.windowProcessor = StreamingWindowProcessor(config: config)
        self.stabilizerSink = StreamingStabilizerSink(config: config.stabilizer)

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info(
            "Initialized StreamingAsrManager with config: chunk=\(config.chunkSeconds)s left=\(config.leftContextSeconds)s right=\(config.rightContextSeconds)s vadEnabled=\(config.vad.isEnabled)"
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

        stabilizerSink.initialize(
            using: asrManager,
            uid: stabilizerUID(for: source),
            logger: logger
        )

        do {
            try await setupVad(for: source)
        } catch {
            stabilizerSink.handleInitializationFailure(logger: logger)
            throw error
        }

        // Reset sliding window state
        windowProcessor.reset()
        segmentIndex = 0
        lastProcessedFrame = 0
        tokenAccumulator.reset()
        stabilizerSink.resetTranscripts()
        processedChunks = 0
        cumulativeVadDroppedSamples = 0
        totalChunkProcessingTime = 0
        maxChunkProcessingTime = 0
        minChunkProcessingTime = .infinity
        firstTokenLatencySeconds = nil

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
                do {
                    // Convert to 16kHz mono (streaming)
                    let samples = try audioConverter.resampleBuffer(pcmBuffer)

                    // Append to raw sample buffer and attempt windowed processing (with VAD gating if enabled)
                    await self.processIncomingSamples(samples)
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Stream ended: no need to flush converter since each conversion is stateless

            await self.flushPendingVadBuffers()

            // Then flush remaining assembled audio (no right-context requirement)
            await self.flushRemaining()
            self.finalizeStabilizerAfterStreamEnd()

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

        finalizeStabilizerAfterStreamEnd()

        // Use the stabilized transcripts as the authoritative final result.
        let stabilizedText = confirmedTranscript + volatileTranscript
        var finalText = stabilizedText
        if finalText.isEmpty,
            tokenAccumulator.trimmedTotalCount == 0,
            let asrManager = asrManager
        {
            let accumulatedTokens = tokenAccumulator.tokens
            if !accumulatedTokens.isEmpty {
                let finalResult = asrManager.processTranscriptionResult(
                    tokenIds: accumulatedTokens,
                    timestamps: [],
                    confidences: [],
                    encoderSequenceLength: 0,
                    audioSamples: [],
                    processingTime: 0
                )
                finalText = finalResult.text
            }
        }

        await resetVadState()

        logger.info("Final transcription: \(finalText.count) characters")
        return finalText
    }

    /// Reset the transcriber for a new session
    public func reset() async throws {
        stabilizerSink.resetTranscripts()
        processedChunks = 0
        startTime = Date()
        firstTokenLatencySeconds = nil
        windowProcessor.reset()

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        tokenAccumulator.reset()
        stabilizerSink.initialize(
            using: asrManager,
            uid: stabilizerUID(for: audioSource),
            logger: logger
        )

        await resetVadState()
        cumulativeVadDroppedSamples = 0
        totalChunkProcessingTime = 0
        maxChunkProcessingTime = 0
        minChunkProcessingTime = .infinity

        logger.info("StreamingAsrManager reset for source: \(String(describing: self.audioSource))")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()
        updateContinuation?.finish()

        finalizeStabilizerAfterStreamEnd()

        await resetVadState()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear the update continuation
    private func clearUpdateContinuation() {
        updateContinuation = nil
    }

    public func metricsSnapshot() -> StreamingAsrEngineMetrics {
        let average: TimeInterval?
        if processedChunks > 0 && totalChunkProcessingTime > 0 {
            average = totalChunkProcessingTime / Double(processedChunks)
        } else {
            average = nil
        }

        let minValue = processedChunks > 0 && minChunkProcessingTime.isFinite ? minChunkProcessingTime : nil
        let maxValue = processedChunks > 0 && maxChunkProcessingTime > 0 ? maxChunkProcessingTime : nil
        let stabilizerMetrics = stabilizerSink.metricsSnapshot()

        return StreamingAsrEngineMetrics(
            chunkCount: processedChunks,
            totalChunkProcessingTime: totalChunkProcessingTime,
            averageChunkProcessingTime: average,
            maxChunkProcessingTime: maxValue,
            minChunkProcessingTime: minValue,
            firstTokenLatency: firstTokenLatencySeconds,
            firstConfirmedTokenLatency: stabilizerMetrics.firstCommitLatencySeconds
        )
    }

    private func recordChunkProcessingTime(_ duration: TimeInterval) {
        guard duration.isFinite else { return }
        totalChunkProcessingTime += duration
        if duration > maxChunkProcessingTime {
            maxChunkProcessingTime = duration
        }
        if duration < minChunkProcessingTime {
            minChunkProcessingTime = duration
        }
    }

    private func recordFirstTokenLatencyIfNeeded(for update: StreamingTranscriptionUpdate) {
        guard firstTokenLatencySeconds == nil else { return }
        guard let startTime else { return }
        let normalized = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        let latency = update.timestamp.timeIntervalSince(startTime)
        guard latency.isFinite else { return }
        firstTokenLatencySeconds = max(0, latency)
    }

    private func buildChunkParameters(
        for window: StreamingWindow,
        additionalOffsetSamples: Int
    ) -> AsrManager.StreamingChunkParameters {
        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame
        let effectiveStartSample = window.startSample + additionalOffsetSamples
        let globalFrameOffset = effectiveStartSample / samplesPerFrame

        let isFinalChunk = window.isFinalChunk

        let frameCount = ASRConstants.calculateEncoderFrames(from: window.samples.count)
        return AsrManager.StreamingChunkParameters(
            actualAudioFrames: frameCount,
            contextFrameAdjustment: 0,
            isLastChunk: isFinalChunk,
            globalFrameOffset: globalFrameOffset
        )
    }

    // MARK: - Private Methods

    private func setupVad(for source: AudioSource) async throws {
        try await vadPipeline.prepare(for: source, logger: logger)
    }

    private func processIncomingSamples(_ samples: [Float]) async {
        await vadPipeline.process(samples: samples, logger: logger) { [self] event in
            switch event {
            case .speech(let segment):
                await self.appendSamplesAndProcess(segment, allowPartialWindows: vadPipeline.isEnabled)
            case .silence(samples: let count, cumulativeDroppedSamples: let cumulative):
                await self.handleSilenceGap(count: count, cumulativeDroppedSamples: cumulative)
            }
        }
    }

    private func flushPendingVadBuffers() async {
        await vadPipeline.flushPending(logger: logger) { [self] event in
            switch event {
            case .speech(let segment):
                await self.appendSamplesAndProcess(segment, allowPartialWindows: true)
            case .silence(samples: let count, cumulativeDroppedSamples: let cumulative):
                await self.handleSilenceGap(count: count, cumulativeDroppedSamples: cumulative)
            }
        }
    }

    private func resetVadState() async {
        await vadPipeline.resetState()
    }

    /// Append new samples and process as many windows as available
    private func appendSamplesAndProcess(
        _ segment: StreamingVadSegment,
        allowPartialWindows: Bool
    ) async {
        guard !segment.samples.isEmpty else { return }
        cumulativeVadDroppedSamples = segment.cumulativeDroppedSamples
        let shouldAllowPartialChunk = allowPartialWindows && segment.isSpeechOnset
        let minimumCenterSamples: Int?
        if shouldAllowPartialChunk {
            let sampleRate: Double
            if config.chunkSeconds > 0 {
                sampleRate = Double(config.chunkSamples) / config.chunkSeconds
            } else {
                sampleRate = 16_000.0
            }
            let onsetSeconds = max(1.0, min(2.0, config.chunkSeconds / 4.0))
            minimumCenterSamples = max(1, Int(onsetSeconds * sampleRate))
        } else {
            minimumCenterSamples = nil
        }
        let windows = windowProcessor.append(
            segment.samples,
            allowPartialChunk: shouldAllowPartialChunk,
            minimumCenterSamples: minimumCenterSamples
        )
        for window in windows {
            await processWindow(window, offsetOverride: nil)
        }
    }

    private func handleSilenceGap(count: Int, cumulativeDroppedSamples: Int) async {
        guard count > 0 else { return }
        windowProcessor.advanceBySilence(count)
        let previousOffset = cumulativeVadDroppedSamples
        if windowProcessor.hasBufferedAudio() {
            let windows = windowProcessor.append([], allowPartialChunk: true, minimumCenterSamples: 1)
            for window in windows {
                await processWindow(window, offsetOverride: previousOffset)
            }
        }
        cumulativeVadDroppedSamples = cumulativeDroppedSamples
    }

    /// Flush any remaining audio at end of stream (no right-context requirement)
    private func flushRemaining() async {
        let windows = windowProcessor.flushRemaining()
        for window in windows {
            await processWindow(window, offsetOverride: nil)
        }
    }

    /// Process a single assembled window: [left, chunk, right]
    private func processWindow(
        _ window: StreamingWindow,
        offsetOverride: Int?
    ) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            let cumulativeOffset = offsetOverride ?? cumulativeVadDroppedSamples
            let parameters = buildChunkParameters(for: window, additionalOffsetSamples: cumulativeOffset)

            let (tokens, timestamps, confidences, _) = try await asrManager.transcribeStreamingChunk(
                window.samples,
                source: audioSource,
                previousTokens: tokenAccumulator.tokens,
                parameters: parameters
            )

            // Update state
            tokenAccumulator.append(tokens)
            lastProcessedFrame = max(lastProcessedFrame, timestamps.max() ?? lastProcessedFrame)
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1
            recordChunkProcessingTime(processingTime)
            #if DEBUG
            let energy = window.samples.reduce(0) { $0 + abs($1) }
            let energyString = String(format: "%.3f", energy)
            logger.debug(
                "Processed streaming window \(segmentIndex) centerStart=\(window.centerStartSample) samples=\(window.samples.count) energy=\(energyString) newTokens=\(tokens.count) totalTokens=\(tokenAccumulator.totalCount) trimmed=\(tokenAccumulator.trimmedTotalCount)"
            )
            #endif

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = asrManager.processTranscriptionResult(
                tokenIds: tokens,  // Only current chunk tokens for progress updates
                timestamps: timestamps,
                confidences: confidences,
                encoderSequenceLength: 0,
                audioSamples: window.samples,
                processingTime: processingTime
            )

            let nowMs = currentStreamTimestampMs()
            let uid = stabilizerUID(for: audioSource)
            let output = stabilizerSink.emitterUpdate(
                uid: uid,
                accumulatedTokens: tokenAccumulator.tokens,
                latestTokens: tokens,
                latestTokenTimings: interim.tokenTimings ?? [],
                interimConfidence: interim.confidence,
                nowMs: nowMs
            )
            for update in output.updates {
                recordFirstTokenLatencyIfNeeded(for: update)
                updateContinuation?.yield(update)
            }
            let trimmed = tokenAccumulator.dropCommittedPrefixIfNeeded(
                totalCommitted: output.totalCommittedCount
            )
            if trimmed > 0 {
                stabilizerSink.discardCommittedTokenPrefix(trimmed, uid: uid)
            }

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Apply encoder-frame offset derived from the absolute window start sample.
    /// Streaming runs in disjoint chunks, so we need to add the global offset to
    /// keep each chunk's token timings aligned to the full audio timeline rather
    /// than resetting back to zero for every window.
    internal static func applyGlobalFrameOffset(to timestamps: [Int], windowStartSample: Int) -> [Int] {
        guard !timestamps.isEmpty else { return timestamps }

        let frameOffset = windowStartSample / ASRConstants.samplesPerEncoderFrame
        guard frameOffset != 0 else { return timestamps }

        return timestamps.map { $0 + frameOffset }
    }

    private func stabilizerUID(for source: AudioSource) -> Int {
        switch source {
        case .microphone:
            return 0
        case .system:
            return 1
        }
    }

    private func currentStreamTimestampMs() -> Int {
        guard let startTime else {
            return Int(Date().timeIntervalSince1970 * 1000.0)
        }
        return Int(Date().timeIntervalSince(startTime) * 1000.0)
    }

    private func flushStabilizerOnStreamEnd() {
        let uid = stabilizerUID(for: audioSource)
        let nowMs = currentStreamTimestampMs()
        guard let result = stabilizerSink.flush(uid: uid, nowMs: nowMs) else { return }
        let output = stabilizerSink.makeUpdates(
            result: result,
            accumulatedTokens: tokenAccumulator.tokens,
            latestTokens: [],
            latestTokenTimings: [],
            interimConfidence: 1.0,
            timestamp: Date()
        )
        for update in output.updates {
            recordFirstTokenLatencyIfNeeded(for: update)
            updateContinuation?.yield(update)
        }
        let trimmed = tokenAccumulator.dropCommittedPrefixIfNeeded(
            totalCommitted: output.totalCommittedCount
        )
        if trimmed > 0 {
            stabilizerSink.discardCommittedTokenPrefix(trimmed, uid: uid)
        }
    }

    private func finalizeStabilizerAfterStreamEnd() {
        flushStabilizerOnStreamEnd()
        stabilizerSink.finalizeAfterStreamEnd(logger: logger)
        let uid = stabilizerUID(for: audioSource)
        stabilizerSink.cleanupState(uid: uid)
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
                logger.info("Recovering from audio buffer error")

            case .audioConversionFailed:
                logger.info("Recovering from audio conversion error")

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
                stabilizerSink.refreshVocabulary(using: asrManager, logger: logger)
            } catch {
                logger.error("Failed to reset decoder state during recovery: \(error)")

                // Last resort: try to reinitialize the ASR manager
                do {
                    let models = try await AsrModels.downloadAndLoad()
                    let newAsrManager = AsrManager(config: config.asrConfig)
                    try await newAsrManager.initialize(models: models)
                    self.asrManager = newAsrManager
                    stabilizerSink.refreshVocabulary(using: newAsrManager, logger: logger)
                    logger.info("Successfully reinitialized ASR manager during error recovery")
                } catch {
                    logger.error("Failed to reinitialize ASR manager during recovery: \(error)")
                }
            }
        }
    }

}
