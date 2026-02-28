@preconcurrency import AVFoundation
import Foundation
import OSLog

private actor StreamingAsrEngine {
    private let manager: AsrManager

    init(config: ASRConfig) {
        manager = AsrManager(config: config)
    }

    func initialize(models: AsrModels) async throws {
        nonisolated(unsafe) let manager = manager
        try await manager.initialize(models: models)
    }

    func resetDecoderState(for source: AudioSource) async throws {
        nonisolated(unsafe) let manager = manager
        try await manager.resetDecoderState(for: source)
    }

    func transcribeStreamingChunk(
        _ chunkSamples: [Float],
        source: AudioSource,
        previousTokens: [Int],
        isLastChunk: Bool
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int) {
        nonisolated(unsafe) let manager = manager
        return try await manager.transcribeStreamingChunk(
            chunkSamples,
            source: source,
            previousTokens: previousTokens,
            isLastChunk: isLastChunk
        )
    }

    func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int],
        confidences: [Float],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval
    ) -> ASRResult {
        manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: timestamps,
            confidences: confidences,
            encoderSequenceLength: encoderSequenceLength,
            audioSamples: audioSamples,
            processingTime: processingTime
        )
    }
}

/// A high-level streaming ASR manager that provides a simple API for real-time transcription
/// Similar to Apple's SpeechAnalyzer, it handles audio conversion and buffering automatically
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
    private var asrEngine: StreamingAsrEngine?
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

    // Vocabulary boosting
    // These are initialized via configureVocabularyBoosting() before start()
    // CtcKeywordSpotter and VocabularyRescorer contain CoreML models which are not Sendable.
    // We manage the safety ourselves by only accessing them from within the actor.
    private var customVocabulary: CustomVocabularyContext?
    nonisolated(unsafe) private var ctcSpotter: CtcKeywordSpotter?
    nonisolated(unsafe) private var vocabularyRescorer: VocabularyRescorer?
    private var vocabSizeConfig: ContextBiasingConstants.VocabSizeConfig?
    private var vocabBoostingEnabled: Bool { customVocabulary != nil && vocabularyRescorer != nil }

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

    /// Configure vocabulary boosting for streaming transcription
    ///
    /// When configured, vocabulary terms will be rescored when text is confirmed during streaming.
    /// This provides real-time vocabulary corrections visible in confirmed updates.
    ///
    /// - Parameters:
    ///   - vocabulary: Custom vocabulary context with terms to detect
    ///   - ctcModels: Pre-loaded CTC models for keyword spotting
    ///   - config: Optional rescorer configuration (default: vocabulary-size-aware config)
    /// - Throws: Error if rescorer initialization fails
    public func configureVocabularyBoosting(
        vocabulary: CustomVocabularyContext,
        ctcModels: CtcModels,
        config: VocabularyRescorer.Config? = nil
    ) async throws {
        self.customVocabulary = vocabulary

        // Create CTC spotter
        let blankId = ctcModels.vocabulary.count
        self.ctcSpotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)

        // Use vocabulary-size-aware config (matching batch mode behavior)
        let vocabSize = vocabulary.terms.count
        let vocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: vocabSize)
        self.vocabSizeConfig = vocabConfig
        let effectiveConfig = config ?? .default

        // Create rescorer
        let ctcModelDir = CtcModels.defaultCacheDirectory(for: ctcModels.variant)
        self.vocabularyRescorer = try await VocabularyRescorer.create(
            spotter: ctcSpotter!,
            vocabulary: vocabulary,
            config: effectiveConfig,
            ctcModelDirectory: ctcModelDir
        )

        let isLargeVocab = vocabSize > ContextBiasingConstants.largeVocabThreshold
        logger.info(
            "Vocabulary boosting configured with \(vocabSize) terms (isLargeVocab: \(isLargeVocab))"
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

        // Initialize ASR engine with provided models
        let newAsrEngine = StreamingAsrEngine(config: config.asrConfig)
        try await newAsrEngine.initialize(models: models)
        try await newAsrEngine.resetDecoderState(for: source)
        asrEngine = newAsrEngine

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokens.removeAll()

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
                do {
                    // Convert to 16kHz mono (streaming)
                    let samples = try audioConverter.resampleBuffer(pcmBuffer)

                    // Append to raw sample buffer and attempt windowed processing
                    await self.appendSamplesAndProcess(samples)
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Stream ended: no need to flush converter since each conversion is stateless

            // Then flush remaining assembled audio (no right-context requirement)
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

        let finalText: String
        if vocabBoostingEnabled {
            // Text-based reconstruction preserves rescored corrections from processWindow().
            // Token-based reconstruction would undo rescoring since it decodes raw tokens.
            var parts: [String] = []
            if !confirmedTranscript.isEmpty { parts.append(confirmedTranscript) }
            if !volatileTranscript.isEmpty { parts.append(volatileTranscript) }
            finalText = parts.joined(separator: " ")
        } else if let asrEngine = asrEngine, !accumulatedTokens.isEmpty {
            let finalResult = await asrEngine.processTranscriptionResult(
                tokenIds: accumulatedTokens,
                timestamps: [],
                confidences: [],  // No per-token confidences needed for final text
                encoderSequenceLength: 0,
                audioSamples: [],  // Not needed for final text conversion
                processingTime: 0
            )
            finalText = finalResult.text
        } else {
            var parts: [String] = []
            if !confirmedTranscript.isEmpty { parts.append(confirmedTranscript) }
            if !volatileTranscript.isEmpty { parts.append(volatileTranscript) }
            finalText = parts.joined(separator: " ")
        }

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
        if let asrEngine = asrEngine {
            try await asrEngine.resetDecoderState(for: audioSource)
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
            await processWindow(window, windowStartSample: leftStartAbs)

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
            let isLastWindow = (nextWindowCenterStart + effectiveChunk) >= currentAbsEnd
            await processWindow(
                window,
                windowStartSample: leftStartAbs,
                isLastChunk: isLastWindow
            )

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
    private func processWindow(
        _ windowSamples: [Float],
        windowStartSample: Int,
        isLastChunk: Bool = false
    ) async {
        guard let asrEngine = asrEngine else { return }

        do {
            let chunkStartTime = Date()

            // Start frame offset is now handled by decoder's timeJump mechanism

            // Call AsrManager directly with deduplication
            let (tokens, timestamps, confidences, _) = try await asrEngine.transcribeStreamingChunk(
                windowSamples,
                source: audioSource,
                previousTokens: accumulatedTokens,
                isLastChunk: isLastChunk
            )

            let adjustedTimestamps = Self.applyGlobalFrameOffset(
                to: timestamps,
                windowStartSample: windowStartSample
            )

            // Update state
            accumulatedTokens.append(contentsOf: tokens)
            lastProcessedFrame = max(lastProcessedFrame, adjustedTimestamps.max() ?? 0)
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = await asrEngine.processTranscriptionResult(
                tokenIds: tokens,  // Only current chunk tokens for progress updates
                timestamps: adjustedTimestamps,
                confidences: confidences,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            logger.debug(
                "Chunk \(self.processedChunks): '\(interim.text)', time: \(String(format: "%.3f", processingTime))s)"
            )

            let totalAudioProcessed = Double(bufferStartIndex + sampleBuffer.count) / 16000.0
            let hasMinimumContext = totalAudioProcessed >= config.minContextForConfirmation
            let isHighConfidence = Double(interim.confidence) >= config.confirmationThreshold
            let shouldConfirm = isHighConfidence && hasMinimumContext

            // Rescore before updating transcript state so finish() returns rescored content
            var displayResult = interim
            if shouldConfirm && vocabBoostingEnabled {
                let chunkLocalTimings =
                    await asrEngine.processTranscriptionResult(
                        tokenIds: tokens,
                        timestamps: timestamps,  // Original chunk-local timestamps (not adjusted)
                        confidences: confidences,
                        encoderSequenceLength: 0,
                        audioSamples: windowSamples,
                        processingTime: processingTime
                    ).tokenTimings ?? []

                if let rescored = await applyVocabularyRescoring(
                    text: interim.text,
                    tokenTimings: chunkLocalTimings,
                    windowSamples: windowSamples
                ) {
                    let detected = rescored.replacements.compactMap { $0.replacementWord }
                    let applied = rescored.replacements.filter { $0.shouldReplace }.compactMap {
                        $0.replacementWord
                    }
                    displayResult = interim.withRescoring(
                        text: rescored.text,
                        detected: detected.isEmpty ? nil : detected,
                        applied: applied.isEmpty ? nil : applied
                    )
                }
            }

            await updateTranscriptionState(with: displayResult, shouldConfirm: shouldConfirm)

            let update = StreamingTranscriptionUpdate(
                text: displayResult.text,
                isConfirmed: shouldConfirm,
                confidence: interim.confidence,
                timestamp: Date(),
                tokenIds: tokens,
                tokenTimings: displayResult.tokenTimings ?? []
            )

            updateContinuation?.yield(update)

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    private func updateTranscriptionState(with result: ASRResult, shouldConfirm: Bool) async {
        let totalAudioProcessed = Double(bufferStartIndex + sampleBuffer.count) / 16000.0

        if shouldConfirm {
            if !volatileTranscript.isEmpty {
                var components: [String] = []
                if !confirmedTranscript.isEmpty {
                    components.append(confirmedTranscript)
                }
                components.append(volatileTranscript)
                confirmedTranscript = components.joined(separator: " ")
            }
            volatileTranscript = result.text
            logger.debug(
                "CONFIRMED (\(result.confidence), \(String(format: "%.1f", totalAudioProcessed))s context): promoted to confirmed; new volatile '\(result.text)'"
            )
        } else {
            volatileTranscript = result.text
            let hasMinimumContext = totalAudioProcessed >= config.minContextForConfirmation
            let reason =
                !hasMinimumContext
                ? "insufficient context (\(String(format: "%.1f", totalAudioProcessed))s)" : "low confidence"
            logger.debug("VOLATILE (\(result.confidence)): \(reason) - updated volatile '\(result.text)'")
        }
    }

    /// Apply vocabulary rescoring to confirmed text using CTC-based constrained decoding.
    ///
    /// This runs CTC inference on the chunk audio and applies vocabulary rescoring
    /// to replace misrecognized words with vocabulary terms when acoustic evidence supports it.
    ///
    /// - Parameters:
    ///   - text: Original transcript text from ASR
    ///   - tokenTimings: Token-level timing information
    ///   - windowSamples: Audio samples for the current window
    /// - Returns: Rescored output if modifications were made, nil otherwise
    private func applyVocabularyRescoring(
        text: String,
        tokenTimings: [TokenTiming],
        windowSamples: [Float]
    ) async -> VocabularyRescorer.RescoreOutput? {
        guard let spotter = ctcSpotter,
            let rescorer = vocabularyRescorer,
            let vocab = customVocabulary,
            !tokenTimings.isEmpty
        else {
            return nil
        }

        do {
            // Run CTC inference on the chunk audio to get log probabilities
            let spotResult = try await spotter.spotKeywordsWithLogProbs(
                audioSamples: windowSamples,
                customVocabulary: vocab,
                minScore: nil
            )

            let logProbs = spotResult.logProbs
            guard !logProbs.isEmpty else {
                logger.debug("Vocabulary rescoring skipped: no log probs from CTC")
                return nil
            }

            // Determine rescoring parameters based on vocabulary size
            let vocabConfig = vocabSizeConfig ?? ContextBiasingConstants.rescorerConfig(forVocabSize: 0)
            let minSimilarity = vocabConfig.minSimilarity
            let cbw = vocabConfig.cbw

            // Apply constrained CTC rescoring
            let rescoreOutput = rescorer.ctcTokenRescore(
                transcript: text,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: spotResult.frameDuration,
                cbw: cbw,
                marginSeconds: 0.5,
                minSimilarity: minSimilarity
            )

            if rescoreOutput.wasModified {
                logger.info(
                    "Vocabulary rescoring applied \(rescoreOutput.replacements.count) replacement(s) in streaming chunk"
                )
                for replacement in rescoreOutput.replacements where replacement.shouldReplace {
                    logger.debug(
                        "  '\(replacement.originalWord)' → '\(replacement.replacementWord ?? "")'"
                    )
                }
                return rescoreOutput
            }

            return nil
        } catch {
            logger.warning("Vocabulary rescoring failed: \(error.localizedDescription)")
            return nil
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
        if let asrEngine = asrEngine {
            do {
                try await asrEngine.resetDecoderState(for: audioSource)
                logger.info("Successfully reset decoder state during error recovery")
            } catch {
                logger.error("Failed to reset decoder state during recovery: \(error)")

                // Last resort: try to reinitialize the ASR manager
                do {
                    let models = try await AsrModels.downloadAndLoad()
                    let newAsrEngine = StreamingAsrEngine(config: config.asrConfig)
                    try await newAsrEngine.initialize(models: models)
                    try await newAsrEngine.resetDecoderState(for: audioSource)
                    self.asrEngine = newAsrEngine
                    logger.info("Successfully reinitialized ASR manager during error recovery")
                } catch {
                    logger.error("Failed to reinitialize ASR manager during recovery: \(error)")
                }
            }
        }
    }
}

/// Configuration for StreamingAsrManager
public struct StreamingAsrConfig: Sendable {
    /// Main chunk size for stable transcription (seconds). Should be 10-11s for best quality
    public let chunkSeconds: TimeInterval
    /// Quick hypothesis chunk size for immediate feedback (seconds). Typical: 1.0s
    public let hypothesisChunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s
    public let leftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency)
    public let rightContextSeconds: TimeInterval
    /// Minimum audio duration before confirming text (seconds). Should be ~10s
    public let minContextForConfirmation: TimeInterval

    /// Confidence threshold for promoting volatile text to confirmed (0.0...1.0)
    public let confirmationThreshold: Double
    /// Default configuration aligned with previous API expectations
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 15.0,
        hypothesisChunkSeconds: 2.0,
        leftContextSeconds: 10.0,
        rightContextSeconds: 2.0,
        minContextForConfirmation: 10.0,
        confirmationThreshold: 0.85
    )

    /// Optimized streaming configuration: Dual-track processing for best experience
    /// Uses ChunkProcessor's proven 11-2-2 approach for stable transcription
    /// Plus quick hypothesis updates for immediate feedback
    public static let streaming = StreamingAsrConfig(
        chunkSeconds: 11.0,  // Match ChunkProcessor for stable transcription
        hypothesisChunkSeconds: 1.0,  // Quick hypothesis updates
        leftContextSeconds: 2.0,  // Match ChunkProcessor left context
        rightContextSeconds: 2.0,  // Match ChunkProcessor right context
        minContextForConfirmation: 10.0,  // Need sufficient context before confirming
        confirmationThreshold: 0.80  // Higher threshold for more stable confirmations
    )

    public init(
        chunkSeconds: TimeInterval = 10.0,
        hypothesisChunkSeconds: TimeInterval = 1.0,
        leftContextSeconds: TimeInterval = 2.0,
        rightContextSeconds: TimeInterval = 2.0,
        minContextForConfirmation: TimeInterval = 10.0,
        confirmationThreshold: Double = 0.85
    ) {
        self.chunkSeconds = chunkSeconds
        self.hypothesisChunkSeconds = hypothesisChunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.minContextForConfirmation = minContextForConfirmation
        self.confirmationThreshold = confirmationThreshold
    }

    /// Backward-compatible convenience initializer used by tests (chunkDuration label)
    public init(
        confirmationThreshold: Double = 0.85,
        chunkDuration: TimeInterval
    ) {
        self.init(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold
        )
    }

    /// Custom configuration factory expected by tests
    public static func custom(
        chunkDuration: TimeInterval,
        confirmationThreshold: Double
    ) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold
        )
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            tdtConfig: TdtConfig()
        )
    }

    // Sample counts at 16 kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var hypothesisChunkSamples: Int { Int(hypothesisChunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }
    var minContextForConfirmationSamples: Int { Int(minContextForConfirmation * 16000) }

    // Backward-compat convenience for existing call-sites/tests
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
}

/// Transcription update from streaming ASR
public struct StreamingTranscriptionUpdate: Sendable {
    /// The transcribed text
    public let text: String

    /// Whether this text is confirmed (high confidence) or volatile (may change)
    public let isConfirmed: Bool

    /// Confidence score (0.0 - 1.0)
    public let confidence: Float

    /// Timestamp of this update
    public let timestamp: Date

    /// Raw token identifiers emitted for this update
    public let tokenIds: [Int]

    /// Token-level timing information aligned with the decoded text
    public let tokenTimings: [TokenTiming]

    /// Human-readable tokens (normalized) for this update
    public var tokens: [String] {
        tokenTimings.map(\.token)
    }

    public init(
        text: String,
        isConfirmed: Bool,
        confidence: Float,
        timestamp: Date,
        tokenIds: [Int] = [],
        tokenTimings: [TokenTiming] = []
    ) {
        self.text = text
        self.isConfirmed = isConfirmed
        self.confidence = confidence
        self.timestamp = timestamp
        self.tokenIds = tokenIds
        self.tokenTimings = tokenTimings
    }
}
