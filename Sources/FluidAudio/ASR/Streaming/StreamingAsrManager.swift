import AVFoundation
import CoreMedia
import Foundation
import OSLog

/// High-level streaming ASR manager for real-time transcription
///
/// Provides a simple API similar to Apple's SpeechAnalyzer, handling audio conversion
/// and buffering automatically. Supports both volatile (real-time) and final (corrected)
/// transcription updates for responsive user interfaces.
///
/// **Basic Usage:**
/// ```swift
/// let streamingAsr = StreamingAsrManager(config: .realtime)
/// try await streamingAsr.start()
///
/// // Listen for UI updates
/// for await snapshot in streamingAsr.snapshots {
///     updateUI(finalized: snapshot.finalized, volatile: snapshot.volatile)
/// }
/// ```
///
/// **For complete examples**, see `Examples/StreamingASR/` directory.
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingAsrManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "StreamingASR")
    private let audioConverter = AudioConverter()
    private let config: StreamingAsrConfig

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Transcription output streams
    private var resultsContinuation: AsyncStream<StreamingTranscriptionResult>.Continuation?
    private var snapshotsContinuation: AsyncStream<StreamingTranscriptSnapshot>.Continuation?

    // ASR components
    private var asrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone

    // Sliding window state
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var accumulatedTokens: [Int] = []

    // Segment tracking to prevent duplicates
    private var finalizedSegments: Set<UUID> = []
    private var segmentFinalTexts: [(UUID, String)] = []  // Track final text per segment in order
    private var lastFinalizedText: String = ""

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]

    // Fixed-duration segmentation: process one center segment at a time, emitting
    // a volatile hypothesis once a small right-context is available, and a final
    // result once full right-context is available, then advance by chunk size.
    private struct CurrentSegment {
        var id: UUID
        var centerStartAbs: Int
        var chunkSamples: Int
        var revision: Int
        var emittedFinal: Bool
        var volatileProgressSamples: Int  // how much of center has been covered by volatile updates
    }
    private var currentSegment: CurrentSegment = .init(
        id: UUID(), centerStartAbs: 0, chunkSamples: 0, revision: 0, emittedFinal: false, volatileProgressSamples: 0
    )

    // Two-tier transcription state (like Apple's Speech API)
    public private(set) var volatileTranscript: String = ""
    public private(set) var finalizedTranscript: String = ""

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

        // Initialize first segment
        currentSegment = CurrentSegment(
            id: UUID(), centerStartAbs: 0, chunkSamples: config.chunkSamples, revision: 0,
            emittedFinal: false, volatileProgressSamples: 0
        )

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
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

            // Stream ended: flush remaining audio and finalize any active range
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

    /// Segment-based results stream (volatile iterations and final replacements)
    public var results: AsyncStream<StreamingTranscriptionResult> {
        AsyncStream { continuation in
            self.resultsContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearResultsContinuation()
                }
            }
        }
    }

    /// Snapshot stream: the concatenated finalized + current volatile transcript
    public var snapshots: AsyncStream<StreamingTranscriptSnapshot> {
        AsyncStream { continuation in
            self.snapshotsContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearSnapshotsContinuation()
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

        // Force process any remaining segments
        await processSegmentsIfPossible(forceFinalize: true)

        // Use the cleaned finalized transcript (now properly deduplicated)
        let finalText = finalizedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)

        // Fallback to accumulated tokens only if finalized transcript is empty
        if finalText.isEmpty && !accumulatedTokens.isEmpty, let asrManager = asrManager {
            let finalResult = asrManager.processTranscriptionResult(
                tokenIds: accumulatedTokens,
                timestamps: [],
                encoderSequenceLength: 0,
                audioSamples: [],  // Not needed for final text conversion
                processingTime: 0
            )
            return finalResult.text
        }

        logger.info("Final transcription: \(finalText.count) characters")
        return finalText
    }

    /// Reset the transcriber for a new session
    public func reset() async throws {
        volatileTranscript = ""
        finalizedTranscript = ""
        processedChunks = 0
        startTime = Date()
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        currentSegment = CurrentSegment(
            id: UUID(), centerStartAbs: 0, chunkSamples: config.chunkSamples, revision: 0, emittedFinal: false,
            volatileProgressSamples: 0
        )

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        accumulatedTokens.removeAll()

        // Reset segment tracking
        finalizedSegments.removeAll()
        segmentFinalTexts.removeAll()
        lastFinalizedText = ""

        logger.info("StreamingAsrManager reset for source: \(String(describing: self.audioSource))")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()
        resultsContinuation?.finish()
        snapshotsContinuation?.finish()

        // Cleanup audio converter state
        await audioConverter.cleanup()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear continuations
    private func clearResultsContinuation() { resultsContinuation = nil }
    private func clearSnapshotsContinuation() { snapshotsContinuation = nil }

    // MARK: - Private Methods

    /// Append new samples and process the current segment when thresholds are met
    private func appendSamplesAndProcess(_ samples: [Float]) async {
        // Append samples to buffer
        sampleBuffer.append(contentsOf: samples)

        await processSegmentsIfPossible()
    }

    /// Flush any remaining audio at end of stream (no right-context requirement)
    private func flushRemaining() async {
        // Finalize the current segment with whatever right-context is available,
        // then advance and attempt to process any remaining partial segments until exhausted.
        await processSegmentsIfPossible(forceFinalize: true)
    }

    /// Process a single assembled window: [left, chunk, right]
    private func processWindow(
        _ windowSamples: [Float],
        actualLeftSeconds: Double,
        segmentID: UUID,
        centerStartAbs: Int,
        chunkSamples: Int,
        revision: Int,
        isFinal: Bool
    ) async {
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

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = asrManager.processTranscriptionResult(
                tokenIds: tokens,  // Only current chunk tokens for progress updates
                timestamps: timestamps,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            logger.debug(
                "Chunk \(self.processedChunks): '\(interim.text)', time: \(String(format: "%.3f", processingTime))s, conf: \(String(format: "%.3f", interim.confidence))"
            )

            // Skip empty text to reduce noise
            let trimmed = interim.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return
            }

            // Filter out garbage outputs (repetitive dots, very low confidence, etc.)
            let isGarbage = isGarbageOutput(trimmed, confidence: interim.confidence)
            if isGarbage {
                logger.debug("Skipping garbage output: '\(trimmed)' (confidence: \(interim.confidence))")
                return
            }

            // Update transcript builder (volatile or final)
            await updateTranscriptBuilder(text: trimmed, isFinal: isFinal, segmentID: segmentID)

            // Build and emit result for this segment (using captured metadata)
            let sampleRate = config.asrConfig.sampleRate
            let start = CMTime(value: CMTimeValue(centerStartAbs), timescale: CMTimeScale(sampleRate))
            let durationSamples = min(chunkSamples, windowSamples.count)
            let duration = CMTime(
                value: CMTimeValue(durationSamples), timescale: CMTimeScale(sampleRate)
            )

            let result = StreamingTranscriptionResult(
                segmentID: segmentID,
                revision: revision,
                attributedText: AttributedString(trimmed),
                audioTimeRange: CMTimeRange(start: start, duration: duration),
                isFinal: isFinal,
                confidence: interim.confidence,
                timestamp: Date()
            )
            resultsContinuation?.yield(result)

            // Emit snapshot
            let snapshot = StreamingTranscriptSnapshot(
                finalized: AttributedString(finalizedTranscript),
                volatile: volatileTranscript.isEmpty ? nil : AttributedString(volatileTranscript),
                lastUpdated: Date()
            )
            snapshotsContinuation?.yield(snapshot)

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Check if the output appears to be garbage (repetitive dots, very low confidence, etc.)
    private func isGarbageOutput(_ text: String, confidence: Float) -> Bool {
        // Filter out very low confidence outputs
        if confidence < 0.3 {
            return true
        }

        // Filter out outputs that are mostly dots or repetitive characters
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Check if it's mostly dots
        let dotCount = trimmed.filter { $0 == "." }.count
        if dotCount >= trimmed.count - 1 && trimmed.count >= 3 {
            return true
        }

        // Check if it's only dots, commas, and spaces
        let allowedChars = CharacterSet(charactersIn: "., ")
        if trimmed.rangeOfCharacter(from: allowedChars.inverted) == nil && trimmed.count >= 3 {
            return true
        }

        // Filter out very short outputs with low confidence
        if text.count < 3 && confidence < 0.5 {
            return true
        }

        return false
    }

    /// Update transcript strings for snapshotting
    private func updateTranscriptBuilder(text: String, isFinal: Bool, segmentID: UUID) async {
        if isFinal {
            // Only process each segment once
            guard !finalizedSegments.contains(segmentID) else {
                return
            }

            finalizedSegments.insert(segmentID)

            let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedText.isEmpty {
                // Store the final text for this specific segment in chronological order
                segmentFinalTexts.append((segmentID, trimmedText))

                // Rebuild the complete finalized transcript from segment texts only
                rebuildFinalizedTranscript()

                lastFinalizedText = trimmedText
            }

            // Clear volatile transcript only after finalizing
            volatileTranscript = ""
        } else {
            // For volatile updates, replace the entire volatile transcript
            volatileTranscript = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    /// Rebuild the finalized transcript from stored segment texts in chronological order
    private func rebuildFinalizedTranscript() {
        finalizedTranscript = segmentFinalTexts.map { $0.1 }.joined(separator: " ")
    }

    /// Process current segment depending on available right-context; optionally force finalization
    private func processSegmentsIfPossible(forceFinalize: Bool = false) async {
        let sampleRate = config.asrConfig.sampleRate
        let left = config.leftContextSamples
        let rightFinal = config.rightContextSamples
        // Use configuration-controlled volatile timing for better customization
        let rightVolatile = config.volatileRightContextSamples
        let stepVolatile = config.volatileStepSamples

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count

        while true {
            // Determine thresholds
            let center = currentSegment.centerStartAbs
            let chunk = currentSegment.chunkSamples

            // If no audio ahead of center, break
            if currentAbsEnd <= center { break }

            // Iterative volatile emissions as progress advances within the center chunk
            let nextProgress = min(chunk, currentSegment.volatileProgressSamples + stepVolatile)
            let progressEndAbs = center + nextProgress
            let canEmitVolatile =
                (nextProgress > currentSegment.volatileProgressSamples)
                && currentAbsEnd >= (progressEndAbs + rightVolatile)
            let canEmitFinal = currentAbsEnd >= (center + chunk + rightFinal)

            if canEmitFinal || forceFinalize {
                let leftStartAbs = max(0, center - left)
                // If forcing final but we have less than a full chunk ahead, cap to available audio
                let availableAhead = max(0, currentAbsEnd - center)
                let effectiveChunk = min(chunk, availableAhead)
                let rightEndAbs = center + effectiveChunk + (forceFinalize ? 0 : rightFinal)
                let startIdx = max(leftStartAbs - bufferStartIndex, 0)
                let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)
                if startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx {
                    let window = Array(sampleBuffer[startIdx..<endIdx])
                    let segID = currentSegment.id
                    let rev = currentSegment.revision + 1
                    // Mark final emitted before processing to avoid re-entry
                    currentSegment.emittedFinal = true
                    currentSegment.revision = rev
                    await self.processWindow(
                        window,
                        actualLeftSeconds: Double(center - leftStartAbs) / Double(sampleRate),
                        segmentID: segID,
                        centerStartAbs: center,
                        chunkSamples: effectiveChunk,
                        revision: rev,
                        isFinal: true
                    )
                }

                // Advance by chunk size
                currentSegment = CurrentSegment(
                    id: UUID(),
                    centerStartAbs: center + chunk,
                    chunkSamples: config.chunkSamples,
                    revision: 0,
                    emittedFinal: false,
                    volatileProgressSamples: 0
                )

                // Trim buffer to keep only what's needed for left context
                let trimToAbs = max(0, currentSegment.centerStartAbs - left)
                let dropCount = max(0, trimToAbs - bufferStartIndex)
                if dropCount > 0 && dropCount <= sampleBuffer.count {
                    sampleBuffer.removeFirst(dropCount)
                    bufferStartIndex += dropCount
                }

                // Recompute end
                currentAbsEnd = bufferStartIndex + sampleBuffer.count
                continue
            }

            if canEmitVolatile && !currentSegment.emittedFinal {
                let leftStartAbs = max(0, center - left)
                let rightEndAbs = progressEndAbs + rightVolatile
                let startIdx = max(leftStartAbs - bufferStartIndex, 0)
                let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)
                if startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx {
                    let window = Array(sampleBuffer[startIdx..<endIdx])
                    let rev = currentSegment.revision + 1
                    currentSegment.revision = rev
                    await self.processWindow(
                        window,
                        actualLeftSeconds: Double(center - leftStartAbs) / Double(sampleRate),
                        segmentID: currentSegment.id,
                        centerStartAbs: center,
                        chunkSamples: nextProgress,
                        revision: rev,
                        isFinal: false
                    )
                    currentSegment.volatileProgressSamples = nextProgress
                    // Continue loop to allow multiple volatile emissions if enough audio is buffered
                    currentAbsEnd = bufferStartIndex + sampleBuffer.count
                    continue
                }
                // Note: do not advance center for volatile
                break
            }

            // Nothing to emit yet
            break
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
    /// Main chunk size for stable transcription (seconds). Should be 10-11s for best quality
    public let chunkSeconds: TimeInterval
    /// Quick hypothesis chunk size for immediate feedback (seconds). Typical: 1.0s
    public let hypothesisChunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s
    public let leftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency)
    public let rightContextSeconds: TimeInterval
    /// Minimum audio duration before finalizing text (seconds). Should be ~10s
    public let minContextForFinalization: TimeInterval

    /// Volatile text update timing controls for responsiveness
    /// Right context for volatile updates (seconds). Lower = more responsive but less accurate
    public let volatileRightContextSeconds: TimeInterval
    /// Step size for volatile text progression (seconds). Smaller = smoother updates
    public let volatileStepSeconds: TimeInterval

    /// Enable debug logging
    public let enableDebug: Bool
    /// Confidence threshold for promoting volatile text to final (0.0...1.0)
    public let finalizationThreshold: Double

    /// Balanced configuration - good quality with reasonable latency
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 11.0,
        hypothesisChunkSeconds: 1.0,
        leftContextSeconds: 2.0,
        rightContextSeconds: 2.0,
        minContextForFinalization: 10.0,
        volatileRightContextSeconds: 0.5,  // 500ms for moderate responsiveness
        volatileStepSeconds: 0.5,  // Update every 500ms
        enableDebug: false,
        finalizationThreshold: 0.85
    )

    /// Optimized for real-time streaming with maximum responsiveness
    /// Best for live transcription where users expect immediate feedback
    public static let realtime = StreamingAsrConfig(
        chunkSeconds: 11.0,  // Proven optimal chunk size
        hypothesisChunkSeconds: 1.0,  // Quick hypothesis updates
        leftContextSeconds: 2.0,  // Minimal left context for speed
        rightContextSeconds: 1.5,  // Reduced right context for faster finalization
        minContextForFinalization: 8.0,  // Faster finalization
        volatileRightContextSeconds: 0.125,  // 125ms - very responsive
        volatileStepSeconds: 0.267,  // ~267ms steps for smooth progression
        enableDebug: false,
        finalizationThreshold: 0.80
    )

    public init(
        chunkSeconds: TimeInterval = 11.0,
        hypothesisChunkSeconds: TimeInterval = 1.0,
        leftContextSeconds: TimeInterval = 2.0,
        rightContextSeconds: TimeInterval = 2.0,
        minContextForFinalization: TimeInterval = 10.0,
        volatileRightContextSeconds: TimeInterval = 0.5,
        volatileStepSeconds: TimeInterval = 0.5,
        enableDebug: Bool = false,
        finalizationThreshold: Double = 0.85
    ) {
        self.chunkSeconds = chunkSeconds
        self.hypothesisChunkSeconds = hypothesisChunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.minContextForFinalization = minContextForFinalization
        self.volatileRightContextSeconds = volatileRightContextSeconds
        self.volatileStepSeconds = volatileStepSeconds
        self.enableDebug = enableDebug
        self.finalizationThreshold = finalizationThreshold
    }

    /// Backward-compatible convenience initializer (deprecated)
    @available(*, deprecated, message: "Use StreamingAsrConfig(.default) or StreamingAsrConfig(.realtime)")
    public init(
        finalizationThreshold: Double = 0.85,
        chunkDuration: TimeInterval,
        enableDebug: Bool = false
    ) {
        self.init(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),
            leftContextSeconds: 2.0,
            rightContextSeconds: 2.0,
            minContextForFinalization: 10.0,
            volatileRightContextSeconds: 0.5,
            volatileStepSeconds: 0.5,
            enableDebug: enableDebug,
            finalizationThreshold: finalizationThreshold
        )
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
    var hypothesisChunkSamples: Int { Int(hypothesisChunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }
    var minContextForFinalizationSamples: Int { Int(minContextForFinalization * 16000) }
    var volatileRightContextSamples: Int { Int(volatileRightContextSeconds * 16000) }
    var volatileStepSamples: Int { Int(volatileStepSeconds * 16000) }

    // Backward-compat convenience for existing call-sites/tests
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
}

/// Segment-based transcription result (volatile iterations and final for a range)
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptionResult: Sendable {
    public let segmentID: UUID
    public let revision: Int
    public let attributedText: AttributedString
    public let audioTimeRange: CMTimeRange
    public let isFinal: Bool
    public let confidence: Float
    public let timestamp: Date

    public init(
        segmentID: UUID,
        revision: Int,
        attributedText: AttributedString,
        audioTimeRange: CMTimeRange,
        isFinal: Bool,
        confidence: Float,
        timestamp: Date
    ) {
        self.segmentID = segmentID
        self.revision = revision
        self.attributedText = attributedText
        self.audioTimeRange = audioTimeRange
        self.isFinal = isFinal
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

/// Snapshot of the full transcript state suitable for simple UIs
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptSnapshot: Sendable {
    public let finalized: AttributedString
    public let volatile: AttributedString?
    public let lastUpdated: Date

    public init(finalized: AttributedString, volatile: AttributedString?, lastUpdated: Date) {
        self.finalized = finalized
        self.volatile = volatile
        self.lastUpdated = lastUpdated
    }
}
