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
/// let streamingAsr = StreamingAsrManager()
/// try await streamingAsr.start()
///
/// // Listen for transcription updates
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

    // Transcription output stream
    private var snapshotsContinuation: AsyncStream<StreamingTranscriptSnapshot>.Continuation?

    // ASR components
    private var asrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone

    // Sliding window state
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var accumulatedTokens: [Int] = []
    private let maxAccumulatedTokens = 5000  // Limit to prevent memory growth

    // Segment tracking to prevent duplicates
    private var finalizedSegments: Set<UUID> = []
    private var segmentFinalTexts: [(UUID, String)] = []  // Track final text per segment in order
    private var timestampedSegments: [TimestampedSegment] = []  
    private let maxSegmentTexts = 100  // Circular buffer limit
    private var cachedFinalizedTranscript: String = ""  // Cache to avoid rebuilding
    private var lastFinalizedText: String = ""

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]

    // Simplified segment processing
    private struct CurrentSegment {
        var id: UUID
        var centerStartAbs: Int
        var chunkSamples: Int
        var revision: Int
        var emittedFinal: Bool
        var lastInterimUpdate: Int  // Last sample position where we emitted an interim result
    }
    private var currentSegment: CurrentSegment = .init(
        id: UUID(), centerStartAbs: 0, chunkSamples: 0, revision: 0, emittedFinal: false, lastInterimUpdate: 0
    )

    // Two-tier transcription state (like Apple's Speech API)
    public private(set) var volatileTranscript: String = ""
    public private(set) var finalizedTranscript: String = "" {
        didSet {
            cachedFinalizedTranscript = finalizedTranscript
        }
    }

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics and performance monitoring
    private var startTime: Date?
    private var processedChunks: Int = 0
    private var lastMemoryCleanup: Date = Date()
    private let memoryCleanupInterval: TimeInterval = 30.0  // Cleanup every 30 seconds

    /// Initialize the streaming ASR manager
    /// - Parameter config: Configuration for streaming behavior
    public init(config: StreamingAsrConfig = .default) {
        self.config = config

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info(
            "Initialized StreamingAsrManager with mode: \(String(describing: config.mode)), chunk=\(config.chunkSeconds)s"
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

        // Reset transcript tracking
        finalizedSegments.removeAll()
        segmentFinalTexts.removeAll()
        timestampedSegments.removeAll()
        finalizedTranscript = ""
        volatileTranscript = ""

        startTime = Date()

        // Initialize first segment
        currentSegment = CurrentSegment(
            id: UUID(), centerStartAbs: 0, chunkSamples: config.chunkSamples, revision: 0,
            emittedFinal: false, lastInterimUpdate: 0
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

        // Reset decoder state and clean up resources
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
            logger.debug("Reset decoder state after finishing")
        }

        // Clean up continuations safely
        snapshotsContinuation?.finish()
        snapshotsContinuation = nil

        // Clean up audio converter state
        await audioConverter.cleanup()

        logger.info("Final transcription: \(finalText.count) characters")
        return finalText
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()

        // Clean up continuations safely
        snapshotsContinuation?.finish()
        snapshotsContinuation = nil

        // Cleanup audio converter state
        await audioConverter.cleanup()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear continuations
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

            // Update state with token window management
            accumulatedTokens.append(contentsOf: tokens)

            // Keep only recent tokens to prevent unbounded growth
            if accumulatedTokens.count > maxAccumulatedTokens {
                let removeCount = accumulatedTokens.count - maxAccumulatedTokens
                accumulatedTokens.removeFirst(removeCount)
                logger.debug("Trimmed \(removeCount) old tokens, keeping \(self.maxAccumulatedTokens) recent tokens")
            }

            lastProcessedFrame = max(lastProcessedFrame, timestamps.max() ?? 0)
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Periodic memory cleanup for long-running streams
            let now = Date()
            if now.timeIntervalSince(lastMemoryCleanup) > memoryCleanupInterval {
                lastMemoryCleanup = now
                logger.debug("Performing periodic memory cleanup after \(self.processedChunks) chunks")
                cleanupOldSegments()
            }

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = asrManager.processTranscriptionResult(
                tokenIds: tokens,  // Only current chunk tokens for progress updates
                timestamps: timestamps,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            // Skip empty text and garbage artifacts to reduce noise
            let trimmed = interim.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || isGarbageText(trimmed) {
                return
            }

            // Calculate timing information for this segment
            let sampleRate = Double(config.asrConfig.sampleRate)
            let startTimeSeconds = Double(centerStartAbs) / sampleRate
            let endTimeSeconds = Double(centerStartAbs + chunkSamples) / sampleRate

            // Update transcript builder (volatile or final)
            await updateTranscriptBuilder(
                text: trimmed,
                isFinal: isFinal,
                segmentID: segmentID,
                startTime: startTimeSeconds,
                endTime: endTimeSeconds
            )

            // Emit snapshot
            let snapshot = StreamingTranscriptSnapshot(
                finalized: AttributedString(finalizedTranscript),
                volatile: volatileTranscript.isEmpty ? nil : AttributedString(volatileTranscript),
                lastUpdated: Date(),
                timestampedSegments: timestampedSegments
            )
            // Safe continuation access
            if let continuation = snapshotsContinuation {
                continuation.yield(snapshot)
            }

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Update transcript strings for snapshotting
    private func updateTranscriptBuilder(
        text: String, isFinal: Bool, segmentID: UUID, startTime: TimeInterval = 0, endTime: TimeInterval = 0
    ) async {
        if isFinal {
            // Only process each segment once
            guard !finalizedSegments.contains(segmentID) else {
                return
            }

            finalizedSegments.insert(segmentID)

            let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedText.isEmpty && !isGarbageText(trimmedText) {
                // Store the final text for this specific segment in chronological order
                segmentFinalTexts.append((segmentID, trimmedText))

                // Store the timestamped segment with timing information
                let timestampedSegment = TimestampedSegment(
                    id: segmentID,
                    text: trimmedText,
                    startTime: startTime,
                    endTime: endTime,
                    confidence: 1.0  // TODO: Pass actual confidence from processing
                )
                timestampedSegments.append(timestampedSegment)

                // Clean up old segments to prevent unbounded memory growth
                cleanupOldSegments()

                // Rebuild the complete finalized transcript from segment texts only
                rebuildFinalizedTranscript()

                lastFinalizedText = trimmedText
            }

            // Clear volatile transcript only after finalizing
            volatileTranscript = ""
        } else {
            // For volatile updates, replace the entire volatile transcript
            let trimmedVolatile = text.trimmingCharacters(in: .whitespacesAndNewlines)
            volatileTranscript = isGarbageText(trimmedVolatile) ? "" : trimmedVolatile
        }
    }

    /// Efficiently rebuild the finalized transcript from stored segment texts
    private func rebuildFinalizedTranscript() {
        guard !segmentFinalTexts.isEmpty else {
            finalizedTranscript = ""
            return
        }

        // Use more efficient concatenation for large transcripts
        if segmentFinalTexts.count > 10 {
            var result = ""
            result.reserveCapacity(segmentFinalTexts.reduce(0) { $0 + $1.1.count } + segmentFinalTexts.count)
            for (index, segment) in segmentFinalTexts.enumerated() {
                if index > 0 { result += " " }
                result += segment.1
            }
            finalizedTranscript = result
        } else {
            finalizedTranscript = segmentFinalTexts.map { $0.1 }.joined(separator: " ")
        }
    }

    /// Clean up old segments to prevent unbounded memory growth
    /// Maintains only the most recent segments as a circular buffer
    private func cleanupOldSegments() {
        guard segmentFinalTexts.count > maxSegmentTexts else { return }

        // Remove oldest segments but keep the finalized transcript intact
        let excessCount = segmentFinalTexts.count - maxSegmentTexts

        // Get UUIDs of segments we're about to remove
        let removedSegmentIDs = Set(segmentFinalTexts.prefix(excessCount).map { $0.0 })

        // Remove from all three collections
        segmentFinalTexts.removeFirst(excessCount)
        timestampedSegments.removeFirst(excessCount)  // Keep in sync with segmentFinalTexts
        finalizedSegments.subtract(removedSegmentIDs)

        logger.debug("Cleaned up \(excessCount) old segments, keeping \(self.segmentFinalTexts.count) recent segments")
    }

    /// Process a final segment with full context
    private func processFinalSegment(center: Int, chunk: Int, currentAbsEnd: Int, forceFinalize: Bool) async -> Bool {
        let sampleRate = config.asrConfig.sampleRate
        let left = config.leftContextSamples
        let rightFinal = config.rightContextSamples

        let canEmitFinal = currentAbsEnd >= (center + chunk + rightFinal) || forceFinalize
        guard canEmitFinal else { return false }

        let leftStartAbs = max(0, center - left)
        let availableAhead = max(0, currentAbsEnd - center)
        let effectiveChunk = min(chunk, availableAhead)
        let rightEndAbs = center + effectiveChunk + (forceFinalize ? 0 : rightFinal)
        let startIdx = max(leftStartAbs - bufferStartIndex, 0)
        let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)

        guard startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx else { return false }

        let window = Array(sampleBuffer[startIdx..<endIdx])
        let segID = currentSegment.id
        let rev = currentSegment.revision + 1

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

        return true
    }

    /// Process an interim segment with limited context
    private func processInterimSegment(center: Int, currentAbsEnd: Int) async -> Bool {
        let sampleRate = config.asrConfig.sampleRate
        let left = config.leftContextSamples
        let interimRightContext = config.interimRightContextSamples
        let interimStep = config.interimUpdateSamples

        let nextInterimPos = currentSegment.lastInterimUpdate + interimStep
        let canEmitInterim =
            !currentSegment.emittedFinal
            && nextInterimPos <= (currentAbsEnd - center)
            && currentAbsEnd >= (center + nextInterimPos + interimRightContext)

        guard canEmitInterim else { return false }

        let leftStartAbs = max(0, center - left)
        let rightEndAbs = center + nextInterimPos + interimRightContext
        let startIdx = max(leftStartAbs - bufferStartIndex, 0)
        let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)

        guard startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx else { return false }

        let window = Array(sampleBuffer[startIdx..<endIdx])
        let rev = currentSegment.revision + 1
        currentSegment.revision = rev
        currentSegment.lastInterimUpdate = nextInterimPos

        await self.processWindow(
            window,
            actualLeftSeconds: Double(center - leftStartAbs) / Double(sampleRate),
            segmentID: currentSegment.id,
            centerStartAbs: center,
            chunkSamples: nextInterimPos,
            revision: rev,
            isFinal: false
        )

        return true
    }

    /// Advance to the next segment and trim buffer
    private func advanceToNextSegment() {
        let left = config.leftContextSamples
        let center = currentSegment.centerStartAbs
        let chunk = currentSegment.chunkSamples

        // Advance to next segment
        currentSegment = CurrentSegment(
            id: UUID(),
            centerStartAbs: center + chunk,
            chunkSamples: config.chunkSamples,
            revision: 0,
            emittedFinal: false,
            lastInterimUpdate: 0
        )

        // Trim buffer to keep only what's needed for left context
        let trimToAbs = max(0, currentSegment.centerStartAbs - left)
        let dropCount = max(0, trimToAbs - bufferStartIndex)
        if dropCount > 0 && dropCount <= sampleBuffer.count {
            sampleBuffer.removeFirst(dropCount)
            bufferStartIndex += dropCount
        }
    }

    /// Simplified segment processing with regular interim updates
    private func processSegmentsIfPossible(forceFinalize: Bool = false) async {
        var currentAbsEnd = bufferStartIndex + sampleBuffer.count

        while true {
            let center = currentSegment.centerStartAbs
            let chunk = currentSegment.chunkSamples

            // If no audio ahead of center, break
            if currentAbsEnd <= center { break }

            // Try to emit a final result first
            if await processFinalSegment(
                center: center, chunk: chunk, currentAbsEnd: currentAbsEnd, forceFinalize: forceFinalize)
            {
                advanceToNextSegment()
                currentAbsEnd = bufferStartIndex + sampleBuffer.count
                continue
            }

            // Try to emit an interim result
            if await processInterimSegment(center: center, currentAbsEnd: currentAbsEnd) {
                currentAbsEnd = bufferStartIndex + sampleBuffer.count
                continue
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

    /// Check if text is garbage/artifacts that should be filtered out
    private func isGarbageText(_ text: String) -> Bool {
        // Filter out common ASR artifacts
        let cleaned = text.replacingOccurrences(of: " ", with: "")

        // Check if it's only periods (any number of them)
        if cleaned.allSatisfy({ $0 == "." }) && !cleaned.isEmpty {
            return true
        }

        // Check if it's only common punctuation artifacts
        let commonArtifacts: Set<Character> = [".", ",", "?", "!", ";", ":", "-"]
        if cleaned.allSatisfy({ commonArtifacts.contains($0) }) && !cleaned.isEmpty {
            return true
        }

        return false
    }

    /// Get current memory usage statistics for monitoring
    public var memoryStats: (sampleBufferSize: Int, accumulatedTokens: Int, segmentTexts: Int, processedChunks: Int) {
        return (
            sampleBufferSize: sampleBuffer.count,
            accumulatedTokens: accumulatedTokens.count,
            segmentTexts: segmentFinalTexts.count,
            processedChunks: processedChunks
        )
    }
}
