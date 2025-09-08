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
/// for await update in streamingAsr.segmentUpdates {
///     if update.isVolatile { /* show realtime text */ } else { /* append final */ }
/// }
/// ```
///
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingAsrManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "StreamingASR")
    private let audioConverter = AudioConverter()
    private let config: StreamingAsrConfig

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Transcription output stream
    // Removed snapshots API; use segmentUpdates only

    // Segment update stream for per-segment emissions
    private var segmentUpdatesContinuation: AsyncStream<StreamingSegmentUpdate>.Continuation?
    private var segmentUpdatesStream: AsyncStream<StreamingSegmentUpdate>?

    // ASR components
    private var asrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone
    // Snapshot of decoder state at the start of the current segment.
    // Used to re-decode the entire segment on finalization so early tokens get confirmed.
    private var segmentStartDecoderState: TdtDecoderState?

    // Sliding window state
    private var accumulatedTokens: [Int] = []
    private let maxAccumulatedTokens = 5000  // Limit to prevent memory growth
    private var lastProcessedFrame: Int = 0  // Global encoder-frame position for streaming alignment

    // No internal accumulation; clients combine final transcript externally

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]

    // Simplified segment processing
    private struct CurrentSegment {
        var id: UUID
        var centerStartAbs: Int
        var chunkSamples: Int
        var isFinalized: Bool
        var lastInterimUpdate: Int  // Last sample position where we emitted an interim result
    }
    
    private var currentSegment: CurrentSegment = .init(
        id: UUID(), centerStartAbs: 0, chunkSamples: 0, isFinalized: false, lastInterimUpdate: 0
    )

    // No public transcript accumulation; clients handle accumulation externally
    private var volatileTranscript: String = ""

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics and performance monitoring
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

        // Initialize segment updates stream
        let (segmentStream, segmentContinuation) = AsyncStream<StreamingSegmentUpdate>.makeStream()
        self.segmentUpdatesStream = segmentStream
        self.segmentUpdatesContinuation = segmentContinuation

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
        accumulatedTokens.removeAll()

        // Reset transcript tracking
        volatileTranscript = ""

        // Initialize first segment
        currentSegment = CurrentSegment(
            id: UUID(), centerStartAbs: 0, chunkSamples: config.chunkSamples,
            isFinalized: false, lastInterimUpdate: 0
        )

        // Capture decoder state snapshot at the start of the first segment
        if let asrManager = asrManager {
            let base = (source == .microphone) ? asrManager.microphoneDecoderState : asrManager.systemDecoderState
            segmentStartDecoderState = try? TdtDecoderState(from: base)
        }

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

    /// Segment update stream: individual segment emissions with timestamps
    public var segmentUpdates: AsyncStream<StreamingSegmentUpdate> {
        guard let stream = segmentUpdatesStream else {
            logger.warning("Segment updates stream was not initialized, creating fallback")
            return AsyncStream { continuation in
                continuation.finish()
            }
        }
        return stream
    }

    /// Stop streaming: flush any pending audio, emit final updates, and close streams.
    public func stop() async throws {
        logger.info("Stopping streaming ASR...")

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

        // Reset decoder state and clean up resources
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
            logger.debug("Reset decoder state after finishing")
        }

        segmentUpdatesContinuation?.finish()
        segmentUpdatesContinuation = nil
        segmentUpdatesStream = nil

        // Clean up audio converter state
        await audioConverter.cleanup()

        logger.info("Streaming stopped and resources cleaned up")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()

        segmentUpdatesContinuation?.finish()
        segmentUpdatesContinuation = nil
        segmentUpdatesStream = nil

        // Cleanup audio converter state
        await audioConverter.cleanup()

        logger.info("StreamingAsrManager cancelled")
    }

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
        leftStartAbs: Int,
        segmentID: UUID,
        centerStartAbs: Int,
        chunkSamples: Int,
        isFinal: Bool,
        isStreamEnding: Bool
    ) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            // Calculate encoder-frame metadata for streaming (NeMo-style)
            let samplesPerFrame = ASRConstants.samplesPerEncoderFrame
            let leftContextLen = max(0, centerStartAbs - leftStartAbs)
            let leftContextFrames = ASRConstants.calculateEncoderFrames(from: leftContextLen)
            let actualFramesForChunk = ASRConstants.calculateEncoderFrames(from: leftContextLen + chunkSamples)
            let globalFrameOffset = leftStartAbs / samplesPerFrame

            // Always skip the left-context frames inside this window.
            // Tells the decoder to start decoding at the beginning of the new chunk region.
            // Any inter-chunk alignment is handled internally by timeJump.
            var contextFrameAdjustment = leftContextFrames

            // Select decoder state for source and run frame-accurate inference
            // For finalization, re-decode the entire segment from the snapshot taken
            // at the segment start so early tokens are included in the final.
            var state: TdtDecoderState
            if isFinal, let snap = segmentStartDecoderState {
                state = snap
                // Clear streaming-specific alignment when re-decoding a finalized segment
                state.timeJump = nil
            } else {
                state = (audioSource == .microphone) ? asrManager.microphoneDecoderState : asrManager.systemDecoderState
            }
            // Pad to model capacity (15s) to satisfy CoreML input constraints
            let paddedWindow = asrManager.padAudioIfNeeded(windowSamples, targetLength: 240_000)
            // For finals, let the decoder see the full window (start from left) and rely on dedup
            // For interims, skip left-context frames inside this window
            let effectiveContextAdjustment = isFinal ? 0 : contextFrameAdjustment

            let (hyp, encLen) = try await asrManager.executeMLInferenceWithTimings(
                paddedWindow,
                originalLength: windowSamples.count,
                actualAudioFrames: actualFramesForChunk,  // decode only left+chunk frames
                enableDebug: config.enableDebug,
                decoderState: &state,
                contextFrameAdjustment: effectiveContextAdjustment,
                isLastChunk: isStreamEnding,
                globalFrameOffset: globalFrameOffset
            )

            // Persist decoder state only on final updates to avoid hypothesis drift from volatile passes
            if isFinal {
                if audioSource == .microphone {
                    asrManager.microphoneDecoderState = state
                } else {
                    asrManager.systemDecoderState = state
                }
            }

            // Update global processed frame tracker
            if hyp.maxTimestamp > 0 { lastProcessedFrame = hyp.maxTimestamp }

            // Token filtering: for interims, drop left-context tokens strictly.
            // For finals, keep everything and rely on dedup (prevents losing early-chunk tokens).
            var filteredTokens: [Int] = []
            var filteredTimestamps: [Int] = []
            var filteredConfidences: [Float] = []

            if !isFinal {
                let startFrameCutoff = globalFrameOffset + leftContextFrames
                for i in 0..<hyp.ySequence.count {
                    let ts = hyp.timestamps.indices.contains(i) ? hyp.timestamps[i] : 0
                    if ts >= startFrameCutoff {
                        filteredTokens.append(hyp.ySequence[i])
                        filteredTimestamps.append(ts)
                        if hyp.tokenConfidences.indices.contains(i) {
                            filteredConfidences.append(hyp.tokenConfidences[i])
                        }
                    }
                }
            } else {
                filteredTokens = hyp.ySequence
                filteredTimestamps = hyp.timestamps
                filteredConfidences = hyp.tokenConfidences
            }

            // Clean intra-chunk punctuation sequences before cross-boundary dedup
            let (punctCleaned, removedIdxSet) = asrManager.cleanPunctuationSequenceTokens(filteredTokens)
            let adjustedByPunct: ([Int], [Int], [Float]) = {
                if removedIdxSet.isEmpty { return (filteredTokens, filteredTimestamps, filteredConfidences) }
                var tks: [Int] = []
                var ts: [Int] = []
                var conf: [Float] = []
                for (i, tok) in filteredTokens.enumerated() {
                    if removedIdxSet.contains(i) { continue }
                    tks.append(tok)
                    ts.append(filteredTimestamps.indices.contains(i) ? filteredTimestamps[i] : 0)
                    if filteredConfidences.indices.contains(i) { conf.append(filteredConfidences[i]) }
                }
                return (tks, ts, conf)
            }()

            // Deduplicate against accumulated tokens across finalized segments
            let (dedupedTokens, removedCount) = asrManager.removeDuplicateTokenSequence(
                previous: accumulatedTokens, current: adjustedByPunct.0, maxOverlap: 30
            )
            let adjustedTimestamps = removedCount > 0 ? Array(adjustedByPunct.1.dropFirst(removedCount)) : adjustedByPunct.1
            let adjustedConfidences = removedCount > 0 ? Array(adjustedByPunct.2.dropFirst(removedCount)) : adjustedByPunct.2

            // For streaming, don't accumulate tokens blindly since they're deduplicated
            // by transcribeStreamingChunk. Only update accumulated tokens if this is final.
            if isFinal {
                // Only add truly new tokens after deduplication
                accumulatedTokens.append(contentsOf: dedupedTokens)

                // Keep only recent tokens to prevent unbounded growth
                if accumulatedTokens.count > maxAccumulatedTokens {
                    let removeCount = accumulatedTokens.count - maxAccumulatedTokens
                    accumulatedTokens.removeFirst(removeCount)
                    logger.debug(
                        "Trimmed \(removeCount) old tokens, keeping \(self.maxAccumulatedTokens) recent tokens")
                }
            }

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Periodic tick for long-running streams (placeholder, no cleanup needed now)
            let now = Date()
            if now.timeIntervalSince(lastMemoryCleanup) > memoryCleanupInterval {
                lastMemoryCleanup = now
                logger.debug("Streaming health check after \(self.processedChunks) chunks")
            }

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            let interim = asrManager.processTranscriptionResult(
                tokenIds: dedupedTokens,  // emit only new tokens
                timestamps: adjustedTimestamps,
                confidences: adjustedConfidences,
                encoderSequenceLength: encLen,
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
                endTime: endTimeSeconds,
                confidence: interim.confidence,
                tokenTimings: interim.tokenTimings
            )

            // No snapshot emission; clients consume segmentUpdates

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Update transcript strings for snapshotting and emit segment updates
    private func updateTranscriptBuilder(
        text: String, 
        isFinal: Bool, 
        segmentID: UUID, 
        startTime: TimeInterval = 0, 
        endTime: TimeInterval = 0,
        confidence: Float = 1.0,
        tokenTimings: [TokenTiming]? = nil
    ) async {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty && !isGarbageText(trimmedText) else { return }

        if isFinal {
            // Emit finalized segment update (clients accumulate)
            let segmentUpdate = StreamingSegmentUpdate(
                segmentId: segmentID,
                text: trimmedText,
                timestamp: Date(),
                isVolatile: false,
                startTime: startTime,
                endTime: endTime,
                confidence: confidence,
                tokenTimings: tokenTimings
            )
            if config.enableDebug {
                print("[StreamingASR] FINAL seg=\(segmentID) text=\(trimmedText)")
            }
            segmentUpdatesContinuation?.yield(segmentUpdate)

            // Clear volatile transcript only after finalizing
            volatileTranscript = ""
        } else {
            // For volatile updates, replace the entire volatile transcript
            volatileTranscript = trimmedText

            // Emit volatile segment update
            let segmentUpdate = StreamingSegmentUpdate(
                segmentId: segmentID,
                text: trimmedText,
                timestamp: Date(),
                isVolatile: true,
                startTime: startTime,
                endTime: endTime,
                confidence: confidence,
                tokenTimings: tokenTimings
            )
            if config.enableDebug {
                print("[StreamingASR] VOLATILE seg=\(segmentID) text=\(trimmedText)")
            }
            segmentUpdatesContinuation?.yield(segmentUpdate)
        }
    }

    // Removed rebuild and cleanup helpers – accumulation now handled by clients

    /// Process a final segment with extended left context for consistent window size
    private func processFinalSegment(center: Int, chunk: Int, currentAbsEnd: Int, forceFinalize: Bool) async -> Bool {
        let sampleRate = config.asrConfig.sampleRate
        let left = config.leftContextSamples
        let rightFinal = config.rightContextSamples

        let canEmitFinal = currentAbsEnd >= (center + chunk + rightFinal) || forceFinalize
        guard canEmitFinal else { return false }

        let availableAhead = max(0, currentAbsEnd - center)
        let effectiveChunk = min(chunk, availableAhead)

        // Calculate extended left context to maintain consistent window size
        let actualLeftContext: Int
        if forceFinalize {
            // For final segments, extend left context so that left + chunk = expected chunk size
            // This ensures the model gets a consistent context window
            let missingChunk = chunk - effectiveChunk  // How much shorter the final chunk is
            let extendedLeft = left + missingChunk  // Compensate with more left context

            // Ensure we don't exceed available audio (can't go before start)
            let maxAvailableLeft = center
            actualLeftContext = min(extendedLeft, maxAvailableLeft)

            logger.debug(
                "Final segment: extending left context from \(left) to \(actualLeftContext) samples (chunk: \(effectiveChunk)/\(chunk), missing: \(missingChunk))"
            )
        } else {
            actualLeftContext = left
        }

        let leftStartAbs = max(0, center - actualLeftContext)
        let rightEndAbs = center + effectiveChunk + (forceFinalize ? 0 : rightFinal)
        let startIdx = max(leftStartAbs - bufferStartIndex, 0)
        let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)

        guard startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx else { return false }

        let window = Array(sampleBuffer[startIdx..<endIdx])
        let segID = currentSegment.id

        currentSegment.isFinalized = true

        // Log window size for debugging
        let windowDuration = Double(window.count) / Double(sampleRate)
        let leftDuration = Double(center - leftStartAbs) / Double(sampleRate)
        logger.debug(
            "Processing final segment: window=\(String(format: "%.1f", windowDuration))s, left=\(String(format: "%.1f", leftDuration))s, chunk=\(String(format: "%.1f", Double(effectiveChunk) / Double(sampleRate)))s"
        )

        await self.processWindow(
            window,
            leftStartAbs: leftStartAbs,
            segmentID: segID,
            centerStartAbs: center,
            chunkSamples: effectiveChunk,
            isFinal: true,
            isStreamEnding: forceFinalize
        )

        return true
    }

    /// Process an interim segment with limited context
    private func processInterimSegment(center: Int, currentAbsEnd: Int) async -> Bool {
        let left = config.leftContextSamples
        let interimRightContext = config.interimRightContextSamples
        let interimStep = config.interimUpdateSamples

        let nextInterimPos = currentSegment.lastInterimUpdate + interimStep
        let canEmitInterim =
            !currentSegment.isFinalized
            && nextInterimPos <= (currentAbsEnd - center)
            && currentAbsEnd >= (center + nextInterimPos + interimRightContext)

        guard canEmitInterim else { return false }

        let leftStartAbs = max(0, center - left)
        let rightEndAbs = center + nextInterimPos + interimRightContext
        let startIdx = max(leftStartAbs - bufferStartIndex, 0)
        let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)

        guard startIdx >= 0, endIdx <= sampleBuffer.count, startIdx < endIdx else { return false }

        let window = Array(sampleBuffer[startIdx..<endIdx])
        currentSegment.lastInterimUpdate = nextInterimPos

        await self.processWindow(
            window,
            leftStartAbs: leftStartAbs,
            segmentID: currentSegment.id,
            centerStartAbs: center,
            chunkSamples: nextInterimPos,
            isFinal: false,
            isStreamEnding: false
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
            isFinalized: false,
            lastInterimUpdate: 0
        )

        // Trim buffer to keep only what's needed for left context
        let trimToAbs = max(0, currentSegment.centerStartAbs - left)
        let dropCount = max(0, trimToAbs - bufferStartIndex)
        if dropCount > 0 && dropCount <= sampleBuffer.count {
            sampleBuffer.removeFirst(dropCount)
            bufferStartIndex += dropCount
        }

        // Update decoder state snapshot for the new segment start
        if let asrManager = asrManager {
            let base = (audioSource == .microphone) ? asrManager.microphoneDecoderState : asrManager.systemDecoderState
            segmentStartDecoderState = try? TdtDecoderState(from: base)
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
    public var memoryStats: (sampleBufferSize: Int, accumulatedTokens: Int, processedChunks: Int) {
        return (
            sampleBufferSize: sampleBuffer.count,
            accumulatedTokens: accumulatedTokens.count,
            processedChunks: processedChunks
        )
    }
}
