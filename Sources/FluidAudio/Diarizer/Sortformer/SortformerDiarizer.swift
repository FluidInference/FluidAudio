import Accelerate
import CoreML
import Foundation
import OSLog

/// Streaming speaker diarization using NVIDIA's Sortformer model.
///
/// Sortformer provides end-to-end streaming diarization with 4 fixed speaker slots,
/// achieving ~11% DER on DI-HARD III in real-time.
///
/// Usage:
/// ```swift
/// let diarizer = SortformerDiarizerPipeline()
/// try await diarizer.initialize(preprocessorPath: url1, mainModelPath: url2)
///
/// // Streaming mode
/// for audioChunk in audioStream {
///     if let result = try diarizer.processSamples(audioChunk) {
///         // Handle speaker probabilities
///     }
/// }
///
/// // Or complete file
/// let result = try diarizer.processComplete(audioSamples)
/// ```
public final class SortformerDiarizer {
    /// Lock for thread-safe access to mutable state
    private let lock = OSAllocatedUnfairLock()

    /// Accumulated results
    public var timeline: SortformerTimeline {
        lock.lock()
        defer { lock.unlock() }
        return _timeline
    }
    
    private var _timeline: SortformerTimeline

    /// Check if diarizer is ready for processing.
    public var isAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _models != nil
    }

    /// Streaming state
    public var state: SortformerStreamingState {
        lock.lock()
        defer { lock.unlock() }
        return _state
    }
    private var _state: SortformerStreamingState

    /// Number of frames processed
    public var numFramesProcessed: Int {
        return state.nextNewFrame
    }

    /// Configuration
    public let config: SortformerConfig

    private let logger = AppLogger(category: "SortformerDiarizerPipeline")
    private let stateUpdater: SortformerStateUpdater

    private var _models: SortformerModels?

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private let melSpectrogram = NeMoMelSpectrogram()
    private let embeddingManager: EmbeddingManager

    // Audio buffering
    private var audioBuffer: [Float] = []
    private var lastAudioSample: Float = 0

    // Feature buffering
    internal var featureBuffer: [Float] = []

    // Chunk tracking
    private var startFeat: Int = 0  // Current position in mel feature stream
    private var diarizerChunkIndex: Int = 0

    // MARK: - Initialization

    public init(
        config: SortformerConfig = .default,
        postProcessingConfig: SortformerTimelineConfig? = nil,
        embeddingConfig: EmbeddingConfig? = nil
    ) {
        self.config = config
        self.stateUpdater = SortformerStateUpdater(config: config)
        self._state = SortformerStreamingState(config: config)
        self.embeddingManager = EmbeddingManager(config: .default)
        try? self.embeddingManager.initialize()
        self._timeline = SortformerTimeline(
            config: postProcessingConfig ?? .default(for: config),
            embeddingManager: embeddingManager
        )
    }

    /// Initialize with CoreML models (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - mainModelPath: Path to Sortformer.mlpackage
    public func initialize(
        mainModelPath: URL
    ) async throws {
        logger.info("Initializing Sortformer diarizer (combined pipeline mode)")

        let loadedModels = try await SortformerModels.load(
            config: config,
            mainModelPath: mainModelPath
        )

        // Use withLock helper to avoid direct NSLock usage in async context
        withLock {
            self._models = loadedModels
            self._state = SortformerStreamingState(config: config)
            self.lastAudioSample = 0
            self.resetBuffersLocked()
        }
        logger.info("Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s")
    }

    /// Execute a closure while holding the lock
    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// Initialize with pre-loaded models.
    public func initialize(models: SortformerModels) {
        lock.lock()
        defer { lock.unlock() }

        self._models = models
        resetBuffersLocked()
        logger.info("Sortformer initialized with pre-loaded models")
    }

    /// Reset all internal state for a new audio stream.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        resetBuffersLocked()
        logger.debug("Sortformer state reset")
    }

    /// Internal reset - caller must hold lock
    private func resetBuffersLocked() {
        audioBuffer = []
        featureBuffer = []
        lastAudioSample = 0
        startFeat = 0
        diarizerChunkIndex = 0
        _timeline.reset()
        _state = SortformerStreamingState(config: config)
        
        audioBuffer.reserveCapacity(config.chunkMelFrames * config.melStride)
        featureBuffer.reserveCapacity((config.chunkMelFrames + config.coreFrames) * config.melFeatures)
    }

    /// Remove a speaker from the streaming state (FIFO + speaker cache).
    ///
    /// Scrubs all trace of the speaker from the internal buffers so future
    /// model calls no longer see that speaker's activity.
    ///
    /// - Parameter speakerIndex: The speaker slot to remove (0..<numSpeakers)
    public func removeSpeaker(at speakerIndex: Int) {
        lock.lock()
        defer { lock.unlock() }
        stateUpdater.removeSpeaker(at: speakerIndex, from: &_state)
        _timeline.freeSlot(speakerIndex)
    }

    /// Cleanup resources.
    public func cleanup() {
        lock.lock()
        defer { lock.unlock() }

        _models = nil
        _state.cleanup()
        resetBuffersLocked()
        logger.info("Sortformer resources cleaned up")
    }

    // MARK: - Streaming Processing

    /// Add audio samples to the processing buffer.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    public func addAudio<C: Collection>(_ samples: C) where C.Element == Float {
        lock.lock()
        defer { lock.unlock() }

        audioBuffer.append(contentsOf: samples)
    }

    /// Process buffered audio and return any new results.
    ///
    /// Call this after adding audio with `addAudio()`.
    ///
    /// - Returns: New chunk results if enough audio was processed, nil otherwise
    public func process() throws -> SortformerTimelineDifference? {
        lock.lock()
        defer { lock.unlock() }
        return try processLocked()
    }

    /// Process a chunk of audio in one call.
    ///
    /// Convenience method that combines `addAudio()` and `process()`.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    /// - Returns: New chunk results if enough audio was processed
    public func processSamples(_ samples: [Float]) throws -> SortformerTimelineDifference? {
        lock.lock()
        defer { lock.unlock() }

        audioBuffer.append(contentsOf: samples)
        return try processLocked()
    }

    /// Internal process - caller must hold lock
    private func processLocked() throws -> SortformerTimelineDifference? {
        guard let models = _models else {
            throw SortformerError.notInitialized
        }
        
        // TODO: INIT EMPTY DIFF
        var difference = SortformerTimelineDifference()
        var ran = false

        // Step 1: Run preprocessor on available audio
        while let (chunkFeatures, chunkLengths) = getNextChunkFeatures() {
            ran = true
            let output = try models.runMainModel(
                chunk: chunkFeatures,
                chunkLength: chunkLengths,
                spkcache: _state.spkcache,
                spkcacheLength: _state.spkcacheLength,
                fifo: _state.fifo,
                fifoLength: _state.fifoLength,
                config: config
            )

            // Raw predictions are already probabilities (model applies sigmoid internally)
            // DO NOT apply sigmoid again
            let probabilities = output.predictions

            // Trim embeddings to actual length
            let embLength = output.chunkLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.preEncoderDims))

            // Update state with correct context values
            let chunk = try stateUpdater.streamingUpdate(
                state: &_state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: diarizerChunkIndex > 0 ? config.chunkLeftContext : 0,
                rightContext: config.chunkRightContext
            )

            let diff = try _timeline.addChunk(chunk, dropOldEmbeddingFrames: true)
            difference.apply(diff)

            diarizerChunkIndex += 1
        }

        if ran {
            return difference
        }
        return nil
    }

    // MARK: - Complete File Processing

    /// Progress callback type: (chunksProcessed, totalChunks)
    public typealias ProgressCallback = (Int, Int) -> Void
    public typealias AsyncProgressCallback = (Int, Int) async -> Void

    /// Process complete audio file.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples (16kHz mono)
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Complete diarization result
    private func processCompleteAsync(
        _ samples: [Float],
        progressCallback: AsyncProgressCallback? = nil
    ) async throws -> SortformerTimeline {
        try lock.withLock {
            guard _models != nil else {
                throw SortformerError.notInitialized
            }
            
            // Reset for fresh processing
            resetBuffersLocked()
            
            // Feed audio to embedding manager for speaker embedding extraction
            embeddingManager.addAudio(from: samples[...], lastAudioSample: 0)
        }
            
        var difference = SortformerTimelineDifference()
        var featureProvider = SortformerFeatureLoader(config: self.config, audio: samples)
        
        while let (chunkIndex, chunkFeatures, chunkLength, leftOffset, rightOffset) = featureProvider.next() {
            let diff = try lock.withLock {
                guard let models = _models else {
                    throw SortformerError.notInitialized
                }
                // Run main model
                let output = try models.runMainModel(
                    chunk: chunkFeatures,
                    chunkLength: chunkLength,
                    spkcache: _state.spkcache,
                    spkcacheLength: _state.spkcacheLength,
                    fifo: _state.fifo,
                    fifoLength: _state.fifoLength,
                    config: config
                )
                
                let probabilities = output.predictions
                
                // Trim embeddings to actual length
                let embLength = output.chunkLength
                let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.preEncoderDims))
                
                // Compute left/right context for prediction extraction
                let leftContext = IndexUtils.roundDiv(leftOffset, config.subsamplingFactor)
                let rightContext = IndexUtils.ceilDiv(rightOffset, config.subsamplingFactor)
                
                // Update state
                let chunk = try stateUpdater.streamingUpdate(
                    state: &_state,
                    chunk: chunkEmbs,
                    preds: probabilities,
                    leftContext: leftContext,
                    rightContext: rightContext
                )
                
                diarizerChunkIndex += 1
                return try _timeline.addChunk(chunk, dropOldEmbeddingFrames: true)
            }
            
            difference.apply(diff)
            
            await progressCallback?(chunkIndex + 1, featureProvider.numChunks)
        }
        
        return try lock.withLock {
            try _timeline.finalize()
            
            // Save updated state
            
            if config.debugMode {
                print(
                    "[DEBUG] Phase 2 complete: diarizerChunks=\(diarizerChunkIndex), totalProbs=\(_timeline.framePredictions.count + _timeline.tentativePredictions.count), totalFrames=\(_state.nextNewFrame)"
                )
                fflush(stdout)
            }
            return _timeline
        }
    }
    
    /// Process complete audio file.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples (16kHz mono)
    ///   - progressCallback: Asynchronous callback for progress updates
    /// - Returns: Complete diarization result
    public func processComplete(
        _ samples: [Float],
        progressCallback: @escaping AsyncProgressCallback
    ) async throws -> SortformerTimeline {
        return try await processCompleteAsync(samples, progressCallback: progressCallback)
    }
    
    /// Process complete audio file.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples (16kHz mono)
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Complete diarization result
    public func processComplete(
        _ samples: [Float],
        progressCallback: ProgressCallback? = nil
    ) throws -> SortformerTimeline {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<SortformerTimeline, Error>!

        Task {
            do {
                let timeline = try await processCompleteAsync(samples) { current, total in
                    // Wrap the sync callback so it can be 'awaited' internally
                    progressCallback?(current, total)
                }
                result = .success(timeline)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        return try result.get()
    }

    // MARK: - Helpers

    /// Preprocess audio into mel features - caller must hold lock
    private func preprocessAudioToFeatures() -> Bool {
        guard audioBuffer.count >= config.melWindow else {
            return false
        }

        // Demand-Driven Optimization:
        // Calculate exactly how many features we need for the next chunk
        // needed = (startFeat + core + RC) - currentFeatureCount

        let featLength = featureBuffer.count / config.melFeatures
        let coreFrames = config.chunkLen * config.subsamplingFactor
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor

        // Calculate absolute target position in feature stream
        // For Chunk 0: startFeat=0. Target=104.
        // For Chunk 1: startFeat=8. Target=112.
        let targetEnd = startFeat + coreFrames + rightContextFrames
        let framesNeeded = targetEnd - featLength

        // If we already have enough frames, we don't strictly need to process more right now.
        // However, to keep the pipeline moving smoothly, we can process if we have a full chunk buffered.
        // But to strictly prioritize efficiency/latency balance as requested:
        if framesNeeded <= 0 {
            return false
        }

        // Calculate audio samples needed to produce 'framesNeeded'
        // If we are appending to existing stream (featureBuffer not empty), we need stride * N.
        // If featureBuffer is empty (start of stream), we need window + (N-1)*stride.

        let samplesNeeded: Int
        if featureBuffer.isEmpty {
            samplesNeeded = (framesNeeded - 1) * config.melStride + config.melWindow
        } else {
            samplesNeeded = framesNeeded * config.melStride
        }

        // Wait until we have enough audio to satisfy the demand
        guard audioBuffer.count >= samplesNeeded else {
            return false
        }

        let audioSlice = audioBuffer.prefix(samplesNeeded)
        let (mel, melFrames, _) = melSpectrogram.computeFlatTransposed(
            audio: audioSlice,
            lastAudioSample: lastAudioSample
        )
        
        self.embeddingManager.addAudio(
            from: audioSlice,
            lastAudioSample: lastAudioSample
        )
        
        guard melFrames > 0 else {
            return false
        }

        featureBuffer.append(contentsOf: mel)
        
        // Remove old audio context
        let samplesConsumed = melFrames * config.melStride
        lastAudioSample = audioBuffer[samplesConsumed - 1] // For preemph filter
        audioBuffer.removeFirst(samplesConsumed)
        return true
    }

    /// Get next chunk features - caller must hold lock
    private func getNextChunkFeatures() -> (mel: [Float], melLength: Int)? {
        let hasNewChunk = preprocessAudioToFeatures()
        let featLength = featureBuffer.count / config.melFeatures
        let coreFrames = config.chunkLen * config.subsamplingFactor
        let leftContextFrames = config.chunkLeftContext * config.subsamplingFactor
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor

        // Calculate end of core chunk
        let endFeat = min(startFeat + coreFrames, featLength)

        // Need at least one core frame
        guard endFeat > startFeat else {
            if hasNewChunk {
                print("Audio and mels out of sync!")
            }
            return nil
        }
        guard endFeat + rightContextFrames <= featLength else {
            if hasNewChunk {
                print("Audio and mels out of sync!")
            }
            return nil
        }
        
        if !hasNewChunk {
            print("Audio and mels out of sync (it's suddenly full????)!")
        }

        // Calculate offsets
        let leftOffset = min(leftContextFrames, startFeat)
        let rightOffset = rightContextFrames

        // Extract chunk with context
        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        let chunkStartIndex = chunkStartFrame * config.melFeatures
        let chunkEndIndex = chunkEndFrame * config.melFeatures

        let mel = Array(featureBuffer[chunkStartIndex..<chunkEndIndex])
        let chunkLength = chunkEndFrame - chunkStartFrame

        // Advance position
        startFeat = endFeat

        // Remove consumed frames from buffer (frames before our new startFeat - leftContext)
        // We keep leftContextFrames history for the next chunk's Left Context
        let newBufferStart = max(0, startFeat - leftContextFrames)
        let framesToRemove = newBufferStart
        if framesToRemove > 0 {
            featureBuffer.removeFirst(framesToRemove * config.melFeatures)
            startFeat -= framesToRemove
        }

        return (mel, chunkLength)
    }
}
