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
    private let lock = NSLock()

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
        lock.lock()
        defer { lock.unlock() }
        return _numFramesProcessed
    }
    private var _numFramesProcessed: Int = 0

    /// Configuration
    public let config: SortformerConfig

    private let logger = AppLogger(category: "SortformerDiarizerPipeline")
    private let stateUpdater: SortformerStateUpdater

    private var _models: SortformerModels?

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private let melSpectrogram = NeMoMelSpectrogram()

    // Audio buffering
    private var audioBuffer: [Float] = []
    private var lastAudioSample: Float = 0

    // Feature buffering
    internal var featureBuffer: [Float] = []

    // Chunk tracking
    private var startFeat: Int = 0  // Current position in mel feature stream
    private var diarizerChunkIndex: Int = 0

    // MARK: - Initialization

    public init(config: SortformerConfig = .default, postProcessingConfig: SortformerPostProcessingConfig = .default) {
        self.config = config
        self.stateUpdater = SortformerStateUpdater(config: config)
        self._state = SortformerStreamingState(config: config)
        self._timeline = SortformerTimeline(config: postProcessingConfig)
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
        self._state = SortformerStreamingState(config: config)
        self.lastAudioSample = 0
        resetBuffersLocked()
        logger.info("Sortformer initialized with pre-loaded models")
    }

    /// Reset all internal state for a new audio stream.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        _state = SortformerStreamingState(config: config)
        lastAudioSample = 0
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

        featureBuffer.reserveCapacity((config.chunkMelFrames + config.coreFrames) * config.melFeatures)
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
        preprocessAudioToFeaturesLocked()
    }

    /// Process buffered audio and return any new results.
    ///
    /// Call this after adding audio with `addAudio()`.
    ///
    /// - Returns: New chunk results if enough audio was processed, nil otherwise
    public func process() throws -> SortformerChunkResult? {
        lock.lock()
        defer { lock.unlock() }

        guard let models = _models else {
            throw SortformerError.notInitialized
        }

        var newPredictions: [Float] = []
        var newTentativePredictions: [Float] = []
        var newFrameCount = 0
        var newTentativeFrameCount = 0

        // Step 1: Run preprocessor on available audio
        while let (chunkFeatures, chunkLengths) = getNextChunkFeaturesLocked() {
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
            let updateResult = try stateUpdater.streamingUpdate(
                state: &_state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: diarizerChunkIndex > 0 ? config.chunkLeftContext : 0,
                rightContext: config.chunkRightContext
            )

            // Accumulate confirmed results
            newPredictions.append(contentsOf: updateResult.confirmed)
            newTentativePredictions = updateResult.tentative
            newFrameCount += updateResult.confirmed.count / config.numSpeakers
            newTentativeFrameCount = updateResult.tentative.count / config.numSpeakers

            diarizerChunkIndex += 1
        }

        // Return new results if any
        if newPredictions.count > 0 {
            let chunk = SortformerChunkResult(
                startFrame: _numFramesProcessed,
                speakerPredictions: newPredictions,
                frameCount: newFrameCount,
                tentativePredictions: newTentativePredictions,
                tentativeFrameCount: newTentativeFrameCount
            )

            _numFramesProcessed += newFrameCount
            _timeline.addChunk(chunk)

            return chunk
        }

        return nil
    }

    /// Process a chunk of audio in one call.
    ///
    /// Convenience method that combines `addAudio()` and `process()`.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    /// - Returns: New chunk results if enough audio was processed
    public func processSamples(_ samples: [Float]) throws -> SortformerChunkResult? {
        lock.lock()
        defer { lock.unlock() }

        audioBuffer.append(contentsOf: samples)
        preprocessAudioToFeaturesLocked()
        return try processLocked()
    }

    /// Internal process - caller must hold lock
    private func processLocked() throws -> SortformerChunkResult? {
        guard let models = _models else {
            throw SortformerError.notInitialized
        }

        var newPredictions: [Float] = []
        var newTentativePredictions: [Float] = []
        var newFrameCount = 0
        var newTentativeFrameCount = 0

        while let (chunkFeatures, chunkLengths) = getNextChunkFeaturesLocked() {
            let output = try models.runMainModel(
                chunk: chunkFeatures,
                chunkLength: chunkLengths,
                spkcache: _state.spkcache,
                spkcacheLength: _state.spkcacheLength,
                fifo: _state.fifo,
                fifoLength: _state.fifoLength,
                config: config
            )

            let probabilities = output.predictions
            let embLength = output.chunkLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.preEncoderDims))

            let updateResult = try stateUpdater.streamingUpdate(
                state: &_state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: diarizerChunkIndex > 0 ? config.chunkLeftContext : 0,
                rightContext: config.chunkRightContext
            )

            newPredictions.append(contentsOf: updateResult.confirmed)
            newTentativePredictions = updateResult.tentative
            newFrameCount += updateResult.confirmed.count / config.numSpeakers
            newTentativeFrameCount = updateResult.tentative.count / config.numSpeakers

            diarizerChunkIndex += 1
        }

        if newPredictions.count > 0 {
            let chunk = SortformerChunkResult(
                startFrame: _numFramesProcessed,
                speakerPredictions: newPredictions,
                frameCount: newFrameCount,
                tentativePredictions: newTentativePredictions,
                tentativeFrameCount: newTentativeFrameCount
            )

            _numFramesProcessed += newFrameCount
            _timeline.addChunk(chunk)

            return chunk
        }

        return nil
    }

    // MARK: - Complete File Processing

    /// Progress callback type: (processedSamples, totalSamples, chunksProcessed)
    public typealias ProgressCallback = (Int, Int, Int) -> Void

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
        lock.lock()
        defer { lock.unlock() }

        guard let models = _models else {
            throw SortformerError.notInitialized
        }

        // Reset for fresh processing
        _state = SortformerStreamingState(config: config)
        lastAudioSample = 0
        resetBuffersLocked()

        var featureProvider = SortformerFeatureLoader(config: self.config, audio: samples)
        var chunksProcessed = 0
        var predictions: [Float] = []
        var lastTentative: [Float] = []

        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48 mel frames core

        while let (chunkFeatures, chunkLength, leftOffset, rightOffset) = featureProvider.next() {
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
            let leftContext = (leftOffset + config.subsamplingFactor / 2) / config.subsamplingFactor
            let rightContext = (rightOffset + config.subsamplingFactor - 1) / config.subsamplingFactor

            // Update state
            let updateResult = try stateUpdater.streamingUpdate(
                state: &_state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: leftContext,
                rightContext: rightContext
            )

            // Accumulate confirmed results (tentative not needed for batch processing)
            predictions.append(contentsOf: updateResult.confirmed)
            lastTentative = updateResult.tentative
            chunksProcessed += 1
            diarizerChunkIndex += 1

            // Progress callback
            // processedFrames is in mel frames (after subsampling)
            // Each mel frame corresponds to melStride samples
            let processedMelFrames = diarizerChunkIndex * coreFrames
            let progress = min(processedMelFrames * config.melStride, samples.count)
            progressCallback?(progress, samples.count, chunksProcessed)
        }
        predictions.append(contentsOf: lastTentative)

        // Save updated state
        _numFramesProcessed = predictions.count / config.numSpeakers

        if config.debugMode {
            print(
                "[DEBUG] Phase 2 complete: diarizerChunks=\(diarizerChunkIndex), totalProbs=\(predictions.count), totalFrames=\(_numFramesProcessed)"
            )
            fflush(stdout)
        }

        _timeline = SortformerTimeline(
            allPredictions: predictions,
            config: _timeline.config,
            isComplete: true
        )

        return _timeline
    }

    // MARK: - Helpers

    /// Preprocess audio into mel features - caller must hold lock
    private func preprocessAudioToFeaturesLocked() {
        guard !audioBuffer.isEmpty else { return }
        if audioBuffer.count < config.melWindow { return }

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
            // We have enough features for the next chunk!
            // Check if we have A LOT of audio buffered (buffer pressure)?
            // If we have > 1 second of audio, maybe process it batch-wise?
            // For now, lazy approach: don't process.
            return
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
        if audioBuffer.count < samplesNeeded { return }

        // We have enough audio! Process exactly what's needed (or slightly more if convenient?)
        // Let's process everything we have, since we paid the initialization cost check.
        // This prevents creating a backlog of unprocessed audio.

        let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(
            audio: audioBuffer,
            lastAudioSample: lastAudioSample
        )

        guard melLength > 0 else { return }

        featureBuffer.append(contentsOf: mel)

        let samplesConsumed = melLength * config.melStride

        if samplesConsumed <= audioBuffer.count {
            lastAudioSample = audioBuffer[samplesConsumed - 1]
            audioBuffer.removeFirst(samplesConsumed)
        } else {
            lastAudioSample = 0
            audioBuffer.removeAll()
        }
    }

    /// Get next chunk features (for testing)
    internal func getNextChunkFeatures() -> (mel: [Float], melLength: Int)? {
        lock.lock()
        defer { lock.unlock() }
        return getNextChunkFeaturesLocked()
    }

    /// Get next chunk features - caller must hold lock
    private func getNextChunkFeaturesLocked() -> (mel: [Float], melLength: Int)? {
        let featLength = featureBuffer.count / config.melFeatures
        let coreFrames = config.chunkLen * config.subsamplingFactor
        let leftContextFrames = config.chunkLeftContext * config.subsamplingFactor
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor

        // Calculate end of core chunk
        let endFeat = min(startFeat + coreFrames, featLength)

        // Need at least one core frame
        guard endFeat > startFeat else { return nil }

        // Ensure we have the full chunk context (Core + RC)
        // This prevents issuing chunks too early with zero right context.
        // Alignment:
        // Chunk 0: startFeat=0. Need 48+56=104 frames. (Returns 104 frames). Matches Batch.
        // Chunk 1: startFeat=8. Need 56+56=112 frames (relative). (Returns 112 frames).
        guard endFeat + rightContextFrames <= featLength else { return nil }

        // Calculate offsets
        let leftOffset = min(leftContextFrames, startFeat)
        // Since we guarded above, we know we have full right context
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
