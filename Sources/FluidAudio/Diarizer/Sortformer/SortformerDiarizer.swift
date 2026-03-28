import Accelerate
import CoreML
import Foundation
import OSLog

/// Streaming speaker diarization using NVIDIA's Sortformer model.
///
/// Sortformer provides end-to-end streaming diarization with 4 fixed speaker slots,
/// achieving ~11% DER on DI-HARD III in real-time.
///
/// - Important: This class is **not** thread-safe.
public final class SortformerDiarizer: Diarizer {
    /// Lock for thread-safe access to mutable state
    private let lock = NSLock()

    /// Accumulated results
    public var timeline: DiarizerTimeline {
        lock.lock()
        defer { lock.unlock() }
        return _timeline
    }

    private var _timeline: DiarizerTimeline

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
    private var _realSamplesReceived: Int = 0

    /// Configuration
    public let config: SortformerConfig

    // MARK: - Diarizer Protocol Properties

    /// Model's target sample rate in Hz
    public var targetSampleRate: Int? { config.sampleRate }

    /// Output frame rate in Hz
    public var modelFrameHz: Double? { 1.0 / Double(config.frameDurationSeconds) }

    /// Number of speaker output tracks
    public var numSpeakers: Int? { config.numSpeakers }

    private let logger = AppLogger(category: "SortformerDiarizerPipeline")
    private let stateUpdater: SortformerStateUpdater

    private var _models: SortformerModels?

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private let melSpectrogram = AudioMelSpectrogram()

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
        timelineConfig: DiarizerTimelineConfig = .sortformerDefault
    ) {
        var timelineConfig = timelineConfig
        timelineConfig.numSpeakers = config.numSpeakers
        self.config = config
        self.stateUpdater = SortformerStateUpdater(config: config)
        self._state = SortformerStreamingState(config: config)
        self._timeline = DiarizerTimeline(config: timelineConfig)
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
        resetBuffersLocked()
        logger.info("Sortformer initialized with pre-loaded models")
    }

    /// Reset all internal state for a new audio stream.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        _state = SortformerStreamingState(config: config)
        resetBuffersLocked()
        logger.debug("Sortformer state reset")
    }

    /// Internal reset - caller must hold lock
    private func resetBuffersLocked(keepingSpeakers: Bool = false) {
        audioBuffer = []
        featureBuffer = []
        lastAudioSample = 0
        startFeat = 0
        diarizerChunkIndex = 0
        _numFramesProcessed = 0
        _realSamplesReceived = 0
        _timeline.reset(keepingSpeakers: keepingSpeakers)

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

    // MARK: - Speaker Priming

    /// Prime the diarizer with enrollment audio to warm up speaker state.
    ///
    /// Processes the audio through the full pipeline to populate the speaker cache
    /// and FIFO buffers, then resets the timeline so subsequent processing starts
    /// from frame 0. Call this after `initialize()` and before streaming real audio.
    ///
    /// - Parameters:
    ///   - samples: Audio samples (16kHz mono) of known speakers
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    ///   - name: The speaker's name.
    ///   - overwriteAssignedSpeakerName: Whether enrollment may overwrite the name on an already-named slot
    ///     if the diarizer assigns the audio to that speaker.
    /// - Throws: `SortformerError.notInitialized` if models not loaded
    public func enrollSpeaker(
        withAudio samples: [Float],
        sourceSampleRate: Double? = nil,
        named name: String? = nil,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool = true
    ) throws -> DiarizerSpeaker? {
        try enrollSpeakerInternal(
            withAudio: samples,
            sourceSampleRate: sourceSampleRate,
            named: name,
            overwritingAssignedSpeakerName: overwriteAssignedSpeakerName
        )
    }

    /// Prime the diarizer with enrollment audio to warm up speaker state.
    ///
    /// - Parameters:
    ///   - samples: Audio samples of known speakers.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    ///   - name: The speaker's name.
    ///   - overwriteAssignedSpeakerName: Whether enrollment may overwrite the name on an already-named slot
    ///     if the diarizer assigns the audio to that speaker.
    public func enrollSpeaker<C: Collection>(
        withAudio samples: C,
        sourceSampleRate: Double? = nil,
        named name: String? = nil,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool = true
    ) throws -> DiarizerSpeaker? where C.Element == Float {
        try enrollSpeakerInternal(
            withAudio: Array(samples),
            sourceSampleRate: sourceSampleRate,
            named: name,
            overwritingAssignedSpeakerName: overwriteAssignedSpeakerName
        )
    }

    private func enrollSpeakerInternal(
        withAudio samples: [Float],
        sourceSampleRate: Double?,
        named name: String?,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool
    ) throws -> DiarizerSpeaker? {
        var description: String {
            guard let name else { return "(no name)" }
            return "named '\(name)'"
        }

        guard !samples.isEmpty else {
            logger.warning("Failed to enroll speaker \(description) because no speech detected")
            return nil
        }
        let normalized = try normalizeSamples(samples, sourceSampleRate: sourceSampleRate)

        return try lock.withLock {
            guard _models != nil else {
                throw SortformerError.notInitialized
            }

            if _timeline.hasSegments {
                logger.warning("Trying to enroll a speaker while timeline has segments; timeline will be reset")
            }

            _timeline.reset(keepingSpeakers: true)
            var occupiedIndices = Set(_timeline.speakers.keys)

            // Clear audio and feature buffers to avoid enrolling this speaker with stale audio.
            startFeat = 0
            lastAudioSample = 0
            diarizerChunkIndex = 0
            audioBuffer.removeAll(keepingCapacity: true)
            featureBuffer.removeAll(keepingCapacity: true)
            _realSamplesReceived = 0
            audioBuffer.append(contentsOf: normalized)

            preprocessAudioToFeaturesLocked()

            var didProcess: Bool = false
            while let _ = try processLocked(updateTimeline: true) { didProcess = true }

            guard didProcess else {
                let minDuration = Float(config.chunkLen + config.chunkRightContext) * config.frameDurationSeconds
                logger.warning(
                    "Failed to enroll speaker \(description): not enough audio was provided. "
                        + "Please provide at least \(String(format: "%.2f", minDuration)) seconds of speech.")
                return nil
            }

            let speaker = _timeline.speakers.values.max { $0.numSpeechFrames < $1.numSpeechFrames }
            let enrolledSpeaker: DiarizerSpeaker?
            if let speaker, speaker.hasSegments {
                // Provide warnings if the diarizer failed to recognize this person as a new speaker
                if let oldName = speaker.name {
                    guard overwriteAssignedSpeakerName else {
                        logger.warning(
                            "Failed to enroll speaker \(description): diarizer matched existing speaker '\(oldName)' "
                                + "at index \(speaker.index) and overwritingAssignedSpeakerName=false"
                        )
                        _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })
                        _numFramesProcessed = 0
                        diarizerChunkIndex = 0
                        startFeat = 0
                        lastAudioSample = 0
                        audioBuffer.removeAll(keepingCapacity: true)
                        featureBuffer.removeAll(keepingCapacity: true)
                        _realSamplesReceived = 0
                        return nil
                    }
                    logger.warning(
                        "Newly-enrolled speaker \(description) will overwrite the old one named \(oldName) at index \(speaker.index)"
                    )
                }
                speaker.name = name
                occupiedIndices.insert(speaker.index)
                enrolledSpeaker = speaker
            } else {
                logger.warning("Failed to enroll speaker \(description) because no speech detected")
                enrolledSpeaker = nil
            }

            _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })
            _numFramesProcessed = 0
            diarizerChunkIndex = 0
            startFeat = 0
            lastAudioSample = 0
            audioBuffer.removeAll(keepingCapacity: true)
            featureBuffer.removeAll(keepingCapacity: true)
            _realSamplesReceived = 0

            logger.info(
                "Enrolled speaker \(description) with \(normalized.count) samples "
                    + "(\(String(format: "%.1f", Float(normalized.count) / Float(config.sampleRate)))s), "
                    + "spkcache=\(_state.spkcacheLength), fifo=\(_state.fifoLength)"
            )

            return enrolledSpeaker
        }
    }

    // MARK: - Streaming Processing

    /// Add audio samples to the processing buffer (protocol conformance).
    ///
    /// - Parameters:
    ///   - samples: Audio samples (16kHz mono)
    ///   - sourceSampleRate: Source audio sample rate
    public func addAudio(_ samples: [Float]) {
        lock.withLock {
            audioBuffer.append(contentsOf: samples)
            _realSamplesReceived += samples.count
            preprocessAudioToFeaturesLocked()
        }
    }

    /// Add audio samples to the processing buffer, resampling when needed.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    public func addAudio(
        _ samples: [Float],
        sourceSampleRate: Double? = nil
    ) throws {
        let normalized = try normalizeSamples(samples, sourceSampleRate: sourceSampleRate)
        lock.withLock {
            audioBuffer.append(contentsOf: normalized)
            _realSamplesReceived += normalized.count
            preprocessAudioToFeaturesLocked()
        }
    }

    /// Add audio samples to the processing buffer (generic variant).
    ///
    /// - Parameters:
    ///   - samples: Audio samples (16kHz mono)
    ///   - sourceSampleRate: Source audio sample rate
    public func addAudio<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double? = nil
    ) throws where C.Element == Float {
        try lock.withLock {
            if let sourceSampleRate, sourceSampleRate != Double(config.sampleRate) {
                let normalized = try normalizeSamples(Array(samples), sourceSampleRate: sourceSampleRate)
                audioBuffer.append(contentsOf: normalized)
                _realSamplesReceived += normalized.count
            } else {
                audioBuffer.append(contentsOf: samples)
                _realSamplesReceived += samples.count
            }
            preprocessAudioToFeaturesLocked()
        }
    }

    /// Process buffered audio and return any new results.
    ///
    /// Call this after adding audio with `addAudio()`.
    /// - Returns: New chunk results if enough audio was processed, nil otherwise
    @discardableResult
    public func process() throws -> DiarizerTimelineUpdate? {
        try lock.withLock {
            return try processLocked()
        }
    }

    /// Add and process a chunk of audio in one call.
    /// - Parameters:
    ///   - samples: Audio samples (16kHz mono)
    ///   - sourceSampleRate: Source audio sample rate
    /// - Returns: New chunk results if enough audio was processed
    public func process<C: Collection>(
        samples: C,
        sourceSampleRate: Double? = nil
    ) throws -> DiarizerTimelineUpdate?
    where C.Element == Float {
        return try lock.withLock {
            if let sourceSampleRate, sourceSampleRate != Double(config.sampleRate) {
                let normalized = try normalizeSamples(Array(samples), sourceSampleRate: sourceSampleRate)
                audioBuffer.append(contentsOf: normalized)
                _realSamplesReceived += normalized.count
            } else {
                audioBuffer.append(contentsOf: samples)
                _realSamplesReceived += samples.count
            }
            preprocessAudioToFeaturesLocked()
            return try processLocked()
        }
    }

    /// Internal process - caller must hold lock
    private func processLocked(updateTimeline: Bool = true) throws -> DiarizerTimelineUpdate? {
        guard let chunk = try makeStreamingChunkLocked() else {
            return nil
        }

        _numFramesProcessed += chunk.finalizedFrameCount
        guard updateTimeline else { return nil }
        return try _timeline.addChunk(chunk)
    }

    private func makeStreamingChunkLocked() throws -> DiarizerChunkResult? {
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
            return DiarizerChunkResult(
                startFrame: _numFramesProcessed,
                finalizedPredictions: newPredictions,
                finalizedFrameCount: newFrameCount,
                tentativePredictions: newTentativePredictions,
                tentativeFrameCount: newTentativeFrameCount
            )
        }

        return nil
    }

    /// Finalize the current streaming session.
    ///
    /// Pads the tail with silence until the last true frame has been emitted as
    /// finalized output, then finalizes the timeline.
    ///
    /// - Returns: The last finalized chunk emitted during finalization, if any.
    @discardableResult
    public func finalizeSession() throws -> DiarizerChunkResult? {
        return try lock.withLock {
            guard _models != nil else {
                throw SortformerError.notInitialized
            }

            var lastResult: DiarizerChunkResult?
            var tentativeToFlush = _timeline.numTentativeFrames
            if let chunk = try makeStreamingChunkLocked() {
                tentativeToFlush = chunk.tentativeFrameCount
                let finalizedChunk = DiarizerChunkResult(
                    startFrame: chunk.startFrame,
                    finalizedPredictions: chunk.finalizedPredictions,
                    finalizedFrameCount: chunk.finalizedFrameCount,
                    tentativePredictions: [],
                    tentativeFrameCount: 0
                )
                _numFramesProcessed = finalizedChunk.startFrame + finalizedChunk.finalizedFrameCount
                try _timeline.addChunk(finalizedChunk)
                lastResult = finalizedChunk
            }

            let targetEndFrame = _numFramesProcessed + min(tentativeToFlush, config.chunkLen)
            let exactPaddingSamples = try exactFinalizationPaddingSamples(targetEndFrame: targetEndFrame)
            if exactPaddingSamples > 0 {
                audioBuffer.append(contentsOf: [Float](repeating: 0, count: exactPaddingSamples))
            }

            while _numFramesProcessed < targetEndFrame {
                let remainingFrames = targetEndFrame - _numFramesProcessed
                let targetFeatureFrames =
                    startFeat
                    + min(remainingFrames * config.subsamplingFactor, config.coreFrames)
                    + config.chunkRightContext * config.subsamplingFactor
                preprocessAudioToFeatureTargetLocked(targetFeatureFrames: targetFeatureFrames)
                guard let chunk = try makeStreamingChunkLocked(), chunk.finalizedFrameCount > 0 else {
                    logger.warning(
                        "Sortformer finalize could not emit enough confirmed frames "
                            + "(\(_numFramesProcessed) / \(targetEndFrame) frames)"
                    )
                    break
                }

                let finalizedFrameCount = min(remainingFrames, chunk.finalizedFrameCount)
                guard finalizedFrameCount > 0 else {
                    break
                }

                let finalizedPredictions = Array(
                    chunk.finalizedPredictions.prefix(finalizedFrameCount * config.numSpeakers)
                )
                let finalizedResult = DiarizerChunkResult(
                    startFrame: _numFramesProcessed,
                    finalizedPredictions: finalizedPredictions,
                    finalizedFrameCount: finalizedFrameCount,
                    tentativePredictions: [],
                    tentativeFrameCount: 0
                )
                _numFramesProcessed += finalizedFrameCount
                try _timeline.addChunk(finalizedResult)
                lastResult = finalizedResult
            }

            _timeline.finalize()
            return lastResult
        }
    }

    // MARK: - Complete File Processing

    /// Progress callback type: (processedSamples, totalSamples, chunksProcessed)
    public typealias ProgressCallback = (Int, Int, Int) -> Void

    /// Process complete audio file.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples (16kHz mono)
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it will keep the speakers if no audio has been added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline after completing the processing
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Complete diarization timeline
    public func processComplete(
        _ samples: [Float],
        sourceSampleRate: Double? = nil,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool = true,
        progressCallback: ProgressCallback? = nil
    ) throws -> DiarizerTimeline {
        try processCompleteInternal(
            samples,
            sourceSampleRate: sourceSampleRate,
            keepingEnrolledSpeakers: keepSpeakers,
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
    }

    /// Process a complete audio buffer and return the resulting timeline.
    ///
    /// - Parameters:
    ///   - samples: Complete mono audio buffer.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it will keep the speakers if no audio has been added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline before returning it.
    ///   - progressCallback: Optional callback receiving `(processedSamples, totalSamples, chunksProcessed)`.
    /// - Returns: The diarization timeline for the provided audio.
    public func processComplete<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double? = nil,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool = true,
        progressCallback: ProgressCallback? = nil
    ) throws -> DiarizerTimeline
    where C.Element == Float {
        try processCompleteInternal(
            Array(samples),
            sourceSampleRate: sourceSampleRate,
            keepingEnrolledSpeakers: keepSpeakers,
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
    }

    /// Process a complete audio file from a URL.
    ///
    /// Reads and resamples the file to ``targetSampleRate``, then delegates to
    /// ``processComplete(_:finalizeOnCompletion:progressCallback:)``.
    ///
    /// - Parameters:
    ///   - audioFileURL: Path to a WAV, CAF, or other audio file.
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers.
    ///   - finalizeOnCompletion: Whether to finalize the timeline after processing
    ///   - progressCallback: Optional callback (processedSamples, totalSamples, chunksProcessed).
    /// - Returns: Finalized timeline with segments.
    public func processComplete(
        audioFileURL: URL,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool = true,
        progressCallback: ((Int, Int, Int) -> Void)? = nil
    ) throws -> DiarizerTimeline {
        let converter = AudioConverter(sampleRate: Double(config.sampleRate))
        let audio = try converter.resampleAudioFile(audioFileURL)

        return try processCompleteInternal(
            audio,
            sourceSampleRate: nil,
            keepingEnrolledSpeakers: keepSpeakers,
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
    }

    private func processCompleteInternal(
        _ samples: [Float],
        sourceSampleRate: Double?,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ProgressCallback?
    ) throws -> DiarizerTimeline {
        let normalized = try normalizeSamples(samples, sourceSampleRate: sourceSampleRate)

        return try lock.withLock {
            guard let models = _models else {
                throw SortformerError.notInitialized
            }

            // Reset for fresh processing
            let keepSpeakers = keepSpeakers ?? (featureBuffer.isEmpty && audioBuffer.isEmpty && diarizerChunkIndex == 0)
            if !keepSpeakers {
                _state = SortformerStreamingState(config: config)
            }
            resetBuffersLocked(keepingSpeakers: keepSpeakers)

            var featureProvider = SortformerFeatureLoader(config: self.config, audio: normalized)

            var chunksProcessed = 0

            var finalizedPredictions: [Float] = []
            var tentativePredictions: [Float] = []

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
                finalizedPredictions.append(contentsOf: updateResult.confirmed)
                tentativePredictions = updateResult.tentative

                chunksProcessed += 1
                diarizerChunkIndex += 1

                // Progress callback
                // processedFrames is in mel frames (after subsampling)
                // Each mel frame corresponds to melStride samples
                let processedMelFrames = diarizerChunkIndex * coreFrames
                let progress = min(processedMelFrames * config.melStride, normalized.count)
                progressCallback?(progress, normalized.count, chunksProcessed)
            }

            // Save updated state
            let numPredictions = finalizedPredictions.count + tentativePredictions.count
            _numFramesProcessed = numPredictions / config.numSpeakers

            if config.debugMode {
                print(
                    "[DEBUG] Phase 2 complete: diarizerChunks=\(diarizerChunkIndex), totalProbs=\(numPredictions), totalFrames=\(_numFramesProcessed)"
                )
                fflush(stdout)
            }

            try _timeline.rebuild(
                finalizedPredictions: finalizedPredictions,
                tentativePredictions: tentativePredictions,
                keepingSpeakers: keepSpeakers,
                isComplete: finalizeOnCompletion
            )

            return _timeline
        }
    }

    // MARK: - Helpers

    /// Preprocess audio into mel features - caller must hold lock
    private func preprocessAudioToFeaturesLocked() {
        let targetFeatureFrames = startFeat + config.coreFrames + config.chunkRightContext * config.subsamplingFactor
        preprocessAudioToFeatureTargetLocked(targetFeatureFrames: targetFeatureFrames)
    }

    private func preprocessAudioToFeatureTargetLocked(targetFeatureFrames: Int) {
        guard !audioBuffer.isEmpty else { return }
        if audioBuffer.count < config.melWindow { return }

        let featLength = featureBuffer.count / config.melFeatures
        let framesNeeded = targetFeatureFrames - featLength
        guard framesNeeded > 0 else { return }

        let samplesNeeded: Int
        if featureBuffer.isEmpty {
            samplesNeeded = (framesNeeded - 1) * config.melStride + config.melWindow
        } else {
            samplesNeeded = framesNeeded * config.melStride
        }

        guard audioBuffer.count >= samplesNeeded else { return }

        let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(
            audio: audioBuffer,
            lastAudioSample: lastAudioSample
        )

        guard melLength > 0 else { return }

        featureBuffer.append(contentsOf: mel)

        // Invert the center-padded frame count formula to compute samples consumed.
        // This ensures samplesConsumed ≤ audioBuffer.count, preserving leftover samples
        // and maintaining preemphasis continuity across streaming chunks.
        let samplesConsumed = (melLength - 1) * config.melStride + config.melWindow - melSpectrogram.nFFT

        if samplesConsumed <= audioBuffer.count {
            lastAudioSample = audioBuffer[samplesConsumed - 1]
            audioBuffer.removeFirst(samplesConsumed)
        } else {
            lastAudioSample = 0
            audioBuffer.removeAll()
        }
    }

    private func normalizeSamples(
        _ samples: [Float],
        sourceSampleRate: Double?
    ) throws -> [Float] {
        guard let sourceSampleRate,
            sourceSampleRate != Double(config.sampleRate)
        else {
            return samples
        }

        return try AudioConverter(sampleRate: Double(config.sampleRate))
            .resample(samples, from: sourceSampleRate)
    }

    private func exactFinalizationPaddingSamples(targetEndFrame: Int) throws -> Int {
        let remainingFrames = max(0, targetEndFrame - _numFramesProcessed)
        let (remainingFeatureFrames, remainingOverflow) = remainingFrames.multipliedReportingOverflow(
            by: config.subsamplingFactor
        )
        guard !remainingOverflow else {
            throw SortformerError.invalidState(
                "Finalization remaining-frame expansion overflowed for \(remainingFrames) frames."
            )
        }

        let (rightContextFeatureFrames, rightContextOverflow) = config.chunkRightContext.multipliedReportingOverflow(
            by: config.subsamplingFactor
        )
        guard !rightContextOverflow else {
            throw SortformerError.invalidState(
                "Finalization right-context expansion overflowed for \(config.chunkRightContext) frames."
            )
        }

        let (targetWithoutContext, startOverflow) = startFeat.addingReportingOverflow(remainingFeatureFrames)
        guard !startOverflow else {
            throw SortformerError.invalidState(
                "Finalization target feature frame calculation overflowed at startFeat=\(startFeat)."
            )
        }

        let (targetFeatureFrames, contextOverflow) = targetWithoutContext.addingReportingOverflow(
            rightContextFeatureFrames)
        guard !contextOverflow else {
            throw SortformerError.invalidState(
                "Finalization target feature frame calculation overflowed after adding right context."
            )
        }

        let currentFeatureFrames = featureBuffer.count / config.melFeatures
        let additionalFeatureFramesNeeded = max(0, targetFeatureFrames - currentFeatureFrames)
        guard additionalFeatureFramesNeeded > 0 else {
            return 0
        }

        let framesAvailableWithoutPadding = producedMelFramesAvailable()
        guard additionalFeatureFramesNeeded > framesAvailableWithoutPadding else {
            return 0
        }

        let requiredBufferedSamples: Int
        if featureBuffer.isEmpty {
            let (additionalSamples, overflow) = max(0, additionalFeatureFramesNeeded - 1).multipliedReportingOverflow(
                by: config.melStride
            )
            guard !overflow else {
                throw SortformerError.invalidState(
                    "Finalization sample requirement overflowed for \(additionalFeatureFramesNeeded) feature frames."
                )
            }
            let (samples, windowOverflow) = additionalSamples.addingReportingOverflow(config.melWindow)
            guard !windowOverflow else {
                throw SortformerError.invalidState(
                    "Finalization sample requirement overflowed after adding melWindow."
                )
            }
            requiredBufferedSamples = samples
        } else {
            let (samples, overflow) = additionalFeatureFramesNeeded.multipliedReportingOverflow(by: config.melStride)
            guard !overflow else {
                throw SortformerError.invalidState(
                    "Finalization sample requirement overflowed for \(additionalFeatureFramesNeeded) buffered frames."
                )
            }
            requiredBufferedSamples = max(config.melWindow, samples)
        }

        return max(0, requiredBufferedSamples - audioBuffer.count)
    }

    private func producedMelFramesAvailable() -> Int {
        guard audioBuffer.count >= config.melWindow else {
            return 0
        }
        // Use center-padded frame count formula matching AudioMelSpectrogram.computeFlatTransposed
        let paddedCount = audioBuffer.count + melSpectrogram.nFFT
        return 1 + (paddedCount - config.melWindow) / config.melStride
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
