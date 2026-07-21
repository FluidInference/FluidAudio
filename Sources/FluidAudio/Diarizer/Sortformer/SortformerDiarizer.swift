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
    private var _finalized: Bool = false

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

    // Audio buffering. `audioBuffer` always starts at sample
    // `melFramesEmitted * melStride - nFFT/2` of the session's virtually
    // center-padded stream; it is seeded with the nFFT/2-zero batch left pad
    // at reset.
    private var audioBuffer: [Float] = []
    private var lastAudioSample: Float = 0
    /// Frames of the session's mel stream emitted so far.
    internal private(set) var melFramesEmitted: Int = 0
    /// Set once `padAndEmitRemainingMelLocked` has flushed the final frames.
    private var melInputExhausted: Bool = false
    private var warnedDroppedAudio: Bool = false

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
        resetMelStreamLocked()
    }

    /// Initialize with CoreML models (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - mainModelPath: Path to Sortformer.mlpackage
    ///   - computeUnits: CoreML compute units. Pass `nil` (default) to auto-resolve — large fp16
    ///     high-context variants fall back to `.cpuOnly` on RAM-constrained devices (issue #726).
    public func initialize(
        mainModelPath: URL,
        computeUnits: MLComputeUnits? = nil
    ) async throws {
        logger.info("Initializing Sortformer diarizer (combined pipeline mode)")

        let loadedModels = try await SortformerModels.load(
            config: config,
            mainModelPath: mainModelPath,
            computeUnits: computeUnits
        )

        validateConfigMatch(loadedModels)

        // Use withLock helper to avoid direct NSLock usage in async context
        withLock {
            self._models = loadedModels
            self._state = SortformerStreamingState(config: config)
            self.resetBuffersLocked()
        }
        logger.info("Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s")
    }

    /// Warn loudly if the diarizer's `config` does not match the streaming parameters baked
    /// into the loaded model. A mismatch (e.g. a `.default` config against a `highContextV2_1`
    /// model) runs but produces incorrect and much slower results — issue #726.
    private func validateConfigMatch(_ models: SortformerModels) {
        guard let embedded = models.embeddedConfig else { return }
        let current = SortformerModels.EmbeddedConfig(
            chunkLen: config.chunkLen,
            chunkLeftContext: config.chunkLeftContext,
            chunkRightContext: config.chunkRightContext,
            fifoLen: config.fifoLen,
            spkcacheLen: config.spkcacheLen
        )
        guard current != embedded else { return }
        logger.error(
            """
            Sortformer config mismatch — diarizer config does not match the loaded model. \
            This produces incorrect and much slower diarization (issue #726). \
            diarizer(chunkLen=\(current.chunkLen), leftCtx=\(current.chunkLeftContext), \
            rightCtx=\(current.chunkRightContext), fifoLen=\(current.fifoLen), \
            spkcacheLen=\(current.spkcacheLen)) \
            vs model(chunkLen=\(embedded.chunkLen), leftCtx=\(embedded.chunkLeftContext), \
            rightCtx=\(embedded.chunkRightContext), fifoLen=\(embedded.fifoLen), \
            spkcacheLen=\(embedded.spkcacheLen)). \
            Construct SortformerDiarizer with the SortformerConfig matching the model variant.
            """
        )
    }

    /// Execute a closure while holding the lock
    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// Initialize with pre-loaded models.
    public func initialize(models: SortformerModels) {
        validateConfigMatch(models)

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
        resetMelStreamLocked()
        diarizerChunkIndex = 0
        _numFramesProcessed = 0
        _finalized = false
        _timeline.reset(keepingSpeakers: keepingSpeakers)

        featureBuffer.reserveCapacity((config.chunkMelFrames + config.coreFrames) * config.melFeatures)
    }

    /// Reset the incremental mel stream: an empty feature buffer and an audio
    /// buffer seeded with the batch left pad (nFFT/2 zeros) - caller must
    /// hold lock.
    private func resetMelStreamLocked() {
        featureBuffer.removeAll(keepingCapacity: true)
        startFeat = 0
        lastAudioSample = 0
        melFramesEmitted = 0
        melInputExhausted = false
        warnedDroppedAudio = false
        _realSamplesReceived = 0
        audioBuffer.removeAll(keepingCapacity: true)
        audioBuffer.append(contentsOf: [Float](repeating: 0, count: melSpectrogram.nFFT / 2))
    }

    /// True (after a one-time warning) when the mel stream is exhausted and
    /// incoming audio must be dropped; buffering it would grow without bound
    /// since nothing consumes audio after `finalizeSession` - caller must
    /// hold lock.
    private func shouldDropAudioLocked() -> Bool {
        guard melInputExhausted else { return false }
        if !warnedDroppedAudio {
            warnedDroppedAudio = true
            logger.warning("Audio added after finalizeSession is ignored; call reset() to start a new session")
        }
        return true
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

            // Every exit (guard failures and thrown errors included) must
            // clear the enrollment clip's mel stream: padAndEmitRemainingMel
            // sets the exhausted latch, which would otherwise silently
            // disable all further streaming until reset().
            defer {
                resetMelStreamLocked()
                diarizerChunkIndex = 0
                _numFramesProcessed = 0
            }

            // Clear audio and feature buffers to avoid enrolling this speaker with stale audio.
            resetMelStreamLocked()
            diarizerChunkIndex = 0
            _realSamplesReceived = normalized.count
            audioBuffer.append(contentsOf: normalized)

            // The enrollment clip is a complete utterance: emit its full
            // mel stream, right pad included.
            padAndEmitRemainingMelLocked()

            // Accumulate per-slot speech frames from the diarizer updates rather than
            // from persisted timeline segments, so enrollment works even when the
            // timeline is configured not to store segments.
            var speechFrames: [Int: Int] = [:]
            var lastUpdate: DiarizerTimelineUpdate?
            var didProcess: Bool = false
            while let update = try processLocked(updateTimeline: true) {
                didProcess = true
                for segment in update.finalizedSegments {
                    speechFrames[segment.speakerIndex, default: 0] += segment.length
                }
                lastUpdate = update
            }

            guard didProcess else {
                let minDuration = Float(config.chunkLen + config.chunkRightContext) * config.frameDurationSeconds
                logger.warning(
                    "Failed to enroll speaker \(description): not enough audio was provided. "
                        + "Please provide at least \(String(format: "%.2f", minDuration)) seconds of speech.")
                return nil
            }

            // Include the trailing still-open (tentative) segment from the final update.
            for segment in lastUpdate?.tentativeSegments ?? [] {
                speechFrames[segment.speakerIndex, default: 0] += segment.length
            }

            let bestSlot = speechFrames.max { $0.value < $1.value }?.key
            let enrolledSpeaker: DiarizerSpeaker?
            if let bestSlot, (speechFrames[bestSlot] ?? 0) > 0 {
                // Provide warnings if the diarizer failed to recognize this person as a new speaker
                if let oldName = _timeline.speakers[bestSlot]?.name {
                    guard overwriteAssignedSpeakerName else {
                        logger.warning(
                            "Failed to enroll speaker \(description): diarizer matched existing speaker '\(oldName)' "
                                + "at index \(bestSlot) and overwritingAssignedSpeakerName=false"
                        )
                        _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })
                        return nil
                    }
                    logger.warning(
                        "Newly-enrolled speaker \(description) will overwrite the old one named \(oldName) at index \(bestSlot)"
                    )
                }
                // Register the enrolled speaker at the chosen slot regardless of whether
                // segments are persisted; the slot came from the updates above.
                enrolledSpeaker = _timeline.upsertSpeaker(named: name, atIndex: bestSlot)
                if enrolledSpeaker != nil {
                    occupiedIndices.insert(bestSlot)
                }
            } else {
                logger.warning("Failed to enroll speaker \(description) because no speech detected")
                enrolledSpeaker = nil
            }

            _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })

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
            guard !shouldDropAudioLocked() else { return }
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
            guard !shouldDropAudioLocked() else { return }
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
            guard !shouldDropAudioLocked() else { return }
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
            guard !shouldDropAudioLocked() else { return nil }
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
            // Cooperative cancellation: throws `CancellationError` if the enclosing Swift
            // `Task` was cancelled, before the next (expensive) inference call.
            try Task.checkCancellation()

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
    /// Drains any remaining full-right-context chunks from the audio buffer
    /// (matching offline mode's `SortformerFeatureLoader` gate), then absorbs
    /// the trailing tentative predictions into the timeline as finalized —
    /// equivalent to offline `rebuild(... isComplete: true)`.
    ///
    /// Idempotent: subsequent calls return `nil`.
    ///
    /// - Returns: A single `DiarizerTimelineUpdate` covering all new finalized
    ///   and tentative segments produced by this call, or `nil` if there were
    ///   no new predictions.
    @discardableResult
    public func finalizeSession() throws -> DiarizerTimelineUpdate? {
        return try lock.withLock {
            guard _models != nil else {
                throw SortformerError.notInitialized
            }
            guard !_finalized else { return nil }

            // Emit the trailing mel frames (whose windows overlap the batch
            // right pad), then drain every remaining full-rc chunk.
            var aggFinalized: [Float] = []
            var aggTentative: [Float] = []
            var didDrain = false
            padAndEmitRemainingMelLocked()
            while let chunk = try makeStreamingChunkLocked() {
                aggFinalized.append(contentsOf: chunk.finalizedPredictions)
                aggTentative = chunk.tentativePredictions
                _numFramesProcessed += chunk.finalizedFrameCount
                didDrain = true
            }

            // Absorb trailing tentative as finalized — mirrors offline
            // rebuild(isComplete: true). If we drained, the last drained chunk's
            // tentative is the trailing region. If we didn't drain, the
            // previously-stored tentative on the timeline is the trailing region.
            let absorbedTentative: [Float]
            if didDrain {
                absorbedTentative = aggTentative
            } else {
                absorbedTentative = _timeline.tentativePredictions
            }
            aggFinalized.append(contentsOf: absorbedTentative)
            _numFramesProcessed += absorbedTentative.count / config.numSpeakers

            let update: DiarizerTimelineUpdate?
            if !aggFinalized.isEmpty {
                update = try _timeline.addPredictions(
                    finalizedPredictions: aggFinalized,
                    tentativePredictions: []
                )
            } else {
                update = nil
            }

            _timeline.finalize()
            _finalized = true
            return update
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
    ///   - progressCallback: Optional callback for progress updates `(processedSamples, totalSamples, chunksProcessed)`.
    /// - Returns: Complete diarization timeline
    /// - Throws: `CancellationError` if the enclosing Swift `Task` is cancelled mid-processing.
    ///   Run this inside a `Task` and call `cancel()` on it to stop early.
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
            let keepSpeakers = keepSpeakers ?? (_realSamplesReceived == 0 && diarizerChunkIndex == 0)
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
                // Cooperative cancellation: when `processComplete` runs inside a Swift `Task`,
                // cancelling that task throws `CancellationError` here before the next inference.
                try Task.checkCancellation()

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

    /// Preprocess audio into mel features - caller must hold lock.
    ///
    /// Emits every frame of the session's batch-equivalent (center-padded)
    /// mel stream whose window is already covered by received samples, so
    /// frame count and values do not depend on how the audio was batched
    /// into `addAudio` calls. Frame k spans samples
    /// `[k * melStride - melWindow/2, k * melStride + melWindow/2)`; the
    /// trailing frames whose windows overlap the batch right pad are emitted
    /// by `padAndEmitRemainingMelLocked` at session end.
    private func preprocessAudioToFeaturesLocked() {
        guard !melInputExhausted else { return }
        let halfWindow = config.melWindow / 2
        guard _realSamplesReceived >= halfWindow else { return }
        let computableFrames = (_realSamplesReceived - halfWindow) / config.melStride + 1
        let count = computableFrames - melFramesEmitted
        guard count > 0 else { return }
        emitMelFramesLocked(count)
    }

    /// Compute the next `count` frames of the mel stream from `audioBuffer`
    /// and append them to `featureBuffer`, consuming `count * melStride`
    /// samples - caller must hold lock and guarantee the buffer covers the
    /// frames' windows.
    private func emitMelFramesLocked(_ count: Int) {
        let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(
            audio: audioBuffer,
            lastAudioSample: lastAudioSample,
            paddingMode: .prePadded,
            expectedFrameCount: count
        )
        assert(melLength == count && mel.count == count * config.melFeatures)
        featureBuffer.append(contentsOf: mel)

        let samplesConsumed = count * config.melStride
        lastAudioSample = audioBuffer[samplesConsumed - 1]
        audioBuffer.removeFirst(samplesConsumed)
        melFramesEmitted += count
    }

    /// Emit the remaining frames of the mel stream by appending the batch
    /// right pad, reaching the exact frame count batch (`.center`) mode
    /// produces for the complete session audio - caller must hold lock.
    /// Idempotent; further audio is ignored once called.
    private func padAndEmitRemainingMelLocked() {
        guard !melInputExhausted else { return }
        melInputExhausted = true
        guard _realSamplesReceived > 0 else { return }

        let totalFrames =
            1 + (_realSamplesReceived + melSpectrogram.nFFT - config.melWindow) / config.melStride
        let remaining = totalFrames - melFramesEmitted
        guard remaining > 0 else { return }

        // The pad is a preemphasis-cancelling decay rather than literal
        // zeros: the in-place preemphasis filter turns it into the exact
        // zeros of the batch right pad.
        let padLength = melSpectrogram.nFFT / 2
        var tail = [Float](repeating: 0, count: padLength)
        if melSpectrogram.preemph != 0, let lastReal = audioBuffer.last {
            var value = lastReal
            for i in 0..<padLength {
                value *= melSpectrogram.preemph
                tail[i] = value
            }
        }
        audioBuffer.append(contentsOf: tail)

        emitMelFramesLocked(remaining)
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

    /// Get next chunk features (for testing)
    internal func getNextChunkFeatures() -> (mel: [Float], melLength: Int)? {
        lock.lock()
        defer { lock.unlock() }
        return getNextChunkFeaturesLocked()
    }

    /// Emit the trailing mel frames as `finalizeSession` would, without
    /// requiring loaded models (for testing).
    internal func padAndEmitRemainingMel() {
        lock.lock()
        defer { lock.unlock() }
        padAndEmitRemainingMelLocked()
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
