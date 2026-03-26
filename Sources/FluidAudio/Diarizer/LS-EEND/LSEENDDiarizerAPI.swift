import AVFoundation
import CoreML
import Foundation

/// Speaker diarization using LS-EEND (Linear Streaming End-to-End Neural Diarization).
///
/// Supports both streaming and offline processing, matching the `SortformerDiarizer` API
/// - Important: This class is **not** thread-safe.
public final class LSEENDDiarizer: Diarizer {
    private let lock = NSLock()
    private let logger = AppLogger(category: "LSEENDDiarizer")

    // MARK: - Diarizer Protocol Properties

    /// Accumulated results
    public var timeline: DiarizerTimeline {
        lock.withLock { return _timeline }
    }

    /// Whether the processor is ready for processing
    public var isAvailable: Bool {
        lock.withLock { return _engine != nil }
    }

    /// Number of confirmed frames processed so far
    public var numFramesProcessed: Int {
        lock.withLock { return _numFramesProcessed }
    }

    /// Model's target sample rate in Hz (e.g., 8000)
    public var targetSampleRate: Int? {
        lock.withLock { return _engine?.targetSampleRate }
    }

    /// Output frame rate in Hz (e.g., 10.0)
    public var modelFrameHz: Double? {
        lock.withLock { return _engine?.modelFrameHz }
    }

    /// Number of real speaker tracks (excluding boundary tracks)
    public var numSpeakers: Int? {
        lock.withLock { return _engine?.metadata.realOutputDim }
    }

    // MARK: - Additional Properties

    /// Compute units for CoreML inference
    public let computeUnits: MLComputeUnits

    /// Post-processing configuration
    public var timelineConfig: DiarizerTimelineConfig {
        lock.withLock { return _timeline.config }
    }

    /// Streaming latency in seconds
    public var streamingLatencySeconds: Double? {
        lock.withLock { return _engine?.streamingLatencySeconds }
    }

    /// Total speaker slots in model output (including boundary tracks)
    public var decodeMaxSpeakers: Int? {
        lock.withLock { return _engine?.decodeMaxSpeakers }
    }

    /// Whether a streaming session is currently active.
    var hasActiveSession: Bool {
        lock.withLock { return _session != nil }
    }

    // MARK: - Private State

    private var _engine: LSEENDInferenceHelper?
    private var _session: LSEENDStreamingSession?
    private var _melSpectrogram: AudioMelSpectrogram?
    private var _timeline: DiarizerTimeline
    private var _numFramesProcessed: Int = 0
    private var _timelineConfig: DiarizerTimelineConfig
    private var _visibleStartFrameOffset: Int = 0

    // Audio buffering
    private var pendingAudio: [Float] = []

    // MARK: - Init

    /// Create a processor with default settings.
    ///
    /// Call `initialize(descriptor:)` before processing audio.
    ///
    /// - Parameters:
    ///   - computeUnits: CoreML compute units (default: `.cpuOnly`)
    ///   - onsetThreshold: Onset threshold for segment detection
    ///   - offsetThreshold: Offset threshold for segment detection
    ///   - onsetPadFrames: Padding frames added before each speech segment
    ///   - offsetPadFrames: Padding frames added after each speech segment
    ///   - minFramesOn: Minimum segment length in frames (shorter segments are discarded)
    ///   - minFramesOff: Minimum gap length in frames (shorter gaps are closed)
    ///   - maxStoredFrames: Maximum number of finalized prediction frames to retain (`nil` = unlimited)
    public init(
        computeUnits: MLComputeUnits = .cpuOnly,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.computeUnits = computeUnits
        // Placeholder timeline until model is loaded and numSpeakers/frameHz are known
        self._timelineConfig = .init(
            numSpeakers: 1,
            frameDurationSeconds: 0.1,
            onsetThreshold: onsetThreshold,
            offsetThreshold: offsetThreshold,
            onsetPadFrames: onsetPadFrames,
            offsetPadFrames: offsetPadFrames,
            minFramesOn: minFramesOn,
            minFramesOff: minFramesOff,
            maxStoredFrames: maxStoredFrames
        )
        self._timeline = DiarizerTimeline(config: _timelineConfig)
    }

    /// Create a processor with default settings.
    ///
    /// Call `initialize(descriptor:)` before processing audio.
    ///
    /// - Parameters:
    ///   - computeUnits: CoreML compute units (default: `.cpuOnly`)
    ///   - onsetThreshold: Onset threshold for segment detection
    ///   - offsetThreshold: Offset threshold for segment detection
    public init(
        computeUnits: MLComputeUnits = .cpuOnly,
        timelineConfig: DiarizerTimelineConfig
    ) {
        self.computeUnits = computeUnits
        self._timelineConfig = timelineConfig
        // Placeholder timeline until model is loaded and numSpeakers/frameHz are known
        self._timeline = DiarizerTimeline(config: timelineConfig)
    }

    // MARK: - Initialization

    /// Initialize with a model descriptor. Loads the CoreML model.
    ///
    /// - Parameter descriptor: Model descriptor specifying variant and file paths
    public func initialize(variant: LSEENDVariant = .dihard3) async throws {
        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        try initialize(descriptor: descriptor)
    }

    /// Initialize with a model descriptor. Loads the CoreML model.
    ///
    /// - Parameter descriptor: Model descriptor specifying variant and file paths
    public func initialize(descriptor: LSEENDModelDescriptor) throws {
        let engine = try LSEENDInferenceHelper(descriptor: descriptor, computeUnits: computeUnits)
        let melSpectrogram = Self.createMelSpectrogram(featureConfig: engine.featureConfig)

        lock.withLock {
            updateTimelineConfig(engine: engine)
            _engine = engine
            _melSpectrogram = melSpectrogram
            _timeline = DiarizerTimeline(config: _timelineConfig)
            _session = nil
            resetBuffersLocked()

            logger.info(
                "Initialized LS-EEND \(descriptor.variant.rawValue): "
                    + "\(engine.metadata.realOutputDim) speakers, "
                    + "\(String(format: "%.1f", engine.modelFrameHz)) Hz, "
                    + "\(String(format: "%.2f", engine.streamingLatencySeconds))s latency"
            )
        }
    }

    /// Initialize with a pre-loaded engine.
    public func initialize(engine: LSEENDInferenceHelper) {
        let melSpectrogram = Self.createMelSpectrogram(featureConfig: engine.featureConfig)

        lock.withLock {
            updateTimelineConfig(engine: engine)
            _engine = engine
            _melSpectrogram = melSpectrogram
            _timeline = DiarizerTimeline(config: _timelineConfig)
            _session = nil
            resetBuffersLocked()

            logger.info("Initialized LS-EEND with pre-loaded engine")
        }
    }

    // MARK: - Speaker Priming

    /// Prime the diarizer with enrollment audio to warm the streaming state.
    ///
    /// This feeds audio through the active streaming session, discards any emitted
    /// predictions, and resets the visible timeline so subsequent calls to
    /// `process()` start again from frame 0 while keeping the warmed model state.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to use for priming.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    ///   - name: The speaker's name.
    ///   - overwriteAssignedSpeakerName: Whether enrollment may overwrite the name on an already-named slot
    ///     if the diarizer assigns the audio to that speaker.
    /// - Throws: ``LSEENDError/modelPredictionFailed(_:)`` if the diarizer is not initialized.
    public func enrollSpeaker(
        withSamples samples: [Float],
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

    /// Prime the diarizer with enrollment audio to warm the streaming state.
    ///
    /// - Parameters:
    ///   - samples: Audio samples to use for priming.
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
        try lock.withLock {
            let description: String = name.map { "named '\($0)'" } ?? "(no name)"
            guard let engine = _engine else {
                throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
            }

            let normalized = try normalizeSamplesLocked(samples, sourceSampleRate: sourceSampleRate) ?? samples
            guard !normalized.isEmpty else {
                logger.warning("Failed to enroll speaker \(description) because no speech detected")
                return nil
            }

            if _timeline.hasSegments {
                logger.warning("Trying to enroll a speaker while timeline has segments; timeline will be reset")
            }

            _timeline.reset(keepingSpeakers: true)
            var occupiedIndices = Set(_timeline.speakers.keys)
            _numFramesProcessed = 0
            _visibleStartFrameOffset = 0
            pendingAudio.removeAll(keepingCapacity: true)

            if _session == nil {
                _session = try engine.createSession(
                    inputSampleRate: engine.targetSampleRate, melSpectrogram: _melSpectrogram!)
            }
            guard let session = _session else {
                return nil
            }

            let update = try session.pushAudio(normalized)
            let didProcess = update.map { !$0.probabilities.isEmpty || !$0.previewProbabilities.isEmpty } ?? false

            guard didProcess else {
                let minimumSeconds = engine.streamingLatencySeconds
                logger.warning(
                    "Failed to enroll speaker \(description): not enough audio was provided. "
                        + "Please provide at least \(String(format: "%.2f", minimumSeconds)) seconds of speech."
                )
                return nil
            }

            if let update {
                let numSpeakers = engine.metadata.realOutputDim
                let result = DiarizerChunkResult(
                    startFrame: max(0, update.startFrame - _visibleStartFrameOffset),
                    finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
                    finalizedFrameCount: update.probabilities.rows,
                    tentativePredictions: flattenRowMajor(update.previewProbabilities, numSpeakers: numSpeakers),
                    tentativeFrameCount: update.previewProbabilities.rows
                )
                _numFramesProcessed += result.finalizedFrameCount
                _ = try _timeline.addChunk(result)
            }

            let speaker = _timeline.speakers.values.max { $0.numSpeechFrames < $1.numSpeechFrames }
            let enrolledSpeaker: DiarizerSpeaker?
            if let speaker, speaker.hasSegments {
                if let oldName = speaker.name {
                    guard overwriteAssignedSpeakerName else {
                        logger.warning(
                            "Failed to enroll speaker \(description): diarizer matched existing speaker '\(oldName)' "
                                + "at index \(speaker.index) and overwritingAssignedSpeakerName=false"
                        )
                        _visibleStartFrameOffset = session.snapshot().probabilities.rows
                        _numFramesProcessed = 0
                        _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })
                        pendingAudio.removeAll(keepingCapacity: true)
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

            _visibleStartFrameOffset = session.snapshot().probabilities.rows
            _numFramesProcessed = 0
            _timeline.reset(keepingSpeakersWhere: { occupiedIndices.contains($0.index) })
            pendingAudio.removeAll(keepingCapacity: true)

            logger.info(
                "Enrolled speaker \(description) with \(normalized.count) samples "
                    + "(\(String(format: "%.1f", Float(normalized.count) / Float(engine.targetSampleRate)))s), "
                    + "visible offset=\(_visibleStartFrameOffset)"
            )

            return enrolledSpeaker
        }
    }

    // MARK: - Streaming (Diarizer Protocol)

    /// Add audio samples to the processing buffer.
    ///
    /// Audio must be at the model's target sample rate (typically 8000 Hz).
    /// Call `process()` after adding audio to run inference.
    public func addAudio(_ samples: [Float]) {
        try? addAudio(samples, sourceSampleRate: nil)
    }

    /// Add audio samples to the processing buffer, resampling when needed.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples to enqueue.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    /// Add audio samples from any `Collection` of `Float` to the processing buffer.
    public func addAudio<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double? = nil
    ) throws where C.Element == Float {
        try lock.withLock {
            if let normalized = try normalizeSamplesLocked(samples, sourceSampleRate: sourceSampleRate) {
                pendingAudio.append(contentsOf: normalized)
            } else {
                pendingAudio.append(contentsOf: samples)
            }
        }
    }

    /// Process buffered audio and return any new results.
    ///
    /// - Returns: New chunk result if inference produced frames, nil otherwise
    public func process() throws -> DiarizerTimelineUpdate? {
        try lock.withLock { return try processLocked() }
    }

    /// Add and process a chunk of audio in one call.
    /// - Parameters:
    ///   - samples: Audio samples to process.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the model rate.
    /// - Returns: New chunk result if inference produced frames, nil otherwise.
    public func process<C: Collection>(
        samples: C,
        sourceSampleRate: Double? = nil
    ) throws -> DiarizerTimelineUpdate? where C.Element == Float {
        try lock.withLock {
            if let normalized = try normalizeSamplesLocked(samples, sourceSampleRate: sourceSampleRate) {
                pendingAudio.append(contentsOf: normalized)
            } else {
                pendingAudio.append(contentsOf: samples)
            }

            return try processLocked()
        }
    }

    /// Internal process — caller must hold lock.
    private func processLocked() throws -> DiarizerTimelineUpdate? {
        guard let engine = _engine else {
            throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
        }

        guard !pendingAudio.isEmpty else { return nil }

        // Lazily create session on first process call
        if _session == nil {
            _session = try engine.createSession(
                inputSampleRate: engine.targetSampleRate, melSpectrogram: _melSpectrogram!)
        }
        guard let session = _session else { return nil }

        // Clear unconditionally (even on throw) so failed audio isn't re-fed.
        // Using defer + direct pass avoids a CoW copy — pushAudio receives a
        // temporary reference, and removeAll runs after it returns (refcount == 1).
        defer { pendingAudio.removeAll(keepingCapacity: true) }

        guard let update = try session.pushAudio(pendingAudio) else {
            return nil
        }

        let numSpeakers = engine.metadata.realOutputDim
        let result = DiarizerChunkResult(
            startFrame: max(0, update.startFrame - _visibleStartFrameOffset),
            finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
            finalizedFrameCount: update.probabilities.rows,
            tentativePredictions: flattenRowMajor(update.previewProbabilities, numSpeakers: numSpeakers),
            tentativeFrameCount: update.previewProbabilities.rows
        )

        _numFramesProcessed += result.finalizedFrameCount
        return try _timeline.addChunk(result)
    }

    // MARK: - Offline (Diarizer Protocol)

    /// Progress callback: (processedSamples, totalSamples, chunksProcessed)
    public typealias ProgressCallback = (Int, Int, Int) -> Void

    /// Process a complete audio buffer.
    ///
    /// Resets state (unless pre-enrolled speakers are kept) and pushes all audio at once, then finalizes.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples at the model's target sample rate.
    ///   - sourceSampleRate: Source audio sample rate (if nil, assumes that it matches the engine's sample rate).
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it keeps the speakers if no more audio was added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline after processing.
    ///   - progressCallback: Optional callback (processedSamples, totalSamples, chunksProcessed).
    /// - Returns: Finalized timeline with segments.
    public func processComplete(
        _ samples: [Float],
        sourceSampleRate: Double? = nil,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool = true,
        progressCallback: ((Int, Int, Int) -> Void)? = nil
    ) throws -> DiarizerTimeline {
        try lock.withLock {
            try processCompleteLocked(
                samples,
                sourceSampleRate: sourceSampleRate,
                keepingEnrolledSpeakers: keepSpeakers,
                finalizeOnCompletion: finalizeOnCompletion,
                progressCallback: progressCallback
            )
        }
    }

    /// Process a complete audio buffer.
    ///
    /// Resets state (unless pre-enrolled speakers are kept) and pushes all audio at once, then finalizes.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples.
    ///   - sourceSampleRate: Source audio sample rate (if `nil`, assumes the model rate).
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it keeps the speakers if no more audio was added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline after processing.
    ///   - progressCallback: Optional callback `(processedSamples, totalSamples, chunksProcessed)`.
    /// - Returns: Finalized timeline with segments.
    public func processComplete<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double? = nil,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline where C.Element == Float {
        try lock.withLock {
            try processCompleteLocked(
                Array(samples),
                sourceSampleRate: sourceSampleRate,
                keepingEnrolledSpeakers: keepSpeakers,
                finalizeOnCompletion: finalizeOnCompletion,
                progressCallback: progressCallback
            )
        }
    }

    /// Process a complete audio file from a URL.
    ///
    /// Reads and resamples the file to ``targetSampleRate``, then delegates to
    /// ``processComplete(_:finalizeOnCompletion:progressCallback:)``.
    ///
    /// - Parameters:
    ///   - audioFileURL: Path to a WAV, CAF, or other audio file.
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it keeps the speakers if no more audio was added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline after processing
    ///   - progressCallback: Optional callback (processedSamples, totalSamples, chunksProcessed).
    /// - Returns: Finalized timeline with segments.
    public func processComplete(
        audioFileURL: URL,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool = true,
        progressCallback: ((Int, Int, Int) -> Void)? = nil
    ) throws -> DiarizerTimeline {
        try lock.withLock {
            guard let engine = _engine else {
                throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
            }

            let converter = AudioConverter(sampleRate: Double(engine.targetSampleRate))
            let audio = try converter.resampleAudioFile(audioFileURL)

            return try processCompleteLocked(
                audio,
                sourceSampleRate: nil,
                keepingEnrolledSpeakers: keepSpeakers,
                finalizeOnCompletion: finalizeOnCompletion,
                progressCallback: progressCallback
            )
        }
    }

    private func processCompleteLocked(
        _ samples: [Float],
        sourceSampleRate: Double?,
        keepingEnrolledSpeakers keepSpeakers: Bool? = nil,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline {
        let normalized = try normalizeSamplesLocked(samples, sourceSampleRate: sourceSampleRate) ?? samples

        guard let engine = _engine else {
            throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
        }

        let keepSpeakers = keepSpeakers ?? (_numFramesProcessed == 0 && pendingAudio.isEmpty)

        _timeline.reset(keepingSpeakers: keepSpeakers)
        _numFramesProcessed = 0
        pendingAudio.removeAll(keepingCapacity: true)
        let useRetainedSession = keepSpeakers && _session != nil
        if !keepSpeakers {
            _visibleStartFrameOffset = 0
            _session = nil
        }

        let session =
            if let retainedSession = _session, useRetainedSession {
                retainedSession
            } else {
                try engine.createSession(
                    inputSampleRate: engine.targetSampleRate, melSpectrogram: _melSpectrogram!)
            }
        let numSpeakers = engine.metadata.realOutputDim

        // Push all audio at once
        if let update = try session.pushAudio(normalized) {
            let chunk = DiarizerChunkResult(
                startFrame: max(0, update.startFrame - _visibleStartFrameOffset),
                finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
                finalizedFrameCount: update.probabilities.rows,
                tentativePredictions: flattenRowMajor(update.previewProbabilities, numSpeakers: numSpeakers),
                tentativeFrameCount: update.previewProbabilities.rows
            )
            _numFramesProcessed += chunk.finalizedFrameCount
            try _timeline.addChunk(chunk)
        }

        progressCallback?(normalized.count, normalized.count, 1)

        // Finalize remaining frames
        if let finalUpdate = try session.finalize() {
            let chunk = DiarizerChunkResult(
                startFrame: max(0, finalUpdate.startFrame - _visibleStartFrameOffset),
                finalizedPredictions: flattenRowMajor(finalUpdate.probabilities, numSpeakers: numSpeakers),
                finalizedFrameCount: finalUpdate.probabilities.rows,
                tentativePredictions: [],
                tentativeFrameCount: 0
            )
            _numFramesProcessed += chunk.finalizedFrameCount
            try _timeline.addChunk(chunk)
        }
        if useRetainedSession {
            _session = nil
        }

        if finalizeOnCompletion {
            _timeline.finalize()
        }
        return _timeline
    }

    // MARK: - Lifecycle (Diarizer Protocol)

    /// Reset all streaming state for a new audio stream.
    ///
    /// Preserves the loaded model. Call `initialize()` again to change models.
    public func reset() {
        lock.withLock {
            _session = nil
            _timeline.reset()
            resetBuffersLocked()
            logger.debug("LS-EEND state reset")
        }
    }

    /// Clean up all resources including the loaded model.
    public func cleanup() {
        lock.withLock {
            _engine = nil
            _session = nil
            _melSpectrogram = nil
            _timeline.reset()
            resetBuffersLocked()
            logger.info("LS-EEND resources cleaned up")
        }
    }

    // MARK: - LS-EEND Specific

    /// Finalize the current streaming session.
    ///
    /// Flushes any remaining frames and finalizes the timeline.
    /// After calling this, `process()` will no longer produce results
    /// until `reset()` is called.
    ///
    /// - Returns: Final chunk result if any remaining frames were flushed, nil otherwise
    @discardableResult
    public func finalizeSession() throws -> DiarizerChunkResult? {
        lock.lock()
        defer { lock.unlock() }

        guard let engine = _engine, let session = _session else { return nil }
        let numSpeakers = engine.metadata.realOutputDim
        var lastResult: DiarizerChunkResult?

        // Flush pending audio first — clear unconditionally so failed audio isn't retained.
        // Using defer + direct pass avoids a CoW copy.
        if !pendingAudio.isEmpty {
            defer { pendingAudio.removeAll(keepingCapacity: true) }
            let pushedUpdate = try session.pushAudio(pendingAudio)
            if let update = pushedUpdate {
                let flushedResult = DiarizerChunkResult(
                    startFrame: _numFramesProcessed,
                    finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
                    finalizedFrameCount: update.probabilities.rows,
                    tentativePredictions: [],
                    tentativeFrameCount: 0
                )
                _numFramesProcessed += flushedResult.finalizedFrameCount
                try _timeline.addChunk(flushedResult)
                lastResult = flushedResult
            }
        }

        if let finalUpdate = try session.finalize() {
            let finalResult = DiarizerChunkResult(
                startFrame: _numFramesProcessed,
                finalizedPredictions: flattenRowMajor(finalUpdate.probabilities, numSpeakers: numSpeakers),
                finalizedFrameCount: finalUpdate.probabilities.rows,
                tentativePredictions: [],
                tentativeFrameCount: 0
            )
            _numFramesProcessed += finalResult.finalizedFrameCount
            try _timeline.addChunk(finalResult)
            lastResult = finalResult
        }
        _timeline.finalize()
        _session = nil

        return lastResult
    }

    // MARK: - Private

    private func resetBuffersLocked() {
        pendingAudio.removeAll(keepingCapacity: true)
        _numFramesProcessed = 0
        _visibleStartFrameOffset = 0
    }

    private func normalizeSamplesLocked<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double?
    ) throws -> [Float]? where C.Element == Float {
        guard let engine = _engine,
            let sourceSampleRate,
            sourceSampleRate != Double(engine.targetSampleRate)
        else {
            return nil
        }

        return try AudioConverter(sampleRate: Double(engine.targetSampleRate))
            .resample(Array(samples), from: sourceSampleRate)
    }

    /// Create a new mel spectrogram instance owned by this diarizer.
    private static func createMelSpectrogram(featureConfig: LSEENDFeatureConfig) -> AudioMelSpectrogram {
        AudioMelSpectrogram(
            sampleRate: featureConfig.sampleRate,
            nMels: featureConfig.nMels,
            nFFT: featureConfig.nFFT,
            hopLength: featureConfig.hopLength,
            winLength: featureConfig.winLength,
            preemph: 0,
            padTo: 1,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )
    }

    private func updateTimelineConfig(engine: LSEENDInferenceHelper) {
        self._timelineConfig.numSpeakers = engine.metadata.realOutputDim
        self._timelineConfig.frameDurationSeconds = Float(1.0 / engine.modelFrameHz)
    }

    /// Convert an LSEENDMatrix to a flat [Float] in row-major layout.
    private func flattenRowMajor(_ matrix: LSEENDMatrix, numSpeakers: Int) -> [Float] {
        guard matrix.rows > 0, matrix.columns > 0 else { return [] }
        return matrix.values
    }
}
