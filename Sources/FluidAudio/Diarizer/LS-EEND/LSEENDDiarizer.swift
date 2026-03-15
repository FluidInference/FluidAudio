import AVFoundation
import CoreML
import Foundation

/// Speaker diarization using LS-EEND (Linear Streaming End-to-End Neural Diarization).
///
/// Supports both streaming and offline processing, matching the `SortformerDiarizer` API:
/// ```swift
/// let processor = LSEENDDiarizer()
/// try processor.initialize(descriptor: .defaultDescriptor(for: .dihard3))
///
/// // Streaming
/// processor.addAudio(audioChunk)
/// if let result = try processor.process() {
///     // Handle speaker probabilities
/// }
///
/// // Or complete file
/// let timeline = try processor.processComplete(audioSamples)
/// ```
public final class LSEENDDiarizer: Diarizer {
    private let lock = NSLock()
    private let logger = AppLogger(category: "LSEENDDiarizer")

    // MARK: - Diarizer Protocol Properties

    /// Accumulated results
    public var timeline: DiarizerTimeline {
        lock.lock()
        defer { lock.unlock() }
        return _timeline
    }

    /// Whether the processor is ready for processing
    public var isAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _engine != nil
    }

    /// Number of confirmed frames processed so far
    public var numFramesProcessed: Int {
        lock.lock()
        defer { lock.unlock() }
        return _numFramesProcessed
    }

    /// Model's target sample rate in Hz (e.g., 8000)
    public var targetSampleRate: Int? {
        lock.lock()
        defer { lock.unlock() }
        return _engine?.targetSampleRate
    }

    /// Output frame rate in Hz (e.g., 10.0)
    public var modelFrameHz: Double? {
        lock.lock()
        defer { lock.unlock() }
        return _engine?.modelFrameHz
    }

    /// Number of real speaker tracks (excluding boundary tracks)
    public var numSpeakers: Int? {
        lock.lock()
        defer { lock.unlock() }
        return _engine?.metadata.realOutputDim
    }

    // MARK: - Additional Properties

    /// Compute units for CoreML inference
    public let computeUnits: MLComputeUnits

    /// Post-processing configuration
    public var timelineConfig: DiarizerTimelineConfig {
        lock.lock()
        defer { lock.unlock() }
        return _timeline.config
    }

    /// Streaming latency in seconds
    public var streamingLatencySeconds: Double? {
        lock.lock()
        defer { lock.unlock() }
        return _engine?.streamingLatencySeconds
    }

    /// Total speaker slots in model output (including boundary tracks)
    public var decodeMaxSpeakers: Int? {
        lock.lock()
        defer { lock.unlock() }
        return _engine?.decodeMaxSpeakers
    }

    // MARK: - Private State

    private var _engine: LSEENDInferenceEngine?
    private var _session: LSEENDStreamingSession?
    private var _melSpectrogram: NeMoMelSpectrogram?
    private var _timeline: DiarizerTimeline
    private var _numFramesProcessed: Int = 0
    private var _timelineConfig: DiarizerTimelineConfig

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
        let engine = try LSEENDInferenceEngine(descriptor: descriptor, computeUnits: computeUnits)
        let melSpectrogram = Self.createMelSpectrogram(featureConfig: engine.featureConfig)

        lock.lock()
        defer { lock.unlock() }

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

    /// Initialize with a pre-loaded engine.
    public func initialize(engine: LSEENDInferenceEngine) {
        let melSpectrogram = Self.createMelSpectrogram(featureConfig: engine.featureConfig)

        lock.lock()
        defer { lock.unlock() }

        updateTimelineConfig(engine: engine)
        _engine = engine
        _melSpectrogram = melSpectrogram
        _timeline = DiarizerTimeline(config: _timelineConfig)
        _session = nil
        resetBuffersLocked()

        logger.info("Initialized LS-EEND with pre-loaded engine")
    }

    // MARK: - Streaming (Diarizer Protocol)

    /// Add audio samples to the processing buffer.
    ///
    /// Audio must be at the model's target sample rate (typically 8000 Hz).
    /// Call `process()` after adding audio to run inference.
    public func addAudio(_ samples: [Float]) {
        lock.lock()
        defer { lock.unlock() }

        pendingAudio.append(contentsOf: samples)
    }

    /// Add audio samples from any `Collection` of `Float` to the processing buffer.
    public func addAudio<C: Collection>(_ samples: C) where C.Element == Float {
        lock.lock()
        defer { lock.unlock() }

        pendingAudio.append(contentsOf: samples)
    }

    /// Process buffered audio and return any new results.
    ///
    /// - Returns: New chunk result if inference produced frames, nil otherwise
    public func process() throws -> DiarizerTimelineUpdate? {
        lock.lock()
        defer { lock.unlock() }

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

        let chunk = pendingAudio
        pendingAudio.removeAll(keepingCapacity: true)

        guard let update = try session.pushAudio(chunk) else {
            return nil
        }

        let numSpeakers = engine.metadata.realOutputDim
        let result = DiarizerChunkResult(
            startFrame: update.startFrame,
            finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
            finalizedFrameCount: update.probabilities.rows,
            tentativePredictions: flattenRowMajor(update.previewProbabilities, numSpeakers: numSpeakers),
            tentativeFrameCount: update.previewProbabilities.rows
        )

        _numFramesProcessed += result.finalizedFrameCount
        return try _timeline.addChunk(result)
    }

    /// Process a chunk of audio in one call.
    ///
    /// Convenience method that combines `addAudio()` and `process()`.
    public func process(samples: [Float]) throws -> DiarizerTimelineUpdate? {
        lock.lock()
        defer { lock.unlock() }

        guard let engine = _engine else {
            throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
        }

        pendingAudio.append(contentsOf: samples)

        guard !pendingAudio.isEmpty else { return nil }

        if _session == nil {
            _session = try engine.createSession(
                inputSampleRate: engine.targetSampleRate, melSpectrogram: _melSpectrogram!)
        }
        guard let session = _session else { return nil }

        let chunk = pendingAudio
        pendingAudio.removeAll(keepingCapacity: true)

        guard let update = try session.pushAudio(chunk) else {
            return nil
        }

        let numSpeakers = engine.metadata.realOutputDim
        let result = DiarizerChunkResult(
            startFrame: update.startFrame,
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
    /// Resets state and pushes all audio at once, then finalizes.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples at the model's target sample rate
    ///   - finalizeOnCompletion: Whether to finalize the timeline after processing
    ///   - progressCallback: Optional callback (processedSamples, totalSamples, chunksProcessed)
    /// - Returns: Finalized timeline with segments
    public func processComplete(
        _ samples: [Float],
        finalizeOnCompletion: Bool = true,
        progressCallback: ((Int, Int, Int) -> Void)? = nil
    ) throws -> DiarizerTimeline {
        lock.lock()
        defer { lock.unlock() }

        guard let engine = _engine else {
            throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
        }

        // Reset for fresh processing
        updateTimelineConfig(engine: engine)
        _timeline = DiarizerTimeline(config: _timelineConfig)
        _numFramesProcessed = 0
        _session = nil
        pendingAudio.removeAll(keepingCapacity: true)

        let session = try engine.createSession(
            inputSampleRate: engine.targetSampleRate, melSpectrogram: _melSpectrogram!)
        let numSpeakers = engine.metadata.realOutputDim

        // Push all audio at once
        if let update = try session.pushAudio(samples) {
            let chunk = DiarizerChunkResult(
                startFrame: update.startFrame,
                finalizedPredictions: flattenRowMajor(update.probabilities, numSpeakers: numSpeakers),
                finalizedFrameCount: update.probabilities.rows,
                tentativePredictions: flattenRowMajor(update.previewProbabilities, numSpeakers: numSpeakers),
                tentativeFrameCount: update.previewProbabilities.rows
            )
            _numFramesProcessed += chunk.finalizedFrameCount
            try _timeline.addChunk(chunk)
        }

        progressCallback?(samples.count, samples.count, 1)

        // Finalize remaining frames
        if let finalUpdate = try session.finalize() {
            let chunk = DiarizerChunkResult(
                startFrame: _numFramesProcessed,
                finalizedPredictions: flattenRowMajor(finalUpdate.probabilities, numSpeakers: numSpeakers),
                finalizedFrameCount: finalUpdate.probabilities.rows,
                tentativePredictions: [],
                tentativeFrameCount: 0
            )
            _numFramesProcessed += chunk.finalizedFrameCount
            try _timeline.addChunk(chunk)
        }

        if finalizeOnCompletion {
            _timeline.finalize()
        }
        return _timeline
    }

    /// Process a complete audio file from a URL.
    ///
    /// Reads and resamples the file to ``targetSampleRate``, then delegates to
    /// ``processComplete(_:finalizeOnCompletion:progressCallback:)``.
    ///
    /// - Parameters:
    ///   - audioFileURL: Path to a WAV, CAF, or other audio file.
    ///   - progressCallback: Optional callback (processedSamples, totalSamples, chunksProcessed).
    /// - Returns: Finalized timeline with segments.
    public func processComplete(
        audioFileURL: URL,
        progressCallback: ((Int, Int, Int) -> Void)? = nil
    ) throws -> DiarizerTimeline {
        let engine = try lock.withLock {
            guard let engine = _engine else {
                throw LSEENDError.modelPredictionFailed("LS-EEND processor not initialized. Call initialize() first.")
            }
            return engine
        }
        let converter = AudioConverter(sampleRate: Double(engine.targetSampleRate))
        let audio = try converter.resampleAudioFile(audioFileURL)
        return try processComplete(audio, progressCallback: progressCallback)
    }

    // MARK: - Lifecycle (Diarizer Protocol)

    /// Reset all streaming state for a new audio stream.
    ///
    /// Preserves the loaded model. Call `initialize()` again to change models.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        _session = nil
        _timeline.reset()
        resetBuffersLocked()
        logger.debug("LS-EEND state reset")
    }

    /// Clean up all resources including the loaded model.
    public func cleanup() {
        lock.lock()
        defer { lock.unlock() }

        _engine = nil
        _session = nil
        _melSpectrogram = nil
        _timeline.reset()
        resetBuffersLocked()
        logger.info("LS-EEND resources cleaned up")
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

        // Flush pending audio first
        if !pendingAudio.isEmpty {
            let chunk = pendingAudio
            pendingAudio.removeAll(keepingCapacity: true)
            let _ = try session.pushAudio(chunk)
        }

        guard let finalUpdate = try session.finalize() else {
            _session = nil
            _timeline.finalize()
            return nil
        }

        let numSpeakers = engine.metadata.realOutputDim
        let result = DiarizerChunkResult(
            startFrame: _numFramesProcessed,
            finalizedPredictions: flattenRowMajor(finalUpdate.probabilities, numSpeakers: numSpeakers),
            finalizedFrameCount: finalUpdate.probabilities.rows,
            tentativePredictions: [],
            tentativeFrameCount: 0
        )
        _numFramesProcessed += result.finalizedFrameCount
        try _timeline.addChunk(result)
        _timeline.finalize()
        _session = nil

        return result
    }

    // MARK: - Private

    private func resetBuffersLocked() {
        pendingAudio.removeAll(keepingCapacity: true)
        _numFramesProcessed = 0
    }

    /// Create a new mel spectrogram instance owned by this diarizer.
    private static func createMelSpectrogram(featureConfig: LSEENDFeatureConfig) -> NeMoMelSpectrogram {
        NeMoMelSpectrogram(
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

    private func updateTimelineConfig(engine: LSEENDInferenceEngine) {
        self._timelineConfig.numSpeakers = engine.metadata.realOutputDim
        self._timelineConfig.frameDurationSeconds = Float(1.0 / engine.modelFrameHz)
    }

    /// Convert an LSEENDMatrix to a flat [Float] in row-major layout.
    private func flattenRowMajor(_ matrix: LSEENDMatrix, numSpeakers: Int) -> [Float] {
        guard matrix.rows > 0, matrix.columns > 0 else { return [] }
        return matrix.values
    }
}
