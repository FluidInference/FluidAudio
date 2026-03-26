import Foundation

// MARK: - Diarizer Protocol

/// Protocol for frame-based speaker diarization processors.
///
/// Both SortformerDiarizer and LS-EEND processors conform to this protocol,
/// providing a unified streaming and offline diarization API.
public protocol Diarizer: AnyObject {
    /// Whether the processor is initialized and ready
    var isAvailable: Bool { get }

    /// Number of confirmed frames processed so far
    var numFramesProcessed: Int { get }

    /// Model's target sample rate in Hz
    var targetSampleRate: Int? { get }

    /// Output frame rate in Hz
    var modelFrameHz: Double? { get }

    /// Number of real speaker output tracks
    var numSpeakers: Int? { get }

    /// Diarization timeline
    var timeline: DiarizerTimeline { get }

    // MARK: Streaming

    /// Add audio samples to the processing buffer.
    ///
    /// Implementations may resample the input when `sourceSampleRate` differs from
    /// the model's target sample rate.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples to enqueue for diarization.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the target rate.
    func addAudio<C: Collection>(_ samples: C, sourceSampleRate: Double?) throws
    where C.Element == Float

    /// Process buffered audio and return any newly available diarization output.
    func process() throws -> DiarizerTimelineUpdate?

    /// Add audio and process it in one call.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples to process.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the target rate.
    /// - Returns: A timeline update containing finalized and tentative output, or `nil`
    ///   if not enough buffered audio was available to emit frames.
    func process<C: Collection>(samples: C, sourceSampleRate: Double?) throws -> DiarizerTimelineUpdate?
    where C.Element == Float

    // MARK: Offline

    /// Process a complete audio buffer and return the resulting timeline.
    ///
    /// - Parameters:
    ///   - samples: Complete mono audio buffer to diarize.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the target rate.
    ///   - keepSpeakers: Whether to keep pre-enrolled speakers. If `nil`, it will keep the speakers if no audio has been added.
    ///   - finalizeOnCompletion: Whether to finalize the timeline before returning it.
    ///   - progressCallback: Optional callback receiving `(processedSamples, totalSamples, chunksProcessed)`.
    /// - Returns: The diarization timeline for the provided audio.
    func processComplete<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double?,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline
    where C.Element == Float

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
    func processComplete(
        audioFileURL: URL,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline

    // MARK: Lifecycle

    /// Reset streaming state while keeping model loaded
    func reset()

    /// Clean up all resources
    func cleanup()

    /// Pre-enroll a speaker before running the diarizer.
    ///
    /// - Parameters:
    ///   - samples: Enrollment audio samples.
    ///   - sourceSampleRate: Sample rate of `samples`, or `nil` if already at the target rate.
    ///   - name: The speaker's name.
    ///   - overwriteAssignedSpeakerName: Whether enrollment may overwrite the name of an already-named slot
    ///     if the diarizer assigns the audio to that speaker.
    /// - Returns: The enrolled speaker.
    func enrollSpeaker<C: Collection>(
        withAudio samples: C,
        sourceSampleRate: Double?,
        named name: String?,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool
    ) throws -> DiarizerSpeaker? where C.Element == Float
}

// MARK: - Post-Processing Configuration

/// Configuration for post-processing diarizer predictions into segments.
///
/// Generalizes Sortformer's `SortformerPostProcessingConfig` for any frame-based
/// diarizer (Sortformer, LS-EEND, etc.).
public struct DiarizerTimelineConfig: Sendable {
    /// Number of speaker output tracks
    public var numSpeakers: Int

    /// Duration of one output frame in seconds
    public var frameDurationSeconds: Float

    /// Onset threshold for detecting the beginning of speech
    public var onsetThreshold: Float

    /// Offset threshold for detecting the end of speech
    public var offsetThreshold: Float

    /// Padding frames added before each speech segment
    public var onsetPadFrames: Int

    /// Padding frames added after each speech segment
    public var offsetPadFrames: Int

    /// Minimum segment length in frames (shorter segments are discarded)
    public var minFramesOn: Int

    /// Minimum gap length in frames (shorter gaps are closed)
    public var minFramesOff: Int

    /// Maximum number of finalized prediction frames to retain (nil = unlimited)
    public var maxStoredFrames: Int?

    // MARK: - Seconds Accessors

    /// Padding duration added before each speech segment in seconds
    public var onsetPadSeconds: Float {
        get { Float(onsetPadFrames) * frameDurationSeconds }
        set { onsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Padding duration added after each speech segment in seconds
    public var offsetPadSeconds: Float {
        get { Float(offsetPadFrames) * frameDurationSeconds }
        set { offsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Minimum duration for a speech segment in seconds
    public var minDurationOn: Float {
        get { Float(minFramesOn) * frameDurationSeconds }
        set { minFramesOn = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Minimum gap duration between speech segments in seconds (shorter gaps are closed)
    public var minDurationOff: Float {
        get { Float(minFramesOff) * frameDurationSeconds }
        set { minFramesOff = Int(round(newValue / frameDurationSeconds)) }
    }

    // MARK: - Presets

    /// Default configuration with no post-processing (pass-through thresholding at 0.5)
    public static func `default`(numSpeakers: Int, frameDurationSeconds: Float) -> DiarizerTimelineConfig {
        DiarizerTimelineConfig(
            numSpeakers: numSpeakers,
            frameDurationSeconds: frameDurationSeconds,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 0,
            minFramesOff: 0
        )
    }

    /// Default timeline configuration for Sortformer (4 speakers, 80ms frames)
    public static let sortformerDefault = Self.default(numSpeakers: 4, frameDurationSeconds: 0.08)

    // MARK: - Init

    /// - Parameters:
    ///   - numSpeakers: Number of speaker output tracks
    ///   - frameDurationSeconds: Duration of one output frame in seconds
    ///   - onsetThreshold: Threshold for detecting the beginning of speech
    ///   - offsetThreshold: Threshold for detecting the end of speech
    ///   - onsetPadFrames: Padding frames added before each speech segment
    ///   - offsetPadFrames: Padding frames added after each speech segment
    ///   - minFramesOn: Minimum segment length in frames (shorter segments are discarded)
    ///   - minFramesOff: Minimum gap length in frames (shorter gaps are closed)
    ///   - maxStoredFrames: Maximum number of finalized prediction frames to retain (nil = unlimited)
    public init(
        numSpeakers: Int? = nil,
        frameDurationSeconds: Float? = nil,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.numSpeakers = numSpeakers ?? 1
        self.frameDurationSeconds = frameDurationSeconds ?? 0.08
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = onsetPadFrames
        self.offsetPadFrames = offsetPadFrames
        self.minFramesOn = minFramesOn
        self.minFramesOff = minFramesOff
        self.maxStoredFrames = maxStoredFrames
    }

    /// - Parameters:
    ///   - numSpeakers: Number of speaker output tracks
    ///   - frameDurationSeconds: Duration of one output frame in seconds
    ///   - onsetThreshold: Threshold for detecting the beginning of speech
    ///   - offsetThreshold: Threshold for detecting the end of speech
    ///   - onsetPadSeconds: Padding duration added before each speech segment
    ///   - offsetPadSeconds: Padding duration added after each speech segment
    ///   - minDurationOn: Minimum segment length in seconds (shorter segments are discarded)
    ///   - minDurationOff: Minimum gap length in seconds (shorter gaps are closed)
    ///   - maxStoredFrames: Maximum number of finalized raw prediction frames to retain (nil = unlimited)
    public init(
        numSpeakers: Int? = nil,
        frameDurationSeconds: Float? = nil,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadSeconds: Float = 0,
        offsetPadSeconds: Float = 0,
        minDurationOn: Float = 0,
        minDurationOff: Float = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.numSpeakers = numSpeakers ?? 1
        self.frameDurationSeconds = frameDurationSeconds ?? 0.08
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = Int(round(onsetPadSeconds / self.frameDurationSeconds))
        self.offsetPadFrames = Int(round(offsetPadSeconds / self.frameDurationSeconds))
        self.minFramesOn = Int(round(minDurationOn / self.frameDurationSeconds))
        self.minFramesOff = Int(round(minDurationOff / self.frameDurationSeconds))
        self.maxStoredFrames = maxStoredFrames
    }
}

// MARK: - Speaker

public final class DiarizerSpeaker: Identifiable, CustomStringConvertible {
    /// Speaker ID
    public let id: UUID

    /// Speaker's string representation
    public var description: String {
        queue.sync { _name ?? "Speaker \(_index)" }
    }

    /// Display name
    public var name: String? {
        get { queue.sync { _name } }
        set { queue.sync(flags: .barrier) { _name = newValue } }
    }

    /// Slot in the diarizer predictions
    public var index: Int {
        get { queue.sync { _index } }
        set { queue.sync(flags: .barrier) { _index = newValue } }
    }

    /// Confirmed/finalized speech segments that belong to this speaker
    public var finalizedSegments: [DiarizerSegment] {
        get { queue.sync { _finalizedSegments } }
        set { queue.sync(flags: .barrier) { _finalizedSegments = newValue } }
    }

    /// Tentative speech segments that belong to this speaker
    public var tentativeSegments: [DiarizerSegment] {
        get { queue.sync { _tentativeSegments } }
        set { queue.sync(flags: .barrier) { _tentativeSegments = newValue } }
    }

    /// Whether this speaker has any segments
    public var hasSegments: Bool {
        queue.sync { !(_finalizedSegments.isEmpty && _tentativeSegments.isEmpty) }
    }

    /// Number of segments (finalized + tentative)
    public var segmentCount: Int {
        queue.sync { _finalizedSegments.count + _tentativeSegments.count }
    }

    /// Number of confirmed segments
    public var finalizedSegmentCount: Int {
        queue.sync { _finalizedSegments.count }
    }

    /// Number of tentative segments
    public var tentativeSegmentCount: Int {
        queue.sync { _tentativeSegments.count }
    }

    /// Last segment (tentative or finalized). Checks tentative segments first, falls back to finalized if none found.
    public var lastSegment: DiarizerSegment? {
        queue.sync { _tentativeSegments.last ?? _finalizedSegments.last }
    }

    /// Total duration of segments in seconds (finalized + tentative)
    public var speechDuration: Float {
        queue.sync {
            return
                (_finalizedSegments.reduce(0.0) { $0 + $1.duration }
                + _tentativeSegments.reduce(0.0) { $0 + $1.duration })
        }
    }

    /// Duration of all finalized segments in seconds
    public var finalizedSpeechDuration: Float {
        queue.sync {
            _finalizedSegments.reduce(0.0) { $0 + $1.duration }
        }
    }

    /// Duration of all tentative segments in seconds
    public var tentativeSpeechDuration: Float {
        queue.sync {
            _tentativeSegments.reduce(0.0) { $0 + $1.duration }
        }
    }

    /// Total number of frames spanned by all segments (finalized + tentative)
    public var numSpeechFrames: Int {
        queue.sync {
            return (_finalizedSegments.reduce(0) { $0 + $1.length } + _tentativeSegments.reduce(0) { $0 + $1.length })
        }
    }

    /// Number of frames in all finalized segments
    public var numFinalizedSpeechFrames: Int {
        queue.sync {
            _finalizedSegments.reduce(0) { $0 + $1.length }
        }
    }

    /// Number of frames in all tentative segments
    public var numTentativeSpeechFrames: Int {
        queue.sync {
            _tentativeSegments.reduce(0) { $0 + $1.length }
        }
    }

    private var _name: String?
    private var _index: Int
    private var _finalizedSegments: [DiarizerSegment] = []
    private var _tentativeSegments: [DiarizerSegment] = []
    private let queue = DispatchQueue(label: "FluidAudio.Diarization.DiarizerSpeaker")

    /// - Parameters:
    ///   - id: Speaker UUID
    ///   - index: Index in diarizer output
    ///   - name: Speaker display name
    public init(
        id: UUID = UUID(),
        index: Int,
        name: String? = nil
    ) {
        self.id = id
        self._index = index
        self._name = name
    }

    /// Finalize all segments
    /// - Parameter minFramesOn: Minimum segment length
    public func finalize(enforcingMinFramesOn minFramesOn: Int? = nil) {
        queue.sync(flags: .barrier) {
            if let minFramesOn {
                _tentativeSegments.removeAll { $0.length < minFramesOn }
            }
            _finalizedSegments.append(contentsOf: _tentativeSegments)
            _tentativeSegments.removeAll()
        }
    }

    /// Clear segments
    public func reset() {
        queue.sync(flags: .barrier) {
            _tentativeSegments.removeAll()
            _finalizedSegments.removeAll()
        }
    }

    /// Clear all tentative segments
    /// - Parameter keepingCapacity: Whether to keep the reserved capacity in the tentative segments list.
    public func removeAllTentative(keepingCapacity: Bool = false) {
        queue.sync(flags: .barrier) {
            _tentativeSegments.removeAll(keepingCapacity: keepingCapacity)
        }
    }

    /// Append a tentative segment
    /// - Parameter segment: The segment to append
    public func appendTentative(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            _tentativeSegments.append(segment)
        }
    }

    /// Append a finalized segment
    /// - Parameter segment: The segment to append
    public func appendFinalized(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            _finalizedSegments.append(segment)
        }
    }

    /// Append a segment, automatically detecting if it's finalized or tentative
    /// - Parameter segment: The segment to append
    public func append(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            if segment.isFinalized {
                _finalizedSegments.append(segment)
            } else {
                _tentativeSegments.append(segment)
            }
        }
    }

    /// Pop last tentative segment
    /// - Returns: The popped segment
    @discardableResult
    public func popLastTentative() -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            _tentativeSegments.popLast()
        }
    }

    /// Pop last finalized segment
    /// - Returns: The popped segment
    @discardableResult
    public func popLastFinalized() -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            return _finalizedSegments.popLast()
        }
    }

    /// Pop last tentative or finalized segment
    /// - Parameter fromFinalized: Whether to pop the segment from the finalized segment list
    /// - Returns: The popped segment
    @discardableResult
    public func popLast(fromFinalized: Bool) -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            return
                (fromFinalized
                ? _finalizedSegments.popLast()
                : _tentativeSegments.popLast())
        }
    }

    /// Pop last segment. Pops the last tentative segment first. Falls back to the last finalized segment if no
    /// tentative segments are found.
    /// - Returns: The popped segment
    @discardableResult
    public func popLast() -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            return _tentativeSegments.popLast() ?? _finalizedSegments.popLast()
        }
    }
}

// MARK: - Segment

/// A single speaker segment from any diarizer.
public struct DiarizerSegment: Sendable, Identifiable, Comparable, Equatable {
    public let id: UUID

    /// Speaker index in diarizer output
    public var speakerIndex: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }

    /// Whether this segment is finalized
    public var isFinalized: Bool

    /// Duration of one frame in seconds
    public let frameDurationSeconds: Float

    /// Confidence in this speech segment (average speech probability from the diarizer)
    public var confidence: Float = 0.0

    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * frameDurationSeconds }

    /// Duration in seconds
    public var duration: Float { Float(endFrame - startFrame) * frameDurationSeconds }

    /// Speaker label
    public var speakerLabel: String { "Speaker \(speakerIndex)" }

    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        frameDurationSeconds: Float,
        confidence: Float = 0
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
        self.confidence = confidence
    }

    public init(
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float,
        confidence: Float = 0
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
        self.confidence = confidence
    }

    /// Check if this overlaps with another segment
    public func overlaps(with other: DiarizerSegment) -> Bool {
        (startFrame <= other.endFrame) && (other.startFrame <= endFrame)
    }

    /// Merge another segment into this one
    public mutating func absorb(_ other: DiarizerSegment) {
        startFrame = min(startFrame, other.startFrame)
        endFrame = max(endFrame, other.endFrame)
    }

    /// Extend the end of this segment
    public mutating func extendEnd(toFrame frame: Int) {
        endFrame = max(endFrame, frame)
    }

    /// Extend the start of this segment
    public mutating func extendStart(toFrame frame: Int) {
        startFrame = min(startFrame, frame)
    }

    public static func < (lhs: DiarizerSegment, rhs: DiarizerSegment) -> Bool {
        return (lhs.startFrame, lhs.endFrame, lhs.speakerIndex) < (rhs.startFrame, rhs.endFrame, rhs.speakerIndex)
    }

    public static func == (lhs: DiarizerSegment, rhs: DiarizerSegment) -> Bool {
        return (lhs.startFrame, lhs.endFrame, lhs.speakerIndex) == (rhs.startFrame, rhs.endFrame, rhs.speakerIndex)
    }
}

// MARK: - Chunk Result

/// Result from a single streaming diarization step (works with any diarizer).
///
/// Maps directly to `SortformerChunkResult` for Sortformer,
/// and wraps `LSEENDStreamingUpdate` for LS-EEND.
public struct DiarizerChunkResult: Sendable {
    /// Speaker probabilities for finalized frames.
    /// Flat array of shape [frameCount, numSpeakers].
    public let finalizedPredictions: [Float]

    /// Number of finalized frames in this result
    public let finalizedFrameCount: Int

    /// Frame index of the first confirmed frame
    public let startFrame: Int

    /// Tentative/preview predictions (may change with future data).
    /// Flat array of shape [tentativeFrameCount, numSpeakers].
    public let tentativePredictions: [Float]

    /// Number of tentative frames
    public let tentativeFrameCount: Int

    /// Frame index of first tentative frame
    public var tentativeStartFrame: Int { startFrame + finalizedFrameCount }

    public init(
        startFrame: Int,
        finalizedPredictions: [Float],
        finalizedFrameCount: Int,
        tentativePredictions: [Float] = [],
        tentativeFrameCount: Int = 0
    ) {
        self.startFrame = startFrame
        self.finalizedPredictions = finalizedPredictions
        self.finalizedFrameCount = finalizedFrameCount
        self.tentativePredictions = tentativePredictions
        self.tentativeFrameCount = tentativeFrameCount
    }

    /// Get probability for a specific speaker at a confirmed frame
    public func probability(speaker: Int, frame: Int, numSpeakers: Int) -> Float {
        guard frame < finalizedFrameCount, speaker < numSpeakers else { return 0 }
        return finalizedPredictions[frame * numSpeakers + speaker]
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int, numSpeakers: Int) -> Float {
        guard frame < tentativeFrameCount, speaker < numSpeakers else { return 0 }
        return tentativePredictions[frame * numSpeakers + speaker]
    }
}

// MARK: - Timeline

/// Complete diarization timeline managing streaming predictions and segments.
///
/// Generalizes `SortformerTimeline` for any frame-based diarizer. Works with
/// both Sortformer (fixed 4 speakers) and LS-EEND (variable speaker count).
public final class DiarizerTimeline {
    private struct ClosedSegmentStats {
        var start: Int
        var end: Int
        var activitySum: Float
        var activeFrameCount: Int
    }

    public enum KeptOnReset {
        case nothing
        case namedSpeakers
        case allSpeakers
        case speakersWithoutSegments
        case speakersWithSegments
    }

    private struct StreamingState {
        var startFrame: Int
        var isSpeaking: Bool
        var activitySum: Float
        var activeFrameCount: Int
        var lastSegment: ClosedSegmentStats?

        init(
            startFrame: Int = 0,
            isSpeaking: Bool = false,
            activitySum: Float = 0,
            activeFrameCount: Int = 0,
            lastSegment: ClosedSegmentStats? = nil
        ) {
            self.startFrame = startFrame
            self.isSpeaking = isSpeaking
            self.activitySum = activitySum
            self.activeFrameCount = activeFrameCount
            self.lastSegment = lastSegment
        }
    }

    /// Post-processing configuration
    public let config: DiarizerTimelineConfig

    /// Finalized frame-wise speaker predictions.
    /// Flat array of shape [numFrames, numSpeakers].
    public var finalizedPredictions: [Float] {
        queue.sync { _finalizedPredictions }
    }

    /// Tentative predictions.
    /// Flat array of shape [numTentative, numSpeakers].
    public var tentativePredictions: [Float] {
        queue.sync { _tentativePredictions }
    }

    /// Total number of finalized frames
    public var numFinalizedFrames: Int {
        queue.sync { _numFinalizedFrames }
    }

    /// Number of tentative frames
    public var numTentativeFrames: Int {
        queue.sync { _tentativePredictions.count / speakerCapacity }
    }

    /// Total number of frames (finalized + tentative)
    public var numFrames: Int {
        queue.sync { _numFinalizedFrames + _tentativePredictions.count / speakerCapacity }
    }

    /// Speakers in the timeline
    public var speakers: [Int: DiarizerSpeaker] {
        get { queue.sync { _speakers } }
        set {
            queue.sync(flags: .barrier) {
                let maxSpeakers = speakerCapacity

                _speakers = newValue.filter { key, _ in
                    key >= 0 && key < maxSpeakers
                }

                for (index, speaker) in _speakers {
                    speaker.index = index
                }
            }
        }
    }

    /// Whether the timeline has any segments
    public var hasSegments: Bool {
        speakers.values.contains(where: \.hasSegments)
    }

    /// Duration of finalized predictions in seconds
    public var finalizedDuration: Float {
        Float(numFinalizedFrames) * config.frameDurationSeconds
    }

    /// Duration of tentative predictions in seconds
    public var tentativeDuration: Float {
        Float(numTentativeFrames) * config.frameDurationSeconds
    }

    /// Duration of all predictions (finalized + tentative) in seconds
    public var duration: Float {
        Float(numFrames) * config.frameDurationSeconds
    }

    /// Maximum number of speakers
    public var speakerCapacity: Int {
        config.numSpeakers
    }

    private var _finalizedPredictions: [Float] = []
    private var _tentativePredictions: [Float] = []
    private var _speakers: [Int: DiarizerSpeaker] = [:]
    private var _numFinalizedFrames: Int = 0

    // Segment builder state
    private var states: [StreamingState]

    private let queue = DispatchQueue(label: "FluidAudio.Diarizer.DiarizerTimeline")

    private static let logger = AppLogger(category: "DiarizerTimeline")

    // MARK: - Init

    /// Initialize for streaming usage
    public init(config: DiarizerTimelineConfig) {
        self.config = config
        states = Array(repeating: .init(), count: config.numSpeakers)
        _speakers = [:]
    }

    /// Initialize with existing probabilities (batch processing or restored state)
    public convenience init(
        finalizedPredictions: [Float],
        tentativePredictions: [Float],
        config: DiarizerTimelineConfig,
        isComplete: Bool = true
    ) throws {
        self.init(config: config)

        try rebuild(
            finalizedPredictions: finalizedPredictions,
            tentativePredictions: tentativePredictions,
            isComplete: isComplete
        )
    }

    /// Initialize with existing probabilities (batch processing or restored state)
    public convenience init(
        allPredictions: [Float],
        config: DiarizerTimelineConfig,
        isComplete: Bool = true
    ) throws {
        try self.init(
            finalizedPredictions: allPredictions,
            tentativePredictions: [],
            config: config,
            isComplete: isComplete
        )
    }

    // MARK: - Streaming API

    /// Add new predictions from the diarizer
    @discardableResult
    public func addPredictions(
        finalizedPredictions: [Float],
        tentativePredictions: [Float]
    ) throws -> DiarizerTimelineUpdate {
        let numFinalized = finalizedPredictions.count / speakerCapacity
        let numTentative = tentativePredictions.count / speakerCapacity

        let chunk = DiarizerChunkResult(
            startFrame: self.numFinalizedFrames,
            finalizedPredictions: finalizedPredictions,
            finalizedFrameCount: numFinalized,
            tentativePredictions: tentativePredictions,
            tentativeFrameCount: numTentative
        )

        return try addChunk(chunk)
    }

    /// Add a new chunk of predictions from the diarizer
    @discardableResult
    public func addChunk(_ chunk: DiarizerChunkResult) throws -> DiarizerTimelineUpdate {
        try queue.sync(flags: .barrier) {
            try verifyPredictionCounts(
                finalized: chunk.finalizedPredictions,
                tentative: chunk.tentativePredictions
            )

            _finalizedPredictions.append(contentsOf: chunk.finalizedPredictions)
            _tentativePredictions = chunk.tentativePredictions

            for speaker in _speakers.values {
                speaker.removeAllTentative(keepingCapacity: true)
            }

            var confirmedCounts = [Int](repeating: 0, count: speakerCapacity)
            for (index, speaker) in _speakers {
                confirmedCounts[index] = speaker.finalizedSegmentCount
            }

            updateSegments(
                predictions: chunk.finalizedPredictions,
                numFrames: chunk.finalizedFrameCount,
                isFinalized: true,
                addTrailingTentative: false
            )

            _numFinalizedFrames += chunk.finalizedFrameCount

            updateSegments(
                predictions: chunk.tentativePredictions,
                numFrames: chunk.tentativeFrameCount,
                isFinalized: false,
                addTrailingTentative: true
            )

            trimPredictions()

            let newConfirmed = _speakers.flatMap { (index, speaker) in
                let startIndex = confirmedCounts[index]
                guard startIndex < speaker.finalizedSegmentCount else {
                    return ArraySlice<DiarizerSegment>()
                }
                return speaker.finalizedSegments.suffix(from: startIndex)
            }

            let newTentative = _speakers.values.flatMap(\.tentativeSegments)

            return DiarizerTimelineUpdate(
                finalizedSegments: newConfirmed,
                tentativeSegments: newTentative,
                chunkResult: chunk
            )
        }
    }

    /// Finalize all tentative data at end of recording
    public func finalize() {
        queue.sync(flags: .barrier) { finalizeLocked() }
    }

    private func finalizeLocked() {
        _finalizedPredictions.append(contentsOf: _tentativePredictions)
        _numFinalizedFrames += _tentativePredictions.count / speakerCapacity
        _tentativePredictions.removeAll()
        for speaker in _speakers.values {
            speaker.finalize(enforcingMinFramesOn: config.minFramesOn)
        }
        trimPredictions()
    }

    /// Reset to initial state
    /// - Parameter condition: Condition for when to keep a speaker. All speakers still have their segments reset.
    public func reset(keepingSpeakersWhere condition: (DiarizerSpeaker) -> Bool) {
        queue.sync(flags: .barrier) { resetLocked(keepingSpeakersWhere: condition) }
    }

    /// Reset to initial state
    /// - Parameter keepingSpeakers: Whether to keep existing speakers enrolled. Their segments are still reset.
    public func reset(keepingSpeakers: Bool = false) {
        queue.sync(flags: .barrier) {
            resetLocked(keepingSpeakers: keepingSpeakers)
        }
    }

    private func resetLocked(keepingSpeakersWhere condition: (DiarizerSpeaker) -> Bool) {
        _finalizedPredictions.removeAll()
        _tentativePredictions.removeAll()
        _numFinalizedFrames = 0
        states = Array(repeating: .init(), count: speakerCapacity)

        _speakers = _speakers.filter {
            condition($0.value)
        }

        for speaker in _speakers.values {
            speaker.reset()
        }
    }

    private func resetLocked(keepingSpeakers: Bool) {
        _finalizedPredictions.removeAll()
        _tentativePredictions.removeAll()
        _numFinalizedFrames = 0
        states = Array(repeating: .init(), count: speakerCapacity)

        if keepingSpeakers {
            for speaker in _speakers.values {
                speaker.reset()
            }
        } else {
            _speakers = [:]
        }
    }

    /// Rebuild the timeline from initial predictions. This is equivalent to reinitializing the timeline.
    /// - Parameters:
    ///   - finalizedPredictions: Finalized prediction matrix `[numFrames, numSpeakers]` flattened
    ///   - tentativePredictions: Tentative prediction matrix `[numFrames, numSpeakers]` flattened
    ///   - keepingSpeakers: Whether to keep the old speaker names and slots.
    ///   - isComplete: Whether to finalize the timeline afterward.
    public func rebuild(
        finalizedPredictions: [Float],
        tentativePredictions: [Float],
        keepingSpeakers: Bool = false,
        isComplete: Bool = true
    ) throws {
        try verifyPredictionCounts(finalized: finalizedPredictions, tentative: tentativePredictions)

        queue.sync(flags: .barrier) {
            resetLocked(keepingSpeakers: keepingSpeakers)
            _finalizedPredictions = finalizedPredictions
            _tentativePredictions = tentativePredictions

            let numFinalizedFrames = finalizedPredictions.count / speakerCapacity
            let numTentativeFrames = tentativePredictions.count / speakerCapacity

            updateSegments(
                predictions: finalizedPredictions,
                numFrames: numFinalizedFrames,
                isFinalized: true,
                addTrailingTentative: false
            )

            _numFinalizedFrames = numFinalizedFrames

            updateSegments(
                predictions: tentativePredictions,
                numFrames: numTentativeFrames,
                isFinalized: false,
                addTrailingTentative: true
            )
            if isComplete {
                finalizeLocked()
            } else {
                trimPredictions()
            }
        }
    }

    // MARK: Speaker Management

    /// Add a speaker to the timeline at a given slot, or update their name if one already exists
    /// - Parameters:
    ///   - name: The speaker's name
    ///   - index: The diarizer index of the speaker. If left as `nil`, the first unused index will be chosen.
    /// - Returns: The upserted speaker if created successfully
    @discardableResult
    public func upsertSpeaker(
        named name: String? = nil,
        atIndex index: Int? = nil
    ) -> DiarizerSpeaker? {
        queue.sync(flags: .barrier) {
            let index = index ?? (0..<speakerCapacity).first { _speakers[$0] == nil }

            // Ensure index is within bounds
            guard let index, index >= 0, index < speakerCapacity else { return nil }

            if let speaker = _speakers[index] {
                // Update old speaker
                speaker.name = name
                return speaker
            }

            // New speaker
            let speaker = DiarizerSpeaker(index: index, name: name)
            _speakers[index] = speaker
            return speaker
        }
    }

    /// Add a speaker to the timeline at a given slot, or replace the old one if it's already occupied
    /// - Parameters:
    ///   - speaker: The new speaker to put in the slot.
    ///   - index: The diarizer index of the speaker. If left as `nil`, the first unused index will be chosen.
    ///   - transferCurrentSegment: Whether the current segment should be moved from the old speaker to the new speaker
    /// - Returns: The upserted speaker if created successfully
    @discardableResult
    public func upsertSpeaker(
        _ speaker: DiarizerSpeaker,
        atIndex index: Int? = nil,
        transferCurrentSegment: Bool = true
    ) -> DiarizerSpeaker? {
        queue.sync(flags: .barrier) {
            // Ensure index is within bounds
            let index = index ?? (0..<speakerCapacity).first { _speakers[$0] == nil }

            guard let index, index >= 0, index < speakerCapacity else {
                return nil
            }

            if transferCurrentSegment,
                states[index].isSpeaking,
                let oldSpeaker = _speakers[index],
                let oldStartFrame = oldSpeaker.lastSegment?.startFrame,
                oldStartFrame >= states[index].startFrame,
                let segment = oldSpeaker.popLast()
            {
                speaker.append(segment)
            }

            if transferCurrentSegment {
                states[index] = StreamingState()
            }

            _speakers[index] = speaker
            speaker.index = index

            return speaker
        }
    }

    /// Remove speaker at a given index
    /// - Parameters:
    ///   - index: Speaker index to remove in diarizer output.
    ///   - clearCurrentSegment: Whether to clear the current segment if the speaker was still talking.
    /// - Returns: The removed speaker.
    @discardableResult
    public func removeSpeaker(
        atIndex index: Int,
        clearCurrentSegment: Bool = false
    ) -> DiarizerSpeaker? {
        guard index >= 0, index < speakerCapacity else {
            return nil
        }

        return queue.sync(flags: .barrier) {
            if clearCurrentSegment {
                states[index] = StreamingState()
            }

            return _speakers.removeValue(forKey: index)
        }
    }

    // MARK: - Query

    /// Get probability for a specific speaker at a finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        queue.sync {
            let frameOffset = (frame - _numFinalizedFrames) * speakerCapacity + _finalizedPredictions.count
            guard frameOffset >= 0,
                frameOffset < _finalizedPredictions.count,
                speaker < speakerCapacity
            else { return .nan }
            return _finalizedPredictions[frameOffset + speaker]
        }
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        queue.sync {
            let frameOffset = (frame - _numFinalizedFrames) * speakerCapacity
            guard frameOffset >= 0,
                frameOffset < _tentativePredictions.count,
                speaker < speakerCapacity
            else { return .nan }
            return _tentativePredictions[frameOffset + speaker]
        }
    }

    // MARK: - Segment Detection

    private func updateSegments(
        predictions: [Float],
        numFrames: Int,
        isFinalized: Bool,
        addTrailingTentative: Bool
    ) {
        guard numFrames > 0 || addTrailingTentative else { return }

        let frameOffset = _numFinalizedFrames
        let numSpeakers = speakerCapacity
        let onset = config.onsetThreshold
        let offset = config.offsetThreshold
        let padOnset = config.onsetPadFrames
        let padOffset = config.offsetPadFrames
        let minFramesOn = config.minFramesOn
        let minFramesOff = config.minFramesOff
        let frameDuration = config.frameDurationSeconds

        let tentativeBuffer = padOnset + padOffset + minFramesOff
        let tentativeStartFrame = isFinalized ? (frameOffset + numFrames) - tentativeBuffer : 0

        for speakerIndex in 0..<numSpeakers {
            let state = states[speakerIndex]

            var start = state.startFrame
            var speaking = state.isSpeaking
            var activitySum = state.activitySum
            var activeFrameCount = state.activeFrameCount
            var lastSegment = state.lastSegment
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = speakerIndex + i * numSpeakers
                let activity = predictions[index]

                if speaking {
                    if activity >= offset {
                        activitySum += activity
                        activeFrameCount += 1
                        continue
                    }

                    speaking = false
                    let end = frameOffset + i + padOffset

                    guard end - start > minFramesOn else {
                        activitySum = 0
                        activeFrameCount = 0
                        continue
                    }

                    wasLastSegmentFinal = isFinalized && (end < tentativeStartFrame)
                    let confidence = activeFrameCount > 0 ? (activitySum / Float(activeFrameCount)) : 0

                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: frameDuration,
                        confidence: confidence
                    )

                    provideSpeaker(forSlot: speakerIndex).append(newSegment)

                    lastSegment = ClosedSegmentStats(
                        start: start,
                        end: end,
                        activitySum: activitySum,
                        activeFrameCount: activeFrameCount
                    )
                    activitySum = 0
                    activeFrameCount = 0

                } else if activity > onset {
                    start = max(0, frameOffset + i - padOnset)
                    speaking = true
                    activitySum = activity
                    activeFrameCount = 1

                    if let lastSegment, start - lastSegment.end <= minFramesOff {
                        start = lastSegment.start
                        activitySum += lastSegment.activitySum
                        activeFrameCount += lastSegment.activeFrameCount
                        _speakers[speakerIndex]?.popLast(fromFinalized: wasLastSegmentFinal)
                    }
                }
            }

            if isFinalized {
                states[speakerIndex].startFrame = start
                states[speakerIndex].isSpeaking = speaking
                states[speakerIndex].activitySum = activitySum
                states[speakerIndex].activeFrameCount = activeFrameCount
                states[speakerIndex].lastSegment = lastSegment
            }

            if addTrailingTentative {
                let end = frameOffset + numFrames + padOffset
                if speaking && (end > start) {
                    let confidence = activeFrameCount > 0 ? (activitySum / Float(activeFrameCount)) : 0
                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: frameDuration,
                        confidence: confidence
                    )
                    provideSpeaker(forSlot: speakerIndex).appendTentative(newSegment)
                }
            }
        }
    }

    private func provideSpeaker(forSlot speakerIndex: Int) -> DiarizerSpeaker {
        if let speaker = _speakers[speakerIndex] { return speaker }

        let newSpeaker = DiarizerSpeaker(index: speakerIndex)
        _speakers[speakerIndex] = newSpeaker
        return newSpeaker
    }

    private func trimPredictions() {
        guard let maxStoredFrames = config.maxStoredFrames else { return }
        let numToRemove = _finalizedPredictions.count - maxStoredFrames * speakerCapacity
        if numToRemove > 0 {
            _finalizedPredictions.removeFirst(numToRemove)
        }
    }

    private func verifyPredictionCounts(finalized: borrowing [Float], tentative: borrowing [Float]) throws {
        guard finalized.count.isMultiple(of: speakerCapacity) else {
            throw DiarizerTimelineError.misalignedFinalizedPredictions(finalized.count, speakerCapacity)
        }

        guard tentative.count.isMultiple(of: speakerCapacity) else {
            throw DiarizerTimelineError.misalignedTentativePredictions(tentative.count, speakerCapacity)
        }
    }
}

// MARK: - Timeline Update

public struct DiarizerTimelineUpdate: Sendable {
    public let finalizedSegments: [DiarizerSegment]
    public let tentativeSegments: [DiarizerSegment]
    public let chunkResult: DiarizerChunkResult

    public init(
        finalizedSegments: [DiarizerSegment] = [],
        tentativeSegments: [DiarizerSegment] = [],
        chunkResult: DiarizerChunkResult
    ) {
        self.chunkResult = chunkResult
        self.finalizedSegments = finalizedSegments
        self.tentativeSegments = tentativeSegments
    }
}

public enum DiarizerTimelineError: Error, LocalizedError {
    case misalignedFinalizedPredictions(Int, Int)
    case misalignedTentativePredictions(Int, Int)

    public var errorDescription: String? {
        switch self {
        case .misalignedFinalizedPredictions(let numPreds, let numSpeakers):
            return
                ("The number of finalized predictions (\(numPreds)) isn't a "
                + "multiple of the speaker count (\(numSpeakers)).")
        case .misalignedTentativePredictions(let numPreds, let numSpeakers):
            return
                ("The number of tentative predictions (\(numPreds)) isn't a "
                + "multiple of the speaker count (\(numSpeakers)).")
        }
    }
}
