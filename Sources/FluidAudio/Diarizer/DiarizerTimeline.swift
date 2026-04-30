import Foundation

// MARK: - Timeline Configuration

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

    /// Value used to measure speech activity (sigmoids or logits)
    public var activityType: DiarizerActivityType

    /// When false, committed segments are only delivered via the per-chunk
    /// `DiarizerTimelineUpdate` and are not persisted on `DiarizerSpeaker`,
    /// nor do they cause entries to be inserted into the speakers map. The
    /// caller becomes the sole owner of segment history.
    public var storeSegments: Bool

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
            minFramesOff: 0,
            activityType: .sigmoids
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
    ///   - storeSegments: When false, segments are only emitted via DiarizerTimelineUpdate and not persisted on speakers
    public init(
        numSpeakers: Int? = nil,
        frameDurationSeconds: Float? = nil,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        activityType: DiarizerActivityType = .sigmoids,
        maxStoredFrames: Int? = nil,
        storeSegments: Bool = true
    ) {
        self.numSpeakers = numSpeakers ?? 1
        self.frameDurationSeconds = frameDurationSeconds ?? 0.08
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = onsetPadFrames
        self.offsetPadFrames = offsetPadFrames
        self.minFramesOn = minFramesOn
        self.minFramesOff = minFramesOff
        self.activityType = activityType
        self.maxStoredFrames = maxStoredFrames
        self.storeSegments = storeSegments
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
    ///   - storeSegments: When false, segments are only emitted via DiarizerTimelineUpdate and not persisted on speakers
    public init(
        numSpeakers: Int? = nil,
        frameDurationSeconds: Float? = nil,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadSeconds: Float,
        offsetPadSeconds: Float,
        minDurationOn: Float,
        minDurationOff: Float,
        activityType: DiarizerActivityType = .sigmoids,
        maxStoredFrames: Int? = nil,
        storeSegments: Bool = true
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
        self.activityType = activityType
        self.storeSegments = storeSegments
    }
}

// MARK: - Chunk Result

/// Result from a single streaming diarization step (works with any diarizer).
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
        startFrame: Int = 0,
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

// MARK: - Speaker

public class DiarizerSpeaker: Identifiable {
    public struct Snapshot {
        public let name: String?
        public let index: Int
        public let finalizedSegments: [DiarizerSegment]
        public let tentativeSegments: [DiarizerSegment]
    }

    /// Serializes mutation of this speaker's segment arrays, index, and name.
    private let lock = NSLock()

    /// Speaker ID
    public let id: UUID

    /// Display name
    public var name: String?

    /// Diarizer output slot
    public var index: Int

    /// Finalized speech segments
    public var finalizedSegments: [DiarizerSegment] {
        get { lock.withLock { _finalizedSegments } }
        set { lock.withLock { _finalizedSegments = newValue } }
    }

    /// Preview speech segments
    public var tentativeSegments: [DiarizerSegment] {
        get { lock.withLock { _tentativeSegments } }
        set { lock.withLock { _tentativeSegments = newValue } }
    }

    /// Speaker's string representation
    public var description: String {
        lock.withLock { name ?? "Speaker \(index)" }
    }

    /// Whether this speaker has any segments
    public var hasSegments: Bool {
        lock.withLock { !(_tentativeSegments.isEmpty && _finalizedSegments.isEmpty) }
    }

    /// Number of segments (finalized + tentative)
    public var segmentCount: Int {
        lock.withLock { _tentativeSegments.count + _finalizedSegments.count }
    }

    /// Number of confirmed segments
    public var finalizedSegmentCount: Int {
        lock.withLock { _finalizedSegments.count }
    }

    /// Number of tentative segments
    public var tentativeSegmentCount: Int {
        lock.withLock { _tentativeSegments.count }
    }

    /// Last segment (tentative or finalized). Checks tentative segments first, falls back to finalized if none found.
    public var lastSegment: DiarizerSegment? {
        lock.withLock { _tentativeSegments.last ?? _finalizedSegments.last }
    }

    /// Total duration of segments in seconds (finalized + tentative)
    public var speechDuration: Float {
        lock.withLock {
            _finalizedSegments.reduce(0) { $0 + $1.duration } + _tentativeSegments.reduce(0) { $0 + $1.duration }
        }
    }

    /// Duration of all finalized segments in seconds
    public var finalizedSpeechDuration: Float {
        lock.withLock { _finalizedSegments.reduce(0) { $0 + $1.duration } }
    }

    /// Duration of all tentative segments in seconds
    public var tentativeSpeechDuration: Float {
        lock.withLock { _tentativeSegments.reduce(0) { $0 + $1.duration } }
    }

    /// Total number of frames spanned by all segments (finalized + tentative)
    public var numSpeechFrames: Int {
        lock.withLock {
            _finalizedSegments.reduce(0) { $0 + $1.length } + _tentativeSegments.reduce(0) { $0 + $1.length }
        }
    }

    /// Number of frames in all finalized segments
    public var numFinalizedSpeechFrames: Int {
        lock.withLock { _finalizedSegments.reduce(0) { $0 + $1.length } }
    }

    /// Number of frames in all tentative segments
    public var numTentativeSpeechFrames: Int {
        lock.withLock { _tentativeSegments.reduce(0) { $0 + $1.length } }
    }

    /// Finalized speech segments
    private var _finalizedSegments: [DiarizerSegment] = []

    /// Preview speech segments
    private var _tentativeSegments: [DiarizerSegment] = []

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
        self.index = index
        self.name = name
    }

    /// Initialize from a snapshot of a diarizer speaker
    public init(from snapshot: consuming Snapshot) {
        self.id = UUID()
        self.index = snapshot.index
        self.name = snapshot.name
        self._finalizedSegments = snapshot.finalizedSegments
        self._tentativeSegments = snapshot.tentativeSegments
    }

    /// Rename the speaker
    public func rename(to name: String?) {
        lock.lock()
        defer { lock.unlock() }
        self.name = name
    }

    /// Reassign diarizer output slot
    public func reassign(toSlot slot: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.index = slot
    }

    /// Finalize all segments
    /// - Parameter minFramesOn: Minimum segment length
    public func finalize() {
        lock.lock()
        defer { lock.unlock() }
        _finalizedSegments.append(contentsOf: _tentativeSegments)
        _tentativeSegments.removeAll()
    }

    /// Clear segments
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        _tentativeSegments.removeAll()
        _finalizedSegments.removeAll()
    }

    public func rollback(to snapshot: consuming Snapshot, keepingName: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        if !keepingName { self.name = snapshot.name }
        self.index = snapshot.index
        self._finalizedSegments = snapshot.finalizedSegments
        self._tentativeSegments = snapshot.tentativeSegments
    }

    public func takeSnapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            name: name,
            index: index,
            finalizedSegments: _finalizedSegments,
            tentativeSegments: _tentativeSegments
        )
    }

    /// Clear all tentative segments
    /// - Parameter keepingCapacity: Whether to keep the reserved capacity in the tentative segments list.
    public func clearTentative(keepingCapacity: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        _tentativeSegments.removeAll(keepingCapacity: keepingCapacity)
    }

    /// Append a tentative segment
    /// - Parameter segment: The segment to append
    public func appendTentative(_ segment: DiarizerSegment) {
        lock.lock()
        defer { lock.unlock() }
        _tentativeSegments.append(segment)
    }

    /// Append a finalized segment
    /// - Parameter segment: The segment to append
    public func appendFinalized(_ segment: DiarizerSegment) {
        lock.lock()
        defer { lock.unlock() }
        _finalizedSegments.append(segment)
    }

    /// Append a segment, automatically detecting if it's finalized or tentative
    /// - Parameter segment: The segment to append
    public func append(_ segment: DiarizerSegment) {
        lock.lock()
        defer { lock.unlock() }
        if segment.isFinalized {
            _finalizedSegments.append(segment)
        } else {
            _tentativeSegments.append(segment)
        }
    }

    /// Pop last tentative segment
    /// - Returns: The popped segment
    @discardableResult
    public func popLastTentative() -> DiarizerSegment? {
        lock.lock()
        defer { lock.unlock() }
        return _tentativeSegments.popLast()
    }

    /// Pop last finalized segment
    /// - Returns: The popped segment
    @discardableResult
    public func popLastFinalized() -> DiarizerSegment? {
        lock.lock()
        defer { lock.unlock() }
        return _finalizedSegments.popLast()
    }

    /// Pop last tentative or finalized segment
    /// - Parameter fromFinalized: Whether to pop the segment from the finalized segment list
    /// - Returns: The popped segment
    @discardableResult
    public func popLast(fromFinalized: Bool) -> DiarizerSegment? {
        lock.lock()
        defer { lock.unlock() }
        return
            (fromFinalized
            ? _finalizedSegments.popLast()
            : _tentativeSegments.popLast())
    }

    /// Pop last segment. Pops the last tentative segment first. Falls back to the last finalized segment if no
    /// tentative segments are found.
    /// - Returns: The popped segment
    @discardableResult
    public func popLast() -> DiarizerSegment? {
        lock.lock()
        defer { lock.unlock() }
        return _tentativeSegments.popLast() ?? _finalizedSegments.popLast()
    }

    /// Pop last segment. Pops the last tentative segment first. Falls back to the last finalized segment if no
    /// tentative segments are found.
    /// - Returns: The popped segment
    @discardableResult
    public func popLast(
        if predicate: @Sendable (DiarizerSegment) throws -> Bool
    ) rethrows -> DiarizerSegment? {
        lock.lock()
        defer { lock.unlock() }
        let last = _tentativeSegments.last ?? _finalizedSegments.last
        guard let last, try predicate(last) else {
            return nil
        }
        return _tentativeSegments.popLast() ?? _finalizedSegments.popLast()
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

    /// Average speech activity in the segment
    public var activity: Float = 0.0

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
        activity: Float = 0
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
        self.activity = activity
    }

    public init(
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float,
        activity: Float = 0
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
        self.activity = activity
    }

    /// Check if this overlaps with another segment
    public func overlaps(with other: DiarizerSegment) -> Bool {
        (startFrame <= other.endFrame) && (other.startFrame <= endFrame)
    }

    /// Merge another segment into this one
    public mutating func absorb(_ other: DiarizerSegment) {
        let lengthFloat = Float(length)
        let otherLengthFloat = Float(other.length)
        let totalLength = lengthFloat + otherLengthFloat
        activity =
            totalLength > 0
            ? (lengthFloat * activity + otherLengthFloat * other.activity) / totalLength
            : 0
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

// MARK: - Activity Type

/// Methods to measure speech activity for segment activity
public enum DiarizerActivityType: Sendable {
    case sigmoids
    case logits

    /// A closure that maps speech probabilities to the desired activity type
    public var evaluationFunction: (Float) -> Float {
        switch self {
        case .sigmoids:
            return { (p: Float) -> Float in return p }
        case .logits:
            return { (p: Float) -> Float in
                let eps: Float = 1e-6
                let clamped = min(max(p, eps), 1 - eps)
                return log(clamped / (1 - clamped))
            }
        }
    }
}

// MARK: - Timeline

/// Complete diarization timeline managing streaming predictions and segments.
///
/// Generalizes `SortformerTimeline` for any frame-based diarizer. Works with
/// both Sortformer (fixed 4 speakers) and LS-EEND (variable speaker count).
public class DiarizerTimeline {
    public struct ConfiguredSnapshot {
        let config: DiarizerTimelineConfig
        let snapshot: Snapshot
    }

    public struct Snapshot {
        public let speakers: [Int: DiarizerSpeaker.Snapshot]
        public let finalizedPredictions: [Float]
        public let tentativePredictions: [Float]
        public let numFinalizedFrames: Int
        internal let scratches: [SegmentScratch]
    }

    public enum KeptOnReset {
        case nothing
        case namedSpeakers
        case allSpeakers
        case speakersWithoutSegments
        case speakersWithSegments
    }

    internal struct SegmentScratch {
        var speaking: Bool = false
        var hasSegment: Bool = false
        var startFrame: Int = .min
        var endFrame: Int = .min
        var activitySum: Float = 0
        var activeFrameCount: Int = 0
    }

    /// Serializes mutation of `speakers`, `scratches`, prediction buffers, and
    /// finalized cursor across threads. NSLock is non-recursive, so public
    /// mutating entry points acquire the lock once and delegate to private
    /// `_unlocked` helpers when they need to call other mutating logic.
    private let lock = NSLock()

    /// Post-processing configuration
    public let config: DiarizerTimelineConfig

    /// Finalized frame-wise speaker predictions.
    /// Flat array of shape [numFrames, numSpeakers].
    public var finalizedPredictions: [Float] {
        lock.withLock { _finalizedPredictions }
    }

    /// Tentative predictions.
    /// Flat array of shape [numTentative, numSpeakers].
    public var tentativePredictions: [Float] {
        lock.withLock { _tentativePredictions }
    }

    /// Total number of finalized frames
    public var numFinalizedFrames: Int {
        lock.withLock { finalizedCursorFrame }
    }

    /// Number of tentative frames
    public var numTentativeFrames: Int {
        lock.withLock { _tentativePredictions.count / speakerCapacity }
    }

    /// Total number of frames (finalized + tentative)
    public var numFrames: Int {
        lock.withLock { finalizedCursorFrame + _tentativePredictions.count / speakerCapacity }
    }

    /// Speakers in the timeline
    public var speakers: [Int: DiarizerSpeaker] {
        lock.withLock { _speakers }
    }

    /// Whether the timeline has any segments
    public var hasSegments: Bool {
        lock.withLock { _speakers.values.contains(where: \.hasSegments) }
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

    private var _speakers: [Int: DiarizerSpeaker]
    private var _finalizedPredictions: [Float] = []
    private var _tentativePredictions: [Float] = []
    private var finalizedCursorFrame: Int = 0
    private var scratches: [SegmentScratch]
    private static let logger = AppLogger(category: "DiarizerTimeline")

    // MARK: - Init

    /// Initialize for streaming usage
    public init(config: DiarizerTimelineConfig) {
        self.config = config
        self._speakers = [:]
        self.scratches = Array(repeating: .init(), count: config.numSpeakers)
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

    /// Initialize from a snapshot
    public init(from snapshot: consuming Snapshot, withConfig config: DiarizerTimelineConfig) {
        self.config = config
        self._finalizedPredictions = snapshot.finalizedPredictions
        self._tentativePredictions = snapshot.tentativePredictions
        self.finalizedCursorFrame = snapshot.numFinalizedFrames
        self.scratches = snapshot.scratches
        self._speakers = [:]
        self._speakers.reserveCapacity(snapshot.speakers.count)

        for (slot, speakerSnapshot) in snapshot.speakers {
            self._speakers[slot] = DiarizerSpeaker(from: speakerSnapshot)
        }
    }

    /// Initialize from a snapshot
    public convenience init(from snapshot: consuming ConfiguredSnapshot) {
        self.init(from: snapshot.snapshot, withConfig: snapshot.config)
    }

    // MARK: - Add Predictions to Timeline

    /// Add new predictions from the diarizer
    @discardableResult
    public func addPredictions(
        finalizedPredictions: [Float],
        tentativePredictions: [Float]
    ) throws -> DiarizerTimelineUpdate {
        lock.lock()
        defer { lock.unlock() }
        let finalizedCount = finalizedPredictions.count / speakerCapacity
        let tentativeCount = tentativePredictions.count / speakerCapacity
        let chunk = DiarizerChunkResult(
            startFrame: finalizedCursorFrame,
            finalizedPredictions: finalizedPredictions,
            finalizedFrameCount: finalizedCount,
            tentativePredictions: tentativePredictions,
            tentativeFrameCount: tentativeCount
        )
        return try _addChunkUnlocked(consume chunk)
    }

    /// Add a new chunk of predictions from the diarizer
    @discardableResult
    public func addChunk(_ chunk: DiarizerChunkResult) throws -> DiarizerTimelineUpdate {
        lock.lock()
        defer { lock.unlock() }
        return try _addChunkUnlocked(chunk)
    }

    private func _addChunkUnlocked(_ chunk: DiarizerChunkResult) throws -> DiarizerTimelineUpdate {
        try verifyPredictionCounts(
            finalized: chunk.finalizedPredictions,
            tentative: chunk.tentativePredictions
        )

        // Update predictions
        if config.maxStoredFrames != 0 {
            _finalizedPredictions.append(contentsOf: chunk.finalizedPredictions)
            trimPredictions()
        }
        _tentativePredictions = chunk.tentativePredictions

        // Clear tentative segments
        for speaker in _speakers.values {
            speaker.clearTentative(keepingCapacity: true)
        }

        // Extract new segments
        var newFinalized: [DiarizerSegment] = []
        var newTentative: [DiarizerSegment] = []

        updateSegments(
            predictions: chunk.finalizedPredictions,
            isFinalized: true,
            addTrailingTentative: false,
            emittingFinalizedTo: &newFinalized,
            emittingTentativeTo: &newTentative
        )

        finalizedCursorFrame += chunk.finalizedFrameCount

        updateSegments(
            predictions: chunk.tentativePredictions,
            isFinalized: false,
            addTrailingTentative: true,
            emittingFinalizedTo: &newFinalized,
            emittingTentativeTo: &newTentative
        )

        return DiarizerTimelineUpdate(
            finalizedSegments: consume newFinalized,
            tentativeSegments: consume newTentative,
            chunkResult: consume chunk
        )
    }

    // MARK: - Finalize Timeline

    /// Finalize all tentative data at end of recording
    public func finalize() {
        lock.lock()
        defer { lock.unlock() }
        _finalizeUnlocked()
    }

    private func _finalizeUnlocked() {
        _finalizedPredictions.append(contentsOf: _tentativePredictions)
        finalizedCursorFrame += _tentativePredictions.count / speakerCapacity
        _tentativePredictions.removeAll()
        for speaker in _speakers.values {
            speaker.finalize()
        }
        trimPredictions()
    }

    // MARK: - Reset Timeline

    /// Reset to initial state
    /// - Parameter condition: Condition for when to keep a speaker. All speakers still have their segments reset.
    public func reset(
        keepingSpeakersWhere condition: (DiarizerSpeaker) -> Bool
    ) {
        lock.lock()
        defer { lock.unlock() }
        _finalizedPredictions.removeAll()
        _tentativePredictions.removeAll()
        finalizedCursorFrame = 0
        scratches = Array(repeating: .init(), count: speakerCapacity)

        _speakers = _speakers.filter { _, speaker in condition(speaker) }
        for speaker in _speakers.values {
            speaker.reset()
        }
    }

    /// Reset to initial state
    /// - Parameter keepingSpeakers: Whether to keep existing speakers enrolled. Their segments are still reset.
    public func reset(keepingSpeakers: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        _resetUnlocked(keepingSpeakers: keepingSpeakers)
    }

    private func _resetUnlocked(keepingSpeakers: Bool) {
        _finalizedPredictions.removeAll()
        _tentativePredictions.removeAll()
        finalizedCursorFrame = 0
        scratches = Array(repeating: .init(), count: speakerCapacity)

        if keepingSpeakers {
            for speaker in _speakers.values {
                speaker.reset()
            }
        } else {
            _speakers.removeAll(keepingCapacity: true)
        }
    }

    // MARK: - Rebuild Timeline

    /// Rebuild the timeline from initial predictions. This is equivalent to reinitializing the timeline.
    /// - Parameters:
    ///   - finalizedPredictions: Finalized prediction matrix `[numFrames, numSpeakers]` flattened
    ///   - tentativePredictions: Tentative prediction matrix `[numFrames, numSpeakers]` flattened
    ///   - keepingSpeakers: Whether to keep the old speaker names and slots.
    ///   - isComplete: Whether to finalize the timeline afterward.
    @discardableResult
    public func rebuild(
        finalizedPredictions: [Float],
        tentativePredictions: [Float],
        keepingSpeakers: Bool = false,
        isComplete: Bool = true
    ) throws -> DiarizerTimelineUpdate {
        lock.lock()
        defer { lock.unlock() }

        try verifyPredictionCounts(
            finalized: finalizedPredictions,
            tentative: tentativePredictions
        )

        var newFinalized: [DiarizerSegment] = []
        var newTentative: [DiarizerSegment] = []

        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: finalizedPredictions,
            finalizedFrameCount: finalizedPredictions.count / speakerCapacity,
            tentativePredictions: tentativePredictions,
            tentativeFrameCount: tentativePredictions.count / speakerCapacity
        )

        _resetUnlocked(keepingSpeakers: keepingSpeakers)
        self._finalizedPredictions = finalizedPredictions
        self._tentativePredictions = tentativePredictions

        updateSegments(
            predictions: finalizedPredictions,
            isFinalized: true,
            addTrailingTentative: false,
            emittingFinalizedTo: &newFinalized,
            emittingTentativeTo: &newTentative
        )

        finalizedCursorFrame = finalizedPredictions.count / speakerCapacity

        updateSegments(
            predictions: tentativePredictions,
            isFinalized: false,
            addTrailingTentative: true,
            emittingFinalizedTo: &newFinalized,
            emittingTentativeTo: &newTentative
        )

        if isComplete {
            _finalizeUnlocked()
        } else {
            trimPredictions()
        }

        return DiarizerTimelineUpdate(
            finalizedSegments: consume newFinalized,
            tentativeSegments: consume newTentative,
            chunkResult: consume chunk
        )
    }

    // MARK: - Timeline Snapshots

    public func rollback(to snapshot: consuming Snapshot, keepingSpeakers: Bool = false) {
        lock.lock()
        defer { lock.unlock() }
        self._finalizedPredictions = snapshot.finalizedPredictions
        self._tentativePredictions = snapshot.tentativePredictions
        self.finalizedCursorFrame = snapshot.numFinalizedFrames
        self.scratches = snapshot.scratches

        for (slot, speakerSnapshot) in snapshot.speakers {
            _speakers[slot]?.rollback(to: speakerSnapshot, keepingName: keepingSpeakers)
        }

        guard !keepingSpeakers else { return }
        _speakers = _speakers.filter { slot, _ in snapshot.speakers[slot] != nil }
    }

    public func takeSnapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        var speakersSnapshots: [Int: DiarizerSpeaker.Snapshot] = [:]
        for (slot, speaker) in _speakers {
            speakersSnapshots[slot] = speaker.takeSnapshot()
        }

        return Snapshot(
            speakers: speakersSnapshots,
            finalizedPredictions: _finalizedPredictions,
            tentativePredictions: _tentativePredictions,
            numFinalizedFrames: finalizedCursorFrame,
            scratches: scratches
        )
    }

    // MARK: - Timeline Speaker Management

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
        lock.lock()
        defer { lock.unlock() }
        let index = index ?? (0..<speakerCapacity).first { _speakers[$0] == nil }

        // Ensure index is within bounds
        guard let index, index >= 0, index < speakerCapacity else { return nil }

        if let speaker = _speakers[index] {
            // Update old speaker
            speaker.rename(to: name)
            return speaker
        }

        // New speaker
        let speaker = DiarizerSpeaker(index: index, name: name)
        _speakers[index] = speaker
        return speaker
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
        lock.lock()
        defer { lock.unlock() }
        // Ensure index is within bounds
        let index = index ?? (0..<speakerCapacity).first { _speakers[$0] == nil }

        guard let index, index >= 0, index < speakerCapacity else {
            return nil
        }

        if transferCurrentSegment,
            scratches[index].speaking,
            let oldSpeaker = _speakers[index],
            let segment = oldSpeaker.popLast(
                if: { [startFrame = scratches[index].startFrame] in
                    $0.startFrame >= startFrame
                })
        {
            speaker.append(segment)
        }

        // Clear current segment if we don't want to transfer it
        if !transferCurrentSegment {
            scratches[index] = SegmentScratch()
        }

        _speakers[index] = speaker
        speaker.reassign(toSlot: index)

        return speaker
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
        lock.lock()
        defer { lock.unlock() }
        guard index >= 0, index < speakerCapacity else {
            return nil
        }
        if clearCurrentSegment {
            scratches[index] = SegmentScratch()
        }

        return _speakers.removeValue(forKey: index)
    }

    // MARK: - Query

    /// Get probability for a specific speaker at a finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        lock.lock()
        defer { lock.unlock() }
        let frameOffset = (frame - finalizedCursorFrame) * speakerCapacity + _finalizedPredictions.count
        guard frameOffset >= 0,
            frameOffset < _finalizedPredictions.count,
            speaker < speakerCapacity
        else { return .nan }
        return _finalizedPredictions[frameOffset + speaker]
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        lock.lock()
        defer { lock.unlock() }
        let frameOffset = (frame - finalizedCursorFrame) * speakerCapacity
        guard frameOffset >= 0,
            frameOffset < _tentativePredictions.count,
            speaker < speakerCapacity
        else { return .nan }
        return _tentativePredictions[frameOffset + speaker]
    }

    // MARK: - Segment Detection

    private func updateSegments(
        predictions: borrowing [Float],
        isFinalized: Bool,
        addTrailingTentative: Bool,
        emittingFinalizedTo finalizedResult: inout [DiarizerSegment],
        emittingTentativeTo tentativeResult: inout [DiarizerSegment]
    ) {
        guard !predictions.isEmpty || addTrailingTentative else { return }

        let frameOffset = finalizedCursorFrame
        let onset = config.onsetThreshold
        let offset = config.offsetThreshold
        let padOnset = config.onsetPadFrames
        let padOffset = config.offsetPadFrames
        let minFramesOn = config.minFramesOn
        let minFramesOff = config.minFramesOff

        let numNewFrames = predictions.count / speakerCapacity
        let endFrame = frameOffset + numNewFrames
        let pad = padOnset + padOffset
        let minSegmentLength = pad + minFramesOn
        let finalizedEndFrame = isFinalized ? endFrame - minFramesOff - pad : .min

        let activityFunc = config.activityType.evaluationFunction

        for speakerIndex in 0..<speakerCapacity {
            var aux = scratches[speakerIndex]

            for i in 0..<numNewFrames {
                let index = i * speakerCapacity + speakerIndex
                let activity = predictions[index]
                let frame = frameOffset + i

                if aux.speaking {
                    if activity >= offset {
                        aux.activitySum += activityFunc(activity)
                        aux.activeFrameCount += 1
                        continue
                    }

                    aux.speaking = false
                    let end = frame + padOffset

                    guard end - aux.startFrame >= minSegmentLength else {
                        continue
                    }

                    aux.endFrame = end
                    aux.hasSegment = true
                } else if activity > onset {
                    let start = frame - padOnset
                    aux.speaking = true

                    guard !aux.hasSegment || start - aux.endFrame > minFramesOff else {
                        aux.activitySum += activityFunc(activity)
                        aux.activeFrameCount += 1
                        aux.hasSegment = false
                        continue
                    }

                    commitSegment(
                        from: &aux,
                        toSlot: speakerIndex,
                        isFinalized: aux.endFrame < finalizedEndFrame,
                        emittingIfFinalizedTo: &finalizedResult,
                        emittingIfTentativeTo: &tentativeResult
                    )

                    aux.startFrame = start
                    aux.activitySum = activityFunc(activity)
                    aux.activeFrameCount = 1
                }
            }

            // Commit final pending segment
            commitSegment(
                from: &aux,
                toSlot: speakerIndex,
                isFinalized: aux.endFrame < finalizedEndFrame,
                emittingIfFinalizedTo: &finalizedResult,
                emittingIfTentativeTo: &tentativeResult
            )

            if isFinalized {
                scratches[speakerIndex] = aux
                continue
            }

            // Add trailing segment (tentative-path only)
            guard addTrailingTentative, aux.speaking else { continue }
            aux.endFrame = endFrame + padOffset
            guard aux.endFrame - aux.startFrame >= minSegmentLength else { continue }
            aux.hasSegment = true

            commitSegment(
                from: &aux,
                toSlot: speakerIndex,
                isFinalized: false,
                emittingIfFinalizedTo: &finalizedResult,
                emittingIfTentativeTo: &tentativeResult
            )
        }
    }

    @inline(__always)
    private func commitSegment(
        from aux: inout SegmentScratch,
        toSlot slot: Int,
        isFinalized: Bool,
        emittingIfFinalizedTo finalizedResult: inout [DiarizerSegment],
        emittingIfTentativeTo tentativeResult: inout [DiarizerSegment]
    ) {
        guard aux.hasSegment else { return }

        let segment = DiarizerSegment(
            speakerIndex: slot,
            startFrame: aux.startFrame,
            endFrame: aux.endFrame,
            finalized: isFinalized,
            frameDurationSeconds: config.frameDurationSeconds,
            activity: aux.activeFrameCount > 0 ? aux.activitySum / Float(aux.activeFrameCount) : 0
        )

        if config.storeSegments {
            let speaker: DiarizerSpeaker
            if let spk = _speakers[slot] {
                speaker = consume spk
            } else {
                let spk = DiarizerSpeaker(index: slot)
                _speakers[slot] = spk
                speaker = consume spk
            }
            speaker.append(segment)
        }

        if isFinalized {
            finalizedResult.append(consume segment)
        } else {
            tentativeResult.append(consume segment)
        }

        aux.hasSegment = false
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
