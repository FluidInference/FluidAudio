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
    
    /// Add audio samples to the processing buffer
    func addAudio(_ samples: [Float])

    /// Process buffered audio and return any new results
    func process() throws -> DiarizerTimelineUpdate?

    /// Add audio and process in one call
    func process(samples: [Float]) throws -> DiarizerTimelineUpdate?

    // MARK: Offline

    /// Process complete audio and return finalized timeline
    func processComplete(
        _ samples: [Float],
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline

    // MARK: Lifecycle

    /// Reset streaming state while keeping model loaded
    func reset()

    /// Clean up all resources
    func cleanup()
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

    public var onsetPadSeconds: Float {
        get { Float(onsetPadFrames) * frameDurationSeconds }
        set { onsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    public var offsetPadSeconds: Float {
        get { Float(offsetPadFrames) * frameDurationSeconds }
        set { offsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    public var minDurationOn: Float {
        get { Float(minFramesOn) * frameDurationSeconds }
        set { minFramesOn = Int(round(newValue / frameDurationSeconds)) }
    }

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
        self.numSpeakers = numSpeakers ?? 0
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
        self.numSpeakers = numSpeakers ?? 0
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

public final class DiarizerSpeaker: @unchecked Sendable, Identifiable, CustomStringConvertible {
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
    
    /// Number of confirmed segments
    public var finalizedSegmentCount: Int {
        queue.sync { _finalizedSegments.count }
    }
    
    /// Number of tentative segments
    public var tentativeSegmentCount: Int {
        queue.sync { _tentativeSegments.count }
    }
    
    private var _name: String?
    private var _index: Int
    private var _finalizedSegments: [DiarizerSegment] = []
    private var _tentativeSegments: [DiarizerSegment] = []
    private let queue = DispatchQueue(label: "FluidAudio.Diarization.DiarizerSpeaker")
    
    public init(
        id: UUID = UUID(),
        index: Int,
        name: String? = nil
    ) {
        self.id = id
        self._index = index
        self._name = name
    }
    
    public func finalize() {
        queue.sync(flags: .barrier) {
            _finalizedSegments.append(contentsOf: _tentativeSegments)
            _tentativeSegments.removeAll()
        }
    }
    
    public func reset() {
        queue.sync(flags: .barrier) {
            _tentativeSegments.removeAll()
            _finalizedSegments.removeAll()
        }
    }
    
    public func removeAllTentative(keepingCapacity: Bool = false) {
        queue.sync(flags: .barrier) {
            _tentativeSegments.removeAll(keepingCapacity: keepingCapacity)
        }
    }
    
    public func appendTentative(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            _tentativeSegments.append(segment)
        }
    }
    
    public func appendFinalized(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            _finalizedSegments.append(segment)
        }
    }
    
    public func append(_ segment: DiarizerSegment) {
        queue.sync(flags: .barrier) {
            if segment.isFinalized {
                _finalizedSegments.append(segment)
            } else {
                _tentativeSegments.append(segment)
            }
        }
    }
    
    @discardableResult
    public func popLastTentative() -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            _tentativeSegments.popLast()
        }
    }
    
    @discardableResult
    public func popLastFinalized() -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            return _finalizedSegments.popLast()
        }
    }
    
    @discardableResult
    public func popLast(fromFinalized: Bool = true) -> DiarizerSegment? {
        queue.sync(flags: .barrier) {
            return (fromFinalized
                    ? _finalizedSegments.popLast()
                    : _tentativeSegments.popLast())
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
        frameDurationSeconds: Float
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    public init(
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
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
public final class DiarizerTimeline: @unchecked Sendable {
    private struct StreamingState {
        var startFrame: Int
        var isSpeaking: Bool
        var lastSegment: (start: Int, end: Int)
        
        init(
            startFrame: Int = 0,
            isSpeaking: Bool = false,
            lastSegment: (start: Int, end: Int) = (-1, -1)
        ) {
            self.startFrame = startFrame
            self.isSpeaking = isSpeaking
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
    
    /// Total number of frames (finalized + tentative)
    @available(*, deprecated,
                message: "`numFrames` now includes tentative frames. Use 'numFinalizedFrames' for only finalized frames.",
                renamed: "numFinalizedFrames")
    public var numFrames: Int {
        queue.sync { _numFinalizedFrames + _tentativePredictions.count / config.numSpeakers }
    }
    
    /// Total number of finalized frames
    public var numFinalizedFrames: Int {
        queue.sync { _numFinalizedFrames }
    }
    
    /// Number of tentative frames
    public var numTentativeFrames: Int {
        queue.sync { _tentativePredictions.count / config.numSpeakers }
    }

    /// Speakers in the timeline
    public var speakers: [Int : DiarizerSpeaker] {
        queue.sync { _speakers }
    }

    /// Duration of all predictions in seconds
    @available(*, deprecated,
                message: "`duration` now includes tentative frames. Use 'finalizedDuration' for only finalized frames.",
                renamed: "finalizedDuration")
    public var duration: Float {
        Float(numFrames) * config.frameDurationSeconds
    }
    
    /// Duration of finalized predictions in seconds
    public var finalizedDuration: Float {
        Float(numFinalizedFrames) * config.frameDurationSeconds
    }

    /// Duration of tentative predictions in seconds
    @available(*, deprecated,
                message: "tentativeDuration now excludes finalized frames. Use 'duration' for the full timeline duration.",
                renamed: "duration")
    public var tentativeDuration: Float {
        Float(numTentativeFrames) * config.frameDurationSeconds
    }

    private var _finalizedPredictions: [Float] = []
    private var _tentativePredictions: [Float] = []
    private var _speakers: [Int : DiarizerSpeaker] = [:]
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
    
    @discardableResult
    public func addChunk(
        finalizedPredictions: [Float],
        tentativePredictions: [Float]
    ) throws -> DiarizerTimelineUpdate {
        let numFinalized = finalizedPredictions.count / config.numSpeakers
        let numTentative = tentativePredictions.count / config.numSpeakers
        
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
            
            let confirmedCounts = Dictionary(uniqueKeysWithValues: _speakers.map { (index, speaker) in
                (index, speaker.finalizedSegmentCount)
            })
            
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
                speaker.finalizedSegments.suffix(from: confirmedCounts[index, default: 0])
            }
            
            let newTentative = _speakers.flatMap(\.value.tentativeSegments)
            
            return DiarizerTimelineUpdate(
                finalizedSegments: newConfirmed,
                tentativeSegments: newTentative,
                chunkResult: chunk
            )
        }
    }

    /// Finalize all tentative data at end of recording
    public func finalize() {
        queue.sync(flags: .barrier) {
            _finalizedPredictions.append(contentsOf: _tentativePredictions)
            _numFinalizedFrames += _tentativePredictions.count / config.numSpeakers
            _tentativePredictions.removeAll()
            for speaker in _speakers.values {
                speaker.finalize()
            }
            trimPredictions()
        }
    }

    /// Reset to initial state
    public func reset() {
        queue.sync(flags: .barrier) {
            _finalizedPredictions.removeAll()
            _tentativePredictions.removeAll()
            _numFinalizedFrames = 0
            _speakers = [:]
            states = Array(repeating: .init(), count: config.numSpeakers)
        }
    }
    
    public func rebuild(
        finalizedPredictions: [Float],
        tentativePredictions: [Float],
        isComplete: Bool = true
    ) throws {
        try verifyPredictionCounts(finalized: finalizedPredictions, tentative: tentativePredictions)
        
        reset()
        queue.sync(flags: .barrier) {
            _finalizedPredictions = finalizedPredictions
            _tentativePredictions = tentativePredictions
            
            let numFinalizedFrames = finalizedPredictions.count / config.numSpeakers
            let numTentativeFrames = tentativePredictions.count / config.numSpeakers
            
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
        }
        
        if isComplete {
            finalize()
        } else {
            queue.sync(flags: .barrier) {
                trimPredictions()
            }
        }
    }

    // MARK: - Query

    /// Get probability for a specific speaker at a finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        queue.sync {
            let frameOffset = (frame - _numFinalizedFrames) * config.numSpeakers + _finalizedPredictions.count
            guard frameOffset >= 0,
                  frameOffset < _finalizedPredictions.count,
                  speaker < config.numSpeakers
            else { return .nan }
            return _finalizedPredictions[frameOffset + speaker]
        }
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        queue.sync {
            let frameOffset = (frame - _numFinalizedFrames) * config.numSpeakers
            guard frameOffset >= 0,
                  frameOffset < _tentativePredictions.count,
                  speaker < config.numSpeakers
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
        guard numFrames > 0 else { return }

        let frameOffset = _numFinalizedFrames
        let numSpeakers = config.numSpeakers
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
            var lastSegment = state.lastSegment
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = speakerIndex + i * numSpeakers

                if speaking {
                    if predictions[index] >= offset {
                        continue
                    }

                    speaking = false
                    let end = frameOffset + i + padOffset

                    guard end - start > minFramesOn else { continue }

                    wasLastSegmentFinal = isFinalized && (end < tentativeStartFrame)

                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: frameDuration
                    )
                    
                    provideSpeaker(forSlot: speakerIndex).append(newSegment)
                    
                    lastSegment = (start, end)

                } else if predictions[index] > onset {
                    start = max(0, frameOffset + i - padOnset)
                    speaking = true

                    if start - lastSegment.end <= minFramesOff {
                        start = lastSegment.start
                        _speakers[speakerIndex]?.popLast(fromFinalized: wasLastSegmentFinal)
                    }
                }
            }

            if isFinalized {
                states[speakerIndex].startFrame = start
                states[speakerIndex].isSpeaking = speaking
                states[speakerIndex].lastSegment = lastSegment
            }

            if addTrailingTentative {
                let end = frameOffset + numFrames + padOffset
                if speaking && (end > start) {
                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: frameDuration
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
        let numToRemove = _finalizedPredictions.count - maxStoredFrames * config.numSpeakers
        if numToRemove > 0 {
            _finalizedPredictions.removeFirst(numToRemove)
        }
    }

    private func verifyPredictionCounts(finalized: borrowing [Float], tentative: borrowing [Float]) throws {
        guard finalized.count.isMultiple(of: config.numSpeakers) else {
            throw DiarizerTimelineError.misalignedFinalizedPredictions(finalized.count, config.numSpeakers)
        }
        
        guard tentative.count.isMultiple(of: config.numSpeakers) else {
            throw DiarizerTimelineError.misalignedTentativePredictions(tentative.count, config.numSpeakers)
        }
    }
}


extension DiarizerTimeline {
    @available(*, deprecated, renamed: "finalizedPredictions")
    public var framePredictions: [Float] { finalizedPredictions }
    
    @available(*, deprecated, message: "Use Timeline.speakers[index].confirmedSegments to access a speaker's confirmed segments.")
    public var segments: [[DiarizerSegment]] {
        var result: [[DiarizerSegment]] = Array(repeating: [], count: config.numSpeakers)
        for (index, speaker) in _speakers {
            result[index] = speaker.finalizedSegments
        }
        return result
    }
    
    @available(*, deprecated, message: "Use Timeline.speakers[index].tentativeSegments to access a speaker's tentative segments.")
    public var tentativeSegments: [[DiarizerSegment]] {
        var result: [[DiarizerSegment]] = Array(repeating: [], count: config.numSpeakers)
        for (index, speaker) in _speakers {
            result[index] = speaker.tentativeSegments
        }
        return result
    }
    
    @available(*, deprecated, renamed: "numTentativeFrames")
    public var numTentative: Int { numTentativeFrames }
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
        case let .misalignedFinalizedPredictions(numPreds, numSpeakers):
            return ("The number of finalized predictions (\(numPreds)) isn't a " +
                    "multiple of the speaker count (\(numSpeakers)).")
        case let .misalignedTentativePredictions(numPreds, numSpeakers):
            return ("The number of tentative predictions (\(numPreds)) isn't a " +
                    "multiple of the speaker count (\(numSpeakers)).")
        }
    }
}
