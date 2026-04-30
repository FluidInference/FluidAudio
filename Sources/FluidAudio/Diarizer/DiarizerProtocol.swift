import Foundation

// MARK: - Diarizer Protocol

/// Protocol for frame-based end-to-end neural diarization pipelines.
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

    /// Finalize the current streaming session and extracts finalized predictions for the tail.
    /// - Returns: The last finalized chunk emitted during finalization, if any.
    @discardableResult
    func finalizeSession() throws -> DiarizerTimelineUpdate?
}
