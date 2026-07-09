import Foundation

/// Cacheable output of the deterministic first phase of the offline diarization pipeline
/// (segmentation + embedding extraction), produced by `OfflineDiarizerManager.prepare(...)`
/// and consumed by `OfflineDiarizerManager.cluster(_:)`.
///
/// ## Why this exists
/// `OfflineDiarizerManager.process(...)` bundles segmentation → embedding extraction →
/// clustering into one call. Segmentation and embedding extraction are deterministic and
/// dominate the runtime (CoreML inference over the whole audio), while clustering (AHC + VBx)
/// is the only stage whose outcome callers may want to vary or re-run — e.g. multi-run
/// candidate selection with exact speaker-count constraints. Splitting the phases lets such
/// callers pay for inference once and re-run only the clustering stage:
///
/// ```swift
/// let prepared = try await manager.prepare(audio: samples)
/// let runA = try manager.cluster(prepared)
/// let runB = try manager.cluster(prepared)   // no re-segmentation, no re-embedding
/// ```
///
/// `process(...)` remains the single-shot convenience and is exactly
/// `prepare(...)` + `cluster(_:)`.
///
/// ## Contents
/// The stored fields are internal implementation types; the struct is an opaque handle from
/// the caller's perspective. It retains the `AudioSampleSource` because clustering's optional
/// post-passes (zero-vote re-embed, short-segment relabel) re-embed exact audio spans.
///
/// Value semantics + `Sendable`: safe to hold across actor boundaries and reuse for any
/// number of `cluster(_:)` calls.
@available(macOS 14.0, iOS 17.0, *)
public struct PreparedDiarization: Sendable {
    /// Audio the pipeline ran over; needed by clustering post-passes that re-embed spans.
    let audioSource: AudioSampleSource

    /// Deterministic segmentation output (per-chunk speaker activation windows).
    let segmentation: SegmentationOutput

    /// Deterministic per-(chunk, local-speaker) embeddings with PLDA projections.
    let timedEmbeddings: [TimedEmbedding]

    /// Time spent loading/converting the audio, carried into `PipelineTimings`.
    let audioLoadingSeconds: TimeInterval

    /// Wall-clock duration of the segmentation task.
    let segmentationSeconds: TimeInterval

    /// Wall-clock duration of the embedding-extraction task.
    let embeddingExtractionSeconds: TimeInterval

    /// Total wall-clock duration of the whole `prepare(...)` phase (segmentation and
    /// embedding run concurrently, so this is less than the sum of the two stage timings).
    let prepareWallSeconds: TimeInterval

    /// Number of embedding vectors extracted. Zero means `cluster(_:)` will throw
    /// `OfflineDiarizationError.noSpeechDetected`.
    public var embeddingCount: Int { timedEmbeddings.count }

    /// Number of segmentation chunks the audio was windowed into.
    public var segmentationChunkCount: Int { segmentation.numChunks }
}
