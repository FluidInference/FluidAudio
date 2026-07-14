import Foundation

/// Cacheable result of deterministic segmentation and embedding extraction.
///
/// Pass it to `OfflineDiarizerManager.cluster(_:)` to re-run clustering without repeating
/// model inference.
@available(macOS 14.0, iOS 17.0, *)
public struct PreparedDiarization: Sendable {
    let audioSource: AudioSampleSource

    let segmentation: SegmentationOutput

    let timedEmbeddings: [TimedEmbedding]

    let audioLoadingSeconds: TimeInterval

    let segmentationSeconds: TimeInterval

    let embeddingExtractionSeconds: TimeInterval

    let prepareWallSeconds: TimeInterval

    public var embeddingCount: Int { timedEmbeddings.count }

    public var segmentationChunkCount: Int { segmentation.numChunks }
}
