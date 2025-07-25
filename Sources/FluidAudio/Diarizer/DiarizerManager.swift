import CoreML
import Foundation
import OSLog


@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    private let config: DiarizerConfig
    private var models: DiarizerModels?
    
    private let segmentationProcessor = SegmentationProcessor()
    private let embeddingExtractor = EmbeddingExtractor()
    private let speakerClustering: SpeakerClustering
    private let audioValidation = AudioValidation()

    public init(config: DiarizerConfig = .default) {
        self.config = config
        self.speakerClustering = SpeakerClustering(config: config)
    }

    public var isAvailable: Bool {
        models != nil
    }

    public var initializationTimings: (downloadTime: TimeInterval, compilationTime: TimeInterval) {
        models.map { ($0.downloadDuration, $0.compilationDuration) } ?? (0, 0)
    }

    public func initialize(models: consuming DiarizerModels) {
        logger.info("Initializing diarization system")
        self.models = consume models
    }

    @available(*, deprecated, message: "Use initialize(models:) instead")
    public func initialize() async throws {
        self.initialize(models: try await .downloadIfNeeded())
    }

    public func cleanup() {
        models = nil
        logger.info("Diarization resources cleaned up")
    }

    public func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float {
        async let result1 = performCompleteDiarization(audio1)
        async let result2 = performCompleteDiarization(audio2)

        guard let segment1 = try await result1.segments.max(by: { $0.qualityScore < $1.qualityScore }),
            let segment2 = try await result2.segments.max(by: { $0.qualityScore < $1.qualityScore })
        else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let distance = speakerClustering.cosineDistance(segment1.embedding, segment2.embedding)
        return max(0, (1.0 - distance) * 100)
    }

    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        return audioValidation.validateEmbedding(embedding)
    }

    public func validateAudio(_ samples: [Float]) -> AudioValidationResult {
        return audioValidation.validateAudio(samples)
    }

    public func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        return speakerClustering.cosineDistance(a, b)
    }

    public func performCompleteDiarization(_ samples: [Float], sampleRate: Int = 16000) throws
        -> DiarizationResult
    {
        guard let models else {
            throw DiarizerError.notInitialized
        }

        let processingStartTime = Date()
        var segmentationTime: TimeInterval = 0
        var embeddingTime: TimeInterval = 0
        var clusteringTime: TimeInterval = 0
        var postProcessingTime: TimeInterval = 0

        let chunkSize = sampleRate * 10
        var allSegments: [TimedSpeakerSegment] = []
        var speakerDB: [String: [Float]] = [:]

        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, samples.count)
            let chunk = samples[chunkStart..<chunkEnd]
            let chunkOffset = Double(chunkStart) / Double(sampleRate)

            let (chunkSegments, chunkTimings) = try processChunkWithSpeakerTracking(
                chunk,
                chunkOffset: chunkOffset,
                speakerDB: &speakerDB,
                models: models,
                sampleRate: sampleRate
            )
            allSegments.append(contentsOf: chunkSegments)

            segmentationTime += chunkTimings.segmentationTime
            embeddingTime += chunkTimings.embeddingTime
            clusteringTime += chunkTimings.clusteringTime
        }

        let postProcessingStartTime = Date()
        let filteredSegments = applyPostProcessingFilters(allSegments)
        postProcessingTime = Date().timeIntervalSince(postProcessingStartTime)

        let totalProcessingTime = Date().timeIntervalSince(processingStartTime)

        let timings = PipelineTimings(
            modelDownloadSeconds: models.downloadDuration,
            modelCompilationSeconds: models.compilationDuration,
            audioLoadingSeconds: 0,
            segmentationSeconds: segmentationTime,
            embeddingExtractionSeconds: embeddingTime,
            speakerClusteringSeconds: clusteringTime,
            postProcessingSeconds: postProcessingTime
        )

        logger.info(
            "Complete diarization finished in \(String(format: "%.2f", totalProcessingTime))s (segmentation: \(String(format: "%.2f", segmentationTime))s, embedding: \(String(format: "%.2f", embeddingTime))s, clustering: \(String(format: "%.2f", clusteringTime))s, post-processing: \(String(format: "%.2f", postProcessingTime))s)"
        )

        return DiarizationResult(
            segments: filteredSegments, speakerDatabase: speakerDB, timings: timings)
    }

    private struct ChunkTimings {
        let segmentationTime: TimeInterval
        let embeddingTime: TimeInterval
        let clusteringTime: TimeInterval
    }

    private func processChunkWithSpeakerTracking(
        _ chunk: ArraySlice<Float>,
        chunkOffset: Double,
        speakerDB: inout [String: [Float]],
        models: DiarizerModels,
        sampleRate: Int = 16000
    ) throws -> ([TimedSpeakerSegment], ChunkTimings) {
        let segmentationStartTime = Date()

        let chunkSize = sampleRate * 10
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            var padded = Array(repeating: 0.0 as Float, count: chunkSize)
            padded.replaceSubrange(0..<chunk.count, with: chunk)
            paddedChunk = padded[...]
        }

        let binarizedSegments = try segmentationProcessor.getSegments(
            audioChunk: paddedChunk,
            segmentationModel: models.segmentationModel
        )
        let slidingFeature = segmentationProcessor.createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        let segmentationTime = Date().timeIntervalSince(segmentationStartTime)
        let embeddingStartTime = Date()

        let embeddings = try embeddingExtractor.getEmbedding(
            audioChunk: paddedChunk,
            binarizedSegments: binarizedSegments,
            slidingWindowFeature: slidingFeature,
            embeddingModel: models.embeddingModel,
            sampleRate: sampleRate
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

        let speakerActivities = speakerClustering.calculateSpeakerActivities(binarizedSegments)

        var speakerLabels: [String] = []
        var activityFilteredCount = 0
        var embeddingInvalidCount = 0
        var clusteringProcessedCount = 0

        for (speakerIndex, activity) in speakerActivities.enumerated() {
            if activity > self.config.minActivityThreshold {
                let embedding = embeddings[speakerIndex]
                if validateEmbedding(embedding) {
                    clusteringProcessedCount += 1
                    let speakerId = speakerClustering.assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
                    speakerLabels.append(speakerId)
                } else {
                    embeddingInvalidCount += 1
                    speakerLabels.append("")
                }
            } else {
                activityFilteredCount += 1
                speakerLabels.append("")
            }
        }

        let clusteringTime = Date().timeIntervalSince(clusteringStartTime)

        let segments = speakerClustering.createTimedSegments(
            binarizedSegments: binarizedSegments,
            slidingWindow: slidingFeature.slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        )

        let timings = ChunkTimings(
            segmentationTime: segmentationTime,
            embeddingTime: embeddingTime,
            clusteringTime: clusteringTime
        )

        return (segments, timings)
    }

    private func applyPostProcessingFilters(_ segments: [TimedSpeakerSegment])
        -> [TimedSpeakerSegment]
    {
        return segments.filter { segment in
            segment.durationSeconds >= self.config.minDurationOn
        }.compactMap { segment in
            return segment
        }
    }
}