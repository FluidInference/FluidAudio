import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager {

    internal let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    internal let config: DiarizerConfig
    private var models: DiarizerModels?

    private let segmentationProcessor = SegmentationProcessor()
    private let speakerClustering: SpeakerClustering
    private var embeddingExtractor: EmbeddingExtractor?
    private let audioValidation = AudioValidation()
    private let memoryOptimizer = ANEMemoryOptimizer.shared

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

        // Initialize EmbeddingExtractor with the embedding model from DiarizerModels
        self.embeddingExtractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)
        logger.info("EmbeddingExtractor initialized with embedding model")

        // Store models after extracting embedding model
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

    public func performCompleteDiarization(
        _ samples: [Float], sampleRate: Int = 16000
    ) throws
        -> DiarizationResult
    {
        guard let models else {
            throw DiarizerError.notInitialized
        }

        var segmentationTime: TimeInterval = 0
        var embeddingTime: TimeInterval = 0
        var clusteringTime: TimeInterval = 0
        var postProcessingTime: TimeInterval = 0

        let chunkSize = sampleRate * 10
        var allSegments: [TimedSpeakerSegment] = []
        var speakerDB: [String: [Float]] = [:]
        var chunkIndex = 0

        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            chunkIndex += 1

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

        let timings = PipelineTimings(
            modelDownloadSeconds: models.downloadDuration,
            modelCompilationSeconds: models.compilationDuration,
            audioLoadingSeconds: 0,
            segmentationSeconds: segmentationTime,
            embeddingExtractionSeconds: embeddingTime,
            speakerClusteringSeconds: clusteringTime,
            postProcessingSeconds: postProcessingTime
        )

        return DiarizationResult(
            segments: filteredSegments, speakerDatabase: speakerDB, timings: timings)
    }

    internal struct ChunkTimings {
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

        // Prepare chunk (same for both paths)
        let chunkSize = sampleRate * 10
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            var padded = Array(repeating: 0.0 as Float, count: chunkSize)
            padded.replaceSubrange(0..<chunk.count, with: chunk)
            paddedChunk = padded[...]
        }

        // Use optimized segmentation with zero-copy
        let (binarizedSegments, _) = try segmentationProcessor.getSegments(
            audioChunk: paddedChunk,
            segmentationModel: models.segmentationModel
        )

        let segmentationTime = Date().timeIntervalSince(segmentationStartTime)

        // Unified and merged models removed - caused performance/stability issues

        // Otherwise use traditional separate model processing
        let slidingFeature = segmentationProcessor.createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        let embeddingStartTime = Date()

        // Use EmbeddingExtractor for embedding extraction
        guard let embeddingExtractor = self.embeddingExtractor else {
            throw DiarizerError.notInitialized
        }

        logger.info("Using EmbeddingExtractor for embedding extraction")

        // Extract masks from sliding window feature
        var masks: [[Float]] = []
        let numSpeakers = slidingFeature.data[0][0].count
        let numFrames = slidingFeature.data[0].count

        for s in 0..<numSpeakers {
            var speakerMask: [Float] = []
            for f in 0..<numFrames {
                // Apply clean frame logic
                let speakerSum = slidingFeature.data[0][f].reduce(0, +)
                let isClean: Float = speakerSum < 2.0 ? 1.0 : 0.0
                speakerMask.append(slidingFeature.data[0][f][s] * isClean)
            }
            masks.append(speakerMask)
        }

        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: Array(paddedChunk),
            masks: masks,
            minActivityThreshold: config.minActivityThreshold
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

        // Calculate speaker activities
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

    private func applyPostProcessingFilters(
        _ segments: [TimedSpeakerSegment]
    )
        -> [TimedSpeakerSegment]
    {
        return segments.filter { segment in
            segment.durationSeconds >= self.config.minDurationOn
        }
    }
}
