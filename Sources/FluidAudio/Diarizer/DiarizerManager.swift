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
            embeddingPreprocessor: models.embeddingPreprocessor,
            batchFrameExtractor: models.batchFrameExtractor,
            sampleRate: sampleRate
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

        // Use unified model if available
        let speakerActivities: [Float]
        let filteredSegments: [[[Float]]]
        
        if let unifiedModel = models.unifiedPostEmbeddingModel {
            logger.info("🚀 Using unified post-embedding model - GPU acceleration enabled!")
            let (activities, filtered) = try processWithUnifiedModel(
                embeddings: embeddings,
                binarizedSegments: binarizedSegments,
                speakerDB: &speakerDB,
                unifiedModel: unifiedModel
            )
            speakerActivities = activities
            filteredSegments = filtered
        } else {
            // Fallback to CPU processing
            speakerActivities = speakerClustering.calculateSpeakerActivities(binarizedSegments)
            filteredSegments = binarizedSegments
        }

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
    
    private func processWithUnifiedModel(
        embeddings: [[Float]],
        binarizedSegments: [[[Float]]],
        speakerDB: inout [String: [Float]],
        unifiedModel: MLModel
    ) throws -> ([Float], [[[Float]]]) {
        // Prepare inputs for unified model
        let numEmbeddings = embeddings.count
        let embeddingSize = embeddings[0].count
        
        // Flatten embeddings to MLMultiArray
        guard let embeddingsArray = try? MLMultiArray(
            shape: [numEmbeddings, embeddingSize] as [NSNumber],
            dataType: .float32
        ) else {
            throw DiarizerError.processingFailed("Failed to create embeddings array")
        }
        
        for i in 0..<numEmbeddings {
            for j in 0..<embeddingSize {
                embeddingsArray[i * embeddingSize + j] = NSNumber(value: embeddings[i][j])
            }
        }
        
        // Convert speaker database to MLMultiArray
        let speakerDBArray: [Float] = speakerDB.values.flatMap { $0 }
        let numSpeakers = speakerDB.count
        
        // Handle empty speaker DB case - use at least 1 row with zeros
        let dbRows = max(numSpeakers, 1)
        guard let speakerDBMLArray = try? MLMultiArray(
            shape: [dbRows, embeddingSize] as [NSNumber],
            dataType: .float32
        ) else {
            throw DiarizerError.processingFailed("Failed to create speaker DB array")
        }
        
        if numSpeakers > 0 {
            for i in 0..<speakerDBArray.count {
                speakerDBMLArray[i] = NSNumber(value: speakerDBArray[i])
            }
        } else {
            // Fill with zeros for empty DB
            for i in 0..<embeddingSize {
                speakerDBMLArray[i] = NSNumber(value: 0.0)
            }
        }
        
        // Convert binarized segments
        let numFrames = binarizedSegments[0].count
        let numSpeakerSlots = binarizedSegments[0][0].count
        
        guard let segmentsArray = try? MLMultiArray(
            shape: [1, numFrames, numSpeakerSlots] as [NSNumber],
            dataType: .float32
        ) else {
            throw DiarizerError.processingFailed("Failed to create segments array")
        }
        
        for f in 0..<numFrames {
            for s in 0..<numSpeakerSlots {
                segmentsArray[f * numSpeakerSlots + s] = NSNumber(value: binarizedSegments[0][f][s])
            }
        }
        
        // Run unified model
        let inputs: [String: Any] = [
            "embeddings": embeddingsArray,
            "speaker_db": speakerDBMLArray,
            "binarized_segments": segmentsArray
        ]
        
        guard let output = try? unifiedModel.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs)),
              let activities = output.featureValue(for: "activities")?.multiArrayValue,
              let validSpeakers = output.featureValue(for: "valid_speakers")?.multiArrayValue,
              let filteredSegments = output.featureValue(for: "filtered_segments")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Unified model prediction failed")
        }
        
        // Convert activities to array
        var activitiesArray: [Float] = []
        for i in 0..<numSpeakerSlots {
            activitiesArray.append(activities[i].floatValue)
        }
        
        // Convert filtered segments back to 3D array
        var filteredSegmentsArray: [[[Float]]] = [Array(
            repeating: Array(repeating: 0.0, count: numSpeakerSlots),
            count: numFrames
        )]
        
        for f in 0..<numFrames {
            for s in 0..<numSpeakerSlots {
                filteredSegmentsArray[0][f][s] = filteredSegments[f * numSpeakerSlots + s].floatValue
            }
        }
        
        return (activitiesArray, filteredSegmentsArray)
    }

    private func applyPostProcessingFilters(_ segments: [TimedSpeakerSegment])
        -> [TimedSpeakerSegment]
    {
        return segments.filter { segment in
            segment.durationSeconds >= self.config.minDurationOn
        }
    }
}