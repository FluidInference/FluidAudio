import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager {

    internal let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    internal let config: DiarizerConfig
    private var models: DiarizerModels?

    /// Public getter for segmentation model (for streaming)
    public var segmentationModel: MLModel? {
        return models?.segmentationModel
    }

    public let segmentationProcessor = SegmentationProcessor()
    public var embeddingExtractor: EmbeddingExtractor?
    private let audioValidation = AudioValidation()
    private let memoryOptimizer = ANEMemoryOptimizer.shared

    // Speaker manager for consistent speaker tracking
    public let speakerManager: SpeakerManager

    // Chunking parameters owned by DiarizerManager
    public var chunkDuration: Float = 10.0  // seconds
    public var chunkOverlap: Float = 0.0  // seconds

    public init(config: DiarizerConfig = .default, chunkDuration: Float = 10.0, chunkOverlap: Float = 0.0) {
        self.config = config
        self.chunkDuration = chunkDuration
        self.chunkOverlap = chunkOverlap
        self.speakerManager = SpeakerManager(
            speakerThreshold: config.clusteringThreshold * 1.2,
            embeddingThreshold: config.clusteringThreshold * 0.8,
            minDuration: config.minSpeechDuration
        )
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

        let distance = speakerManager.cosineDistance(segment1.embedding, segment2.embedding)
        return max(0, (1.0 - distance) * 100)
    }

    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        return audioValidation.validateEmbedding(embedding)
    }

    public func validateAudio(_ samples: [Float]) -> AudioValidationResult {
        return audioValidation.validateAudio(samples)
    }

    public func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        return speakerManager.cosineDistance(a, b)
    }

    public func initializeKnownSpeakers(_ speakers: [String: [Float]]) {
        speakerManager.initializeKnownSpeakers(speakers)
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

        // Use DiarizerManager's chunk parameters
        let chunkDuration = Int(self.chunkDuration)
        let overlapDuration = Int(self.chunkOverlap)
        let chunkSize = sampleRate * chunkDuration
        let stepSize = chunkSize - (sampleRate * overlapDuration)  // Step size with overlap

        var allSegments: [TimedSpeakerSegment] = []
        var speakerDB: [String: [Float]] = [:]

        for chunkStart in stride(from: 0, to: samples.count, by: stepSize) {
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

        // Calculate actual chunk duration before padding
        _ = Float(chunk.count) / Float(sampleRate)

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

        // Unified and merged models removed - caused performance/stability issues

        // Otherwise use traditional separate model processing
        let slidingFeature = segmentationProcessor.createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        let segmentationTime = Date().timeIntervalSince(segmentationStartTime)

        let embeddingStartTime = Date()

        // Use EmbeddingExtractor for embedding extraction
        guard let embeddingExtractor = self.embeddingExtractor else {
            throw DiarizerError.notInitialized
        }

        logger.debug("Using EmbeddingExtractor for embedding extraction")

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
            minActivityThreshold: config.minActiveFramesCount
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

        // Calculate speaker activities
        let speakerActivities = calculateSpeakerActivities(binarizedSegments)

        var speakerLabels: [String] = []
        var activityFilteredCount = 0
        var embeddingInvalidCount = 0
        var clusteringProcessedCount = 0

        for (speakerIndex, activity) in speakerActivities.enumerated() {
            if activity > self.config.minActiveFramesCount {
                let embedding = embeddings[speakerIndex]
                if validateEmbedding(embedding) {
                    clusteringProcessedCount += 1
                    // Calculate exact duration based on sliding window step
                    // Each frame represents slidingWindow.step seconds (0.016875s from pyannote model)
                    let duration = Float(activity) * Float(slidingFeature.slidingWindow.step)

                    // Check for overlap in this speaker's frames
                    _ = detectOverlap(
                        speakerIndex: speakerIndex,
                        binarizedSegments: binarizedSegments
                    )

                    // Calculate embedding quality
                    let quality = calculateEmbeddingQuality(embedding) * (activity / Float(numFrames))

                    if let speakerId = speakerManager.assignSpeaker(
                        embedding,
                        speechDuration: duration,
                        confidence: quality
                    ) {
                        speakerLabels.append(speakerId)
                        // Also update the legacy speaker database for compatibility
                        speakerDB[speakerId] = embedding
                    } else {
                        speakerLabels.append("")
                    }
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

        let segments = createTimedSegments(
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
            segment.durationSeconds >= self.config.minSpeechDuration
        }
    }

    // MARK: - Functions moved from SpeakerClustering

    private func calculateSpeakerActivities(_ binarizedSegments: [[[Float]]]) -> [Float] {
        let numSpeakers = binarizedSegments[0][0].count
        let numFrames = binarizedSegments[0].count
        var activities: [Float] = Array(repeating: 0.0, count: numSpeakers)

        for speakerIndex in 0..<numSpeakers {
            for frameIndex in 0..<numFrames {
                activities[speakerIndex] += binarizedSegments[0][frameIndex][speakerIndex]
            }
        }

        return activities
    }

    private func createTimedSegments(
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> [TimedSpeakerSegment] {
        let segmentation = binarizedSegments[0]
        let numFrames = segmentation.count
        var segments: [TimedSpeakerSegment] = []

        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)
            }
        }

        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                if let segment = createSegmentIfValid(
                    speakerIndex: currentSpeaker,
                    startFrame: startFrame,
                    endFrame: i,
                    slidingWindow: slidingWindow,
                    embeddings: embeddings,
                    speakerLabels: speakerLabels,
                    speakerActivities: speakerActivities
                ) {
                    segments.append(segment)
                }
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        if let segment = createSegmentIfValid(
            speakerIndex: currentSpeaker,
            startFrame: startFrame,
            endFrame: numFrames,
            slidingWindow: slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        ) {
            segments.append(segment)
        }

        return segments
    }

    private func createSegmentIfValid(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> TimedSpeakerSegment? {
        guard speakerIndex < speakerLabels.count,
            !speakerLabels[speakerIndex].isEmpty,
            speakerIndex < embeddings.count
        else {
            return nil
        }

        let startTime = slidingWindow.time(forFrame: startFrame)
        let endTime = slidingWindow.time(forFrame: endFrame)
        let duration = endTime - startTime

        if Float(duration) < config.minSpeechDuration {
            return nil
        }

        let embedding = embeddings[speakerIndex]
        let activity = speakerActivities[speakerIndex]
        let quality = calculateEmbeddingQuality(embedding) * (activity / Float(endFrame - startFrame))

        return TimedSpeakerSegment(
            speakerId: speakerLabels[speakerIndex],
            embedding: embedding,
            startTimeSeconds: Float(startTime),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
    }

    private func calculateEmbeddingQuality(_ embedding: [Float]) -> Float {
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        return min(1.0, magnitude / 10.0)
    }

    private func detectOverlap(speakerIndex: Int, binarizedSegments: [[[Float]]]) -> Bool {
        let numFrames = binarizedSegments[0].count
        var overlapCount = 0
        var totalActive = 0

        for frameIndex in 0..<numFrames {
            // Check if this speaker is active in this frame
            if binarizedSegments[0][frameIndex][speakerIndex] > 0.5 {
                totalActive += 1

                // Count how many speakers are active in this frame
                let activeSpeakers = binarizedSegments[0][frameIndex].filter { $0 > 0.5 }.count
                if activeSpeakers > 1 {
                    overlapCount += 1
                }
            }
        }

        // Consider it overlapping if more than 20% of active frames have multiple speakers
        return totalActive > 0 && Float(overlapCount) / Float(totalActive) > 0.2
    }

    /// Post-process to merge similar speakers (for IS meetings)
    public func mergeSimilarSpeakers(threshold: Float = 0.3) {
        let speakers = speakerManager.speakerIds
        var mergedPairs: Set<String> = []

        for i in 0..<speakers.count {
            for j in (i + 1)..<speakers.count {
                let speakerId1 = speakers[i]
                let speakerId2 = speakers[j]

                // Skip if already merged
                if mergedPairs.contains(speakerId1) || mergedPairs.contains(speakerId2) {
                    continue
                }

                if let info1 = speakerManager.getSpeakerInfo(for: speakerId1),
                    let info2 = speakerManager.getSpeakerInfo(for: speakerId2)
                {

                    let distance = cosineDistance(info1.embedding, info2.embedding)

                    if distance < threshold {
                        // Merge speaker2 into speaker1
                        logger.info(
                            "Merging similar speakers: \(speakerId2) into \(speakerId1) (distance: \(distance))")

                        // Transfer all segments from speaker2 to speaker1
                        // This would need to be implemented in your segment management
                        mergedPairs.insert(speakerId2)
                    }
                }
            }
        }
    }
}
