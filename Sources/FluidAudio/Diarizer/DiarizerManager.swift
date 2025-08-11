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

    public init(config: DiarizerConfig = .default) {
        self.config = config
        self.speakerManager = SpeakerManager(
            speakerThreshold: config.clusteringThreshold * 1.2,
            embeddingThreshold: config.clusteringThreshold * 0.8,
            minSpeechDuration: config.minSpeechDuration
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

    /// Compare two audio samples to determine if they're from the same speaker.
    ///
    /// Performs diarization on both audio samples and compares the dominant speaker
    /// from each to calculate similarity.
    ///
    /// - Parameters:
    ///   - audio1: First audio sample (16kHz mono)
    ///   - audio2: Second audio sample (16kHz mono)
    /// - Returns: Similarity score (0-100, higher = more similar)
    /// - Throws: DiarizerError if processing fails
    ///
    /// Example:
    /// ```swift
    /// let similarity = try await diarizer.compareSpeakers(audio1: sample1, audio2: sample2)
    /// if similarity > 80 {
    ///     print("Likely the same speaker")
    /// }
    /// ```
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

    /// Validate embedding format.
    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        return audioValidation.validateEmbedding(embedding)
    }

    /// Validate audio quality.
    public func validateAudio(_ samples: [Float]) -> AudioValidationResult {
        return audioValidation.validateAudio(samples)
    }

    /// Initialize with known speaker profiles.
    public func initializeKnownSpeakers(_ speakers: [String: [Float]]) {
        speakerManager.initializeKnownSpeakers(speakers)
    }

    /// Perform complete speaker diarization on audio samples.
    ///
    /// Processes the entire audio to identify "who spoke when" by:
    /// - Detecting speech segments
    /// - Extracting speaker embeddings
    /// - Clustering speakers
    /// - Tracking consistent speaker IDs
    ///
    /// - Parameters:
    ///   - samples: Audio samples (should be 16kHz mono)
    ///   - sampleRate: Sample rate (default: 16000)
    /// - Returns: `DiarizationResult` containing:
    ///   - `segments`: Array of speaker segments with speaker IDs, timestamps, and embeddings
    ///   - `speakerDatabase`: Dictionary mapping speaker IDs to embeddings (only when debugMode enabled)
    ///   - `timings`: Performance metrics (only when debugMode enabled)
    /// - Throws: DiarizerError if not initialized or processing fails
    ///
    /// ```
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

        // Use DiarizerManager's chunk parameters with proper rounding
        let chunkDuration = Int(config.chunkDuration.rounded())
        let overlapDuration = Int(config.chunkOverlap.rounded())
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

        // Only populate debug info if debugMode is enabled
        if config.debugMode {
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
        } else {
            // Return minimal result for production use
            return DiarizationResult(segments: filteredSegments)
        }
    }

    internal struct ChunkTimings {
        let segmentationTime: TimeInterval
        let embeddingTime: TimeInterval
        let clusteringTime: TimeInterval
    }

    /// Process a single audio chunk with consistent speaker tracking.
    ///
    /// This function maintains speaker consistency across chunks by:
    /// - Using SpeakerManager to track and assign consistent speaker IDs
    /// - Updating speaker embeddings as more data becomes available
    /// - Handling both known and new speakers
    ///
    /// - Parameters:
    ///   - chunk: Audio chunk to process
    ///   - chunkOffset: Time offset of this chunk in the full audio
    ///   - speakerDB: Mutable speaker database (deprecated, use speakerManager instead)
    ///   - models: Diarization models for processing
    ///   - sampleRate: Audio sample rate
    /// - Returns: Tuple of (segments with speaker IDs, timing metrics)
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

        var speakerIds: [String] = []
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

                    // Calculate embedding quality
                    let quality = calculateEmbeddingQuality(embedding) * (activity / Float(numFrames))

                    if let speakerId = speakerManager.assignSpeaker(
                        embedding,
                        speechDuration: duration,
                        confidence: quality
                    ) {
                        speakerIds.append(speakerId)
                        // Also update the legacy speaker database for compatibility
                        speakerDB[speakerId] = embedding
                    } else {
                        speakerIds.append("")
                    }
                } else {
                    embeddingInvalidCount += 1
                    speakerIds.append("")
                }
            } else {
                activityFilteredCount += 1
                speakerIds.append("")
            }
        }

        let clusteringTime = Date().timeIntervalSince(clusteringStartTime)

        let segments = createTimedSegments(
            binarizedSegments: binarizedSegments,
            slidingWindow: slidingFeature.slidingWindow,
            embeddings: embeddings,
            speakerIds: speakerIds,
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
        // Filter by minimum duration
        let filtered = segments.filter { segment in
            segment.durationSeconds >= self.config.minSpeechDuration
        }

        // Finally merge nearby segments from the same speaker
        return mergeNearbySpeakerSegments(filtered)
    }

    /// Merge close segments from same speaker.
    private func mergeNearbySpeakerSegments(_ segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return segments }

        // Group segments by speaker
        var speakerSegments: [String: [TimedSpeakerSegment]] = [:]
        for segment in segments {
            speakerSegments[segment.speakerId, default: []].append(segment)
        }

        var mergedSegments: [TimedSpeakerSegment] = []

        for (speakerId, speakerSegs) in speakerSegments {
            let sorted = speakerSegs.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
            var merged: [TimedSpeakerSegment] = []
            var current = sorted[0]

            for i in 1..<sorted.count {
                let next = sorted[i]
                let gap = next.startTimeSeconds - current.endTimeSeconds

                // Merge if gap is less than minSilenceGap
                if gap < config.minSilenceGap {
                    // Extend current segment to include next
                    current = TimedSpeakerSegment(
                        speakerId: speakerId,
                        embedding: current.embedding,  // Keep first embedding
                        startTimeSeconds: current.startTimeSeconds,
                        endTimeSeconds: next.endTimeSeconds,
                        qualityScore: max(current.qualityScore, next.qualityScore)
                    )
                } else {
                    // Gap too large, save current and start new
                    merged.append(current)
                    current = next
                }
            }
            merged.append(current)
            mergedSegments.append(contentsOf: merged)
        }

        // Sort by start time for consistent output
        return mergedSegments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
    }

    // MARK: - Functions moved from SpeakerClustering

    /// Count activity frames per speaker.
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

    /// Convert frames to timed segments.
    private func createTimedSegments(
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerIds: [String],
        speakerActivities: [Float]
    ) -> [TimedSpeakerSegment] {
        let segmentation = binarizedSegments[0]
        let numFrames = segmentation.count
        let numSpeakers = segmentation[0].count
        var segments: [TimedSpeakerSegment] = []

        // Enhanced overlap detection: Process each speaker independently
        // This allows multiple speakers to be active in the same time frame
        for speakerIndex in 0..<numSpeakers {
            // Skip inactive speakers
            if speakerActivities[speakerIndex] < config.minActiveFramesCount {
                continue
            }

            // Track continuous segments for this speaker
            var isActive = false
            var startFrame = 0

            for frameIdx in 0..<numFrames {
                let frameActivity = segmentation[frameIdx][speakerIndex]

                // Dynamic threshold based on frame context
                // Lower threshold if other speakers are active (overlap detection)
                var activityThreshold: Float = 0.3

                // Check if other speakers are active in this frame
                for otherSpeaker in 0..<numSpeakers {
                    if otherSpeaker != speakerIndex && segmentation[frameIdx][otherSpeaker] > 0.3 {
                        // Another speaker is active, lower our threshold to detect overlap
                        activityThreshold = 0.15
                        break
                    }
                }

                if frameActivity > activityThreshold && !isActive {
                    // Start of a new segment
                    isActive = true
                    startFrame = frameIdx
                } else if frameActivity <= activityThreshold && isActive {
                    // End of current segment
                    if let segment = createSegmentIfValid(
                        speakerIndex: speakerIndex,
                        startFrame: startFrame,
                        endFrame: frameIdx,
                        slidingWindow: slidingWindow,
                        embeddings: embeddings,
                        speakerIds: speakerIds,
                        speakerActivities: speakerActivities
                    ) {
                        segments.append(segment)
                    }
                    isActive = false
                }
            }

            // Handle segment that extends to the end
            if isActive {
                if let segment = createSegmentIfValid(
                    speakerIndex: speakerIndex,
                    startFrame: startFrame,
                    endFrame: numFrames,
                    slidingWindow: slidingWindow,
                    embeddings: embeddings,
                    speakerIds: speakerIds,
                    speakerActivities: speakerActivities
                ) {
                    segments.append(segment)
                }
            }
        }

        // Sort segments by start time for consistent output
        segments.sort {
            $0.startTimeSeconds < $1.startTimeSeconds
        }

        return segments
    }

    /// Create segment if duration is valid.
    private func createSegmentIfValid(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerIds: [String],
        speakerActivities: [Float]
    ) -> TimedSpeakerSegment? {
        guard speakerIndex < speakerIds.count,
            !speakerIds[speakerIndex].isEmpty,
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
            speakerId: speakerIds[speakerIndex],
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

    /// Merge similar speakers (for IS meetings).
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

                    let distance = speakerManager.cosineDistance(info1.embedding, info2.embedding)

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
