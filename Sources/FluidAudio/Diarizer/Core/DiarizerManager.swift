import Accelerate
import CoreML
import Foundation
import OSLog

public final class DiarizerManager {

    internal let logger = AppLogger(category: "Diarizer")
    internal let config: DiarizerConfig
    private var models: DiarizerModels?
    /// Public getter for segmentation model (for streaming)
    public var segmentationModel: MLModel? {
        return models?.segmentationModel
    }

    public let segmentationProcessor = SegmentationProcessor()
    public var embeddingExtractor: EmbeddingExtractor?
    private let audioValidation = AudioValidation()
    private let memoryOptimizer = ANEMemoryOptimizer()

    // Speaker manager for consistent speaker tracking
    public let speakerManager: SpeakerManager

    public init(config: DiarizerConfig = .default) {
        self.config = config
        self.speakerManager = SpeakerManager(
            // Speaker assignment threshold: 0.9x clustering threshold
            // Slightly more aggressive to reduce over-segmentation
            speakerThreshold: config.clusteringThreshold * 1.2,
            // Embedding update threshold: 0.8x clustering threshold
            // More aggressive (lower threshold) to update embeddings with high-confidence matches
            embeddingThreshold: config.clusteringThreshold * 0.8,
            minSpeechDuration: config.minSpeechDuration,
            minEmbeddingUpdateDuration: config.minEmbeddingUpdateDuration
        )
    }

    public var isAvailable: Bool {
        models != nil
    }

    public func initialize(models: consuming DiarizerModels) {
        logger.info("Initializing diarization system")

        self.embeddingExtractor = EmbeddingExtractor(embeddingModel: models.embeddingModel)
        logger.info("EmbeddingExtractor initialized with embedding model")

        self.models = consume models
    }

    public func cleanup() {
        models = nil
        logger.info("Diarization resources cleaned up")
    }

    /// Validate embedding format.
    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        return audioValidation.validateEmbedding(embedding)
    }

    /// Validate audio quality.
    ///
    /// - Parameter samples: Audio samples - accepts any Collection of Float
    ///                     (Array, ArraySlice, or custom collections)
    /// - Returns: AudioValidationResult with validity status and any issues
    public func validateAudio<C>(_ samples: C) -> AudioValidationResult
    where C: Collection, C.Element == Float {
        return audioValidation.validateAudio(samples)
    }

    /// Accepts Speaker structs and adds them to the in-memory database.
    ///
    /// - Parameter speakers: Array of Speaker structs with embeddings and metadata
    public func initializeKnownSpeakers(_ speakers: [Speaker]) {
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
    ///   - samples: Audio samples (16kHz mono) - accepts any RandomAccessCollection of Float
    ///             (Array, ArraySlice, ContiguousArray, or custom collections)
    ///   - sampleRate: Sample rate (default: 16000)
    /// - Returns: `DiarizationResult` containing:
    ///   - `segments`: Array of speaker segments with speaker IDs, timestamps, and embeddings
    ///   - `speakerDatabase`: Dictionary mapping speaker IDs to embeddings (only when debugMode enabled)
    ///   - `timings`: Performance metrics (only when debugMode enabled)
    /// - Throws: DiarizerError if not initialized or processing fails
    ///
    /// Example usage:
    /// ```swift
    /// // With Array (traditional)
    /// let audioArray = [Float](repeating: 0, count: 160000)
    /// let result = try diarizer.performCompleteDiarization(audioArray)
    ///
    /// // With ArraySlice (zero-copy view)
    /// let audioSlice = audioArray[8000..<24000]  // No copy!
    /// let result = try diarizer.performCompleteDiarization(audioSlice)
    ///
    /// // With ContiguousArray (performance-optimized)
    /// let audioContiguous = ContiguousArray<Float>(audioArray)
    /// let result = try diarizer.performCompleteDiarization(audioContiguous)
    /// ```
    public func performCompleteDiarization<C>(
        _ samples: C, sampleRate: Int = 16000, atTime startTime: TimeInterval = 0
    ) throws -> DiarizationResult
    where C: RandomAccessCollection, C.Element == Float, C.Index == Int {
        guard let models else {
            throw DiarizerError.notInitialized
        }

        var segmentationTime: TimeInterval = 0
        var embeddingTime: TimeInterval = 0
        var clusteringTime: TimeInterval = 0
        var postProcessingTime: TimeInterval = 0

        let chunkDuration = Int(config.chunkDuration.rounded())
        let overlapDuration = Int(config.chunkOverlap.rounded())
        let chunkSize = sampleRate * chunkDuration
        let stepSize = chunkSize - (sampleRate * overlapDuration)
        var chunkBuffer = [Float](repeating: 0.0, count: max(chunkSize, 1))

        var allSegments: [TimedSpeakerSegment] = []

        let startIndex = samples.startIndex
        let endIndex = samples.endIndex
        let totalSamples = samples.distance(from: startIndex, to: endIndex)

        for chunkStartOffset in stride(from: 0, to: totalSamples, by: stepSize) {
            let chunkStart = samples.index(startIndex, offsetBy: chunkStartOffset)
            let remainingSamples = samples.distance(from: chunkStart, to: endIndex)
            let chunkEndOffset = min(chunkSize, remainingSamples)
            let chunkEnd = samples.index(chunkStart, offsetBy: chunkEndOffset)
            let chunk = samples[chunkStart..<chunkEnd]
            let chunkOffset = Double(chunkStartOffset) / Double(sampleRate) + startTime

            let (chunkSegments, chunkTimings) = try processChunkWithSpeakerTracking(
                chunk,
                chunkOffset: chunkOffset,
                models: models,
                sampleRate: sampleRate,
                chunkSize: chunkSize,
                chunkBuffer: &chunkBuffer
            )
            allSegments.append(contentsOf: chunkSegments)

            segmentationTime += chunkTimings.segmentationTime
            embeddingTime += chunkTimings.embeddingTime
            clusteringTime += chunkTimings.clusteringTime
        }

        let postProcessingStartTime = Date()
        let filteredSegments = allSegments  // No post-processing
        postProcessingTime = Date().timeIntervalSince(postProcessingStartTime)

        if config.debugMode {
            let timings = PipelineTimings(
                modelCompilationSeconds: models.compilationDuration,
                audioLoadingSeconds: 0,
                segmentationSeconds: segmentationTime,
                embeddingExtractionSeconds: embeddingTime,
                speakerClusteringSeconds: clusteringTime,
                postProcessingSeconds: postProcessingTime
            )

            // Build speakerDatabase from speakerManager for debug output
            let speakerDB = speakerManager.getAllSpeakers().reduce(into: [String: [Float]]()) { result, item in
                result[item.key] = item.value.currentEmbedding
            }

            return DiarizationResult(
                segments: filteredSegments, speakerDatabase: speakerDB, timings: timings)
        } else {
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
    ///   - chunk: Audio chunk to process (can be any RandomAccessCollection)
    ///   - chunkOffset: Time offset of this chunk in the full audio
    ///   - models: Diarization models for processing
    ///   - sampleRate: Audio sample rate
    /// - Returns: Tuple of (segments with speaker IDs, timing metrics)
    private func processChunkWithSpeakerTracking<C>(
        _ chunk: C,
        chunkOffset: Double,
        models: DiarizerModels,
        sampleRate: Int = 16000,
        chunkSize: Int,
        chunkBuffer: inout [Float]
    ) throws -> ([TimedSpeakerSegment], ChunkTimings)
    where C: RandomAccessCollection, C.Element == Float, C.Index == Int {
        let segmentationStartTime = Date()

        let chunkCount = chunk.distance(from: chunk.startIndex, to: chunk.endIndex)
        let copyCount = min(chunkCount, chunkSize)

        if chunkBuffer.count != chunkSize {
            chunkBuffer = [Float](repeating: 0.0, count: chunkSize)
        }

        chunkBuffer.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            vDSP_vclr(baseAddress, 1, vDSP_Length(chunkSize))

            guard copyCount > 0 else { return }

            let copied =
                chunk.withContiguousStorageIfAvailable { storage -> Bool in
                    storage.withUnsafeBufferPointer { src in
                        vDSP_mmov(
                            src.baseAddress!,
                            baseAddress,
                            vDSP_Length(copyCount),
                            vDSP_Length(1),
                            vDSP_Length(1),
                            vDSP_Length(chunkSize)
                        )
                    }
                    return true
                } ?? false

            if !copied {
                var index = chunk.startIndex
                for i in 0..<copyCount {
                    baseAddress.advanced(by: i).pointee = chunk[index]
                    index = chunk.index(after: index)
                }
            }
        }

        let paddedChunk: ArraySlice<Float> = chunkBuffer[0..<chunkSize]

        let (binarizedSegments, _) = try segmentationProcessor.getSegments(
            audioChunk: paddedChunk,
            segmentationModel: models.segmentationModel
        )

        let slidingFeature = segmentationProcessor.createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        let segmentationTime = Date().timeIntervalSince(segmentationStartTime)

        let embeddingStartTime = Date()

        guard let embeddingExtractor = self.embeddingExtractor else {
            throw DiarizerError.notInitialized
        }

        var masks: [[Float]] = []
        let numSpeakers = slidingFeature.data[0][0].count
        let numFrames = slidingFeature.data[0].count

        for s in 0..<numSpeakers {
            var speakerMask: [Float] = []
            for f in 0..<numFrames {
                let speakerSum = slidingFeature.data[0][f].reduce(0, +)
                let isClean: Float = speakerSum < 2.0 ? 1.0 : 0.0  // Clean frame: only 1 speaker
                speakerMask.append(slidingFeature.data[0][f][s] * isClean)
            }
            masks.append(speakerMask)
        }

        let embeddings = try embeddingExtractor.getEmbeddings(
            audio: paddedChunk,
            masks: masks,
            minActivityThreshold: config.minActiveFramesCount
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

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
                    // Each frame = 0.016875s (pyannote model step size)
                    let duration = Float(activity) * Float(slidingFeature.slidingWindow.step)

                    let quality = calculateEmbeddingQuality(embedding) * (activity / Float(numFrames))

                    if let speaker = speakerManager.assignSpeaker(
                        embedding,
                        speechDuration: duration,
                        confidence: quality
                    ) {
                        speakerIds.append(speaker.id)
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

        // Process each speaker independently for overlap detection
        for speakerIndex in 0..<numSpeakers {
            if speakerActivities[speakerIndex] < config.minActiveFramesCount {
                continue
            }

            var isActive = false
            var startFrame = 0

            for frameIdx in 0..<numFrames {
                let frameActivity = segmentation[frameIdx][speakerIndex]

                // Dynamic threshold: lower when other speakers active
                var activityThreshold: Float = 0.3

                for otherSpeaker in 0..<numSpeakers {
                    if otherSpeaker != speakerIndex && segmentation[frameIdx][otherSpeaker] > 0.3 {
                        activityThreshold = 0.15  // Detect overlap
                        break
                    }
                }

                if frameActivity > activityThreshold && !isActive {
                    isActive = true
                    startFrame = frameIdx
                } else if frameActivity <= activityThreshold && isActive {
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
        let magnitude = sqrt(vDSP.sumOfSquares(embedding))
        return min(1.0, magnitude / 10.0)
    }

}
