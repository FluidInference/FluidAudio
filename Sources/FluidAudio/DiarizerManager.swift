import CoreML
import Foundation
import OSLog

public struct DiarizerConfig: Sendable {
    public var clusteringThreshold: Float = 0.7  // Similarity threshold for grouping speakers (0.0-1.0, higher = stricter)
    public var minDurationOn: Float = 1.0  // Minimum duration (seconds) for a speaker segment to be considered valid
    public var minDurationOff: Float = 0.5  // Minimum silence duration (seconds) between different speakers
    public var numClusters: Int = -1  // Number of speakers to detect (-1 = auto-detect)
    public var minActivityThreshold: Float = 10.0  // Minimum activity threshold (frames) for speaker to be considered active
    public var debugMode: Bool = false

    public static let `default` = DiarizerConfig()
    
    /// Platform-optimized configuration for iOS devices
    #if os(iOS)
    public static let iosOptimized = DiarizerConfig(
        clusteringThreshold: 0.7,
        minDurationOn: 1.0,
        minDurationOff: 0.5,
        numClusters: -1,
        minActivityThreshold: 10.0,
        debugMode: false
    )
    #endif

    public init(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        minActivityThreshold: Float = 10.0,
        debugMode: Bool = false
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.minActivityThreshold = minActivityThreshold
        self.debugMode = debugMode
    }
}

/// Detailed timing breakdown for each stage of the diarization pipeline
public struct PipelineTimings: Sendable, Codable {
    public let modelDownloadSeconds: TimeInterval
    public let modelCompilationSeconds: TimeInterval
    public let audioLoadingSeconds: TimeInterval
    public let segmentationSeconds: TimeInterval
    public let embeddingExtractionSeconds: TimeInterval
    public let speakerClusteringSeconds: TimeInterval
    public let postProcessingSeconds: TimeInterval
    public let totalInferenceSeconds: TimeInterval  // segmentation + embedding + clustering
    public let totalProcessingSeconds: TimeInterval  // all stages combined

    public init(
        modelDownloadSeconds: TimeInterval = 0,
        modelCompilationSeconds: TimeInterval = 0,
        audioLoadingSeconds: TimeInterval = 0,
        segmentationSeconds: TimeInterval = 0,
        embeddingExtractionSeconds: TimeInterval = 0,
        speakerClusteringSeconds: TimeInterval = 0,
        postProcessingSeconds: TimeInterval = 0
    ) {
        self.modelDownloadSeconds = modelDownloadSeconds
        self.modelCompilationSeconds = modelCompilationSeconds
        self.audioLoadingSeconds = audioLoadingSeconds
        self.segmentationSeconds = segmentationSeconds
        self.embeddingExtractionSeconds = embeddingExtractionSeconds
        self.speakerClusteringSeconds = speakerClusteringSeconds
        self.postProcessingSeconds = postProcessingSeconds
        self.totalInferenceSeconds =
            segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
        self.totalProcessingSeconds =
            modelDownloadSeconds + modelCompilationSeconds + audioLoadingSeconds
            + segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
            + postProcessingSeconds
    }

    /// Calculate percentage breakdown of time spent in each stage
    public var stagePercentages: [String: Double] {
        guard totalProcessingSeconds > 0 else {
            return [:]
        }

        return [
            "Model Download": (modelDownloadSeconds / totalProcessingSeconds) * 100,
            "Model Compilation": (modelCompilationSeconds / totalProcessingSeconds) * 100,
            "Audio Loading": (audioLoadingSeconds / totalProcessingSeconds) * 100,
            "Segmentation": (segmentationSeconds / totalProcessingSeconds) * 100,
            "Embedding Extraction": (embeddingExtractionSeconds / totalProcessingSeconds) * 100,
            "Speaker Clustering": (speakerClusteringSeconds / totalProcessingSeconds) * 100,
            "Post Processing": (postProcessingSeconds / totalProcessingSeconds) * 100,
        ]
    }

    /// Identify the bottleneck stage (stage taking the most time)
    public var bottleneckStage: String {
        let stages = [
            ("Model Download", modelDownloadSeconds),
            ("Model Compilation", modelCompilationSeconds),
            ("Audio Loading", audioLoadingSeconds),
            ("Segmentation", segmentationSeconds),
            ("Embedding Extraction", embeddingExtractionSeconds),
            ("Speaker Clustering", speakerClusteringSeconds),
            ("Post Processing", postProcessingSeconds),
        ]

        return stages.max(by: { $0.1 < $1.1 })?.0 ?? "Unknown"
    }
}

extension Duration {

    /// Converts this duration to a Foundation TimeInterval - i.e. a `Double` number of seconds.
    ///
    internal var timeInterval: TimeInterval {
        self / .seconds(1)
    }
}

/// Complete diarization result with consistent speaker IDs and embeddings
public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]
    public let speakerDatabase: [String: [Float]]  // Speaker ID → representative embedding
    public let timings: PipelineTimings

    public init(
        segments: [TimedSpeakerSegment], speakerDatabase: [String: [Float]],
        timings: PipelineTimings = PipelineTimings()
    ) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
        self.timings = timings
    }
}

/// Speaker segment with embedding and consistent ID across chunks
public struct TimedSpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerId: String  // "Speaker 1", "Speaker 2", etc.
    public let embedding: [Float]  // Voice characteristics
    public let startTimeSeconds: Float  // When segment starts
    public let endTimeSeconds: Float  // When segment ends
    public let qualityScore: Float  // Embedding quality

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    public init(
        speakerId: String, embedding: [Float], startTimeSeconds: Float, endTimeSeconds: Float,
        qualityScore: Float
    ) {
        self.speakerId = speakerId
        self.embedding = embedding
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.qualityScore = qualityScore
    }
}

public struct SpeakerEmbedding: Sendable {
    public let embedding: [Float]
    public let qualityScore: Float
    public let durationSeconds: Float

    public init(embedding: [Float], qualityScore: Float, durationSeconds: Float) {
        self.embedding = embedding
        self.qualityScore = qualityScore
        self.durationSeconds = durationSeconds
    }
}

public struct ModelPaths: Sendable {
    public let segmentationPath: URL
    public let embeddingPath: URL

    public init(segmentationPath: URL, embeddingPath: URL) {
        self.segmentationPath = segmentationPath
        self.embeddingPath = embeddingPath
    }
}

/// Audio validation result
public struct AudioValidationResult: Sendable {
    public let isValid: Bool
    public let durationSeconds: Float
    public let issues: [String]

    public init(isValid: Bool, durationSeconds: Float, issues: [String] = []) {
        self.isValid = isValid
        self.durationSeconds = durationSeconds
        self.issues = issues
    }
}

// MARK: - Error Types

public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case modelCompilationFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .modelCompilationFailed:
            return "Failed to compile CoreML models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}

private struct Segment: Hashable {
    let start: Double
    let end: Double
}

private struct SlidingWindow {
    var start: Double
    var duration: Double
    var step: Double

    func time(forFrame index: Int) -> Double {
        return start + Double(index) * step
    }

    func segment(forFrame index: Int) -> Segment {
        let s = time(forFrame: index)
        return Segment(start: s, end: s + duration)
    }
}

private struct SlidingWindowFeature {
    var data: [[[Float]]]  // (1, 589, 3)
    var slidingWindow: SlidingWindow
}

// MARK: - Diarizer Implementation

/// Speaker diarization manager
@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    private let config: DiarizerConfig

    private var models: DiarizerModels?

    public init(config: DiarizerConfig = .default) {
        self.config = config
    }

    public var isAvailable: Bool {
        models != nil
    }

    /// Get the initialization timing data
    public var initializationTimings: (downloadTime: TimeInterval, compilationTime: TimeInterval) {
        models.map { ($0.downloadTime.timeInterval, $0.compilationTime.timeInterval) } ?? (0, 0)
    }


    public func initialize(models: consuming DiarizerModels) {
        logger.info("Initializing diarization system")
        self.models = consume models
    }

    @available(*, deprecated, message: "Use initialize(models:) instead")
    public func initialize() async throws {
        self.initialize(models: try await .downloadIfNeeded())
    }

    /// Clean up resources
    public func cleanup() {
        models = nil
        logger.info("Diarization resources cleaned up")
    }

    private func getSegments(audioChunk: ArraySlice<Float>, chunkSize: Int = 160_000) throws -> [[[Float]]] {

        guard let segmentationModel = models?.segmentationModel else {
            throw DiarizerError.notInitialized
        }

        let audioArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: chunkSize)],
            dataType: .float32
        )
        var offset = 0
        for sample in audioChunk.prefix(chunkSize) {
            audioArray[offset] = NSNumber(value: sample)

            // In order for the loop body to even execute, chunkSize > 0.
            // Thus, offset begins at 0 and increments by 1 to some positive Int, which can never overflow.
            offset &+= 1
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])

        let output = try segmentationModel.prediction(from: input)

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output from segmentation model")
        }

        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        var segments = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: combinations), count: frames),
            count: 1)

        for f in 0..<frames {
            for c in 0..<combinations {
                let index = f * combinations + c
                segments[0][f][c] = segmentOutput[index].floatValue
            }
        }

        return powersetConversion(segments)
    }

    private func powersetConversion(_ segments: [[[Float]]]) -> [[[Float]]] {
        let powerset: [[Int]] = [
            [],  // 0
            [0],  // 1
            [1],  // 2
            [2],  // 3
            [0, 1],  // 4
            [0, 2],  // 5
            [1, 2],  // 6
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        var binarized = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers),
                count: numFrames
            ),
            count: batchSize
        )

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                // Find index of max value in this frame
                guard let bestIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) else {
                    continue
                }

                // Mark the corresponding speakers as active
                for speaker in powerset[bestIdx] {
                    binarized[b][f][speaker] = 1.0
                }
            }
        }

        return binarized
    }

    private func createSlidingWindowFeature(
        binarizedSegments: [[[Float]]], chunkOffset: Double = 0.0
    ) -> SlidingWindowFeature {
        let slidingWindow = SlidingWindow(
            start: chunkOffset,
            duration: 0.0619375,
            step: 0.016875
        )

        return SlidingWindowFeature(
            data: binarizedSegments,
            slidingWindow: slidingWindow
        )
    }

    private func getEmbedding(
        audioChunk: ArraySlice<Float>,
        binarizedSegments: [[[Float]]],
        slidingWindowFeature: SlidingWindowFeature,
        embeddingModel: MLModel,
        sampleRate: Int = 16000
    ) throws -> [[Float]] {
        let chunkSize = 10 * sampleRate
        let audioTensor = audioChunk
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count

        // Compute clean_frames = 1.0 where active speakers < 2
        var cleanFrames = Array(
            repeating: Array(repeating: 0.0 as Float, count: 1), count: numFrames)

        for f in 0..<numFrames {
            let frame = slidingWindowFeature.data[0][f]
            let speakerSum = frame.reduce(0, +)
            cleanFrames[f][0] = (speakerSum < 2.0) ? 1.0 : 0.0
        }

        // Multiply slidingWindowSegments.data by cleanFrames
        var cleanSegmentData = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers), count: numFrames),
            count: 1
        )

        for f in 0..<numFrames {
            for s in 0..<numSpeakers {
                cleanSegmentData[0][f][s] = slidingWindowFeature.data[0][f][s] * cleanFrames[f][0]
            }
        }

        // Transpose mask shape to (numSpeakers, 589)
        var cleanMasks: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: numFrames), count: numSpeakers)

        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                cleanMasks[s][f] = cleanSegmentData[0][f][s]
            }
        }

        // Prepare MLMultiArray inputs
        guard
            let waveformArray = try? MLMultiArray(
                shape: [numSpeakers, chunkSize] as [NSNumber], dataType: .float32),
            let maskArray = try? MLMultiArray(
                shape: [numSpeakers, numFrames] as [NSNumber], dataType: .float32)
        else {
            throw DiarizerError.processingFailed("Failed to allocate MLMultiArray for embeddings")
        }

        for s in 0..<numSpeakers {
            for i in 0..<min(chunkSize, audioTensor.count) {
                waveformArray[s * chunkSize + i] = NSNumber(value: audioTensor[audioTensor.startIndex + i])
            }
        }

        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                maskArray[s * numFrames + f] = NSNumber(value: cleanMasks[s][f])
            }
        }

        // Run model
        let inputs: [String: Any] = [
            "waveform": waveformArray,
            "mask": maskArray,
        ]

        guard
            let output = try? embeddingModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: inputs)),
            let multiArray = output.featureValue(for: "embedding")?.multiArrayValue
        else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }

        return convertToSendableArray(multiArray)
    }

    private func convertToSendableArray(_ multiArray: MLMultiArray) -> [[Float]] {
        let shape = multiArray.shape.map { $0.intValue }
        let numRows = shape[0]
        let numCols = shape[1]
        let strides = multiArray.strides.map { $0.intValue }

        var result: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: numCols), count: numRows)

        for i in 0..<numRows {
            for j in 0..<numCols {
                let index = i * strides[0] + j * strides[1]
                result[i][j] = multiArray[index].floatValue
            }
        }

        return result
    }

    private func getAnnotation(
        annotation: inout [Segment: Int],
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow
    ) {
        let segmentation = binarizedSegments[0]  // shape: [589][3]
        let numFrames = segmentation.count

        // Step 1: argmax to get dominant speaker per frame
        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)  // fallback
            }
        }

        // Step 2: group contiguous same-speaker segments
        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                let startTime = slidingWindow.time(forFrame: startFrame)
                let endTime = slidingWindow.time(forFrame: i)

                let segment = Segment(start: startTime, end: endTime)
                annotation[segment] = currentSpeaker  // Use raw speaker index
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        // Final segment
        let finalStart = slidingWindow.time(forFrame: startFrame)
        let finalEnd = slidingWindow.segment(forFrame: numFrames - 1).end
        let finalSegment = Segment(start: finalStart, end: finalEnd)
        annotation[finalSegment] = currentSpeaker  // Use raw speaker index
    }


    // MARK: - Audio Analysis

    /// Compare similarity between two audio samples using efficient diarization
    public func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float {
        // Use the efficient method to get embeddings
        async let result1 = performCompleteDiarization(audio1)
        async let result2 = performCompleteDiarization(audio2)

        // Get the most representative embedding from each audio
        guard let segment1 = try await result1.segments.max(by: { $0.qualityScore < $1.qualityScore }),
            let segment2 = try await result2.segments.max(by: { $0.qualityScore < $1.qualityScore })
        else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let distance = cosineDistance(segment1.embedding, segment2.embedding)
        return max(0, (1.0 - distance) * 100)  // Convert to similarity percentage
    }

    /// Validate if an embedding is valid
    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        guard !embedding.isEmpty else { return false }

        // Check for NaN or infinite values
        guard embedding.allSatisfy({ $0.isFinite }) else { return false }

        // Check magnitude
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        guard magnitude > 0.1 else { return false }

        return true
    }

    /// Validate audio quality and characteristics
    public func validateAudio(_ samples: [Float]) -> AudioValidationResult {
        let duration = Float(samples.count) / 16000.0
        var issues: [String] = []

        if duration < 1.0 {
            issues.append("Audio too short (minimum 1 second)")
        }

        if samples.isEmpty {
            issues.append("No audio data")
        }

        // Check for silence
        let rmsEnergy = calculateRMSEnergy(samples)
        if rmsEnergy < 0.01 {
            issues.append("Audio too quiet or silent")
        }

        return AudioValidationResult(
            isValid: issues.isEmpty,
            durationSeconds: duration,
            issues: issues
        )
    }

    // MARK: - Utility Functions

    /// Calculate cosine distance between two embeddings
    public func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            logger.debug(
                "🔍 CLUSTERING DEBUG: Invalid embeddings for distance calculation - a.count: \(a.count), b.count: \(b.count)"
            )
            return Float.infinity
        }

        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }

        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)

        guard magnitudeA > 0 && magnitudeB > 0 else {
            logger.warning(
                "🔍 CLUSTERING DEBUG: Zero magnitude embedding detected - magnitudeA: \(magnitudeA), magnitudeB: \(magnitudeB)"
            )
            return Float.infinity
        }

        let similarity = dotProduct / (magnitudeA * magnitudeB)
        let distance = 1 - similarity

        // DEBUG: Log distance calculation details
        logger.debug(
            "🔍 CLUSTERING DEBUG: cosineDistance - similarity: \(String(format: "%.4f", similarity)), distance: \(String(format: "%.4f", distance)), magA: \(String(format: "%.4f", magnitudeA)), magB: \(String(format: "%.4f", magnitudeB))"
        )

        return distance
    }

    private func calculateRMSEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let squaredSum = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(squaredSum / Float(samples.count))
    }

    private func calculateEmbeddingQuality(_ embedding: [Float]) -> Float {
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        // Simple quality score based on magnitude
        return min(1.0, magnitude / 10.0)
    }

    /// Select the embedding for the most active speaker based on speaker activity
    private func selectMostActiveSpeaker(
        embeddings: [[Float]],
        binarizedSegments: [[[Float]]]
    ) -> (embedding: [Float], activity: Float) {
        guard !embeddings.isEmpty, !binarizedSegments.isEmpty else {
            return ([], 0.0)
        }

        let numSpeakers = min(embeddings.count, binarizedSegments[0][0].count)
        var speakerActivities: [Float] = []

        // Calculate total activity for each speaker
        for speakerIndex in 0..<numSpeakers {
            var totalActivity: Float = 0.0
            let numFrames = binarizedSegments[0].count

            for frameIndex in 0..<numFrames {
                totalActivity += binarizedSegments[0][frameIndex][speakerIndex]
            }

            speakerActivities.append(totalActivity)
        }

        // Find the most active speaker
        guard
            let maxActivityIndex = speakerActivities.indices.max(by: {
                speakerActivities[$0] < speakerActivities[$1]
            })
        else {
            return (embeddings[0], 0.0)
        }

        let maxActivity = speakerActivities[maxActivityIndex]
        let normalizedActivity = maxActivity / Float(binarizedSegments[0].count)

        return (embeddings[maxActivityIndex], normalizedActivity)
    }

    // MARK: - Combined Efficient Diarization

    /// Perform complete diarization with consistent speaker IDs across chunks
    /// This is more efficient than calling performSegmentation + extractEmbedding separately
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

        let chunkSize = sampleRate * 10  // 10 seconds
        var allSegments: [TimedSpeakerSegment] = []
        var speakerDB: [String: [Float]] = [:]  // Global speaker database

        // Process in 10-second chunks
        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, samples.count)
            let chunk = samples[chunkStart..<chunkEnd]
            let chunkOffset = Double(chunkStart) / Double(sampleRate)

            let (chunkSegments, chunkTimings) = try processChunkWithSpeakerTracking(
                chunk,
                chunkOffset: chunkOffset,
                speakerDB: &speakerDB,
                sampleRate: sampleRate
            )
            allSegments.append(contentsOf: chunkSegments)

            // Accumulate timing from all chunks
            segmentationTime += chunkTimings.segmentationTime
            embeddingTime += chunkTimings.embeddingTime
            clusteringTime += chunkTimings.clusteringTime
        }

        let postProcessingStartTime = Date()
        // Post-processing: filter segments, apply duration constraints, etc.
        let filteredSegments = applyPostProcessingFilters(allSegments)
        postProcessingTime = Date().timeIntervalSince(postProcessingStartTime)

        let totalProcessingTime = Date().timeIntervalSince(processingStartTime)

        let timings = PipelineTimings(
            modelDownloadSeconds: models.downloadTime.timeInterval,
            modelCompilationSeconds: models.compilationTime.timeInterval,
            audioLoadingSeconds: 0,  // Will be set by CLI
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

    /// Timing data for chunk processing
    private struct ChunkTimings {
        let segmentationTime: TimeInterval
        let embeddingTime: TimeInterval
        let clusteringTime: TimeInterval
    }

    /// Process a single chunk with speaker tracking and timing
    private func processChunkWithSpeakerTracking(
        _ chunk: ArraySlice<Float>,
        chunkOffset: Double,
        speakerDB: inout [String: [Float]],
        sampleRate: Int = 16000
    ) throws -> ([TimedSpeakerSegment], ChunkTimings) {
        let segmentationStartTime = Date()

        let chunkSize = sampleRate * 10  // 10 seconds
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            var padded = Array(repeating: 0.0 as Float, count: chunk.count)
            padded.replaceSubrange(0..<chunk.count, with: chunk)
            paddedChunk = padded[...]
        }

        // Step 1: Get segmentation (when speakers are active)
        let binarizedSegments = try getSegments(audioChunk: paddedChunk)
        let slidingFeature = createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        let segmentationTime = Date().timeIntervalSince(segmentationStartTime)
        let embeddingStartTime = Date()

        // Step 2: Get embeddings using same segmentation results
        guard let embeddingModel = models?.embeddingModel else {
            throw DiarizerError.notInitialized
        }

        let embeddings = try getEmbedding(
            audioChunk: paddedChunk,
            binarizedSegments: binarizedSegments,
            slidingWindowFeature: slidingFeature,
            embeddingModel: embeddingModel,
            sampleRate: sampleRate
        )

        let embeddingTime = Date().timeIntervalSince(embeddingStartTime)
        let clusteringStartTime = Date()

        // Step 3: Calculate speaker activities
        let speakerActivities = calculateSpeakerActivities(binarizedSegments)

        // Step 4: Assign consistent speaker IDs using global database
        var speakerLabels: [String] = []
        var activityFilteredCount = 0
        var embeddingInvalidCount = 0
        var clusteringProcessedCount = 0

        for (speakerIndex, activity) in speakerActivities.enumerated() {
            if activity > self.config.minActivityThreshold {  // Use configurable activity threshold
                let embedding = embeddings[speakerIndex]
                if validateEmbedding(embedding) {
                    clusteringProcessedCount += 1
                    let speakerId = assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
                    speakerLabels.append(speakerId)
                } else {
                    embeddingInvalidCount += 1
                    speakerLabels.append("")  // Invalid embedding
                }
            } else {
                activityFilteredCount += 1
                speakerLabels.append("")  // No activity
            }
        }

        let clusteringTime = Date().timeIntervalSince(clusteringStartTime)

        // Step 5: Create temporal segments with consistent speaker IDs
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

    /// Apply post-processing filters to segments
    private func applyPostProcessingFilters(_ segments: [TimedSpeakerSegment])
        -> [TimedSpeakerSegment]
    {
        return segments.filter { segment in
            // Apply minimum duration filter
            segment.durationSeconds >= self.config.minDurationOn
        }.compactMap { segment in
            // Additional post-processing could be added here
            return segment
        }
    }

    /// Calculate total activity for each speaker across all frames
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

    /// Assign speaker ID using global database (like main.swift)
    private func assignSpeaker(embedding: [Float], speakerDB: inout [String: [Float]]) -> String {
        if speakerDB.isEmpty {
            let speakerId = "Speaker 1"
            speakerDB[speakerId] = embedding
            return speakerId
        }

        var minDistance: Float = Float.greatestFiniteMagnitude
        var identifiedSpeaker: String? = nil
        var allDistances: [(String, Float)] = []

        for (speakerId, refEmbedding) in speakerDB {
            let distance = cosineDistance(embedding, refEmbedding)
            allDistances.append((speakerId, distance))
            if distance < minDistance {
                minDistance = distance
                identifiedSpeaker = speakerId
            }
        }

        if let bestSpeaker = identifiedSpeaker {
            if minDistance > self.config.clusteringThreshold {
                // New speaker
                let newSpeakerId = "Speaker \(speakerDB.count + 1)"
                speakerDB[newSpeakerId] = embedding
                return newSpeakerId
            } else {
                // Existing speaker - update embedding (exponential moving average)
                updateSpeakerEmbedding(bestSpeaker, embedding, speakerDB: &speakerDB)
                return bestSpeaker
            }
        }

        return "Unknown"
    }

    /// Update speaker embedding with exponential moving average
    private func updateSpeakerEmbedding(
        _ speakerId: String, _ newEmbedding: [Float], speakerDB: inout [String: [Float]],
        alpha: Float = 0.9
    ) {
        guard var oldEmbedding = speakerDB[speakerId] else { return }

        for i in 0..<oldEmbedding.count {
            oldEmbedding[i] = alpha * oldEmbedding[i] + (1 - alpha) * newEmbedding[i]
        }
        speakerDB[speakerId] = oldEmbedding
    }

    /// Create timed segments with speaker IDs
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

        // Find dominant speaker per frame
        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)
            }
        }

        // Group contiguous same-speaker segments
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

        // Final segment
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

    /// Create a segment if the speaker is valid
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

        // Check minimum duration requirement
        if Float(duration) < self.config.minDurationOn {
            return nil
        }

        let embedding = embeddings[speakerIndex]
        let activity = speakerActivities[speakerIndex]
        let quality =
            calculateEmbeddingQuality(embedding) * (activity / Float(endFrame - startFrame))

        return TimedSpeakerSegment(
            speakerId: speakerLabels[speakerIndex],
            embedding: embedding,
            startTimeSeconds: Float(startTime),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
    }
}
