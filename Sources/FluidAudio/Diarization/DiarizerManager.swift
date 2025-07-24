//
//  DiarizerManager.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog

// MARK: - Internal Types

// MARK: - Sliding Window Support

struct Segment {
    var start: Double
    var end: Double

    var center: Double { (start + end) / 2 }
    var duration: Double { end - start }
}

struct SlidingWindow {
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

    // ML models
    private var segmentationModel: MLModel?
    private var embeddingModel: MLModel?

    // Timing tracking
    private var modelDownloadTime: TimeInterval = 0
    private var modelCompilationTime: TimeInterval = 0

    /// Initialize DiarizerManager with configuration only (models will be loaded later)
    public init(config: DiarizerConfig = .default) {
        self.config = config
    }

    /// Initialize DiarizerManager with pre-loaded models
    public init(config: DiarizerConfig = .default, models: DiarizationModels) {
        self.config = config
        self.segmentationModel = models.segmentation
        self.embeddingModel = models.embedding
        self.modelDownloadTime = 0 // No download needed
        self.modelCompilationTime = 0 // Already compiled
        logger.info("DiarizerManager initialized with provided models")
    }

    public var isAvailable: Bool {
        return segmentationModel != nil && embeddingModel != nil
    }

    /// Get the initialization timing data
    public var initializationTimings: (downloadTime: TimeInterval, compilationTime: TimeInterval) {
        return (modelDownloadTime, modelCompilationTime)
    }

    /// Initialize with auto-downloaded models
    /// - Parameter directory: Optional directory to load models from
    public func initialize(from directory: URL? = nil) async throws {
        let initStartTime = Date()
        logger.info("Initializing diarization system")

        let downloadStartTime = Date()
        let models = try await DiarizationModels.load(from: directory)
        self.modelDownloadTime = Date().timeIntervalSince(downloadStartTime)

        let compilationStartTime = Date()
        try await initialize(models: models)
        self.modelCompilationTime = Date().timeIntervalSince(compilationStartTime)

        let totalInitTime = Date().timeIntervalSince(initStartTime)
        logger.info(
            "Diarization system initialized successfully in \(String(format: "%.2f", totalInitTime))s (download: \(String(format: "%.2f", self.modelDownloadTime))s, compilation: \(String(format: "%.2f", self.modelCompilationTime))s)"
        )
    }

    /// Initialize with pre-loaded models
    public func initialize(segmentationModel: MLModel, embeddingModel: MLModel) async throws {
        logger.info("Initializing diarization system with pre-loaded models")

        self.segmentationModel = segmentationModel
        self.embeddingModel = embeddingModel
        self.modelDownloadTime = 0 // No download needed
        self.modelCompilationTime = 0 // Already compiled

        logger.info("Diarization system initialized successfully")
    }

    /// Initialize with pre-loaded models using convenience struct
    public func initialize(models: DiarizationModels) async throws {
        try await initialize(segmentationModel: models.segmentation, embeddingModel: models.embedding)
    }
    
    /// Initialize with DiarizerModels (for test compatibility)
    public func initialize(models: DiarizerModels) {
        self.segmentationModel = models.segmentationModel
        self.embeddingModel = models.embeddingModel
        self.modelDownloadTime = TimeInterval(models.downloadTime.components.seconds)
        self.modelCompilationTime = TimeInterval(models.compilationTime.components.seconds)
        logger.info("DiarizerManager initialized with DiarizerModels")
    }

    /// Load models with automatic recovery on compilation failures
    private func loadModelsWithAutoRecovery(
        segmentationURL: URL, embeddingURL: URL, maxRetries: Int = 2
    ) async throws {

        var lastError: Error?

        for attempt in 0...maxRetries {
            do {
                if attempt > 0 {
                    logger.info("🔄 Recovery attempt \(attempt) of \(maxRetries)")
                }

                // Try to load models with optimized configuration
                let config = MLModelConfiguration()
                config.allowLowPrecisionAccumulationOnGPU = true

                // Use cpuAndNeuralEngine only if we're running in CI environment
                let isCI = ProcessInfo.processInfo.environment["CI"] != nil
                config.computeUnits = isCI ? .cpuAndNeuralEngine : .all

                logger.info("Loading segmentation model from: \(segmentationURL.lastPathComponent)")
                self.segmentationModel = try MLModel(contentsOf: segmentationURL, configuration: config)

                logger.info("Loading embedding model from: \(embeddingURL.lastPathComponent)")
                self.embeddingModel = try MLModel(contentsOf: embeddingURL, configuration: config)

                logger.info("✅ Models loaded successfully")
                return

            } catch {
                lastError = error
                logger.error("❌ Model loading failed (attempt \(attempt + 1)): \(error.localizedDescription)")

                // If this was our last attempt, throw the error
                if attempt >= maxRetries {
                    throw DiarizerError.modelCompilationFailed
                }

                // Perform recovery: delete and re-download
                logger.info("🔧 Attempting recovery by re-downloading models...")
                try await performModelRecovery()
            }
        }

        // This should never be reached, but just in case
        throw lastError ?? DiarizerError.modelCompilationFailed
    }

    /// Perform model recovery by deleting and re-downloading
    private func performModelRecovery() async throws {
        logger.info("🗑️ Deleting corrupted model files...")

        // Get the models directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let baseDirectory = appSupport.appendingPathComponent("FluidAudio")
        let modelsDirectory = baseDirectory.appendingPathComponent("speaker-diarization-coreml")

        // Delete the entire models directory
        if FileManager.default.fileExists(atPath: modelsDirectory.path) {
            try FileManager.default.removeItem(at: modelsDirectory)
            logger.info("✅ Deleted corrupted models directory")
        }

        // Re-download models
        logger.info("📥 Re-downloading models...")
        _ = try await downloadModels()
        logger.info("✅ Models re-downloaded successfully")
    }

    /// Download diarization models
    public func downloadModels() async throws -> ModelPaths {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let baseDirectory = appSupport.appendingPathComponent("FluidAudio")

        _ = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: ["pyannote_segmentation.mlmodelc", "wespeaker.mlmodelc"],
            directory: baseDirectory
        )

        // Construct paths from the downloaded models
        let modelsDir = baseDirectory.appendingPathComponent("speaker-diarization-coreml")
        let segmentationPath = modelsDir.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingPath = modelsDir.appendingPathComponent("wespeaker.mlmodelc")

        return ModelPaths(segmentationPath: segmentationPath, embeddingPath: embeddingPath)
    }

    // MARK: - Inference Methods

    /// Process audio file and return diarization results
    public func performCompleteDiarization(
        _ audioSamples: [Float], sampleRate: Int = 16000
    ) async throws -> DiarizationResult {
        let timings = PipelineTimings()

        guard isAvailable else {
            logger.error("Models not loaded - segmentationModel: \(self.segmentationModel != nil), embeddingModel: \(self.embeddingModel != nil)")
            throw DiarizerError.notInitialized
        }

        do {
            let chunkSize = sampleRate * 10  // 10 seconds
            var allSegments: [TimedSpeakerSegment] = []
            var speakerDB: [String: [Float]] = [:]  // Global speaker database

            var segmentationTotalTime = 0.0

            // Process in 10-second chunks
            for chunkStart in stride(from: 0, to: audioSamples.count, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, audioSamples.count)
                let chunk = Array(audioSamples[chunkStart..<chunkEnd])
                let chunkOffset = Double(chunkStart) / Double(sampleRate)

                // Process chunk with timing
                let segmentationStartTime = Date()
                let chunkSegments = try processChunkWithSpeakerTracking(
                    chunk,
                    chunkOffset: chunkOffset,
                    speakerDB: &speakerDB,
                    sampleRate: sampleRate
                )
                segmentationTotalTime += Date().timeIntervalSince(segmentationStartTime)

                allSegments.append(contentsOf: chunkSegments)
            }

            // Post-process all segments
            let postProcessStartTime = Date()
            let processedSegments = postProcessSegments(allSegments)
            let postProcessTime = Date().timeIntervalSince(postProcessStartTime)

            let finalTimings = PipelineTimings(
                modelDownloadSeconds: self.modelDownloadTime,
                modelCompilationSeconds: self.modelCompilationTime,
                segmentationSeconds: segmentationTotalTime,
                embeddingExtractionSeconds: 0,  // Included in segmentation time
                speakerClusteringSeconds: 0,     // Included in segmentation time
                postProcessingSeconds: postProcessTime
            )

            return DiarizationResult(
                segments: processedSegments,
                speakerDatabase: speakerDB,
                timings: finalTimings
            )

        } catch {
            print("❌ Diarization error: \(error.localizedDescription)")
            logger.error("Diarization failed: \(error.localizedDescription)")
            return DiarizationResult(segments: [], speakerDatabase: [:], timings: timings)
        }
    }

    /// Process audio chunk for online diarization
    public func processDiarizationChunk(
        _ audioSamples: [Float],
        chunkStartTime: Float,
        existingSpeakers: [String: [Float]] = [:],
        sampleRate: Int = 16000
    ) -> DiarizationChunk {

        guard isAvailable else {
            logger.error("Models not loaded")
            return DiarizationChunk(
                startTime: chunkStartTime,
                endTime: chunkStartTime,
                segments: [],
                speakerEmbeddings: [:]
            )
        }

        do {
            var speakerDB = existingSpeakers
            let chunkOffset = Double(chunkStartTime)

            // Process chunk with existing speaker database
            let segments = try processChunkWithSpeakerTracking(
                audioSamples,
                chunkOffset: chunkOffset,
                speakerDB: &speakerDB,
                sampleRate: sampleRate
            )

            let processedSegments = postProcessSegments(segments)
            let chunkEndTime = chunkStartTime + Float(audioSamples.count) / Float(sampleRate)

            return DiarizationChunk(
                startTime: chunkStartTime,
                endTime: chunkEndTime,
                segments: processedSegments,
                speakerEmbeddings: speakerDB
            )

        } catch {
            logger.error("Chunk diarization failed: \(error)")
            return DiarizationChunk(
                startTime: chunkStartTime,
                endTime: chunkStartTime,
                segments: [],
                speakerEmbeddings: [:]
            )
        }
    }

    // MARK: - Private Implementation Methods

    /// Process a single chunk with speaker tracking across chunks
    private func processChunkWithSpeakerTracking(
        _ chunk: [Float],
        chunkOffset: Double,
        speakerDB: inout [String: [Float]],
        sampleRate: Int = 16000
    ) throws -> [TimedSpeakerSegment] {
        let chunkSize = sampleRate * 10  // 10 seconds
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            paddedChunk += Array(repeating: 0.0, count: chunkSize - chunk.count)
        }

        // Step 1: Get segmentation (when speakers are active)
        let binarizedSegments = try getSegments(audioChunk: paddedChunk)
        let slidingFeature = createSlidingWindowFeature(
            binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        // Step 2: Get embeddings using same segmentation results
        guard let embeddingModel = self.embeddingModel else {
            throw DiarizerError.notInitialized
        }

        let embeddings = try getEmbedding(
            audioChunk: paddedChunk,
            binarizedSegments: binarizedSegments,
            slidingWindowFeature: slidingFeature,
            embeddingModel: embeddingModel,
            sampleRate: sampleRate
        )

        // Step 3: Calculate speaker activities
        let speakerActivities = calculateSpeakerActivities(binarizedSegments)

        // Step 4: Assign consistent speaker IDs using global database
        var speakerLabels: [String] = []

        for (speakerIndex, activity) in speakerActivities.enumerated() {
            if activity > self.config.minActivityThreshold {
                let embedding = embeddings[speakerIndex]
                if validateEmbedding(embedding) {
                    let speakerId = assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
                    speakerLabels.append(speakerId)
                } else {
                    speakerLabels.append("")  // Invalid embedding
                }
            } else {
                speakerLabels.append("")  // No activity
            }
        }

        // Step 5: Create temporal segments with consistent speaker IDs
        return createTimedSegments(
            binarizedSegments: binarizedSegments,
            slidingWindow: slidingFeature.slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        )
    }





    private func postProcessSegments(_ segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        var processed = segments

        // 1. Merge nearby segments from same speaker
        processed = mergeNearbySegments(processed)

        // 2. Filter out very short segments
        processed = processed.filter { $0.durationSeconds >= config.minDurationOn }

        // 3. Sort by start time
        processed.sort { $0.startTimeSeconds < $1.startTimeSeconds }

        return processed
    }

    private func mergeNearbySegments(_ segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        var merged: [TimedSpeakerSegment] = []
        var currentSegment = segments[0]

        for i in 1..<segments.count {
            let nextSegment = segments[i]

            // Check if same speaker and close enough
            if currentSegment.speakerId == nextSegment.speakerId &&
               nextSegment.startTimeSeconds - currentSegment.endTimeSeconds < config.minDurationOff {
                // Merge segments
                currentSegment = TimedSpeakerSegment(
                    speakerId: currentSegment.speakerId,
                    embedding: averageEmbeddings([currentSegment.embedding, nextSegment.embedding]),
                    startTimeSeconds: currentSegment.startTimeSeconds,
                    endTimeSeconds: nextSegment.endTimeSeconds,
                    qualityScore: (currentSegment.qualityScore + nextSegment.qualityScore) / 2
                )
            } else {
                // Save current and start new
                merged.append(currentSegment)
                currentSegment = nextSegment
            }
        }

        merged.append(currentSegment)
        return merged
    }

    private func buildSpeakerDatabase(from segments: [TimedSpeakerSegment]) -> [String: [Float]] {
        var database: [String: [[Float]]] = [:]

        // Collect all embeddings for each speaker
        for segment in segments {
            database[segment.speakerId, default: []].append(segment.embedding)
        }

        // Average embeddings for each speaker
        var averagedDatabase: [String: [Float]] = [:]
        for (speaker, embeddings) in database {
            averagedDatabase[speaker] = averageEmbeddings(embeddings)
        }

        return averagedDatabase
    }

    // MARK: - Core Processing Methods

    private func getSegments(audioChunk: [Float], chunkSize: Int = 160_000) throws -> [[[Float]]] {
        guard let segmentationModel = self.segmentationModel else {
            throw DiarizerError.notInitialized
        }

        let audioArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: chunkSize)], dataType: .float32)
        for i in 0..<min(audioChunk.count, chunkSize) {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }

        let input = pyannote_segmentationInput(audio: audioArray)
        let output = try segmentationModel.prediction(from: input)

        // Try different possible output names
        var outputFeature: MLFeatureValue?
        for name in ["segments", "output", "var_2086", "output0", "Identity"] {
            if let feature = output.featureValue(for: name) {
                outputFeature = feature
                break
            }
        }

        guard let feature = outputFeature,
              let segmentOutput = feature.multiArrayValue else {
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
        audioChunk: [Float],
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

        // Flatten audio tensor to shape (numSpeakers, 160000)
        var audioBatch: [[Float]] = []
        for _ in 0..<numSpeakers {
            audioBatch.append(audioTensor)
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
            for i in 0..<chunkSize {
                waveformArray[s * chunkSize + i] = NSNumber(value: audioBatch[s][i])
            }
        }

        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                maskArray[s * numFrames + f] = NSNumber(value: cleanMasks[s][f])
            }
        }

        // Run model
        let input = wespeakerInput(audio: waveformArray, mask: maskArray)
        let output = try embeddingModel.prediction(from: input)

        // Try different possible output names
        var embeddingFeature: MLFeatureValue?
        for name in ["embedding", "embeddings", "output", "var_448", "Identity"] {
            if let feature = output.featureValue(for: name) {
                embeddingFeature = feature
                break
            }
        }

        guard let feature = embeddingFeature,
              let multiArray = feature.multiArrayValue else {
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

    /// Assign speaker ID using global database
    private func assignSpeaker(embedding: [Float], speakerDB: inout [String: [Float]]) -> String {
        if speakerDB.isEmpty {
            let speakerId = "Speaker 1"
            speakerDB[speakerId] = embedding
            return speakerId
        }

        var minDistance: Float = Float.greatestFiniteMagnitude
        var identifiedSpeaker: String? = nil

        for (speakerId, refEmbedding) in speakerDB {
            let distance = cosineDistance(embedding, refEmbedding)
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

    /// Compare similarity between two audio samples using efficient diarization
    public func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float {
        // Use the efficient method to get embeddings
        let result1 = try await performCompleteDiarization(audio1)
        let result2 = try await performCompleteDiarization(audio2)

        // Get the most representative embedding from each audio
        guard let segment1 = result1.segments.max(by: { $0.qualityScore < $1.qualityScore }),
              let segment2 = result2.segments.max(by: { $0.qualityScore < $1.qualityScore })
        else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let distance = cosineDistance(segment1.embedding, segment2.embedding)
        return max(0, (1.0 - distance) * 100)  // Convert to similarity percentage
    }

    /// Clean up resources
    public func cleanup() async {
        segmentationModel = nil
        embeddingModel = nil
        logger.info("Diarization resources cleaned up")
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

    public func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        normA = sqrt(normA)
        normB = sqrt(normB)

        guard normA > 0 && normB > 0 else { return Float.infinity }
        let similarity = dotProduct / (normA * normB)
        return 1 - similarity
    }

    // MARK: - Utility Functions

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        normA = sqrt(normA)
        normB = sqrt(normB)

        guard normA > 0 && normB > 0 else { return 0 }
        return dotProduct / (normA * normB)
    }

    private func calculateEmbeddingQuality(_ embedding: [Float]) -> Float {
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        // Simple quality score based on magnitude
        return min(1.0, magnitude / 10.0)
    }

    private func calculateRMSEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let squaredSum = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(squaredSum / Float(samples.count))
    }


    private func averageEmbeddings(_ embeddings: [[Float]]) -> [Float] {
        guard !embeddings.isEmpty else { return [] }

        let dimension = embeddings[0].count
        var average = Array(repeating: Float(0), count: dimension)

        for embedding in embeddings {
            for i in 0..<dimension {
                average[i] += embedding[i]
            }
        }

        let count = Float(embeddings.count)
        for i in 0..<dimension {
            average[i] /= count
        }

        return average
    }

    // MARK: - Audio Validation

    /// Validate audio data before processing
    public func validateAudio(_ audioSamples: [Float], sampleRate: Int = 16000) -> AudioValidationResult {
        let durationSeconds = Float(audioSamples.count) / Float(sampleRate)
        var issues: [String] = []

        // Check if audio is empty
        if audioSamples.isEmpty {
            issues.append("No audio data")
            return AudioValidationResult(
                isValid: false,
                durationSeconds: 0,
                issues: issues
            )
        }

        // Check duration
        if durationSeconds < 1.0 {
            issues.append("Audio too short (minimum 1 second)")
        }

        if durationSeconds > 3600 {
            issues.append("Audio too long (maximum 1 hour)")
        }

        // Check if audio is silent
        let maxAmplitude = audioSamples.map { abs($0) }.max() ?? 0
        if maxAmplitude < 0.001 {
            issues.append("Audio too quiet or silent")
        }

        // Check sample rate
        if sampleRate != 16000 {
            issues.append("Sample rate should be 16kHz for optimal performance")
        }

        return AudioValidationResult(
            isValid: issues.isEmpty,
            durationSeconds: durationSeconds,
            issues: issues
        )
    }
}

// MARK: - CoreML Model Input/Output Classes

@available(macOS 13.0, iOS 16.0, *)
class pyannote_segmentationInput: MLFeatureProvider {
    var audio: MLMultiArray

    var featureNames: Set<String> {
        return ["audio"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "audio" {
            return MLFeatureValue(multiArray: audio)
        }
        return nil
    }

    init(audio: MLMultiArray) {
        self.audio = audio
    }
}

@available(macOS 13.0, iOS 16.0, *)
class wespeakerInput: MLFeatureProvider {
    var waveform: MLMultiArray
    var mask: MLMultiArray

    var featureNames: Set<String> {
        return ["waveform", "mask"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "waveform" {
            return MLFeatureValue(multiArray: waveform)
        } else if featureName == "mask" {
            return MLFeatureValue(multiArray: mask)
        }
        return nil
    }

    init(audio: MLMultiArray, mask: MLMultiArray) {
        self.waveform = audio
        self.mask = mask
    }
}
