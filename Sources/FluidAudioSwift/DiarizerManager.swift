
import Foundation
import OSLog
import CoreML

public struct DiarizerConfig: Sendable {
    public var clusteringThreshold: Float = 0.7 // Similarity threshold for grouping speakers (0.0-1.0, higher = stricter)
    public var minDurationOn: Float = 1.0 // Minimum duration (seconds) for a speaker segment to be considered valid
    public var minDurationOff: Float = 0.5 // Minimum silence duration (seconds) between different speakers
    public var numClusters: Int = -1  // Number of speakers to detect (-1 = auto-detect)
    public var debugMode: Bool = false
    public var modelCacheDirectory: URL?

    public static let `default` = DiarizerConfig()

    public init(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.debugMode = debugMode
        self.modelCacheDirectory = modelCacheDirectory
    }
}

/// Represents a speaker segment with timing and speaker information
public struct SpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerClusterId: Int
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let confidenceScore: Float

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    public init(speakerClusterId: Int, startTimeSeconds: Float, endTimeSeconds: Float, confidenceScore: Float = 1.0) {
        self.speakerClusterId = speakerClusterId
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.confidenceScore = confidenceScore
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
    public let segmentationPath: String
    public let embeddingPath: String

    public init(segmentationPath: String, embeddingPath: String) {
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
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
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
    var data: [[[Float]]] // (1, 589, 3)
    var slidingWindow: SlidingWindow
}

// MARK: - Diarizer Implementation

/// Speaker diarization manager
@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager: @unchecked Sendable {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    private let config: DiarizerConfig

    // ML models
    private var segmentationModel: MLModel?
    private var embeddingModel: MLModel?

    public init(config: DiarizerConfig = .default) {
        self.config = config
    }

    public var isAvailable: Bool {
        return segmentationModel != nil && embeddingModel != nil
    }

    public func initialize() async throws {
        logger.info("Initializing diarization system")

        try await cleanupBrokenModels()

        let modelPaths = try await downloadModels()

        let segmentationURL = URL(fileURLWithPath: modelPaths.segmentationPath)
        let embeddingURL = URL(fileURLWithPath: modelPaths.embeddingPath)

        self.segmentationModel = try MLModel(contentsOf: segmentationURL)
        self.embeddingModel = try MLModel(contentsOf: embeddingURL)

        logger.info("Diarization system initialized successfully")
    }

    private func cleanupBrokenModels() async throws {
        let modelsDirectory = getModelsDirectory()
        let segmentationModelPath = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingModelPath = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc")

        if FileManager.default.fileExists(atPath: segmentationModelPath.path) &&
           !isModelCompiled(at: segmentationModelPath) {
            logger.info("Removing broken segmentation model")
            try FileManager.default.removeItem(at: segmentationModelPath)
        }

        if FileManager.default.fileExists(atPath: embeddingModelPath.path) &&
           !isModelCompiled(at: embeddingModelPath) {
            logger.info("Removing broken embedding model")
            try FileManager.default.removeItem(at: embeddingModelPath)
        }
    }

    public func performSegmentation(_ samples: [Float], sampleRate: Int = 16000) async throws -> [SpeakerSegment] {
        guard segmentationModel != nil else {
            throw DiarizerError.notInitialized
        }

        logger.info("Processing \(samples.count) samples for speaker segmentation")

        let chunkSize = sampleRate * 10 // 10 seconds
        var allSegments: [SpeakerSegment] = []

        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, samples.count)
            let chunk = Array(samples[chunkStart..<chunkEnd])

            let chunkSegments = try await processChunk(
                chunk,
                chunkOffset: Double(chunkStart) / Double(sampleRate)
            )
            allSegments.append(contentsOf: chunkSegments)
        }

        return allSegments
    }

    public func extractEmbedding(from samples: [Float]) async throws -> SpeakerEmbedding? {
        guard let embeddingModel = self.embeddingModel else {
            throw DiarizerError.notInitialized
        }

        let chunkSize = 16000 * 10 // 10 seconds
        var paddedChunk = samples
        if samples.count < chunkSize {
            paddedChunk += Array(repeating: 0.0, count: chunkSize - samples.count)
        }

        // Get speaker segments first
        let binarizedSegments = try getSegments(audioChunk: paddedChunk)
        let slidingFeature = createSlidingWindowFeature(binarizedSegments: binarizedSegments)

        let embeddings = try getEmbedding(
            audioChunk: paddedChunk,
            binarizedSegments: binarizedSegments,
            slidingWindowFeature: slidingFeature,
            embeddingModel: embeddingModel
        )

        guard !embeddings.isEmpty, !embeddings[0].isEmpty else {
            return nil
        }

        let embedding = embeddings[0]
        let qualityScore = calculateEmbeddingQuality(embedding)
        let duration = Float(samples.count) / 16000.0

        return SpeakerEmbedding(
            embedding: embedding,
            qualityScore: qualityScore,
            durationSeconds: duration
        )
    }

    private func processChunk(_ chunk: [Float], chunkOffset: Double) async throws -> [SpeakerSegment] {
        let chunkSize = 16000 * 10 // 10 seconds
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            paddedChunk += Array(repeating: 0.0, count: chunkSize - chunk.count)
        }

        // Get speaker segments
        let binarizedSegments = try getSegments(audioChunk: paddedChunk)
        let slidingFeature = createSlidingWindowFeature(binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        var annotations: [Segment: Int] = [:]

        getAnnotation(
            annotation: &annotations,
            binarizedSegments: binarizedSegments,
            slidingWindow: slidingFeature.slidingWindow
        )

        return annotations.map { (segment, speakerIndex) in
            SpeakerSegment(
                speakerClusterId: speakerIndex,
                startTimeSeconds: Float(segment.start),
                endTimeSeconds: Float(segment.end),
                confidenceScore: 1.0
            )
        }
    }

    private func getSegments(audioChunk: [Float], chunkSize: Int = 160_000) throws -> [[[Float]]] {
        guard let segmentationModel = self.segmentationModel else {
            throw DiarizerError.notInitialized
        }

        let audioArray = try MLMultiArray(shape: [1, 1, NSNumber(value: chunkSize)], dataType: .float32)
        for i in 0..<min(audioChunk.count, chunkSize) {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])

        let output = try segmentationModel.prediction(from: input)

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output from segmentation model")
        }

        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        var segments = Array(repeating: Array(repeating: Array(repeating: 0.0 as Float, count: combinations), count: frames), count: 1)

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
            [], // 0
            [0], // 1
            [1], // 2
            [2], // 3
            [0, 1], // 4
            [0, 2], // 5
            [1, 2], // 6
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

    private func createSlidingWindowFeature(binarizedSegments: [[[Float]]], chunkOffset: Double = 0.0) -> SlidingWindowFeature {
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
        chunkSize: Int = 10 * 16000
    ) throws -> [[Float]] {
        let audioTensor = audioChunk
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count

        // Compute clean_frames = 1.0 where active speakers < 2
        var cleanFrames = Array(repeating: Array(repeating: 0.0 as Float, count: 1), count: numFrames)

        for f in 0..<numFrames {
            let frame = slidingWindowFeature.data[0][f]
            let speakerSum = frame.reduce(0, +)
            cleanFrames[f][0] = (speakerSum < 2.0) ? 1.0 : 0.0
        }

        // Multiply slidingWindowSegments.data by cleanFrames
        var cleanSegmentData = Array(
            repeating: Array(repeating: Array(repeating: 0.0 as Float, count: numSpeakers), count: numFrames),
            count: 1
        )

        for f in 0..<numFrames {
            for s in 0..<numSpeakers {
                cleanSegmentData[0][f][s] = slidingWindowFeature.data[0][f][s] * cleanFrames[f][0]
            }
        }

        // Flatten audio tensor to shape (3, 160000)
        var audioBatch: [[Float]] = []
        for _ in 0..<3 {
            audioBatch.append(audioTensor)
        }

        // Transpose mask shape to (3, 589)
        var cleanMasks: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numFrames), count: numSpeakers)

        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                cleanMasks[s][f] = cleanSegmentData[0][f][s]
            }
        }

        // Prepare MLMultiArray inputs
        guard let waveformArray = try? MLMultiArray(shape: [3, chunkSize] as [NSNumber], dataType: .float32),
              let maskArray = try? MLMultiArray(shape: [3, numFrames] as [NSNumber], dataType: .float32) else {
            throw DiarizerError.processingFailed("Failed to allocate MLMultiArray for embeddings")
        }

        for s in 0..<3 {
            for i in 0..<chunkSize {
                waveformArray[s * chunkSize + i] = NSNumber(value: audioBatch[s][i])
            }
        }

        for s in 0..<3 {
            for f in 0..<numFrames {
                maskArray[s * numFrames + f] = NSNumber(value: cleanMasks[s][f])
            }
        }

        // Run model
        let inputs: [String: Any] = [
            "waveform": waveformArray,
            "mask": maskArray,
        ]

        guard let output = try? embeddingModel.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs)),
              let multiArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }

        return convertToSendableArray(multiArray)
    }

    private func convertToSendableArray(_ multiArray: MLMultiArray) -> [[Float]] {
        let shape = multiArray.shape.map { $0.intValue }
        let numRows = shape[0]
        let numCols = shape[1]
        let strides = multiArray.strides.map { $0.intValue }

        var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numCols), count: numRows)

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
        let segmentation = binarizedSegments[0] // shape: [589][3]
        let numFrames = segmentation.count

        // Step 1: argmax to get dominant speaker per frame
        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0) // fallback
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
                annotation[segment] = currentSpeaker // Use raw speaker index
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        // Final segment
        let finalStart = slidingWindow.time(forFrame: startFrame)
        let finalEnd = slidingWindow.segment(forFrame: numFrames - 1).end
        let finalSegment = Segment(start: finalStart, end: finalEnd)
        annotation[finalSegment] = currentSpeaker // Use raw speaker index
    }

    // MARK: - Model Management

    /// Download required models for diarization
    public func downloadModels() async throws -> ModelPaths {
        logger.info("Downloading diarization models from Hugging Face")

        let modelsDirectory = getModelsDirectory()

        let segmentationModelPath = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc").path
        let embeddingModelPath = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc").path

        // Force redownload - remove existing models first
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: segmentationModelPath))
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: embeddingModelPath))
        logger.info("Removed existing models to force fresh download")

        // Download segmentation model bundle from Hugging Face
        try await downloadMLModelCBundle(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "pyannote_segmentation.mlmodelc",
            outputPath: URL(fileURLWithPath: segmentationModelPath)
        )
        logger.info("Downloaded segmentation model bundle from Hugging Face")

        // Download embedding model bundle from Hugging Face
        try await downloadMLModelCBundle(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "wespeaker.mlmodelc",
            outputPath: URL(fileURLWithPath: embeddingModelPath)
        )
        logger.info("Downloaded embedding model bundle from Hugging Face")

        logger.info("Successfully downloaded and compiled diarization models from Hugging Face")
        return ModelPaths(segmentationPath: segmentationModelPath, embeddingPath: embeddingModelPath)
    }

    /// Check if a model is properly compiled
    private func isModelCompiled(at url: URL) -> Bool {
        let coreMLDataPath = url.appendingPathComponent("coremldata.bin")
        return FileManager.default.fileExists(atPath: coreMLDataPath.path)
    }

    /// Download a complete .mlmodelc bundle from Hugging Face
    private func downloadMLModelCBundle(repoPath: String, modelName: String, outputPath: URL) async throws {
        logger.info("Downloading \(modelName) bundle from Hugging Face")

        // Create output directory
        try FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)

        // Files typically found in a .mlmodelc bundle
        let bundleFiles = [
            "model.mil",
            "coremldata.bin",
            "metadata.json"
        ]

        // Weight files that are referenced by model.mil
        let weightFiles = [
            "weights/weight.bin"
        ]

        // Download each file in the bundle
        for fileName in bundleFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(fileName)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(fileName)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(fileName) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(fileName) for \(modelName) - file may not exist")
                    // Create empty file if it doesn't exist (some files are optional)
                    if fileName == "metadata.json" {
                        let destinationPath = outputPath.appendingPathComponent(fileName)
                        try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                    }
                }
            } catch {
                logger.warning("Error downloading \(fileName): \(error.localizedDescription)")
                // For critical files, create minimal versions
                if fileName == "coremldata.bin" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try Data().write(to: destinationPath)
                } else if fileName == "metadata.json" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                }
            }
                }

        // Download weight files
        for weightFile in weightFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(weightFile)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(weightFile)

                    // Create weights directory if it doesn't exist
                    let weightsDir = destinationPath.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(weightFile) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(weightFile) for \(modelName)")
                    throw DiarizerError.modelDownloadFailed
                }
            } catch {
                logger.error("Critical error downloading \(weightFile): \(error.localizedDescription)")
                throw DiarizerError.modelDownloadFailed
            }
        }

        // Also try to download analytics directory if it exists
        let analyticsURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/analytics/coremldata.bin")!
        do {
            let (tempFile, response) = try await URLSession.shared.download(from: analyticsURL)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                let analyticsDir = outputPath.appendingPathComponent("analytics")
                try FileManager.default.createDirectory(at: analyticsDir, withIntermediateDirectories: true)
                let destinationPath = analyticsDir.appendingPathComponent("coremldata.bin")
                try? FileManager.default.removeItem(at: destinationPath)
                try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                logger.info("Downloaded analytics/coremldata.bin for \(modelName)")
            }
        } catch {
            logger.info("Analytics directory not found or not needed for \(modelName)")
        }

        logger.info("Completed downloading \(modelName) bundle")
    }

    /// Compile a model
    private func compileModel(at sourceURL: URL, outputPath: URL) async throws -> URL {
        logger.info("Compiling model from \(sourceURL.lastPathComponent)")

        // Remove existing compiled model if it exists
        if FileManager.default.fileExists(atPath: outputPath.path) {
            try FileManager.default.removeItem(at: outputPath)
        }

        // Compile the model
        let compiledModelURL = try await MLModel.compileModel(at: sourceURL)

        // Move to the desired location
        try FileManager.default.moveItem(at: compiledModelURL, to: outputPath)

        // Clean up the source file
        try? FileManager.default.removeItem(at: sourceURL)

        logger.info("Successfully compiled model to \(outputPath.lastPathComponent)")
        return outputPath
    }

    private func getModelsDirectory() -> URL {
        let directory: URL

        if let customDirectory = config.modelCacheDirectory {
            directory = customDirectory.appendingPathComponent("coreml", isDirectory: true)
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            directory = appSupport.appendingPathComponent("SpeakerKitModels/coreml", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    // MARK: - Audio Analysis

    /// Compare similarity between two audio samples
    public func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float {
        let embedding1 = try await extractEmbedding(from: audio1)
        let embedding2 = try await extractEmbedding(from: audio2)

        guard let emb1 = embedding1, let emb2 = embedding2 else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let distance = cosineDistance(emb1.embedding, emb2.embedding)
        return max(0, (1.0 - distance) * 100) // Convert to similarity percentage
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
            logger.error("Invalid embeddings for distance calculation")
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
            logger.info("Zero magnitude embedding detected")
            return Float.infinity
        }

        let similarity = dotProduct / (magnitudeA * magnitudeB)
        return 1 - similarity
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

    // MARK: - Cleanup

    /// Clean up resources
    public func cleanup() async {
        segmentationModel = nil
        embeddingModel = nil
        logger.info("Diarization resources cleaned up")
    }
}

