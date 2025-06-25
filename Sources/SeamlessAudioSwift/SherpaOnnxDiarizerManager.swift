//
//  SherpaOnnxDiarizerManager.swift
//  SeamlessAudioSwift
//
//  Created by Kiko Wei on 2025-06-24.
//

import Foundation
import OSLog
import SherpaOnnxWrapper

// MARK: - SherpaOnnx Implementation

/// SherpaOnnx implementation of speaker diarization
@available(macOS 13.0, iOS 16.0, *)
public final class SherpaOnnxDiarizerManager: DiarizerManager, @unchecked Sendable {
    public let backend: DiarizerBackend = .sherpaOnnx

    private let logger = Logger(subsystem: "com.speakerkit", category: "SherpaOnnxDiarizer")
    private let config: DiarizerConfig

    // SherpaOnnx components
    private var diarizer: SherpaOnnxOfflineSpeakerDiarizationWrapper?
    private var embeddingExtractor: OpaquePointer?

    public init(config: DiarizerConfig = .default) {
        self.config = config
    }

    // MARK: - Initialization

    /// Initialize the speaker diarization system
    /// This downloads models if needed and sets up the processing pipeline
    public func initialize() async throws {
        logger.info("Initializing SherpaOnnx diarization system")

        let modelPaths = try await downloadModels()

        // Setup diarizer with SherpaOnnx config
        var diarizationConfig = sherpaOnnxOfflineSpeakerDiarizationConfig(
            segmentation: sherpaOnnxOfflineSpeakerSegmentationModelConfig(
                pyannote: sherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(
                    model: modelPaths.segmentationPath
                )
            ),
            embedding: sherpaOnnxSpeakerEmbeddingExtractorConfig(
                model: modelPaths.embeddingPath
            ),
            clustering: sherpaOnnxFastClusteringConfig(
                numClusters: config.numClusters,
                threshold: config.clusteringThreshold
            ),
            minDurationOn: config.minDurationOn,
            minDurationOff: config.minDurationOff
        )
        diarizationConfig.segmentation.debug = config.debugMode ? 1 : 0

        self.diarizer = SherpaOnnxOfflineSpeakerDiarizationWrapper(config: &diarizationConfig)

        // Setup embedding extractor
        var embeddingConfig = sherpaOnnxSpeakerEmbeddingExtractorConfig(
            model: modelPaths.embeddingPath,
            numThreads: 1,
            debug: config.debugMode ? 1 : 0,
            provider: "cpu"
        )

        self.embeddingExtractor = SherpaOnnxCreateSpeakerEmbeddingExtractor(&embeddingConfig)

        logger.info("SherpaOnnx diarization system initialized successfully")
    }

    /// Check if the diarization system is ready for use
    public var isAvailable: Bool {
        return diarizer != nil && embeddingExtractor != nil
    }

    // MARK: - Core Processing

    /// Perform speaker segmentation on audio samples
    /// - Parameters:
    ///   - samples: Audio samples as Float array (16kHz recommended)
    ///   - sampleRate: Sample rate of the audio (default: 16000)
    /// - Returns: Array of speaker segments with timing and speaker IDs
    public func performSegmentation(_ samples: [Float], sampleRate: Int = 16000) async throws -> [SpeakerSegment] {
        guard let diarizer = self.diarizer else {
            throw DiarizerError.notInitialized
        }

        logger.info("Processing \(samples.count) samples for speaker segmentation")

        let sherpaSegments = diarizer.process(samples: samples)

        return sherpaSegments.map { segment in
            SpeakerSegment(
                speakerClusterId: segment.speaker,
                startTimeSeconds: segment.start,
                endTimeSeconds: segment.end,
                confidenceScore: 1.0 // SherpaOnnx doesn't provide confidence scores
            )
        }
    }

    /// Extract speaker embedding from audio samples
    /// - Parameter samples: Audio samples as Float array
    /// - Returns: Speaker embedding with quality metrics, or nil if extraction fails
    public func extractEmbedding(from samples: [Float]) async throws -> SpeakerEmbedding? {
        guard let embeddingExtractor = self.embeddingExtractor else {
            throw DiarizerError.notInitialized
        }

        // Create stream for this batch of samples
        let stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(embeddingExtractor)
        guard stream != nil else {
            logger.error("Failed to create embedding stream")
            return nil
        }

        defer {
            SherpaOnnxDestroyOnlineStream(stream)
        }

        // Feed audio samples to the stream
        SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples, Int32(samples.count))

        // Check if ready for processing
        guard SherpaOnnxSpeakerEmbeddingExtractorIsReady(embeddingExtractor, stream) != 0 else {
            logger.info("Not ready for processing, need more audio")
            return nil
        }

        // Compute the embedding
        guard let embeddingPtr = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(embeddingExtractor, stream) else {
            logger.error("Failed to compute embedding")
            return nil
        }

        defer {
            SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embeddingPtr)
        }

        // Get embedding dimension and convert to Swift array
        let embeddingDim = SherpaOnnxSpeakerEmbeddingExtractorDim(embeddingExtractor)
        var embeddingArray: [Float] = []
        for i in 0..<embeddingDim {
            embeddingArray.append(embeddingPtr[Int(i)])
        }

        let qualityScore = calculateEmbeddingQuality(embeddingArray)
        let duration = Float(samples.count) / 16000.0

        return SpeakerEmbedding(
            embedding: embeddingArray,
            qualityScore: qualityScore,
            durationSeconds: duration
        )
    }

    // MARK: - Model Management

    /// Download required models for diarization
    /// Models are cached locally and only downloaded if not present
    public func downloadModels() async throws -> ModelPaths {
        logger.info("Downloading diarization models")

        let modelsDirectory = getModelsDirectory()

        let segmentationModelPath = modelsDirectory.appendingPathComponent("pyannote_segmentation_3.onnx").path
        let embeddingModelPath = modelsDirectory.appendingPathComponent("3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx").path

        // Check if models already exist
        if FileManager.default.fileExists(atPath: segmentationModelPath) &&
            FileManager.default.fileExists(atPath: embeddingModelPath) {
            logger.info("Diarization models already exist locally")
            return ModelPaths(segmentationPath: segmentationModelPath, embeddingPath: embeddingModelPath)
        }

        // Download segmentation model
        if !FileManager.default.fileExists(atPath: segmentationModelPath) {
            let segmentationURL = URL(string: "https://assets.slipbox.ai/dist/pyannote_segmentation_3.onnx")!
            let (tempFile, _) = try await URLSession.shared.download(from: segmentationURL)
            try FileManager.default.moveItem(at: tempFile, to: URL(fileURLWithPath: segmentationModelPath))
            logger.info("Downloaded segmentation model")
        }

        // Download embedding model
        if !FileManager.default.fileExists(atPath: embeddingModelPath) {
            let embeddingURL = URL(string: "https://assets.slipbox.ai/dist/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx")!
            let (tempFile, _) = try await URLSession.shared.download(from: embeddingURL)
            try FileManager.default.moveItem(at: tempFile, to: URL(fileURLWithPath: embeddingModelPath))
            logger.info("Downloaded embedding model")
        }

        logger.info("Successfully downloaded diarization models")
        return ModelPaths(segmentationPath: segmentationModelPath, embeddingPath: embeddingModelPath)
    }

    private func getModelsDirectory() -> URL {
        let directory: URL

        if let customDirectory = config.modelCacheDirectory {
            directory = customDirectory.appendingPathComponent("sherpa-onnx", isDirectory: true)
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            directory = appSupport.appendingPathComponent("SpeakerKitModels/sherpa-onnx", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    // MARK: - Audio Analysis

    /// Compare similarity between two audio samples
    /// - Parameters:
    ///   - audio1: First audio sample
    ///   - audio2: Second audio sample
    /// - Returns: Similarity score as percentage (0-100)
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
        if let extractor = embeddingExtractor {
            SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor)
        }
        embeddingExtractor = nil
        diarizer = nil
        logger.info("SherpaOnnx diarization resources cleaned up")
    }
}

