import Foundation

public enum EmbeddingMode: String, Sendable, CaseIterable {
    case full = "full"              // Full embedding extraction (10.5s)
    case fast = "fast"              // Fast mode with optimizations
    case skip = "skip"              // Skip embedding extraction entirely
    case cached = "cached"          // Use cached embeddings if available
}

public struct DiarizerConfig: Sendable {
    public var clusteringThreshold: Float = 0.7
    public var minDurationOn: Float = 1.0
    public var minDurationOff: Float = 0.5
    public var numClusters: Int = -1
    public var minActivityThreshold: Float = 10.0
    public var debugMode: Bool = false
    public var embeddingMode: EmbeddingMode = .full

    public static let `default` = DiarizerConfig()

    public init(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        minActivityThreshold: Float = 10.0,
        debugMode: Bool = false,
        embeddingMode: EmbeddingMode = .full
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.minActivityThreshold = minActivityThreshold
        self.debugMode = debugMode
        self.embeddingMode = embeddingMode
    }
}

public struct PipelineTimings: Sendable, Codable {
    public let modelDownloadSeconds: TimeInterval
    public let modelCompilationSeconds: TimeInterval
    public let audioLoadingSeconds: TimeInterval
    public let segmentationSeconds: TimeInterval
    public let embeddingExtractionSeconds: TimeInterval
    public let speakerClusteringSeconds: TimeInterval
    public let postProcessingSeconds: TimeInterval
    public let totalInferenceSeconds: TimeInterval
    public let totalProcessingSeconds: TimeInterval

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

public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]
    public let speakerDatabase: [String: [Float]]
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

public struct TimedSpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerId: String
    public let embedding: [Float]
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let qualityScore: Float

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