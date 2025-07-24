//
//  DiarizerTypes.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

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
    
    /// Identify the bottleneck stage
    public var bottleneckStage: String {
        let stages = [
            ("Model Download", modelDownloadSeconds),
            ("Model Compilation", modelCompilationSeconds),
            ("Audio Loading", audioLoadingSeconds),
            ("Segmentation", segmentationSeconds),
            ("Embedding Extraction", embeddingExtractionSeconds),
            ("Speaker Clustering", speakerClusteringSeconds),
            ("Post Processing", postProcessingSeconds)
        ]
        
        let maxStage = stages.max(by: { $0.1 < $1.1 })
        return maxStage?.0 ?? "Unknown"
    }
}

/// Complete diarization result including speaker segments and timings
public struct DiarizationResult: Sendable, Codable {
    public let segments: [TimedSpeakerSegment]
    public let speakerDatabase: [String: [Float]]  // Speaker ID to embedding mapping
    public let timings: PipelineTimings

    public init(
        segments: [TimedSpeakerSegment], speakerDatabase: [String: [Float]],
        timings: PipelineTimings = PipelineTimings()
    ) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
        self.timings = timings
    }

    /// Calculate summary statistics about the diarization
    public var summary: DiarizationSummary {
        let totalDuration = segments.last?.endTimeSeconds ?? 0
        let uniqueSpeakers = Set(segments.map { $0.speakerId }).count
        let totalSpeechDuration = segments.reduce(0) { $0 + ($1.endTimeSeconds - $1.startTimeSeconds) }
        let speechRatio = totalDuration > 0 ? totalSpeechDuration / totalDuration : 0

        return DiarizationSummary(
            totalDurationSeconds: totalDuration,
            uniqueSpeakers: uniqueSpeakers,
            totalSpeechDurationSeconds: totalSpeechDuration,
            speechRatio: speechRatio,
            segmentCount: segments.count
        )
    }
}

/// Summary statistics for a diarization result
public struct DiarizationSummary: Sendable, Codable {
    public let totalDurationSeconds: Float
    public let uniqueSpeakers: Int
    public let totalSpeechDurationSeconds: Float
    public let speechRatio: Float
    public let segmentCount: Int
}

/// Individual speaker segment with timing and embedding information
public struct TimedSpeakerSegment: Sendable, Codable {
    public let speakerId: String
    public let embedding: [Float]
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let qualityScore: Float  // 0.0 to 1.0, higher is better

    public var durationSeconds: Float {
        return endTimeSeconds - startTimeSeconds
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

/// Speaker embedding from audio segment
public struct SpeakerEmbedding: Sendable {
    public let embedding: [Float]
    public let qualityScore: Float  // 0.0 to 1.0
    public let durationSeconds: Float

    public init(embedding: [Float], qualityScore: Float, durationSeconds: Float) {
        self.embedding = embedding
        self.qualityScore = qualityScore
        self.durationSeconds = durationSeconds
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
    case modelLoadFailed

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarizer not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download diarization models"
        case .modelCompilationFailed:
            return "Failed to compile CoreML models"
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embeddings"
        case .invalidAudioData:
            return "Invalid audio data provided"
        case .processingFailed(let reason):
            return "Diarization processing failed: \(reason)"
        case .modelLoadFailed:
            return "Failed to load CoreML models"
        }
    }
}

// MARK: - Internal Types

/// For online diarization chunks
public struct DiarizationChunk: Sendable {
    public let startTime: Float
    public let endTime: Float
    public let segments: [TimedSpeakerSegment]
    public let speakerEmbeddings: [String: [Float]]  // Speaker ID to embedding

    public init(
        startTime: Float,
        endTime: Float,
        segments: [TimedSpeakerSegment],
        speakerEmbeddings: [String: [Float]]
    ) {
        self.startTime = startTime
        self.endTime = endTime
        self.segments = segments
        self.speakerEmbeddings = speakerEmbeddings
    }
}