#if os(macOS)
//
//  AsrBenchmarkTypes.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import Foundation

/// ASR evaluation metrics
public struct ASRMetrics: Sendable {
    public let wer: Double  // Word Error Rate
    public let cer: Double  // Character Error Rate
    public let insertions: Int
    public let deletions: Int
    public let substitutions: Int
    public let totalWords: Int
    public let totalCharacters: Int

    public init(
        wer: Double, cer: Double, insertions: Int, deletions: Int, substitutions: Int, totalWords: Int,
        totalCharacters: Int
    ) {
        self.wer = wer
        self.cer = cer
        self.insertions = insertions
        self.deletions = deletions
        self.substitutions = substitutions
        self.totalWords = totalWords
        self.totalCharacters = totalCharacters
    }
}

/// Streaming-specific metrics for ASR benchmarking
public struct StreamingMetrics: Sendable {
    public let avgChunkProcessingTime: Double  // Average time to process each chunk
    public let maxChunkProcessingTime: Double  // Maximum time to process any chunk
    public let minChunkProcessingTime: Double  // Minimum time to process any chunk
    public let totalChunks: Int  // Total number of chunks processed
    public let firstTokenLatency: Double?  // Time to first token (if measurable)
    public let streamingRTFx: Double  // Streaming real-time factor
    public let chunkDuration: Double  // Configured chunk duration in seconds

    public init(
        avgChunkProcessingTime: Double,
        maxChunkProcessingTime: Double,
        minChunkProcessingTime: Double,
        totalChunks: Int,
        firstTokenLatency: Double? = nil,
        streamingRTFx: Double,
        chunkDuration: Double
    ) {
        self.avgChunkProcessingTime = avgChunkProcessingTime
        self.maxChunkProcessingTime = maxChunkProcessingTime
        self.minChunkProcessingTime = minChunkProcessingTime
        self.totalChunks = totalChunks
        self.firstTokenLatency = firstTokenLatency
        self.streamingRTFx = streamingRTFx
        self.chunkDuration = chunkDuration
    }
}

/// Single ASR benchmark result
public struct ASRBenchmarkResult: Sendable {
    public let fileName: String
    public let hypothesis: String
    public let reference: String
    public let metrics: ASRMetrics
    public let processingTime: TimeInterval
    public let audioLength: TimeInterval
    public let rtfx: Double  // Real-Time Factor (inverse)
    public let streamingMetrics: StreamingMetrics?  // Optional streaming metrics

    public init(
        fileName: String, hypothesis: String, reference: String, metrics: ASRMetrics, processingTime: TimeInterval,
        audioLength: TimeInterval, streamingMetrics: StreamingMetrics? = nil
    ) {
        self.fileName = fileName
        self.hypothesis = hypothesis
        self.reference = reference
        self.metrics = metrics
        self.processingTime = processingTime
        self.audioLength = audioLength
        self.rtfx = audioLength / processingTime
        self.streamingMetrics = streamingMetrics
    }
}

/// ASR benchmark configuration
///
/// ## LibriSpeech Dataset Subsets
/// - **test-clean**: Clean, studio-quality recordings with clear speech from native speakers
///   - Easier benchmark subset with minimal noise/accents
///   - Expected WER: 2-6% for good ASR systems
///   - Use for baseline performance evaluation
///
/// - **test-other**: More challenging recordings with various acoustic conditions
///   - Includes accented speech, background noise, and non-native speakers
///   - Expected WER: 5-15% for good ASR systems
///   - Use for robustness testing
///
/// Both subsets contain ~5.4 hours of audio from different speakers reading books.
public struct ASRBenchmarkConfig: Sendable {
    public let dataset: String
    public let subset: String
    public let maxFiles: Int?
    public let debugMode: Bool
    public let longAudioOnly: Bool
    public let testStreaming: Bool
    public let streamingChunkDuration: Double
    public let useStreamingEou: Bool

    public init(
        dataset: String = "librispeech", subset: String = "test-clean", maxFiles: Int? = nil, debugMode: Bool = false,
        longAudioOnly: Bool = false, testStreaming: Bool = false, streamingChunkDuration: Double = 0.1,
        useStreamingEou: Bool = false
    ) {
        self.dataset = dataset
        self.subset = subset
        self.maxFiles = maxFiles
        self.debugMode = debugMode
        self.longAudioOnly = longAudioOnly
        self.testStreaming = testStreaming
        self.streamingChunkDuration = streamingChunkDuration
        self.useStreamingEou = useStreamingEou
    }
}

/// LibriSpeech file representation
public struct LibriSpeechFile {
    public let fileName: String
    public let audioPath: URL
    public let transcript: String
}
#endif
