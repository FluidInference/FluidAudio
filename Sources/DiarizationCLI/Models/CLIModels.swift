#if os(macOS)
import Foundation
import FluidAudio

// MARK: - Errors

public enum CLIError: Error {
case invalidArgument(String)
}

// MARK: - Processing Results

public struct ProcessingResult: Codable {
public let audioFile: String
public let durationSeconds: Float
public let processingTimeSeconds: TimeInterval
public let realTimeFactor: Float
public let segments: [TimedSpeakerSegment]
public let speakerCount: Int
public let config: DiarizerConfig
public let timestamp: Date

public init(
    audioFile: String, durationSeconds: Float, processingTimeSeconds: TimeInterval,
    realTimeFactor: Float, segments: [TimedSpeakerSegment], speakerCount: Int,
    config: DiarizerConfig
) {
    self.audioFile = audioFile
    self.durationSeconds = durationSeconds
    self.processingTimeSeconds = processingTimeSeconds
    self.realTimeFactor = realTimeFactor
    self.segments = segments
    self.speakerCount = speakerCount
    self.config = config
    self.timestamp = Date()
}
}

// MARK: - Benchmark Results

public struct BenchmarkResult: Codable {
public let meetingId: String
public let durationSeconds: Float
public let processingTimeSeconds: TimeInterval
public let realTimeFactor: Float
public let der: Float
public let jer: Float
public let segments: [TimedSpeakerSegment]
public let speakerCount: Int
public let groundTruthSpeakerCount: Int
public let timings: PipelineTimings

/// Total time including audio loading
public var totalExecutionTime: TimeInterval {
    return timings.totalProcessingSeconds + timings.audioLoadingSeconds
}
}

public struct BenchmarkSummary: Codable {
public let dataset: String
public let averageDER: Float
public let averageJER: Float
public let processedFiles: Int
public let totalFiles: Int
public let results: [BenchmarkResult]
public let timestamp: Date

public init(
    dataset: String, averageDER: Float, averageJER: Float, processedFiles: Int,
    totalFiles: Int,
    results: [BenchmarkResult]
) {
    self.dataset = dataset
    self.averageDER = averageDER
    self.averageJER = averageJER
    self.processedFiles = processedFiles
    self.totalFiles = totalFiles
    self.results = results
    self.timestamp = Date()
}
}

public struct DiarizationMetrics {
public let der: Float
public let jer: Float
public let missRate: Float
public let falseAlarmRate: Float
public let speakerErrorRate: Float
public let mappedSpeakerCount: Int  // Number of predicted speakers that mapped to ground truth
}

// MARK: - VAD Benchmark Data Structures

public struct VadTestFile {
public let name: String
public let expectedLabel: Int  // 0 = no speech, 1 = speech
public let url: URL
}

public struct VadBenchmarkResult {
public let testName: String
public let accuracy: Float
public let precision: Float
public let recall: Float
public let f1Score: Float
public let processingTime: TimeInterval
public let totalFiles: Int
public let correctPredictions: Int
}

// MARK: - AMI Dataset

public enum AMIVariant: String, CaseIterable {
case sdm = "sdm"  // Single Distant Microphone (Mix-Headset.wav)
case ihm = "ihm"  // Individual Headset Microphones (Headset-0.wav)

public var displayName: String {
    switch self {
    case .sdm: return "Single Distant Microphone"
    case .ihm: return "Individual Headset Microphones"
    }
}

public var filePattern: String {
    switch self {
    case .sdm: return "Mix-Headset.wav"
    case .ihm: return "Headset-0.wav"
    }
}
}

// MARK: - DiarizerConfig Extensions

extension DiarizerConfig: Codable {
enum CodingKeys: String, CodingKey {
    case clusteringThreshold
    case minDurationOn
    case minDurationOff
    case numClusters
    case minActivityThreshold
    case debugMode
    case modelCacheDirectory
}

public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(clusteringThreshold, forKey: .clusteringThreshold)
    try container.encode(minDurationOn, forKey: .minDurationOn)
    try container.encode(minDurationOff, forKey: .minDurationOff)
    try container.encode(numClusters, forKey: .numClusters)
    try container.encode(minActivityThreshold, forKey: .minActivityThreshold)
    try container.encode(debugMode, forKey: .debugMode)
    try container.encodeIfPresent(modelCacheDirectory, forKey: .modelCacheDirectory)
}

public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let clusteringThreshold = try container.decode(Float.self, forKey: .clusteringThreshold)
    let minDurationOn = try container.decode(Float.self, forKey: .minDurationOn)
    let minDurationOff = try container.decode(Float.self, forKey: .minDurationOff)
    let numClusters = try container.decode(Int.self, forKey: .numClusters)
    let minActivityThreshold = try container.decode(
        Float.self, forKey: .minActivityThreshold)
    let debugMode = try container.decode(Bool.self, forKey: .debugMode)
    let modelCacheDirectory = try container.decodeIfPresent(
        URL.self, forKey: .modelCacheDirectory)

    self.init(
        clusteringThreshold: clusteringThreshold,
        minDurationOn: minDurationOn,
        minDurationOff: minDurationOff,
        numClusters: numClusters,
        minActivityThreshold: minActivityThreshold,
        debugMode: debugMode,
        modelCacheDirectory: modelCacheDirectory
    )
}
}

// MARK: - TimedSpeakerSegment Extensions

extension TimedSpeakerSegment: Codable {
enum CodingKeys: String, CodingKey {
    case speakerId
    case embedding
    case startTimeSeconds
    case endTimeSeconds
    case qualityScore
}

public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(speakerId, forKey: .speakerId)
    try container.encode(embedding, forKey: .embedding)
    try container.encode(startTimeSeconds, forKey: .startTimeSeconds)
    try container.encode(endTimeSeconds, forKey: .endTimeSeconds)
    try container.encode(qualityScore, forKey: .qualityScore)
}

public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let speakerId = try container.decode(String.self, forKey: .speakerId)
    let embedding = try container.decode([Float].self, forKey: .embedding)
    let startTimeSeconds = try container.decode(Float.self, forKey: .startTimeSeconds)
    let endTimeSeconds = try container.decode(Float.self, forKey: .endTimeSeconds)
    let qualityScore = try container.decode(Float.self, forKey: .qualityScore)

    self.init(
        speakerId: speakerId,
        embedding: embedding,
        startTimeSeconds: startTimeSeconds,
        endTimeSeconds: endTimeSeconds,
        qualityScore: qualityScore
    )
}
}
#endif
