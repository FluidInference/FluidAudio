#if os(macOS)
import FluidAudio
import Foundation

// MARK: - Data Structures

struct ProcessingResult: Codable {
    let audioFile: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
    let config: DiarizerConfig?
    let metrics: DiarizationMetrics?
    let timings: PipelineTimings?
    let timestamp: Date

    init(
        audioFile: String, durationSeconds: Float, processingTimeSeconds: TimeInterval,
        realTimeFactor: Float, segments: [TimedSpeakerSegment], speakerCount: Int,
        config: DiarizerConfig?, metrics: DiarizationMetrics? = nil,
        timings: PipelineTimings? = nil
    ) {
        self.audioFile = audioFile
        self.durationSeconds = durationSeconds
        self.processingTimeSeconds = processingTimeSeconds
        self.realTimeFactor = realTimeFactor
        self.segments = segments
        self.speakerCount = speakerCount
        self.config = config
        self.metrics = metrics
        self.timings = timings
        self.timestamp = Date()
    }
}

struct BenchmarkResult: Codable {
    let meetingId: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let der: Float
    let jer: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
    let groundTruthSpeakerCount: Int
    let timings: PipelineTimings

    /// Total time including audio loading
    var totalExecutionTime: TimeInterval {
        return timings.totalProcessingSeconds + timings.audioLoadingSeconds
    }
}

struct BenchmarkSummary: Codable {
    let dataset: String
    let averageDER: Float
    let averageJER: Float
    let processedFiles: Int
    let totalFiles: Int
    let results: [BenchmarkResult]
    let timestamp: Date

    init(
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

// MARK: - Extensions for Codable Support

// Make DiarizerConfig Codable for output
extension DiarizerConfig: Codable {
    enum CodingKeys: String, CodingKey {
        case clusteringThreshold
        case minDurationOn
        case minDurationOff
        case numClusters
        case minActivityThreshold
        case debugMode
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(clusteringThreshold, forKey: .clusteringThreshold)
        try container.encode(minSpeechDuration, forKey: .minDurationOn)
        try container.encode(minSilenceGap, forKey: .minDurationOff)
        try container.encode(numClusters, forKey: .numClusters)
        try container.encode(minActiveFramesCount, forKey: .minActivityThreshold)
        try container.encode(debugMode, forKey: .debugMode)
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

        self.init(
            clusteringThreshold: clusteringThreshold,
            minSpeechDuration: minDurationOn,
            minSilenceGap: minDurationOff,
            numClusters: numClusters,
            minActiveFramesCount: minActivityThreshold,
            debugMode: debugMode
        )
    }
}

// Make TimedSpeakerSegment Codable for CLI output
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

enum CLIError: Error {
    case invalidArgument(String)
}

// MARK: - Performance Assessment

enum PerformanceAssessment {
    case excellent  // DER < 20.0% - Competitive with state-of-the-art
    case good  // DER < 30.0% - Above research baseline
    case needsWork  // DER < 50.0% - Needs parameter tuning
    case critical  // DER >= 50.0% - Much worse than expected

    var exitCode: Int32 {
        switch self {
        case .excellent, .good:
            return 0  // Success
        case .needsWork:
            return 1  // Warning/needs work
        case .critical:
            return 2  // Critical failure
        }
    }

    var description: String {
        switch self {
        case .excellent:
            return "🎉 Pass"
        case .good:
            return "Pass"
        case .needsWork:
            return "⚠️ Needs Work"
        case .critical:
            return "🚨 Critical"
        }
    }

    static func assess(
        der: Float, jer: Float, rtf: Float,
        customThresholds: (der: Float?, jer: Float?, rtf: Float?) = (nil, nil, nil)
    ) -> PerformanceAssessment {
        // Check custom thresholds first
        if let derThreshold = customThresholds.der, der > derThreshold {
            return .needsWork
        }
        if let jerThreshold = customThresholds.jer, jer > jerThreshold {
            return .needsWork
        }
        if let rtfThreshold = customThresholds.rtf, rtf > rtfThreshold {
            return .needsWork
        }

        // If custom thresholds are set and all pass, return excellent
        if customThresholds.der != nil || customThresholds.jer != nil
            || customThresholds.rtf != nil
        {
            return .excellent
        }

        // Use default thresholds
        if der < 20.0 {
            return .excellent
        } else if der < 30.0 {
            return .good
        } else if der < 50.0 {
            return .needsWork
        } else {
            return .critical
        }
    }
}

// MARK: - VAD Benchmark Data Structures

struct VadTestFile {
    let name: String
    let expectedLabel: Int  // 0 = no speech, 1 = speech
    let url: URL
}

struct VadBenchmarkResult {
    let testName: String
    let accuracy: Float
    let precision: Float
    let recall: Float
    let f1Score: Float
    let processingTime: TimeInterval
    let totalFiles: Int
    let correctPredictions: Int
}

#endif
