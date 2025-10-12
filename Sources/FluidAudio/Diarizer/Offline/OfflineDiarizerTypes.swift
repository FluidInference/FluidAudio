import Foundation

/// Errors surfaced by the offline diarization pipeline.
@available(macOS 13.0, iOS 16.0, *)
public enum OfflineDiarizationError: Error, LocalizedError {
    case modelNotLoaded(String)
    case invalidConfiguration(String)
    case invalidBatchSize(String)
    case processingFailed(String)
    case noSpeechDetected
    case exportFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .invalidBatchSize(let message):
            return "Invalid batch size: \(message)"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .noSpeechDetected:
            return "No speech detected in audio"
        case .exportFailed(let message):
            return "Failed to export data: \(message)"
        }
    }
}

/// Configuration values tuned to pyannote's community-1 pipeline.
/// Groups knobs by pipeline stage while keeping legacy property accessors
/// to minimize downstream churn.
@available(macOS 13.0, iOS 16.0, *)
public struct OfflineDiarizerConfig: Sendable {

    /// Segmentation parameters. Threshold fields are ignored by powerset models like community-1 but
    /// remain for compatibility with non-powerset pipelines.
    public struct Segmentation: Sendable {
        public var windowDurationSeconds: Double
        public var sampleRate: Int
        public var minDurationOn: Double
        public var minDurationOff: Double
        public var stepRatio: Double
        public var speechOnsetThreshold: Float
        public var speechOffsetThreshold: Float
        public var speakerOnsetThreshold: Float
        public var speakerOffsetThreshold: Float
        public var maxEmptyClassProbability: Float

        public static let community = Segmentation(
            windowDurationSeconds: 10.0,
            sampleRate: 16_000,
            minDurationOn: 0.0,
            minDurationOff: 0.0,
            stepRatio: 0.1,
            speechOnsetThreshold: 0.5,
            speechOffsetThreshold: 0.5,
            speakerOnsetThreshold: 0.0,
            speakerOffsetThreshold: 0.0,
            maxEmptyClassProbability: 1.0
        )

        public init(
            windowDurationSeconds: Double,
            sampleRate: Int,
            minDurationOn: Double,
            minDurationOff: Double,
            stepRatio: Double,
            speechOnsetThreshold: Float,
            speechOffsetThreshold: Float,
            speakerOnsetThreshold: Float,
            speakerOffsetThreshold: Float,
            maxEmptyClassProbability: Float
        ) {
            self.windowDurationSeconds = windowDurationSeconds
            self.sampleRate = sampleRate
            self.minDurationOn = minDurationOn
            self.minDurationOff = minDurationOff
            self.stepRatio = stepRatio
            self.speechOnsetThreshold = speechOnsetThreshold
            self.speechOffsetThreshold = speechOffsetThreshold
            self.speakerOnsetThreshold = speakerOnsetThreshold
            self.speakerOffsetThreshold = speakerOffsetThreshold
            self.maxEmptyClassProbability = maxEmptyClassProbability
        }
    }

    public struct Embedding: Sendable {
        public var batchSize: Int
        public var excludeOverlap: Bool
        public var minSegmentDurationSeconds: Double

        public static let community = Embedding(
            batchSize: 32,
            excludeOverlap: true,
            minSegmentDurationSeconds: 0.5
        )

        public init(
            batchSize: Int,
            excludeOverlap: Bool,
            minSegmentDurationSeconds: Double
        ) {
            self.batchSize = batchSize
            self.excludeOverlap = excludeOverlap
            self.minSegmentDurationSeconds = minSegmentDurationSeconds
        }
    }

    public struct Clustering: Sendable {
        /// Euclidean distance threshold for unit-normalized embeddings.
        public var threshold: Double
        public var minClusterSize: Int
        public var warmStartFa: Double
        public var warmStartFb: Double

        public static let community = Clustering(
            threshold: 0.7045654963945799,
            minClusterSize: 12,
            warmStartFa: 0.07,
            warmStartFb: 0.8
        )

        public init(
            threshold: Double,
            minClusterSize: Int,
            warmStartFa: Double,
            warmStartFb: Double
        ) {
            self.threshold = threshold
            self.minClusterSize = minClusterSize
            self.warmStartFa = warmStartFa
            self.warmStartFb = warmStartFb
        }
    }

    public struct VBx: Sendable {
        public var loopProbability: Double
        public var maxIterations: Int
        public var convergenceTolerance: Double

        public static let community = VBx(
            loopProbability: 0.98,
            maxIterations: 20,
            convergenceTolerance: 1e-4
        )

        public init(
            loopProbability: Double,
            maxIterations: Int,
            convergenceTolerance: Double
        ) {
            self.loopProbability = loopProbability
            self.maxIterations = maxIterations
            self.convergenceTolerance = convergenceTolerance
        }
    }

    public struct PostProcessing: Sendable {
        public var minGapDurationSeconds: Double

        public static let community = PostProcessing(minGapDurationSeconds: 0.1)

        public init(minGapDurationSeconds: Double) {
            self.minGapDurationSeconds = minGapDurationSeconds
        }
    }

    public struct Export: Sendable {
        public var embeddingsPath: String?

        public init(embeddingsPath: String? = nil) {
            self.embeddingsPath = embeddingsPath
        }

        public static let none = Export()
    }

    public var segmentation: Segmentation
    public var embedding: Embedding
    public var clustering: Clustering
    public var vbx: VBx
    public var postProcessing: PostProcessing
    public var export: Export

    public init(
        segmentation: Segmentation = .community,
        embedding: Embedding = .community,
        clustering: Clustering = .community,
        vbx: VBx = .community,
        postProcessing: PostProcessing = .community,
        export: Export = .none
    ) {
        self.segmentation = segmentation
        self.embedding = embedding
        self.clustering = clustering
        self.vbx = vbx
        self.postProcessing = postProcessing
        self.export = export
    }

    public init(
        clusteringThreshold: Double = Clustering.community.threshold,
        minClusterSize: Int = Clustering.community.minClusterSize,
        Fa: Double = Clustering.community.warmStartFa,
        Fb: Double = Clustering.community.warmStartFb,
        loopProbability: Double = VBx.community.loopProbability,
        windowDuration: Double = Segmentation.community.windowDurationSeconds,
        sampleRate: Int = Segmentation.community.sampleRate,
        segmentationStepRatio: Double = Segmentation.community.stepRatio,
        embeddingBatchSize: Int = Embedding.community.batchSize,
        embeddingExcludeOverlap: Bool = Embedding.community.excludeOverlap,
        minSegmentDuration: Double = Embedding.community.minSegmentDurationSeconds,
        minGapDuration: Double = PostProcessing.community.minGapDurationSeconds,
        speechOnsetThreshold: Float = Segmentation.community.speechOnsetThreshold,
        speechOffsetThreshold: Float = Segmentation.community.speechOffsetThreshold,
        segmentationOnsetThreshold: Float = Segmentation.community.speakerOnsetThreshold,
        segmentationOffsetThreshold: Float = Segmentation.community.speakerOffsetThreshold,
        maxEmptyClassProbability: Float = Segmentation.community.maxEmptyClassProbability,
        segmentationMinDurationOn: Double = Segmentation.community.minDurationOn,
        segmentationMinDurationOff: Double = Segmentation.community.minDurationOff,
        maxVBxIterations: Int = VBx.community.maxIterations,
        convergenceTolerance: Double = VBx.community.convergenceTolerance,
        embeddingExportPath: String? = nil
    ) {
        self.init(
            segmentation: Segmentation(
                windowDurationSeconds: windowDuration,
                sampleRate: sampleRate,
                minDurationOn: segmentationMinDurationOn,
                minDurationOff: segmentationMinDurationOff,
                stepRatio: segmentationStepRatio,
                speechOnsetThreshold: speechOnsetThreshold,
                speechOffsetThreshold: speechOffsetThreshold,
                speakerOnsetThreshold: segmentationOnsetThreshold,
                speakerOffsetThreshold: segmentationOffsetThreshold,
                maxEmptyClassProbability: maxEmptyClassProbability
            ),
            embedding: Embedding(
                batchSize: embeddingBatchSize,
                excludeOverlap: embeddingExcludeOverlap,
                minSegmentDurationSeconds: minSegmentDuration
            ),
            clustering: Clustering(
                threshold: clusteringThreshold,
                minClusterSize: minClusterSize,
                warmStartFa: Fa,
                warmStartFb: Fb
            ),
            vbx: VBx(
                loopProbability: loopProbability,
                maxIterations: maxVBxIterations,
                convergenceTolerance: convergenceTolerance
            ),
            postProcessing: PostProcessing(minGapDurationSeconds: minGapDuration),
            export: Export(embeddingsPath: embeddingExportPath)
        )
    }

    /// Number of samples processed per segmentation window.
    public var samplesPerWindow: Int {
        Int(Double(segmentation.sampleRate) * segmentation.windowDurationSeconds)
    }

    public var samplesPerStep: Int {
        max(1, Int(Double(samplesPerWindow) * segmentation.stepRatio))
    }

    /// Validate configuration values and throw if they fall outside expected ranges.
    public func validate() throws {
        let maxClusteringThreshold = sqrt(2.0)
        guard clustering.threshold > 0, clustering.threshold <= maxClusteringThreshold else {
            throw OfflineDiarizationError.invalidConfiguration(
                "clustering.threshold must be within (0, sqrt(2)], got \(clustering.threshold)"
            )
        }

        guard clustering.minClusterSize >= 1 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "clustering.minClusterSize must be >= 1, got \(clustering.minClusterSize)"
            )
        }

        guard clustering.warmStartFa > 0, clustering.warmStartFb > 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "clustering warm-start Fa/Fb must be positive (Fa=\(clustering.warmStartFa), Fb=\(clustering.warmStartFb))"
            )
        }

        guard vbx.loopProbability > 0, vbx.loopProbability <= 1 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "vbx.loopProbability must be within (0, 1], got \(vbx.loopProbability)"
            )
        }

        guard segmentation.windowDurationSeconds > 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentation.windowDurationSeconds must be positive, got \(segmentation.windowDurationSeconds)"
            )
        }

        guard segmentation.sampleRate > 0 else {
            throw OfflineDiarizationError.invalidConfiguration("sampleRate must be positive")
        }

        guard segmentation.stepRatio > 0, segmentation.stepRatio <= 1 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentation.stepRatio must be within (0, 1], got \(segmentation.stepRatio)"
            )
        }

        guard embedding.batchSize > 0 else {
            throw OfflineDiarizationError.invalidBatchSize("embeddingBatchSize must be > 0")
        }

        guard embedding.batchSize <= 32 else {
            throw OfflineDiarizationError.invalidBatchSize(
                "embeddingBatchSize must be <= 32 to fit PLDA batch limits"
            )
        }

        guard vbx.maxIterations > 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "maxVBxIterations must be > 0, got \(vbx.maxIterations)"
            )
        }

        guard vbx.convergenceTolerance > 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "convergenceTolerance must be positive"
            )
        }

        guard embedding.minSegmentDurationSeconds >= 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "embedding.minSegmentDuration must be >= 0"
            )
        }

        guard postProcessing.minGapDurationSeconds >= 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "minGapDuration must be >= 0"
            )
        }

        guard segmentation.minDurationOn >= 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentation.minDurationOn must be >= 0"
            )
        }

        guard segmentation.minDurationOff >= 0 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentation.minDurationOff must be >= 0"
            )
        }

        guard segmentation.speechOnsetThreshold >= 0, segmentation.speechOnsetThreshold <= 1 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "speechOnsetThreshold must be within [0, 1], got \(segmentation.speechOnsetThreshold)"
            )
        }

        guard segmentation.speechOffsetThreshold >= 0,
            segmentation.speechOffsetThreshold <= segmentation.speechOnsetThreshold
        else {
            throw OfflineDiarizationError.invalidConfiguration(
                "speechOffsetThreshold must be within [0, speechOnsetThreshold], got \(segmentation.speechOffsetThreshold)"
            )
        }

        guard segmentation.speakerOnsetThreshold >= 0, segmentation.speakerOnsetThreshold <= 1 else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentationOnsetThreshold must be within [0, 1], got \(segmentation.speakerOnsetThreshold)"
            )
        }

        guard segmentation.speakerOffsetThreshold >= 0,
            segmentation.speakerOffsetThreshold <= segmentation.speakerOnsetThreshold
        else {
            throw OfflineDiarizationError.invalidConfiguration(
                "segmentationOffsetThreshold must be within [0, segmentationOnsetThreshold], got \(segmentation.speakerOffsetThreshold)"
            )
        }

        guard segmentation.maxEmptyClassProbability >= 0,
            segmentation.maxEmptyClassProbability <= 1
        else {
            throw OfflineDiarizationError.invalidConfiguration(
                "maxEmptyClassProbability must be within [0, 1], got \(segmentation.maxEmptyClassProbability)"
            )
        }
    }

    public var clusteringThreshold: Double {
        get { clustering.threshold }
        set { clustering.threshold = newValue }
    }

    public var minClusterSize: Int {
        get { clustering.minClusterSize }
        set { clustering.minClusterSize = newValue }
    }

    public var Fa: Double {
        get { clustering.warmStartFa }
        set { clustering.warmStartFa = newValue }
    }

    public var Fb: Double {
        get { clustering.warmStartFb }
        set { clustering.warmStartFb = newValue }
    }

    public var loopProbability: Double {
        get { vbx.loopProbability }
        set { vbx.loopProbability = newValue }
    }

    public var windowDuration: Double {
        get { segmentation.windowDurationSeconds }
        set { segmentation.windowDurationSeconds = newValue }
    }

    public var sampleRate: Int {
        get { segmentation.sampleRate }
        set { segmentation.sampleRate = newValue }
    }

    public var embeddingBatchSize: Int {
        get { embedding.batchSize }
        set { embedding.batchSize = newValue }
    }

    public var maxVBxIterations: Int {
        get { vbx.maxIterations }
        set { vbx.maxIterations = newValue }
    }

    public var convergenceTolerance: Double {
        get { vbx.convergenceTolerance }
        set { vbx.convergenceTolerance = newValue }
    }

    public var embeddingExcludeOverlap: Bool {
        get { embedding.excludeOverlap }
        set { embedding.excludeOverlap = newValue }
    }

    @available(*, deprecated, renamed: "embeddingExcludeOverlap")
    public var shouldExcludeOverlaps: Bool {
        get { embeddingExcludeOverlap }
        set { embeddingExcludeOverlap = newValue }
    }

    public var minSegmentDuration: Double {
        get { embedding.minSegmentDurationSeconds }
        set { embedding.minSegmentDurationSeconds = newValue }
    }

    public var minGapDuration: Double {
        get { postProcessing.minGapDurationSeconds }
        set { postProcessing.minGapDurationSeconds = newValue }
    }

    public var embeddingExportPath: String? {
        get { export.embeddingsPath }
        set { export.embeddingsPath = newValue }
    }

    public var speechOnsetThreshold: Float {
        get { segmentation.speechOnsetThreshold }
        set { segmentation.speechOnsetThreshold = newValue }
    }

    public var speechOffsetThreshold: Float {
        get { segmentation.speechOffsetThreshold }
        set { segmentation.speechOffsetThreshold = newValue }
    }

    public var segmentationOnsetThreshold: Float {
        get { segmentation.speakerOnsetThreshold }
        set { segmentation.speakerOnsetThreshold = newValue }
    }

    public var segmentationOffsetThreshold: Float {
        get { segmentation.speakerOffsetThreshold }
        set { segmentation.speakerOffsetThreshold = newValue }
    }

    public var maxEmptyClassProbability: Float {
        get { segmentation.maxEmptyClassProbability }
        set { segmentation.maxEmptyClassProbability = newValue }
    }

    public var segmentationMinDurationOn: Double {
        get { segmentation.minDurationOn }
        set { segmentation.minDurationOn = newValue }
    }

    public var segmentationMinDurationOff: Double {
        get { segmentation.minDurationOff }
        set { segmentation.minDurationOff = newValue }
    }

    public var segmentationStepRatio: Double {
        get { segmentation.stepRatio }
        set { segmentation.stepRatio = newValue }
    }

    public static var `default`: OfflineDiarizerConfig {
        OfflineDiarizerConfig()
    }
}

/// Raw segmentation logits over the local powerset predictions for each chunk.
@available(macOS 13.0, iOS 16.0, *)
struct SegmentationLogits: Sendable {
    let chunkIndex: Int
    let startSample: Int
    let endSample: Int
    let logits: [[Float]]  // frames × classes
}

/// Segmentation output aggregated across all processed windows.
@available(macOS 13.0, iOS 16.0, *)
public struct SegmentationOutput: Sendable {
    public let logProbs: [[[Float]]]
    /// Soft speaker activity weights per chunk/frame/speaker (0.0...1.0 values).
    public let speakerWeights: [[[Float]]]
    public let numChunks: Int
    public let numFrames: Int
    public let numSpeakers: Int
    public let chunkOffsets: [Double]
    public let frameDuration: Double

    public init(
        logProbs: [[[Float]]],
        speakerWeights: [[[Float]]] = [],
        numChunks: Int,
        numFrames: Int,
        numSpeakers: Int,
        chunkOffsets: [Double] = [],
        frameDuration: Double = 0
    ) {
        self.logProbs = logProbs
        self.speakerWeights = speakerWeights
        self.numChunks = numChunks
        self.numFrames = numFrames
        self.numSpeakers = numSpeakers
        self.chunkOffsets = chunkOffsets
        self.frameDuration = frameDuration
    }
}

/// Result returned by the VBx refinement step.
@available(macOS 13.0, iOS 16.0, *)
public struct VBxOutput: Sendable {
    public let gamma: [[Double]]
    public let pi: [Double]
    public let hardClusters: [[Int]]
    public let centroids: [[Double]]
    public let numClusters: Int
    public let elbos: [Double]

    public init(
        gamma: [[Double]],
        pi: [Double],
        hardClusters: [[Int]],
        centroids: [[Double]],
        numClusters: Int,
        elbos: [Double]
    ) {
        self.gamma = gamma
        self.pi = pi
        self.hardClusters = hardClusters
        self.centroids = centroids
        self.numClusters = numClusters
        self.elbos = elbos
    }
}

/// Intermediate representation of an embedding associated with its timeline.
@available(macOS 13.0, iOS 16.0, *)
struct TimedEmbedding: Sendable {
    let chunkIndex: Int
    let speakerIndex: Int
    let startFrame: Int
    let endFrame: Int
    let frameWeights: [Float]
    let startTime: Double
    let endTime: Double
    let embedding256: [Float]
    let rho128: [Double]
}
