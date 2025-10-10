import Foundation

/// Configuration for `StreamingAsrManager`.
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingAsrConfig: Sendable {
    /// Main chunk size for stable transcription (seconds). Should be 10-11s for best quality.
    public let chunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s.
    public let leftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency).
    public let rightContextSeconds: TimeInterval
    /// Stabilizer configuration for commit-on-prefix streaming.
    public let stabilizer: StreamingStabilizerConfig
    /// Voice activity detection configuration for streaming audio.
    public let vad: StreamingVadConfig

    /// Default configuration aligned with previous API expectations.
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 15.0,
        leftContextSeconds: 10.0,
        rightContextSeconds: 2.0,
        stabilizer: StreamingStabilizerConfig(),
        vad: .disabled
    )

    /// Optimized streaming configuration: tuned for balanced latency and stability.
    /// Uses ChunkProcessor's proven 11-2-2 approach for stable transcription while still
    /// emitting low-latency volatile updates via the stabilizer.
    public static let streaming = StreamingAsrConfig(
        chunkSeconds: 11.2,  // Match ChunkProcessor center duration
        leftContextSeconds: 1.6,  // Align overlap with ChunkProcessor
        rightContextSeconds: 1.6,  // Align overlap with ChunkProcessor
        stabilizer: StreamingStabilizerConfig.preset(.highStability),
        vad: .default
    )

    public init(
        chunkSeconds: TimeInterval = 10.0,
        leftContextSeconds: TimeInterval = 2.0,
        rightContextSeconds: TimeInterval = 2.0,
        stabilizer: StreamingStabilizerConfig = StreamingStabilizerConfig(),
        vad: StreamingVadConfig = .disabled
    ) {
        self.chunkSeconds = chunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.stabilizer = stabilizer
        self.vad = vad
    }

    /// Backward-compatible convenience initializer used by tests (`chunkDuration` label).
    public init(chunkDuration: TimeInterval) {
        self.init(
            chunkSeconds: chunkDuration,
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            stabilizer: StreamingStabilizerConfig(),
            vad: .disabled
        )
    }

    /// Custom configuration factory expected by tests.
    public static func custom(
        chunkDuration: TimeInterval
    ) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkDuration,
            leftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            stabilizer: StreamingStabilizerConfig(),
            vad: .disabled
        )
    }

    // MARK: - Derived Parameters

    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            tdtConfig: TdtConfig()
        )
    }

    /// Sample counts at 16 kHz.
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }

    /// Backward-compatibility conveniences for existing call-sites/tests.
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }

    public func withStabilizer(_ stabilizer: StreamingStabilizerConfig) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkSeconds,
            leftContextSeconds: leftContextSeconds,
            rightContextSeconds: rightContextSeconds,
            stabilizer: stabilizer,
            vad: vad
        )
    }

    public func withVad(_ vad: StreamingVadConfig) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkSeconds,
            leftContextSeconds: leftContextSeconds,
            rightContextSeconds: rightContextSeconds,
            stabilizer: stabilizer,
            vad: vad
        )
    }
}

/// Configuration options for streaming VAD gating within `StreamingAsrManager`.
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingVadConfig: Sendable {
    /// Whether VAD gating is enabled.
    public let isEnabled: Bool
    /// Configuration passed to the underlying `VadManager` when auto-initializing.
    public let vadConfig: VadConfig
    /// Segmentation behavior used when emitting speech events during streaming.
    public let segmentationConfig: VadSegmentationConfig

    public static let `default` = StreamingVadConfig(isEnabled: true)
    public static let disabled = StreamingVadConfig(isEnabled: false)

    public init(
        isEnabled: Bool = true,
        vadConfig: VadConfig = .default,
        segmentationConfig: VadSegmentationConfig = .default
    ) {
        self.isEnabled = isEnabled
        self.vadConfig = vadConfig
        self.segmentationConfig = segmentationConfig
    }
}

/// Transcription update from streaming ASR.
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptionUpdate: Sendable {
    /// The transcribed text.
    public let text: String
    /// Whether this text is confirmed (high confidence) or volatile (may change).
    public let isConfirmed: Bool
    /// Confidence score (0.0 - 1.0).
    public let confidence: Float
    /// Timestamp of this update.
    public let timestamp: Date
    /// Raw token identifiers emitted for this update.
    public let tokenIds: [Int]
    /// Token-level timing information aligned with the decoded text.
    public let tokenTimings: [TokenTiming]

    /// Human-readable tokens (normalized) for this update.
    public var tokens: [String] {
        tokenTimings.map(\.token)
    }

    public init(
        text: String,
        isConfirmed: Bool,
        confidence: Float,
        timestamp: Date,
        tokenIds: [Int] = [],
        tokenTimings: [TokenTiming] = []
    ) {
        self.text = text
        self.isConfirmed = isConfirmed
        self.confidence = confidence
        self.timestamp = timestamp
        self.tokenIds = tokenIds
        self.tokenTimings = tokenTimings
    }
}
