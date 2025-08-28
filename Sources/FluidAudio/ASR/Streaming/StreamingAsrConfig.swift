import CoreMedia

/// Streaming performance mode presets
@available(macOS 13.0, iOS 16.0, *)
public enum StreamingMode: Sendable {
    /// Fast updates with lower accuracy - 5s chunks, 0.5s updates
    case lowLatency
    /// Balanced quality and responsiveness - 10s chunks, 1s updates
    case balanced
    /// Best quality with higher latency - 15s chunks, 2s updates
    case highAccuracy

    public var chunkSeconds: TimeInterval {
        switch self {
        case .lowLatency: return 5.0
        case .balanced: return 10.0
        case .highAccuracy: return 15.0
        }
    }

    internal var leftContextSeconds: TimeInterval {
        switch self {
        case .lowLatency: return 1.0
        case .balanced: return 2.0
        case .highAccuracy: return 3.0
        }
    }

    internal var rightContextSeconds: TimeInterval {
        switch self {
        case .lowLatency: return 1.0
        case .balanced: return 2.0
        case .highAccuracy: return 2.5
        }
    }

    internal var updateFrequency: TimeInterval {
        switch self {
        case .lowLatency: return 0.5
        case .balanced: return 1.0
        case .highAccuracy: return 2.0
        }
    }
}

@available(macOS 13.0, iOS 16.0, *)
public struct StreamingAsrConfig: Sendable {
    /// Performance mode preset
    public let mode: StreamingMode
    /// Enable debug logging
    public let enableDebug: Bool
    /// Override chunk duration (optional - uses mode default if nil)
    public let chunkDuration: TimeInterval?
    /// Override UI update frequency (optional - uses mode default if nil)
    public let updateFrequency: TimeInterval?

    /// Default balanced configuration
    public static let `default`: StreamingAsrConfig = StreamingAsrConfig(mode: .balanced)

    public init(
        mode: StreamingMode = .balanced,
        enableDebug: Bool = false,
        chunkDuration: TimeInterval? = nil,
        updateFrequency: TimeInterval? = nil
    ) {
        self.mode = mode
        self.enableDebug = enableDebug
        self.chunkDuration = chunkDuration
        self.updateFrequency = updateFrequency
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            enableDebug: enableDebug,
            tdtConfig: TdtConfig()
        )
    }

    // Computed properties with fallbacks to mode defaults
    var chunkSeconds: TimeInterval { chunkDuration ?? mode.chunkSeconds }
    var leftContextSeconds: TimeInterval { mode.leftContextSeconds }
    var rightContextSeconds: TimeInterval { mode.rightContextSeconds }
    var interimUpdateFrequency: TimeInterval { updateFrequency ?? mode.updateFrequency }

    // Sample counts at 16 kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }
    var interimUpdateSamples: Int { Int(interimUpdateFrequency * 16000) }

    // Simplified interim processing parameters
    var interimRightContextSamples: Int { Int(rightContextSeconds * 0.5 * 16000) }  // Half of right context for interim

    // Backward-compat convenience
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
}
