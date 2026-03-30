@preconcurrency import CoreML
import Foundation

/// Catalogues all available true streaming ASR model variants with cache-aware encoders.
///
/// These are models with native streaming architectures that maintain encoder cache states
/// across chunks. This does **not** include Parakeet TDT, which uses an offline encoder
/// in a sliding-window pseudo-streaming mode (use `AsrModelVersion` + `SlidingWindowAsrManager`
/// directly for TDT).
///
/// Call `createManager()` to instantiate the appropriate streaming ASR manager.
///
/// Following the `CtcModelVariant` pattern for consistency.
public enum StreamingModelVariant: String, CaseIterable, Sendable {
    // MARK: - Parakeet EOU (cache-aware streaming encoder, 120M params)

    /// Parakeet EOU 120M with 160ms chunks (lowest latency)
    case parakeetEou160ms = "parakeet-eou-160ms"
    /// Parakeet EOU 120M with 320ms chunks (balanced)
    case parakeetEou320ms = "parakeet-eou-320ms"
    /// Parakeet EOU 120M with 1280ms chunks (highest throughput)
    case parakeetEou1280ms = "parakeet-eou-1280ms"

    // MARK: - Nemotron Speech Streaming (cache-aware streaming, 0.6B params)

    /// Nemotron 0.6B with 560ms chunks (balanced)
    case nemotron560ms = "nemotron-560ms"
    /// Nemotron 0.6B with 1120ms chunks (best accuracy)
    case nemotron1120ms = "nemotron-1120ms"

    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .parakeetEou160ms: return "Parakeet EOU 120M (160ms)"
        case .parakeetEou320ms: return "Parakeet EOU 120M (320ms)"
        case .parakeetEou1280ms: return "Parakeet EOU 120M (1280ms)"
        case .nemotron560ms: return "Nemotron 0.6B (560ms)"
        case .nemotron1120ms: return "Nemotron 0.6B (1120ms)"
        }
    }

    /// The HuggingFace repo for this variant's CoreML models
    public var repo: Repo {
        switch self {
        case .parakeetEou160ms: return .parakeetEou160
        case .parakeetEou320ms: return .parakeetEou320
        case .parakeetEou1280ms: return .parakeetEou1280
        case .nemotron560ms: return .nemotronStreaming560
        case .nemotron1120ms: return .nemotronStreaming1120
        }
    }

    /// Engine family grouping for factory dispatch
    public var engineFamily: EngineFamily {
        switch self {
        case .parakeetEou160ms, .parakeetEou320ms, .parakeetEou1280ms:
            return .parakeetEou
        case .nemotron560ms, .nemotron1120ms:
            return .nemotron
        }
    }

    /// The streaming chunk size for EOU variants (nil for non-EOU)
    public var eouChunkSize: StreamingChunkSize? {
        switch self {
        case .parakeetEou160ms: return .ms160
        case .parakeetEou320ms: return .ms320
        case .parakeetEou1280ms: return .ms1280
        default: return nil
        }
    }

    /// The streaming chunk size for Nemotron variants (nil for non-Nemotron)
    public var nemotronChunkSize: NemotronChunkSize? {
        switch self {
        case .nemotron560ms: return .ms560
        case .nemotron1120ms: return .ms1120
        default: return nil
        }
    }

    /// Create a streaming ASR manager for this variant.
    ///
    /// The returned manager is not yet loaded — call `loadModels()` before use.
    ///
    /// - Parameter configuration: Optional `MLModelConfiguration` override.
    /// - Returns: A streaming ASR manager conforming to `StreamingAsrManager`.
    public func createManager(
        configuration: MLModelConfiguration? = nil
    ) -> any StreamingAsrManager {
        let mlConfig = configuration ?? MLModelConfiguration()
        switch engineFamily {
        case .parakeetEou:
            let chunkSize = eouChunkSize ?? .ms160
            return StreamingEouAsrManager(configuration: mlConfig, chunkSize: chunkSize)
        case .nemotron:
            let chunkSize = nemotronChunkSize ?? .ms1120
            return StreamingNemotronAsrManager(configuration: mlConfig, requestedChunkSize: chunkSize)
        }
    }

    /// Engine family types for true streaming models
    public enum EngineFamily: String, Sendable {
        /// Parakeet EOU: cache-aware streaming with end-of-utterance detection
        case parakeetEou = "parakeet-eou"
        /// Nemotron: cache-aware streaming with encoder cache states
        case nemotron = "nemotron"
    }
}
