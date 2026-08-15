import Foundation

/// Chunk size variant for Nemotron streaming
public enum NemotronChunkSize: Int, Sendable, CaseIterable {
    case ms2240 = 2240  // 2.24s - default; highest throughput (+50% RTFx w/ B1 vs 1120ms), WER-neutral
    case ms1120 = 1120  // 1.12s - the trained chunk; lower latency
    case ms560 = 560  // 0.56s - lowest latency tier

    public var repo: Repo {
        switch self {
        case .ms2240: return .nemotronStreaming2240
        case .ms1120: return .nemotronStreaming1120
        case .ms560: return .nemotronStreaming560
        }
    }

    /// HuggingFace remote subdirectory path (matches Repo.subdirectory)
    public var subdirectory: String {
        "nemotron_coreml_\(rawValue)ms"
    }
}

/// Encoder file name for Nemotron streaming (int8 quantized only)
public enum NemotronEncoder {
    static let fileName = "encoder_int8.mlmodelc"
}
