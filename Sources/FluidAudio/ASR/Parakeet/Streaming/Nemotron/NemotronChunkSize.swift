import Foundation

/// Chunk size variant for Nemotron streaming
public enum NemotronChunkSize: Int, Sendable, CaseIterable {
    case ms1120 = 1120  // 1.12s - original, best accuracy
    case ms560 = 560  // 0.56s - lower latency, same accuracy
    case ms160 = 160  // 0.16s - very low latency
    case ms80 = 80  // 0.08s - ultra low latency

    public var repo: Repo {
        switch self {
        case .ms1120: return .nemotronStreaming1120
        case .ms560: return .nemotronStreaming560
        case .ms160: return .nemotronStreaming160
        case .ms80: return .nemotronStreaming80
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
