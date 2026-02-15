import Foundation

/// Per-word timestamp alignment from the forced aligner.
public struct WordAlignment: Sendable {
    public let word: String
    public let startMs: Double
    public let endMs: Double

    public init(word: String, startMs: Double, endMs: Double) {
        self.word = word
        self.startMs = startMs
        self.endMs = endMs
    }
}

/// Result of a forced alignment operation.
public struct ForcedAlignmentResult: Sendable {
    public let alignments: [WordAlignment]
    public let latencyMs: Double

    public init(alignments: [WordAlignment], latencyMs: Double) {
        self.alignments = alignments
        self.latencyMs = latencyMs
    }
}

/// Errors from the forced aligner pipeline.
public enum ForcedAlignerError: Error, LocalizedError {
    case modelNotFound(String)
    case encoderFailed(String)
    case decoderFailed(String)
    case tokenizerFailed(String)
    case alignmentFailed(String)
    case modelsNotLoaded

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "ForcedAligner model not found: \(name)"
        case .encoderFailed(let detail):
            return "Audio encoder failed: \(detail)"
        case .decoderFailed(let detail):
            return "Decoder failed: \(detail)"
        case .tokenizerFailed(let detail):
            return "Tokenizer failed: \(detail)"
        case .alignmentFailed(let detail):
            return "Alignment failed: \(detail)"
        case .modelsNotLoaded:
            return "ForcedAligner models not loaded. Call loadModels() first."
        }
    }
}
