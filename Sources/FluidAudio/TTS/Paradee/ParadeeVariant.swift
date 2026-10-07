import Foundation

/// Paradee weight precision. Both variants share the same graphs, vocab and
/// English frontend; only the weights and repo subdirectory differ.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum ParadeeVariant: String, Sendable, CaseIterable {
    /// int8 per-channel weights (12 MB). Default, as upstream recommends its int8 build.
    case int8
    /// fp32 weights (34 MB); matches PyTorch to rounding.
    case fp32

    /// Subdirectory under `FluidInference/paradee-8m-coreml/`.
    public var subdirectory: String { rawValue }

    /// HuggingFace repo case for this variant's subdirectory.
    public var repo: Repo {
        switch self {
        case .int8: return .paradeeInt8
        case .fp32: return .paradeeFp32
        }
    }
}
