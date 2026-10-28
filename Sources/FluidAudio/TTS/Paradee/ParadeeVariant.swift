#if TTS
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

    /// v1.0 subdirectory under `FluidInference/paradee-8m-coreml/`.
    public var subdirectory: String { rawValue }

    /// HuggingFace repo case for this variant of v1.0.
    public var repo: Repo { repo(for: .v10) }

    /// HuggingFace repo case for this variant of `version`.
    public func repo(for version: ParadeeVersion) -> Repo {
        switch (version, self) {
        case (.v10, .int8): return .paradeeInt8
        case (.v10, .fp32): return .paradeeFp32
        case (.v12, .int8): return .paradeeV12Int8
        case (.v12, .fp32): return .paradeeV12Fp32
        }
    }
}
#endif
