import Foundation

/// Inflect v2 model size. Both share the same pipeline, symbol table, and
/// English frontend — only the CoreML weights and repo subdirectory differ.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum InflectVariant: String, Sendable, CaseIterable {
    /// 9.36M params (~19 MB fp16). Higher quality.
    case micro
    /// 3.97M params (~8 MB fp16). Smallest footprint, fastest.
    case nano

    /// Subdirectory under `FluidInference/inflect-v2-coreml/`.
    public var subdirectory: String { rawValue }
}
