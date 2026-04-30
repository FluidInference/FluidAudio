@preconcurrency import CoreML
import Foundation

/// Four CoreML models for the CosyVoice3 inference pipeline.
///
/// `Sendable` conformance leans on `@preconcurrency import CoreML` (same
/// pattern as `TtsModels`). `MLModel` is reference-typed but its predict
/// surface is internally synchronized, and these instances are only handed
/// to actors that own them for their lifetime, so crossing actor isolation
/// is safe in practice.
public struct CosyVoice3Models: Sendable {
    public let prefill: MLModel
    public let decode: MLModel
    public let flow: MLModel
    public let hift: MLModel

    public init(prefill: MLModel, decode: MLModel, flow: MLModel, hift: MLModel) {
        self.prefill = prefill
        self.decode = decode
        self.flow = flow
        self.hift = hift
    }
}
