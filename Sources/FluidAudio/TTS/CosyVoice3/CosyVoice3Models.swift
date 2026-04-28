@preconcurrency import CoreML
import Foundation

/// Four CoreML models for the CosyVoice3 inference pipeline.
public struct CosyVoice3Models {
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
