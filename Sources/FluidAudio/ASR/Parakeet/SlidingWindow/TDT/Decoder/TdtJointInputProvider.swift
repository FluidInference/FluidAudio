import CoreML
import Foundation

/// Reusable input provider for TDT joint model inference.
///
/// This class holds pre-allocated MLMultiArray tensors for encoder and decoder features,
/// allowing zero-copy joint network execution. By reusing the same arrays across
/// inference calls, we avoid repeated allocations and improve ANE performance.
///
/// Usage:
/// ```swift
/// let provider = ReusableJointInputProvider(
///     encoderStep: encoderStepArray,  // Shape: [1, 1024]
///     decoderStep: decoderStepArray   // Shape: [1, 640]
/// )
/// let output = try jointModel.prediction(from: provider)
/// ```
internal final class ReusableJointInputProvider: NSObject, MLFeatureProvider {
    /// Encoder feature tensor (shape: [1, hidden_dim])
    let encoderStep: MLMultiArray

    /// Decoder feature tensor (shape: [1, decoder_dim])
    let decoderStep: MLMultiArray

    /// Initialize with pre-allocated encoder and decoder step tensors.
    ///
    /// - Parameters:
    ///   - encoderStep: MLMultiArray for encoder features (typically [1, 1024])
    ///   - decoderStep: MLMultiArray for decoder features (typically [1, 640])
    init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
        self.encoderStep = encoderStep
        self.decoderStep = decoderStep
        super.init()
    }

    var featureNames: Set<String> {
        ["encoder_step", "decoder_step"]
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "encoder_step":
            return MLFeatureValue(multiArray: encoderStep)
        case "decoder_step":
            return MLFeatureValue(multiArray: decoderStep)
        default:
            return nil
        }
    }
}
