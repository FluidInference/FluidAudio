@preconcurrency import CoreML
import Foundation

/// Encoder feature provider: `mel`, `mel_length`, and `prompt_id` for the multilingual (prompted) encoders.
final class UnifiedEncoderFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String>

    let mel: MLFeatureValue
    let melLength: MLFeatureValue
    let promptId: MLFeatureValue?

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "mel": return mel
        case "mel_length": return melLength
        case "prompt_id": return promptId
        default: return nil
        }
    }

    init(mel: MLMultiArray, melLength: MLMultiArray, promptId: MLMultiArray? = nil) {
        self.mel = MLFeatureValue(multiArray: mel)
        self.melLength = MLFeatureValue(multiArray: melLength)
        self.promptId = promptId.map { MLFeatureValue(multiArray: $0) }
        self.featureNames = promptId == nil ? ["mel", "mel_length"] : ["mel", "mel_length", "prompt_id"]
    }
}

/// Hyperoptimized Decoder Model Feature Provider
final class UnifiedDecoderFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["targets", "target_length", "h_in", "c_in"]

    let hIn: MLFeatureValue
    let cIn: MLFeatureValue
    let targets: MLFeatureValue
    let targetLength: MLFeatureValue

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName.count {
        case 4: return featureName.first == "h" ? hIn : cIn
        case 7: return targets
        default: return targetLength
        }
    }

    init(
        targets: MLMultiArray,
        targetLength: MLMultiArray,
        hIn: MLMultiArray,
        cIn: MLMultiArray
    ) {
        self.targets = MLFeatureValue(multiArray: targets)
        self.targetLength = MLFeatureValue(multiArray: targetLength)
        self.hIn = MLFeatureValue(multiArray: hIn)
        self.cIn = MLFeatureValue(multiArray: cIn)
    }
}

/// Hyperoptimized Joint Decision Model Feature Provider
final class UnifiedJointDecisionFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["encoder_step", "decoder_step"]

    let encoderStep: MLFeatureValue
    let decoderStep: MLFeatureValue

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName.first == "e" { return encoderStep }
        return decoderStep
    }

    init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
        self.encoderStep = MLFeatureValue(multiArray: encoderStep)
        self.decoderStep = MLFeatureValue(multiArray: decoderStep)
    }
}
