import CoreML
import Foundation

/// Zero-copy MLFeatureProvider for chaining model outputs to inputs.
public class ZeroCopyFeatureProvider: NSObject, MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    public init(features: [String: MLFeatureValue]) {
        self.features = features
        super.init()
    }

    public var featureNames: Set<String> {
        Set(features.keys)
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }

    /// Create a provider that chains output from one model to input of another
    public static func chain(
        from outputProvider: MLFeatureProvider,
        outputName: String,
        to inputName: String
    ) -> ZeroCopyFeatureProvider? {
        guard let outputValue = outputProvider.featureValue(for: outputName) else {
            return nil
        }

        return ZeroCopyFeatureProvider(features: [inputName: outputValue])
    }
}
