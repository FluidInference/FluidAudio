@preconcurrency import CoreML
import Foundation

/// On-device CUA-S1-FORMS decision scoring for existing document entities and form actions.
///
/// Provide a description of one UI element and strings such as `fill Email: a@example.com`,
/// `check`, `click`, and `skip`. The model scores those options in one stateless pass.
/// Document extraction, UI observation, action ordering, and execution belong to the caller.
/// This research checkpoint was verified on a small form demo, not arbitrary computer-use tasks.
public actor CuaS1FormsManager {
    /// Maximum UTF-8 bytes encoded from a context.
    public static let contextByteLimit = CuaS1FormsInput.contextByteLimit
    /// Maximum UTF-8 bytes encoded from each option.
    public static let optionByteLimit = CuaS1FormsInput.optionByteLimit
    /// Maximum supplied options in the supported model artifact.
    public static let maximumOptions = CuaS1FormsInput.maximumOptions

    private let model: MLModel

    /// Initialize with an already loaded model after validating its tensor interface.
    public init(model: MLModel) throws {
        try Self.validateModel(model.modelDescription)
        self.model = model
    }

    /// Download and load the model using FluidAudio's shared cache and offline policy.
    /// - Parameter cacheDirectory: The parent Models directory, not the repository subdirectory.
    public static func load(
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ProgressHandler? = nil
    ) async throws -> CuaS1FormsManager {
        let root = cacheDirectory ?? defaultCacheDirectory()
        let name = ModelNames.CuaS1Forms.modelFile
        let models = try await ModelHub.loadModels(
            .cuaS1Forms, modelNames: [name], directory: root,
            computeUnits: computeUnits, progressHandler: progressHandler)
        guard let model = models[name] else {
            throw CuaS1FormsError.invalidModel("Downloaded model is missing \(name)")
        }
        return try CuaS1FormsManager(model: model)
    }

    /// Load a local `.mlpackage` or `.mlmodelc`, compiling the package locally when necessary.
    /// This path does not download assets or recover by accessing the network.
    public static func load(
        from modelURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> CuaS1FormsManager {
        guard modelURL.isFileURL else {
            throw CuaS1FormsError.invalidModel("A local file URL is required")
        }
        let compiledURL: URL
        switch modelURL.pathExtension {
        case "mlpackage":
            compiledURL = try await MLModel.compileModel(at: modelURL)
        case "mlmodelc":
            compiledURL = modelURL
        default:
            throw CuaS1FormsError.invalidModel("Expected a .mlpackage or .mlmodelc URL")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        return try CuaS1FormsManager(model: model)
    }

    /// Score 2–32 nonempty options against one nonempty context.
    ///
    /// Text is truncated by UTF-8 bytes exactly as in the upstream checkpoint. The result
    /// reports any truncation; options beyond the supported capacity are rejected, not dropped.
    /// Calls on a manager are serialized by the actor; only Sendable values leave it.
    public func score(context: String, options: [String]) throws -> CuaS1FormsResult {
        try Task.checkCancellation()
        let encoded = try CuaS1FormsInput(context: context, options: options)
        return try autoreleasepool {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "context_ids": makeArray(encoded.contextIDs, shape: [1, Self.contextByteLimit]),
                "option_ids": makeArray(encoded.optionIDs, shape: [1, Self.maximumOptions, Self.optionByteLimit]),
                "option_mask": makeArray(encoded.optionMask, shape: [1, Self.maximumOptions]),
            ])
            let prediction = try model.prediction(from: features)
            let logits = try readOutput("logits", from: prediction)
            let probabilities = try readOutput("probabilities", from: prediction)
            guard probabilities.allSatisfy({ $0 >= 0 && $0 <= 1 }) else {
                throw CuaS1FormsError.invalidOutput("Probabilities must be between zero and one")
            }
            let live = Array(probabilities.prefix(options.count))
            guard abs(live.reduce(0, +) - 1) <= 0.001 else {
                throw CuaS1FormsError.invalidOutput("Live probabilities do not sum to one")
            }
            guard probabilities.dropFirst(options.count).allSatisfy({ $0 == 0 }) else {
                throw CuaS1FormsError.invalidOutput("Padded options received probability mass")
            }
            var selectedIndex = 0
            for index in live.indices.dropFirst() where live[index] > live[selectedIndex] {
                selectedIndex = index
            }
            return CuaS1FormsResult(
                selectedIndex: selectedIndex, selectedOption: options[selectedIndex],
                probabilities: live, logits: Array(logits.prefix(options.count)),
                contextWasTruncated: encoded.contextWasTruncated,
                truncatedOptionIndices: encoded.truncatedOptionIndices)
        }
    }

    private func makeArray(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for (index, value) in values.enumerated() { pointer[index] = value }
        return array
    }

    private func readOutput(_ name: String, from output: MLFeatureProvider) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue,
            array.shape.map({ $0.intValue }) == [1, Self.maximumOptions], array.dataType == .float32
        else {
            throw CuaS1FormsError.invalidOutput("\(name) must be float32 [1, 32]")
        }
        let values = (0..<Self.maximumOptions).map { array[$0].floatValue }
        guard values.allSatisfy(\.isFinite) else {
            throw CuaS1FormsError.invalidOutput("\(name) contains a nonfinite value")
        }
        return values
    }

    private static func validateModel(_ description: MLModelDescription) throws {
        let inputs = description.inputDescriptionsByName
        let expected = [
            "context_ids": [1, contextByteLimit],
            "option_ids": [1, maximumOptions, optionByteLimit],
            "option_mask": [1, maximumOptions],
        ]
        guard Set(inputs.keys) == Set(expected.keys) else {
            throw CuaS1FormsError.invalidModel("Unexpected input names")
        }
        for (name, shape) in expected {
            guard let constraint = inputs[name]?.multiArrayConstraint,
                constraint.dataType == .int32, constraint.shape.map({ $0.intValue }) == shape
            else {
                throw CuaS1FormsError.invalidModel("Unexpected shape or type for \(name)")
            }
        }
        for name in ["logits", "probabilities"] {
            guard let constraint = description.outputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.dataType == .float32, constraint.shape.map({ $0.intValue }) == [1, maximumOptions]
            else {
                throw CuaS1FormsError.invalidModel("Unexpected shape or type for \(name)")
            }
        }
    }

    private static func defaultCacheDirectory() -> URL {
        let manager = FileManager.default
        let base =
            manager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? manager.temporaryDirectory
        return base.appendingPathComponent("FluidAudio/Models", isDirectory: true)
    }
}
