@preconcurrency import CoreML
import Foundation

/// On-device laya typed decisions: `choice`, `score`, and `noul` answers with calibrated probabilities,
/// one encoder pass per question and no generated tokens.
///
/// The Core ML conversion ships fixed-length buckets (128, 256, 512, 1024 tokens). A prompt runs on the
/// smallest loaded bucket that fits; longer states are truncated on the right by the largest one,
/// exactly as laya does for its own `max_len`. Calls on a manager are serialized by the actor.
public actor LayaManager {
    /// Option slots in every exported bucket.
    public static let maximumOptions = 32

    /// Which buckets to load and where each should run.
    public struct Configuration: Sendable {
        /// Sequence-length buckets to load, e.g. `[128, 512]`. Each bucket maps its own 640 MB of weights.
        public var lengths: [Int]
        /// Compute units per bucket. Defaults: 128 → CPU+ANE (fastest measured), longer → all units,
        /// where the GPU wins because ANE attention cost grows quadratically with length.
        public var computeUnits: [Int: MLComputeUnits]

        public init(lengths: [Int] = [128, 512], computeUnits: [Int: MLComputeUnits] = [:]) {
            self.lengths = lengths
            self.computeUnits = computeUnits
        }

        func units(for length: Int) -> MLComputeUnits {
            computeUnits[length] ?? (length <= 128 ? .cpuAndNeuralEngine : .all)
        }
    }

    private struct Bucket: Sendable {
        let length: Int
        let model: MLModel
    }

    private let buckets: [Bucket]
    private let tokenizer: LayaTokenizer
    private let builder: LayaSequenceBuilder
    private let temperature: [Float]
    private let temperatureByOptions: [String: Float]

    /// Loaded bucket lengths, ascending.
    public nonisolated let lengths: [Int]
    /// Option-token budget of the checkpoint (`head_max_len`).
    public nonisolated let headMaxLength: Int

    /// Wrap already loaded bucket models plus their tokenizer. Models must share one checkpoint.
    public init(models: [MLModel], tokenizer: LayaTokenizer) throws {
        guard !models.isEmpty else { throw LayaError.invalidModel("At least one bucket model is required") }
        var buckets: [Bucket] = []
        var calibration: LayaCalibration?
        for model in models {
            let metadata = try LayaCalibration(model: model)
            if let existing = calibration, !existing.sharesCheckpoint(with: metadata) {
                throw LayaError.invalidModel("Bucket models come from different checkpoints")
            }
            calibration = metadata
            try Self.validate(model.modelDescription, length: metadata.length)
            buckets.append(Bucket(length: metadata.length, model: model))
        }
        guard let calibration else { throw LayaError.invalidModel("Missing calibration metadata") }
        self.buckets = buckets.sorted { $0.length < $1.length }
        self.lengths = self.buckets.map(\.length)
        self.tokenizer = tokenizer
        self.headMaxLength = calibration.headMaxLength
        self.temperature = calibration.temperature
        self.temperatureByOptions = calibration.temperatureByOptions
        self.builder = LayaSequenceBuilder(tokenizer: tokenizer, headMaxLength: calibration.headMaxLength)
    }

    /// Download the requested buckets and tokenizer from HuggingFace into FluidAudio's shared cache and load them.
    /// - Parameter cacheDirectory: The parent Models directory, not the repository subdirectory.
    public static func load(
        cacheDirectory: URL? = nil,
        configuration: Configuration = Configuration(),
        progressHandler: ProgressHandler? = nil
    ) async throws -> LayaManager {
        let root = cacheDirectory ?? defaultCacheDirectory()
        let repoDirectory = root.appendingPathComponent(Repo.laya.folderName)
        let modelFiles = try configuration.lengths.map { try ModelNames.Laya.modelFile(length: $0) }
        let required = Set(modelFiles + [ModelNames.Laya.tokenizerFile])
        return try await ModelHub.loadWithRecovery(
            .laya, directory: root, requiredFiles: required, progressHandler: progressHandler
        ) {
            try await load(from: repoDirectory, configuration: configuration)
        }
    }

    /// Load bucket bundles (`.mlmodelc` or `.mlpackage`) and `tokenizer.json` from a local directory.
    /// This path never downloads or recovers by accessing the network.
    public static func load(
        from directory: URL,
        configuration: Configuration = Configuration()
    ) async throws -> LayaManager {
        guard directory.isFileURL else { throw LayaError.invalidAsset("A local directory URL is required") }
        let tokenizerURL = directory.appendingPathComponent(ModelNames.Laya.tokenizerFile)
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw LayaError.invalidAsset("Missing \(ModelNames.Laya.tokenizerFile) in \(directory.path)")
        }
        let tokenizer = try LayaTokenizer(tokenizerJsonURL: tokenizerURL)
        var models: [MLModel] = []
        for length in configuration.lengths {
            let bundle = directory.appendingPathComponent(try ModelNames.Laya.modelFile(length: length))
            let package = bundle.deletingPathExtension().appendingPathExtension("mlpackage")
            let compiledURL: URL
            if FileManager.default.fileExists(atPath: bundle.path) {
                compiledURL = bundle
            } else if FileManager.default.fileExists(atPath: package.path) {
                let prepared = try CuaS1FormsPackage.prepare(package)
                defer { prepared.cleanup() }
                compiledURL = try await MLModel.compileModel(at: prepared.url)
            } else {
                throw LayaError.invalidAsset("Missing \(bundle.lastPathComponent) in \(directory.path)")
            }
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = configuration.units(for: length)
            models.append(try await MLModel.load(contentsOf: compiledURL, configuration: modelConfiguration))
        }
        return try LayaManager(models: models, tokenizer: tokenizer)
    }

    /// Answer one typed question about a state.
    ///
    /// The state is plain text; serialize structured state to JSON yourself if needed. The prompt runs on the
    /// smallest loaded bucket that holds it without truncation, otherwise on the largest with the state cut.
    public func answer(state: String, question: LayaQuestion) throws -> LayaAnswer {
        try Task.checkCancellation()
        let parts = try builder.parts(state: state, question: question)
        guard let bucket = buckets.first(where: { parts.untruncatedLength <= $0.length }) ?? buckets.last else {
            throw LayaError.invalidModel("No bucket loaded")
        }
        let sequence = try builder.sequence(from: parts, maximumLength: bucket.length)
        let output = try autoreleasepool { try predict(sequence, question: question, bucket: bucket) }
        return calibrate(output, question: question, sequence: sequence, bucket: bucket)
    }

    /// Answer several questions about the same state, one encoder pass each.
    public func answer(state: String, questions: [LayaQuestion]) throws -> [LayaAnswer] {
        try questions.map { try answer(state: state, question: $0) }
    }

    /// Token ids for text, exposed for parity checks against the reference tokenizer.
    public nonisolated func encode(_ text: String) -> [Int] {
        tokenizer.encode(text)
    }

    /// The exact model-input token sequence for a question at a bucket length, for parity checks.
    public func tokenSequence(state: String, question: LayaQuestion, length: Int) throws -> (ids: [Int], markers: [Int])
    {
        let sequence = try builder.sequence(state: state, question: question, maximumLength: length)
        return (sequence.ids, sequence.markers)
    }

    // MARK: - Prediction

    private struct RawOutput {
        let logits: [Float]
        let probabilities: [Float]
        let actionProbability: Float
    }

    private func predict(
        _ sequence: LayaSequenceBuilder.Sequence, question: LayaQuestion, bucket: Bucket
    ) throws
        -> RawOutput
    {
        let length = bucket.length
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let attention = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        let markerMap = try MLMultiArray(
            shape: [1, NSNumber(value: Self.maximumOptions), NSNumber(value: length)], dataType: .float32)
        let questionType = try MLMultiArray(shape: [1, 3], dataType: .float32)
        let idPointer = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPointer = attention.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<length {
            idPointer[index] = index < sequence.ids.count ? Int32(sequence.ids[index]) : Int32(tokenizer.padTokenId)
            maskPointer[index] = index < sequence.ids.count ? 1 : 0
        }
        let markerPointer = markerMap.dataPointer.assumingMemoryBound(to: Float.self)
        markerPointer.initialize(repeating: 0, count: Self.maximumOptions * length)
        for (row, position) in sequence.markers.enumerated() {
            markerPointer[row * length + position] = 1
        }
        let typePointer = questionType.dataPointer.assumingMemoryBound(to: Float.self)
        typePointer.initialize(repeating: 0, count: 3)
        typePointer[question.type.index] = 1

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds, "attention_mask": attention, "marker_map": markerMap, "question_type": questionType,
        ])
        let prediction = try bucket.model.prediction(from: features)
        let count = sequence.markers.count
        let logits = try readOutput("logits", from: prediction, count: Self.maximumOptions)
        let probabilities = try readOutput("probabilities", from: prediction, count: Self.maximumOptions)
        let action = try readOutput("action_probabilities", from: prediction, count: 2)
        guard logits.allSatisfy(\.isFinite), probabilities.allSatisfy(\.isFinite), action.allSatisfy(\.isFinite) else {
            throw LayaError.invalidOutput("Model returned non-finite values")
        }
        return RawOutput(
            logits: Array(logits.prefix(count)), probabilities: Array(probabilities.prefix(count)),
            actionProbability: action[0])
    }

    private func calibrate(
        _ output: RawOutput, question: LayaQuestion, sequence: LayaSequenceBuilder.Sequence, bucket: Bucket
    ) -> LayaAnswer {
        let count = output.logits.count
        let scale =
            temperatureByOptions[Self.temperatureBucket(question.type, optionCount: count)]
            ?? temperature[question.type.index]
        let scaled = output.logits.map { Double($0) / max(1e-3, Double(scale)) }
        let peak = scaled.max() ?? 0
        let exponentials = scaled.map { exp($0 - peak) }
        let total = exponentials.reduce(0, +)
        let probabilities = exponentials.map { $0 / total }
        let confidence: Double
        if question.type == .noul, probabilities.count == 2 {
            confidence = max(probabilities[1], 1 - probabilities[1])
        } else {
            let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
            confidence = count < 2 ? 1 : min(1, max(0, 1 - entropy / log(Double(count))))
        }
        return LayaAnswer(
            question: question,
            probabilities: probabilities.map { Float($0) },
            rawProbabilities: output.probabilities,
            logits: output.logits,
            confidence: Float(confidence),
            actionProbability: output.actionProbability,
            tokenCount: sequence.ids.count,
            bucketLength: bucket.length,
            stateWasTruncated: sequence.stateWasTruncated)
    }

    /// Key for laya's per-cardinality temperature table (`temp_bucket`).
    static func temperatureBucket(_ type: LayaQuestionType, optionCount: Int) -> String {
        let size = optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
        return "\(type.rawValue):\(size)"
    }

    private func readOutput(_ name: String, from output: MLFeatureProvider, count: Int) throws -> [Float] {
        guard let array = output.featureValue(for: name)?.multiArrayValue, array.count == count,
            array.dataType == .float32
        else {
            throw LayaError.invalidOutput("\(name) must be float32 with \(count) values")
        }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        return (0..<count).map { pointer[$0] }
    }

    private static func validate(_ description: MLModelDescription, length: Int) throws {
        let inputs = description.inputDescriptionsByName
        let expected: [String: ([Int], MLMultiArrayDataType)] = [
            "input_ids": ([1, length], .int32),
            "attention_mask": ([1, length], .int32),
            "marker_map": ([1, maximumOptions, length], .float32),
            "question_type": ([1, 3], .float32),
        ]
        guard Set(inputs.keys) == Set(expected.keys) else {
            throw LayaError.invalidModel("Unexpected input names \(inputs.keys.sorted())")
        }
        for (name, (shape, type)) in expected {
            guard let constraint = inputs[name]?.multiArrayConstraint, constraint.dataType == type,
                constraint.shape.map({ $0.intValue }) == shape
            else {
                throw LayaError.invalidModel("Unexpected shape or type for \(name)")
            }
        }
        for (name, shape) in [
            "logits": [1, maximumOptions], "probabilities": [1, maximumOptions], "action_probabilities": [1, 2],
        ] {
            guard let constraint = description.outputDescriptionsByName[name]?.multiArrayConstraint,
                constraint.dataType == .float32, constraint.shape.map({ $0.intValue }) == shape
            else {
                throw LayaError.invalidModel("Unexpected shape or type for \(name)")
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

/// Calibration and shape metadata written into each bucket by the Mobius conversion.
struct LayaCalibration: Equatable {
    let length: Int
    let maxOptions: Int
    let headMaxLength: Int
    let temperature: [Float]
    let temperatureByOptions: [String: Float]

    /// Same calibration and option budget; the bucket length is expected to differ.
    func sharesCheckpoint(with other: LayaCalibration) -> Bool {
        maxOptions == other.maxOptions && headMaxLength == other.headMaxLength && temperature == other.temperature
            && temperatureByOptions == other.temperatureByOptions
    }

    init(model: MLModel) throws {
        guard let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] else {
            throw LayaError.invalidModel("Missing creator-defined metadata")
        }
        func integer(_ key: String) throws -> Int {
            guard let value = metadata[key].flatMap(Int.init) else {
                throw LayaError.invalidModel("Metadata \(key) is missing")
            }
            return value
        }
        self.length = try integer("length")
        self.maxOptions = try integer("max_options")
        self.headMaxLength = try integer("head_max_len")
        guard let temperatureData = metadata["temperature"]?.data(using: .utf8),
            let temperature = try? JSONDecoder().decode([Float].self, from: temperatureData), temperature.count == 3
        else {
            throw LayaError.invalidModel("Metadata temperature must list three values")
        }
        self.temperature = temperature
        guard let byOptionsData = metadata["temperature_by_options"]?.data(using: .utf8),
            let byOptions = try? JSONDecoder().decode([String: Float].self, from: byOptionsData)
        else {
            throw LayaError.invalidModel("Metadata temperature_by_options must be a JSON object")
        }
        self.temperatureByOptions = byOptions
        guard maxOptions == LayaManager.maximumOptions else {
            throw LayaError.invalidModel("Expected \(LayaManager.maximumOptions) option slots, found \(maxOptions)")
        }
    }
}
