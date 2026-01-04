@preconcurrency import CoreML
import Foundation
import OSLog

/// Models container for the hybrid TDT-CTC 110M model.
/// Contains all 5 components with shared encoder.
public struct HybridAsrModels: Sendable {
    public let preprocessor: MLModel
    public let encoder: MLModel
    public let ctcHead: MLModel
    public let decoder: MLModel
    public let jointDecision: MLModel
    public let vocabulary: [Int: String]
    public let vocabSize: Int
    public let blankId: Int

    private static let logger = AppLogger(category: "HybridAsrModels")

    public init(
        preprocessor: MLModel,
        encoder: MLModel,
        ctcHead: MLModel,
        decoder: MLModel,
        jointDecision: MLModel,
        vocabulary: [Int: String]
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.ctcHead = ctcHead
        self.decoder = decoder
        self.jointDecision = jointDecision
        self.vocabulary = vocabulary
        self.vocabSize = vocabulary.count
        self.blankId = vocabulary.count  // Blank token is after vocab
    }

    /// Load hybrid models from a directory containing all components.
    public static func load(from directory: URL) async throws -> HybridAsrModels {
        logger.info("Loading hybrid models from \(directory.path)")

        // Load all CoreML models
        let preprocessorURL = directory.appendingPathComponent("Preprocessor.mlmodelc")
        let encoderURL = directory.appendingPathComponent("Encoder.mlmodelc")
        let ctcHeadURL = directory.appendingPathComponent("CTCHead.mlmodelc")
        let decoderURL = directory.appendingPathComponent("Decoder.mlmodelc")
        let jointURL = directory.appendingPathComponent("JointDecision.mlmodelc")
        let vocabURL = directory.appendingPathComponent("vocab.json")

        // Verify all files exist
        let fm = FileManager.default
        for url in [preprocessorURL, encoderURL, ctcHeadURL, decoderURL, jointURL, vocabURL] {
            guard fm.fileExists(atPath: url.path) else {
                throw AsrModelsError.modelNotFound(url.lastPathComponent, url)
            }
        }

        // Load models in parallel
        async let preprocessor = try MLModel.load(contentsOf: preprocessorURL)
        async let encoder = try MLModel.load(contentsOf: encoderURL)
        async let ctcHead = try MLModel.load(contentsOf: ctcHeadURL)
        async let decoder = try MLModel.load(contentsOf: decoderURL)
        async let joint = try MLModel.load(contentsOf: jointURL)

        // Load vocabulary
        let vocabData = try Data(contentsOf: vocabURL)
        let vocabDict = try JSONDecoder().decode([String: String].self, from: vocabData)
        var vocabulary: [Int: String] = [:]
        for (key, value) in vocabDict {
            if let id = Int(key) {
                vocabulary[id] = value
            }
        }

        logger.info("Loaded hybrid models: vocab_size=\(vocabulary.count)")

        return try await HybridAsrModels(
            preprocessor: preprocessor,
            encoder: encoder,
            ctcHead: ctcHead,
            decoder: decoder,
            jointDecision: joint,
            vocabulary: vocabulary
        )
    }

    /// Default model path in Application Support
    public static func defaultModelPath() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("FluidAudio/Models/parakeet-ctc-110m-coreml")
    }

    /// Download and load hybrid models
    public static func downloadAndLoad() async throws -> HybridAsrModels {
        let modelPath = defaultModelPath()
        if FileManager.default.fileExists(atPath: modelPath.path) {
            return try await load(from: modelPath)
        }
        throw AsrModelsError.modelNotFound(
            //"parakeet-tdt-ctc-110m-hybrid",
            "parakeet-ctc-110m-coreml",
            modelPath
        )
    }
}
