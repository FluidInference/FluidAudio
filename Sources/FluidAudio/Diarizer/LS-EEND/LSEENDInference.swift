import Foundation
import CoreML
import Accelerate

public class LSEENDModel {
    public let metadata: LSEENDMetadata

    private let model: MLModel

    private let lock = NSLock()

    private static let logger = AppLogger(category: "LS-EEND Model")

    // MARK: - Init

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuOnly) throws {
        // Load the model from the URL
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)

        // Load the config from metadata
        guard let userMetadata = self.model.modelDescription.metadata[.creatorDefinedKey] as? [String: Any],
            let json = userMetadata["config"] as? String
        else {
            throw LSEENDError.initializationFailed("No `config` found in model metadata")
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.metadata = try decoder.decode(LSEENDMetadata.self, from: Data(json.utf8))
    }

    /// Download LS-EEND models from HuggingFace.
    ///
    /// - Parameters:
    ///   - variant: The model variant to load (default: `.dihard3`).
    ///   - stepSize: The model step size to load (default: `.step100ms`).
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: Model compute units (`.cpuOnly` seems to be fastest for this model)
    /// - Returns: LS-EEND Model Wrapper
    public static func loadFromHuggingFace(
        variant: LSEENDVariant = .dihard3,
        stepSize: LSEENDStepSize = .step100ms,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> LSEENDModel {
        let directory =
            cacheDirectory
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models")

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let repo = variant.repo
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let modelRelPath = variant.fileName(forStep: stepSize)
        let fullRelPath = repo.subPath.map { "\($0)/\(modelRelPath)" } ?? modelRelPath
        let modelURL = repoPath.appendingPathComponent(fullRelPath)

        let modelExists = FileManager.default.fileExists(atPath: modelURL.path)

        if !modelExists {
            // Narrow to just the one mlmodelc — listing the whole step dir
            // is fine here since each step dir contains only its own mlmodelc.
            logger.info("Models not found in cache at \(modelURL.path); downloading \(fullRelPath)…")
            try await DownloadUtils.downloadSubdirectory(
                repo, subdirectory: fullRelPath, to: repoPath
            )
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LSEENDError.initializationFailed(
                "HF download completed but mlmodelc missing at \(modelURL.path). "
                    + "Expected HF path: \(modelRelPath)"
            )
        }

        return try LSEENDModel(modelURL: modelURL, computeUnits: computeUnits)
    }

    // MARK: - Inference

    public func predict(from input: LSEENDInput) throws -> [Float] {
        try autoreleasepool {
            lock.lock()
            defer { lock.unlock() }

            let prediction = try model.prediction(from: input)

            guard let probsMA = prediction.featureValue(for: "probs")?.multiArrayValue,
                let encKvMA = prediction.featureValue(for: "enc_kv_new")?.multiArrayValue,
                let encScaleMA = prediction.featureValue(for: "enc_scale_new")?.multiArrayValue,
                let encConvCacheMA = prediction.featureValue(for: "enc_conv_cache_new")?.multiArrayValue,
                let cnnWindowMA = prediction.featureValue(for: "cnn_window_new")?.multiArrayValue,
                let decKvMA = prediction.featureValue(for: "dec_kv_new")?.multiArrayValue,
                let decScaleMA = prediction.featureValue(for: "dec_scale_new")?.multiArrayValue
            else {
                throw LSEENDError.inferenceFailed("Failed to extract predictions from CoreML model.")
            }

            // Update state
            input.state.encRetKv = encKvMA
            input.state.encRetScale = encScaleMA
            input.state.encConvCache = encConvCacheMA
            input.state.cnnWindow = cnnWindowMA
            input.state.decRetKv = decKvMA
            input.state.decRetScale = decScaleMA

            // Copy speaker sigmoids and skip warmup frames
            let warmup = input.warmupFrames
            let outputFrames = metadata.chunkSize - warmup
            let outputSpeakers = metadata.maxSpeakers
            guard outputFrames > 0, outputSpeakers > 0 else { return [] }

            guard probsMA.strides.last?.intValue == 1 else {
                throw LSEENDError.inferenceFailed(
                    "Probs innermost stride must be 1. CoreML model produced strides: \(probsMA.strides).")
            }
            let frameStride = probsMA.strides[1].intValue

            var probsOut = [Float](repeating: 0, count: outputFrames * outputSpeakers)
            let maBase = probsMA.dataPointer.assumingMemoryBound(to: Float.self)

            probsOut.withUnsafeMutableBufferPointer { flatPtr in
                vDSP_mmov(
                    maBase + warmup * frameStride,
                    flatPtr.baseAddress!,
                    vDSP_Length(outputSpeakers),
                    vDSP_Length(outputFrames),
                    vDSP_Length(frameStride),
                    vDSP_Length(outputSpeakers)
                )
            }

            return probsOut
        }
    }
}

public class LSEENDInput: MLFeatureProvider {
    public var state: LSEENDState
    public let melFeatures: MLMultiArray
    public let decoderMask: MLMultiArray
    public var warmupFrames: Int = 0

    public var featureNames: Set<String> {
        [
            "features",
            "enc_kv", "enc_scale",
            "enc_conv_cache", "cnn_window",
            "dec_kv", "dec_scale",
            "valid_mask",
        ]
    }

    public init(from metadata: LSEENDMetadata, state: consuming LSEENDState? = nil) throws {
        self.state = try state ?? LSEENDState(from: metadata)
        let T = NSNumber(value: metadata.chunkSize)
        let M = NSNumber(value: metadata.melFrames)
        let N = NSNumber(value: metadata.nMels)
        self.melFeatures = try MLMultiArray(shape: [1, M, N], dataType: .float32)
        self.decoderMask = try MLMultiArray(shape: [T], dataType: .float32)
    }

    /// Reset state
    @inline(__always)
    public func resetState() {
        state.reset()
    }

    @inline(__always)
    public func loadInputs<C: AccelerateBuffer>(
        melFeatures newMelFeatures: C,
        decoderMask newDecoderMask: C,
        warmupFrames: Int? = nil
    ) throws where C.Element == Float {
        try Self.load(decoderMask, from: newDecoderMask)
        try Self.load(melFeatures, from: newMelFeatures)
        self.warmupFrames = warmupFrames ?? newDecoderMask.withUnsafeBufferPointer { $0.count(where: \.isZero) }
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "features": return MLFeatureValue(multiArray: melFeatures)
        case "enc_kv": return MLFeatureValue(multiArray: state.encRetKv)
        case "enc_scale": return MLFeatureValue(multiArray: state.encRetScale)
        case "enc_conv_cache": return MLFeatureValue(multiArray: state.encConvCache)
        case "cnn_window": return MLFeatureValue(multiArray: state.cnnWindow)
        case "dec_kv": return MLFeatureValue(multiArray: state.decRetKv)
        case "dec_scale": return MLFeatureValue(multiArray: state.decRetScale)
        case "valid_mask": return MLFeatureValue(multiArray: decoderMask)
        default: return nil
        }
    }

    @inline(__always)
    private static func load<C: AccelerateBuffer>(
        _ multiArray: MLMultiArray,
        from buffer: C
    ) throws {
        guard buffer.count == multiArray.count else {
            throw LSEENDError.invalidInputSize(
                "Input size mismatch: new=\(buffer.count) expected=\(multiArray.count)")
        }

        _ = buffer.withUnsafeBufferPointer { buf in
            memcpy(
                multiArray.dataPointer, buf.baseAddress,
                buf.count * MemoryLayout<Float>.stride)
        }
    }
}
