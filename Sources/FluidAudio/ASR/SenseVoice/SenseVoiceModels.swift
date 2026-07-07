@preconcurrency import CoreML
import Foundation

/// SenseVoice encoder weight precision. fp16 and int8 run on the Neural Engine
/// (int8 is ~half the size, accuracy-neutral); fp32 is the non-ANE fallback.
public enum SenseVoiceEncoderPrecision: String, Sendable {
    case fp16
    case int8
    case fp32

    var modelName: String {
        switch self {
        case .fp16: return ModelNames.SenseVoice.encoder
        case .int8: return ModelNames.SenseVoice.encoderInt8
        case .fp32: return ModelNames.SenseVoice.encoderFp32
        }
    }

    var computeUnits: MLComputeUnits {
        self == .fp32 ? .all : .cpuAndNeuralEngine
    }
}

/// Loaded SenseVoiceSmall CoreML models + vocabulary.
///
/// 3 stages from `FluidInference/sensevoice-small-coreml`:
///   - `preprocessor` (fp32, CPU): waveform → [1, T, 560] LFR features
///   - `encoder` (fp16 on ANE, or fp32 fallback): features + lang/textnorm → CTC logits
///   - `vocabulary`: 25055 SentencePiece tokens (id → piece)
public struct SenseVoiceModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "SenseVoiceModels")

    public init(preprocessor: MLModel, encoder: MLModel, vocabulary: [Int: String]) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.vocabulary = vocabulary
    }

    /// Download (if needed) and load all SenseVoice models.
    ///
    /// - Parameter precision: encoder weight precision (`.fp16` default ANE,
    ///   `.int8` ~half size on ANE, `.fp32` fallback for non-ANE hardware).
    public static func downloadAndLoad(
        precision: SenseVoiceEncoderPrecision = .fp16,
        progressHandler: ProgressHandler? = nil
    ) async throws -> SenseVoiceModels {
        let directory = try await download(precision: precision, progressHandler: progressHandler)
        return try load(from: directory, precision: precision)
    }

    /// Download the repo into the shared model cache; returns the model directory.
    ///
    /// `precision` ensures the requested encoder variant is present — a cache that
    /// predates a variant (e.g. fp16-only) re-fetches just the missing file
    /// (`ModelHub.download` skips files already on disk).
    public static func download(
        precision: SenseVoiceEncoderPrecision = .fp16,
        force: Bool = false,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.senseVoiceSmall.folderName, isDirectory: true)

        if !force && modelsExist(at: targetDir, precision: precision) {
            logger.info("SenseVoice models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }

        logger.info("Downloading SenseVoice models from HuggingFace...")
        try await ModelHub.download(
            .senseVoiceSmall,
            to: modelsRoot,
            variant: precision.rawValue,
            progressHandler: progressHandler
        )
        logger.info("Successfully downloaded SenseVoice models")
        return targetDir
    }

    public static func modelsExist(
        at directory: URL, precision: SenseVoiceEncoderPrecision = .fp16
    ) -> Bool {
        let fm = FileManager.default
        let required = [
            ModelNames.SenseVoice.preprocessorFile,
            precision.modelName + ".mlmodelc",
            ModelNames.SenseVoice.vocabularyFile,
        ]
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// Load models from a directory that already contains the artifacts.
    public static func load(
        from directory: URL, precision: SenseVoiceEncoderPrecision = .fp16
    ) throws -> SenseVoiceModels {
        // Preprocessor must run fp32 on CPU (power-spectrum/log exceed fp16 range,
        // and the big identity convs fail ANE compile).
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        // fp16/int8 encoders are correct on the Neural Engine; fp32 runs anywhere.
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = precision.computeUnits

        let preprocessor = try loadModel(
            named: ModelNames.SenseVoice.preprocessor, from: directory, configuration: cpuConfig)
        let encoder = try loadModel(named: precision.modelName, from: directory, configuration: encoderConfig)
        let vocabulary = try loadVocabulary(from: directory)

        logger.info("Loaded SenseVoice (encoder: \(precision.rawValue), vocab: \(vocabulary.count))")
        return SenseVoiceModels(preprocessor: preprocessor, encoder: encoder, vocabulary: vocabulary)
    }

    // MARK: - Private

    private static func loadModel(
        named name: String, from directory: URL, configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            modelURL = try MLModel.compileModel(at: packagePath)
        } else {
            throw ASRError.processingFailed("SenseVoice model not found: \(name)")
        }
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let path = directory.appendingPathComponent(ModelNames.SenseVoice.vocabularyFile)
        let data = try Data(contentsOf: path)
        // Canonical format: JSON array ["<unk>", "<s>", ...].
        if let arr = try? JSONSerialization.jsonObject(with: data) as? [String] {
            var v: [Int: String] = [:]
            for (i, tok) in arr.enumerated() { v[i] = tok }
            return v
        }
        if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            var v: [Int: String] = [:]
            for (k, tok) in dict { if let i = Int(k) { v[i] = tok } }
            return v
        }
        throw ASRError.processingFailed("Failed to parse vocab.json (expected array or dict)")
    }

    private static func modelsRootDirectory() -> URL {
        let fm = FileManager.default
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            return
                appSupport
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return fm.temporaryDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
