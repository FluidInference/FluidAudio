@preconcurrency import CoreML
import Foundation

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
    /// - Parameter useFp32Encoder: load the fp32 encoder fallback instead of the
    ///   fp16/ANE encoder. Use on hardware without a Neural Engine (the fp16
    ///   encoder is correct only on ANE).
    public static func downloadAndLoad(
        useFp32Encoder: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SenseVoiceModels {
        let directory = try await download(progressHandler: progressHandler)
        return try load(from: directory, useFp32Encoder: useFp32Encoder)
    }

    /// Download the repo into the shared model cache; returns the model directory.
    public static func download(
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.senseVoiceSmall.folderName, isDirectory: true)

        if !force && modelsExist(at: targetDir) {
            logger.info("SenseVoice models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }

        logger.info("Downloading SenseVoice models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.senseVoiceSmall, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded SenseVoice models")
        return targetDir
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let required = [
            ModelNames.SenseVoice.preprocessorFile,
            ModelNames.SenseVoice.encoderFile,
            ModelNames.SenseVoice.vocabularyFile,
        ]
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// Load models from a directory that already contains the artifacts.
    public static func load(from directory: URL, useFp32Encoder: Bool = false) throws -> SenseVoiceModels {
        // Preprocessor must run fp32 on CPU (power-spectrum/log exceed fp16 range,
        // and the big identity convs fail ANE compile).
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        // The fp16 encoder is numerically correct only on the Neural Engine;
        // the fp32 fallback can run anywhere.
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = useFp32Encoder ? .all : .cpuAndNeuralEngine

        let preprocessor = try loadModel(
            named: ModelNames.SenseVoice.preprocessor, from: directory, configuration: cpuConfig)
        let encoderName = useFp32Encoder ? ModelNames.SenseVoice.encoderFp32 : ModelNames.SenseVoice.encoder
        let encoder = try loadModel(named: encoderName, from: directory, configuration: encoderConfig)
        let vocabulary = try loadVocabulary(from: directory)

        logger.info("Loaded SenseVoice (encoder: \(useFp32Encoder ? "fp32" : "fp16/ANE"), vocab: \(vocabulary.count))")
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
            return appSupport
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return fm.temporaryDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
