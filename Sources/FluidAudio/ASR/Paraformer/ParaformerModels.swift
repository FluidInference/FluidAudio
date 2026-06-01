@preconcurrency import CoreML
import Foundation

/// Loaded Paraformer-large (zh) CoreML models + vocabulary.
///
/// 4 stages from `FluidInference/paraformer-large-zh-coreml`:
///   - `preprocessor` (fp32, CPU): waveform -> [1,T,560] LFR features
///   - `encoder` (fp16, ANE): SANM encoder
///   - `cifAlphas` (fp16, ANE): enc_out -> per-frame alphas (host integrate-and-fire)
///   - `decoder` (fp16, ANE): parallel decoder -> token logits
///   - `vocabulary`: 8404 CharTokenizer tokens (id -> char)
public struct ParaformerModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let cifAlphas: MLModel
    public let decoder: MLModel
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "ParaformerModels")

    public init(
        preprocessor: MLModel, encoder: MLModel, cifAlphas: MLModel, decoder: MLModel, vocabulary: [Int: String]
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.cifAlphas = cifAlphas
        self.decoder = decoder
        self.vocabulary = vocabulary
    }

    public static func downloadAndLoad(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> ParaformerModels {
        let directory = try await download(progressHandler: progressHandler)
        return try load(from: directory)
    }

    public static func download(
        force: Bool = false, progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.paraformerLargeZh.folderName, isDirectory: true)
        if !force && modelsExist(at: targetDir) {
            logger.info("Paraformer models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }
        logger.info("Downloading Paraformer models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.paraformerLargeZh, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded Paraformer models")
        return targetDir
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let required = [
            ModelNames.ParaformerZh.preprocessorFile,
            ModelNames.ParaformerZh.encoderFile,
            ModelNames.ParaformerZh.cifAlphasFile,
            ModelNames.ParaformerZh.decoderFile,
            ModelNames.ParaformerZh.vocabularyFile,
        ]
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    public static func load(from directory: URL) throws -> ParaformerModels {
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        let ane = MLModelConfiguration()
        ane.computeUnits = .cpuAndNeuralEngine

        let pre = try loadModel(named: ModelNames.ParaformerZh.preprocessor, from: directory, configuration: cpu)
        let enc = try loadModel(named: ModelNames.ParaformerZh.encoder, from: directory, configuration: ane)
        let cif = try loadModel(named: ModelNames.ParaformerZh.cifAlphas, from: directory, configuration: ane)
        let dec = try loadModel(named: ModelNames.ParaformerZh.decoder, from: directory, configuration: ane)
        let vocab = try loadVocabulary(from: directory)
        logger.info("Loaded Paraformer (vocab: \(vocab.count))")
        return ParaformerModels(preprocessor: pre, encoder: enc, cifAlphas: cif, decoder: dec, vocabulary: vocab)
    }

    // MARK: - Private

    private static func loadModel(
        named name: String, from directory: URL, configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        let pkg = directory.appendingPathComponent("\(name).mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: pkg.path) {
            url = try MLModel.compileModel(at: pkg)
        } else {
            throw ASRError.processingFailed("Paraformer model not found: \(name)")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let path = directory.appendingPathComponent(ModelNames.ParaformerZh.vocabularyFile)
        let data = try Data(contentsOf: path)
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
