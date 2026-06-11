@preconcurrency import CoreML
import Foundation

/// Loaded FSMN-VAD CoreML models.
///
/// 2 stages from `FluidInference/fsmn-vad-coreml`:
///   - `preprocessor` (fp32, CPU): waveform -> [1, T, 400] features (fbank80 + LFR m=5,n=1)
///   - `scorer` (fp16, ANE): features -> [1, T, 248] frame scores (col 0 = silence prob)
public struct FsmnVadModels: Sendable {

    public let preprocessor: MLModel
    public let scorer: MLModel

    private static let logger = AppLogger(category: "FsmnVadModels")

    public init(preprocessor: MLModel, scorer: MLModel) {
        self.preprocessor = preprocessor
        self.scorer = scorer
    }

    public static func downloadAndLoad(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> FsmnVadModels {
        try load(from: try await download(progressHandler: progressHandler))
    }

    public static func download(
        force: Bool = false, progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let root = modelsRootDirectory()
        let dir = root.appendingPathComponent(Repo.fsmnVad.folderName, isDirectory: true)
        if !force && modelsExist(at: dir) {
            logger.info("FSMN-VAD models already present at: \(dir.path)")
            return dir
        }
        if force { try? FileManager.default.removeItem(at: dir) }
        logger.info("Downloading FSMN-VAD models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.fsmnVad, to: root, progressHandler: progressHandler)
        return dir
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        return [ModelNames.FsmnVad.preprocessorFile, ModelNames.FsmnVad.scorerFile].allSatisfy {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    public static func load(from directory: URL) throws -> FsmnVadModels {
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        let ane = MLModelConfiguration()
        ane.computeUnits = .cpuAndNeuralEngine
        let pre = try loadModel(named: ModelNames.FsmnVad.preprocessor, from: directory, configuration: cpu)
        let scorer = try loadModel(named: ModelNames.FsmnVad.scorer, from: directory, configuration: ane)
        logger.info("Loaded FSMN-VAD models")
        return FsmnVadModels(preprocessor: pre, scorer: scorer)
    }

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
            throw ASRError.processingFailed("FSMN-VAD model not found: \(name)")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
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
