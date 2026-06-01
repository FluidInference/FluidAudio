@preconcurrency import CoreML
import Foundation

/// Loaded CAM++ CoreML models (speaker embedding).
///
/// 2 stages from `FluidInference/campplus-coreml`:
///   - `preprocessor` (fp32, CPU): waveform -> [1, T, 80] fbank
///   - `model` (fp16, ANE): fbank -> [1, 192] speaker embedding
public struct CampPlusModels: Sendable {

    public let preprocessor: MLModel
    public let model: MLModel

    private static let logger = AppLogger(category: "CampPlusModels")

    public init(preprocessor: MLModel, model: MLModel) {
        self.preprocessor = preprocessor
        self.model = model
    }

    public static func downloadAndLoad(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CampPlusModels {
        try load(from: try await download(progressHandler: progressHandler))
    }

    public static func download(
        force: Bool = false, progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.campPlus.folderName, isDirectory: true)
        if !force && modelsExist(at: targetDir) {
            logger.info("CAM++ models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }
        logger.info("Downloading CAM++ models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.campPlus, to: modelsRoot, progressHandler: progressHandler)
        return targetDir
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        return [ModelNames.CampPlus.preprocessorFile, ModelNames.CampPlus.modelFile].allSatisfy {
            fm.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    public static func load(from directory: URL) throws -> CampPlusModels {
        let cpu = MLModelConfiguration()
        cpu.computeUnits = .cpuOnly
        // CAM++ uses a dynamic time dim (RangeDim) which the ANE compiler rejects;
        // it's tiny (~7.2M), so run on CPU/GPU. Dynamic length avoids padding
        // corrupting the statistics-pooled embedding.
        let gpu = MLModelConfiguration()
        gpu.computeUnits = .cpuAndGPU
        let pre = try loadModel(named: ModelNames.CampPlus.preprocessor, from: directory, configuration: cpu)
        let model = try loadModel(named: ModelNames.CampPlus.model, from: directory, configuration: gpu)
        logger.info("Loaded CAM++ speaker-embedding models")
        return CampPlusModels(preprocessor: pre, model: model)
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
            throw ASRError.processingFailed("CAM++ model not found: \(name)")
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
