import CoreML
import Foundation
import OSLog

@available(macOS 13.0, *)
public struct TTSModels {
    public let kokoro: MLModel

    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "TTSModels")

    public init(kokoro: MLModel) {
        self.kokoro = kokoro
    }

    public static func download(
        from repo: String = "FluidInference/coreml-kokoro",
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> TTSModels {
        let modelName = "kokoro.mlmodelc"
        let modelURL = try await downloadModel(
            from: repo,
            modelName: modelName,
            progressHandler: progressHandler
        )

        let kokoro = try await loadCompiledModel(at: modelURL, modelName: modelName)

        return TTSModels(kokoro: kokoro)
    }

    private static func downloadModel(
        from repo: String,
        modelName: String,
        progressHandler: DownloadUtils.ProgressHandler?
    ) async throws -> URL {
        let cacheDirectory = try getCacheDirectory()
        let modelPath = cacheDirectory.appendingPathComponent(repo).appendingPathComponent(modelName)

        if FileManager.default.fileExists(atPath: modelPath.path) {
            logger.info("Using cached model: \(modelName)")
            return modelPath
        }

        logger.info("Downloading model: \(modelName) from \(repo)")

        let baseURL = "https://huggingface.co/\(repo)/resolve/main"
        let files = [
            "coremldata.bin",
            "model.mil",
            "weights/weight.bin",
        ]

        try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: modelPath.appendingPathComponent("weights"),
            withIntermediateDirectories: true
        )

        for (index, file) in files.enumerated() {
            let fileURL = URL(string: "\(baseURL)/\(modelName)/\(file)")!
            let destinationURL = modelPath.appendingPathComponent(file)

            let progress: Double = Double(index) / Double(files.count)
            progressHandler?(progress)

            try await downloadFile(from: fileURL, to: destinationURL)
        }

        progressHandler?(1.0)
        return modelPath
    }

    private static func downloadFile(from url: URL, to destination: URL) async throws {
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TTSError.downloadFailed("Failed to download file from \(url)")
        }

        try data.write(to: destination)
    }

    private static func loadCompiledModel(at url: URL, modelName: String) async throws -> MLModel {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            configuration.allowLowPrecisionAccumulationOnGPU = true

            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            logger.error("Failed to load model \(modelName): \(error.localizedDescription)")

            if let nsError = error as NSError?,
                nsError.domain == "com.apple.CoreML"
            {
                logger.info("Attempting to re-download corrupted model")
                try? FileManager.default.removeItem(at: url)
                throw TTSError.corruptedModel(modelName)
            }

            throw error
        }
    }

    private static func getCacheDirectory() throws -> URL {
        let baseDirectory: URL
        #if os(macOS)
        baseDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        baseDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")

        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory,
                withIntermediateDirectories: true
            )
        }

        return cacheDirectory
    }

    public static func cacheDirectoryURL() throws -> URL {
        return try getCacheDirectory()
    }

    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()
        options.usesCPUOnly = false
        return options
    }
}

public enum TTSError: LocalizedError {
    case downloadFailed(String)
    case corruptedModel(String)
    case modelNotFound(String)
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .corruptedModel(let name):
            return "Model \(name) is corrupted"
        case .modelNotFound(let name):
            return "Model \(name) not found"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}
