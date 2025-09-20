import CoreML
import Foundation
import OSLog

@available(macOS 13.0, *)
public struct TtsModels {
    public let kokoro: MLModel

    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "TtsModels")

    public init(kokoro: MLModel) {
        self.kokoro = kokoro
    }

    public static func download(
        from repo: String = "FluidInference/kokoro-82m-coreml",
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> TtsModels {
        let modelName = ModelNames.TTS.kokoroBundle
        let cacheDirectory = try getCacheDirectory()
        // Pass Models subdirectory so models end up in ~/.cache/fluidaudio/Models/kokoro/
        let modelsDirectory = cacheDirectory.appendingPathComponent("Models")
        let dict = try await DownloadUtils.loadModels(
            .kokoro,
            modelNames: [modelName],
            directory: modelsDirectory,
            computeUnits: .cpuAndNeuralEngine
        )
        guard let kokoro = dict[modelName] else {
            throw TTSError.modelNotFound(modelName)
        }
        return TtsModels(kokoro: kokoro)
    }

    private static func getCacheDirectory() throws -> URL {
        let baseDirectory: URL
        #if os(macOS)
        baseDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard let first = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw TTSError.processingFailed("Failed to locate caches directory")
        }
        baseDirectory = first
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
