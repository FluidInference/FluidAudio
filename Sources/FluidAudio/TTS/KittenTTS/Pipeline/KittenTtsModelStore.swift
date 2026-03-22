@preconcurrency import CoreML
import Foundation
import OSLog

/// Actor-based store for KittenTTS CoreML models and voice embeddings.
///
/// Manages loading and caching of the CoreML model (5s or 10s variant)
/// and the voice embedding data (binary float32 files).
public actor KittenTtsModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KittenTtsModelStore")

    private let kittenVariant: KittenTtsVariant
    private var model5s: MLModel?
    private var model10s: MLModel?
    private var voiceCache: [String: [Float]] = [:]
    private var repoDirectory: URL?
    private let directory: URL?

    /// - Parameters:
    ///   - variant: Which KittenTTS variant to use (nano or mini).
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    public init(variant: KittenTtsVariant, directory: URL? = nil) {
        self.kittenVariant = variant
        self.directory = directory
    }

    /// The KittenTTS variant this store manages.
    public var variant: KittenTtsVariant { kittenVariant }

    /// Load all models and voices from cache, downloading if needed.
    public func loadIfNeeded() async throws {
        guard model10s == nil else { return }

        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            KittenTtsConstants.defaultModelsSubdirectory)

        let repo: Repo = kittenVariant == .nano ? .kittenTtsNano : .kittenTtsMini
        let repoDir = modelsDirectory.appendingPathComponent(repo.folderName)

        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: nil)
        let allPresent = requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }

        if !allPresent {
            logger.info("Downloading KittenTTS \(self.kittenVariant.rawValue) models from HuggingFace...")
            try await DownloadUtils.downloadRepo(repo, to: modelsDirectory)
        } else {
            logger.info("KittenTTS \(self.kittenVariant.rawValue) models found in cache")
        }

        self.repoDirectory = repoDir

        // Use CPU+GPU to maintain float32 precision (avoid ANE float16 artifacts).
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        // Load both 5s and 10s models
        let variant5s = ModelNames.KittenTTS.Variant.fiveSecond
        let variant10s = ModelNames.KittenTTS.Variant.tenSecond
        let fileName5s =
            kittenVariant == .nano ? variant5s.nanoFileName() : variant5s.miniFileName()
        let fileName10s =
            kittenVariant == .nano ? variant10s.nanoFileName() : variant10s.miniFileName()

        let modelURL5s = repoDir.appendingPathComponent(fileName5s)
        let modelURL10s = repoDir.appendingPathComponent(fileName10s)

        model5s = try MLModel(contentsOf: modelURL5s, configuration: config)
        logger.info("Loaded \(fileName5s)")

        model10s = try MLModel(contentsOf: modelURL10s, configuration: config)
        logger.info("Loaded \(fileName10s)")

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info(
            "KittenTTS \(self.kittenVariant.rawValue) models loaded in \(String(format: "%.2f", elapsed))s"
        )
    }

    /// Get the 5-second model.
    public func fiveSecondModel() throws -> MLModel {
        guard let model = model5s else {
            throw KittenTTSError.modelNotFound("KittenTTS 5s model not loaded")
        }
        return model
    }

    /// Get the 10-second model.
    public func tenSecondModel() throws -> MLModel {
        guard let model = model10s else {
            throw KittenTTSError.modelNotFound("KittenTTS 10s model not loaded")
        }
        return model
    }

    /// Select the appropriate model based on token count.
    public func model(for tokenCount: Int) throws -> (MLModel, ModelNames.KittenTTS.Variant) {
        let variant: ModelNames.KittenTTS.Variant =
            tokenCount <= ModelNames.KittenTTS.Variant.fiveSecond.maxTokens
            ? .fiveSecond : .tenSecond
        let model =
            variant == .fiveSecond
            ? try fiveSecondModel()
            : try tenSecondModel()
        return (model, variant)
    }

    /// Load and cache voice embedding data for the given voice name.
    public func voiceData(for voice: String) throws -> [Float] {
        if let cached = voiceCache[voice] {
            return cached
        }
        guard let repoDir = repoDirectory else {
            throw KittenTTSError.modelNotFound("KittenTTS repository not loaded")
        }

        let voicesDir = repoDir.appendingPathComponent(ModelNames.KittenTTS.voicesDir)
        let voiceFile = voicesDir.appendingPathComponent("\(voice).bin")

        guard FileManager.default.fileExists(atPath: voiceFile.path) else {
            throw KittenTTSError.modelNotFound(
                "Voice '\(voice)' not found at \(voiceFile.path)")
        }

        let data = try Data(contentsOf: voiceFile)

        let expectedSize: Int
        if kittenVariant == .nano {
            // Nano: 256 floats = 1024 bytes
            expectedSize = KittenTtsConstants.nanoVoiceDim * MemoryLayout<Float>.size
        } else {
            // Mini: 400 × 256 floats = 409600 bytes
            expectedSize =
                KittenTtsConstants.miniVoiceRows * KittenTtsConstants.miniVoiceDim
                * MemoryLayout<Float>.size
        }

        guard data.count == expectedSize else {
            throw KittenTTSError.corruptedModel(
                "Voice '\(voice)' has unexpected size \(data.count) bytes (expected \(expectedSize))"
            )
        }

        let floatCount = data.count / MemoryLayout<Float>.size
        var floats = [Float](repeating: 0, count: floatCount)
        _ = floats.withUnsafeMutableBytes { buffer in
            data.copyBytes(to: buffer)
        }

        voiceCache[voice] = floats
        logger.info("Loaded voice '\(voice)' (\(floatCount) floats)")
        return floats
    }

    // MARK: - Private

    private func cacheDirectory() throws -> URL {
        let baseDirectory: URL
        #if os(macOS)
        baseDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(
                for: .cachesDirectory, in: .userDomainMask
            ).first
        else {
            throw KittenTTSError.processingFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory
    }
}
