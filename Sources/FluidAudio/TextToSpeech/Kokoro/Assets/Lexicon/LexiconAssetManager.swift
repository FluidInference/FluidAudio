import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
actor LexiconAssetManager {
    static let shared = LexiconAssetManager()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "LexiconAssetManager")
    private let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"

    private func downloadFileIfNeeded(filename: String, urlPath: String) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)
        guard !FileManager.default.fileExists(atPath: localURL.path) else {
            return
        }

        logger.info("Downloading \(filename)...")
        guard let downloadURL = URL(string: "\(baseURL)/\(urlPath)") else {
            throw TTSError.modelNotFound("Invalid URL for \(filename)")
        }

        let (data, response) = try await DownloadUtils.sharedSession.data(from: downloadURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw TTSError.modelNotFound("Failed to download \(filename)")
        }

        try data.write(to: localURL)
        logger.info("Downloaded \(filename) (\(data.count) bytes)")
    }

    private func ensureLexiconFiles() async throws {
        try await downloadFileIfNeeded(filename: "us_gold.json", urlPath: "us_gold.json")
        try await downloadFileIfNeeded(filename: "us_silver.json", urlPath: "us_silver.json")
    }

    private func ensureLexiconCache() async {
        do {
            try await downloadFileIfNeeded(filename: "us_lexicon_cache.json", urlPath: "us_lexicon_cache.json")
        } catch {
            logger.warning("Failed to download lexicon cache (will fall back to merge): \(error.localizedDescription)")
        }
    }

    private func ensureEspeakAssets() async {
        let cacheDir = try? TtsModels.cacheDirectoryURL()
        guard let cacheDir else { return }
        let modelsDirectory = cacheDir.appendingPathComponent("Models")
        _ = try? await DownloadUtils.ensureEspeakDataBundle(in: modelsDirectory)
    }

    func ensureCoreAssets() async throws {
        await ensureLexiconCache()
        try await ensureLexiconFiles()
        await ensureEspeakAssets()
    }

    static func ensureCoreAssets() async throws {
        try await LexiconAssetManager.shared.ensureCoreAssets()
    }
}
