import Foundation

/// Kokoro TTS resource downloader (voice embeddings, eSpeak data)
public enum TtsResourceDownloader {

    private static let logger = AppLogger(category: "TtsResourceDownloader")

    /// Download a voice embedding JSON file from HuggingFace
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        // Try to download pre-converted JSON first
        let jsonURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(voice).json"

        if let url = URL(string: jsonURL) {
            do {
                // Use DownloadUtils.sharedSession for consistent proxy and configuration handling
                let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    logger.info("Downloaded voice embedding JSON for \(voice)")
                    return data
                }
            } catch {
                // JSON not available, try to download .pt file
                logger.warning("Could not download \(voice).json: \(error.localizedDescription)")
            }
        }

        var downloadedPtPath: String?

        // Download the .pt file for future conversion
        let ptURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(voice).pt"
        if let url = URL(string: ptURL) {
            do {
                // Use DownloadUtils.sharedSession for consistent proxy and configuration handling
                let (ptData, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    // Save .pt file to cache
                    let cacheDir = try TtsModels.cacheDirectoryURL()
                    let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
                    try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

                    let ptFileURL = voicesDir.appendingPathComponent("\(voice).pt")
                    try ptData.write(to: ptFileURL)
                    downloadedPtPath = ptFileURL.path
                    logger.info(
                        "Downloaded voice embedding .pt file for \(voice) (\(ptData.count) bytes)")
                    logger.notice(
                        "Run 'python3 extract_voice_embeddings.py' to convert \(voice).pt to JSON format"
                    )
                }
            } catch {
                logger.warning("Could not download \(voice).pt: \(error.localizedDescription)")
            }
        }

        if let path = downloadedPtPath {
            throw TTSError.processingFailed(
                "Voice embedding JSON unavailable for \(voice). Downloaded .pt to \(path); run 'python3 extract_voice_embeddings.py' to convert it."
            )
        }

        throw TTSError.modelNotFound("Voice embedding JSON for \(voice)")
    }

    /// Ensure a voice embedding is available in cache
    public static func ensureVoiceEmbedding(voice: String = "af_heart") async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")

        // Create directory if needed
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let jsonFile = "\(voice).json"
        let jsonURL = voicesDir.appendingPathComponent(jsonFile)

        // Skip if already cached
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            return
        }

        // Try to download
        let data = try await downloadVoiceEmbedding(voice: voice)
        try data.write(to: jsonURL)
        logger.info("Voice embedding cached: \(voice)")
    }

    /// Ensure the eSpeak NG data bundle is available locally for Kokoro G2P
    @available(macOS 13.0, iOS 16.0, *)
    public static func ensureEspeakDataBundle(in modelsDirectory: URL) async throws -> URL {
        let repo = Repo.kokoro
        let repoPath = modelsDirectory.appendingPathComponent(repo.folderName)
        let bundleRoot = repoPath.appendingPathComponent(
            "Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data"
        )
        let voices = bundleRoot.appendingPathComponent("voices")

        if FileManager.default.fileExists(atPath: voices.path) {
            return bundleRoot
        }

        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        logger.info("Downloading eSpeak NG data bundle from HuggingFace…")

        let zipPath = repoPath.appendingPathComponent("espeak-ng.zip")
        let zipURL = URL(string: "https://huggingface.co/\(repo.remotePath)/resolve/main/espeak-ng.zip")!

        if !FileManager.default.fileExists(atPath: zipPath.path) {
            try FileManager.default.createDirectory(
                at: zipPath.deletingLastPathComponent(), withIntermediateDirectories: true)

            let (tempURL, _) = try await DownloadUtils.sharedSession.download(from: zipURL)
            try FileManager.default.moveItem(at: tempURL, to: zipPath)
            logger.info("Downloaded espeak-ng-data.zip")
        }

        #if os(macOS)
        let resourcesDir = repoPath.appendingPathComponent("Resources")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", zipPath.path, "-d", resourcesDir.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()

        if process.terminationStatus == 0 {
            logger.info("Extracted espeak-ng-data successfully")
        }
        #else
        logger.warning(
            "Skipping espeak-ng.zip extraction on this platform; expecting pre-extracted Resources bundle"
        )
        #endif

        guard FileManager.default.fileExists(atPath: voices.path) else {
            throw TTSError.downloadFailed("eSpeak NG data bundle missing 'voices' after extraction")
        }
        return bundleRoot
    }
}
