import Foundation

/// Downloads voice embeddings from HuggingFace
public enum VoiceEmbeddingDownloader {

    private static let logger = AppLogger(category: "VoiceEmbeddingDownloader")

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
}
