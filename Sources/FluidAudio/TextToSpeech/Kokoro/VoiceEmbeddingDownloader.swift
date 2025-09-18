import Foundation

/// Downloads voice embeddings from HuggingFace
@available(macOS 13.0, iOS 16.0, *)
public enum VoiceEmbeddingDownloader {

    /// Download a voice embedding JSON file from HuggingFace.
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        let canonicalVoice = KokoroVoiceCatalog.canonicalVoiceId(for: voice) ?? voice

        // Try to download pre-converted JSON first
        let jsonURL =
            "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(canonicalVoice).json"

        if let url = URL(string: jsonURL) {
            do {
                // Use DownloadUtils.sharedSession for consistent proxy and configuration handling
                let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    print("Downloaded voice embedding: \(canonicalVoice).json from HuggingFace")
                    return data
                }
            } catch {
                print("Could not download \(canonicalVoice).json: \(error.localizedDescription)")
            }
        }

        // Download the .pt file for future conversion, but still signal failure for immediate use
        let ptURL =
            "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(canonicalVoice).pt"
        if let url = URL(string: ptURL) {
            do {
                // Use DownloadUtils.sharedSession for consistent proxy and configuration handling
                let (ptData, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    let cacheDir = try TtsModels.cacheDirectoryURL()
                    let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
                    try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

                    let ptFileURL = voicesDir.appendingPathComponent("\(canonicalVoice).pt")
                    try ptData.write(to: ptFileURL)
                    print("Downloaded voice embedding .pt file: \(canonicalVoice).pt (\(ptData.count) bytes)")
                    print("Note: Run 'python3 extract_voice_embeddings.py' to convert .pt to JSON format")
                }
            } catch {
                print("Could not download \(canonicalVoice).pt: \(error.localizedDescription)")
            }
        }

        throw TTSError.modelNotFound("Voice embedding for \(canonicalVoice)")
    }

    /// Ensure a voice embedding is available in cache
    public static func ensureVoiceEmbedding(
        voice: String = KokoroVoiceCatalog.defaultVoiceId
    ) async throws {
        let canonicalVoice = KokoroVoiceCatalog.canonicalVoiceId(for: voice) ?? voice
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")

        // Create directory if needed
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let jsonFile = "\(canonicalVoice).json"
        let jsonURL = voicesDir.appendingPathComponent(jsonFile)

        // Skip if already cached
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            return
        }

        let data = try await downloadVoiceEmbedding(voice: canonicalVoice)
        try data.write(to: jsonURL)
        print("Voice embedding cached: \(canonicalVoice)")
    }
}
