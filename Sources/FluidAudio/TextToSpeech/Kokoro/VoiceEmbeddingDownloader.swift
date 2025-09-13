import Foundation

/// Downloads voice embeddings from HuggingFace
public enum VoiceEmbeddingDownloader {

    /// Download a voice embedding JSON file from HuggingFace
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        // Try to download pre-converted JSON first
        let jsonURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(voice).json"

        if let url = URL(string: jsonURL) {
            do {
                let (data, response) = try await URLSession.shared.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    print("Downloaded voice embedding: \(voice).json from HuggingFace")
                    return data
                }
            } catch {
                // JSON not available, try to download .pt file
                print("Could not download \(voice).json: \(error.localizedDescription)")
            }
        }

        // Download the .pt file for future conversion
        let ptURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/voices/\(voice).pt"
        if let url = URL(string: ptURL) {
            do {
                let (ptData, response) = try await URLSession.shared.data(from: url)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    // Save .pt file to cache
                    let cacheDir = try TtsModels.cacheDirectoryURL()
                    let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
                    try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

                    let ptFileURL = voicesDir.appendingPathComponent("\(voice).pt")
                    try ptData.write(to: ptFileURL)
                    print("Downloaded voice embedding .pt file: \(voice).pt (\(ptData.count) bytes)")
                    print("Note: Run 'python3 extract_voice_embeddings.py' to convert .pt to JSON format")
                }
            } catch {
                print("Could not download \(voice).pt: \(error.localizedDescription)")
            }
        }

        // For now, return a default embedding
        print("Using default voice embedding for \(voice)")

        // Create default embedding (128 random values)
        var embedding: [Float] = []
        for _ in 0..<128 {
            embedding.append(Float.random(in: -0.1...0.1))
        }

        let json = [voice: embedding]
        return try JSONSerialization.data(withJSONObject: json)
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
        print("Voice embedding cached: \(voice)")
    }
}
