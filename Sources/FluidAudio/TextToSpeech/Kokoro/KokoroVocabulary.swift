import Foundation
import OSLog

/// Minimal vocabulary loader for KokoroDirect
@available(macOS 13.0, *)
public struct KokoroVocabulary {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroVocabulary")
    private static var vocabulary: [String: Int32] = [:]
    private static var isLoaded = false

    /// Load vocabulary from kokoro_correct_vocab.json
    public static func loadVocabulary() {
        guard !isLoaded else { return }

        // Use Models/kokoro subdirectory
        let cacheDir: URL
        do {
            cacheDir = try TtsModels.cacheDirectoryURL()
        } catch {
            logger.error("Failed to get cache directory: \(error)")
            fatalError("Failed to get cache directory: \(error)")
        }

        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let vocabURL = kokoroDir.appendingPathComponent("vocab_index.json")

        // Download if missing
        if !FileManager.default.fileExists(atPath: vocabURL.path) {
            logger.info("Vocabulary file not found in cache, downloading...")
            do {
                try downloadVocabularyFile(to: cacheDir)
            } catch {
                logger.error("Failed to download vocabulary: \(error)")
                fatalError("Failed to download vocabulary: \(error)")
            }
        }

        do {
            let data = try Data(contentsOf: vocabURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // The vocab can be either a dict or other structure
                if let vocab = json["vocab"] as? [String: Int] {
                    // Dict format: phoneme -> id
                    vocabulary = vocab.mapValues { Int32($0) }
                    isLoaded = true
                    logger.info("Loaded \(vocabulary.count) vocabulary entries from dict")
                } else if let vocab = json["vocab"] as? [String: Any] {
                    // Try to parse as String->Any then convert
                    var parsedVocab: [String: Int32] = [:]
                    for (key, value) in vocab {
                        if let intValue = value as? Int {
                            parsedVocab[key] = Int32(intValue)
                        } else if let doubleValue = value as? Double {
                            parsedVocab[key] = Int32(doubleValue)
                        }
                    }
                    vocabulary = parsedVocab
                    isLoaded = true
                    logger.info("Loaded \(vocabulary.count) vocabulary entries from Any dict")
                } else {
                    logger.error("Unexpected vocab format in vocab_index.json")
                }
            }
        } catch {
            logger.error("Failed to load vocabulary: \(error)")
            fatalError("Failed to load vocabulary from vocab_index.json: \(error)")
        }
    }

    /// Get the full vocabulary dictionary
    public static func getVocabulary() -> [String: Int32] {
        loadVocabulary()
        return vocabulary
    }

    /// Download vocabulary file if missing
    private static func downloadVocabularyFile(to cacheDir: URL) throws {
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create directory if needed
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let baseURL = "https://huggingface.co/\(DownloadUtils.Repo.kokoro.rawValue)/resolve/main"
        let fileName = "vocab_index.json"
        let localPath = kokoroDir.appendingPathComponent(fileName)

        if !FileManager.default.fileExists(atPath: localPath.path) {
            guard let remoteURL = URL(string: "\(baseURL)/\(fileName)") else {
                throw TTSError.downloadFailed("Invalid vocab URL base: \(baseURL)")
            }
            logger.info("Downloading \(fileName)...")
            let (data, response) = try awaitURLData(remoteURL)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw TTSError.downloadFailed("Failed to download \(fileName)")
            }
            try data.write(to: localPath)
            logger.info("Downloaded \(fileName) to cache")
        }
    }

}

@available(macOS 13.0, *)
private func awaitURLData(_ url: URL) throws -> (Data, URLResponse) {
    let semaphore = DispatchSemaphore(value: 0)
    var outData: Data?
    var outResp: URLResponse?
    var outErr: Error?
    let task = URLSession.shared.dataTask(with: url) { data, resp, err in
        outData = data
        outResp = resp
        outErr = err
        semaphore.signal()
    }
    task.resume()
    semaphore.wait()
    if let e = outErr { throw e }
    return (outData ?? Data(), outResp ?? URLResponse())
}
