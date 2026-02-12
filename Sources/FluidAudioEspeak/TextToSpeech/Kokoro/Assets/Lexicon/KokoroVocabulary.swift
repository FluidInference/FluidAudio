import FluidAudio
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Minimal vocabulary loader for KokoroDirect backed by an actor for thread safety.
public actor KokoroVocabulary {

    public static let shared = KokoroVocabulary()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroVocabulary")
    private var vocabulary: [String: Int32] = [:]
    private var isLoaded = false

    /// Get the full vocabulary dictionary, loading it from disk (and downloading if required).
    public func getVocabulary() async throws -> [String: Int32] {
        if !isLoaded {
            try await loadVocabulary()
        }
        return vocabulary
    }

    private func loadVocabulary() async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let vocabURL = kokoroDir.appendingPathComponent("vocab_index.json")

        if !FileManager.default.fileExists(atPath: vocabURL.path) {
            logger.info("Vocabulary file not found in cache, downloading...")
            try await downloadVocabularyFile(to: cacheDir)
        }

        let data: Data
        do {
            data = try Data(contentsOf: vocabURL)
        } catch {
            logger.error("Failed to read vocabulary at \(vocabURL.path): \(error.localizedDescription)")
            throw TTSError.processingFailed("Failed to read Kokoro vocabulary: \(error.localizedDescription)")
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            logger.error("Unexpected vocabulary JSON structure")
            throw TTSError.processingFailed("Unexpected Kokoro vocabulary format")
        }

        if let vocab = json["vocab"] as? [String: Int] {
            vocabulary = vocab.mapValues { Int32($0) }
            isLoaded = true
            logger.info("Loaded \(vocabulary.count) vocabulary entries from integer map")
            return
        }

        if let vocab = json["vocab"] as? [String: Any] {
            var parsed: [String: Int32] = [:]
            parsed.reserveCapacity(vocab.count)
            for (key, value) in vocab {
                if let intValue = value as? Int {
                    parsed[key] = Int32(intValue)
                } else if let doubleValue = value as? Double {
                    parsed[key] = Int32(doubleValue)
                }
            }
            vocabulary = parsed
            isLoaded = true
            logger.info("Loaded \(vocabulary.count) vocabulary entries from generic map")
            return
        }

        logger.error("Missing 'vocab' key in vocabulary JSON")
        throw TTSError.processingFailed("Missing 'vocab' key in Kokoro vocabulary")
    }

    private func downloadVocabularyFile(to cacheDir: URL) async throws {
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let fileName = "vocab_index.json"
        let localPath = kokoroDir.appendingPathComponent(fileName)
        let remoteURL = try ModelRegistry.resolveModel(Repo.kokoro.remotePath, fileName)

        let descriptor = AssetDownloader.Descriptor(
            description: fileName,
            remoteURL: remoteURL,
            destinationURL: localPath
        )

        do {
            _ = try await AssetDownloader.ensure(descriptor, logger: logger)
        } catch {
            logger.error("Failed to download vocabulary: \(error.localizedDescription)")
            throw TTSError.downloadFailed("Failed to obtain Kokoro vocabulary: \(error.localizedDescription)")
        }
    }
}
