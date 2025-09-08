import Foundation
import OSLog

/// Minimal vocabulary loader for KokoroDirectTTS
@available(macOS 13.0, *)
public struct KokoroVocabulary {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroVocabulary")
    private static var vocabulary: [String: Int32] = [:]
    private static var isLoaded = false

    /// Load vocabulary from kokoro_correct_vocab.json
    public static func loadVocabulary() {
        guard !isLoaded else { return }

        let currentDir = FileManager.default.currentDirectoryPath
        let vocabURL = URL(fileURLWithPath: currentDir).appendingPathComponent("vocab_index.json")

        do {
            let data = try Data(contentsOf: vocabURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let vocab = json["vocab"] as? [String: Int]
            {
                vocabulary = vocab.mapValues { Int32($0) }
                isLoaded = true
                logger.info("Loaded \(vocabulary.count) vocabulary entries")
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

}
