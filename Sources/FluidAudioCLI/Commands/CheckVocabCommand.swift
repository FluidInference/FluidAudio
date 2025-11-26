import Foundation
import FluidAudio

enum CheckVocabCommand {
    static func run(arguments: [String]) async {
        do {
            // Ensure assets are downloaded/loaded
            let _ = try await TtsResourceDownloader.ensureZhAssetsInCache()
            let vocab = try await KokoroVocabulary.shared.getVocabulary()
            
            print("Vocabulary size: \(vocab.count)")
            
            let tokensToCheck = ["↓", "↗", "→", "↘"]
            
            print("--- Token Check ---")
            for t in tokensToCheck {
                if let id = vocab[t] {
                    print("'\(t)': \(id)")
                    for s in t.unicodeScalars {
                        print("  Unicode: \(s.value)")
                    }
                } else {
                    print("'\(t)': NOT FOUND")
                }
            }
            
            // Print some random tokens to see the style
            print("\n--- Sample Tokens ---")
            for (k, _) in vocab.prefix(20) {
                print(k)
            }
            
        } catch {
            print("Error: \(error)")
        }
    }
}
