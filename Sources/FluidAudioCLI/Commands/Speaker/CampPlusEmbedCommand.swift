#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `campplus-embed <a.wav> [b.wav]`
/// One file → prints the 192-d embedding norm/preview; two files → cosine similarity
/// (speaker verification).
enum CampPlusEmbedCommand {
    private static let logger = AppLogger(category: "CampPlusEmbed")

    static func run(arguments: [String]) async {
        let paths = arguments.filter { !$0.hasPrefix("-") }
        guard let a = paths.first else {
            print("Usage: fluidaudio campplus-embed <a.wav> [b.wav]")
            return
        }
        do {
            let embedder = try await CampPlusEmbedder.load()
            let ea = try await embedder.embed(audioURL: URL(fileURLWithPath: a))
            if paths.count >= 2 {
                let eb = try await embedder.embed(audioURL: URL(fileURLWithPath: paths[1]))
                let cos = CampPlusEmbedder.cosine(ea, eb)
                print(String(format: "cosine = %.4f  (%@)", cos, cos >= 0.5 ? "same speaker" : "different"))
            } else {
                print("embedding: dim=\(ea.count), first 5 = \(ea.prefix(5).map { String(format: "%.3f", $0) })")
            }
        } catch {
            logger.error("CAM++ embed failed: \(error)")
        }
    }
}
#endif
