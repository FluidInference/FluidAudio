#if os(macOS)
import FluidAudio
import Foundation

/// Compare the speaker identity of two audio files using DiarizerManager's
/// 256-dim speaker embedding extractor and cosine similarity.
///
/// Usage:
///   fluidaudio speaker-similarity <a.wav> <b.wav>
///       [--threshold 0.65] [--json]
///
/// Output (default text mode):
///   distance    : 0.1234   (0 = identical, 2 = opposite)
///   similarity  : 0.8766   (1 = identical, -1 = opposite)
///   same speaker: yes      (similarity > threshold)
enum SpeakerSimilarityCommand {
    private static let logger = AppLogger(category: "SpeakerSimilarity")

    static func run(arguments: [String]) async {
        var positional: [String] = []
        var threshold: Float = 0.65
        var json = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--threshold":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    threshold = v
                    i += 1
                } else {
                    fputs("ERROR: --threshold requires a Float\n", stderr)
                    exit(1)
                }
            case "--json":
                json = true
            case "-h", "--help":
                printUsage()
                return
            default:
                positional.append(arguments[i])
            }
            i += 1
        }

        guard positional.count == 2 else {
            fputs("ERROR: speaker-similarity expects exactly 2 audio paths\n", stderr)
            printUsage()
            exit(1)
        }
        let pathA = positional[0]
        let pathB = positional[1]

        let manager: DiarizerManager
        do {
            let config = DiarizerConfig()
            manager = DiarizerManager(config: config)
            let models = try await DiarizerModels.downloadIfNeeded()
            manager.initialize(models: models)
        } catch {
            logger.error("Failed to initialize diarizer models: \(error)")
            exit(1)
        }

        let converter = AudioConverter()
        let samplesA: [Float]
        let samplesB: [Float]
        do {
            samplesA = try converter.resampleAudioFile(path: pathA)
            samplesB = try converter.resampleAudioFile(path: pathB)
        } catch {
            logger.error("Failed to load audio: \(error)")
            exit(1)
        }

        let embA: [Float]
        let embB: [Float]
        do {
            embA = try manager.extractSpeakerEmbedding(from: samplesA)
            embB = try manager.extractSpeakerEmbedding(from: samplesB)
        } catch {
            logger.error("Failed to extract speaker embedding: \(error)")
            exit(1)
        }

        let distance = SpeakerUtilities.cosineDistance(embA, embB)
        let similarity = 1.0 - distance
        let sameSpeaker = similarity > threshold

        if json {
            // Hand-roll JSON to avoid pulling in JSONSerialization for a flat 5-key dict.
            let payload =
                "{\"file_a\":\"\(escape(pathA))\","
                + "\"file_b\":\"\(escape(pathB))\","
                + "\"distance\":\(distance),"
                + "\"similarity\":\(similarity),"
                + "\"threshold\":\(threshold),"
                + "\"same_speaker\":\(sameSpeaker)}"
            print(payload)
        } else {
            print("file a      : \(pathA)")
            print("file b      : \(pathB)")
            print(String(format: "distance    : %.4f  (0 = identical, 2 = opposite)", distance))
            print(String(format: "similarity  : %.4f  (1 = identical, -1 = opposite)", similarity))
            print(String(format: "threshold   : %.4f", threshold))
            print("same speaker: \(sameSpeaker ? "yes" : "no")")
        }
    }

    private static func escape(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio speaker-similarity <a.wav> <b.wav> [options]

            Compares two audio files using a 256-dim speaker embedding and
            reports the cosine similarity / distance.

            Options:
                --threshold <float>   Decision threshold for "same speaker"
                                      (default: 0.65)
                --json                Emit a single-line JSON object
                -h, --help            Show this message
            """
        )
    }
}
#endif
