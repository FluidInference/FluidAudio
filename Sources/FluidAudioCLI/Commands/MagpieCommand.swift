#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// CLI surface for the Magpie TTS Multilingual Swift port.
///
/// Subcommands:
///   - `download`             Fetch models + constants + tokenizer data from HuggingFace.
///   - `text`                 Synthesize text → WAV.
public enum MagpieCommand {

    private static let logger = AppLogger(category: "MagpieCommand")

    public static func run(arguments: [String]) async {
        guard let sub = arguments.first else {
            printUsage()
            return
        }
        let rest = Array(arguments.dropFirst())
        switch sub {
        case "download":
            await runDownload(arguments: rest)
        case "text":
            await runText(arguments: rest)
        case "help", "--help", "-h":
            printUsage()
        default:
            logger.error("Unknown magpie subcommand: \(sub)")
            printUsage()
            exit(1)
        }
    }

    // MARK: - download

    private static func runDownload(arguments: [String]) async {
        var languageCodes: [String] = ["en"]
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--languages" || arg == "-l", i + 1 < arguments.count {
                languageCodes = arguments[i + 1].split(separator: ",").map(String.init)
                i += 1
            }
            i += 1
        }
        let langs: Set<MagpieLanguage> = Set(languageCodes.compactMap { MagpieLanguage(rawValue: $0) })
        if langs.isEmpty {
            logger.error("No valid language codes provided")
            exit(1)
        }
        do {
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: langs)
            logger.info("Magpie assets ready at: \(repoDir.path)")
        } catch {
            logger.error("Magpie download failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - text

    private static func runText(arguments: [String]) async {
        var text: String? = nil
        var output = "magpie.wav"
        var speakerIdx = MagpieSpeaker.john.rawValue
        var languageCode = "en"
        var cfg: Float = MagpieConstants.defaultCfgScale
        var topK = MagpieConstants.defaultTopK
        var temperature = MagpieConstants.defaultTemperature
        var seed: UInt64? = nil
        var allowIpa = true

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let idx = Int(arguments[i + 1]) {
                    speakerIdx = idx
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--cfg":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    cfg = v
                    i += 1
                }
            case "--topk":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    topK = v
                    i += 1
                }
            case "--temperature":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    temperature = v
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--no-ipa-override":
                allowIpa = false
            default:
                if text == nil { text = arg }
            }
            i += 1
        }

        guard let text = text, !text.isEmpty else {
            logger.error("Missing text argument")
            printUsage()
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx); valid range 0..<\(MagpieConstants.numSpeakers)")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }

        do {
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let opts = MagpieSynthesisOptions(
                temperature: temperature,
                topK: topK,
                maxSteps: MagpieConstants.maxSteps,
                minFrames: MagpieConstants.minFrames,
                cfgScale: cfg,
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: allowIpa)
            let start = Date()
            let result = try await manager.synthesize(
                text: text, speaker: speaker, language: language, options: opts)
            let elapsed = Date().timeIntervalSince(start)

            let wav = try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate))
            let outURL = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try wav.write(to: outURL)

            let audioSecs = result.durationSeconds
            let rtfx = elapsed > 0 ? audioSecs / elapsed : 0
            logger.info("Magpie synthesis complete")
            logger.info("  Speaker: \(speaker.displayName), Language: \(language.rawValue)")
            logger.info("  Codes: \(result.codeCount), EOS: \(result.finishedOnEos)")
            logger.info(
                "  Audio: \(String(format: "%.3f", audioSecs))s, Synthesis: \(String(format: "%.3f", elapsed))s, RTFx: \(String(format: "%.2f", rtfx))x"
            )
            logger.info("  Output: \(outURL.path)")
        } catch {
            logger.error("Magpie synthesis failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - usage

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio magpie <subcommand> [options]

            Subcommands:
              download                Download Magpie models + constants + tokenizers
                --languages en,es,de    Comma-separated language codes (default: en)

              text "<text>"           Synthesize text and write a WAV file
                --output, -o PATH       Output WAV path (default: magpie.wav)
                --speaker N             Speaker index 0-4 (default: 0 = John)
                --language CODE         Language code (en, es, de, fr, it, vi, zh, hi)
                --cfg FLOAT             CFG guidance scale (default: 1.0 = off)
                --topk N                Top-K sampling (default: 80)
                --temperature FLOAT     Sampling temperature (default: 0.6)
                --seed N                Deterministic RNG seed
                --no-ipa-override       Disable `|…|` IPA pass-through

            IPA override example:
              fluidaudio magpie text "Hello | ˈ n ɛ m o ʊ | Text." --output demo.wav

            """
        )
    }
}
#endif
