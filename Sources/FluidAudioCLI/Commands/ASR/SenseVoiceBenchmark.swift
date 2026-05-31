#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `sensevoice-benchmark [--languages en_us,cmn_hans_cn] [--samples 100] [--fp32]`
///
/// FLEURS WER/CER for SenseVoiceSmall on Apple Silicon. Reuses the shared
/// `FLEURSBenchmark` downloader + `WERCalculator`. CoreML(ANE) vs the published
/// numbers; cross-checks the conversion is accuracy-neutral.
enum SenseVoiceBenchmark {
    private static let logger = AppLogger(category: "SenseVoiceBenchmark")

    static func run(arguments: [String]) async {
        var languages = ["en_us", "cmn_hans_cn"]
        var samplesPerLanguage = 100
        var precision: SenseVoiceEncoderPrecision = .fp16
        var verbose = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--languages":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1].split(separator: ",").map(String.init)
                    i += 1
                }
            case "--samples", "-n":
                if i + 1 < arguments.count {
                    samplesPerLanguage =
                        arguments[i + 1].lowercased() == "all" ? Int.max : (Int(arguments[i + 1]) ?? 100)
                    i += 1
                }
            case "--int8":
                precision = .int8
            case "--fp32":
                precision = .fp32
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                print(
                    "Usage: fluidaudio sensevoice-benchmark [--languages en_us,cmn_hans_cn] [--samples N|all] [--int8|--fp32]"
                )
                return
            default:
                break
            }
            i += 1
        }

        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/FLEURS").path

        do {
            logger.info("Loading SenseVoice (encoder: \(precision.rawValue))...")
            let manager = try await SenseVoiceManager.load(precision: precision)

            let fleurs = FLEURSBenchmark(
                config: .init(
                    languages: languages, samplesPerLanguage: samplesPerLanguage,
                    outputFile: "", cacheDir: cacheDir, debugMode: verbose))
            logger.info("Downloading FLEURS (\(languages.joined(separator: ", ")))...")
            try await fleurs.downloadFLEURS(languages: languages)
            let allSamples = try fleurs.loadFLEURSSamples(languages: languages)

            let converter = AudioConverter(sampleRate: 16_000)
            for language in languages {
                let samples = allSamples.filter { $0.language == language }.prefix(samplesPerLanguage)
                var wordErrors = 0
                var totalWords = 0
                var charErrors = 0.0
                var totalChars = 0
                var audioSec = 0.0
                var procSec = 0.0
                var n = 0
                for sample in samples {
                    guard let audio = try? converter.resampleAudioFile(path: sample.audioPath) else { continue }
                    let t0 = Date()
                    guard let hyp = try? await manager.transcribe(audio: audio) else { continue }
                    procSec += Date().timeIntervalSince(t0)
                    audioSec += Double(audio.count) / 16_000.0
                    let m = WERCalculator.calculateWERAndCER(hypothesis: hyp, reference: sample.transcription)
                    wordErrors += m.insertions + m.deletions + m.substitutions
                    totalWords += m.totalWords
                    charErrors += m.cer * Double(m.totalCharacters)
                    totalChars += m.totalCharacters
                    n += 1
                }
                let wer = totalWords > 0 ? 100.0 * Double(wordErrors) / Double(totalWords) : 0
                let cer = totalChars > 0 ? 100.0 * charErrors / Double(totalChars) : 0
                let rtfx = procSec > 0 ? audioSec / procSec : 0
                logger.info(
                    "[\(language)] n=\(n)  WER=\(String(format: "%.2f", wer))%  CER=\(String(format: "%.2f", cer))%  RTFx=\(String(format: "%.0f", rtfx))"
                )
            }
        } catch {
            logger.error("Benchmark failed: \(error)")
        }
    }
}
#endif
