#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// Transcribe audio using the corrected Cohere cache-external pipeline,
/// with optional mixed-precision model loading (e.g. INT8 encoder + FP16
/// decoder).
enum CohereTranscribeCommand {
    private static let logger = AppLogger(category: "CohereMixed")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var encoderDir: String?
        var decoderDir: String?
        var vocabDir: String?
        var language: CohereAsrConfig.Language = .english
        var maxTokens = 108
        var repetitionPenalty: Float = 1.1
        var noRepeatNgram = 3
        var computeUnits: MLComputeUnits = .all
        var decoderVariant: CoherePipeline.DecoderVariant = .v2

        var i = 1
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--encoder-dir":
                if i + 1 < arguments.count {
                    encoderDir = arguments[i + 1]
                    i += 1
                }
            case "--decoder-dir":
                if i + 1 < arguments.count {
                    decoderDir = arguments[i + 1]
                    i += 1
                }
            case "--vocab-dir":
                if i + 1 < arguments.count {
                    vocabDir = arguments[i + 1]
                    i += 1
                }
            case "--language", "-l":
                if i + 1 < arguments.count,
                    let lang = CohereAsrConfig.Language(rawValue: arguments[i + 1].lowercased())
                {
                    language = lang
                    i += 1
                } else {
                    logger.error("Unknown language '\(arguments[safe: i + 1] ?? "<nil>")'")
                    exit(1)
                }
            case "--max-tokens":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    maxTokens = v
                    i += 1
                }
            case "--repetition-penalty":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    repetitionPenalty = v
                    i += 1
                }
            case "--no-repeat-ngram":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    noRepeatNgram = v
                    i += 1
                }
            case "--cpu-only":
                computeUnits = .cpuOnly
            case "--cpu-gpu":
                computeUnits = .cpuAndGPU
            case "--decoder-variant":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v2": decoderVariant = .v2
                    case "v1": decoderVariant = .v1
                    default:
                        logger.error("Unknown decoder variant '\(arguments[i + 1])' (expected v1 or v2)")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Ignoring unknown option: \(arg)")
            }
            i += 1
        }

        // Resolve encoder/decoder/vocab directories: explicit flags win,
        // otherwise all default to --model-dir.
        let encDir = encoderDir ?? modelDir
        let decDir = decoderDir ?? modelDir
        let vocDir = vocabDir ?? modelDir ?? decoderDir ?? encoderDir
        guard let encDir = encDir, let decDir = decDir, let vocDir = vocDir else {
            logger.error(
                "Need --model-dir, or --encoder-dir + --decoder-dir (+ optional --vocab-dir)")
            printUsage()
            exit(1)
        }

        await transcribe(
            audioFile: audioFile,
            encoderDir: URL(fileURLWithPath: encDir),
            decoderDir: URL(fileURLWithPath: decDir),
            vocabDir: URL(fileURLWithPath: vocDir),
            language: language,
            maxTokens: maxTokens,
            repetitionPenalty: repetitionPenalty,
            noRepeatNgram: noRepeatNgram,
            computeUnits: computeUnits,
            decoderVariant: decoderVariant
        )
    }

    private static func transcribe(
        audioFile: String,
        encoderDir: URL,
        decoderDir: URL,
        vocabDir: URL,
        language: CohereAsrConfig.Language,
        maxTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        computeUnits: MLComputeUnits,
        decoderVariant: CoherePipeline.DecoderVariant
    ) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Cohere mixed pipeline requires macOS 14 or later")
            return
        }

        do {
            logger.info("Loading audio: \(audioFile)")
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)
            logger.info(
                "Audio duration: \(String(format: "%.2f", duration))s (\(samples.count) samples)")

            logger.info("Encoder dir:  \(encoderDir.path)")
            logger.info("Decoder dir:  \(decoderDir.path)")
            logger.info("Vocab dir:    \(vocabDir.path)")
            logger.info("Language:     \(language.englishName) (\(language.rawValue))")

            logger.info("Decoder:      \(decoderVariant)")

            let loadStart = CFAbsoluteTimeGetCurrent()
            let models = try await CoherePipeline.loadModels(
                encoderDir: encoderDir,
                decoderDir: decoderDir,
                vocabDir: vocabDir,
                decoderVariant: decoderVariant,
                computeUnits: computeUnits)
            let loadSecs = CFAbsoluteTimeGetCurrent() - loadStart
            logger.info("Models loaded in \(String(format: "%.2f", loadSecs))s")

            let pipeline = CoherePipeline()
            let result = try await pipeline.transcribe(
                audio: samples,
                models: models,
                language: language,
                maxNewTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                noRepeatNgram: noRepeatNgram)

            let rtfx = duration / max(result.totalSeconds, 1e-9)

            logger.info(String(repeating: "=", count: 60))
            logger.info("COHERE TRANSCRIBE (fixed pipeline)")
            logger.info(String(repeating: "=", count: 60))
            print(result.text)
            logger.info("")
            logger.info("Tokens generated: \(result.tokenIds.count)")
            logger.info("Encoder time:  \(String(format: "%.3f", result.encoderSeconds))s")
            logger.info("Decoder time:  \(String(format: "%.3f", result.decoderSeconds))s")
            logger.info("Total time:    \(String(format: "%.3f", result.totalSeconds))s")
            logger.info("RTFx:          \(String(format: "%.2f", rtfx))x")
        } catch {
            logger.error("cohere-transcribe failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe — fixed cache-external pipeline (mixed precision supported)

            Usage: fluidaudio cohere-transcribe <audio_file> [options]

            Model locations (choose one pattern):
                --model-dir <path>              Single dir with encoder + decoder + vocab.json
                --encoder-dir <path>            (mixed) INT8 or FP16 encoder .mlmodelc dir
                --decoder-dir <path>            (mixed) INT8 or FP16 decoder .mlmodelc dir
                --vocab-dir <path>              Directory containing vocab.json (defaults to decoder-dir)

            Expected files inside each dir:
                cohere_encoder.mlmodelc
                cohere_decoder_cache_external_v2.mlmodelc  (v2 — ANE-resident static shapes)
                cohere_decoder_cache_external.mlmodelc     (v1 — FP16 dynamic RangeDim)
                vocab.json  (in --vocab-dir / --model-dir)

            Decode options:
                --language, -l <code>           en / fr / de / es / it / pt / nl / pl / el /
                                                 ar / ja / zh / ko / vi  (default: en)
                --max-tokens <n>                Max decoded tokens (default: 108)
                --repetition-penalty <f>        CTRL-style penalty, 1.0 disables (default: 1.1)
                --no-repeat-ngram <n>           Forbid repeating n-grams, 0 disables (default: 3)
                --decoder-variant <v1|v2>       Decoder to load (default: v2)

            Compute units:
                --cpu-only                      Force CPU
                --cpu-gpu                       CPU + GPU (skip ANE)
                (default: all, includes ANE)

            Example (INT8 encoder + FP16 decoder):
                fluidaudio cohere-transcribe audio.wav \\
                    --encoder-dir /path/to/q8 \\
                    --decoder-dir /path/to/f16 \\
                    --vocab-dir  /path/to/f16 \\
                    --language en
            """
        )
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
#endif
