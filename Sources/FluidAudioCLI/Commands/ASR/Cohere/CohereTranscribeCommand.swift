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
        var maxTokens: Int?
        var repetitionPenalty: Float = 1.1
        var noRepeatNgram = 3
        var computeUnits = CohereComputeUnitsOption()
        var decoderVariant: CoherePipeline.DecoderVariant = .v3
        var repeatCount = 1

        var i = 1
        while i < arguments.count {
            let arg = arguments[i]
            if computeUnits.consume(arguments, at: &i, logger: logger) {
                i += 1
                continue
            }
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
            case "--repeat":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    repeatCount = max(1, v)
                    i += 1
                }
            case "--decoder-variant":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v3": decoderVariant = .v3
                    case "v2": decoderVariant = .v2
                    case "v1": decoderVariant = .v1
                    default:
                        logger.error(
                            "Unknown decoder variant '\(arguments[i + 1])' (expected v1, v2 or v3)")
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
            decoderVariant: decoderVariant,
            repeatCount: repeatCount
        )
    }

    private static func transcribe(
        audioFile: String,
        encoderDir: URL,
        decoderDir: URL,
        vocabDir: URL,
        language: CohereAsrConfig.Language,
        maxTokens: Int?,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        computeUnits: CohereComputeUnitsOption,
        decoderVariant: CoherePipeline.DecoderVariant,
        repeatCount: Int
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
                computeUnits: computeUnits.shared,
                encoderComputeUnits: computeUnits.encoder,
                decoderComputeUnits: computeUnits.decoder)
            let loadSecs = CFAbsoluteTimeGetCurrent() - loadStart
            logger.info("Models loaded in \(String(format: "%.2f", loadSecs))s")
            logger.info(
                "Encoder compute: \(CoherePipeline.describe(models.encoderComputeUnits)), "
                    + "decoder compute: \(CoherePipeline.describe(models.decoderComputeUnits))")
            logFootprint("after load")

            let pipeline = CoherePipeline()
            var result: CoherePipeline.TranscriptionResult?
            // `--repeat n` re-runs the same audio so warm per-call timings and
            // the steady-state footprint are visible after the first-call
            // compile (#1045). The report below covers the last run.
            for run in 1...repeatCount {
                let r = try await pipeline.transcribeLong(
                    audio: samples,
                    models: models,
                    language: language,
                    maxNewTokens: maxTokens,
                    repetitionPenalty: repetitionPenalty,
                    noRepeatNgram: noRepeatNgram)
                if repeatCount > 1 {
                    logger.info(
                        "Run \(run)/\(repeatCount): encoder \(String(format: "%.3f", r.encoderSeconds))s, "
                            + "decoder \(String(format: "%.3f", r.decoderSeconds))s")
                    logFootprint("after run \(run)")
                }
                result = r
            }
            guard let result else { return }

            let rtfx = duration / max(result.totalSeconds, 1e-9)

            logger.info(String(repeating: "=", count: 60))
            logger.info("COHERE TRANSCRIBE (fixed pipeline)")
            logger.info(String(repeating: "=", count: 60))
            print(result.text)
            logger.info("")
            logger.info("Tokens generated: \(result.tokenIds.count)")
            logger.info("Decoder cache:    \(models.cacheLength) slots")
            if result.cappedWindows > 0 {
                logger.warning(
                    "Windows that hit the decode cap: \(result.cappedWindows) (speech dropped)")
            }
            logger.info("Encoder time:  \(String(format: "%.3f", result.encoderSeconds))s")
            logger.info("Decoder time:  \(String(format: "%.3f", result.decoderSeconds))s")
            logger.info("Total time:    \(String(format: "%.3f", result.totalSeconds))s")
            logger.info("RTFx:          \(String(format: "%.2f", rtfx))x")
            logFootprint("after transcribe")
        } catch {
            logger.error("cohere-transcribe failed: \(error)")
            exit(1)
        }
    }

    /// Physical footprint (what Xcode's memory gauge and Activity Monitor
    /// report), so the memory cost of a compute-unit choice is visible from
    /// the CLI (#1045).
    private static func logFootprint(_ stage: String) {
        guard let bytes = SystemInfo.currentPhysicalFootprintBytes(),
            let peak = SystemInfo.peakPhysicalFootprintBytes()
        else { return }
        logger.info("Footprint \(stage): \(bytes / 1_048_576) MB (peak \(peak / 1_048_576) MB)")
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
                cohere_decoder_cache_external_v3.mlmodelc  (v3 — static shapes, 256-slot cache)
                cohere_decoder_cache_external_v2.mlmodelc  (v2 — static shapes, 108-slot cache)
                cohere_decoder_cache_external.mlmodelc     (v1 — FP16 dynamic RangeDim)
                vocab.json  (in --vocab-dir / --model-dir)

            Decode options:
                --language, -l <code>           en / fr / de / es / it / pt / nl / pl / el /
                                                 ar / ja / zh / ko / vi  (default: en)
                --max-tokens <n>                Max decoded tokens per 35 s window
                                                (default: all the cache holds — 99 v2, 247 v3)
                --repetition-penalty <f>        CTRL-style penalty, 1.0 disables (default: 1.1)
                --no-repeat-ngram <n>           Forbid repeating n-grams, 0 disables (default: 3)
                --decoder-variant <v1|v2|v3>    Decoder to load (default: v3)
                --repeat <n>                    Transcribe the file n times in one process and
                                                log per-run timings + footprint (default: 1)

            \(CohereComputeUnitsOption.usage)

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
