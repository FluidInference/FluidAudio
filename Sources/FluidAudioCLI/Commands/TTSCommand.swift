import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    private static let logger = AppLogger(category: "TTSCommand")
    private static let benchmarkSentences: [String] = [
        "Quick check to measure short output speed.",
        "The new release pipeline needs reliable voice synthesis benchmarks "
            + "to track regressions in latency and throughput across updates.",
        "I can't believe we finally made it to the summit after climbing for twelve exhausting hours "
            + "through wind and rain, but wow, this view of the endless mountain ranges stretching to the horizon "
            + "makes every single difficult step completely worth the journey.",
        "Benchmarking medium-length sentences helps reveal how the system balances clarity with speed.",
        "Some users only ever generate brief prompts, while others expect multi-paragraph narrations for reports.",
        "Latency tends to spike when processing punctuation-heavy text, so this sentence includes commas, semicolons, and—of course—dashes.",
        "During real-world use, people may speak in long, meandering ways that stretch the models ability to sustain natural cadence and intonation over dozens of words, testing both quality and throughput.",
        "Short.",
        "In the midst of testing how synthetic speech systems perform under stress, we decided to craft an especially long passage that meanders through several interconnected themes—starting with the simple observation that voice interfaces have become part of everyday life, moving into a reflection on how early text-to-speech systems were criticized for sounding robotic and unnatural, drifting further into technical details about neural vocoders, attention mechanisms, and latency bottlenecks in hardware pipelines, and then circling back to the human element: the way people perceive rhythm, tone, and emotion in spoken language, which makes evaluation of generated audio far more complex than measuring raw throughput or accuracy, because speech is not only a vehicle for information but also an instrument of connection, persuasion, and empathy; so when a benchmark sentence grows this long, with commas and semicolons and digressions that twist and turn like winding mountain roads, it becomes an excellent test of whether the synthesizer can maintain not just intelligibility but also coherence, flow, and a sense of natural cadence across dozens and dozens of words without faltering, stuttering, or flattening into monotony.",
        "After hours of careful preparation, countless revisions to the experiment setup, and no shortage of nervous anticipation, the team finally gathered around the workstation to watch the synthesizer process an unusually long passage of text that meandered across ideas—touching on the history of voice interfaces, the challenges of real-time inference on limited hardware, and the subtle artistry of making synthetic voices sound natural—before concluding with the hopeful reminder that progress, while sometimes slow and uneven, is always worth the patience it demands."
    ]

    public static func run(arguments: [String]) async {
        var output = "output.wav"
        var voice = "af_heart"
        var metricsPath: String? = nil
        var chunkDirectory: String? = nil
        var variantPreference: ModelNames.TTS.Variant? = nil
        var text: String? = nil
        var benchmarkMode = false

        var i = 0
        while i < arguments.count {
            let argument = arguments[i]
            switch argument {
            case "--help", "-h":
                printUsage()
                return
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--voice", "-v":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--metrics":
                if i + 1 < arguments.count {
                    metricsPath = arguments[i + 1]
                    i += 1
                }
            case "--chunk-dir":
                if i + 1 < arguments.count {
                    chunkDirectory = arguments[i + 1]
                    i += 1
                }
            case "--variant", "--model-variant":
                if i + 1 < arguments.count {
                    let value = arguments[i + 1].lowercased()
                    switch value {
                    case "5", "5s", "short":
                        variantPreference = .fiveSecond
                    case "15", "15s", "long":
                        variantPreference = .fifteenSecond
                    default:
                        logger.warning("Unknown variant preference '\(arguments[i + 1])'; ignoring")
                    }
                    i += 1
                }
            case "--auto-download":
                // No-op: downloads are always ensured by the CLI
                ()
            case "--benchmark":
                benchmarkMode = true
            default:
                if text == nil {
                    text = argument
                } else {
                    logger.warning("Ignoring unexpected argument '\(argument)'")
                }
            }
            i += 1
        }

        if benchmarkMode {
            await runBenchmark(
                outputPath: output,
                voice: voice,
                metricsPath: metricsPath,
                chunkDirectory: chunkDirectory,
                variantPreference: variantPreference
            )
            return
        }

        guard let text = text else {
            printUsage()
            return
        }

        do {
            // Timing buckets
            let tStart = Date()

            let manager = TtSManager()

            let tLoad0 = Date()
            try await manager.initialize()
            let tLoad1 = Date()

            let tSynth0 = Date()
            let requestedVoice = voice.trimmingCharacters(in: .whitespacesAndNewlines)
            let usedVoice = requestedVoice.isEmpty ? "af_heart" : requestedVoice
            let detailed = try await manager.synthesizeDetailed(
                text: text,
                voice: requestedVoice.isEmpty ? nil : requestedVoice,
                variantPreference: variantPreference
            )
            let wav = detailed.audio
            let tSynth1 = Date()

            // Write WAV
            let outURL = URL(fileURLWithPath: output)
            try wav.write(to: outURL)
            logger.info("Saved output WAV: \(outURL.path)")

            var chunkFileMap: [Int: String] = [:]
            if let chunkDirectory = chunkDirectory {
                let dirURL = URL(fileURLWithPath: chunkDirectory, isDirectory: true)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                for chunk in detailed.chunks {
                    let fileName = String(format: "chunk_%03d.wav", chunk.index)
                    let fileURL = dirURL.appendingPathComponent(fileName)
                    let chunkData = try AudioWAV.data(from: chunk.samples, sampleRate: 24_000)
                    try chunkData.write(to: fileURL)
                    chunkFileMap[chunk.index] = fileURL.path
                }
                logger.info("Saved \(chunkFileMap.count) chunk WAV files to \(dirURL.path)")
            }

            // Metrics
            if let metricsPath = metricsPath {
                let loadS = tLoad1.timeIntervalSince(tLoad0)
                let synthS = tSynth1.timeIntervalSince(tSynth0)
                let totalS = tSynth1.timeIntervalSince(tStart)

                // Approx audio seconds from WAV header (24 kHz mono)
                let audioSecs: Double = {
                    // 44-byte header typical, but use Data length minus header if possible.
                    // We store raw bytes; Safe estimate: payload / (24000 * 2)
                    let bytes = wav.count
                    let payload = max(0, bytes - 44)
                    return Double(payload) / Double(24000 * 2)
                }()
                let rtf = audioSecs > 0 ? (synthS / audioSecs) : 0
                let realtimeSpeed = rtf > 0 ? (1.0 / rtf) : 0

                // Run ASR on the generated audio for comparison
                var asrHypothesis: String? = nil
                var werValue: Double? = nil

                logger.info("--- Running ASR for TTS evaluation ---")
                do {
                    // Load ASR models and initialize
                    let models = try await AsrModels.downloadAndLoad()
                    let asr = AsrManager()
                    try await asr.initialize(models: models)

                    // Transcribe the generated audio file
                    let transcription = try await asr.transcribe(outURL)
                    asrHypothesis = transcription.text

                    // Calculate WER metrics using shared utility
                    let werMetrics = WERCalculator.calculateWERMetrics(
                        hypothesis: transcription.text, reference: text)
                    werValue = werMetrics.wer

                    logger.info("Reference: \(text)")
                    logger.info("ASR Output: \(transcription.text)")
                    logger.info(String(format: "WER: %.1f%%", werValue! * 100))

                    // Clean up ASR resources
                    asr.cleanup()
                } catch {
                    logger.warning("ASR evaluation failed: \(error.localizedDescription)")
                }

                var metricsDict: [String: Any] = [
                    "inference_time_s": synthS,
                    "realtime_speed": realtimeSpeed,
                    "audio_duration_s": audioSecs,
                    "model_load_time_s": loadS,
                    "total_time_s": totalS,
                ]

                if let variantPreference {
                    metricsDict["variant_preference"] = variantPreference == .fiveSecond ? "5s" : "15s"
                }

                // Add ASR comparison if available
                if let asrHypothesis = asrHypothesis {
                    metricsDict["asr_hypothesis"] = asrHypothesis
                    if let werValue = werValue {
                        metricsDict["wer"] = werValue
                    }
                }

                if !detailed.chunks.isEmpty {
                    let frameSamples = 600
                    var totalChunkSamples = 0
                    var chunkLogLines: [String] = []

                    detailed.chunks.enumerated().forEach { index, chunk in
                        let chunkSeconds = Double(chunk.samples.count) / 24_000.0
                        let frameCount = frameSamples > 0 ? chunk.samples.count / frameSamples : 0
                        totalChunkSamples += chunk.samples.count
                        let line = String(
                            format: "Chunk %d duration: %.3fs (%d frames)", index + 1, chunkSeconds,
                            frameCount)
                        chunkLogLines.append(line)
                    }
                    logger.info(chunkLogLines.joined(separator: "\n"))
                    let chunkMetrics = detailed.chunks.map { chunk -> [String: Any] in
                        var entry: [String: Any] = [
                            "index": chunk.index,
                            "text": chunk.text,
                            "pause_after_ms": chunk.pauseAfterMs,
                            "tokens": chunk.tokenCount,
                        ]
                        entry["word_count"] = chunk.wordCount
                        if !chunk.words.isEmpty {
                            entry["normalized_words"] = chunk.words
                        }
                        let chunkSeconds = Double(chunk.samples.count) / 24_000.0
                        let frameCount = frameSamples > 0 ? chunk.samples.count / frameSamples : 0
                        entry["audio_duration_s"] = chunkSeconds
                        entry["frame_count"] = frameCount
                        let variantLabel: String = {
                            switch chunk.variant {
                            case .fiveSecond:
                                return "kokoro_24_5s_v2"
                            case .fifteenSecond:
                                return "kokoro_24_15s"
                            }
                        }()
                        entry["model_variant"] = variantLabel
                        if let path = chunkFileMap[chunk.index] {
                            entry["audio_file"] = path
                        }
                        return entry
                    }
                    metricsDict["chunks"] = chunkMetrics
                    let totalFrames = frameSamples > 0 ? totalChunkSamples / frameSamples : 0
                    logger.info(
                        "Total audio duration: \(String(format: "%.3f", audioSecs))s (\(totalFrames) frames)")
                } else {
                    let frames = Int((audioSecs * 24_000.0) / 600.0)
                    logger.info(
                        "Total audio duration: \(String(format: "%.3f", audioSecs))s (\(frames) frames)")
                }

                let dict: [String: Any] = [
                    "text": text,
                    "voice": usedVoice,
                    "output": outURL.path,
                    "metrics": metricsDict,
                ]

                // Write JSON
                let json = try JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted])
                let mURL = URL(fileURLWithPath: metricsPath)
                try json.write(to: mURL)
                logger.info("Metrics saved: \(mURL.path)")
            }
        } catch {
            logger.error("Error: \(error.localizedDescription)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--metrics metrics.json]

            Options:
              --output, -o         Output WAV path (default: output.wav)
              --voice, -v          Voice name (default: af_heart)
              --benchmark          Run a predefined benchmarking suite with multiple sentences
              --variant            Force Kokoro 5s or 15s model (values: 5s,15s)
              --metrics            Write timing metrics to a JSON file (also runs ASR for evaluation)
              --chunk-dir          Directory where individual chunk WAVs will be written
              (models/dictionary auto-download is always on in CLI)
              --help, -h           Show this help
            """
        )
    }
}

@available(macOS 13.0, *)
extension TTS {
    private struct BenchmarkResult {
        let text: String
        let audioDuration: Double
        let synthesisDuration: Double
        let rtf: Double
        let rtfx: Double
        let outputPath: String?
    }

    private static func runBenchmark(
        outputPath: String,
        voice: String,
        metricsPath: String?,
        chunkDirectory: String?,
        variantPreference: ModelNames.TTS.Variant?
    ) async {
        do {
            let manager = TtSManager()

            let initStart = Date()
            try await manager.initialize()
            let initEnd = Date()

            let requestedVoice = voice.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedVoice = requestedVoice.isEmpty ? nil : requestedVoice
            let usedVoice = normalizedVoice ?? "af_heart"

            var results: [BenchmarkResult] = []
            var totalAudioDuration: Double = 0
            var totalSynthesisDuration: Double = 0

            for (index, sentence) in benchmarkSentences.enumerated() {
                let synthStart = Date()
                let detailed = try await manager.synthesizeDetailed(
                    text: sentence,
                    voice: normalizedVoice,
                    variantPreference: variantPreference
                )
                let synthEnd = Date()

                let audioDuration = audioDurationSeconds(for: detailed)
                let synthesisDuration = synthEnd.timeIntervalSince(synthStart)
                let rtf = audioDuration > 0 ? synthesisDuration / audioDuration : 0
                let rtfx = synthesisDuration > 0 ? audioDuration / synthesisDuration : 0

                let sampleOutputURL = benchmarkOutputURL(basePath: outputPath, index: index)
                try detailed.audio.write(to: sampleOutputURL)
                logger.info("Saved benchmark sample \(index + 1) to \(sampleOutputURL.path)")

                if let chunkDirectory {
                    try writeChunks(
                        detailed: detailed,
                        baseDirectory: chunkDirectory,
                        sampleIndex: index
                    )
                }

                let result = BenchmarkResult(
                    text: sentence,
                    audioDuration: audioDuration,
                    synthesisDuration: synthesisDuration,
                    rtf: rtf,
                    rtfx: rtfx,
                    outputPath: sampleOutputURL.path
                )

                totalAudioDuration += audioDuration
                totalSynthesisDuration += synthesisDuration
                results.append(result)
            }

            printBenchmarkTable(
                voice: usedVoice,
                initializationDuration: initEnd.timeIntervalSince(initStart),
                results: results,
                totalAudioDuration: totalAudioDuration,
                totalSynthesisDuration: totalSynthesisDuration,
                variantPreference: variantPreference
            )

            if let metricsPath {
                try writeBenchmarkMetrics(
                    to: metricsPath,
                    initializationDuration: initEnd.timeIntervalSince(initStart),
                    voice: usedVoice,
                    variantPreference: variantPreference,
                    results: results,
                    totalAudioDuration: totalAudioDuration,
                    totalSynthesisDuration: totalSynthesisDuration
                )
            }
        } catch {
            logger.error("Benchmark run failed: \(error.localizedDescription)")
        }
    }

    private static func benchmarkOutputURL(basePath: String, index: Int) -> URL {
        let baseURL = URL(fileURLWithPath: basePath)
        let directoryURL: URL
        let fileStem: String
        let fileExtension: String

        if baseURL.pathExtension.isEmpty {
            directoryURL = baseURL.deletingLastPathComponent()
            fileStem = baseURL.lastPathComponent.isEmpty ? "output" : baseURL.lastPathComponent
            fileExtension = "wav"
        } else {
            directoryURL = baseURL.deletingLastPathComponent()
            fileStem = baseURL.deletingPathExtension().lastPathComponent
            fileExtension = baseURL.pathExtension
        }

        let fileName = String(format: "%@_benchmark_%02d.%@", fileStem, index + 1, fileExtension)
        if directoryURL.path.isEmpty {
            return URL(fileURLWithPath: fileName)
        }
        return directoryURL.appendingPathComponent(fileName)
    }

    private static func audioDurationSeconds(for detailed: KokoroSynthesizer.SynthesisResult) -> Double {
        let totalSamples = detailed.chunks.reduce(0) { $0 + $1.samples.count }
        if totalSamples > 0 {
            return Double(totalSamples) / 24_000.0
        }

        let bytes = detailed.audio.count
        let payload = max(0, bytes - 44)
        return Double(payload) / Double(24_000 * 2)
    }

    private static func writeChunks(
        detailed: KokoroSynthesizer.SynthesisResult,
        baseDirectory: String,
        sampleIndex: Int
    ) throws {
        let baseURL = URL(fileURLWithPath: baseDirectory, isDirectory: true)
        let sampleDirectory = baseURL.appendingPathComponent(
            String(format: "sample_%02d", sampleIndex + 1), isDirectory: true)
        try FileManager.default.createDirectory(at: sampleDirectory, withIntermediateDirectories: true)

        for chunk in detailed.chunks {
            let fileName = String(format: "chunk_%03d.wav", chunk.index)
            let fileURL = sampleDirectory.appendingPathComponent(fileName)
            let chunkData = try AudioWAV.data(from: chunk.samples, sampleRate: 24_000)
            try chunkData.write(to: fileURL)
        }
    }

    private static func printBenchmarkTable(
        voice: String,
        initializationDuration: TimeInterval,
        results: [BenchmarkResult],
        totalAudioDuration: Double,
        totalSynthesisDuration: Double,
        variantPreference: ModelNames.TTS.Variant?
    ) {
        let indexWidth = 6
        let charsWidth = 8
        let durationWidth = 12
        let ratioWidth = 10

        print("")
        let initString = String(format: "%.3fs", initializationDuration)
        print("TTS benchmark for voice \(voice) (warm-up took an extra \(initString))")
        if let variantPreference {
            print("Variant preference: \(variantPreferenceLabel(variantPreference))")
        }

        let header = [
            padded("Test", width: indexWidth),
            padded("Chars", width: charsWidth),
            padded("Ouput (s)", width: durationWidth),
            padded("Inf(s)", width: durationWidth),
            padded("RTFx", width: ratioWidth),
        ].joined(separator: " ")
        print(header)

        for (index, result) in results.enumerated() {
            let audioString = formattedRatio(result.audioDuration)
            let synthString = formattedRatio(result.synthesisDuration)
            let rtfxString = "\(formattedRatio(result.rtfx))x"

            let row = [
                padded(String(index + 1), width: indexWidth),
                padded(String(result.text.count), width: charsWidth),
                padded(audioString, width: durationWidth),
                padded(synthString, width: durationWidth),
                padded(rtfxString, width: ratioWidth),
            ].joined(separator: " ")
            print(row)
        }

        let totalRTFx = totalSynthesisDuration > 0 ? totalAudioDuration / totalSynthesisDuration : 0
        let totalRow = [
            padded("Total", width: indexWidth),
            padded("-", width: charsWidth),
            padded(String(format: "%.3f", totalAudioDuration), width: durationWidth),
            padded(String(format: "%.3f", totalSynthesisDuration), width: durationWidth),
            padded(formattedRatio(totalRTFx), width: ratioWidth),
        ].joined(separator: " ")
        print(totalRow)
        print("")
    }

    private static func writeBenchmarkMetrics(
        to metricsPath: String,
        initializationDuration: TimeInterval,
        voice: String,
        variantPreference: ModelNames.TTS.Variant?,
        results: [BenchmarkResult],
        totalAudioDuration: Double,
        totalSynthesisDuration: Double
    ) throws {
        let runs: [[String: Any]] = results.enumerated().map { index, result in
            var entry: [String: Any] = [
                "index": index + 1,
                "text": result.text,
                "character_count": result.text.count,
                "audio_duration_s": result.audioDuration,
                "synthesis_time_s": result.synthesisDuration,
                "rtf": result.rtf,
                "rtfx": result.rtfx,
            ]

            if let outputPath = result.outputPath {
                entry["output"] = outputPath
            }

            return entry
        }

        var dictionary: [String: Any] = [
            "voice": voice,
            "runs": runs,
            "total_audio_duration_s": totalAudioDuration,
            "total_synthesis_time_s": totalSynthesisDuration,
            "initialization_time_s": initializationDuration,
        ]

        if let variantPreference {
            dictionary["variant_preference"] = variantPreferenceLabel(variantPreference)
        }

        let json = try JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted])
        try json.write(to: URL(fileURLWithPath: metricsPath))
        logger.info("Benchmark metrics saved to \(metricsPath)")
    }

    private static func padded(_ text: String, width: Int) -> String {
        if text.count >= width { return text }
        return text + String(repeating: " ", count: width - text.count)
    }

    private static func formattedRatio(_ value: Double) -> String {
        guard value.isFinite, value > 0 else { return "n/a" }
        return String(format: "%.3f", value)
    }

    private static func variantPreferenceLabel(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }
}
