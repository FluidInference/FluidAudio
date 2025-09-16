#if os(macOS)
import FluidAudio
import Foundation

/// CLI command that surfaces VadManager's segmentation and streaming APIs.
@available(macOS 13.0, *)
enum VadAnalyzeCommand {
    private static let logger = AppLogger(category: "VadAnalyze")

    private enum Mode: String {
        case segmentation
        case streaming
        case both

        static func from(_ value: String) -> Mode? {
            Mode(rawValue: value.lowercased())
        }
    }

    private struct Options {
        var audioPath: String?
        var mode: Mode = .both
        var threshold: Float?
        var debug: Bool = false
        var returnSeconds: Bool = false
        var timeResolution: Int = 1
        var minSpeechDuration: TimeInterval?
        var minSilenceDuration: TimeInterval?
        var maxSpeechDuration: TimeInterval?
        var speechPadding: TimeInterval?
        var silenceSplitThreshold: Float?
        var negativeThreshold: Float?
        var negativeThresholdOffset: Float?
        var minSilenceAtMaxSpeech: TimeInterval?
        var useMaxSilenceAtMaxSpeech: Bool = true
        var chunkDuration: TimeInterval = Double(VadManager.chunkSize) / Double(VadManager.sampleRate)
    }

    static func run(arguments: [String]) async {
        var options = Options()
        var index = 0

        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--mode":
                options.mode = parseString(arguments, &index, transform: Mode.from) ?? options.mode
            case "--threshold":
                options.threshold = parseString(arguments, &index, transform: Float.init)
            case "--debug":
                options.debug = true
            case "--seconds":
                options.returnSeconds = true
            case "--time-resolution":
                if let value = parseString(arguments, &index, transform: Int.init) {
                    options.timeResolution = max(0, value)
                }
            case "--min-speech-ms":
                options.minSpeechDuration = parseDurationMillis(arguments, &index)
            case "--min-silence-ms":
                options.minSilenceDuration = parseDurationMillis(arguments, &index)
            case "--max-speech-s":
                if let value = parseString(arguments, &index, transform: Double.init) {
                    options.maxSpeechDuration = max(0, value)
                }
            case "--pad-ms":
                options.speechPadding = parseDurationMillis(arguments, &index)
            case "--silence-split-threshold":
                options.silenceSplitThreshold = parseString(arguments, &index, transform: Float.init)
            case "--neg-threshold":
                options.negativeThreshold = parseString(arguments, &index, transform: Float.init)
            case "--neg-offset":
                options.negativeThresholdOffset = parseString(arguments, &index, transform: Float.init)
            case "--min-silence-max-ms":
                options.minSilenceAtMaxSpeech = parseDurationMillis(arguments, &index)
            case "--use-last-silence":
                options.useMaxSilenceAtMaxSpeech = false
            case "--chunk-ms":
                if let value = parseString(arguments, &index, transform: Double.init) {
                    options.chunkDuration = max(0.01, value / 1000.0)
                }
            default:
                if arg.hasPrefix("--") {
                    logger.warning("Unknown option: \(arg)")
                } else if options.audioPath == nil {
                    options.audioPath = arg
                } else {
                    logger.warning("Ignoring extra argument: \(arg)")
                }
            }
            index += 1
        }

        guard let audioPath = options.audioPath else {
            logger.error("No audio file provided")
            printUsage()
            exit(1)
        }

        do {
            let samples = try AudioConverter().resampleAudioFile(path: audioPath)
            let manager = try await VadManager(
                config: VadConfig(
                    threshold: options.threshold ?? VadConfig.default.threshold,
                    debugMode: options.debug
                )
            )

            let segmentationConfig = buildSegmentationConfig(options: options)

            if options.mode == .segmentation || options.mode == .both {
                await runSegmentation(
                    manager: manager,
                    samples: samples,
                    config: segmentationConfig
                )
            }

            if options.mode == .streaming || options.mode == .both {
                await runStreaming(
                    manager: manager,
                    samples: samples,
                    options: options,
                    config: segmentationConfig
                )
            }
        } catch {
            logger.error("VAD analysis failed: \(error)")
            exit(1)
        }
    }

    private static func runSegmentation(
        manager: VadManager,
        samples: [Float],
        config: VadSegmentationConfig
    ) async {
        do {
            logger.info("📍 Running offline speech segmentation...")
            let start = Date()
            let segments = try await manager.segmentSpeech(samples, config: config)
            let duration = Date().timeIntervalSince(start)
            logger.info(
                "Detected \(segments.count) speech segments in \(String(format: "%.2f", duration))s"
            )

            for (index, segment) in segments.enumerated() {
                let startSample = segment.startSample(sampleRate: VadManager.sampleRate)
                let endSample = segment.endSample(sampleRate: VadManager.sampleRate)
                let startTime = segment.startTime
                let endTime = segment.endTime
                logger.info(
                    "Segment #\(index + 1): samples \(startSample)-\(endSample) ("
                        + String(format: "%.2fs-%.2fs", startTime, endTime) + ")"
                )
            }
        } catch {
            logger.error("Segmentation failed: \(error)")
        }
    }

    private static func runStreaming(
        manager: VadManager,
        samples: [Float],
        options: Options,
        config: VadSegmentationConfig
    ) async {
        do {
            logger.info("📶 Running streaming simulation...")
            var state = await manager.makeStreamState()
            let chunkSamples = max(1, Int(options.chunkDuration * Double(VadManager.sampleRate)))
            var emittedEvents: [VadStreamEvent] = []

            for start in stride(from: 0, to: samples.count, by: chunkSamples) {
                let end = min(start + chunkSamples, samples.count)
                let chunk = Array(samples[start..<end])
                let result = try await manager.processStreamingChunk(
                    chunk,
                    state: state,
                    config: config,
                    returnSeconds: options.returnSeconds,
                    timeResolution: options.timeResolution
                )
                state = result.state
                if let event = result.event {
                    emittedEvents.append(event)
                    logStreamEvent(event, returnSeconds: options.returnSeconds)
                }
            }

            if state.triggered {
                logger.info("Flushing trailing silence to close open segments...")
                let silenceChunk = [Float](repeating: 0, count: chunkSamples)
                var flushState = state
                var guardCounter = 0
                while flushState.triggered && guardCounter < 8 {
                    let flush = try await manager.processStreamingChunk(
                        silenceChunk,
                        state: flushState,
                        config: config,
                        returnSeconds: options.returnSeconds,
                        timeResolution: options.timeResolution
                    )
                    flushState = flush.state
                    if let event = flush.event {
                        emittedEvents.append(event)
                        logStreamEvent(event, returnSeconds: options.returnSeconds)
                    }
                    guardCounter += 1
                }
                state = flushState
            }

            logger.info("Streaming simulation produced \(emittedEvents.count) events")
        } catch {
            logger.error("Streaming simulation failed: \(error)")
        }
    }

    private static func logStreamEvent(_ event: VadStreamEvent, returnSeconds: Bool) {
        let label = event.isStart ? "Speech Start" : "Speech End"
        if returnSeconds, let time = event.time {
            let formatted = String(format: "%.3fs", time)
            logger.info("  • \(label) at \(formatted)")
        } else {
            logger.info("  • \(label) at sample \(event.sampleIndex)")
        }
    }

    private static func buildSegmentationConfig(options: Options) -> VadSegmentationConfig {
        var config = VadSegmentationConfig.default
        if let value = options.minSpeechDuration { config.minSpeechDuration = value }
        if let value = options.minSilenceDuration { config.minSilenceDuration = value }
        if let value = options.maxSpeechDuration {
            config.maxSpeechDuration = value.isInfinite ? .infinity : value
        }
        if let value = options.speechPadding { config.speechPadding = value }
        if let value = options.silenceSplitThreshold { config.silenceThresholdForSplit = value }
        if let value = options.negativeThreshold { config.negativeThreshold = value }
        if let value = options.negativeThresholdOffset { config.negativeThresholdOffset = value }
        if let value = options.minSilenceAtMaxSpeech { config.minSilenceAtMaxSpeech = value }
        config.useMaxPossibleSilenceAtMaxSpeech = options.useMaxSilenceAtMaxSpeech
        return config
    }

    private static func parseString<T>(
        _ arguments: [String],
        _ index: inout Int,
        transform: (String) -> T?
    ) -> T? {
        guard index + 1 < arguments.count else {
            logger.error("Missing value for option \(arguments[index])")
            return nil
        }
        let raw = arguments[index + 1]
        if let value = transform(raw) {
            index += 1
            return value
        }
        logger.error("Invalid value '\(raw)' for option \(arguments[index])")
        return nil
    }

    private static func parseDurationMillis(
        _ arguments: [String],
        _ index: inout Int
    ) -> TimeInterval? {
        guard let value = parseString(arguments, &index, transform: Double.init) else { return nil }
        return max(0, value) / 1000.0
    }

    private static func printUsage() {
        logger.info(
            """

            VAD Analyze Command Usage:
                fluidaudio vad-analyze <audio_file> [options]

            Options:
                --mode <segmentation|streaming|both>    Select analysis mode (default: both)
                --threshold <float>                      Override VAD probability threshold
                --debug                                  Enable VadManager debug logging
                --seconds                                Emit streaming timestamps in seconds
                --time-resolution <int>                  Decimal precision when using --seconds (default: 1)
                --chunk-ms <double>                      Streaming chunk size in milliseconds (default: 256)
                --min-speech-ms <double>                 Minimum speech span considered valid
                --min-silence-ms <double>                Required trailing silence duration
                --max-speech-s <double>                  Maximum length of a single speech segment
                --pad-ms <double>                        Padding applied around detected speech
                --silence-split-threshold <float>        Minimum silence probability for max-duration splits
                --neg-threshold <float>                  Override negative threshold for hysteresis
                --neg-offset <float>                     Offset from threshold for negative threshold
                --min-silence-max-ms <double>            Silence guard used when hitting max speech duration
                --use-last-silence                       Prefer the last candidate silence when splitting

            Examples:
                fluidaudio vad-analyze audio.wav --mode segmentation
                fluidaudio vad-analyze audio.wav --mode streaming --seconds --chunk-ms 128
            """
        )
    }
}
#endif
