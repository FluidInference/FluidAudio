#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// CLI command that surfaces VadManager's segmentation and streaming APIs.
enum VadAnalyzeCommand {
    private static let logger = AppLogger(category: "VadAnalyze")

    private struct Options {
        var audioPath: String?
        var streaming: Bool = false
        var threshold: Float?
        var debug: Bool = false
        var minSpeechDuration: TimeInterval?
        var minSilenceDuration: TimeInterval?
        var maxSpeechDuration: TimeInterval?
        var speechPadding: TimeInterval?
        var silenceSplitThreshold: Float?
        var negativeThreshold: Float?
        var negativeThresholdOffset: Float?
        var minSilenceAtMaxSpeech: TimeInterval?
        var useMaxSilenceAtMaxSpeech: Bool = true
        var exportPath: String?
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
            case "--streaming":
                options.streaming = true
            case "--threshold":
                options.threshold = parseString(arguments, &index, transform: Float.init)
            case "--debug":
                options.debug = true
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
            case "--export-wav":
                options.exportPath = parseString(arguments, &index) { raw in
                    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    return trimmed.isEmpty ? nil : trimmed
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
                    defaultThreshold: options.threshold ?? VadConfig.default.defaultThreshold,
                    debugMode: options.debug
                )
            )

            let segmentationConfig = buildSegmentationConfig(options: options)

            let shouldRunSegmentation = !options.streaming || options.exportPath != nil
            var segments: [VadSegment] = []

            if shouldRunSegmentation {
                segments = await runSegmentation(
                    manager: manager,
                    samples: samples,
                    config: segmentationConfig
                )

                if let exportPath = options.exportPath {
                    exportSpeechSegments(
                        segments: segments,
                        samples: samples,
                        destinationPath: exportPath
                    )
                }
            }

            if options.streaming {
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
    ) async -> [VadSegment] {
        do {
            logger.info("📍 Running offline speech segmentation...")
            let inferenceStart = Date()
            let results = try await manager.process(samples)

            let segments = await manager.segmentSpeech(
                from: results,
                totalSamples: samples.count,
                config: config
            )

            let totalModelTime = results.reduce(0.0) { $0 + $1.processingTime }
            let audioSeconds = Double(samples.count) / Double(VadManager.sampleRate)
            let rtf = totalModelTime > 0 ? audioSeconds / totalModelTime : 0

            let duration = Date().timeIntervalSince(inferenceStart)
            logger.info(
                "Detected \(segments.count) speech segments in \(String(format: "%.2f", duration))s"
            )

            if !results.isEmpty {
                logger.info(
                    String(
                        format: "RTFx: %.2fx (audio: %.2fs, inference: %.2fs)",
                        rtf,
                        audioSeconds,
                        totalModelTime
                    )
                )
            }

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
            return segments
        } catch {
            logger.error("Segmentation failed: \(error)")
            return []
        }
    }

    private enum ExportError: Error {
        case failedToCreateBuffer
        case failedToAccessChannelData
    }

    private static func exportSpeechSegments(
        segments: [VadSegment],
        samples: [Float],
        destinationPath: String
    ) {
        guard !segments.isEmpty else {
            logger.info("No speech segments detected; nothing to export")
            return
        }

        let sampleRate = VadManager.sampleRate

        var totalSamples = 0
        var ranges: [Range<Int>] = []
        ranges.reserveCapacity(segments.count)

        for segment in segments {
            guard
                let range = clampedRange(
                    for: segment,
                    sampleCount: samples.count,
                    sampleRate: sampleRate
                )
            else { continue }
            ranges.append(range)
            totalSamples += range.count
        }

        guard totalSamples > 0 else {
            logger.info("Speech segments were zero-length after clamping; nothing to export")
            return
        }

        var concatenated = [Float]()
        concatenated.reserveCapacity(totalSamples)

        for range in ranges {
            concatenated.append(contentsOf: samples[range])
        }

        do {
            let expandedPath = (destinationPath as NSString).expandingTildeInPath
            try writeWav(samples: concatenated, sampleRate: sampleRate, to: expandedPath)
            let durationSeconds = Double(concatenated.count) / Double(sampleRate)
            logger.info(
                "Saved \(segments.count) speech segments to \(expandedPath) (~\(String(format: "%.2f", durationSeconds))s)"
            )
        } catch {
            logger.error("Failed to export speech segments: \(error)")
        }
    }

    private static func clampedRange(
        for segment: VadSegment,
        sampleCount: Int,
        sampleRate: Int
    ) -> Range<Int>? {
        guard sampleCount > 0 else { return nil }
        let startSample = segment.startSample(sampleRate: sampleRate)
        let endSample = segment.endSample(sampleRate: sampleRate)
        let clampedStart = max(0, min(startSample, sampleCount))
        let clampedEnd = max(clampedStart, min(endSample, sampleCount))
        return clampedStart < clampedEnd ? clampedStart..<clampedEnd : nil
    }

    private static func writeWav(samples: [Float], sampleRate: Int, to path: String) throws {
        guard !samples.isEmpty else { return }

        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false
            )
        else {
            throw ExportError.failedToCreateBuffer
        }

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else {
            throw ExportError.failedToCreateBuffer
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)

        guard let channelData = buffer.floatChannelData else {
            throw ExportError.failedToAccessChannelData
        }

        samples.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            channelData[0].update(from: base, count: samples.count)
        }

        let url = URL(fileURLWithPath: path)
        let directoryURL = url.deletingLastPathComponent()
        if !directoryURL.path.isEmpty && directoryURL.path != "." {
            try FileManager.default.createDirectory(
                at: directoryURL,
                withIntermediateDirectories: true
            )
        }

        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings
        )
        try audioFile.write(from: buffer)
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
            let chunkSamples = VadManager.chunkSize
            var emittedEvents: [VadStreamEvent] = []

            for start in stride(from: 0, to: samples.count, by: chunkSamples) {
                let end = min(start + chunkSamples, samples.count)
                let chunk = Array(samples[start..<end])
                let result = try await manager.processStreamingChunk(
                    chunk,
                    state: state,
                    config: config,
                    returnSeconds: true
                )
                state = result.state
                if let event = result.event {
                    emittedEvents.append(event)
                    logStreamEvent(event)
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
                        returnSeconds: true
                    )
                    flushState = flush.state
                    if let event = flush.event {
                        emittedEvents.append(event)
                        logStreamEvent(event)
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

    private static func logStreamEvent(_ event: VadStreamEvent) {
        let label = event.isStart ? "Speech Start" : "Speech End"
        if let time = event.time {
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
                --streaming                              Run streaming simulation instead of offline segmentation
                --threshold <float>                      Override VAD probability threshold
                --debug                                  Enable VadManager debug logging
                --min-speech-ms <double>                 Minimum speech span considered valid
                --min-silence-ms <double>                Required trailing silence duration
                --max-speech-s <double>                  Maximum length of a single speech segment
                --pad-ms <double>                        Padding applied around detected speech
                --silence-split-threshold <float>        Minimum silence probability for max-duration splits
                --neg-threshold <float>                  Override negative threshold for hysteresis
                --neg-offset <float>                     Offset from threshold for negative threshold
                --min-silence-max-ms <double>            Silence guard used when hitting max speech duration
                --use-last-silence                       Prefer the last candidate silence when splitting
                --export-wav <path>                      Write detected speech segments to a single WAV file

            Examples:
                fluidaudio vad-analyze audio.wav
                fluidaudio vad-analyze audio.wav --streaming
            """
        )
    }
}
#endif
