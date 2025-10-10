#if os(macOS)
import AVFoundation
import AudioToolbox
import Darwin
import FluidAudio
import Foundation

/// Command to perform stabilized streaming transcription from the microphone.
@available(macOS 13.0, *)
enum MicTranscribeCommand {
    private static let logger = AppLogger(category: "MicTranscribe")
    private static let confirmedColor = "\u{001B}[32m"
    private static let volatileColor = "\u{001B}[33m"
    private static let resetColor = "\u{001B}[0m"
    private static let timestampFormatterQueue = DispatchQueue(
        label: "com.fluidaudio.mictranscribe.timestamp"
    )
    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        return formatter
    }()

    static func run(arguments: [String]) async {
        var showMetadata = false
        var modelVersion: AsrModelVersion = .v3
        var autoStopDuration: TimeInterval?

        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--metadata":
                showMetadata = true
            case "--model-version":
                guard index + 1 < arguments.count else {
                    logError("--model-version expects a value (v2|v3)")
                    exit(1)
                }
                let value = arguments[index + 1].lowercased()
                switch value {
                case "v2", "2":
                    modelVersion = .v2
                case "v3", "3":
                    modelVersion = .v3
                default:
                    logError("Invalid model version: \(arguments[index + 1]). Use 'v2' or 'v3'")
                    exit(1)
                }
                index += 1
            case "--duration":
                guard index + 1 < arguments.count, let seconds = TimeInterval(arguments[index + 1]),
                    seconds > 0
                else {
                    logError("--duration expects a positive number of seconds")
                    exit(1)
                }
                autoStopDuration = seconds
                index += 1
            default:
                logWarning("Unknown option: \(arguments[index])")
            }
            index += 1
        }

        let config = micStreamingConfig()

        var updateTask: Task<Void, Never>?
        let confirmedAccumulator = ConfirmedAccumulator()

        do {
            logInfo(
                "Downloading and loading ASR models (version: \(modelVersionDescription(modelVersion)))..."
            )
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)

            let streamingAsr = StreamingAsrManager(config: config)
            try await streamingAsr.start(models: models, source: .microphone)

            let engine = AVAudioEngine()
            let inputNode = engine.inputNode
            let inputFormat = inputNode.outputFormat(forBus: 0)
            let bufferSize = recommendedBufferSize(for: inputFormat)

            logInfo(
                """
                Microphone input ready:
                  Sample rate: \(String(format: "%.0f", inputFormat.sampleRate)) Hz
                  Channels: \(inputFormat.channelCount)
                  Buffer size: \(bufferSize) frames (~\(String(format: "%.0f", Double(bufferSize) / inputFormat.sampleRate * 1000)) ms)
                  Stabilizer: enabled (window=\(config.stabilizer.windowSize))
                """
            )

            updateTask = Task {
                let timestampFormatter: DateFormatter = {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "HH:mm:ss.SSS"
                    return formatter
                }()

                for await update in await streamingAsr.transcriptionUpdates {
                    let label = coloredLabel(for: update)
                    if showMetadata {
                        let timestampString = timestampFormatter.string(from: update.timestamp)
                        logInfo(
                            "\(label) '\(update.text)' (conf: \(String(format: "%.3f", update.confidence)), timestamp: \(timestampString))"
                        )
                        if !update.tokenTimings.isEmpty {
                            for (idx, timing) in update.tokenTimings.enumerated() {
                                logInfo(
                                    "  [\(idx)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                                )
                            }
                        }
                        if update.isConfirmed {
                            await confirmedAccumulator.append(update.text)
                        }
                    } else {
                        logInfo(
                            "\(label) '\(update.text)' (conf: \(String(format: "%.2f", update.confidence)))"
                        )
                        if update.isConfirmed {
                            await confirmedAccumulator.append(update.text)
                        }
                    }
                }
            }

            try configureEngine(engine, bufferSize: bufferSize, format: inputFormat, streamingAsr: streamingAsr)

            logInfo(statusMessage(for: autoStopDuration))

            let stopReason = await waitForStop(duration: autoStopDuration)
            logInfo("Stopping microphone capture (\(stopReason.description))")

            inputNode.removeTap(onBus: 0)
            if engine.isRunning {
                engine.stop()
            }

            let finalText = try await streamingAsr.finish()

            if let task = updateTask {
                task.cancel()
                await task.value
            }

            let confirmedParagraph = await confirmedAccumulator.aggregatedText()

            logInfo("" + String(repeating: "=", count: 50))
            logInfo("LIVE MICROPHONE TRANSCRIPTION")
            logInfo(String(repeating: "=", count: 50))
            if !confirmedParagraph.isEmpty {
                logInfo("Confirmed transcript:")
                logInfo("\(confirmedColor)\(confirmedParagraph)\(resetColor)")
            } else {
                logInfo("Confirmed transcript: <none>")
            }

            if finalText != confirmedParagraph && !finalText.isEmpty {
                logInfo("Full transcription (including last flush):")
                logInfo("\(confirmedColor)\(finalText)\(resetColor)")
            }
        } catch {
            logError("Microphone transcription failed: \(error)")
            updateTask?.cancel()
            exit(1)
        }
    }

    private static func modelVersionDescription(_ version: AsrModelVersion) -> String {
        switch version {
        case .v2:
            return "v2"
        case .v3:
            return "v3"
        }
    }

    private static func micStreamingConfig() -> StreamingAsrConfig {
        let stabilizer =
            StreamingStabilizerConfig
            .preset(.balanced)
            .withMaxWaitMilliseconds(750)

        return StreamingAsrConfig(
            chunkSeconds: 1.5,
            leftContextSeconds: 1.0,
            rightContextSeconds: 0.4,
            stabilizer: stabilizer
        )
    }

    private static func configureEngine(
        _ engine: AVAudioEngine,
        bufferSize: AVAudioFrameCount,
        format: AVAudioFormat,
        streamingAsr: StreamingAsrManager
    ) throws {
        let inputNode = engine.inputNode

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: format) { buffer, _ in
            guard buffer.frameLength > 0 else {
                return
            }

            guard let copy = makeBufferCopy(from: buffer) else {
                logDebug("Dropping microphone frame: unable to copy buffer")
                return
            }

            Task(priority: .userInitiated) {
                await streamingAsr.streamAudio(copy)
            }
        }

        engine.prepare()
        try engine.start()
    }

    private static func coloredLabel(for update: StreamingTranscriptionUpdate) -> String {
        if update.isConfirmed {
            return "\(confirmedColor)CONFIRMED\(resetColor)"
        } else {
            return "\(volatileColor)VOLATILE\(resetColor)"
        }
    }

    private static func waitForEnterKey() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInteractive).async {
                _ = readLine()
                continuation.resume()
            }
        }
    }

    private static func makeBufferCopy(from buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let frameLength = buffer.frameLength
        guard frameLength > 0 else {
            return nil
        }

        guard
            let copy = AVAudioPCMBuffer(
                pcmFormat: buffer.format,
                frameCapacity: frameLength
            )
        else {
            return nil
        }

        copy.frameLength = frameLength

        let sourceListPointer = buffer.audioBufferList
        let destinationListPointer = copy.mutableAudioBufferList

        let bufferCount = Int(sourceListPointer.pointee.mNumberBuffers)

        let sourceBuffers = withUnsafePointer(to: sourceListPointer.pointee.mBuffers) { pointer in
            UnsafeRawPointer(pointer).assumingMemoryBound(to: AudioBuffer.self)
        }

        let destinationBuffers = withUnsafeMutablePointer(to: &destinationListPointer.pointee.mBuffers) { pointer in
            UnsafeMutableRawPointer(pointer).assumingMemoryBound(to: AudioBuffer.self)
        }

        for index in 0..<bufferCount {
            let sourceBuffer = sourceBuffers.advanced(by: index)
            let destinationBuffer = destinationBuffers.advanced(by: index)

            let size = Int(sourceBuffer.pointee.mDataByteSize)
            guard size > 0 else { continue }

            guard let srcData = sourceBuffer.pointee.mData else { continue }
            guard let dstData = destinationBuffer.pointee.mData else { continue }

            memcpy(dstData, srcData, size)
            destinationBuffer.pointee.mDataByteSize = sourceBuffer.pointee.mDataByteSize
        }

        return copy
    }

    private static func recommendedBufferSize(for format: AVAudioFormat) -> AVAudioFrameCount {
        let sampleRate = max(format.sampleRate, 16_000)
        let targetDuration = 0.2  // seconds
        let frames = Int(sampleRate * targetDuration)
        let alignment = 256
        let alignedFrames = max(alignment, (frames / alignment) * alignment)
        return AVAudioFrameCount(min(alignedFrames, 16_384))
    }

    private static func statusMessage(for duration: TimeInterval?) -> String {
        if let duration {
            return
                "🎙️ Recording... stabilized streaming is active. Auto-stop after \(String(format: "%.1f", duration))s."
        } else {
            return "🎙️ Recording... stabilized streaming is active. Press ENTER to stop or press CTRL+C to finish."
        }
    }

    private static func waitForStop(duration: TimeInterval?) async -> StopReason {
        await withTaskGroup(of: StopReason?.self, returning: StopReason.self) { group in
            group.addTask {
                await waitForInterrupt()
                return .interrupted
            }

            group.addTask {
                await waitForEnterKey()
                return .userRequested
            }

            if let duration {
                group.addTask {
                    await waitForDuration(duration)
                    return .autoTimeout
                }
            }

            guard let reason = await group.next() ?? nil else {
                group.cancelAll()
                return .userRequested
            }

            group.cancelAll()
            return reason
        }
    }

    private static func waitForDuration(_ duration: TimeInterval) async {
        try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
    }

    private static func waitForInterrupt() async {
        var iterator = signalStream(for: SIGINT).makeAsyncIterator()
        _ = await iterator.next()
    }

    private static func signalStream(for signalNumber: Int32) -> AsyncStream<Void> {
        AsyncStream { continuation in
            signal(signalNumber, SIG_IGN)
            let queue = DispatchQueue(label: "com.fluidaudio.mictranscribe.signal", qos: .userInteractive)
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: queue)
            source.setEventHandler {
                continuation.yield(())
                continuation.finish()
            }
            source.setCancelHandler {
                signal(signalNumber, SIG_DFL)
            }
            continuation.onTermination = { _ in
                source.cancel()
            }
            source.activate()
        }
    }

    private static func logInfo(_ message: @autoclosure () -> String) {
        logger.info(formattedMessage(message()))
    }

    private static func logWarning(_ message: @autoclosure () -> String) {
        logger.warning(formattedMessage(message()))
    }

    private static func logError(_ message: @autoclosure () -> String) {
        logger.error(formattedMessage(message()))
    }

    private static func logDebug(_ message: @autoclosure () -> String) {
        logger.debug(formattedMessage(message()))
    }

    private static func formattedMessage(_ message: String) -> String {
        "\(message)"
    }

    private static func currentTimestamp() -> String {
        timestampFormatterQueue.sync {
            timestampFormatter.string(from: Date())
        }
    }

    private static func printUsage() {
        logInfo(
            """

            Microphone Transcribe Command Usage:
                fluidaudio mic-transcribe [options]

            Options:
                --help, -h             Show this help message
                --metadata             Print token-level metadata for each update
                --model-version <v2|v3> Select ASR model version (default: v3)
                --duration <seconds>   Automatically stop after the specified duration

            Examples:
                fluidaudio mic-transcribe
                fluidaudio mic-transcribe --metadata
                fluidaudio mic-transcribe --duration 30

            Press ENTER (or CTRL+C) to finish recording when no duration is provided.
            """
        )
    }

    private actor ConfirmedAccumulator {
        private var text: String = ""

        func append(_ segment: String) {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }

            if text.isEmpty {
                text = trimmed
                return
            }

            if needsSpace(before: trimmed) {
                text.append(" ")
            }
            text.append(trimmed)
        }

        func aggregatedText() -> String {
            text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        private func needsSpace(before segment: String) -> Bool {
            guard let first = segment.first else { return false }
            let punctuation: Set<Character> = [".", ",", "!", "?", ";", ":", ")", "]", "}", "'"]
            if punctuation.contains(first) {
                return false
            }
            if text.last == " " {
                return false
            }
            return true
        }
    }

    private enum StopReason: String {
        case userRequested
        case autoTimeout
        case interrupted

        var description: String {
            switch self {
            case .userRequested:
                return "user requested stop"
            case .autoTimeout:
                return "duration elapsed"
            case .interrupted:
                return "received CTRL+C"
            }
        }
    }
}
#endif
