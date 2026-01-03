#if os(macOS)
@preconcurrency import AVFoundation
import FluidAudio
import Foundation

/// Thread-safe tracker for transcription updates and audio position
actor TranscriptionTracker {
    private var volatileUpdates: [String] = []
    private var confirmedUpdates: [String] = []
    private var currentAudioPosition: Double = 0.0
    private let startTime: Date
    private var latestUpdate: StreamingTranscriptionUpdate?
    private var latestConfirmedUpdate: StreamingTranscriptionUpdate?
    private var tokenTimingMap: [TokenKey: TokenTiming] = [:]

    init() {
        self.startTime = Date()
    }

    func addVolatileUpdate(_ text: String) {
        volatileUpdates.append(text)
    }

    func addConfirmedUpdate(_ text: String) {
        confirmedUpdates.append(text)
    }

    func updateAudioPosition(_ position: Double) {
        currentAudioPosition = position
    }

    func getCurrentAudioPosition() -> Double {
        return currentAudioPosition
    }

    func getElapsedProcessingTime() -> Double {
        return Date().timeIntervalSince(startTime)
    }

    func getVolatileCount() -> Int {
        return volatileUpdates.count
    }

    func getConfirmedCount() -> Int {
        return confirmedUpdates.count
    }

    func record(update: StreamingTranscriptionUpdate) {
        latestUpdate = update

        if update.isConfirmed {
            latestConfirmedUpdate = update

            for timing in update.tokenTimings {
                let key = TokenKey(
                    tokenId: timing.tokenId,
                    startMilliseconds: Int((timing.startTime * 1000).rounded())
                )
                tokenTimingMap[key] = timing
            }
        }
    }

    func metadataSnapshot() -> (timings: [TokenTiming], isConfirmed: Bool)? {
        if !tokenTimingMap.isEmpty {
            let timings = tokenTimingMap.values.sorted { lhs, rhs in
                if lhs.startTime == rhs.startTime {
                    return lhs.tokenId < rhs.tokenId
                }
                return lhs.startTime < rhs.startTime
            }
            return (timings, true)
        }

        if let update = latestConfirmedUpdate ?? latestUpdate, !update.tokenTimings.isEmpty {
            let timings = update.tokenTimings.sorted { lhs, rhs in
                if lhs.startTime == rhs.startTime {
                    return lhs.tokenId < rhs.tokenId
                }
                return lhs.startTime < rhs.startTime
            }
            return (timings, update.isConfirmed)
        }

        return nil
    }

    func latestUpdateSnapshot() -> StreamingTranscriptionUpdate? {
        latestConfirmedUpdate ?? latestUpdate
    }

    private struct TokenKey: Hashable {
        let tokenId: Int
        let startMilliseconds: Int
    }
}

/// Word-level timing information
struct WordTiming: Codable, Sendable {
    let word: String
    let startTime: TimeInterval
    let endTime: TimeInterval
    let confidence: Float
}

/// JSON output model for transcription results
struct TranscriptionJSONOutput: Codable {
    let audioFile: String
    let mode: String
    let modelVersion: String
    let text: String
    let durationSeconds: TimeInterval?
    let processingTimeSeconds: TimeInterval?
    let rtfx: Float?
    let confidence: Float?
    let wordTimings: [WordTiming]
    let timingsConfirmed: Bool?
}

/// Helper to merge tokens into word-level timings
///
/// This merger assumes that the ASR tokenizer produces subword tokens where:
/// - Tokens starting with whitespace (space, newline, tab) indicate word boundaries
/// - Multiple consecutive tokens without leading whitespace form a single word
/// - This pattern is typical for BPE (Byte Pair Encoding) tokenizers like SentencePiece
enum WordTimingMerger {
    /// Merge token timings into word-level timings by detecting word boundaries
    ///
    /// - Parameter tokenTimings: Array of token-level timing information from the ASR model
    /// - Returns: Array of word-level timing information with merged tokens
    ///
    /// Example: Tokens `[" H", "ello", " wor", "ld"]` → Words `["Hello", "world"]`
    static func mergeTokensIntoWords(_ tokenTimings: [TokenTiming]) -> [WordTiming] {
        guard !tokenTimings.isEmpty else { return [] }

        var wordTimings: [WordTiming] = []
        var currentWord = ""
        var currentStartTime: TimeInterval?
        var currentEndTime: TimeInterval = 0
        var currentConfidences: [Float] = []

        for timing in tokenTimings {
            let token = timing.token

            // Check if token starts with whitespace (indicates new word boundary)
            if token.hasPrefix(" ") || token.hasPrefix("\n") || token.hasPrefix("\t") {
                // Finish previous word if exists
                if !currentWord.isEmpty, let startTime = currentStartTime {
                    wordTimings.append(
                        WordTiming(
                            word: currentWord,
                            startTime: startTime,
                            endTime: currentEndTime,
                            confidence: averageConfidence(currentConfidences)
                        ))
                }

                // Start new word (trim leading whitespace)
                currentWord = token.trimmingCharacters(in: .whitespacesAndNewlines)
                currentStartTime = timing.startTime
                currentEndTime = timing.endTime
                currentConfidences = [timing.confidence]
            } else {
                // Continue current word or start first word if no whitespace prefix
                if currentStartTime == nil {
                    currentStartTime = timing.startTime
                }
                currentWord += token
                currentEndTime = timing.endTime
                currentConfidences.append(timing.confidence)
            }
        }

        // Add final word
        if !currentWord.isEmpty, let startTime = currentStartTime {
            wordTimings.append(
                WordTiming(
                    word: currentWord,
                    startTime: startTime,
                    endTime: currentEndTime,
                    confidence: averageConfidence(currentConfidences)
                ))
        }

        return wordTimings
    }

    /// Calculate average confidence from an array of confidence scores
    /// - Parameter confidences: Array of confidence values
    /// - Returns: Average confidence, or 0.0 if array is empty
    private static func averageConfidence(_ confidences: [Float]) -> Float {
        confidences.isEmpty ? 0.0 : confidences.reduce(0, +) / Float(confidences.count)
    }
}

/// Command to transcribe audio files using batch or streaming mode
enum TranscribeCommand {
    private static let logger = AppLogger(category: "Transcribe")

    static func run(arguments: [String]) async {
        // Parse arguments
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var streamingMode = false
        var showMetadata = false
        var wordTimestamps = false
        var outputJsonPath: String?
        var modelVersion: AsrModelVersion = .v3  // Default to v3

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--streaming":
                streamingMode = true
            case "--metadata":
                showMetadata = true
            case "--word-timestamps":
                wordTimestamps = true
            case "--output-json":
                if i + 1 < arguments.count {
                    outputJsonPath = arguments[i + 1]
                    i += 1
                }
            case "--model-version":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v2", "2":
                        modelVersion = .v2
                    case "v3", "3":
                        modelVersion = .v3
                    default:
                        logger.error("Invalid model version: \(arguments[i + 1]). Use 'v2' or 'v3'")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with 1-second chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile, showMetadata: showMetadata, wordTimestamps: wordTimestamps,
                outputJsonPath: outputJsonPath, modelVersion: modelVersion)
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(
                audioFile: audioFile, showMetadata: showMetadata, wordTimestamps: wordTimestamps,
                outputJsonPath: outputJsonPath, modelVersion: modelVersion)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String, showMetadata: Bool, wordTimestamps: Bool, outputJsonPath: String?,
        modelVersion: AsrModelVersion
    ) async {
        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)
            let asrManager = AsrManager(config: .default)
            try await asrManager.initialize(models: models)

            logger.info("ASR Manager initialized successfully")

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                logger.error("Failed to create audio buffer")
                return
            }

            try audioFileHandle.read(into: buffer)

            // Convert audio to the format expected by ASR (16kHz mono Float array)
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(audioFileHandle.length) / format.sampleRate
            logger.info("Processing \(String(format: "%.2f", duration))s of audio (\(samples.count) samples)\n")

            // Process with ASR Manager
            logger.info("Transcribing file: \(audioFileURL) ...")
            let startTime = Date()
            let result = try await asrManager.transcribe(audioFileURL)
            let processingTime = Date().timeIntervalSince(startTime)

            // Print results
            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

            if let outputJsonPath = outputJsonPath {
                let wordTimings = WordTimingMerger.mergeTokensIntoWords(result.tokenTimings ?? [])
                let modelVersionLabel = modelVersion == .v2 ? "v2" : "v3"
                let output = TranscriptionJSONOutput(
                    audioFile: audioFile,
                    mode: "batch",
                    modelVersion: modelVersionLabel,
                    text: result.text,
                    durationSeconds: result.duration,
                    processingTimeSeconds: result.processingTime,
                    rtfx: result.rtfx,
                    confidence: result.confidence,
                    wordTimings: wordTimings,
                    timingsConfirmed: nil
                )
                try writeJsonOutput(output, to: outputJsonPath)
                logger.info("💾 JSON results saved to: \(outputJsonPath)")
            }

            // Print word-level timestamps if requested
            if wordTimestamps {
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
                    let wordTimings = WordTimingMerger.mergeTokensIntoWords(tokenTimings)
                    logger.info("\nWord-level timestamps:")
                    for (index, word) in wordTimings.enumerated() {
                        logger.info(
                            "  [\(index)] \(String(format: "%.3f", word.startTime))s - \(String(format: "%.3f", word.endTime))s: \"\(word.word)\" (conf: \(String(format: "%.3f", word.confidence)))"
                        )
                    }
                } else {
                    logger.info("\nWord-level timestamps: Not available (no token timings)")
                }
            }

            if showMetadata {
                logger.info("Metadata:")
                logger.info("  Confidence: \(String(format: "%.3f", result.confidence))")
                logger.info("  Duration: \(String(format: "%.3f", result.duration))s")
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
                    let startTime = tokenTimings.first?.startTime ?? 0.0
                    let endTime = tokenTimings.last?.endTime ?? result.duration
                    logger.info("  Start time: \(String(format: "%.3f", startTime))s")
                    logger.info("  End time: \(String(format: "%.3f", endTime))s")
                    logger.info("Token Timings:")
                    for (index, timing) in tokenTimings.enumerated() {
                        logger.info(
                            "    [\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                        )
                    }
                } else {
                    logger.info("  Start time: 0.000s")
                    logger.info("  End time: \(String(format: "%.3f", result.duration))s")
                    logger.info("  Token timings: Not available")
                }
            }

            let rtfx = duration / processingTime

            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", processingTime))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            if !showMetadata {
                logger.info("  Confidence: \(String(format: "%.3f", result.confidence))")
            }

            if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty {
                let debugDump = tokenTimings.enumerated().map { index, timing in
                    let start = String(format: "%.3f", timing.startTime)
                    let end = String(format: "%.3f", timing.endTime)
                    let confidence = String(format: "%.3f", timing.confidence)
                    return
                        "[\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(start)s, end: \(end)s, conf: \(confidence))"
                }.joined(separator: ", ")
                logger.debug("Token timings (count: \(tokenTimings.count)): \(debugDump)")
            }

            // Cleanup
            asrManager.cleanup()

        } catch {
            logger.error("Batch transcription failed: \(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(
        audioFile: String, showMetadata: Bool, wordTimestamps: Bool, outputJsonPath: String?,
        modelVersion: AsrModelVersion
    ) async {
        // Use optimized streaming configuration
        let config = StreamingAsrConfig.streaming

        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)

        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)

            // Start the engine with the models
            try await streamingAsr.start(models: models)

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                logger.error("Failed to create audio buffer")
                return
            }

            try audioFileHandle.read(into: buffer)

            // Calculate streaming parameters - align with StreamingAsrConfig chunk size
            let chunkDuration = config.chunkSeconds  // Use same chunk size as streaming config
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            // Track transcription updates
            let tracker = TranscriptionTracker()

            // Listen for updates in real-time
            let updateTask = Task {
                let timestampFormatter: DateFormatter = {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "HH:mm:ss.SSS"
                    return formatter
                }()

                for await update in await streamingAsr.transcriptionUpdates {
                    await tracker.record(update: update)

                    // Debug: show transcription updates
                    let updateType = update.isConfirmed ? "CONFIRMED" : "VOLATILE"
                    if showMetadata {
                        let timestampString = timestampFormatter.string(from: update.timestamp)
                        let timingSummary = streamingTimingSummary(for: update)
                        logger.info(
                            "[\(updateType)] '\(update.text)' (conf: \(String(format: "%.3f", update.confidence)), timestamp: \(timestampString))"
                        )
                        logger.info("  \(timingSummary)")
                        if !update.tokenTimings.isEmpty {
                            for (index, timing) in update.tokenTimings.enumerated() {
                                logger.info(
                                    "    [\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                                )
                            }
                        }
                    } else {
                        logger.info(
                            "[\(updateType)] '\(update.text)' (conf: \(String(format: "%.2f", update.confidence)))")
                    }

                    if update.isConfirmed {
                        await tracker.addConfirmedUpdate(update.text)
                    } else {
                        await tracker.addVolatileUpdate(update.text)
                    }
                }
            }

            // Stream audio chunks continuously - no artificial delays
            var position = 0

            logger.info("Streaming audio continuously (no artificial delays)...")
            logger.info(
                "Using \(String(format: "%.1f", chunkDuration))s chunks with \(String(format: "%.1f", config.leftContextSeconds))s left context, \(String(format: "%.1f", config.rightContextSeconds))s right context"
            )
            logger.info("Watch for real-time hypothesis updates being replaced by confirmed text\n")

            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerChunk, remainingSamples)

                // Create a chunk buffer
                guard
                    let chunkBuffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(chunkSize)
                    )
                else {
                    break
                }

                // Copy samples to chunk
                for channel in 0..<Int(format.channelCount) {
                    if let sourceData = buffer.floatChannelData?[channel],
                        let destData = chunkBuffer.floatChannelData?[channel]
                    {
                        for i in 0..<chunkSize {
                            destData[i] = sourceData[position + i]
                        }
                    }
                }
                chunkBuffer.frameLength = AVAudioFrameCount(chunkSize)

                // Update audio time position in tracker
                let audioTimePosition = Double(position) / format.sampleRate
                await tracker.updateAudioPosition(audioTimePosition)

                // Stream the chunk immediately - no waiting
                await streamingAsr.streamAudio(chunkBuffer)

                position += chunkSize

                // Small yield to allow other tasks to progress
                await Task.yield()
            }

            // Allow brief time for final processing
            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds

            // Finalize transcription
            let finalText = try await streamingAsr.finish()

            // Cancel update task
            updateTask.cancel()

            // Show final results with actual processing performance
            let processingTime = await tracker.getElapsedProcessingTime()
            let finalRtfx = processingTime > 0 ? totalDuration / processingTime : 0

            logger.info("" + String(repeating: "=", count: 50))
            logger.info("STREAMING TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(finalText)

            if let outputJsonPath = outputJsonPath {
                let snapshot = await tracker.metadataSnapshot()
                let wordTimings = WordTimingMerger.mergeTokensIntoWords(snapshot?.timings ?? [])
                let latestUpdate = await tracker.latestUpdateSnapshot()
                let modelVersionLabel = modelVersion == .v2 ? "v2" : "v3"
                let output = TranscriptionJSONOutput(
                    audioFile: audioFile,
                    mode: "streaming",
                    modelVersion: modelVersionLabel,
                    text: finalText,
                    durationSeconds: totalDuration,
                    processingTimeSeconds: processingTime,
                    rtfx: Float(finalRtfx),
                    confidence: latestUpdate?.confidence,
                    wordTimings: wordTimings,
                    timingsConfirmed: snapshot?.isConfirmed
                )
                try writeJsonOutput(output, to: outputJsonPath)
                logger.info("💾 JSON results saved to: \(outputJsonPath)")
            }

            // Print word-level timestamps if requested
            if wordTimestamps {
                if let snapshot = await tracker.metadataSnapshot() {
                    let wordTimings = WordTimingMerger.mergeTokensIntoWords(snapshot.timings)
                    logger.info("\nWord-level timestamps:")
                    for (index, word) in wordTimings.enumerated() {
                        logger.info(
                            "  [\(index)] \(String(format: "%.3f", word.startTime))s - \(String(format: "%.3f", word.endTime))s: \"\(word.word)\" (conf: \(String(format: "%.3f", word.confidence)))"
                        )
                    }
                } else {
                    logger.info("\nWord-level timestamps: Not available (no token timings)")
                }
            }

            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", totalDuration))s")
            logger.info("  Processing time: \(String(format: "%.2f", processingTime))s")
            logger.info("  RTFx: \(String(format: "%.2f", finalRtfx))x")

            if showMetadata {
                if let snapshot = await tracker.metadataSnapshot() {
                    let summaryLabel =
                        snapshot.isConfirmed
                        ? "Confirmed token timings"
                        : "Latest token timings (volatile)"
                    logger.info(summaryLabel + ":")
                    let summary = streamingTimingSummary(timings: snapshot.timings)
                    logger.info("  \(summary)")
                    for (index, timing) in snapshot.timings.enumerated() {
                        logger.info(
                            "    [\(index)] '\(timing.token)' (id: \(timing.tokenId), start: \(String(format: "%.3f", timing.startTime))s, end: \(String(format: "%.3f", timing.endTime))s, conf: \(String(format: "%.3f", timing.confidence)))"
                        )
                    }
                } else {
                    logger.info("Token timings: not available for this session")
                }
            }

        } catch {
            logger.error("Streaming transcription failed: \(error)")
        }
    }

    private static func streamingTimingSummary(for update: StreamingTranscriptionUpdate) -> String {
        streamingTimingSummary(timings: update.tokenTimings)
    }

    private static func streamingTimingSummary(timings: [TokenTiming]) -> String {
        guard !timings.isEmpty else {
            return "Token timings: none"
        }

        let start = timings.map(\.startTime).min() ?? 0
        let end = timings.map(\.endTime).max() ?? start
        let tokenCount = timings.count
        let startText = String(format: "%.3f", start)
        let endText = String(format: "%.3f", end)

        let preview = timings.map(\.token).prefix(6)
        let previewText =
            preview.isEmpty ? "n/a" : preview.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        let ellipsis = timings.count > preview.count ? "…" : ""

        return
            "Token timings: count=\(tokenCount), start=\(startText)s, end=\(endText)s, preview='\(previewText)\(ellipsis)'"
    }

    private static func printUsage() {
        let logger = AppLogger(category: "Transcribe")
        logger.info(
            """

            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]

            Options:
                --help, -h         Show this help message
                --streaming        Use streaming mode with chunk simulation
                --metadata         Show confidence, start time, and end time in results
                --word-timestamps  Show word-level timestamps for each word in the transcription
                --output-json <file>  Save full transcription result to JSON (includes word timings)
                --model-version <version>  ASR model version to use: v2 or v3 (default: v3)

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe audio.wav --metadata         # Batch mode with metadata
                fluidaudio transcribe audio.wav --word-timestamps  # Batch mode with word timestamps
                fluidaudio transcribe audio.wav --streaming --metadata # Streaming mode with metadata
                fluidaudio transcribe audio.wav --output-json results.json

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Simulates real-time streaming with chunk processing
            - Shows incremental transcription updates
            - Uses StreamingAsrManager with sliding window processing

            Metadata option:
            - Shows confidence score for transcription accuracy
            - Batch mode: Shows duration and token-based start/end times (if available)
            - Streaming mode: Shows timestamps for each transcription update
            - Works with both batch and streaming modes

            Word timestamps option:
            - Shows start and end times for each word in the transcription
            - Merges subword tokens into complete words with timing information
            - Displays confidence scores for each word
            - Works with both batch and streaming modes

            Output JSON option:
            - Saves transcription output and timings to the specified JSON file
            - Includes merged word-level timings
            """
        )
    }

    private static func writeJsonOutput(_ output: TranscriptionJSONOutput, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
    }
}
#endif
