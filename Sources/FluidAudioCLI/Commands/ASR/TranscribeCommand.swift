#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

// Helper for stderr output
fileprivate struct StderrOutputStream: TextOutputStream {
    func write(_ string: String) {
        if let data = string.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }
}
fileprivate var stderr = StderrOutputStream()

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

    private struct TokenKey: Hashable {
        let tokenId: Int
        let startMilliseconds: Int
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
        var modelVersion: AsrModelVersion = .v3  // Default to v3
        var customWordsPath: String?
        var customWords: [String] = []
        var customVocabularyPath: String?
        var customVocabulary: CustomVocabularyContext?

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
            case "--custom-words":
                if i + 1 < arguments.count {
                    customWordsPath = arguments[i + 1]
                    i += 1
                } else {
                    logger.error("Missing path after --custom-words")
                    exit(1)
                }
            case "--custom-vocab":
                if i + 1 < arguments.count {
                    customVocabularyPath = arguments[i + 1]
                    i += 1
                } else {
                    logger.error("Missing path after --custom-vocab")
                    exit(1)
                }
            default:
                logger.warning("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Load custom words if provided
        if let path = customWordsPath {
            do {
                let url = URL(fileURLWithPath: path)
                let contents = try String(contentsOf: url, encoding: .utf8)
                customWords =
                    contents
                    .split(whereSeparator: { $0.isNewline })
                    .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                logger.info("Loaded \(customWords.count) custom words from \(path)")
            } catch {
                logger.error("Failed to load custom words at \(path): \(error.localizedDescription)")
                exit(1)
            }
        }

        // Load structured custom vocabulary (for CTC keyword boosting) if provided
        if let vocabPath = customVocabularyPath {
            do {
                let url = URL(fileURLWithPath: vocabPath)
                customVocabulary = try CustomVocabularyContext.load(from: url)
                logger.info("Loaded custom vocabulary from \(vocabPath) (terms: \(customVocabulary?.terms.count ?? 0))")
            } catch {
                logger.error("Failed to load custom vocabulary at \(vocabPath): \(error.localizedDescription)")
                exit(1)
            }
        }

        if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with 1-second chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile, showMetadata: showMetadata, modelVersion: modelVersion,
                customWords: customWords, customVocabulary: customVocabulary)
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(
                audioFile: audioFile, showMetadata: showMetadata, modelVersion: modelVersion,
                customWords: customWords, customVocabulary: customVocabulary)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String,
        showMetadata: Bool,
        modelVersion: AsrModelVersion,
        customWords: [String],
        customVocabulary: CustomVocabularyContext?
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

            let duration = Double(audioFileHandle.length) / format.sampleRate
            logger.info("Processing \(String(format: "%.2f", duration))s of audio\n")

            // Process with ASR Manager
            logger.info("Transcribing file: \(audioFileURL) ...")
            let startTime = Date()
            let baseResult = try await asrManager.transcribe(
                audioFileURL, customVocabulary: customVocabulary)
            let processingTime = Date().timeIntervalSince(startTime)

            var result = baseResult

            if let vocabulary = customVocabulary {
                let monoSamples = extractMonoSamples(from: buffer)
                result = await applyCtcKeywordBoostIfNeeded(
                    samples: monoSamples,
                    baseResult: result,
                    customVocabulary: vocabulary
                )
            }

            if !customWords.isEmpty {
                let rewritten = rewrite(text: result.text, using: customWords)
                result = ASRResult(
                    text: rewritten,
                    confidence: result.confidence,
                    duration: result.duration,
                    processingTime: result.processingTime,
                    tokenTimings: result.tokenTimings
                )
            }

            // Print results
            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

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

    /// Simple post-processing rewrite that replaces `<unk>` placeholders with
    /// the supplied custom words (in order) if they are not already present.
    private static func rewrite(text: String, using customWords: [String]) -> String {
        guard !customWords.isEmpty else { return text }

        let lowercased = text.lowercased()
        let remaining = customWords.filter { !lowercased.contains($0.lowercased()) }
        guard !remaining.isEmpty else { return text }

        let nsText = text as NSString
        let pattern = "(?i)<unk>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }

        let mutable = NSMutableString(string: text)
        var offset = 0
        var replacementIndex = 0

        for match in regex.matches(in: text, range: NSRange(location: 0, length: nsText.length)) {
            guard replacementIndex < remaining.count else { break }
            let replacement = remaining[replacementIndex]
            replacementIndex += 1

            let adjustedRange = NSRange(
                location: match.range.location + offset,
                length: match.range.length
            )
            mutable.replaceCharacters(in: adjustedRange, with: replacement)
            offset += (replacement as NSString).length - match.range.length
        }

        return String(mutable)
    }

    /// Convert the first channel of a PCM buffer to a mono sample array.
    private static func extractMonoSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        let frameLength = Int(buffer.frameLength)
        guard
            frameLength > 0,
            let channelData = buffer.floatChannelData?[0]
        else {
            return []
        }
        return Array(UnsafeBufferPointer(start: channelData, count: frameLength))
    }

    /// Apply CTC keyword boosting using CTC detections as a word presence signal.
    ///
    /// Strategy: CTC acts as a detector for which keywords are present in the audio.
    /// For each detected keyword, find the most similar word(s) in the TDT transcript
    /// and replace with the canonical form, regardless of timing alignment.
    private static func applyCtcKeywordBoostIfNeeded(
        samples: [Float],
        baseResult: ASRResult,
        customVocabulary: CustomVocabularyContext
    ) async -> ASRResult {
        let debug = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        guard !customVocabulary.terms.isEmpty else {
            return baseResult
        }

        do {
            // Step 1: Run CTC to detect which keywords are present
            let spotter = try await CtcKeywordSpotter.makeDefault()
            let detections = try await spotter.spotKeywords(
                audioSamples: samples,
                customVocabulary: customVocabulary,
                minScore: customVocabulary.minCtcScore
            )

            if debug {
                print("[DEBUG] CTC detected \(detections.count) keywords", to: &stderr)
                for detection in detections {
                    print(
                        "[DEBUG]   '\(detection.term.text)' score=\(String(format: "%.2f", detection.score))",
                        to: &stderr)
                }
            }

            guard !detections.isEmpty else {
                return baseResult
            }

            // Step 2: Use KeywordMerger to apply corrections (string-based for now)
            let mergeResult = KeywordMerger.applyCorrections(
                detections: detections,
                toTranscript: baseResult.text,
                vocabulary: customVocabulary,
                debugMode: debug
            )

            guard mergeResult.correctedText != baseResult.text else {
                return baseResult
            }

            logger.info(
                "CTC keyword boost updated transcript from:\n\(baseResult.text)\nTO:\n\(mergeResult.correctedText)")

            return ASRResult(
                text: mergeResult.correctedText,
                confidence: baseResult.confidence,
                duration: baseResult.duration,
                processingTime: baseResult.processingTime,
                tokenTimings: baseResult.tokenTimings,
                performanceMetrics: baseResult.performanceMetrics
            )
        } catch {
            if debug {
                print("[DEBUG] CTC keyword boost exception: \(error)", to: &stderr)
            }
            logger.warning("CTC keyword boost failed: \(error)")
            return baseResult
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(
        audioFile: String,
        showMetadata: Bool,
        modelVersion: AsrModelVersion,
        customWords: [String],
        customVocabulary: CustomVocabularyContext?
    ) async {
        // Use optimized streaming configuration
        let config = StreamingAsrConfig.streaming

        // Create StreamingAsrManager
        let streamingAsr = StreamingAsrManager(config: config)

        do {
            if customVocabulary != nil {
                logger.warning("Custom vocabulary is applied only in batch mode; streaming ignores CTC boosting.")
            }
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
            let finalOutput = customWords.isEmpty ? finalText : rewrite(text: finalText, using: customWords)

            // Cancel update task
            updateTask.cancel()

            // Show final results with actual processing performance
            let processingTime = await tracker.getElapsedProcessingTime()
            let finalRtfx = processingTime > 0 ? totalDuration / processingTime : 0

            logger.info("" + String(repeating: "=", count: 50))
            logger.info("STREAMING TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(finalOutput)
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
                --model-version <version>  ASR model version to use: v2 or v3 (default: v3)
                --custom-vocab <path>     JSON custom vocabulary for CTC keyword boosting (batch only)
                --custom-words <path>      Newline-delimited custom words to replace <unk> tokens (post-process only)

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe audio.wav --metadata         # Batch mode with metadata
                fluidaudio transcribe audio.wav --streaming --metadata # Streaming mode with metadata

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
            """
        )
    }
}
#endif
