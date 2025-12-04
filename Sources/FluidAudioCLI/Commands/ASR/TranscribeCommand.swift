#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Terminal color codes for ANSI output
enum TerminalColor {
    static let green = "\u{001B}[32m"  // Confirmed text
    static let purple = "\u{001B}[35m"  // Volatile text (magenta)
    static let reset = "\u{001B}[0m"  // Reset color

    static var enabled: Bool {
        ProcessInfo.processInfo.environment["TERM"] != nil
    }
}

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

/// Word-level timing information
struct WordTiming: Sendable {
    let word: String
    let startTime: TimeInterval
    let endTime: TimeInterval
    let confidence: Float
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
        // Check for help flag first
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

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
        var modelVersion: AsrModelVersion = .v3  // Default to v3
        var realtimeChunkMs: Int = 500  // Default 500ms chunks for realistic streaming
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
            case "--word-timestamps":
                wordTimestamps = true
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
            case "--realtime-chunk-size":
                if i + 1 < arguments.count {
                    let sizeStr = arguments[i + 1].lowercased()
                    if let ms = Int(sizeStr.replacingOccurrences(of: "ms", with: "")) {
                        realtimeChunkMs = max(10, min(5000, ms))  // Clamp to 10ms-5000ms
                    } else {
                        logger.error("Invalid chunk size: \(arguments[i + 1]). Use format like '500ms'")
                        exit(1)
                    }
                    i += 1
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
        // Supports both JSON format and simple text format (auto-detected by extension)
        if let vocabPath = customVocabularyPath {
            do {
                let url = URL(fileURLWithPath: vocabPath)
                let isJson = url.pathExtension.lowercased() == "json"

                if isJson {
                    customVocabulary = try CustomVocabularyContext.loadWithSentencePieceTokenization(from: url)
                    logger.info(
                        "Loaded custom vocabulary (JSON) from \(vocabPath) (terms: \(customVocabulary?.terms.count ?? 0))"
                    )
                } else {
                    // Simple text format: one word per line, optionally "word: alias1, alias2, ..."
                    customVocabulary = try CustomVocabularyContext.loadFromSimpleFormatWithTokenization(from: url)
                    logger.info(
                        "Loaded custom vocabulary (text) from \(vocabPath) (terms: \(customVocabulary?.terms.count ?? 0))"
                    )
                }
            } catch {
                logger.error("Failed to load custom vocabulary at \(vocabPath): \(error.localizedDescription)")
                exit(1)
            }
        }

        if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with \(realtimeChunkMs)ms chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile,
                showMetadata: showMetadata,
                wordTimestamps: wordTimestamps,
                modelVersion: modelVersion,
                realtimeChunkMs: realtimeChunkMs,
                customWords: customWords,
                customVocabulary: customVocabulary
            )
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(
                audioFile: audioFile,
                showMetadata: showMetadata,
                wordTimestamps: wordTimestamps,
                modelVersion: modelVersion,
                customWords: customWords,
                customVocabulary: customVocabulary
            )
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String,
        showMetadata: Bool,
        wordTimestamps: Bool,
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

            let samples = try AudioConverter().resampleBuffer(buffer)
            let duration = Double(audioFileHandle.length) / format.sampleRate
            logger.info(
                "Processing \(String(format: "%.2f", duration))s of audio (\(samples.count) samples)\n")

            // Process with ASR Manager
            logger.info("Transcribing file: \(audioFileURL) ...")
            let startTime = Date()
            let baseResult = try await asrManager.transcribe(
                audioFileURL, customVocabulary: customVocabulary)
            let processingTime = Date().timeIntervalSince(startTime)

            var result = baseResult

            // Note: CTC keyword boosting is already applied internally by AsrManager.transcribe()
            // when customVocabulary is provided. No need to call applyCtcKeywordBoostIfNeeded again.

            if !customWords.isEmpty {
                let rewritten = rewrite(text: result.text, using: customWords)
                result = ASRResult(
                    text: rewritten,
                    confidence: result.confidence,
                    duration: result.duration,
                    processingTime: result.processingTime,
                    tokenTimings: result.tokenTimings,
                    performanceMetrics: result.performanceMetrics,
                    ctcDetectedTerms: result.ctcDetectedTerms,
                    ctcAppliedTerms: result.ctcAppliedTerms
                )
            }

            // Print results
            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

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

    /// Test streaming transcription
    private static func testStreamingTranscription(
        audioFile: String,
        showMetadata: Bool,
        wordTimestamps: Bool,
        modelVersion: AsrModelVersion,
        realtimeChunkMs: Int,
        customWords: [String],
        customVocabulary: CustomVocabularyContext?
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

            // Enable custom vocabulary for CTC keyword boosting in streaming mode
            if let vocabulary = customVocabulary {
                await streamingAsr.setCustomVocabulary(vocabulary)
                logger.info("Custom vocabulary enabled for streaming with \(vocabulary.terms.count) terms")
            }

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

            // Calculate streaming parameters - use requested chunk size for simulation
            let chunkDurationSeconds = Double(realtimeChunkMs) / 1000.0
            let samplesPerChunk = Int(chunkDurationSeconds * format.sampleRate)
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

                    // Color-coded output: green = confirmed, purple = volatile
                    let color = update.isConfirmed ? TerminalColor.green : TerminalColor.purple
                    let coloredText =
                        TerminalColor.enabled ? "\(color)\(update.text)\(TerminalColor.reset)" : update.text

                    if showMetadata {
                        let timestampString = timestampFormatter.string(from: update.timestamp)
                        print(
                            "\(coloredText) (conf: \(String(format: "%.3f", update.confidence)), timestamp: \(timestampString))"
                        )
                    } else {
                        print(
                            "\(coloredText) (conf: \(String(format: "%.2f", update.confidence)))")
                    }

                    if update.isConfirmed {
                        await tracker.addConfirmedUpdate(update.text)
                    } else {
                        await tracker.addVolatileUpdate(update.text)
                    }
                }
            }

            // Stream audio chunks with real-time simulation
            var position = 0

            logger.info("Streaming audio with real-time simulation (\(realtimeChunkMs)ms chunks)...")
            logger.info("Waiting \(realtimeChunkMs)ms between chunks to simulate real-time audio arrival")
            logger.info("Purple text = volatile (awaiting validation), Green text = confirmed by LocalAgreement-2\n")

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

                // Stream the chunk
                await streamingAsr.streamAudio(chunkBuffer)

                position += chunkSize

                // Simulate real-time audio arrival by waiting chunk duration
                let chunkDurationNanoseconds = UInt64(chunkDurationSeconds * 1_000_000_000)
                try await Task.sleep(nanoseconds: chunkDurationNanoseconds)
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
        print(
            """

            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --streaming             Use streaming mode with chunk simulation
                --metadata              Show confidence, start time, and end time in results
                --word-timestamps       Show word-level timestamps for each word in the transcription
                --model-version <ver>   ASR model version to use: v2 or v3 (default: v3)
                --realtime-chunk-size   Size of chunks to simulate real-time streaming (default: 500ms)
                <size>                  Format: e.g., "500ms", "100ms", "2000ms" (range: 10ms-5000ms)
                --custom-vocab <path>   Custom vocabulary for CTC keyword boosting (batch only)
                                        Supports JSON format (.json) or simple text format (.txt):
                                        - JSON: {"terms": [{"text": "word", "aliases": ["alias1"]}]}
                                        - Text: One word per line, optionally "word: alias1, alias2, ..."
                --custom-words <path>   Newline-delimited custom words to replace <unk> tokens (post-process only)

            Examples:
                fluidaudio transcribe audio.wav                           # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming               # Streaming mode with 500ms chunks
                fluidaudio transcribe audio.wav --streaming --metadata    # Streaming with metadata
                fluidaudio transcribe audio.wav --streaming --realtime-chunk-size 100ms   # Small chunks (more realistic)
                fluidaudio transcribe audio.wav --streaming --realtime-chunk-size 2000ms  # Larger chunks

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Immediate chunk-by-chunk processing using VAD-style API
            - Processes chunks as they arrive without buffering (true streaming)
            - Shows incremental transcription updates using LocalAgreement-2 validation
            - Color-coded output: Purple = provisional (awaiting validation), Green = confirmed
            - Confirmed text grows as LocalAgreement-2 validates tokens across chunks
            - Stateful decoder maintains context across chunks for better continuity
            - Default 500ms chunks simulate realistic microphone input at real-time speed

            Realtime chunk size:
            - Simulates how audio arrives from a microphone (e.g., 500ms at a time)
            - Smaller chunks (100-300ms) more closely simulate real microphones
            - Larger chunks (1000-5000ms) reduce processing frequency
            - Each chunk waits for its duration before the next arrives (e.g., 500ms wait for 500ms chunk)

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
            """
        )
    }
}
#endif
