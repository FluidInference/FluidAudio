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
    private var latestUpdate: SlidingWindowTranscriptionUpdate?
    private var latestConfirmedUpdate: SlidingWindowTranscriptionUpdate?
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

    func record(update: SlidingWindowTranscriptionUpdate) {
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

    func latestUpdateSnapshot() -> SlidingWindowTranscriptionUpdate? {
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
        var customVocabPath: String?
        var modelDir: String?
        var parakeetVariant: StreamingModelVariant?

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
                    case "tdt-ctc-110m", "110m":
                        modelVersion = .tdtCtc110m
                    default:
                        logger.error(
                            "Invalid model version: \(arguments[i + 1]). Use 'v2', 'v3', or 'tdt-ctc-110m'")
                        exit(1)
                    }
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--custom-vocab":
                if i + 1 < arguments.count {
                    customVocabPath = arguments[i + 1]
                    i += 1
                }
            case "--parakeet-variant":
                if i + 1 < arguments.count {
                    guard let variant = StreamingModelVariant(rawValue: arguments[i + 1]) else {
                        let validVariants = StreamingModelVariant.allCases.map(\.rawValue).joined(
                            separator: ", ")
                        logger.error(
                            "Unknown variant: \(arguments[i + 1]). Valid: \(validVariants)")
                        exit(1)
                    }
                    parakeetVariant = variant
                    i += 1
                }
            default:
                logger.warning("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        if let variant = parakeetVariant {
            logger.info(
                "Using \(variant.displayName) via StreamingAsrManager protocol.\n"
            )
            await runWithEngine(
                audioFile: audioFile, variant: variant)
        } else if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with 1-second chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile, showMetadata: showMetadata, wordTimestamps: wordTimestamps,
                outputJsonPath: outputJsonPath, modelVersion: modelVersion, customVocabPath: customVocabPath)
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(
                audioFile: audioFile, showMetadata: showMetadata, wordTimestamps: wordTimestamps,
                outputJsonPath: outputJsonPath, modelVersion: modelVersion, customVocabPath: customVocabPath,
                modelDir: modelDir)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String, showMetadata: Bool, wordTimestamps: Bool, outputJsonPath: String?,
        modelVersion: AsrModelVersion, customVocabPath: String?, modelDir: String? = nil
    ) async {
        do {
            // Initialize ASR models
            let models: AsrModels
            if let modelDir = modelDir {
                let dir = URL(fileURLWithPath: modelDir)
                models = try await AsrModels.load(from: dir, version: modelVersion)
            } else {
                models = try await AsrModels.downloadAndLoad(version: modelVersion)
            }
            let tdtConfig = TdtConfig(blankId: modelVersion.blankId)
            let asrConfig = ASRConfig(
                tdtConfig: tdtConfig,
                encoderHiddenSize: modelVersion.encoderHiddenSize
            )
            let asrManager = AsrManager(config: asrConfig)
            try await asrManager.loadModels(models)

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
            var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
            let startTime = Date()
            var result = try await asrManager.transcribe(audioFileURL, decoderState: &decoderState)
            let processingTime = Date().timeIntervalSince(startTime)

            // Apply vocabulary rescoring if custom vocab is provided
            if let vocabPath = customVocabPath {
                logger.info("Applying vocabulary boosting from: \(vocabPath)")

                // Load vocabulary with CTC tokenization
                let (customVocab, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(from: vocabPath)
                logger.info("Loaded \(customVocab.terms.count) vocabulary terms")

                // Create CTC spotter
                let blankId = ctcModels.vocabulary.count
                let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)

                // Run CTC keyword spotting to get log probabilities
                let spotResult = try await spotter.spotKeywordsWithLogProbs(
                    audioSamples: samples,
                    customVocabulary: customVocab,
                    minScore: nil
                )

                // Create rescorer and apply constrained CTC rescoring
                let logProbs = spotResult.logProbs
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty, !logProbs.isEmpty {
                    let ctcModelDir = CtcModels.defaultCacheDirectory(for: ctcModels.variant)

                    let vocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: customVocab.terms.count)
                    let rescorerConfig = VocabularyRescorer.Config.default

                    let rescorer = try await VocabularyRescorer.create(
                        spotter: spotter,
                        vocabulary: customVocab,
                        config: rescorerConfig,
                        ctcModelDirectory: ctcModelDir
                    )

                    // Use vocabulary-size-aware parameters
                    let minSimilarity = vocabConfig.minSimilarity
                    let cbw = vocabConfig.cbw

                    let rescoreOutput = rescorer.ctcTokenRescore(
                        transcript: result.text,
                        tokenTimings: tokenTimings,
                        logProbs: logProbs,
                        frameDuration: spotResult.frameDuration,
                        cbw: cbw,
                        marginSeconds: 0.5,
                        minSimilarity: minSimilarity
                    )

                    if rescoreOutput.wasModified {
                        logger.info("Vocabulary boosting applied \(rescoreOutput.replacements.count) replacement(s)")
                        for replacement in rescoreOutput.replacements where replacement.shouldReplace {
                            logger.info(
                                "  '\(replacement.originalWord)' → '\(replacement.replacementWord ?? "")' (score: \(String(format: "%.2f", replacement.replacementScore ?? 0)))"
                            )
                        }
                        result = ASRResult(
                            text: rescoreOutput.text,
                            confidence: result.confidence,
                            duration: result.duration,
                            processingTime: result.processingTime,
                            tokenTimings: result.tokenTimings
                        )
                    } else {
                        logger.info("No vocabulary replacements made")
                    }
                }
            }

            // Print results
            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

            if let outputJsonPath = outputJsonPath {
                let wordTimings = WordTimingMerger.mergeTokensIntoWords(result.tokenTimings ?? [])
                let modelVersionLabel: String
                switch modelVersion {
                case .v2: modelVersionLabel = "v2"
                case .v3: modelVersionLabel = "v3"
                case .tdtCtc110m: modelVersionLabel = "tdt-ctc-110m"
                case .ctcZhCn: modelVersionLabel = "ctc-zh-cn"
                case .ctcJa: modelVersionLabel = "ctc-ja"
                case .tdtJa: modelVersionLabel = "tdt-ja"
                }
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
            await asrManager.cleanup()

        } catch {
            logger.error("Batch transcription failed: \(error)")
        }
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(
        audioFile: String, showMetadata: Bool, wordTimestamps: Bool, outputJsonPath: String?,
        modelVersion: AsrModelVersion, customVocabPath: String?
    ) async {
        // Use optimized streaming configuration
        let config = SlidingWindowAsrConfig.streaming

        // Create SlidingWindowAsrManager
        let streamingAsr = SlidingWindowAsrManager(config: config)

        do {
            // Initialize ASR models
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)

            // Configure vocabulary boosting if custom vocab is provided (Option 3: Hybrid Rescoring)
            if let vocabPath = customVocabPath {
                logger.info("Configuring vocabulary boosting for streaming mode from: \(vocabPath)")

                // Load vocabulary with CTC tokenization
                let (customVocab, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(from: vocabPath)
                logger.info("Loaded \(customVocab.terms.count) vocabulary terms for streaming")

                // Configure vocabulary boosting on the streaming manager
                try await streamingAsr.configureVocabularyBoosting(
                    vocabulary: customVocab,
                    ctcModels: ctcModels
                )
            }

            // Load models and start the engine
            try await streamingAsr.loadModels(models)
            try await streamingAsr.startStreaming()

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

            // Calculate streaming parameters - align with SlidingWindowAsrConfig chunk size
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
                let modelVersionLabel: String
                switch modelVersion {
                case .v2: modelVersionLabel = "v2"
                case .v3: modelVersionLabel = "v3"
                case .tdtCtc110m: modelVersionLabel = "tdt-ctc-110m"
                case .ctcZhCn: modelVersionLabel = "ctc-zh-cn"
                case .ctcJa: modelVersionLabel = "ctc-ja"
                case .tdtJa: modelVersionLabel = "tdt-ja"
                }
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

    private static func streamingTimingSummary(for update: SlidingWindowTranscriptionUpdate) -> String {
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

    /// Run transcription using the universal StreamingAsrManager protocol
    private static func runWithEngine(
        audioFile: String, variant: StreamingModelVariant
    ) async {
        do {
            let engine = variant.createManager()

            logger.info("Loading \(variant.displayName) models...")
            let loadStart = Date()
            try await engine.loadModels()
            let loadTime = Date().timeIntervalSince(loadStart)
            logger.info("Models loaded in \(String(format: "%.2f", loadTime))s")

            // Load audio file
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            else {
                logger.error("Failed to create audio buffer")
                return
            }
            try audioFileHandle.read(into: buffer)

            // Set up partial transcript callback
            await engine.setPartialTranscriptCallback { partial in
                if !partial.isEmpty {
                    logger.info("[PARTIAL] \(partial)")
                }
            }

            // Feed audio in 1-second chunks
            let samplesPerChunk = Int(format.sampleRate)  // 1 second
            let totalSamples = Int(buffer.frameLength)
            let processStart = Date()

            var offset = 0
            while offset < totalSamples {
                let remaining = totalSamples - offset
                let chunkSize = min(samplesPerChunk, remaining)
                let chunkFrameCount = AVAudioFrameCount(chunkSize)

                guard
                    let chunkBuffer = AVAudioPCMBuffer(
                        pcmFormat: format, frameCapacity: chunkFrameCount)
                else { break }

                chunkBuffer.frameLength = chunkFrameCount
                if let src = buffer.floatChannelData, let dst = chunkBuffer.floatChannelData {
                    for ch in 0..<Int(format.channelCount) {
                        dst[ch].update(from: src[ch].advanced(by: offset), count: chunkSize)
                    }
                }

                try await engine.appendAudio(chunkBuffer)
                try await engine.processBufferedAudio()
                offset += chunkSize
            }

            let transcript = try await engine.finish()
            let processingTime = Date().timeIntervalSince(processStart)
            let rtfx = totalDuration / processingTime

            logger.info("\n--- Transcription Result ---")
            logger.info("Model: \(variant.displayName)")
            logger.info("Audio: \(String(format: "%.2f", totalDuration))s")
            logger.info("Time:  \(String(format: "%.2f", processingTime))s")
            logger.info("RTFx:  \(String(format: "%.2f", rtfx))x")
            logger.info("Text:  \(transcript)")
        } catch {
            logger.error("Engine transcription failed: \(error.localizedDescription)")
        }
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
                --model-version <version>  ASR model version: v2, v3, or tdt-ctc-110m (default: v3)
                --model-dir <path>     Path to local model directory (skips download)
                --custom-vocab <file>  Apply vocabulary boosting using terms from file (batch mode only)
                --parakeet-variant <variant>  Use any Parakeet model via StreamingAsrManager protocol

            Streaming variants (for --parakeet-variant):
                parakeet-eou-160ms, parakeet-eou-320ms, parakeet-eou-1280ms,
                nemotron-560ms, nemotron-1120ms
                (TDT models use --streaming + --model-version instead)

            Examples:
                fluidaudio transcribe audio.wav                    # Batch mode (default)
                fluidaudio transcribe audio.wav --streaming        # Streaming mode
                fluidaudio transcribe audio.wav --metadata         # Batch mode with metadata
                fluidaudio transcribe audio.wav --word-timestamps  # Batch mode with word timestamps
                fluidaudio transcribe audio.wav --streaming --metadata # Streaming mode with metadata
                fluidaudio transcribe audio.wav --output-json results.json
                fluidaudio transcribe audio.wav --custom-vocab vocab.txt  # With vocabulary boosting
                fluidaudio transcribe audio.wav --parakeet-variant parakeet-eou-320ms  # Any engine via protocol

            Batch mode (default):
            - Direct processing using AsrManager for fastest results
            - Processes entire audio file at once

            Streaming mode:
            - Simulates real-time streaming with chunk processing
            - Shows incremental transcription updates
            - Uses SlidingWindowAsrManager with sliding window processing

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

            Custom vocabulary option:
            - Boosts recognition of domain-specific terms (company names, jargon, proper nouns)
            - File format: one term per line (e.g., "NVIDIA", "PyTorch", "TensorRT")
            - Uses CTC-based constrained decoding for accurate replacement
            - Works in both batch and streaming modes
            - In streaming mode, corrections appear when text is confirmed (hybrid rescoring)
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
