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

/// Command to transcribe audio files using batch, streaming, or parakeet-variant mode
enum TranscribeCommand {
    private static let logger = AppLogger(category: "Transcribe")

    // MARK: - Argument Parsing

    private struct ParsedArgs {
        var showMetadata = false
        var wordTimestamps = false
        var outputJsonPath: String?
        var modelVersion: AsrModelVersion = .v3
        var modelDir: String?
        var customVocabPath: String?
        var parakeetVariant: StreamingModelVariant?
        var language: Language?
        var encoderPrecision: ParakeetEncoderPrecision = .int8
        var melChunkContext = true
        var dualDecodeArbitration = false
        var seamGapRepair = true
        var streamingMode = false

        // Streaming mode (SlidingWindowAsrConfig)
        var chunkSeconds: TimeInterval = 11.0
        var hypothesisChunkSeconds: TimeInterval = 1.0
        var leftContextSeconds: TimeInterval = 2.0
        var rightContextSeconds: TimeInterval = 2.0
        var minConfirmationContext: TimeInterval = 10.0
        var confirmationThreshold: Double = 0.80

        // Parakeet variant mode
        var variantChunkSeconds: TimeInterval = 1.0

        // Vocabulary boosting overrides (optional; nil = auto from vocab size)
        var vocabMinSimilarity: Float?
        var vocabCbw: Float?
        var vocabMargin: Double?
        // #702 opt-in short-term over-fire controls (nil = library default = disabled).
        var vocabShortTermTaperPivot: Int?
        var vocabSpotterMinSim: Float?
        var vocabSpotterMinSimMulti: Float?
        // #724: disable the acoustic spotter-rescue pass (pre-#634 behavior).
        var vocabDisableSpotterRescue: Bool = false
    }

    private static func parseArguments(_ args: [String]) -> ParsedArgs? {
        var parsed = ParsedArgs()
        var i = 0

        while i < args.count {
            switch args[i] {
            case "--streaming":
                parsed.streamingMode = true
            case "--metadata":
                parsed.showMetadata = true
            case "--word-timestamps":
                parsed.wordTimestamps = true
            case "--output-json":
                if i + 1 < args.count {
                    parsed.outputJsonPath = args[i + 1]
                    i += 1
                }
            case "--model-version":
                if i + 1 < args.count {
                    switch args[i + 1].lowercased() {
                    case "v2", "2":
                        parsed.modelVersion = .v2
                    case "v3", "3":
                        parsed.modelVersion = .v3
                    case "tdt-ctc-110m", "110m":
                        parsed.modelVersion = .tdtCtc110m
                    default:
                        fputs(
                            "ERROR: Invalid model version: \(args[i + 1]). Use 'v2', 'v3', or 'tdt-ctc-110m'\n", stderr)
                        fflush(stderr)
                        return nil
                    }
                    i += 1
                }
            case "--model-dir":
                if i + 1 < args.count {
                    parsed.modelDir = args[i + 1]
                    i += 1
                }
            case "--custom-vocab":
                if i + 1 < args.count {
                    parsed.customVocabPath = args[i + 1]
                    i += 1
                }
            case "--parakeet-variant":
                if i + 1 < args.count {
                    guard let variant = StreamingModelVariant(rawValue: args[i + 1]) else {
                        let validVariants = StreamingModelVariant.allCases.map(\.rawValue).joined(separator: ", ")
                        fputs("ERROR: Unknown variant: \(args[i + 1]). Valid: \(validVariants)\n", stderr)
                        fflush(stderr)
                        return nil
                    }
                    parsed.parakeetVariant = variant
                    i += 1
                }
            case "--language":
                if i + 1 < args.count {
                    guard let lang = Language(rawValue: args[i + 1].lowercased()) else {
                        let valid = Language.allCases.map(\.rawValue).joined(separator: ", ")
                        fputs("ERROR: Unknown language: \(args[i + 1]). Valid: \(valid)\n", stderr)
                        fflush(stderr)
                        return nil
                    }
                    parsed.language = lang
                    i += 1
                }
            case "--encoder-precision":
                if i + 1 < args.count {
                    guard let precision = ParakeetEncoderPrecision(rawValue: args[i + 1].lowercased()) else {
                        let valid = ParakeetEncoderPrecision.allCases.map(\.rawValue).joined(separator: ", ")
                        fputs("ERROR: Unknown encoder precision: \(args[i + 1]). Valid: \(valid)\n", stderr)
                        fflush(stderr)
                        return nil
                    }
                    parsed.encoderPrecision = precision
                    i += 1
                }
            case "--no-mel-context":
                parsed.melChunkContext = false
            case "--dual-decode-arbitration":
                parsed.dualDecodeArbitration = true
            case "--no-seam-gap-repair":
                parsed.seamGapRepair = false

            // Streaming mode config
            case "--chunk-seconds":
                if i + 1 < args.count {
                    parsed.chunkSeconds = TimeInterval(args[i + 1]) ?? 11.0
                    i += 1
                }
            case "--hypothesis-chunk-seconds":
                if i + 1 < args.count {
                    parsed.hypothesisChunkSeconds = TimeInterval(args[i + 1]) ?? 1.0
                    i += 1
                }
            case "--left-context-seconds":
                if i + 1 < args.count {
                    parsed.leftContextSeconds = TimeInterval(args[i + 1]) ?? 2.0
                    i += 1
                }
            case "--right-context-seconds":
                if i + 1 < args.count {
                    parsed.rightContextSeconds = TimeInterval(args[i + 1]) ?? 2.0
                    i += 1
                }
            case "--min-confirmation-context":
                if i + 1 < args.count {
                    parsed.minConfirmationContext = TimeInterval(args[i + 1]) ?? 10.0
                    i += 1
                }
            case "--confirmation-threshold":
                if i + 1 < args.count {
                    parsed.confirmationThreshold = Double(args[i + 1]) ?? 0.80
                    i += 1
                }

            // Parakeet variant mode config
            case "--variant-chunk-seconds":
                if i + 1 < args.count {
                    parsed.variantChunkSeconds = TimeInterval(args[i + 1]) ?? 1.0
                    i += 1
                }

            // Vocabulary boosting overrides
            case "--vocab-min-similarity":
                if i + 1 < args.count {
                    parsed.vocabMinSimilarity = Float(args[i + 1])
                    i += 1
                }
            case "--vocab-cbw":
                if i + 1 < args.count {
                    parsed.vocabCbw = Float(args[i + 1])
                    i += 1
                }
            case "--vocab-margin":
                if i + 1 < args.count {
                    parsed.vocabMargin = Double(args[i + 1])
                    i += 1
                }
            case "--vocab-short-term-taper-pivot":
                if i + 1 < args.count {
                    parsed.vocabShortTermTaperPivot = Int(args[i + 1])
                    i += 1
                }
            case "--vocab-spotter-min-sim":
                if i + 1 < args.count {
                    parsed.vocabSpotterMinSim = Float(args[i + 1])
                    i += 1
                }
            case "--vocab-spotter-min-sim-multi":
                if i + 1 < args.count {
                    parsed.vocabSpotterMinSimMulti = Float(args[i + 1])
                    i += 1
                }
            case "--vocab-disable-spotter-rescue":
                parsed.vocabDisableSpotterRescue = true

            default:
                fputs("WARNING: Unknown option: \(args[i])\n", stderr)
                fflush(stderr)
            }
            i += 1
        }

        return parsed
    }

    // MARK: - Run Dispatcher

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        guard let audioFile = arguments.first, !audioFile.isEmpty else {
            fputs("ERROR: No audio file specified\n", stderr)
            fflush(stderr)
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        guard let parsed = parseArguments(Array(arguments.dropFirst())) else {
            exit(1)
        }

        if let variant = parsed.parakeetVariant {
            logger.info("Using \(variant.displayName) via StreamingAsrManager protocol.\n")
            await runWithVariant(audioFile: audioFile, args: parsed)
        } else if parsed.streamingMode {
            logger.info("Streaming mode enabled: simulating real-time audio with 1-second chunks.\n")
            await runStreaming(audioFile: audioFile, args: parsed)
        } else {
            logger.info("Using batch mode with direct processing\n")
            await runBatch(audioFile: audioFile, args: parsed)
        }
    }

    // MARK: - Batch Mode

    private static func runBatch(
        audioFile: String, args: ParsedArgs
    ) async {
        do {
            let models: AsrModels
            if let modelDir = args.modelDir {
                let dir = URL(fileURLWithPath: modelDir)
                models = try await AsrModels.load(
                    from: dir, version: args.modelVersion, encoderPrecision: args.encoderPrecision)
            } else {
                models = try await AsrModels.downloadAndLoad(
                    version: args.modelVersion, encoderPrecision: args.encoderPrecision)
            }
            let tdtConfig = TdtConfig(blankId: args.modelVersion.blankId)
            let asrConfig = ASRConfig(
                tdtConfig: tdtConfig,
                encoderHiddenSize: args.modelVersion.encoderHiddenSize,
                melChunkContext: args.melChunkContext,
                dualDecodeArbitration: args.dualDecodeArbitration,
                seamGapRepair: args.seamGapRepair
            )
            let asrManager = AsrManager(config: asrConfig)
            try await asrManager.loadModels(models)

            logger.info("ASR Manager initialized successfully")

            let audioFileURL = URL(fileURLWithPath: audioFile)
            logger.info("Transcribing file: \(audioFileURL) ...")
            var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
            let startTime = Date()

            let customVocabularySamples: [Float]?
            var result: ASRResult
            if args.customVocabPath != nil {
                let samples = try AudioConverter().resampleAudioFile(audioFileURL)
                customVocabularySamples = samples
                result = try await asrManager.transcribe(
                    samples,
                    decoderState: &decoderState,
                    language: args.language
                )
            } else {
                customVocabularySamples = nil
                result = try await asrManager.transcribe(
                    audioFileURL, decoderState: &decoderState, language: args.language)
            }
            let processingTime = Date().timeIntervalSince(startTime)
            let duration = result.duration
            logger.info("Processed \(String(format: "%.2f", duration))s of audio\n")

            // Apply vocabulary rescoring if custom vocab is provided
            if let vocabPath = args.customVocabPath, let samples = customVocabularySamples {
                logger.info("Applying vocabulary boosting from: \(vocabPath)")

                let (customVocab, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(from: vocabPath)
                logger.info("Loaded \(customVocab.terms.count) vocabulary terms")

                let blankId = ctcModels.vocabulary.count
                let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)

                let spotResult = try await spotter.spotKeywordsWithLogProbs(
                    audioSamples: samples,
                    customVocabulary: customVocab,
                    minScore: nil
                )

                let logProbs = spotResult.logProbs
                if let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty, !logProbs.isEmpty {
                    let ctcModelDir = CtcModels.defaultCacheDirectory(for: ctcModels.variant)

                    let vocabConfig = ContextBiasingConstants.rescorerConfig(forVocabSize: customVocab.terms.count)
                    // #702: opt-in short-term over-fire controls. Each flag
                    // falls back to the library default (disabled) when unset.
                    let rescorerConfig = VocabularyRescorer.Config(
                        shortTermCbwTaperPivot: args.vocabShortTermTaperPivot
                            ?? ContextBiasingConstants.defaultShortTermCbwTaperPivot,
                        spotterRescueMinSimilarity: args.vocabSpotterMinSim
                            ?? ContextBiasingConstants.defaultSpotterRescueMinSimilarity,
                        spotterRescueMultiWordMinSimilarity: args.vocabSpotterMinSimMulti
                            ?? ContextBiasingConstants.defaultSpotterRescueMultiWordMinSimilarity,
                        // #724: `--vocab-disable-spotter-rescue` forces the acoustic
                        // rescue pass off; otherwise follow the library/env default
                        // (on unless `FLUID_SPOTTER_RESCUE` disables it).
                        spotterRescueEnabled: args.vocabDisableSpotterRescue
                            ? false : ContextBiasingConstants.defaultSpotterRescueEnabled
                    )

                    let rescorer = try await VocabularyRescorer.create(
                        spotter: spotter,
                        vocabulary: customVocab,
                        config: rescorerConfig,
                        ctcModelDirectory: ctcModelDir
                    )

                    // Vocabulary-size-aware default, overridable via CLI.
                    // Per-term minSimilarity overrides are applied inside the
                    // rescorer regardless of this global value.
                    let minSimilarity: Float = args.vocabMinSimilarity ?? vocabConfig.minSimilarity
                    let cbw: Float = args.vocabCbw ?? vocabConfig.cbw
                    let marginSeconds: Double = args.vocabMargin ?? ContextBiasingConstants.defaultMarginSeconds

                    let rescoreOutput = rescorer.ctcTokenRescore(
                        transcript: result.text,
                        tokenTimings: tokenTimings,
                        logProbs: logProbs,
                        frameDuration: spotResult.frameDuration,
                        cbw: cbw,
                        marginSeconds: marginSeconds,
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

            logger.info("" + String(repeating: "=", count: 50))
            logger.info("BATCH TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(result.text)

            if let outputJsonPath = args.outputJsonPath {
                let wordTimings = WordTimingMerger.mergeTokensIntoWords(result.tokenTimings ?? [])
                let modelVersionLabel: String
                switch args.modelVersion {
                case .v2: modelVersionLabel = "v2"
                case .v3: modelVersionLabel = "v3"
                case .tdtCtc110m: modelVersionLabel = "tdt-ctc-110m"
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

            if args.wordTimestamps {
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

            if args.showMetadata {
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
            if !args.showMetadata {
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

            await asrManager.cleanup()

        } catch {
            logger.error("Batch transcription failed: \(error)")
        }
    }

    // MARK: - Streaming Mode

    private static func runStreaming(
        audioFile: String, args: ParsedArgs
    ) async {
        let config = SlidingWindowAsrConfig(
            chunkSeconds: args.chunkSeconds,
            hypothesisChunkSeconds: args.hypothesisChunkSeconds,
            leftContextSeconds: args.leftContextSeconds,
            rightContextSeconds: args.rightContextSeconds,
            minContextForConfirmation: args.minConfirmationContext,
            confirmationThreshold: args.confirmationThreshold
        )

        let streamingAsr = SlidingWindowAsrManager(config: config)

        do {
            // Pass encoder precision + model dir to model loading when available
            let models: AsrModels
            if let modelDir = args.modelDir {
                let dir = URL(fileURLWithPath: modelDir)
                models = try await AsrModels.load(
                    from: dir, version: args.modelVersion, encoderPrecision: args.encoderPrecision)
            } else {
                models = try await AsrModels.downloadAndLoad(
                    version: args.modelVersion, encoderPrecision: args.encoderPrecision)
            }

            if let vocabPath = args.customVocabPath {
                logger.info("Configuring vocabulary boosting for streaming mode from: \(vocabPath)")

                let (customVocab, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(from: vocabPath)
                logger.info("Loaded \(customVocab.terms.count) vocabulary terms for streaming")

                try await streamingAsr.configureVocabularyBoosting(
                    vocabulary: customVocab,
                    ctcModels: ctcModels
                )
            }

            try await streamingAsr.loadModels(models)
            try await streamingAsr.startStreaming()

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

            let chunkDuration = config.chunkSeconds
            let samplesPerChunk = Int(chunkDuration * format.sampleRate)
            let totalDuration = Double(audioFileHandle.length) / format.sampleRate

            let tracker = TranscriptionTracker()

            let updateTask = Task {
                let timestampFormatter: DateFormatter = {
                    let formatter = DateFormatter()
                    formatter.dateFormat = "HH:mm:ss.SSS"
                    return formatter
                }()

                for await update in await streamingAsr.transcriptionUpdates {
                    await tracker.record(update: update)

                    let updateType = update.isConfirmed ? "CONFIRMED" : "VOLATILE"
                    if args.showMetadata {
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

            var position = 0

            logger.info("Streaming audio continuously (no artificial delays)...")
            logger.info(
                "Using \(String(format: "%.1f", chunkDuration))s chunks with \(String(format: "%.1f", config.leftContextSeconds))s left context, \(String(format: "%.1f", config.rightContextSeconds))s right context"
            )
            logger.info("Watch for real-time hypothesis updates being replaced by confirmed text\n")

            while position < Int(buffer.frameLength) {
                let remainingSamples = Int(buffer.frameLength) - position
                let chunkSize = min(samplesPerChunk, remainingSamples)

                guard
                    let chunkBuffer = AVAudioPCMBuffer(
                        pcmFormat: format,
                        frameCapacity: AVAudioFrameCount(chunkSize)
                    )
                else {
                    break
                }

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

                let audioTimePosition = Double(position) / format.sampleRate
                await tracker.updateAudioPosition(audioTimePosition)

                await streamingAsr.streamAudio(chunkBuffer)

                position += chunkSize

                await Task.yield()
            }

            try await Task.sleep(nanoseconds: 500_000_000)

            let finalText = try await streamingAsr.finish()

            updateTask.cancel()

            let processingTime = await tracker.getElapsedProcessingTime()
            let finalRtfx = processingTime > 0 ? totalDuration / processingTime : 0

            logger.info("" + String(repeating: "=", count: 50))
            logger.info("STREAMING TRANSCRIPTION RESULTS")
            logger.info(String(repeating: "=", count: 50))
            logger.info("Final transcription:")
            print(finalText)

            if let outputJsonPath = args.outputJsonPath {
                let snapshot = await tracker.metadataSnapshot()
                let wordTimings = WordTimingMerger.mergeTokensIntoWords(snapshot?.timings ?? [])
                let latestUpdate = await tracker.latestUpdateSnapshot()
                let modelVersionLabel: String
                switch args.modelVersion {
                case .v2: modelVersionLabel = "v2"
                case .v3: modelVersionLabel = "v3"
                case .tdtCtc110m: modelVersionLabel = "tdt-ctc-110m"
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

            if args.wordTimestamps {
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

            if args.showMetadata {
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

    // MARK: - Helpers

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

    // MARK: - Parakeet Variant Mode

    private static func runWithVariant(
        audioFile: String, args: ParsedArgs
    ) async {
        guard let variant = args.parakeetVariant else { return }
        do {
            let engine = variant.createManager()

            logger.info("Loading \(variant.displayName) models...")
            let loadStart = Date()
            try await engine.loadModels()
            let loadTime = Date().timeIntervalSince(loadStart)
            logger.info("Models loaded in \(String(format: "%.2f", loadTime))s")

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

            await engine.setPartialTranscriptCallback { partial in
                if !partial.isEmpty {
                    logger.info("[PARTIAL] \(partial)")
                }
            }

            let samplesPerChunk = Int(Double(format.sampleRate) * args.variantChunkSeconds)
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

    // MARK: - Usage

    private static func printUsage() {
        let usage = """
            Transcribe Command Usage:
                fluidaudio transcribe <audio_file> [options]

            COMMON OPTIONS:
                --help, -h                     Show this help message
                --streaming                    Use streaming mode with chunk simulation
                --metadata                     Show confidence, start time, and end time
                --word-timestamps              Show word-level timestamps in results
                --output-json <file>           Save full transcription to JSON
                --model-version <v2|v3|110m>   ASR model version (default: v3)
                --model-dir <path>             Local model directory (skips download)
                --encoder-precision <int8|int4> Encoder quantization (default: int8)
                --language <code>              Language hint (e.g., en, de, fr, es)
                --custom-vocab <file>          Apply vocabulary boosting in batch mode
                --no-mel-context               Disable 80ms mel-context prepend for long-form batch ASR
                --dual-decode-arbitration      Enable v3/no-mel long-form boundary arbitration

            STREAMING MODE OPTIONS (--streaming, SlidingWindowAsrManager):
                --chunk-seconds <sec>                Audio chunk size (default: 11.0)
                --hypothesis-chunk-seconds <sec>     Hypothesis update interval (default: 1.0)
                --left-context-seconds <sec>         Left context per window (default: 2.0)
                --right-context-seconds <sec>        Right context lookahead (default: 2.0)
                --min-confirmation-context <sec>     Min audio before confirming text (default: 10.0)
                --confirmation-threshold <0-1>       Confidence to promote volatile→confirmed (default: 0.80)

            VOCABULARY BOOSTING OPTIONS (--custom-vocab):
                --custom-vocab <file>            Vocabulary terms file (one per line)
                --vocab-min-similarity <0-1>     Minimum string similarity for replacement (default: auto)
                --vocab-cbw <float>              Context-biasing weight boost (default: auto)
                --vocab-margin <sec>             CTC frame alignment margin (default: 0.5)

              Short-term over-fire controls (#702, opt-in; default off — these
              improve precision on SHORT keyword vocabularies at some recall
              cost on distinctive-name vocabularies):
                --vocab-short-term-taper-pivot <n>   Reduce boost for terms with < n tokens (e.g. 5)
                --vocab-spotter-min-sim <0-1>        Min similarity for acoustic single-word rescue (e.g. 0.30)
                --vocab-spotter-min-sim-multi <0-1>  Min similarity for multi-word rescue (e.g. 0.50)
                --vocab-disable-spotter-rescue       Turn the acoustic rescue OFF entirely (#724; pre-#634
                                                     behavior — biggest over-fire reduction for short vocabs)

            PARAKEET VARIANT MODE (--parakeet-variant):
                --parakeet-variant <variant>     Engine: parakeet-eou-160ms, nemotron-560ms, …
                --variant-chunk-seconds <sec>    Audio chunk size for variant engine (default: 1.0)

            Examples:
                fluidaudio transcribe audio.wav                          # Batch mode
                fluidaudio transcribe audio.wav --streaming              # Streaming mode
                fluidaudio transcribe audio.wav --streaming --chunk-seconds 5.0
                fluidaudio transcribe audio.wav --output-json out.json
                fluidaudio transcribe audio.wav --custom-vocab terms.txt
                fluidaudio transcribe audio.wav --custom-vocab terms.txt --vocab-min-similarity 0.6
                fluidaudio transcribe audio.wav --parakeet-variant parakeet-eou-320ms

            Modes:
                batch (default)       Direct AsrManager processing for fastest results
                streaming (--streaming) SlidingWindowAsrManager with real-time updates
                variant (--parakeet-variant)  StreamingAsrManager protocol engines
            """
        fputs(usage, stderr)
        fflush(stderr)
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
