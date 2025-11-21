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
        var customVocabPath: String? = nil
        var customVocab: CustomVocabularyContext? = nil
        var useCtcKeywordBoost = false

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
            case "--custom-vocab":
                if i + 1 < arguments.count {
                    customVocabPath = arguments[i + 1]
                    i += 1
                } else {
                    logger.error("Missing path after --custom-vocab")
                    exit(1)
                }
            case "--ctc-keyword-boost":
                useCtcKeywordBoost = true
            default:
                logger.warning("Warning: Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Load custom vocabulary if provided
        if let path = customVocabPath {
            do {
                let url = URL(fileURLWithPath: path)
                customVocab = try CustomVocabularyContext.load(from: url)
                logger.info(
                    "Loaded custom vocabulary: \(customVocab!.terms.count) terms, alpha=\(String(format: "%.2f", customVocab!.alpha))"
                )
            } catch {
                logger.error("Failed to load custom vocabulary at \(path): \(error.localizedDescription)")
                exit(1)
            }
        }

        if streamingMode {
            logger.info(
                "Streaming mode enabled: simulating real-time audio with 1-second chunks.\n"
            )
            await testStreamingTranscription(
                audioFile: audioFile, showMetadata: showMetadata, modelVersion: modelVersion,
                customVocabulary: customVocab)
        } else {
            logger.info("Using batch mode with direct processing\n")
            await testBatchTranscription(
                audioFile: audioFile, showMetadata: showMetadata, modelVersion: modelVersion,
                customVocabulary: customVocab, useCtcKeywordBoost: useCtcKeywordBoost)
        }
    }

    /// Test batch transcription using AsrManager directly
    private static func testBatchTranscription(
        audioFile: String,
        showMetadata: Bool,
        modelVersion: AsrModelVersion,
        customVocabulary: CustomVocabularyContext?,
        useCtcKeywordBoost: Bool
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
            let baseResult = try await asrManager.transcribe(audioFileURL)
            let processingTime = Date().timeIntervalSince(startTime)

            // Optionally apply CTC-based keyword boosting when a custom vocabulary is provided.
            let result: ASRResult
            if useCtcKeywordBoost, let vocab = customVocabulary {
                result = await applyCtcKeywordBoostIfNeeded(
                    samples: samples,
                    baseResult: baseResult,
                    customVocabulary: vocab
                )
            } else {
                result = baseResult
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

    /// Apply CTC keyword boosting using CTC detections as a word presence signal.
    ///
    /// Strategy: CTC acts as a detector for which keywords are present in the audio.
    /// For each detected keyword, find the most similar word(s) in the TDT transcript
    /// and replace with the canonical form, regardless of timing alignment.
    private static func applyCtcKeywordBoostIfNeeded(
        samples: [Float],
        baseResult: ASRResult,
        customVocabulary: CustomVocabularyContext,
        minScore: Float = -10.0,
        minSimilarity: Float = 0.50,
        minCombinedConfidence: Float = 0.54
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
                minScore: minScore
            )

            if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
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

            // Step 2: Build set of detected canonical keywords
            let detectedKeywords = Set(detections.map { $0.term.text.lowercased() })

            // Step 3: Split TDT transcript into words
            var words = baseResult.text.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
            guard !words.isEmpty else {
                return baseResult
            }

            if debug {
                print("[DEBUG] TDT transcript has \(words.count) words", to: &stderr)
                print("[DEBUG] Searching for best matches for \(detectedKeywords.count) detected keywords", to: &stderr)
            }

            // Step 4: For each detected keyword, find best matching word(s) in transcript
            struct Replacement {
                let wordIndex: Int
                let canonical: String
                let similarity: Float
                let ctcScore: Float
                let originalWord: String

                var combinedConfidence: Float {
                    // Normalize CTC score from [-10, -5] range to [0, 1]
                    // Better CTC scores (closer to 0) = higher confidence
                    let normalizedCtcScore = max(0, min(1, (ctcScore + 10) / 5))
                    // Weight similarity more heavily (60%) than CTC confidence (40%)
                    return 0.6 * similarity + 0.4 * normalizedCtcScore
                }
            }

            var replacements: [Replacement] = []

            for detection in detections {
                let canonical = detection.term.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !canonical.isEmpty else { continue }

                let canonicalWords = canonical.split(whereSeparator: { $0.isWhitespace }).map { String($0) }

                // Skip if canonical already exists verbatim in transcript
                if baseResult.text.range(of: canonical, options: [.caseInsensitive]) != nil {
                    if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                        print("[DEBUG] '\(canonical)' already exists verbatim, skipping", to: &stderr)
                    }
                    continue
                }

                // Find best matching word or phrase in transcript
                var bestMatch: (index: Int, similarity: Float, spanLength: Int)?

                // Try single-word matches
                for (i, word) in words.enumerated() {
                    let cleanWord = word.trimmingCharacters(in: .punctuationCharacters)

                    // For multi-word canonical, compare against first word
                    let targetWord = canonicalWords[0]
                    let similarity = characterSimilarity(cleanWord, targetWord)

                    if similarity >= minSimilarity {
                        if let existing = bestMatch {
                            if similarity > existing.similarity {
                                bestMatch = (i, similarity, canonicalWords.count)
                            }
                        } else {
                            bestMatch = (i, similarity, canonicalWords.count)
                        }
                    }
                }

                // Try multi-word matches if canonical has multiple words
                if canonicalWords.count > 1 {
                    for startIdx in 0..<words.count {
                        let endIdx = min(startIdx + canonicalWords.count, words.count)
                        let span = words[startIdx..<endIdx]
                        let spanText = span.map { $0.trimmingCharacters(in: .punctuationCharacters) }.joined(
                            separator: " ")

                        let similarity = characterSimilarity(spanText, canonical)

                        if similarity >= minSimilarity {
                            if let existing = bestMatch {
                                // Prefer longer spans and higher similarity
                                if span.count > existing.spanLength
                                    || (span.count == existing.spanLength && similarity > existing.similarity)
                                {
                                    bestMatch = (startIdx, similarity, span.count)
                                }
                            } else {
                                bestMatch = (startIdx, similarity, span.count)
                            }
                        }
                    }
                }

                if let match = bestMatch {
                    let originalSpan = words[match.index..<min(match.index + match.spanLength, words.count)]
                    let replacement = Replacement(
                        wordIndex: match.index,
                        canonical: canonical,
                        similarity: match.similarity,
                        ctcScore: detection.score,
                        originalWord: originalSpan.joined(separator: " ")
                    )

                    // Filter by combined confidence
                    if replacement.combinedConfidence >= minCombinedConfidence {
                        replacements.append(replacement)

                        if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                            print(
                                "[DEBUG] Match: '\(originalSpan.joined(separator: " "))' -> '\(canonical)' (similarity: \(String(format: "%.2f", match.similarity)), ctc: \(String(format: "%.2f", detection.score)), combined: \(String(format: "%.2f", replacement.combinedConfidence)))",
                                to: &stderr)
                        }
                    } else if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                        print(
                            "[DEBUG] Rejected: '\(originalSpan.joined(separator: " "))' -> '\(canonical)' (similarity: \(String(format: "%.2f", match.similarity)), ctc: \(String(format: "%.2f", detection.score)), combined: \(String(format: "%.2f", replacement.combinedConfidence)) < \(String(format: "%.2f", minCombinedConfidence)))",
                            to: &stderr)
                    }
                }
            }

            guard !replacements.isEmpty else {
                if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                    print("[DEBUG] No suitable replacements found", to: &stderr)
                }
                return baseResult
            }

            // Step 5: Sort by span length (prefer multi-word), then combined confidence, and apply non-overlapping replacements
            replacements.sort { lhs, rhs in
                let lhsSpanLength = lhs.canonical.split(whereSeparator: { $0.isWhitespace }).count
                let rhsSpanLength = rhs.canonical.split(whereSeparator: { $0.isWhitespace }).count

                // Prefer longer spans (multi-word matches)
                if lhsSpanLength != rhsSpanLength {
                    return lhsSpanLength > rhsSpanLength
                }

                // Then prefer higher similarity
                return lhs.similarity > rhs.similarity
            }

            var usedIndices = Set<Int>()
            var finalReplacements: [Replacement] = []

            for replacement in replacements {
                let spanLength = replacement.canonical.split(whereSeparator: { $0.isWhitespace }).count
                let range = replacement.wordIndex..<min(replacement.wordIndex + spanLength, words.count)

                // Check if this range overlaps with any already used indices
                let overlaps = range.contains { usedIndices.contains($0) }

                if !overlaps {
                    finalReplacements.append(replacement)
                    range.forEach { usedIndices.insert($0) }
                }
            }

            // Step 6: Apply replacements from end to start
            finalReplacements.sort { $0.wordIndex > $1.wordIndex }

            for replacement in finalReplacements {
                let canonicalWords = replacement.canonical.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
                let spanLength = canonicalWords.count
                let endIdx = min(replacement.wordIndex + spanLength, words.count)

                // Preserve capitalization if original word started with uppercase
                let originalSpan = words[replacement.wordIndex..<endIdx]
                let finalWords = zip(
                    canonicalWords,
                    originalSpan + Array(repeating: "", count: max(0, canonicalWords.count - originalSpan.count))
                ).map { canonical, original in
                    if !original.isEmpty && original.first?.isUppercase == true && canonical.first?.isLowercase == true
                    {
                        return canonical.prefix(1).uppercased() + canonical.dropFirst()
                    }
                    return canonical
                }

                words.replaceSubrange(replacement.wordIndex..<endIdx, with: finalWords)
            }

            let updatedText = words.joined(separator: " ")

            guard updatedText != baseResult.text else {
                return baseResult
            }

            if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                print("[DEBUG] Applied \(finalReplacements.count) replacements", to: &stderr)
            }

            logger.info("CTC keyword boost updated transcript from:\n\(baseResult.text)\nTO:\n\(updatedText)")

            return ASRResult(
                text: updatedText,
                confidence: baseResult.confidence,
                duration: baseResult.duration,
                processingTime: baseResult.processingTime,
                tokenTimings: baseResult.tokenTimings,
                performanceMetrics: baseResult.performanceMetrics
            )
        } catch {
            if ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1" {
                print("[DEBUG] CTC keyword boost exception: \(error)", to: &stderr)
            }
            logger.warning("CTC keyword boost failed: \(error)")
            return baseResult
        }
    }

    /// Compute character-level Levenshtein similarity with uppercase boost
    private static func characterSimilarity(_ a: String, _ b: String) -> Float {
        let aNorm = a.lowercased()
        let bNorm = b.lowercased()
        let distance = levenshteinDistance(aNorm, bNorm)
        let maxLen = max(aNorm.count, bNorm.count)
        guard maxLen > 0 else { return 1.0 }

        var similarity = 1.0 - Float(distance) / Float(maxLen)

        // Boost if both start with uppercase
        if a.first?.isUppercase == true && b.first?.isUppercase == true {
            similarity += 0.1
        }

        return min(similarity, 1.0)
    }

    /// Compute Levenshtein distance between two strings
    private static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let m = aChars.count
        let n = bChars.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        for i in 1...m {
            for j in 1...n {
                let cost = aChars[i - 1] == bChars[j - 1] ? 0 : 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
            }
        }

        return dp[m][n]
    }

    /// Test streaming transcription
    private static func testStreamingTranscription(
        audioFile: String, showMetadata: Bool, modelVersion: AsrModelVersion, customVocabulary: CustomVocabularyContext?
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
                --custom-vocab <path>      Load custom vocabulary JSON for CTC keyword boosting
                --ctc-keyword-boost        Enable CTC keyword spotting when using a custom vocab

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
