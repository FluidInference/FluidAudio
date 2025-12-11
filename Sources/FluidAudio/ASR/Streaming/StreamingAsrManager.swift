import AVFoundation
import Foundation
import OSLog

/// Represents a word that was corrected during transcription (for UI highlighting)
public struct CorrectedWord: Sendable, Equatable {
    /// Character range in the corrected/final text string
    public let range: Range<Int>
    /// The original (misspelled) word
    public let original: String
    /// The corrected (canonical) word
    public let corrected: String

    public init(range: Range<Int>, original: String, corrected: String) {
        self.range = range
        self.original = original
        self.corrected = corrected
    }
}

/// A high-level streaming ASR manager that provides a simple API for real-time transcription
/// Similar to Apple's SpeechAnalyzer, it handles audio conversion and buffering automatically
public actor StreamingAsrManager {

    // MARK: - Cached Regex Patterns
    // Relaxed to {2,} to handle long suffixes like 'ftriaxone'
    private static let splitWordPattern = try! NSRegularExpression(
        pattern: #"(\w{3,})\.\s+([a-z]{2,})(?=[,.\s]|$)"#, options: [])
    private static let noSpacePattern = try! NSRegularExpression(
        pattern: #"(\w{3,})\.([a-z]{2,})(?=[,.\s]|$)"#, options: [])

    private let logger = AppLogger(category: "StreamingASR")
    private let audioConverter: AudioConverter = AudioConverter()
    private let config: StreamingAsrConfig

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Transcription output stream
    private var updateContinuation: AsyncStream<StreamingTranscriptionUpdate>.Continuation?

    // ASR components
    private var asrManager: AsrManager?
    private var hypothesisAsrManager: AsrManager?
    private var recognizerTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone

    // Sliding window state
    private var segmentIndex: Int = 0
    private var lastProcessedFrame: Int = 0
    private var accumulatedTokens: [Int] = []
    private var hypothesisAccumulatedTokens: [Int] = []
    private var lastConfirmedAudioFrame: Int = 0  // Track confirmed audio boundary (encoder frames)
    private var lastConfirmedAudioSample: Int = 0  // Track confirmed audio boundary (samples)
    private var confirmedTokenCount: Int = 0  // Number of tokens that have been confirmed
    private let maxHypothesisTokenHistory = 96
    private let maxHypothesisOverlapTokens = 64

    // Accumulated hypothesis data since last confirmation (for proper stabilization)
    private var accumulatedHypothesisTokens: [Int] = []
    private var accumulatedHypothesisTimestamps: [Int] = []
    private var accumulatedHypothesisConfidences: [Float] = []

    // Hypothesis stabilization for reducing text fluctuation
    private var hypothesisStabilizer: HypothesisStabilizer

    // Raw sample buffer for sliding-window assembly (absolute indexing)
    private var sampleBuffer: [Float] = []
    private var bufferStartIndex: Int = 0  // absolute index of sampleBuffer[0]
    private var nextWindowCenterStart: Int = 0  // absolute index where next chunk (center) begins
    private var nextHypothesisWindowStart: Int = 0  // absolute index for hypothesis updates

    // Two-tier transcription state (like Apple's Speech API)
    public private(set) var volatileTranscript: String = ""
    public private(set) var confirmedTranscript: String = ""

    // Internal buffer for main track's unconfirmed text (decoupled from display)
    private var pendingMainTranscription: String = ""

    /// Tracks word corrections applied during streaming (for UI highlighting)
    public private(set) var accumulatedCorrections: [CorrectedWord] = []

    /// Corrections for the current chunk being processed
    private var currentChunkCorrections: [CorrectedWord] = []

    /// Corrections for pendingMainTranscription (to be accumulated when confirmed)
    private var pendingMainCorrections: [CorrectedWord] = []

    /// The audio source this stream is configured for
    public var source: AudioSource {
        return audioSource
    }

    // Metrics
    private var startTime: Date?
    private var processedChunks: Int = 0
    private var supportsHypothesisTrack: Bool {
        config.hypothesisChunkSeconds > 0 && config.hypothesisChunkSeconds < config.chunkSeconds
    }

    // Custom vocabulary and CTC keyword spotting for word boosting
    private var customVocabulary: CustomVocabularyContext?
    private var ctcKeywordSpotter: CtcKeywordSpotter?
    private var ctcModels: CtcModels?

    /// Initialize the streaming ASR manager
    /// - Parameter config: Configuration for streaming behavior
    public init(config: StreamingAsrConfig = .default) {
        self.config = config

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        // Initialize hypothesis stabilizer with config-appropriate settings
        self.hypothesisStabilizer = HypothesisStabilizer(
            config: config.hypothesisStabilizerConfig
        )

        logger.info(
            "Initialized StreamingAsrManager with config: chunk=\(config.chunkSeconds)s hypo=\(config.hypothesisChunkSeconds)s left=\(config.leftContextSeconds)s right=\(config.rightContextSeconds)s"
        )
    }

    /// Start the streaming ASR engine
    /// This will download models if needed and begin processing
    /// - Parameter source: The audio source to use (default: microphone)
    public func start(source: AudioSource = .microphone) async throws {
        logger.info("Starting streaming ASR engine for source: \(String(describing: source))...")

        // Initialize ASR models
        let models = try await AsrModels.downloadAndLoad()
        try await start(models: models, source: source)
    }

    /// Start the streaming ASR engine with pre-loaded models
    /// - Parameters:
    ///   - models: Pre-loaded ASR models to use
    ///   - source: The audio source to use (default: microphone)
    public func start(models: AsrModels, source: AudioSource = .microphone) async throws {
        logger.info(
            "Starting streaming ASR engine with pre-loaded models for source: \(String(describing: source))..."
        )

        self.audioSource = source

        // Initialize ASR manager with provided models
        asrManager = AsrManager(config: config.asrConfig)
        try await asrManager?.initialize(models: models)
        hypothesisAsrManager = supportsHypothesisTrack ? AsrManager(config: config.asrConfig) : nil
        if let hypothesisAsrManager {
            try await hypothesisAsrManager.initialize(models: models)
        }

        // Reset decoder state for the specific source
        try await asrManager?.resetDecoderState(for: source)
        try await hypothesisAsrManager?.resetDecoderState(for: source)

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        lastConfirmedAudioFrame = 0
        lastConfirmedAudioSample = 0
        confirmedTokenCount = 0
        accumulatedTokens.removeAll()
        hypothesisAccumulatedTokens.removeAll()
        accumulatedHypothesisTokens.removeAll()
        accumulatedHypothesisTimestamps.removeAll()
        accumulatedHypothesisConfidences.removeAll()
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        nextWindowCenterStart = 0
        nextHypothesisWindowStart = 0

        // Reset hypothesis stabilizer for clean state
        hypothesisStabilizer.reset()

        startTime = Date()

        // Start background recognition task
        recognizerTask = Task {
            logger.info("Recognition task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
                do {
                    // Convert to 16kHz mono (streaming)
                    let samples = try audioConverter.resampleBuffer(pcmBuffer)

                    // Append to raw sample buffer and attempt windowed processing
                    await self.appendSamplesAndProcess(samples)
                } catch {
                    let streamingError = StreamingAsrError.audioBufferProcessingFailed(error)
                    logger.error(
                        "Audio buffer processing error: \(streamingError.localizedDescription)")
                    await attemptErrorRecovery(error: streamingError)
                }
            }

            // Stream ended: no need to flush converter since each conversion is stateless

            // Then flush remaining assembled audio (no right-context requirement)
            await self.flushRemaining()

            logger.info("Recognition task completed")
        }

        logger.info("Streaming ASR engine started successfully")
    }

    /// Stream audio data for transcription
    /// - Parameter buffer: Audio buffer in any format (will be converted to 16kHz mono)
    public func streamAudio(_ buffer: AVAudioPCMBuffer) {
        inputBuilder.yield(buffer)
    }

    /// Update custom vocabulary for context biasing during streaming
    /// - Parameter vocabulary: Custom vocabulary context, or nil to disable context biasing
    public func setCtcModels(_ models: CtcModels) {
        self.ctcModels = models
    }

    public func setCustomVocabulary(_ vocabulary: CustomVocabularyContext?) {
        // Tokenize terms for CTC keyword spotting if they don't have ctcTokenIds
        if var vocab = vocabulary {
            vocab = tokenizeVocabularyForCtc(vocab)
            self.customVocabulary = vocab
            asrManager?.setCustomVocabulary(vocab)
            hypothesisAsrManager?.setCustomVocabulary(vocab)
            logger.info("Custom vocabulary set: \(vocab.terms.count) terms")
        } else {
            self.customVocabulary = nil
            asrManager?.setCustomVocabulary(nil)
            hypothesisAsrManager?.setCustomVocabulary(nil)
            logger.info("Custom vocabulary disabled")
        }
    }

    /// Tokenize vocabulary terms for CTC keyword spotting.
    /// This also expands aliases into separate CTC detection entries, allowing the acoustic
    /// model to detect alias pronunciations (e.g., "xertec") while the canonical form ("zyrtec")
    /// is used for text replacement.
    private func tokenizeVocabularyForCtc(_ vocabulary: CustomVocabularyContext) -> CustomVocabularyContext {
        do {
            let tokenizer = try CtcTokenizer()
            var expandedTerms: [CustomVocabularyTerm] = []
            var tokenizedCount = 0
            var aliasExpansionCount = 0

            for term in vocabulary.terms {
                // 1. Add the canonical term with its own CTC tokens
                let canonicalTokenIds = term.ctcTokenIds ?? tokenizer.encode(term.text)
                let canonicalTerm = CustomVocabularyTerm(
                    text: term.text,
                    weight: term.weight,
                    aliases: term.aliases,
                    tokenIds: term.tokenIds,
                    ctcTokenIds: canonicalTokenIds
                )
                expandedTerms.append(canonicalTerm)

                if term.ctcTokenIds == nil {
                    tokenizedCount += 1
                    logger.debug("Tokenized '\(term.text)': \(canonicalTokenIds)")
                }

                // 2. Expand aliases: create additional CTC detection entries for each alias
                // These use the alias text for CTC acoustic matching, but keep the canonical
                // term.text so KeywordMerger replaces with the correct spelling.
                if let aliases = term.aliases {
                    for alias in aliases {
                        let aliasTokenIds = tokenizer.encode(alias)
                        // Create a term that:
                        // - Uses alias tokens for CTC acoustic detection
                        // - Keeps canonical text for KeywordMerger replacement
                        let aliasTerm = CustomVocabularyTerm(
                            text: term.text,  // Canonical form for replacement
                            weight: term.weight,
                            aliases: term.aliases,
                            tokenIds: term.tokenIds,
                            ctcTokenIds: aliasTokenIds  // Alias tokens for acoustic matching
                        )
                        expandedTerms.append(aliasTerm)
                        aliasExpansionCount += 1
                        logger.debug("Tokenized alias '\(alias)' -> '\(term.text)': \(aliasTokenIds)")
                    }
                }
            }

            if tokenizedCount > 0 || aliasExpansionCount > 0 {
                logger.info(
                    "Auto-tokenized \(tokenizedCount) vocabulary terms for CTC, expanded \(aliasExpansionCount) aliases"
                )
            }

            return CustomVocabularyContext(
                terms: expandedTerms,
                minCtcScore: vocabulary.minCtcScore,
                minSimilarity: vocabulary.minSimilarity,
                minCombinedConfidence: vocabulary.minCombinedConfidence
            )
        } catch {
            logger.warning(
                "CTC tokenization failed: \(error.localizedDescription). Using vocabulary without CTC token IDs.")
            return vocabulary
        }
    }

    /// Get current custom vocabulary
    public func getCustomVocabulary() -> CustomVocabularyContext? {
        return customVocabulary
    }

    /// Get or lazily create the CTC keyword spotter for word boosting
    private func getOrCreateKeywordSpotter() async throws -> CtcKeywordSpotter {
        if let spotter = ctcKeywordSpotter {
            return spotter
        }

        let spotter: CtcKeywordSpotter
        if let models = ctcModels {
            spotter = CtcKeywordSpotter(models: models)
            logger.info("CTC keyword spotter initialized with pre-loaded models")
        } else {
            spotter = try await CtcKeywordSpotter.makeDefault()
            logger.info("CTC keyword spotter initialized with default models (lazy load)")
        }

        ctcKeywordSpotter = spotter
        return spotter
    }

    /// Apply CTC keyword corrections to transcription text using the audio samples
    /// - Parameters:
    ///   - text: Original transcription text
    ///   - audioSamples: Audio samples for CTC analysis
    /// - Returns: Tuple of (corrected text, corrections with character ranges)

    /// Fix chunk boundary artifacts where words are split with erroneous punctuation
    /// Detects patterns like "word. lowercase" and merges them into single words

    /// Detects patterns like "word. lowercase" and merges them into single words
    private func fixChunkBoundaryArtifacts(_ text: String) -> String {
        var result = text

        // Pattern 1: "word. lowercase" → "wordlowercase" (split word with period)
        let range1 = NSRange(result.startIndex..., in: result)
        result = Self.splitWordPattern.stringByReplacingMatches(
            in: result,
            options: [],
            range: range1,
            withTemplate: "$1$2"
        )

        // Pattern 2: "word.word" (no space) → "wordword" (already joined but has period)
        let range2 = NSRange(result.startIndex..., in: result)
        result = Self.noSpacePattern.stringByReplacingMatches(
            in: result,
            options: [],
            range: range2,
            withTemplate: "$1$2"
        )

        return result
    }

    private func applyKeywordCorrections(
        to text: String, tokenTimings: [TokenTiming], audioSamples: [Float]
    ) async -> (text: String, corrections: [CorrectedWord]) {
        // Apply artifact cleaning BEFORE keyword correction
        let cleanedText = fixChunkBoundaryArtifacts(text)

        guard let vocabulary = customVocabulary, !vocabulary.terms.isEmpty else {
            logger.debug("CTC correction skipped: no custom vocabulary")
            return (cleanedText, [])
        }

        guard !tokenTimings.isEmpty else {
            logger.debug("CTC correction skipped: no token timings available")
            return (cleanedText, [])
        }

        do {
            let spotter = try await getOrCreateKeywordSpotter()
            logger.debug("CTC spotting \(vocabulary.terms.count) terms in \(audioSamples.count) samples")

            let detections = try await spotter.spotKeywords(
                audioSamples: audioSamples,
                customVocabulary: vocabulary,
                minScore: vocabulary.minCtcScore
            )

            logger.info("CTC detected \(detections.count) keywords: \(detections.map { $0.term.text })")

            guard !detections.isEmpty else {
                return (cleanedText, [])
            }

            // Use VocabularyRescorer for principled CTC-based rescoring
            // This compares acoustic evidence for original words vs vocabulary terms
            // and only replaces when vocabulary term has significantly higher score
            let rescorer = VocabularyRescorer(
                spotter: spotter,
                vocabulary: vocabulary
            )

            let rescoreOutput = try await rescorer.rescore(
                transcript: cleanedText,
                audioSamples: audioSamples,
                detections: detections
            )

            let replacementCount = rescoreOutput.replacements.filter { $0.shouldReplace }.count
            if replacementCount > 0 {
                logger.info(
                    "Applied \(replacementCount) vocabulary corrections via CTC rescoring"
                )
            } else {
                logger.debug(
                    "CTC: \(detections.count) detections but no replacements applied to '\(cleanedText.prefix(50))...'")
            }

            // Compute character ranges by finding each corrected word in the result text
            var corrections: [CorrectedWord] = []
            var searchStart = rescoreOutput.text.startIndex
            for replacement in rescoreOutput.replacements where replacement.shouldReplace {
                guard let replacementWord = replacement.replacementWord else { continue }
                if let range = rescoreOutput.text.range(
                    of: replacementWord,
                    range: searchStart..<rescoreOutput.text.endIndex
                ) {
                    let startOffset = rescoreOutput.text.distance(
                        from: rescoreOutput.text.startIndex, to: range.lowerBound)
                    let endOffset = rescoreOutput.text.distance(
                        from: rescoreOutput.text.startIndex, to: range.upperBound)
                    corrections.append(
                        CorrectedWord(
                            range: startOffset..<endOffset,
                            original: replacement.originalWord,
                            corrected: replacementWord
                        ))
                    searchStart = range.upperBound
                }
            }

            return (rescoreOutput.text, corrections)
        } catch {
            logger.warning("CTC keyword correction failed: \(error.localizedDescription)")
            return (text, [])
        }
    }

    /// Get an async stream of transcription updates
    public var transcriptionUpdates: AsyncStream<StreamingTranscriptionUpdate> {
        AsyncStream { continuation in
            self.updateContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearUpdateContinuation()
                }
            }
        }
    }

    /// Finish streaming and get the final transcription
    /// - Returns: The complete transcription text
    public func finish() async throws -> String {
        logger.info("Finishing streaming ASR...")

        // Signal end of input
        inputBuilder.finish()

        // Wait for recognition task to complete
        do {
            try await recognizerTask?.value
        } catch {
            logger.error("Recognition task failed: \(error)")
            throw error
        }

        // Build final text from confirmed transcript + unconfirmed tokens
        // This ensures no content is lost between confirmations
        var finalText: String
        var components: [String] = []

        logger.debug(
            "finish() state: confirmedTokenCount=\(confirmedTokenCount), accumulatedTokens=\(accumulatedTokens.count)"
        )

        // The most reliable approach: regenerate text from ALL accumulated tokens
        // This ensures no content is lost, regardless of confirmation state issues
        if let asrManager = asrManager, !accumulatedTokens.isEmpty {
            let fullResult = asrManager.processTranscriptionResult(
                tokenIds: accumulatedTokens,
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: 0
            )
            var fullText = fullResult.text.trimmingCharacters(in: .whitespacesAndNewlines)

            // Apply vocabulary corrections to the full text if available
            if let vocabulary = customVocabulary, !vocabulary.terms.isEmpty, !sampleBuffer.isEmpty {
                let (correctedText, _) = await applyKeywordCorrections(
                    to: fullText,
                    tokenTimings: fullResult.tokenTimings ?? [],
                    audioSamples: sampleBuffer
                )
                fullText = correctedText
            }

            finalText = fullText
        } else if !confirmedTranscript.isEmpty || !pendingMainTranscription.isEmpty {
            // Fallback to text-based assembly if no tokens
            if !confirmedTranscript.isEmpty {
                components.append(confirmedTranscript)
            }
            if !pendingMainTranscription.isEmpty {
                components.append(pendingMainTranscription)
            }
            finalText = components.joined(separator: " ")
        } else {
            finalText = ""
        }

        logger.info("Final transcription: \(finalText.count) characters")
        return finalText
    }

    /// Reset the transcriber for a new session
    public func reset() async throws {
        volatileTranscript = ""
        confirmedTranscript = ""
        pendingMainTranscription = ""
        accumulatedCorrections = []
        currentChunkCorrections = []
        pendingMainCorrections = []
        processedChunks = 0
        startTime = Date()
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        nextWindowCenterStart = 0
        nextHypothesisWindowStart = 0

        // Reset decoder state for the current audio source
        if let asrManager = asrManager {
            try await asrManager.resetDecoderState(for: audioSource)
        }
        if let hypothesisAsrManager = hypothesisAsrManager {
            try await hypothesisAsrManager.resetDecoderState(for: audioSource)
        }

        // Reset sliding window state
        segmentIndex = 0
        lastProcessedFrame = 0
        lastConfirmedAudioFrame = 0
        lastConfirmedAudioSample = 0
        confirmedTokenCount = 0
        accumulatedTokens.removeAll()
        hypothesisAccumulatedTokens.removeAll()
        accumulatedHypothesisTokens.removeAll()
        accumulatedHypothesisTimestamps.removeAll()
        accumulatedHypothesisConfidences.removeAll()

        // Reset hypothesis stabilizer
        hypothesisStabilizer.reset()

        logger.info("StreamingAsrManager reset for source: \(String(describing: self.audioSource))")
    }

    /// Cancel streaming without getting results
    public func cancel() async {
        inputBuilder.finish()
        recognizerTask?.cancel()
        updateContinuation?.finish()

        logger.info("StreamingAsrManager cancelled")
    }

    /// Clear the update continuation
    private func clearUpdateContinuation() {
        updateContinuation = nil
    }

    // MARK: - Private Methods

    /// Append new samples and process as many windows as available
    private func appendSamplesAndProcess(_ samples: [Float]) async {
        // Append samples to buffer
        sampleBuffer.append(contentsOf: samples)

        // Emit quick hypothesis updates if enabled
        await processHypothesisWindows()

        // Process while we have at least chunk + right ahead of the current center start
        let chunk = config.chunkSamples
        let right = config.rightContextSamples
        let left = config.leftContextSamples
        let requiredLeft =
            supportsHypothesisTrack
            ? max(config.leftContextSamples, config.hypothesisLeftContextSamples)
            : config.leftContextSamples

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd >= (nextWindowCenterStart + chunk + right) {
            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + chunk + right
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)

            logger.debug(
                "Main window check: bufferStart=\(bufferStartIndex), bufferCount=\(sampleBuffer.count), nextCenter=\(nextWindowCenterStart), startIdx=\(startIdx), endIdx=\(endIdx)"
            )

            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx {
                logger.warning(
                    "Main window bounds issue: startIdx=\(startIdx), endIdx=\(endIdx), bufferCount=\(sampleBuffer.count)"
                )
                break
            }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            await processWindow(window, windowStartSample: leftStartAbs)

            // Advance by chunk size
            nextWindowCenterStart += chunk

            // Trim buffer to keep only what's needed for left context
            let trimAnchor =
                supportsHypothesisTrack
                ? min(nextWindowCenterStart, nextHypothesisWindowStart)
                : nextWindowCenterStart
            let trimToAbs = max(0, trimAnchor - requiredLeft)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Flush any remaining audio at end of stream (no right-context requirement)
    private func flushRemaining() async {
        await processHypothesisWindows()

        let chunk = config.chunkSamples
        let left = config.leftContextSamples
        let requiredLeft =
            supportsHypothesisTrack
            ? max(config.leftContextSamples, config.hypothesisLeftContextSamples)
            : config.leftContextSamples

        var currentAbsEnd = bufferStartIndex + sampleBuffer.count
        while currentAbsEnd > nextWindowCenterStart {  // process until we exhaust
            // If we have less than a chunk ahead, process the final partial chunk
            let availableAhead = currentAbsEnd - nextWindowCenterStart
            if availableAhead <= 0 { break }
            let effectiveChunk = min(chunk, availableAhead)

            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs = nextWindowCenterStart + effectiveChunk
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = max(rightEndAbs - bufferStartIndex, startIdx)
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx { break }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            await processWindow(window, windowStartSample: leftStartAbs)

            nextWindowCenterStart += effectiveChunk

            // Trim
            let trimAnchor =
                supportsHypothesisTrack
                ? min(nextWindowCenterStart, nextHypothesisWindowStart)
                : nextWindowCenterStart
            let trimToAbs = max(0, trimAnchor - requiredLeft)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= sampleBuffer.count {
                sampleBuffer.removeFirst(dropCount)
                bufferStartIndex += dropCount
            }

            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }

        // Flush any remaining hypothesis audio (even if less than hypothesis chunk)
        await processRemainingHypothesisAudio()
    }

    /// Process a single assembled window: [left, chunk, right]
    private func processWindow(_ windowSamples: [Float], windowStartSample: Int) async {
        guard let asrManager = asrManager else { return }

        do {
            let chunkStartTime = Date()

            // Reset decoder state before each chunk for stateless processing
            // This ensures each overlapping window is transcribed independently
            // and overlap is handled via token deduplication, not decoder state
            try await asrManager.resetDecoderState(for: audioSource)

            // Call AsrManager directly with deduplication
            let (tokens, timestamps, confidences, _) = try await asrManager.transcribeStreamingChunk(
                windowSamples,
                source: audioSource,
                previousTokens: accumulatedTokens
            )

            logger.debug(
                "processWindow chunk \(processedChunks): windowSamples=\(windowSamples.count) (\(Double(windowSamples.count)/16000.0)s), got \(tokens.count) new tokens, accumulated before: \(accumulatedTokens.count)"
            )

            let adjustedTimestamps = Self.applyGlobalFrameOffset(
                to: timestamps,
                windowStartSample: windowStartSample
            )

            // Update state
            accumulatedTokens.append(contentsOf: tokens)
            lastProcessedFrame = max(lastProcessedFrame, adjustedTimestamps.max() ?? 0)
            // Note: lastConfirmedAudioFrame only updated when text is actually confirmed
            segmentIndex += 1

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            processedChunks += 1

            // Convert only the current chunk tokens to text for clean incremental updates
            // The final result will use all accumulated tokens for proper deduplication
            var interim = asrManager.processTranscriptionResult(
                tokenIds: tokens,  // Only current chunk tokens for progress updates
                timestamps: adjustedTimestamps,
                confidences: confidences,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            // Apply CTC keyword corrections if custom vocabulary is set
            if customVocabulary != nil {
                let (correctedText, chunkCorrections) = await applyKeywordCorrections(
                    to: interim.text,
                    tokenTimings: interim.tokenTimings ?? [],
                    audioSamples: windowSamples)
                if correctedText != interim.text {
                    interim = ASRResult(
                        text: correctedText,
                        confidence: interim.confidence,
                        duration: interim.duration,
                        processingTime: interim.processingTime,
                        tokenTimings: interim.tokenTimings,
                        performanceMetrics: interim.performanceMetrics,
                        ctcDetectedTerms: interim.ctcDetectedTerms,
                        ctcAppliedTerms: interim.ctcAppliedTerms
                    )
                    // Store corrections for this chunk
                    currentChunkCorrections = chunkCorrections
                } else {
                    currentChunkCorrections = []
                }
            } else {
                currentChunkCorrections = []
            }

            logger.debug(
                "Chunk \(self.processedChunks): '\(interim.text)', time: \(String(format: "%.3f", processingTime))s)"
            )

            // Apply confidence-based confirmation logic (uses configured threshold)
            let lockedCountForBonus = supportsHypothesisTrack ? hypothesisStabilizer.statistics.locked : 0
            let didConfirm = await updateTranscriptionState(
                with: interim,
                allowConfirmation: true,
                lockedTokenCount: lockedCountForBonus
            )

            // Emit update based on progressive confidence model
            let shouldConfirm = didConfirm

            let normalizedInterimText = interim.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalizedInterimText.isEmpty else {
                logger.debug("Skipping emission of empty transcription update; keeping existing display text")
                return
            }

            let combinedText = [confirmedTranscript, volatileTranscript]
                .filter { !$0.isEmpty }
                .joined(separator: " ")

            let update = StreamingTranscriptionUpdate(
                text: interim.text,
                isConfirmed: shouldConfirm,
                combinedText: combinedText,
                confidence: interim.confidence,
                timestamp: Date(),
                tokenIds: tokens,
                tokenTimings: interim.tokenTimings ?? []
            )

            if shouldConfirm {
                // Only update confirmed audio boundary when text is actually confirmed
                lastConfirmedAudioFrame = lastProcessedFrame
                // Track confirmed position in samples (end of the processed window)
                lastConfirmedAudioSample = windowStartSample + windowSamples.count
                // Track confirmed token count for proper finish() handling
                confirmedTokenCount = accumulatedTokens.count
                hypothesisAccumulatedTokens.removeAll(keepingCapacity: true)
                // Clear accumulated hypothesis data since it's now confirmed
                accumulatedHypothesisTokens.removeAll(keepingCapacity: true)
                accumulatedHypothesisTimestamps.removeAll(keepingCapacity: true)
                accumulatedHypothesisConfidences.removeAll(keepingCapacity: true)
                // Reset the hypothesis stabilizer completely - confirmed text supersedes
                // any locked hypothesis tokens, and we start fresh for the next segment
                hypothesisStabilizer.reset()
                // Reset hypothesis decoder state to avoid state drift after confirmation
                if let hypothesisAsrManager = hypothesisAsrManager {
                    try? await hypothesisAsrManager.resetDecoderState(for: audioSource)
                }
            }

            updateContinuation?.yield(update)

        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Model processing error: \(streamingError.localizedDescription)")

            // Attempt error recovery
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Emit quick hypothesis windows for immediate feedback
    private func processHypothesisWindows() async {
        guard supportsHypothesisTrack,
            let hypothesisAsrManager = hypothesisAsrManager
        else { return }

        let hypothesisSamples = config.hypothesisChunkSamples
        guard hypothesisSamples > 0 else { return }

        let left = config.hypothesisLeftContextSamples
        let right = config.rightContextSamples
        var currentAbsEnd = bufferStartIndex + sampleBuffer.count

        // Skip hypothesis windows that would start before confirmed audio
        // The hypothesis should only show preview of UNCONFIRMED audio
        if nextHypothesisWindowStart < lastConfirmedAudioSample {
            nextHypothesisWindowStart = lastConfirmedAudioSample
        }

        while currentAbsEnd >= (nextHypothesisWindowStart + hypothesisSamples + right) {
            // Allow left context from confirmed audio (for accuracy) while later filtering any
            // confirmed tokens before emitting the hypothesis text.
            let unconstrainedLeftStart = nextHypothesisWindowStart - left
            // Allow hypothesis windows to borrow confirmed audio as left context for stability
            // while still constraining emitted tokens below.
            let leftStartAbs = max(0, unconstrainedLeftStart)
            let rightEndAbs = nextHypothesisWindowStart + hypothesisSamples + right
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx {
                break
            }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            await processHypothesisWindow(
                window,
                windowStartSample: leftStartAbs,
                hypothesisAsrManager: hypothesisAsrManager
            )

            nextHypothesisWindowStart += hypothesisSamples
            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Flush any remaining hypothesis audio when stream ends
    private func processRemainingHypothesisAudio() async {
        guard supportsHypothesisTrack,
            let hypothesisAsrManager = hypothesisAsrManager
        else { return }

        let hypothesisSamples = config.hypothesisChunkSamples
        guard hypothesisSamples > 0 else { return }

        let left = config.hypothesisLeftContextSamples
        var currentAbsEnd = bufferStartIndex + sampleBuffer.count

        // Skip to confirmed boundary if needed
        if nextHypothesisWindowStart < lastConfirmedAudioSample {
            nextHypothesisWindowStart = lastConfirmedAudioSample
        }

        while currentAbsEnd > nextHypothesisWindowStart {
            let availableAhead = currentAbsEnd - nextHypothesisWindowStart
            if availableAhead <= 0 { break }
            let effectiveChunk = min(hypothesisSamples, availableAhead)

            // Allow left context from confirmed audio for higher quality hypotheses; actual emission
            // will drop tokens that overlap confirmed regions.
            let unconstrainedLeftStart = nextHypothesisWindowStart - left
            let leftStartAbs = max(0, unconstrainedLeftStart)
            let rightEndAbs = nextHypothesisWindowStart + effectiveChunk
            let startIdx = max(leftStartAbs - bufferStartIndex, 0)
            let endIdx = min(rightEndAbs - bufferStartIndex, sampleBuffer.count)
            if startIdx < 0 || endIdx > sampleBuffer.count || startIdx >= endIdx { break }

            let window = Array(sampleBuffer[startIdx..<endIdx])
            await processHypothesisWindow(
                window,
                windowStartSample: leftStartAbs,
                hypothesisAsrManager: hypothesisAsrManager
            )

            nextHypothesisWindowStart += effectiveChunk
            currentAbsEnd = bufferStartIndex + sampleBuffer.count
        }
    }

    /// Decode and publish a hypothesis window
    private func processHypothesisWindow(
        _ windowSamples: [Float],
        windowStartSample: Int,
        hypothesisAsrManager: AsrManager
    ) async {
        do {
            let chunkStartTime = Date()

            // Reset hypothesis decoder state before each window to avoid state drift
            // The hypothesis track should decode each window independently to prevent
            // accumulation of LSTM state errors that cause gibberish output
            try await hypothesisAsrManager.resetDecoderState(for: audioSource)

            // Decode the hypothesis window independently (no previousTokens - stateless)
            // This produces clean output for each window without state contamination
            let (rawTokens, rawTimestamps, rawConfidences, _) =
                try await hypothesisAsrManager
                .transcribeStreamingChunk(
                    windowSamples,
                    source: audioSource,
                    previousTokens: []  // Stateless - no deduplication against previous
                )

            let tokens = rawTokens
            let timestamps = rawTimestamps
            let confidences = rawConfidences

            // Skip emitting empty hypotheses so we don't clear existing text
            guard !tokens.isEmpty else {
                logger.debug(
                    "Hypothesis window produced no new tokens (start=\(nextHypothesisWindowStart))"
                )
                return
            }

            let adjustedTimestamps = Self.applyGlobalFrameOffset(
                to: timestamps,
                windowStartSample: windowStartSample
            )

            // Drop any tokens that fall inside already-confirmed audio. This lets us reuse
            // confirmed audio as left context for decoding while keeping the hypothesis text
            // strictly to unconfirmed regions.
            let unconfirmed = zip(tokens, zip(adjustedTimestamps, confidences))
                .filter { $0.1.0 > lastConfirmedAudioFrame }
            let filteredTokens = unconfirmed.map(\.0)
            let filteredTimestamps = unconfirmed.map { $0.1.0 }
            let filteredConfidences = unconfirmed.map { $0.1.1 }

            guard !filteredTokens.isEmpty else {
                logger.debug("Hypothesis tokens all overlapped confirmed audio; keeping previous hypothesis text")
                return
            }

            // Apply hypothesis stabilization to reduce text fluctuation
            // The stabilizer tracks token stability across updates and locks stable tokens
            let stabilized = hypothesisStabilizer.stabilize(
                newTokens: filteredTokens,
                timestamps: filteredTimestamps,
                confidences: filteredConfidences
            )

            // Use stabilized output - only emit if there are changes or new tokens
            let outputTokens = stabilized.allTokens
            let outputTimestamps = stabilized.allTimestamps
            let outputConfidences = stabilized.allConfidences

            // Avoid re-emitting purely overlapping tokens; if nothing changed and no new tokens past
            // the overlap boundary, skip the UI update to keep text stable.
            let overlap = hypothesisOverlapCount(
                previous: hypothesisAccumulatedTokens,
                current: filteredTokens
            )
            let hasNewTokensBeyondOverlap = filteredTokens.count > overlap
            if !stabilized.hasChanges && !hasNewTokensBeyondOverlap {
                logger.debug("Hypothesis window contained only overlapping tokens; no update emitted")
                return
            }

            guard !outputTokens.isEmpty else {
                logger.debug("Stabilizer produced no output tokens")
                return
            }

            // Convert stabilized tokens to text
            let processingTime = Date().timeIntervalSince(chunkStartTime)

            let interim = hypothesisAsrManager.processTranscriptionResult(
                tokenIds: outputTokens,
                timestamps: outputTimestamps,
                confidences: outputConfidences,
                encoderSequenceLength: 0,
                audioSamples: windowSamples,
                processingTime: processingTime
            )

            // Log stabilization statistics periodically
            let stats = hypothesisStabilizer.statistics
            if stats.updates % 10 == 0 {
                logger.debug(
                    "Stabilizer stats: locked=\(stats.locked), volatile=\(stats.volatile), updates=\(stats.updates)"
                )
            }

            // Track hypothesis token history to improve overlap suppression on subsequent windows.
            hypothesisAccumulatedTokens = Array(
                (hypothesisAccumulatedTokens + outputTokens).suffix(maxHypothesisTokenHistory)
            )

            // Update the public volatile transcript for immediate feedback (UI display)
            // The pendingMainTranscription (internal logic state) remains untouched until the
            // Main Track confirms the text, preserving the stability of the confirmation logic.
            volatileTranscript = interim.text

            // Create update with stability information
            let normalizedInterimText = interim.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalizedInterimText.isEmpty else {
                logger.debug("Skipping emission of empty hypothesis update; preserving previous text")
                return
            }

            let combinedText = [confirmedTranscript, interim.text]
                .filter { !$0.isEmpty }
                .joined(separator: " ")

            let update = StreamingTranscriptionUpdate(
                text: interim.text,
                isConfirmed: false,
                combinedText: combinedText,
                confidence: interim.confidence,
                timestamp: Date(),
                tokenIds: outputTokens,
                tokenTimings: interim.tokenTimings ?? [],
                lockedTokenCount: stabilized.lockedCount
            )

            updateContinuation?.yield(update)
        } catch {
            let streamingError = StreamingAsrError.modelProcessingFailed(error)
            logger.error("Hypothesis processing error: \(streamingError.localizedDescription)")
            await attemptErrorRecovery(error: streamingError)
        }
    }

    /// Calculate overlap between previously emitted hypothesis tokens and current tokens
    /// Uses a combination of exact suffix-prefix matching and LCS for fuzzy matching
    private func hypothesisOverlapCount(previous: [Int], current: [Int]) -> Int {
        guard !previous.isEmpty, !current.isEmpty else { return 0 }

        let punctuationTokens: Set<Int> = [7883, 7952, 7948]  // ., ?, !
        if let last = previous.last,
            let first = current.first,
            last == first,
            punctuationTokens.contains(first)
        {
            return 1
        }

        let overlapLimit = min(maxHypothesisOverlapTokens, previous.count, current.count)
        guard overlapLimit >= 1 else { return 0 }

        // Phase 1: Try exact suffix-prefix match (fastest, most reliable)
        for length in stride(from: overlapLimit, through: 1, by: -1) {
            let prevSuffix = previous.suffix(length)
            let currPrefix = current.prefix(length)
            if prevSuffix.elementsEqual(currPrefix) {
                return length
            }
        }

        // Phase 2: Try LCS-based fuzzy matching for cases where exact match fails
        // This handles cases where the decoder produces slightly different tokens
        // for the same audio due to window boundary effects
        let lcsOverlap = findLCSBasedOverlap(
            previousSuffix: Array(previous.suffix(min(20, previous.count))),
            currentPrefix: Array(current.prefix(min(20, current.count)))
        )

        return lcsOverlap
    }

    /// Find overlap using Longest Common Subsequence with position constraints
    /// Requires that the LCS elements appear in order and near the boundary
    private func findLCSBasedOverlap(previousSuffix: [Int], currentPrefix: [Int]) -> Int {
        guard !previousSuffix.isEmpty && !currentPrefix.isEmpty else { return 0 }

        let m = previousSuffix.count
        let n = currentPrefix.count

        // Build LCS length table
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 1...m {
            for j in 1...n {
                if previousSuffix[i - 1] == currentPrefix[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        let lcsLength = dp[m][n]

        // Require at least 3 matching tokens for LCS-based overlap
        // and the matches should cover at least 40% of the shorter sequence
        let minRequiredMatches = 3
        let minCoverage = 0.4
        let shortestLength = min(m, n)

        guard lcsLength >= minRequiredMatches else { return 0 }
        guard Double(lcsLength) / Double(shortestLength) >= minCoverage else { return 0 }

        // Trace back to find where the LCS elements end in currentPrefix
        // This tells us how many tokens in current are "covered" by the overlap
        var i = m
        var j = n
        var lastMatchedCurrentIndex = -1

        while i > 0 && j > 0 {
            if previousSuffix[i - 1] == currentPrefix[j - 1] {
                lastMatchedCurrentIndex = max(lastMatchedCurrentIndex, j - 1)
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        // Return the number of tokens in currentPrefix that should be skipped
        // (up to and including the last matched position)
        if lastMatchedCurrentIndex >= 0 {
            return lastMatchedCurrentIndex + 1
        }

        return 0
    }

    /// Update transcription state based on confidence and context duration
    /// Also considers stability of locked tokens from hypothesis stabilizer
    @discardableResult
    private func updateTranscriptionState(
        with result: ASRResult,
        allowConfirmation: Bool,
        lockedTokenCount: Int = 0
    ) async -> Bool {
        let totalAudioProcessed = Double(bufferStartIndex + sampleBuffer.count) / 16000.0
        let hasMinimumContext = totalAudioProcessed >= config.minContextForConfirmation
        let isHighConfidence = Double(result.confidence) >= config.confirmationThreshold

        // Stability-based confirmation: locked tokens from hypothesis stabilizer
        // can be treated as more reliable since they've survived multiple decodes
        let hasSignificantLockedTokens = lockedTokenCount >= 3
        let stabilityBonus = hasSignificantLockedTokens ? 0.1 : 0.0
        let adjustedConfidenceThreshold = config.confirmationThreshold - stabilityBonus

        let isEffectivelyHighConfidence = Double(result.confidence) >= adjustedConfidenceThreshold

        // Progressive confidence model:
        // 1. Always show text immediately as volatile for responsiveness
        // 2. Confirm text when we have high confidence AND sufficient context
        // 3. Give stability bonus to text with many locked tokens
        let shouldConfirm =
            allowConfirmation
            && (isHighConfidence || (isEffectivelyHighConfidence && hasSignificantLockedTokens))
            && hasMinimumContext

        let incomingText = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let incomingIsEmpty = incomingText.isEmpty

        // Do not clear or confirm when the decoder produces an empty/erased hypothesis; keep the
        // previous volatile text visible until we have new tokens to show.
        if incomingIsEmpty {
            logger.debug(
                "VOLATILE (\(result.confidence)): empty hypothesis - preserving existing volatile '\(pendingMainTranscription)'"
            )
            return false
        }

        if shouldConfirm {
            // Move volatile text to confirmed and set new text as volatile
            if !pendingMainTranscription.isEmpty {
                var components: [String] = []
                if !confirmedTranscript.isEmpty {
                    components.append(confirmedTranscript)
                }

                // Smart punctuation: if volatile text ends with sentence punctuation
                // but the new text starts with lowercase (indicating continuation),
                // defer the punctuation to avoid mid-sentence breaks
                let textToConfirm = adjustPunctuationForContinuation(
                    textToConfirm: pendingMainTranscription,
                    followingText: result.text
                )
                components.append(textToConfirm)
                confirmedTranscript = components.joined(separator: " ")

                // Accumulate corrections: offset pendingMainCorrections by position in confirmed text
                // The confirmed text now includes textToConfirm, so offset = old confirmedTranscript length + space
                let offsetBase = confirmedTranscript.count - textToConfirm.count
                for correction in pendingMainCorrections {
                    let adjustedRange =
                        (correction.range.lowerBound + offsetBase)..<(correction.range.upperBound + offsetBase)
                    accumulatedCorrections.append(
                        CorrectedWord(
                            range: adjustedRange,
                            original: correction.original,
                            corrected: correction.corrected
                        ))
                }
                if !pendingMainCorrections.isEmpty {
                    logger.debug(
                        "Accumulated \(pendingMainCorrections.count) corrections, total: \(accumulatedCorrections.count)"
                    )
                }
            }
            // Move current chunk's corrections to pending (for next confirmation cycle)
            pendingMainCorrections = currentChunkCorrections
            pendingMainTranscription = result.text
            volatileTranscript = result.text
            let stabilityNote = hasSignificantLockedTokens ? " [stability boost]" : ""
            logger.debug(
                "CONFIRMED (\(result.confidence), \(String(format: "%.1f", totalAudioProcessed))s context)\(stabilityNote): promoted to confirmed; new volatile '\(result.text)'"
            )
        } else {
            // Only update volatile text (hypothesis)
            pendingMainCorrections = currentChunkCorrections
            pendingMainTranscription = result.text
            volatileTranscript = result.text
            let reason =
                !hasMinimumContext
                ? "insufficient context (\(String(format: "%.1f", totalAudioProcessed))s)"
                : "low confidence (\(String(format: "%.2f", result.confidence)))"
            logger.debug("VOLATILE (\(result.confidence)): \(reason) - updated volatile '\(result.text)'")
        }

        return shouldConfirm
    }

    /// Adjust punctuation when confirming text to avoid mid-sentence breaks
    /// If text ends with sentence-ending punctuation but following text starts with lowercase,
    /// the sentence likely continues, so we strip the trailing punctuation
    private func adjustPunctuationForContinuation(textToConfirm: String, followingText: String) -> String {
        let trimmedConfirm = textToConfirm.trimmingCharacters(in: .whitespaces)
        let trimmedFollowing = followingText.trimmingCharacters(in: .whitespaces)

        // If no following text, keep punctuation as-is
        guard !trimmedFollowing.isEmpty else { return textToConfirm }

        // Check if confirmed text ends with sentence-ending punctuation
        let sentenceEndingPunctuation: Set<Character> = [".", "!", "?"]
        guard let lastChar = trimmedConfirm.last, sentenceEndingPunctuation.contains(lastChar) else {
            return textToConfirm
        }

        // Check if following text starts with lowercase (indicating continuation)
        guard let firstFollowingChar = trimmedFollowing.first else { return textToConfirm }

        // If following starts with lowercase letter, the sentence continues - strip punctuation
        if firstFollowingChar.isLowercase {
            var adjusted = trimmedConfirm
            adjusted.removeLast()
            return adjusted.trimmingCharacters(in: .whitespaces)
        }

        // If following starts with uppercase or non-letter, keep the punctuation
        return textToConfirm
    }

    /// Apply encoder-frame offset derived from the absolute window start sample.
    /// Streaming runs in disjoint chunks, so we need to add the global offset to
    /// keep each chunk's token timings aligned to the full audio timeline rather
    /// than resetting back to zero for every window.

    /// Helper to decode tokens for correction logic
    private func decodeTokens(_ tokens: [Int], with manager: AsrManager) -> String? {
        let vocab = manager.vocabulary
        let strings = tokens.compactMap { vocab[$0] }
        if strings.isEmpty { return nil }
        return strings.joined().replacingOccurrences(of: " ", with: " ")
    }

    internal static func applyGlobalFrameOffset(to timestamps: [Int], windowStartSample: Int) -> [Int] {
        guard !timestamps.isEmpty else { return timestamps }

        let frameOffset = windowStartSample / ASRConstants.samplesPerEncoderFrame
        guard frameOffset != 0 else { return timestamps }

        return timestamps.map { $0 + frameOffset }
    }

    /// Attempt to recover from processing errors
    private func attemptErrorRecovery(error: Error) async {
        logger.warning("Attempting error recovery for: \(error)")

        // Handle specific error types with targeted recovery
        if let streamingError = error as? StreamingAsrError {
            switch streamingError {
            case .modelsNotLoaded:
                logger.error("Models not loaded - cannot recover automatically")

            case .streamAlreadyExists:
                logger.error("Stream already exists - cannot recover automatically")

            case .audioBufferProcessingFailed:
                logger.info("Recovering from audio buffer error")

            case .audioConversionFailed:
                logger.info("Recovering from audio conversion error")

            case .modelProcessingFailed:
                logger.info("Recovering from model processing error - resetting decoder state")
                await resetDecoderForRecovery()

            case .bufferOverflow:
                logger.info("Buffer overflow handled automatically")

            case .invalidConfiguration:
                logger.error("Configuration error cannot be recovered automatically")
            }
        } else {
            // Generic recovery for non-streaming errors
            await resetDecoderForRecovery()
        }
    }

    /// Reset decoder state for error recovery
    private func resetDecoderForRecovery() async {
        var recoveredModels: AsrModels?

        if let asrManager = asrManager {
            do {
                try await asrManager.resetDecoderState(for: audioSource)
                logger.info("Successfully reset decoder state during error recovery")
            } catch {
                logger.error("Failed to reset decoder state during recovery: \(error)")

                // Last resort: try to reinitialize the ASR manager
                do {
                    if recoveredModels == nil {
                        recoveredModels = try await AsrModels.downloadAndLoad()
                    }
                    let newAsrManager = AsrManager(config: config.asrConfig)
                    if let models = recoveredModels {
                        try await newAsrManager.initialize(models: models)
                        try await newAsrManager.resetDecoderState(for: audioSource)
                        self.asrManager = newAsrManager
                        logger.info("Successfully reinitialized ASR manager during error recovery")
                    }
                } catch {
                    logger.error("Failed to reinitialize ASR manager during recovery: \(error)")
                }
            }
        }

        if supportsHypothesisTrack, let hypothesisAsrManager = hypothesisAsrManager {
            do {
                try await hypothesisAsrManager.resetDecoderState(for: audioSource)
                logger.info("Successfully reset hypothesis decoder state during error recovery")
            } catch {
                logger.error("Failed to reset hypothesis decoder state during recovery: \(error)")

                do {
                    if recoveredModels == nil {
                        recoveredModels = try await AsrModels.downloadAndLoad()
                    }
                    if let models = recoveredModels {
                        let newHypothesisManager = AsrManager(config: config.asrConfig)
                        try await newHypothesisManager.initialize(models: models)
                        try await newHypothesisManager.resetDecoderState(for: audioSource)
                        self.hypothesisAsrManager = newHypothesisManager
                        logger.info("Successfully reinitialized hypothesis ASR manager during error recovery")
                    }
                } catch {
                    logger.error("Failed to reinitialize hypothesis ASR manager during recovery: \(error)")
                }
            }
        }
    }
}

/// Configuration for StreamingAsrManager
public struct StreamingAsrConfig: Sendable {
    /// Main chunk size for stable transcription (seconds). Should be 10-11s for best quality
    public let chunkSeconds: TimeInterval
    /// Quick hypothesis chunk size for immediate feedback (seconds). Typical: 1.0s
    public let hypothesisChunkSeconds: TimeInterval
    /// Left context appended to each window (seconds). Typical: 10.0s
    public let leftContextSeconds: TimeInterval
    /// Additional left context used for hypothesis windows to improve early decoding (seconds)
    public let hypothesisLeftContextSeconds: TimeInterval
    /// Right context lookahead (seconds). Typical: 2.0s (adds latency)
    public let rightContextSeconds: TimeInterval
    /// Minimum audio duration before confirming text (seconds). Should be ~10s
    public let minContextForConfirmation: TimeInterval

    /// Confidence threshold for promoting volatile text to confirmed (0.0...1.0)
    public let confirmationThreshold: Double

    /// Configuration for hypothesis stabilization (reduces text fluctuation)
    public let hypothesisStabilizerConfig: HypothesisStabilizer.Config

    /// Default configuration aligned with previous API expectations
    public static let `default` = StreamingAsrConfig(
        chunkSeconds: 15.0,
        hypothesisChunkSeconds: 2.0,
        leftContextSeconds: 10.0,
        hypothesisLeftContextSeconds: 10.0,
        rightContextSeconds: 2.0,
        minContextForConfirmation: 10.0,
        confirmationThreshold: 0.70,
        hypothesisStabilizerConfig: .default
    )

    /// Optimized streaming configuration: Dual-track processing for best experience
    /// Uses ChunkProcessor's proven 11-2-2 approach for stable transcription
    /// Plus quick hypothesis updates for immediate feedback
    public static let streaming = StreamingAsrConfig(
        chunkSeconds: 11.0,  // Match ChunkProcessor for stable transcription
        hypothesisChunkSeconds: 1.0,  // Quick hypothesis updates
        leftContextSeconds: 2.0,  // Match ChunkProcessor left context
        hypothesisLeftContextSeconds: 8.0,  // Longer history for hypothesis decoding
        rightContextSeconds: 2.0,  // Match ChunkProcessor right context
        minContextForConfirmation: 10.0,  // Need sufficient context before confirming
        confirmationThreshold: 0.70,  // Balanced threshold for reliable confirmations
        hypothesisStabilizerConfig: .default
    )

    public init(
        chunkSeconds: TimeInterval = 10.0,
        hypothesisChunkSeconds: TimeInterval = 1.0,
        leftContextSeconds: TimeInterval = 2.0,
        hypothesisLeftContextSeconds: TimeInterval? = nil,
        rightContextSeconds: TimeInterval = 2.0,
        minContextForConfirmation: TimeInterval = 10.0,
        confirmationThreshold: Double = 0.70,
        hypothesisStabilizerConfig: HypothesisStabilizer.Config = .default
    ) {
        self.chunkSeconds = chunkSeconds
        self.hypothesisChunkSeconds = hypothesisChunkSeconds
        self.leftContextSeconds = leftContextSeconds
        self.hypothesisLeftContextSeconds = hypothesisLeftContextSeconds ?? max(leftContextSeconds, 8.0)
        self.rightContextSeconds = rightContextSeconds
        self.minContextForConfirmation = minContextForConfirmation
        self.confirmationThreshold = confirmationThreshold
        self.hypothesisStabilizerConfig = hypothesisStabilizerConfig
    }

    /// Backward-compatible convenience initializer used by tests (chunkDuration label)
    public init(
        confirmationThreshold: Double = 0.70,
        chunkDuration: TimeInterval
    ) {
        self.init(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            hypothesisLeftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold,
            hypothesisStabilizerConfig: .default
        )
    }

    /// Custom configuration factory expected by tests
    public static func custom(
        chunkDuration: TimeInterval,
        confirmationThreshold: Double
    ) -> StreamingAsrConfig {
        StreamingAsrConfig(
            chunkSeconds: chunkDuration,
            hypothesisChunkSeconds: min(1.0, chunkDuration / 2.0),  // Default to half chunk duration
            leftContextSeconds: 10.0,
            hypothesisLeftContextSeconds: 10.0,
            rightContextSeconds: 2.0,
            minContextForConfirmation: 10.0,
            confirmationThreshold: confirmationThreshold,
            hypothesisStabilizerConfig: .default
        )
    }

    // Internal ASR configuration
    var asrConfig: ASRConfig {
        ASRConfig(
            sampleRate: 16000,
            tdtConfig: TdtConfig()
        )
    }

    // Sample counts at 16 kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var hypothesisChunkSamples: Int { Int(hypothesisChunkSeconds * 16000) }
    var leftContextSamples: Int { Int(leftContextSeconds * 16000) }
    var hypothesisLeftContextSamples: Int { Int(hypothesisLeftContextSeconds * 16000) }
    var rightContextSamples: Int { Int(rightContextSeconds * 16000) }
    var minContextForConfirmationSamples: Int { Int(minContextForConfirmation * 16000) }

    // Backward-compat convenience for existing call-sites/tests
    var chunkDuration: TimeInterval { chunkSeconds }
    var bufferCapacity: Int { Int(15.0 * 16000) }
    var chunkSizeInSamples: Int { chunkSamples }
}

/// Transcription update from streaming ASR
public struct StreamingTranscriptionUpdate: Sendable {
    /// The transcribed text
    public let text: String

    /// Whether this text is confirmed (high confidence) or volatile (may change)
    public let isConfirmed: Bool

    /// Aggregate text including previously confirmed transcript plus this update's text.
    /// Useful for UI display without manual reconstruction.
    public let combinedText: String

    /// Confidence score (0.0 - 1.0)
    public let confidence: Float

    /// Timestamp of this update
    public let timestamp: Date

    /// Raw token identifiers emitted for this update
    public let tokenIds: [Int]

    /// Token-level timing information aligned with the decoded text
    public let tokenTimings: [TokenTiming]

    /// Number of tokens that are "locked" (stable and won't change)
    /// For hypothesis updates, locked tokens have survived multiple consecutive updates
    /// and are highly likely to be correct. Only the tokens after the locked portion may change.
    public let lockedTokenCount: Int

    /// Human-readable tokens (normalized) for this update
    public var tokens: [String] {
        tokenTimings.map(\.token)
    }

    public init(
        text: String,
        isConfirmed: Bool,
        combinedText: String? = nil,
        confidence: Float,
        timestamp: Date,
        tokenIds: [Int] = [],
        tokenTimings: [TokenTiming] = [],
        lockedTokenCount: Int = 0
    ) {
        self.text = text
        self.isConfirmed = isConfirmed
        self.combinedText = combinedText ?? text
        self.confidence = confidence
        self.timestamp = timestamp
        self.tokenIds = tokenIds
        self.tokenTimings = tokenTimings
        self.lockedTokenCount = lockedTokenCount
    }
}
