import AVFoundation
import CoreML
import Foundation
import OSLog

public enum AudioSource: Sendable {
    case microphone
    case system
}

public final class AsrManager {

    internal let logger = AppLogger(category: "ASR")
    internal let config: ASRConfig
    private let audioConverter: AudioConverter = AudioConverter()

    internal var preprocessorModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?

    /// The AsrModels instance if initialized with models
    private var asrModels: AsrModels?

    /// Token duration optimization model

    /// Cached vocabulary loaded once during initialization
    internal var vocabulary: [Int: String] = [:]
    #if DEBUG
    // Test-only setter
    internal func setVocabularyForTesting(_ vocab: [Int: String]) {
        vocabulary = vocab
    }
    #endif

    /// Custom vocabulary context for context biasing
    private var customVocabulary: CustomVocabularyContext?
    /// Cached CTC keyword spotter to avoid per-call initialization cost
    private var cachedCtcKeywordSpotter: CtcKeywordSpotter?

    // TODO:: the decoder state should be moved higher up in the API interface
    internal var microphoneDecoderState: TdtDecoderState
    internal var systemDecoderState: TdtDecoderState

    // Cached prediction options for reuse
    internal lazy var predictionOptions: MLPredictionOptions = {
        AsrModels.optimizedPredictionOptions()
    }()

    public init(config: ASRConfig = .default) {
        self.config = config

        self.microphoneDecoderState = TdtDecoderState.make()
        self.systemDecoderState = TdtDecoderState.make()

        // Pre-warm caches if possible
        Task {
            await sharedMLArrayCache.prewarm(shapes: [
                ([NSNumber(value: 1), NSNumber(value: 240_000)], .float32),
                ([NSNumber(value: 1)], .int32),
                (
                    [
                        NSNumber(value: 2),
                        NSNumber(value: 1),
                        NSNumber(value: ASRConstants.decoderHiddenSize),
                    ], .float32
                ),
            ])
        }
    }

    public var isAvailable: Bool {
        let baseModelsReady = encoderModel != nil && decoderModel != nil && jointModel != nil
        guard baseModelsReady else { return false }

        if asrModels?.usesSplitFrontend == true {
            return preprocessorModel != nil
        }

        return true
    }

    /// Initialize ASR Manager with pre-loaded models
    /// - Parameter models: Pre-loaded ASR models
    public func initialize(models: AsrModels) async throws {
        logger.info("Initializing AsrManager with provided models")

        self.asrModels = models
        self.preprocessorModel = models.preprocessor
        self.encoderModel = models.encoder
        self.decoderModel = models.decoder
        self.jointModel = models.joint
        self.vocabulary = models.vocabulary

        logger.info("Token duration optimization model loaded successfully")

        logger.info("AsrManager initialized successfully with provided models")
    }

    /// Update custom vocabulary for context biasing without reinitializing ASR
    /// - Parameter vocabulary: New custom vocabulary context, or nil to disable context biasing
    public func setCustomVocabulary(_ vocabulary: CustomVocabularyContext?) {
        // Tokenize terms for CTC keyword spotting if they don't have ctcTokenIds
        if var vocab = vocabulary {
            vocab = tokenizeVocabularyForCtc(vocab)
            self.customVocabulary = vocab
            logger.info(
                "Custom vocabulary updated: \(vocab.terms.count) terms, "
                    + "thresholds: similarity=\(String(format: "%.2f", vocab.minSimilarity)), "
                    + "combined=\(String(format: "%.2f", vocab.minCombinedConfidence))")
        } else {
            self.customVocabulary = nil
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
                // The text field remains the canonical form (for replacement), but ctcTokenIds
                // are derived from the alias spelling (for acoustic detection).
                if let aliases = term.aliases {
                    for alias in aliases {
                        let aliasTokenIds = tokenizer.encode(alias)
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

            logger.info(
                "CTC tokenization: \(tokenizedCount) terms tokenized, \(aliasExpansionCount) alias expansions, total \(expandedTerms.count) CTC entries"
            )

            return CustomVocabularyContext(
                terms: expandedTerms,
                alpha: vocabulary.alpha,
                contextScore: vocabulary.contextScore,
                depthScaling: vocabulary.depthScaling,
                scorePerPhrase: vocabulary.scorePerPhrase,
                minCtcScore: vocabulary.minCtcScore,
                minSimilarity: vocabulary.minSimilarity,
                minCombinedConfidence: vocabulary.minCombinedConfidence
            )
        } catch {
            logger.warning("Failed to tokenize vocabulary for CTC: \(error.localizedDescription)")
            return vocabulary
        }
    }

    /// Tokenize vocabulary terms for TDT beam search (uses Parakeet TDT vocabulary).
    /// This is needed for beam search vocabulary biasing which operates on TDT token IDs.
    private func tokenizeVocabularyForTdt(_ vocabulary: CustomVocabularyContext) -> CustomVocabularyContext {
        // Skip if already tokenized
        let needsTokenization = vocabulary.terms.contains { $0.tokenIds == nil }
        guard needsTokenization else { return vocabulary }

        var updatedTerms: [CustomVocabularyTerm] = []
        var tokenizedCount = 0

        for term in vocabulary.terms {
            if let existingTokenIds = term.tokenIds {
                // Already has TDT tokens
                updatedTerms.append(term)
            } else {
                // Tokenize using TDT vocabulary
                let tokenIds = tokenizeTextForTdt(term.text)
                let updatedTerm = CustomVocabularyTerm(
                    text: term.text,
                    weight: term.weight,
                    aliases: term.aliases,
                    tokenIds: tokenIds.isEmpty ? nil : tokenIds,
                    ctcTokenIds: term.ctcTokenIds
                )
                updatedTerms.append(updatedTerm)
                if !tokenIds.isEmpty {
                    tokenizedCount += 1
                    logger.debug("TDT tokenized '\(term.text)': \(tokenIds)")
                }

                // Also tokenize aliases to allow boosting them during beam search
                if let aliases = term.aliases {
                    for alias in aliases {
                        let aliasTokens = tokenizeTextForTdt(alias)
                        if !aliasTokens.isEmpty {
                            // Add alias as a separate term, pointing to the same text/weight
                            // Note: We keep the text as the canonical form, but this term effectively
                            // boosts the alias path in the Trie. Beam search will output the alias tokens.
                            // Post-processing (CTC/Rescorer) is needed to map it back if desired,
                            // or we rely on the user providing the alias because they accept it as output.
                            let aliasTerm = CustomVocabularyTerm(
                                text: term.text,
                                weight: term.weight,
                                aliases: nil,  // Prevent recursion
                                tokenIds: aliasTokens,
                                ctcTokenIds: nil
                            )
                            updatedTerms.append(aliasTerm)
                            logger.debug("TDT tokenized alias '\(alias)' -> '\(term.text)': \(aliasTokens)")
                        }
                    }
                }
            }
        }

        if tokenizedCount > 0 {
            logger.info("TDT tokenization: \(tokenizedCount) terms tokenized for beam search")
        }

        return CustomVocabularyContext(
            terms: updatedTerms,
            alpha: vocabulary.alpha,
            contextScore: vocabulary.contextScore,
            depthScaling: vocabulary.depthScaling,
            scorePerPhrase: vocabulary.scorePerPhrase,
            minCtcScore: vocabulary.minCtcScore,
            minSimilarity: vocabulary.minSimilarity,
            minCombinedConfidence: vocabulary.minCombinedConfidence
        )
    }

    /// Tokenize text using TDT vocabulary (reverse lookup from vocabulary dictionary)
    private func tokenizeTextForTdt(_ text: String) -> [Int] {
        // Use simple subword tokenization based on the loaded vocabulary
        // The vocabulary maps token_id -> text, we need text -> token_id
        guard !vocabulary.isEmpty else { return [] }

        // Build reverse mapping (text -> tokenId) if not cached
        let reverseVocab = buildReverseVocabulary()

        // Add leading space (SentencePiece convention) but preserve case
        // The model outputs case-sensitive tokens, so we need case-sensitive matching
        let normalizedText = " " + text

        var tokens: [Int] = []
        var position = normalizedText.startIndex

        while position < normalizedText.endIndex {
            var bestMatch: (token: String, id: Int)? = nil
            var bestLength = 0

            // Try to find longest matching token at current position
            for (tokenText, tokenId) in reverseVocab {
                let remaining = String(normalizedText[position...])
                if remaining.hasPrefix(tokenText) && tokenText.count > bestLength {
                    bestMatch = (tokenText, tokenId)
                    bestLength = tokenText.count
                }
            }

            if let match = bestMatch {
                tokens.append(match.id)
                position = normalizedText.index(position, offsetBy: match.token.count)
            } else {
                // Skip single character if no match found
                position = normalizedText.index(after: position)
            }
        }

        return tokens
    }

    /// Build reverse vocabulary mapping (token_text -> token_id)
    private func buildReverseVocabulary() -> [String: Int] {
        var reverse: [String: Int] = [:]
        for (id, text) in vocabulary {
            reverse[text] = id
        }
        return reverse
    }

    /// Get current custom vocabulary
    public func getCustomVocabulary() -> CustomVocabularyContext? {
        return customVocabulary
    }

    private func createFeatureProvider(
        features: [(name: String, array: MLMultiArray)]
    ) throws
        -> MLFeatureProvider
    {
        var featureDict: [String: MLFeatureValue] = [:]
        for (name, array) in features {
            featureDict[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: featureDict)
    }

    internal func createScalarArray(
        value: Int, shape: [NSNumber] = [1], dataType: MLMultiArrayDataType = .int32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }

    func preparePreprocessorInput(
        _ audioSamples: [Float], actualLength: Int? = nil
    ) async throws
        -> MLFeatureProvider
    {
        let audioLength = audioSamples.count
        let actualAudioLength = actualLength ?? audioLength  // Use provided actual length or default to sample count

        // Use ANE-aligned array from cache
        let audioArray = try await sharedMLArrayCache.getArray(
            shape: [1, audioLength] as [NSNumber],
            dataType: .float32
        )

        // Use optimized memory copy
        audioSamples.withUnsafeBufferPointer { buffer in
            let destPtr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: audioLength)
            memcpy(destPtr, buffer.baseAddress!, audioLength * MemoryLayout<Float>.stride)
        }

        // Pass the actual audio length, not the padded length
        let lengthArray = try createScalarArray(value: actualAudioLength)

        return try createFeatureProvider(features: [
            ("audio_signal", audioArray),
            ("audio_length", lengthArray),
        ])
    }

    private func prepareDecoderInput(
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try createScalarArray(value: 0, shape: [1, 1])
        let targetLengthArray = try createScalarArray(value: 1)

        return try createFeatureProvider(features: [
            ("targets", targetArray),
            ("target_length", targetLengthArray),
            ("h_in", hiddenState),
            ("c_in", cellState),
        ])
    }

    internal func initializeDecoderState(decoderState: inout TdtDecoderState) async throws {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        // Reset the existing decoder state to clear all cached values including predictorOutput
        decoderState.reset()

        let initDecoderInput = try prepareDecoderInput(
            hiddenState: decoderState.hiddenState,
            cellState: decoderState.cellState
        )

        let initDecoderOutput = try await decoderModel.compatPrediction(
            from: initDecoderInput,
            options: predictionOptions
        )

        decoderState.update(from: initDecoderOutput)

    }

    private func loadModel(
        path: URL,
        name: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        do {
            let model = try MLModel(contentsOf: path, configuration: configuration)
            return model
        } catch {
            logger.error("Failed to load \(name) model: \(error)")

            throw ASRError.modelLoadFailed
        }
    }
    private static func getDefaultModelsDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent(
            "FluidAudio", isDirectory: true)
        let directory = appDirectory.appendingPathComponent("Models/Parakeet", isDirectory: true)

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    public func resetState() {
        microphoneDecoderState = TdtDecoderState.make()
        systemDecoderState = TdtDecoderState.make()
    }

    public func cleanup() {
        preprocessorModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        // Reset decoder states using fresh allocations for deterministic behavior
        microphoneDecoderState = TdtDecoderState.make()
        systemDecoderState = TdtDecoderState.make()
        logger.info("AsrManager resources cleaned up")
    }

    internal func tdtDecodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        originalAudioSamples: [Float],
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0,
        customVocabulary: CustomVocabularyContext? = nil
    ) async throws -> TdtHypothesis {
        // Use beam search if enabled and jointSingleStep model is available
        if config.useBeamSearch,
            let jointSingleStepModel = asrModels?.jointSingleStep,
            let customVocabulary = customVocabulary
        {
            logger.info("Using beam search decoding with vocabulary biasing (frame-by-frame)")
            return try await beamSearchDecode(
                encoderOutput: encoderOutput,
                jointSingleStepModel: jointSingleStepModel,
                customVocabulary: customVocabulary,
                decoderState: &decoderState,
                globalFrameOffset: globalFrameOffset
            )
        }

        // Route to appropriate decoder based on model version
        switch asrModels!.version {
        case .v2:
            let decoder = TdtDecoderV2(config: config)
            return try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualAudioFrames,
                decoderModel: decoderModel!,
                jointModel: jointModel!,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset,
                customVocabulary: customVocabulary
            )
        case .v3:
            let decoder = TdtDecoderV3(config: config)
            return try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualAudioFrames,
                decoderModel: decoderModel!,
                jointModel: jointModel!,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )
        }
    }

    /// Beam search decoding with vocabulary biasing using frame-by-frame JointDecisionSingleStep model
    private func beamSearchDecode(
        encoderOutput: MLMultiArray,
        jointSingleStepModel: MLModel,
        customVocabulary: CustomVocabularyContext,
        decoderState: inout TdtDecoderState,
        globalFrameOffset: Int
    ) async throws -> TdtHypothesis {
        // Tokenize vocabulary using TDT tokens if not already tokenized
        let tokenizedVocabulary = tokenizeVocabularyForTdt(customVocabulary)

        // Build vocabulary trie for biasing
        let trie = VocabularyTrie(vocabulary: tokenizedVocabulary)
        logger.debug("Beam search trie has \(trie.count) entries")

        // Create beam search decoder with vocabulary biasing
        let beamDecoder = BeamSearchDecoder(
            config: config.beamSearchConfig,
            vocabularyTrie: trie
        )

        // Get initial decoder states
        let initialH = decoderState.hiddenState
        let initialC = decoderState.cellState

        // Run beam search decoding with frame-by-frame joint model
        let (tokens, timestamps) = try beamDecoder.decode(
            encoderOutput: encoderOutput,
            jointSingleStepModel: jointSingleStepModel,
            decoderModel: decoderModel!,
            initialHState: initialH,
            initialCState: initialC
        )

        // Convert to TdtHypothesis
        var hypothesis = TdtHypothesis(decState: decoderState)
        hypothesis.ySequence = tokens
        hypothesis.timestamps = timestamps.map { $0 + globalFrameOffset }
        hypothesis.tokenConfidences = [Float](repeating: 0.9, count: tokens.count)  // Beam search doesn't track per-token confidence
        hypothesis.lastToken = tokens.last

        logger.info("Beam search decoded \(tokens.count) tokens with vocabulary biasing (frame-by-frame)")

        return hypothesis
    }

    /// Transcribe audio from an AVAudioPCMBuffer.
    ///
    /// Performs speech-to-text transcription on the provided audio buffer. The decoder state is automatically
    /// reset after transcription completes, ensuring each transcription call is independent. This enables
    /// efficient batch processing where multiple files are transcribed without state carryover.
    ///
    /// - Parameters:
    ///   - audioBuffer: The audio buffer to transcribe
    ///   - source: The audio source type (microphone or system audio)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioBuffer: AVAudioPCMBuffer,
        source: AudioSource = .microphone,
        customVocabulary: CustomVocabularyContext? = nil
    ) async throws -> ASRResult {
        let audioFloatArray = try audioConverter.resampleBuffer(audioBuffer)

        let result = try await transcribe(audioFloatArray, source: source, customVocabulary: customVocabulary)

        return result
    }

    /// Transcribe audio from a file URL.
    ///
    /// Performs speech-to-text transcription on the audio file at the provided URL. The decoder state is
    /// automatically reset after transcription completes, ensuring each transcription call is independent.
    ///
    /// - Parameters:
    ///   - url: The URL to the audio file
    ///   - source: The audio source type (defaults to .system)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails, models are not initialized, or the file cannot be read
    public func transcribe(
        _ url: URL,
        source: AudioSource = .system,
        customVocabulary: CustomVocabularyContext? = nil
    ) async throws -> ASRResult {
        let audioFloatArray = try audioConverter.resampleAudioFile(url)

        let result = try await transcribe(audioFloatArray, source: source, customVocabulary: customVocabulary)

        return result
    }

    /// Transcribe audio from raw float samples.
    ///
    /// Performs speech-to-text transcription on raw audio samples at 16kHz. The decoder state is
    /// automatically reset after transcription completes, ensuring each transcription call is independent
    /// and enabling efficient batch processing of multiple audio files.
    ///
    /// - Parameters:
    ///   - audioSamples: Array of 16-bit audio samples at 16kHz
    ///   - source: The audio source type (microphone or system audio)
    /// - Returns: An ASRResult containing the transcribed text and token timings
    /// - Throws: ASRError if transcription fails or models are not initialized
    public func transcribe(
        _ audioSamples: [Float],
        source: AudioSource = .microphone,
        customVocabulary: CustomVocabularyContext? = nil
    ) async throws -> ASRResult {
        // Use parameter if provided, otherwise fall back to stored vocabulary
        let effectiveVocabulary = customVocabulary ?? self.customVocabulary

        var result: ASRResult
        switch source {
        case .microphone:
            result = try await transcribeWithState(
                audioSamples, decoderState: &microphoneDecoderState, customVocabulary: effectiveVocabulary)
        case .system:
            result = try await transcribeWithState(
                audioSamples, decoderState: &systemDecoderState, customVocabulary: effectiveVocabulary)
        }

        // Optional CTC keyword boosting and metrics
        if let effectiveVocabulary, !effectiveVocabulary.terms.isEmpty {
            let debug = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"
            logger.info("CTC boost: using vocabulary with \(effectiveVocabulary.terms.count) terms")
            do {
                let spotter = try await getKeywordSpotter()

                // CTC model can only process ~15 seconds (240,000 samples) at a time
                // For longer audio, process in overlapping chunks and merge detections
                let maxChunkSamples = 240_000
                let chunkOverlap = 32_000  // 2 seconds overlap to catch words at boundaries
                let sampleRate = 16000.0
                var allDetections: [CtcKeywordSpotter.KeywordDetection] = []
                var spotResult: CtcKeywordSpotter.SpotKeywordsResult?

                if audioSamples.count <= maxChunkSamples {
                    // Short audio - use new API that returns log-probs for principled rescoring
                    let result = try await spotter.spotKeywordsWithLogProbs(
                        audioSamples: audioSamples,
                        customVocabulary: effectiveVocabulary,
                        minScore: effectiveVocabulary.minCtcScore
                    )
                    allDetections = result.detections
                    spotResult = result  // Keep log-probs for principled rescoring
                } else {
                    // Long audio - process in overlapping chunks
                    // Overlap ensures words at chunk boundaries are detected
                    let chunkStep = maxChunkSamples - chunkOverlap
                    var chunkStart = 0
                    var chunkIndex = 0

                    // Time-aware deduplication: track (term, timeWindow) -> best detection
                    // Time window of 2 seconds groups duplicate detections from overlapping chunks
                    let deduplicationWindowSeconds = 2.0
                    var bestDetectionsByTermAndTime: [String: CtcKeywordSpotter.KeywordDetection] = [:]

                    while chunkStart < audioSamples.count {
                        let chunkEnd = min(chunkStart + maxChunkSamples, audioSamples.count)
                        let chunk = Array(audioSamples[chunkStart..<chunkEnd])

                        // Calculate time offset for this chunk
                        let chunkTimeOffset = Double(chunkStart) / sampleRate

                        if debug && chunkIndex % 20 == 0 {
                            logger.debug(
                                "CTC chunk \(chunkIndex): samples \(chunkStart)-\(chunkEnd), offset \(String(format: "%.1f", chunkTimeOffset))s"
                            )
                        }

                        let chunkDetections = try await spotter.spotKeywords(
                            audioSamples: chunk,
                            customVocabulary: effectiveVocabulary,
                            minScore: effectiveVocabulary.minCtcScore
                        )

                        // Adjust detection times and deduplicate by (term, time_window)
                        for detection in chunkDetections {
                            // Create adjusted detection with absolute timestamps
                            let adjustedStartTime = detection.startTime + chunkTimeOffset
                            let adjustedEndTime = detection.endTime + chunkTimeOffset

                            // Create deduplication key: term + quantized time window
                            let timeWindow = Int(adjustedStartTime / deduplicationWindowSeconds)
                            let dedupeKey = "\(detection.term.text)@\(timeWindow)"

                            // Keep the detection with the highest score for each (term, time_window)
                            // Prefer detections WITH aliases when scores are close (within 1.0)
                            if let existing = bestDetectionsByTermAndTime[dedupeKey] {
                                let hasNewAliases = detection.term.aliases != nil && !detection.term.aliases!.isEmpty
                                let hasExistingAliases =
                                    existing.term.aliases != nil && !existing.term.aliases!.isEmpty
                                let scoreDiff = detection.score - existing.score

                                // Replace if: (a) significantly better score, OR
                                //             (b) similar score but new has aliases and existing doesn't
                                let shouldReplace =
                                    scoreDiff > 1.0
                                    || (scoreDiff > -1.0 && hasNewAliases && !hasExistingAliases)

                                if shouldReplace {
                                    // Replace with higher-scoring detection (adjusted times)
                                    let adjusted = CtcKeywordSpotter.KeywordDetection(
                                        term: detection.term,
                                        score: detection.score,
                                        totalFrames: detection.totalFrames,
                                        startFrame: detection.startFrame + Int(Double(chunkStart) / sampleRate * 100),
                                        endFrame: detection.endFrame + Int(Double(chunkStart) / sampleRate * 100),
                                        startTime: adjustedStartTime,
                                        endTime: adjustedEndTime
                                    )
                                    bestDetectionsByTermAndTime[dedupeKey] = adjusted
                                }
                            } else {
                                // First detection for this (term, time_window)
                                let adjusted = CtcKeywordSpotter.KeywordDetection(
                                    term: detection.term,
                                    score: detection.score,
                                    totalFrames: detection.totalFrames,
                                    startFrame: detection.startFrame + Int(Double(chunkStart) / sampleRate * 100),
                                    endFrame: detection.endFrame + Int(Double(chunkStart) / sampleRate * 100),
                                    startTime: adjustedStartTime,
                                    endTime: adjustedEndTime
                                )
                                bestDetectionsByTermAndTime[dedupeKey] = adjusted
                            }
                        }

                        chunkStart += chunkStep
                        chunkIndex += 1
                    }

                    allDetections = Array(bestDetectionsByTermAndTime.values)
                    logger.info(
                        "CTC boost: processed \(chunkIndex) chunks, found \(allDetections.count) unique detections")
                }

                let detections = allDetections

                let detectedTerms = detections.map { $0.term.text }

                if debug {
                    logger.info("CTC boost: detected \(detections.count) keywords")
                }

                if !detections.isEmpty {
                    // Use VocabularyRescorer for principled CTC-based rescoring
                    // This compares acoustic evidence for original words vs vocabulary terms
                    // and only replaces when vocabulary term has significantly higher score
                    let rescorer = VocabularyRescorer(
                        spotter: spotter,
                        vocabulary: effectiveVocabulary
                    )

                    // Use principled scoring when we have cached log-probs (short audio),
                    // fall back to legacy API for chunked audio (log-probs not available)
                    let rescoreOutput: VocabularyRescorer.RescoreOutput
                    if let spotResult = spotResult {
                        // Short audio: use principled CTC scoring with cached log-probs
                        rescoreOutput = rescorer.rescore(
                            transcript: result.text,
                            spotResult: spotResult
                        )
                    } else {
                        // Long audio (chunked): fall back to heuristic scoring
                        // Log-probs from different chunks can't be easily combined
                        let legacyResult = CtcKeywordSpotter.SpotKeywordsResult(
                            detections: detections,
                            logProbs: [],
                            frameDuration: 0,
                            totalFrames: 0
                        )
                        rescoreOutput = rescorer.rescore(
                            transcript: result.text,
                            spotResult: legacyResult
                        )
                    }

                    let appliedTerms = rescoreOutput.replacements
                        .filter { $0.shouldReplace }
                        .compactMap { $0.replacementWord }
                    let loweredCorrected = rescoreOutput.text.lowercased()
                    let filteredDetected =
                        detectedTerms
                        .filter { loweredCorrected.contains($0.lowercased()) }

                    // Convert rescoring replacements to WordCorrection with character positions
                    let corrections = Self.computeWordCorrectionsFromRescoring(
                        from: rescoreOutput.replacements.filter { $0.shouldReplace },
                        in: rescoreOutput.text
                    )

                    // Always log correction info for debugging
                    let replacementCount = rescoreOutput.replacements.filter { $0.shouldReplace }.count
                    logger.info(
                        "CTC rescore: \(replacementCount) replacements → \(corrections.count) corrections")
                    if debug {
                        for replacement in rescoreOutput.replacements {
                            let action = replacement.shouldReplace ? "REPLACED" : "KEPT"
                            logger.info(
                                "  [\(action)] '\(replacement.originalWord)' → '\(replacement.replacementWord ?? "-")': \(replacement.reason)"
                            )
                        }
                    }

                    result = ASRResult(
                        text: rescoreOutput.text,
                        confidence: result.confidence,
                        duration: result.duration,
                        processingTime: result.processingTime,
                        tokenTimings: result.tokenTimings,
                        performanceMetrics: result.performanceMetrics,
                        ctcDetectedTerms: filteredDetected.isEmpty ? nil : filteredDetected,
                        ctcAppliedTerms: appliedTerms.isEmpty ? nil : appliedTerms,
                        corrections: corrections.isEmpty ? nil : corrections
                    )
                } else if !detectedTerms.isEmpty {
                    let loweredText = result.text.lowercased()
                    let filteredDetected = detectedTerms.filter { loweredText.contains($0.lowercased()) }
                    result = ASRResult(
                        text: result.text,
                        confidence: result.confidence,
                        duration: result.duration,
                        processingTime: result.processingTime,
                        tokenTimings: result.tokenTimings,
                        performanceMetrics: result.performanceMetrics,
                        ctcDetectedTerms: filteredDetected.isEmpty ? nil : filteredDetected,
                        ctcAppliedTerms: nil
                    )
                }
            } catch {
                logger.warning("CTC keyword boost failed: \(error.localizedDescription)")
                print("[ERROR] CTC keyword boost failed: \(error)")
            }
        }

        // Stateless architecture: reset decoder state after each transcription to ensure
        // independent processing for batch operations without state carryover
        try await self.resetDecoderState()

        return result
    }

    private func getKeywordSpotter() async throws -> CtcKeywordSpotter {
        if let cachedCtcKeywordSpotter {
            return cachedCtcKeywordSpotter
        }
        let spotter = try await CtcKeywordSpotter.makeDefault()
        cachedCtcKeywordSpotter = spotter
        return spotter
    }

    /// Convert KeywordMerger replacements to WordCorrection with character positions
    /// - Parameters:
    ///   - replacements: Replacements from KeywordMerger
    ///   - correctedText: The final corrected text
    /// - Returns: Array of WordCorrection with character ranges
    private static func computeWordCorrections(
        from replacements: [KeywordMerger.MergeResult.Replacement],
        in correctedText: String
    ) -> [WordCorrection] {
        guard !replacements.isEmpty else { return [] }

        var corrections: [WordCorrection] = []
        var searchStartIndex = correctedText.startIndex

        // Sort replacements by word index to process in order
        let sortedReplacements = replacements.sorted { $0.wordIndex < $1.wordIndex }

        for replacement in sortedReplacements {
            let canonical = replacement.canonicalText

            // Find the canonical text in the corrected text starting from searchStartIndex
            if let range = correctedText.range(
                of: canonical,
                options: .caseInsensitive,
                range: searchStartIndex..<correctedText.endIndex
            ) {
                let startOffset = correctedText.distance(from: correctedText.startIndex, to: range.lowerBound)
                let endOffset = correctedText.distance(from: correctedText.startIndex, to: range.upperBound)

                corrections.append(
                    WordCorrection(
                        range: startOffset..<endOffset,
                        original: replacement.originalText,
                        corrected: canonical
                    ))

                // Move search start past this match to handle duplicates correctly
                searchStartIndex = range.upperBound
            }
        }

        return corrections
    }

    /// Convert VocabularyRescorer replacements to WordCorrection with character positions
    /// - Parameters:
    ///   - replacements: Replacements from VocabularyRescorer
    ///   - correctedText: The corrected transcript text
    /// - Returns: Array of WordCorrection objects with character ranges
    private static func computeWordCorrectionsFromRescoring(
        from replacements: [VocabularyRescorer.RescoringResult],
        in correctedText: String
    ) -> [WordCorrection] {
        guard !replacements.isEmpty else { return [] }

        var corrections: [WordCorrection] = []
        var searchStartIndex = correctedText.startIndex

        for replacement in replacements {
            guard let replacementWord = replacement.replacementWord else { continue }

            // Find the replacement word in the corrected text starting from searchStartIndex
            if let range = correctedText.range(
                of: replacementWord,
                options: .caseInsensitive,
                range: searchStartIndex..<correctedText.endIndex
            ) {
                let startOffset = correctedText.distance(from: correctedText.startIndex, to: range.lowerBound)
                let endOffset = correctedText.distance(from: correctedText.startIndex, to: range.upperBound)

                corrections.append(
                    WordCorrection(
                        range: startOffset..<endOffset,
                        original: replacement.originalWord,
                        corrected: replacementWord
                    ))

                // Move search start past this match to handle duplicates correctly
                searchStartIndex = range.upperBound
            }
        }

        return corrections
    }

    // Reset both decoder states
    public func resetDecoderState() async throws {
        try await resetDecoderState(for: .microphone)
        try await resetDecoderState(for: .system)
    }

    /// Reset the decoder state for a specific audio source
    /// This should be called when starting a new transcription session or switching between different audio files
    public func resetDecoderState(for source: AudioSource) async throws {
        switch source {
        case .microphone:
            try await initializeDecoderState(decoderState: &microphoneDecoderState)
        case .system:
            try await initializeDecoderState(decoderState: &systemDecoderState)
        }
    }

    internal func normalizedTimingToken(_ token: String) -> String {
        token.replacingOccurrences(of: "▁", with: " ")
    }

    internal func convertTokensWithExistingTimings(
        _ tokenIds: [Int], timings: [TokenTiming]
    ) -> (
        text: String, timings: [TokenTiming]
    ) {
        guard !tokenIds.isEmpty else { return ("", []) }

        // SentencePiece-compatible decoding algorithm:
        // 1. Convert token IDs to token strings
        var tokens: [String] = []
        var tokenInfos: [(token: String, tokenId: Int, timing: TokenTiming?)] = []

        for (index, tokenId) in tokenIds.enumerated() {
            if let token = vocabulary[tokenId], !token.isEmpty {
                tokens.append(token)
                let timing = index < timings.count ? timings[index] : nil
                tokenInfos.append((token: token, tokenId: tokenId, timing: timing))
            }
        }

        // 2. Concatenate all tokens (this is how SentencePiece works)
        let concatenated = tokens.joined()

        // 3. Replace ▁ with space (SentencePiece standard)
        let text = concatenated.replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)

        // 4. For now, return original timings as-is
        // Note: Proper timing alignment would require tracking character positions
        // through the concatenation and replacement process
        let adjustedTimings = tokenInfos.compactMap { info in
            info.timing.map { timing in
                TokenTiming(
                    token: normalizedTimingToken(info.token),
                    tokenId: info.tokenId,
                    startTime: timing.startTime,
                    endTime: timing.endTime,
                    confidence: timing.confidence
                )
            }
        }

        return (text, adjustedTimings)
    }

    internal func extractFeatureValue(
        from provider: MLFeatureProvider, key: String, errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }

    internal func extractFeatureValues(
        from provider: MLFeatureProvider, keys: [(key: String, errorSuffix: String)]
    ) throws -> [String: MLMultiArray] {
        var results: [String: MLMultiArray] = [:]
        for (key, errorSuffix) in keys {
            results[key] = try extractFeatureValue(
                from: provider, key: key, errorMessage: "Invalid \(errorSuffix)")
        }
        return results
    }
}
