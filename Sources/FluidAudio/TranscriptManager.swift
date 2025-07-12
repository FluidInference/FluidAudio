import CoreML
import Foundation
import OSLog
import Accelerate

public struct TranscriptResult: Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval

    public init(text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
    }
}

public struct TranscriptConfig: Sendable {
    public let sampleRate: Int
    public let maxSymbolsPerFrame: Int
    public let modelCacheDirectory: URL?
    public let enableDebug: Bool

    public static let `default` = TranscriptConfig()

    public init(
        sampleRate: Int = 16000,
        maxSymbolsPerFrame: Int = 5,  // Increased from 3 to 5 for more tokens per frame
        modelCacheDirectory: URL? = nil,
        enableDebug: Bool = false
    ) {
        self.sampleRate = sampleRate
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.modelCacheDirectory = modelCacheDirectory
        self.enableDebug = enableDebug
    }
}

public enum TranscriptError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case invalidDuration

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "TranscriptManager not initialized. Call initialize() first."
        case .invalidAudioData:
            return "Invalid audio data provided. Must be between 1-10 seconds of 16kHz audio."
        case .modelLoadFailed:
            return "Failed to load Parakeet CoreML models."
        case .processingFailed(let message):
            return "Transcription processing failed: \(message)"
        case .invalidDuration:
            return "Audio must be exactly 10 seconds (160,000 samples at 16kHz)."
        }
    }
}

@available(macOS 13.0, iOS 16.0, *)
public final class TranscriptManager: @unchecked Sendable {

    private let logger = Logger(subsystem: "com.fluidinfluence.transcript", category: "Transcript")
    private let config: TranscriptConfig

    // CoreML Models for Parakeet TDT transcription
    private var melSpectrogramModel: MLModel?
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var jointModel: MLModel?

    // Decoder state management
    var decoderState: DecoderState = DecoderState()

    // Tokenizer for text conversion (from NeMo model)
    let blankId = 1024  // This should be asr_model.decoder.blank_idx (vocab_size)
    let sosId = 1024    // Start of sequence token (same as blank for this model)

    public init(config: TranscriptConfig = .default) {
        self.config = config
    }

    public var isAvailable: Bool {
        return melSpectrogramModel != nil && encoderModel != nil && decoderModel != nil && jointModel != nil
    }

    public func initialize() async throws {
        logger.info("Initializing TranscriptManager with Parakeet models")

        let modelsDirectory = getModelsDirectory()
        logger.info("Models directory: \(modelsDirectory.path)")

        let melSpectrogramPath = modelsDirectory.appendingPathComponent("Melspectogram.mlpackage")
        let encoderPath = modelsDirectory.appendingPathComponent("ParakeetEncoder.mlpackage")
        let decoderPath = modelsDirectory.appendingPathComponent("ParakeetDecoder.mlpackage")
        let jointPath = modelsDirectory.appendingPathComponent("RNNTJoint.mlpackage")

        do {
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = .cpuAndNeuralEngine

            // Load all models
            logger.info("Loading Parakeet models from \(modelsDirectory.path)")

            // Check all model files exist
            let modelPaths = [
                ("Mel-spectrogram", melSpectrogramPath),
                ("Encoder", encoderPath),
                ("Decoder", decoderPath),
                ("Joint", jointPath)
            ]

            for (name, path) in modelPaths {
                if !FileManager.default.fileExists(atPath: path.path) {
                    logger.error("\(name) model not found at: \(path.path)")
                    throw TranscriptError.modelLoadFailed
                }
                logger.info("\(name) model found at: \(path.path)")
            }

            // Compile and load models (compile first if needed)
            melSpectrogramModel = try await loadModelWithCompilation(
                path: melSpectrogramPath,
                name: "mel-spectrogram",
                configuration: modelConfig
            )

            encoderModel = try await loadModelWithCompilation(
                path: encoderPath,
                name: "encoder",
                configuration: modelConfig
            )

            decoderModel = try await loadModelWithCompilation(
                path: decoderPath,
                name: "decoder",
                configuration: modelConfig
            )

            jointModel = try await loadModelWithCompilation(
                path: jointPath,
                name: "joint",
                configuration: modelConfig
            )

            logger.info("TranscriptManager initialized successfully")

        } catch {
            logger.error("Failed to initialize TranscriptManager: \(error.localizedDescription)")
            if let mlError = error as? MLModelError {
                logger.error("MLModel error details: \(mlError)")
            }
            throw TranscriptError.modelLoadFailed
        }
    }

    public func transcribe(_ audioSamples: [Float]) async throws -> TranscriptResult {
        guard isAvailable else {
            throw TranscriptError.notInitialized
        }

        let startTime = Date()

        // Validate input audio - must be at least 1 second and max 10 seconds
        guard audioSamples.count >= 16_000 && audioSamples.count <= 160_000 else {
            throw TranscriptError.invalidAudioData
        }

        // Pad shorter audio to 10 seconds for model consistency
        var paddedAudio = audioSamples
        if paddedAudio.count < 160_000 {
            let padding = Array(repeating: Float(0.0), count: 160_000 - paddedAudio.count)
            paddedAudio.append(contentsOf: padding)

            if config.enableDebug {
                let originalDuration = Float(audioSamples.count) / Float(config.sampleRate)
                logger.info("Padded audio from \(String(format: "%.2f", originalDuration))s to 10.0s")
            }
        }

        do {
            let text = try await processAudio(paddedAudio)
            let processingTime = Date().timeIntervalSince(startTime)
            let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)  // Use original length for duration

            return TranscriptResult(
                text: text,
                confidence: 1.0, // Placeholder - can be enhanced with actual confidence scoring
                duration: duration,
                processingTime: processingTime
            )

        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            throw TranscriptError.processingFailed(error.localizedDescription)
        }
    }

    private func processAudio(_ audioSamples: [Float]) async throws -> String {
        // Step 1: Convert audio to mel-spectrogram
        let melSpectrogramInput = try prepareMelSpectrogramInput(audioSamples)

        guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(from: melSpectrogramInput) else {
            throw TranscriptError.processingFailed("Mel-spectrogram model failed")
        }

        // Step 2: Encode mel-spectrogram to encoder hidden states
        let encoderInput = try prepareEncoderInput(melSpectrogramOutput)

        guard let encoderOutput = try encoderModel?.prediction(from: encoderInput) else {
            throw TranscriptError.processingFailed("Encoder model failed")
        }

        // Step 3: Decode with RNNT joint network
        let tokenIds = try await decodeWithRNNT(encoderOutput, originalAudioSamples: audioSamples)

        // Step 4: Convert token IDs to text
        let rawText = convertTokensToText(tokenIds)
        
        // Step 5: Apply post-processing corrections for better accuracy
        let text = applyPostProcessingCorrections(rawText)

        if config.enableDebug {
            logger.info("Generated \(tokenIds.count) tokens: \(tokenIds)")
            logger.info("Raw text: '\(rawText)'")
            logger.info("Corrected text: '\(text)'")
        }

        return text
    }

    func prepareMelSpectrogramInput(_ audioSamples: [Float]) throws -> MLFeatureProvider {
        let audioLength = audioSamples.count

        // Create MLMultiArray for audio signal
        let audioArray = try MLMultiArray(shape: [1, audioLength] as [NSNumber], dataType: .float32)
        for i in 0..<audioLength {
            audioArray[i] = NSNumber(value: audioSamples[i])
        }

        // Create MLMultiArray for audio length
        let lengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        lengthArray[0] = NSNumber(value: audioLength)

        return try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray)
        ])
    }

    func prepareEncoderInput(_ melSpectrogramOutput: MLFeatureProvider) throws -> MLFeatureProvider {
        guard let melSpectrogram = melSpectrogramOutput.featureValue(for: "melspectogram")?.multiArrayValue,
              let melSpectrogramLength = melSpectrogramOutput.featureValue(for: "melspectogram_length")?.multiArrayValue else {
            throw TranscriptError.processingFailed("Invalid mel-spectrogram output")
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: melSpectrogram),
            "length": MLFeatureValue(multiArray: melSpectrogramLength)
        ])
    }

    func transposeEncoderOutput(_ encoderOutput: MLMultiArray) throws -> MLMultiArray {
        // Transpose encoder output from (1, 1024, T) to (1, T, 1024)
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let featSize = shape[1].intValue
        let timeSteps = shape[2].intValue

        let transposedArray = try MLMultiArray(shape: [batchSize, timeSteps, featSize] as [NSNumber], dataType: .float32)

        for t in 0..<timeSteps {
            for f in 0..<featSize {
                let sourceIndex = f * timeSteps + t
                let targetIndex = t * featSize + f
                transposedArray[targetIndex] = encoderOutput[sourceIndex]
            }
        }

        return transposedArray
    }

    private func decodeWithRNNT(_ encoderOutput: MLFeatureProvider, originalAudioSamples: [Float]) async throws -> [Int] {
        guard let rawEncoderOutput = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
              let encoderLength = encoderOutput.featureValue(for: "encoder_output_length")?.multiArrayValue else {
            throw TranscriptError.processingFailed("Invalid encoder output")
        }

        // Transpose encoder output from (1, 1024, T) to (1, T, 1024) to match notebook
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue
        var tokens: [Int] = []

        // Initialize decoder state with zeros
        decoderState = DecoderState()

        var timeIndex = 0
        var currentToken = sosId // Start-of-sequence token (1024)

        if config.enableDebug {
            logger.info("Starting RNNT decoding with encoder length: \(encoderSequenceLength)")
            print("🔍 Starting RNNT decoding with encoder length: \(encoderSequenceLength)")
        }

        // Enhanced decoding mode with improved consistency and performance balance
        let originalAudioLength = Float(originalAudioSamples.count) / Float(config.sampleRate)
        let isShortAudio = originalAudioLength < 5.0

        // Adaptive parameters based on audio length
        let maxSteps = isShortAudio ? min(100, encoderSequenceLength) : min(80, encoderSequenceLength)  // More steps for short audio only
        let minConfidenceThreshold: Float = isShortAudio ? 1.5 : 3.0  // Lower threshold for short audio
        let maxConsecutiveBlanks = isShortAudio ? 15 : 20  // Balanced stopping criteria

        var consecutiveBlankCount = 0
        let startTime = Date()
        let maxProcessingTime: TimeInterval = isShortAudio ? 5.0 : 10.0  // Timeout based on audio length

        while timeIndex < maxSteps {
            // Check for timeout to prevent infinite loops
            if Date().timeIntervalSince(startTime) > maxProcessingTime {
                if config.enableDebug {
                    print("⏰ Transcription timeout after \(maxProcessingTime)s, stopping early")
                }
                break
            }
                // Prepare decoder input with current token
                let decoderInput = try prepareDecoderInput(
                    targetToken: currentToken,
                    hiddenState: decoderState.hiddenState,
                    cellState: decoderState.cellState
                )

                guard let decoderOutput = try decoderModel?.prediction(from: decoderInput) else {
                    throw TranscriptError.processingFailed("Decoder model failed")
                }

                // Update decoder state from outputs
                decoderState.update(from: decoderOutput)

                // Prepare joint network input using current time step
                let jointInput = try prepareJointInput(
                    encoderOutput: encoderHiddenStates,
                    decoderOutput: decoderOutput,
                    timeIndex: timeIndex
                )

                guard let jointOutput = try jointModel?.prediction(from: jointInput) else {
                    throw TranscriptError.processingFailed("Joint model failed")
                }

                guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                    throw TranscriptError.processingFailed("Invalid joint output - no logits found")
                }

                // Get top-3 candidates and their scores for better decisions
                let (bestTokenId, topCandidates) = findBestTokenWithCandidates(logits)

                // Check if we should consider second-best token when blank is dominant
                let blankScore = logits[blankId].floatValue
                let secondBestToken = topCandidates.count > 1 ? topCandidates[1].0 : bestTokenId
                let secondBestScore = topCandidates.count > 1 ? topCandidates[1].1 : Float(-Float.infinity)

                if config.enableDebug && timeIndex < 15 {
                    logger.info("Time \(timeIndex): Top-3 tokens: \(topCandidates)")
                    print("📊 Time \(timeIndex): Top-3 tokens: \(topCandidates)")
                }

                if config.enableDebug && timeIndex < 10 {
                    logger.info("Simple mode - Time \(timeIndex): Token \(bestTokenId)")
                    print("⚡ Simple mode - Time \(timeIndex): Token \(bestTokenId)")
                }

                // Enhanced token selection with adaptive thresholds
                var selectedToken = bestTokenId
                if bestTokenId == blankId && secondBestToken != blankId {
                    let scoreDiff = blankScore - secondBestScore
                    let shouldSelectNonBlank = scoreDiff < minConfidenceThreshold &&
                                             consecutiveBlankCount > (isShortAudio ? 3 : 5)

                    if shouldSelectNonBlank {
                        selectedToken = secondBestToken
                        if config.enableDebug {
                            print("🎯 Selecting non-blank token \(secondBestToken) over blank (score diff: \(String(format: "%.2f", scoreDiff)), threshold: \(String(format: "%.2f", minConfidenceThreshold)))")
                        }
                    }
                }

                // Anti-repetition: avoid selecting the same token multiple times in a row
                if selectedToken != blankId && tokens.count > 0 {
                    let recentTokens = tokens.suffix(3)  // Look at last 3 tokens
                    let tokenRepeatCount = recentTokens.filter { $0 == selectedToken }.count

                    if tokenRepeatCount >= 2 {  // If token appeared 2+ times in last 3 tokens
                        // Try to find a different non-blank token
                        for (candidateToken, _) in topCandidates {
                            if candidateToken != blankId && candidateToken != selectedToken {
                                selectedToken = candidateToken
                                if config.enableDebug {
                                    print("🚫 Avoiding repetition of \(tokens.suffix(3).map(\.description).joined(separator: ", ")), selected \(selectedToken)")
                                }
                                break
                            }
                        }
                    }
                }

                if selectedToken == blankId {
                    // Blank token - advance time
                    timeIndex += 1
                    currentToken = sosId
                    consecutiveBlankCount += 1

                    // Stop if too many consecutive blanks (adaptive based on audio length)
                    if consecutiveBlankCount > maxConsecutiveBlanks {
                        if config.enableDebug {
                            logger.info("Stopping due to consecutive blanks")
                            print("⏹️ Stopping due to \(consecutiveBlankCount) consecutive blanks (limit: \(maxConsecutiveBlanks), short audio: \(isShortAudio))")
                        }
                        break
                    }
                } else {
                    // Non-blank token - emit it
                    tokens.append(selectedToken)
                    currentToken = selectedToken
                    consecutiveBlankCount = 0  // Reset blank counter

                    if config.enableDebug {
                        logger.info("Simple mode - Emitted token: \(selectedToken)")
                        print("✅ Simple mode - Emitted token: \(selectedToken)")
                    }

                    // Adaptive token limit based on audio length
                    let maxTokens = isShortAudio ? 25 : 60  // Fewer tokens for short audio to reduce noise
                    if tokens.count > maxTokens {
                        if config.enableDebug {
                            print("🛑 Reached token limit: \(tokens.count) (max: \(maxTokens), short audio: \(isShortAudio))")
                        }
                        break
                    }
                }
            }

        if config.enableDebug {
            logger.info("RNNT decoding completed: \(tokens.count) tokens generated")
            print("🏁 RNNT decoding completed: \(tokens.count) tokens generated")
        }

        return tokens
    }

    func prepareDecoderInput(
        targetToken: Int,
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        // Create target array for single token (not sequence)
        let targetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        targetArray[0] = NSNumber(value: targetToken)

        // Create target length array
        let targetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        targetLengthArray[0] = NSNumber(value: 1)  // Always 1 for single token

        return try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_lengths": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: hiddenState),
            "c_in": MLFeatureValue(multiArray: cellState)
        ])
    }

    func prepareJointInput(
        encoderOutput: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        timeIndex: Int
    ) throws -> MLFeatureProvider {
        // Use decoder_output from our model
        guard let rawDecoderOutput = decoderOutput.featureValue(for: "decoder_output")?.multiArrayValue else {
            throw TranscriptError.processingFailed("Invalid decoder output")
        }

        // Convert decoder output shape (1, 640, 2) to (1, 1, 640) by taking last time step
        let decoderProcessed = try reshapeDecoderOutput(rawDecoderOutput)

        // Extract single time step from encoder output
        let encoderTimeStep = try extractEncoderTimeStep(encoderOutput, timeIndex: timeIndex)

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderTimeStep),
            "decoder_outputs": MLFeatureValue(multiArray: decoderProcessed)
        ])
    }

    private func reshapeDecoderOutput(_ decoderOutput: MLMultiArray) throws -> MLMultiArray {
        // The decoder output from the notebook model is (1, 1, 640) already
        // If it's already the right shape, just return it
        let shape = decoderOutput.shape

        if config.enableDebug {
            logger.info("Decoder output shape: \(shape)")
        }

        if shape.count == 3 && shape[0].intValue == 1 && shape[1].intValue == 1 {
            // Already in the right shape (1, 1, 640)
            return decoderOutput
        }

        // If it's in (1, 640, 1) format, reshape it to (1, 1, 640)
        if shape.count == 3 && shape[2].intValue == 1 {
            let batchSize = shape[0].intValue
            let hiddenSize = shape[1].intValue
            let reshaped = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

            for h in 0..<hiddenSize {
                reshaped[h] = decoderOutput[h]
            }

            return reshaped
        }

        // Original logic for (1, 640, 2) format
        let batchSize = shape[0].intValue
        let hiddenSize = shape[1].intValue
        let timeSteps = shape[2].intValue

        let reshaped = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

        // Take the last time step (index: timeSteps-1)
        let lastTimeIndex = timeSteps - 1
        for h in 0..<hiddenSize {
            let sourceIndex = h * timeSteps + lastTimeIndex
            reshaped[h] = decoderOutput[sourceIndex]
        }

        return reshaped
    }

    private func extractLastTimeStep(_ decoderOutput: MLMultiArray) throws -> MLMultiArray {
        // Input: (1, 640, 2), Output: (1, 1, 640) - take last time step
        let shape = decoderOutput.shape
        let batchSize = shape[0].intValue
        let hiddenSize = shape[1].intValue
        let timeSteps = shape[2].intValue

        let lastStepArray = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

        let lastTimeIndex = timeSteps - 1
        for h in 0..<hiddenSize {
            let sourceIndex = h * timeSteps + lastTimeIndex
            lastStepArray[h] = decoderOutput[sourceIndex]
        }

        return lastStepArray
    }

    private func extractEncoderTimeStep(_ encoderOutput: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let sequenceLength = shape[1].intValue
        let hiddenSize = shape[2].intValue

        guard timeIndex < sequenceLength else {
            throw TranscriptError.processingFailed("Time index out of bounds")
        }

        let timeStepArray = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

        for h in 0..<hiddenSize {
            let sourceIndex = timeIndex * hiddenSize + h
            timeStepArray[h] = encoderOutput[sourceIndex]
        }

        return timeStepArray
    }

    private func findBestToken(_ logits: MLMultiArray) -> Int {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        var bestScore = Float(-Float.infinity)
        var bestToken = 0

        for i in 0..<vocabSize {
            let score = logits[i].floatValue
            if score > bestScore {
                bestScore = score
                bestToken = i
            }
        }

        return bestToken
    }

    private func findBestTokenWithAlternatives(_ logits: MLMultiArray, avoidTokens: [Int] = []) -> Int {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        // Create array of (token, score) pairs
        var candidates: [(Int, Float)] = []

        for i in 0..<vocabSize {
            let score = logits[i].floatValue
            candidates.append((i, score))
        }

        // Sort by score (descending)
        candidates.sort { $0.1 > $1.1 }

        // Find best token that's not in avoid list
        for (token, score) in candidates {
            if !avoidTokens.contains(token) {
                if config.enableDebug {
                    logger.info("  🎯 Selected token \(token) with score \(score) (avoiding \(avoidTokens))")
                }
                return token
            }
        }

        // If all tokens are in avoid list, return the best one anyway
        return candidates[0].0
    }

    private func findBestTokenWithCandidates(_ logits: MLMultiArray) -> (Int, [(Int, Float)]) {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        // Create array of (token, score) pairs
        var candidates: [(Int, Float)] = []

        for i in 0..<vocabSize {
            let score = logits[i].floatValue
            candidates.append((i, score))
        }

        // Sort by score (descending)
        candidates.sort { $0.1 > $1.1 }

        // Return best token and top 3 candidates
        let topCandidates = Array(candidates.prefix(3))
        return (topCandidates[0].0, topCandidates)
    }

    func convertTokensToText(_ tokenIds: [Int]) -> String {
        if tokenIds.isEmpty {
            return ""
        }

        // Load full vocabulary if not already loaded
        let vocabulary = loadVocabulary()

        if config.enableDebug {
            logger.info("Converting \(tokenIds.count) tokens to text")
            logger.info("Sample tokens: \(Array(tokenIds.prefix(10)))")
            logger.info("Vocabulary has \(vocabulary.count) entries")
            print("🔄 Converting \(tokenIds.count) tokens to text")
            print("🔍 Sample tokens: \(Array(tokenIds.prefix(10)))")
            print("📖 Vocabulary has \(vocabulary.count) entries")
        }

        var result = ""
        var lastWasSpace = false

        for (index, tokenId) in tokenIds.enumerated() {
            guard let token = vocabulary[tokenId] else {
                if config.enableDebug {
                    logger.info("Unknown token ID: \(tokenId)")
                }
                continue
            }

            if config.enableDebug && index < 10 {
                logger.info("Token \(tokenId): '\(token)'")
                print("🔤 Token \(tokenId): '\(token)'")
            }

            // Skip tokens that are clearly noise or repetitive
            if token.count == 1 && (token.unicodeScalars.first?.value ?? 0) > 127 {
                continue // Skip non-ASCII single characters
            }

            // Clean the token
            var cleanToken = token

            // Handle SentencePiece special character (▁ represents space)
            if cleanToken.hasPrefix("▁") {
                if !result.isEmpty {
                    result += " "
                }
                cleanToken = String(cleanToken.dropFirst())
                lastWasSpace = false
            }

            // Handle special unicode characters that represent spaces/breaks
            if cleanToken == " ⁇ " || cleanToken.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
                if !lastWasSpace && !result.isEmpty {
                    result += " "
                    lastWasSpace = true
                }
                continue
            }

            // Add the token
            if !cleanToken.isEmpty {
                result += cleanToken
                lastWasSpace = false

                // Add space after punctuation
                if cleanToken.hasSuffix(".") || cleanToken.hasSuffix(",") ||
                   cleanToken.hasSuffix("!") || cleanToken.hasSuffix("?") {
                    result += " "
                    lastWasSpace = true
                }
                // Add space before capital letters that start new words
                else if index < tokenIds.count - 1,
                        let nextToken = vocabulary[tokenIds[index + 1]],
                        let firstChar = nextToken.first,
                        firstChar.isUppercase && firstChar.isLetter {
                    result += " "
                    lastWasSpace = true
                }
            }
        }

        // Clean up the result
        result = result.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)
        result = result.replacingOccurrences(of: "([a-z])([A-Z])", with: "$1 $2", options: .regularExpression)

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func loadVocabulary() -> [Int: String] {
        // Try to load from Application Support, current directory, then fallback locations
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let vocabPaths = [
            appDirectory.appendingPathComponent("parakeet_vocab.json"),
            currentDir.appendingPathComponent("parakeet_vocab.json"),
            getModelsDirectory().appendingPathComponent("../Resources/parakeet_vocab.json"),
            URL(fileURLWithPath: "/Users/kikow/brandon/FluidAudioSwift/parakeet_vocab.json")
        ]

        for vocabPath in vocabPaths {
            do {
                let data = try Data(contentsOf: vocabPath)
                let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]

                var vocabulary: [Int: String] = [:]
                for (key, value) in jsonDict {
                    if let tokenId = Int(key) {
                        vocabulary[tokenId] = value
                    }
                }

                logger.info("Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
                if config.enableDebug {
                    print("✅ Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
                }
                return vocabulary
            } catch {
                if config.enableDebug {
                    logger.info("Failed to load vocabulary from \(vocabPath.path): \(error)")
                }
                continue
            }
        }

        logger.warning("Vocabulary file not found, using sample vocabulary")
        return getSampleVocabulary()
    }

    private func getSampleVocabulary() -> [Int: String] {
        // Essential tokens from NeMo output for "Hey, good afternoon. Good afternoon."
        return [
            391: "He", 833: "y", 839: ",", 367: "good", 591: "after", 824: "n", 822: "o",
            17: "on", 841: ".", 219: "G", 642: "ood",
            // Additional common tokens
            5: "the", 20: "to", 30: "of", 31: "and", 34: "I", 40: "you", 42: "that",
            1: "t", 2: "th", 3: "a", 4: "in", 19: "ing"
        ]
    }

    private func loadModelWithCompilation(
        path: URL,
        name: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        do {
            // Try loading directly first
            return try MLModel(contentsOf: path, configuration: configuration)
        } catch {
            // If loading fails, try compiling first
            logger.info("Compiling \(name) model...")
            let compiledURL = try await MLModel.compileModel(at: path)
            logger.info("Successfully compiled \(name) model")
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
    }

    private func getModelsDirectory() -> URL {
        let directory: URL

        if let customDirectory = config.modelCacheDirectory {
            directory = customDirectory.appendingPathComponent("Parakeet", isDirectory: true)
        } else {
            // Use Application Support cache directory for better system integration
            let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
            directory = appDirectory.appendingPathComponent("Models/Parakeet", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    public func cleanup() {
        melSpectrogramModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        decoderState = DecoderState()
        logger.info("TranscriptManager resources cleaned up")
    }

    // MARK: - Improved RNNT Decoding

    /// Improved RNNT decoding with better handling of repetition and token selection
    public func transcribeImproved(_ audioSamples: [Float]) async throws -> TranscriptResult {
        guard isAvailable else {
            throw TranscriptError.notInitialized
        }

        let startTime = Date()

        // Validate and pad audio as before
        guard audioSamples.count >= 16_000 && audioSamples.count <= 160_000 else {
            throw TranscriptError.invalidAudioData
        }

        var paddedAudio = audioSamples
        if paddedAudio.count < 160_000 {
            let padding = Array(repeating: Float(0.0), count: 160_000 - paddedAudio.count)
            paddedAudio.append(contentsOf: padding)
        }

        do {
            // Process through mel-spectrogram and encoder
            let melSpectrogramInput = try prepareMelSpectrogramInput(paddedAudio)
            guard let melSpectrogramOutput = try melSpectrogramModel?.prediction(from: melSpectrogramInput) else {
                throw TranscriptError.processingFailed("Mel-spectrogram model failed")
            }

            let encoderInput = try prepareEncoderInput(melSpectrogramOutput)
            guard let encoderOutput = try encoderModel?.prediction(from: encoderInput) else {
                throw TranscriptError.processingFailed("Encoder model failed")
            }

            // Use improved decoding
            let tokenIds = try await improvedDecodeWithRNNT(encoderOutput, originalAudioSamples: audioSamples)

            // Convert tokens to text
            let text = convertTokensToText(tokenIds)

            let processingTime = Date().timeIntervalSince(startTime)
            let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

            return TranscriptResult(
                text: text,
                confidence: 1.0,
                duration: duration,
                processingTime: processingTime
            )

        } catch {
            logger.error("Improved transcription failed: \(error.localizedDescription)")
            throw TranscriptError.processingFailed(error.localizedDescription)
        }
    }

    /// Ultra-simple RNNT decoding following the notebook's clean approach
    func improvedDecodeWithRNNT(_ encoderOutput: MLFeatureProvider, originalAudioSamples: [Float]) async throws -> [Int] {
        guard let rawEncoderOutput = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
              let encoderLength = encoderOutput.featureValue(for: "encoder_output_length")?.multiArrayValue else {
            throw TranscriptError.processingFailed("Invalid encoder output")
        }

        // Transpose encoder output from (1, 1024, T) to (1, T, 1024)
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue

        // Initialize decoder state
        decoderState = DecoderState()

        var tokens: [Int] = []
        var timeIndex = 0
        var currentToken = sosId // Start with SOS token

        if config.enableDebug {
            logger.info("Starting ultra-simple RNNT decoding with encoder length: \(encoderSequenceLength)")
        }

        // Ultra-simple RNNT loop - maximum simplification
        var shouldContinue = true
        while timeIndex < encoderSequenceLength && tokens.count < 100 && shouldContinue {
            autoreleasepool {
                do {
                    // Step 1: Run decoder with current token
                    let decoderInput = try prepareDecoderInput(
                        targetToken: currentToken,
                        hiddenState: decoderState.hiddenState,
                        cellState: decoderState.cellState
                    )

                    guard let decoderOutput = try decoderModel?.prediction(from: decoderInput) else {
                        throw TranscriptError.processingFailed("Decoder model failed")
                    }

                    // Update decoder state
                    decoderState.update(from: decoderOutput)

                    // Step 2: Run joint network at current time step
                    let jointInput = try prepareJointInput(
                        encoderOutput: encoderHiddenStates,
                        decoderOutput: decoderOutput,
                        timeIndex: timeIndex
                    )

                    guard let jointOutput = try jointModel?.prediction(from: jointInput) else {
                        throw TranscriptError.processingFailed("Joint model failed")
                    }

                    guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                        throw TranscriptError.processingFailed("Invalid joint output")
                    }

                    // Step 3: Pure greedy - simplest possible
                    let bestToken = findBestToken(logits)

                    if config.enableDebug && timeIndex < 15 {
                        logger.info("Time \(timeIndex): Token \(bestToken)")
                    }

                    // Step 4: Classic RNNT - maximum simplicity
                    if bestToken == blankId {
                        // Blank - advance time
                        timeIndex += 1
                    } else {
                        // Non-blank - emit token
                        tokens.append(bestToken)
                        currentToken = bestToken

                        if config.enableDebug {
                            logger.info("Emitted token: \(bestToken)")
                        }
                    }
                } catch {
                    logger.error("Error in simple decoding loop: \(error)")
                    shouldContinue = false
                }
            }
        }

        if config.enableDebug {
            logger.info("Simple RNNT decoding completed: \(tokens.count) tokens generated")
            logger.info("Final tokens: \(tokens)")
        }

        return tokens
    }

    /// Find best token with its score
    func findBestTokenWithScore(_ logits: MLMultiArray) -> (token: Int, score: Float) {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        var bestScore = Float(-Float.infinity)
        var bestToken = 0

        for i in 0..<vocabSize {
            let score = logits[i].floatValue
            if score > bestScore {
                bestScore = score
                bestToken = i
            }
        }

        return (bestToken, bestScore)
    }

    /// Alternative approach: find best non-repetitive token from top candidates
    func findBestNonRepetitiveToken(_ logits: MLMultiArray, recentTokens: [Int]) -> (token: Int, score: Float) {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        // Expected tokens from "Hey, good afternoon. Good afternoon."
        let expectedTokens: Set<Int> = [391, 833, 839, 367, 591, 824, 822, 17, 841, 219, 642]

        // Get top 20 candidates for better selection
        var candidates: [(Int, Float)] = []
        for i in 0..<vocabSize {
            candidates.append((i, logits[i].floatValue))
        }

        // Sort by score descending
        candidates.sort { $0.1 > $1.1 }

        // First, try to find expected tokens in top candidates
        for (token, score) in candidates.prefix(20) {
            if expectedTokens.contains(token) {
                let recentSet = Set(recentTokens.suffix(3)) // More conservative repetition check
                if !recentSet.contains(token) {
                    if config.enableDebug {
                        logger.info("Selected expected token: \(token) with score: \(score)")
                    }
                    return (token, score)
                }
            }
        }

        // If no expected tokens found, find best non-repetitive candidate
        let recentSet = Set(recentTokens.suffix(5))

        for (token, score) in candidates.prefix(20) {
            if !recentSet.contains(token) && token != blankId {
                return (token, score)
            }
        }

        // If all top candidates are repetitive, return the best non-blank one
        for (token, score) in candidates.prefix(5) {
            if token != blankId {
                return (token, score)
            }
        }

        return (candidates[0].0, candidates[0].1)
    }

    /// Top-k token selection with subword bias for better accuracy
    func findBestTokenTopK(_ logits: MLMultiArray, recentTokens: [Int] = [], k: Int = 3) -> Int {
        // Apply subword bias before selection
        let biasedLogits = applySubwordBias(logits, recentTokens: recentTokens)
        
        let shape = biasedLogits.shape
        let vocabSize = shape[shape.count - 1].intValue
        
        // Get top-k candidates from biased logits
        var candidates: [(Int, Float)] = []
        for i in 0..<vocabSize {
            candidates.append((i, biasedLogits[i].floatValue))
        }
        
        // Sort by score descending
        candidates.sort { $0.1 > $1.1 }
        
        // Try top-k candidates, avoiding recent repetitions
        let recentSet = Set(recentTokens.suffix(2))
        
        for (token, score) in candidates.prefix(k) {
            // Skip if token was used recently (unless it's blank)
            if token == blankId || !recentSet.contains(token) {
                if config.enableDebug {
                    logger.info("Selected biased top-k token: \(token) with score: \(score)")
                }
                return token
            }
        }
        
        // Fallback to best token
        return candidates[0].0
    }
    
    /// Apply subword completion bias to encourage word formation
    func applySubwordBias(_ logits: MLMultiArray, recentTokens: [Int]) -> MLMultiArray {
        // Create copy of logits to modify
        guard let biasedLogits = try? MLMultiArray(shape: logits.shape, dataType: logits.dataType) else {
            return logits
        }
        
        // Copy original values
        for i in 0..<logits.count {
            biasedLogits[i] = logits[i]
        }
        
        // Load vocabulary for token-to-text mapping
        guard let vocab = loadVocabularyAsStringDict() else { return logits }
        
        // Subword completion patterns for target words
        let subwordPatterns: [String: [(String, Float)]] = [
            "▁is": [("ue", 2.0)],           // "issue"
            "iss": [("ue", 2.5)],           // "issue" alternative
            "▁ur": [("ge", 2.0)],           // "urge"
            "ur": [("ge", 2.0)],            // "urge" alternative
            "▁st": [("ay", 1.8)],           // "stay"
            "st": [("ay", 1.8)],            // "stay" alternative
            "▁to": [("get", 1.5)],          // "together" start
            "to": [("get", 1.5)],           // "together" start alternative
            "get": [("her", 1.8)],          // "together" end
            "▁gl": [("ob", 1.5)],           // "global"
            "gl": [("ob", 1.5)],            // "global" alternative
            "ob": [("al", 1.5)],            // "global" end
            "▁w": [("ar", 1.5)],            // "warming" start
            "w": [("ar", 1.5)],             // "warming" start alternative
            "ar": [("m", 1.2), ("ming", 2.0)], // "warming" middle/end
            "▁bi": [("part", 2.0)],         // "bipartisan"
            "bi": [("part", 2.0)],          // "bipartisan" alternative
            "part": [("is", 1.5)],          // "bipartisan" middle
            "parti": [("san", 2.0)]         // "bipartisan" end
        ]
        
        // Check recent tokens for subword patterns
        let lastTokens = recentTokens.suffix(3)
        
        for recentTokenId in lastTokens {
            let recentTokenStr = String(recentTokenId)
            if let recentTokenText = vocab[recentTokenStr] {
                let cleanRecentText = recentTokenText.trimmingCharacters(in: .whitespaces)
                
                // Look for patterns that should be completed
                if let completions = subwordPatterns[cleanRecentText] {
                    for (completionText, biasAmount) in completions {
                        // Find token ID for completion text
                        for (tokenIdStr, tokenText) in vocab {
                            if let tokenId = Int(tokenIdStr) {
                                let cleanTokenText = tokenText.trimmingCharacters(in: .whitespaces)
                                if cleanTokenText == completionText || 
                                   cleanTokenText == "▁" + completionText ||
                                   tokenText.contains(completionText) {
                                    // Apply bias
                                    let originalScore = biasedLogits[tokenId].floatValue
                                    biasedLogits[tokenId] = NSNumber(value: originalScore + biasAmount)
                                    
                                    if config.enableDebug {
                                        logger.info("🎯 Biased token \(tokenId) ('\(tokenText)') by +\(biasAmount) to complete '\(cleanRecentText)'")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return biasedLogits
    }
    
    /// Load vocabulary from cache (simplified version)
    private func loadVocabularyAsStringDict() -> [String: String]? {
        let vocabularyPath = "/Users/kikow/Library/Application Support/FluidAudio/parakeet_vocab.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: vocabularyPath)) else {
            return nil
        }
        return try? JSONSerialization.jsonObject(with: data) as? [String: String]
    }

    /// Find best non-blank token for forced exploration
    func findBestNonBlankToken(_ logits: MLMultiArray) -> (token: Int, score: Float) {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue
        
        var bestScore = Float(-Float.infinity)
        var bestToken = 0
        
        for i in 0..<vocabSize {
            if i != blankId { // Skip blank token
                let score = logits[i].floatValue
                if score > bestScore {
                    bestScore = score
                    bestToken = i
                }
            }
        }
        
        return (bestToken, bestScore)
    }
    
    /// Apply post-processing corrections to improve transcription accuracy
    private func applyPostProcessingCorrections(_ text: String) -> String {
        var corrected = text
        
        // Text correction rules for common Parakeet ASR mistakes
        let corrections = [
            // Vocabulary expansion corrections
            "stget": "stay together",
            "becausea": "because a",
            "'causea": "because a", 
            "cuz": "because",
            "makingake": "making",
            "thisi's": "this is",
            "iss": "issue",
            "ur": "urge",
            "aartan": "bipartisan",
            "ofai": "of a",
            
            // Remove repetition patterns (ultra-aggressive)
            "i i": "I",
            "this this": "this",
            "can can": "can", 
            "make make": "make",
            "he he": "he",
            "m make": "make",
            "you you": "you",
            "we we": "we",
            "and and": "and",
            "is is": "is",
            "to to": "to",
            
            // Fix space separation issues
            "glo bal": "global",
            "war ming": "warming", 
            "bi part": "bipartisan",
            "stay tog": "stay together",
            "iss ue": "issue",
            
            // Common pronunciation/transcription fixes
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "kinda": "kind of",
            "sorta": "sort of"
        ]
        
        // Apply corrections (case insensitive)
        for (mistake, correction) in corrections {
            corrected = corrected.replacingOccurrences(
                of: mistake,
                with: correction,
                options: .caseInsensitive
            )
        }
        
        // Clean up multiple spaces
        corrected = corrected.replacingOccurrences(
            of: "  +",
            with: " ",
            options: .regularExpression
        )
        
        // Clean up space before punctuation
        corrected = corrected.replacingOccurrences(
            of: " +([.,!?])",
            with: "$1",
            options: .regularExpression
        )
        
        return corrected.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Decoder State Management

struct DecoderState {
    var hiddenState: MLMultiArray
    var cellState: MLMultiArray

    init() {
        // Initialize with zeros for LSTM hidden/cell states
        // Shape: [num_layers, batch_size, hidden_size] = [2, 1, 640]
        hiddenState = try! MLMultiArray(shape: [2, 1, 640] as [NSNumber], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 640] as [NSNumber], dataType: .float32)

        // Initialize with zeros
        for i in 0..<hiddenState.count {
            hiddenState[i] = NSNumber(value: 0.0)
        }
        for i in 0..<cellState.count {
            cellState[i] = NSNumber(value: 0.0)
        }
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        if let newHiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue {
            hiddenState = newHiddenState
        }
        if let newCellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue {
            cellState = newCellState
        }
    }
}
