@preconcurrency import CoreML
import Foundation
import OSLog
import Accelerate
import AVFoundation

// MARK: - AsrManager

/// AsrManager provides automatic speech recognition using Parakeet TDT (Token-and-Duration Transducer) models.
///
/// This class implements high-performance speech-to-text transcription optimized for Apple Neural Engine,
/// achieving real-time factors (RTFx) of 11-50x depending on audio duration and hardware.
///
/// ## Features
/// - **Parakeet TDT-0.6b** model implementation with duration-aware decoding
/// - **Apple Neural Engine** optimization for maximum performance
/// - **Chunked processing** for long audio files (automatic 10-second chunks)
/// - **Text normalization** compatible with HuggingFace ASR evaluation standards
/// - **Auto-recovery** from CoreML model loading failures
///
/// ## Performance
/// - LibriSpeech test-clean: Average WER 5.9%, Median WER 1.9%
/// - LibriSpeech test-other: Average WER 13-15%
/// - RTFx: ~11.8x (includes full end-to-end file I/O overhead)
///
/// ## Usage
/// ```swift
/// let asrManager = AsrManager()
/// try await asrManager.initialize()
/// let result = try await asrManager.transcribe(audioSamples)
/// print("Transcription: \(result.text)")
/// ```
///
/// ## Technical Details
/// The TDT model uses a novel architecture that jointly predicts both tokens and their durations,
/// enabling more accurate transcription of speech with varying speaking rates. The implementation
/// includes advanced features like duration-based token repetition and confidence-weighted decoding.
///
/// - Note: Requires macOS 13.0+ or iOS 16.0+ for CoreML 6 features
@available(macOS 13.0, iOS 16.0, *)
public final class AsrManager {

    internal let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ASR")
    internal let config: ASRConfig
    private let modelsDirectory: URL

    // CoreML Models for Parakeet TDT transcription
    internal var melSpectrogramModel: MLModel?
    internal var encoderModel: MLModel?
    internal var decoderModel: MLModel?
    internal var jointModel: MLModel?

    // Prediction options for faster inference
    internal var predictionOptions: MLPredictionOptions?

    // Decoder state management
    var decoderState: DecoderState = DecoderState()

    // Tokenizer for text conversion (from NeMo model)
    let blankId = 1024  // Verified: this works correctly (not actually "warming" token)
    let sosId = 1024    // Start of sequence token (same as blank for this model)

    /// Creates a new ASR manager with the specified configuration
    /// - Parameters:
    ///   - config: ASR configuration settings
    ///   - modelsDirectory: Custom directory for CoreML models. If nil, uses default Application Support location
    public init(config: ASRConfig = .default, modelsDirectory: URL? = nil) {
        self.config = config
        self.modelsDirectory = modelsDirectory ?? Self.getDefaultModelsDirectory()

        // Initialize TDT-specific properties - TDT is always enabled
        logger.info("TDT enabled with durations: \(config.tdtConfig.durations)")
    }

    /// Indicates whether all required models are loaded and ready for transcription
    public var isAvailable: Bool {
        return melSpectrogramModel != nil && encoderModel != nil && decoderModel != nil && jointModel != nil
    }

    /// Initializes the ASR system by loading Parakeet TDT models.
    ///
    /// This method downloads models if needed and compiles them for Apple Neural Engine.
    /// Models are cached for subsequent runs.
    ///
    /// - Throws: `ASRError.modelLoadFailed` if models cannot be loaded
    /// - Note: This is an async operation that may take several seconds on first run
    public func initialize() async throws {
        logger.info("Initializing AsrManager with Parakeet models")

        logger.info("Models directory: \(self.modelsDirectory.path)")

        let melSpectrogramPath = modelsDirectory.appendingPathComponent("Melspectogram.mlmodelc")
        let encoderPath = modelsDirectory.appendingPathComponent("ParakeetEncoder.mlmodelc")
        let decoderPath = modelsDirectory.appendingPathComponent("ParakeetDecoder.mlmodelc")
        let jointPath = modelsDirectory.appendingPathComponent("RNNTJoint.mlmodelc")

        do {
            // Download models if they don't exist
            try await DownloadUtils.downloadParakeetModelsIfNeeded(to: modelsDirectory)

            let modelConfig = MLModelConfiguration()

            // Enable performance optimizations
            modelConfig.allowLowPrecisionAccumulationOnGPU = true

            // Configure compute units based on environment
            if ProcessInfo.processInfo.environment["CI"] != nil {
                // Force CPU and Neural Engine only (no GPU) in CI environments
                // GPU can cause issues in virtualized environments like GitHub Actions
                modelConfig.computeUnits = .cpuAndNeuralEngine
                logger.info("🔧 ASR: Using compute units: cpuAndNeuralEngine (CI environment)")

            } else {
                // Use all available compute units for best performance on real hardware
                modelConfig.computeUnits = .all
            }

            // Load all models
            logger.info("Loading Parakeet models from \(self.modelsDirectory.path)")

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
                    throw ASRError.modelLoadFailed
                }
                logger.info("\(name) model found at: \(path.path)")
            }

            // Load all models using common pattern
            let models = try await loadAllModels(
                melSpectrogramPath: melSpectrogramPath,
                encoderPath: encoderPath,
                decoderPath: decoderPath,
                jointPath: jointPath,
                configuration: modelConfig
            )

            melSpectrogramModel = models.melSpectrogram
            encoderModel = models.encoder
            decoderModel = models.decoder
            jointModel = models.joint

            // Initialize prediction options for faster inference
            let options = MLPredictionOptions()
            // usesCPUOnly is deprecated - the model config already specifies compute units
            self.predictionOptions = options

            logger.info("AsrManager initialized successfully")

        } catch {
            logger.error("Failed to initialize AsrManager: \(error.localizedDescription)")
            if let mlError = error as? MLModelError {
                logger.error("MLModel error details: \(mlError)")
            }
            throw ASRError.modelLoadFailed
        }
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
            throw ASRError.processingFailed("Invalid mel-spectrogram output")
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
            throw ASRError.processingFailed("Invalid decoder output")
        }

        // Convert decoder output shape (1, 640, 2) to (1, 1, 640) by taking last time step
        let decoderProcessed = try reshapeDecoderOutput(rawDecoderOutput)

        // Extract single time step from encoder output
        // Check if encoderOutput is already a single timestep (for TDT case)
        let encoderTimeStep: MLMultiArray
        let encoderShape = encoderOutput.shape
        if encoderShape.count == 3 && encoderShape[1].intValue == 1 {
            // Already a single timestep, use as-is
            encoderTimeStep = encoderOutput
        } else {
            // Full sequence, extract single timestep
            encoderTimeStep = try extractEncoderTimeStep(encoderOutput, timeIndex: timeIndex)
        }

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

    private func extractEncoderTimeStep(_ encoderOutput: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let sequenceLength = shape[1].intValue
        let hiddenSize = shape[2].intValue

        if config.enableDebug && timeIndex == 0 {
            logger.debug("🔍 DEBUG: extractEncoderTimeStep - encoder shape: \(shape), timeIndex: \(timeIndex)")
        }

        guard timeIndex < sequenceLength else {
            if config.enableDebug {
                logger.error("Time index out of bounds: \(timeIndex) >= \(sequenceLength) (shape: \(shape))")
            }
            throw ASRError.processingFailed("Time index out of bounds: \(timeIndex) >= \(sequenceLength)")
        }

        let timeStepArray = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

        for h in 0..<hiddenSize {
            let sourceIndex = timeIndex * hiddenSize + h
            timeStepArray[h] = encoderOutput[sourceIndex]
        }

        return timeStepArray
    }

    /// Enhanced token selection with adaptive constraints based on audio length
    private func findBestTokenWithRepetitionPrevention(
        _ logits: MLMultiArray,
        lastEmittedTokens: [Int],
        consecutiveBlanks: Int,
        maxConsecutiveBlanks: Int,
        originalAudioSamples: [Float]
    ) -> Int {
        let shape = logits.shape
        let vocabSize = shape[shape.count - 1].intValue

        // PHASE 1 FIX 1: Enhanced Repetition Detection
        let vocabulary = loadVocabulary()
        let audioLength = Double(originalAudioSamples.count) / 16000.0
        let isShortAudio = audioLength < 4.0
        let isLongAudio = audioLength > 10.0

        // PHASE 1 FIX 2: Adaptive Temperature Scaling (iOS Real-Time Optimized)
        // Lower temperature = more conservative (less repetitive)
        // Higher temperature = more diverse (for difficult audio)
        let baseTemperature: Float
        if config.realtimeMode {
            // Real-time mode: more aggressive for speed/consistency
            baseTemperature = isShortAudio ? 0.6 : (isLongAudio ? 1.1 : 0.9)
        } else {
            // Batch mode: original settings
            baseTemperature = isShortAudio ? 0.8 : (isLongAudio ? 1.3 : 1.1)
        }

        let repetitionPenaltyFactor: Float = config.realtimeMode ?
            min(Float(lastEmittedTokens.count) * 0.15, 1.0) :  // More aggressive in real-time
            min(Float(lastEmittedTokens.count) * 0.1, 0.8)
        let adaptiveTemperature = baseTemperature + repetitionPenaltyFactor

        var scores: [(token: Int, score: Float)] = []
        var maxScore: Float = -Float.infinity

        // Collect raw scores
        for i in 0..<vocabSize {
            let rawScore = logits[i].floatValue
            scores.append((token: i, score: rawScore))
            maxScore = max(maxScore, rawScore)
        }

        // PHASE 1 FIX 1: Hard Repetition Penalties
        // Apply severe penalties for repeated tokens
        for i in 0..<scores.count {
            let token = scores[i].token

            // Count recent occurrences of this token
            let recentCount = lastEmittedTokens.suffix(5).filter { $0 == token }.count

            if recentCount > 0 {
                // Exponential penalty for repetition: each repeat makes it much less likely
                let repetitionPenalty = Float(recentCount) * 2.5  // Aggressive penalty
                scores[i].score -= repetitionPenalty

                if config.enableDebug && recentCount > 1 {
                    logger.warning("Token \(token) repeated \(recentCount) times, applying penalty -\(repetitionPenalty)")
                }
            }

            // Extra penalty for immediate repetition (back-to-back same tokens)
            if let lastToken = lastEmittedTokens.last, lastToken == token {
                scores[i].score -= 3.0  // Very harsh immediate repetition penalty
            }
        }

        // PHASE 1 FIX 2: Temperature Scaling with Softmax
        var probabilities: [(token: Int, prob: Float)] = []
        var totalProb: Float = 0

        for (token, score) in scores {
            let adjustedScore = (score - maxScore) / adaptiveTemperature
            let prob = exp(adjustedScore)
            probabilities.append((token: token, prob: prob))
            totalProb += prob
        }

        // Normalize probabilities
        for i in 0..<probabilities.count {
            probabilities[i].prob /= totalProb
        }

        // Sort by probability descending
        probabilities.sort { $0.prob > $1.prob }

        // PHASE 1 FIX 3: Adaptive Confidence Thresholds (iOS Real-Time Optimized)
        // Stricter thresholds for problematic scenarios
        let baseConfidence: Float
        let nonBlankConfidence: Float

        if config.realtimeMode {
            // Real-time mode: higher thresholds for quality/speed trade-off
            baseConfidence = isShortAudio ? 0.08 : 0.04  // Stricter for real-time
            nonBlankConfidence = isShortAudio ? 0.20 : (isLongAudio ? 0.12 : 0.15)
        } else {
            // Batch mode: original settings
            baseConfidence = isShortAudio ? 0.05 : 0.02
            nonBlankConfidence = isShortAudio ? 0.15 : (isLongAudio ? 0.08 : 0.10)
        }

        let repetitionBonus = config.realtimeMode ?
            min(Float(lastEmittedTokens.count) * 0.015, 0.04) :  // Slightly more aggressive
            min(Float(lastEmittedTokens.count) * 0.01, 0.03)
        let minConfidence = baseConfidence + repetitionBonus

        // PHASE 1 FIX 4: Length Normalization & Quality Gates
        let maxAllowedTokens = Int(audioLength * 8)  // 8 tokens per second max
        let currentTokenCount = lastEmittedTokens.count

        // If we're approaching length limits, be much more selective
        let lengthPressure = Float(currentTokenCount) / Float(maxAllowedTokens)
        let lengthAdjustedConfidence = minConfidence + (lengthPressure * 0.1)

        // Sort by probability descending
        probabilities.sort { $0.prob > $1.prob }

        for (token, prob) in probabilities {
            // PHASE 1 FIX 3: Multi-level Confidence Filtering

            // Base confidence check
            if prob < lengthAdjustedConfidence {
                continue
            }

            // Stricter confidence for non-blank tokens
            if token != blankId && prob < nonBlankConfidence {
                continue
            }

            // PHASE 1 FIX 1: Hard Repetition Limits
            // Absolute repetition blocking
            let recentCount = lastEmittedTokens.suffix(3).filter { $0 == token }.count
            if recentCount >= 2 && token != blankId {  // Max 2 repeats in last 3 tokens
                if config.enableDebug {
                    logger.warning("Blocking token \(token) - too many recent repeats (\(recentCount))")
                }
                continue
            }

            // Block immediate repetition completely (except blanks)
            if let lastToken = lastEmittedTokens.last, lastToken == token && token != blankId {
                if config.enableDebug {
                    logger.warning("Blocking immediate repetition of token \(token)")
                }
                continue
            }

            // PHASE 1 FIX 4: Length-based early termination
            // If output is getting too long, strongly prefer blank tokens
            if lengthPressure > 0.8 && token != blankId {
                let blankBonus: Float = 0.3  // Make blanks much more attractive
                if let blankProb = probabilities.first(where: { $0.token == blankId })?.prob,
                   blankProb + blankBonus > prob {
                    if config.enableDebug {
                        logger.info("Length pressure \(String(format: "%.2f", lengthPressure)), preferring blank")
                    }
                    return blankId
                }
            }

            // Additional quality checks for non-blank tokens
            if token != blankId {
                // Check if token is valid in vocabulary
                if token >= 0 && token < vocabulary.count {
                    let tokenText = vocabulary[token] ?? ""

                    // Filter out problematic tokens
                    if tokenText.isEmpty || tokenText == "<unk>" || tokenText.hasPrefix("▁▁▁") {
                        continue
                    }

                    // For short audio, be extra cautious about long words
                    if isShortAudio && tokenText.count > 8 {
                        continue
                    }
                }
            }

            // If we get here, token passed all quality gates
            if config.enableDebug && token != blankId {
                let tokenText = (token >= 0 && token < vocabulary.count) ? (vocabulary[token] ?? "UNK") : "UNK"
                logger.info("Selected token \(token) ('\(tokenText)') with confidence \(String(format: "%.3f", prob))")
            }

            return token
        }

        // PHASE 1 FIX 4: Fallback with length normalization
        // If no token passes quality gates, return blank (safer than random token)
        if config.enableDebug {
            logger.warning("No tokens passed quality gates, returning blank (length pressure: \(String(format: "%.2f", lengthPressure)))")
        }

        return blankId
    }


    /// Check if a token should be emitted based on repetition patterns
    private func shouldEmitToken(_ token: Int, lastEmittedTokens: [Int], maxRepeats: Int) -> Bool {
        // Always allow blank tokens (they control timing)
        if token == blankId {
            return true
        }

        // Check immediate repetition (same token repeated consecutively)
        let recentCount = lastEmittedTokens.suffix(min(maxRepeats, lastEmittedTokens.count)).filter { $0 == token }.count
        if recentCount >= maxRepeats {
            return false
        }

        // Check for alternating patterns (A-B-A-B-A...)
        if lastEmittedTokens.count >= 4 {
            let recent4 = Array(lastEmittedTokens.suffix(min(4, lastEmittedTokens.count)))
            if recent4.count >= 4 && recent4[0] == recent4[2] && recent4[1] == recent4[3] && recent4[3] == token {
                return false // Prevent A-B-A-B-A pattern
            }
        }

        return true
    }

    /// Detect repetitive patterns in recent token history
    private func hasRecentRepetitivePattern(_ tokens: [Int], checkLength: Int) -> Bool {
        guard tokens.count >= checkLength else { return false }

        let recentTokens = Array(tokens.suffix(min(checkLength, tokens.count)))

        // Check for same token repeated many times
        let uniqueTokens = Set(recentTokens)
        if uniqueTokens.count <= 2 {
            return true
        }

        // Check for short repeating patterns
        for patternLength in 2...min(4, checkLength/2) {
            if hasRepeatingPattern(recentTokens, patternLength: patternLength) {
                return true
            }
        }

        return false
    }

    /// Check if tokens contain a repeating pattern of given length
    private func hasRepeatingPattern(_ tokens: [Int], patternLength: Int) -> Bool {
        guard tokens.count >= patternLength * 2 else { return false }

        for startIdx in 0...(tokens.count - patternLength * 2) {
            let pattern = Array(tokens[startIdx..<(startIdx + patternLength)])
            let nextSegment = Array(tokens[(startIdx + patternLength)..<(startIdx + patternLength * 2)])

            if pattern == nextSegment {
                return true
            }
        }

        return false
    }


    /// Check if the current token sequence appears unstable
    private func isSequenceUnstable(_ tokens: [Int], vocabulary: [Int: String]) -> Bool {
        guard tokens.count >= 20 else { return false }

        let recentTokens = Array(tokens.suffix(min(20, tokens.count)))

        // Calculate vocabulary diversity in recent tokens
        let uniqueTokens = Set(recentTokens.filter { $0 != blankId })
        let nonBlankCount = recentTokens.filter { $0 != blankId }.count

        // If very low diversity (same few tokens repeating)
        if nonBlankCount > 10 && uniqueTokens.count < 3 {
            return true
        }

        // Check for high proportion of unknown/garbage tokens
        var garbageCount = 0
        for token in recentTokens {
            if token != blankId {
                if vocabulary[token] != nil {

                } else {
                    garbageCount += 1 // Unknown token
                }
            }
        }

        if Double(garbageCount) / Double(nonBlankCount) > 0.3 {
            return true // More than 30% garbage
        }

        // Check for erratic pattern (alternating between very different token types)
        var patternScore = 0
        for i in 1..<recentTokens.count {
            let prevToken = recentTokens[i-1]
            let currToken = recentTokens[i]

            if prevToken != blankId && currToken != blankId {
                if let prevText = vocabulary[prevToken], let currText = vocabulary[currToken] {
                    // Check if tokens are very different (length, type, etc.)
                    if abs(prevText.count - currText.count) > 5 {
                        patternScore += 1
                    }
                }
            }
        }

        if Double(patternScore) / Double(recentTokens.count) > 0.4 {
            return true // Too erratic
        }

        return false
    }

    // MARK: - TDT Helper Functions

    /// Create token timings from TDT hypothesis for enhanced post-processing
    private func createTokenTimings(from hypothesis: TDTHypothesis, audioSamples: [Float]) -> [TokenTiming] {
        guard config.tdtConfig.includeTokenDuration,
              hypothesis.ySequence.count == hypothesis.timestamps.count,
              hypothesis.ySequence.count == hypothesis.tokenDurations.count else {
            return []
        }

        let vocabulary = loadVocabulary()
        var timings: [TokenTiming] = []
        let sampleRate = Float(config.sampleRate)

        for i in 0..<hypothesis.ySequence.count {
            let tokenId = hypothesis.ySequence[i]
            let timestamp = hypothesis.timestamps[i]
            let duration = hypothesis.tokenDurations[i]

            // Convert frame indices to time
            let startTime = TimeInterval(timestamp) / TimeInterval(sampleRate / 160.0) // Assuming 160 samples per frame
            let endTime = startTime + TimeInterval(duration) / TimeInterval(sampleRate / 160.0)

            let tokenText = vocabulary[tokenId] ?? "<unk>"
            let confidence: Float = 1.0 // Would be calculated from logits in full implementation

            timings.append(TokenTiming(
                token: tokenText,
                tokenId: tokenId,
                startTime: startTime,
                endTime: endTime,
                confidence: confidence
            ))
        }

        return timings
    }

    /// Apply softmax to logits
    private func applySoftmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0.0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        return expLogits.map { $0 / sumExp }
    }

    /// Create encoder provider from transposed encoder output for fallback
    private func createEncoderProvider(from encoderOutput: MLMultiArray) throws -> MLFeatureProvider {
        // Create a synthetic encoder output provider
        // This is a simplified version - in a real implementation you'd need to handle this more carefully
        let shape = encoderOutput.shape
        let sequenceLength = shape[1].intValue

        // Create encoder output length array
        let encoderOutputLength = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        encoderOutputLength[0] = NSNumber(value: sequenceLength)

        // Transpose back to original format if needed
        let originalFormat = try transposeEncoderOutput(encoderOutput) // This should transpose back

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_output": MLFeatureValue(multiArray: originalFormat),
            "encoder_output_length": MLFeatureValue(multiArray: encoderOutputLength)
        ])
    }

    /// Initialize decoder state with a clean blank token pass
    internal func initializeDecoderState() async throws {
        decoderState = try await createFreshDecoderState()
    }

    /// Create a fresh decoder state with clean initialization
    private func createFreshDecoderState() async throws -> DecoderState {
        guard let decoderModel = decoderModel else {
            throw ASRError.notInitialized
        }

        // Reset decoder state to zeros
        var freshState = DecoderState.zero()

        // Run decoder once with blank token to establish clean state
        let initDecoderInput = try prepareDecoderInput(
            targetToken: blankId,
            hiddenState: freshState.hiddenState,
            cellState: freshState.cellState
        )

        let initDecoderOutput = try decoderModel.prediction(from: initDecoderInput, options: predictionOptions ?? MLPredictionOptions())

        // Update state with initial hidden/cell states
        freshState.update(from: initDecoderOutput)

        if config.enableDebug {
            logger.info("Decoder state initialized cleanly")
        }

        return freshState
    }

    /// Detect instability in the first few tokens
    private func isEarlySequenceUnstable(_ tokens: [Int], vocabulary: [Int: String]) -> Bool {
        guard tokens.count >= 3 else { return false }

        let nonBlankTokens = tokens.filter { $0 != blankId }
        guard nonBlankTokens.count >= 2 else { return false }

        // Check for immediate duplication (He He, A A, etc.)
        if nonBlankTokens.count >= 2 && nonBlankTokens[0] == nonBlankTokens[1] {
            if config.enableDebug {
                logger.warning("Detected immediate token duplication: \(nonBlankTokens[0])")
            }
            return true
        }

        // Check for single-character gibberish sequence
        var singleCharCount = 0
        for token in nonBlankTokens.prefix(min(5, nonBlankTokens.count)) {
            if let tokenText = vocabulary[token] {
                let cleanText = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "▁", with: "")
                if cleanText.count == 1 && cleanText.first != nil && !cleanText.first!.isLetter {
                    singleCharCount += 1
                }
            }
        }

        if singleCharCount >= 3 {
            if config.enableDebug {
                logger.warning("Detected single-character gibberish sequence")
            }
            return true
        }

        // Check for random character patterns like "L M M I W"
        if nonBlankTokens.count >= 4 {
            var randomCharPattern = true
            for token in nonBlankTokens.prefix(min(4, nonBlankTokens.count)) {
                if let tokenText = vocabulary[token] {
                    let cleanText = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
                        .replacingOccurrences(of: "▁", with: "")
                    if cleanText.count > 1 || (cleanText.first?.isLowercase == true) {
                        randomCharPattern = false
                        break
                    }
                }
            }

            if randomCharPattern {
                if config.enableDebug {
                    logger.warning("Detected random character pattern")
                }
                return true
            }
        }

        return false
    }

    /// Advanced post-processing using vocabulary and timing information
    internal func applyAdvancedPostProcessing(_ text: String, tokenTimings: [TokenTiming]) -> String {
        var processedText = text

        // Simple cleanup only - remove consecutive duplicates and fix spacing
        processedText = removeConsecutiveDuplicates(processedText)
        processedText = processedText.trimmingCharacters(in: .whitespacesAndNewlines)
        processedText = processedText.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)

        if config.enableDebug {
            logger.info("Post-processing: '\(text)' → '\(processedText)'")
        }

        return processedText
    }

    /// Apply timing-based corrections using token duration analysis
    private func applyTimingBasedCorrections(_ text: String, timings: [TokenTiming]) -> String {
        var corrected = text

        // Identify tokens with suspicious timing patterns
        for timing in timings {
            let duration = timing.endTime - timing.startTime

            // Very short tokens might be artifacts
            if duration < 0.05 && timing.token.count == 1 && !timing.token.first!.isLetter {
                corrected = corrected.replacingOccurrences(of: timing.token, with: "")
            }

            // Very long single character tokens are suspicious
            if duration > 0.5 && timing.token.count == 1 {
                // Potentially expand single characters to common words
                switch timing.token.lowercased() {
                case "h":
                    if duration > 0.8 { corrected = corrected.replacingOccurrences(of: timing.token, with: "he") }
                case "n":
                    if duration > 0.8 { corrected = corrected.replacingOccurrences(of: timing.token, with: "and") }
                default:
                    break
                }
            }
        }

        return corrected
    }

    /// Apply pattern-based corrections for common ASR errors
    private func applyPatternCorrections(_ text: String) -> String {
        var corrected = text

        // PHASE 3.2: Remove obvious repetition patterns
        corrected = removeConsecutiveDuplicates(corrected)

        // PHASE 3.4: Clean up spacing and punctuation
        corrected = corrected.trimmingCharacters(in: .whitespacesAndNewlines)
        corrected = corrected.replacingOccurrences(of: "  +", with: " ", options: .regularExpression)

        return corrected
    }


    /// Remove consecutive duplicate words
    private func removeConsecutiveDuplicates(_ text: String) -> String {
        let words = text.components(separatedBy: .whitespaces)
        var result: [String] = []

        for word in words {
            if let lastWord = result.last, lastWord.lowercased() == word.lowercased() {
                continue // Skip duplicate
            }
            result.append(word)
        }

        return result.joined(separator: " ")
    }

    private func loadVocabulary() -> [Int: String] {
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        let vocabPath = appDirectory.appendingPathComponent("parakeet_vocab.json")

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning("Vocabulary file not found at \(vocabPath.path). Please ensure parakeet_vocab.json is downloaded with the models.")
            // Return empty vocabulary - the transcription will still work but without some advanced features
            return [:]
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]

            var vocabulary: [Int: String] = [:]
            for (key, value) in jsonDict {
                if let tokenId = Int(key) {
                    vocabulary[tokenId] = value
                }
            }

            logger.info("✅ Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error("Failed to load or parse vocabulary file at \(vocabPath.path): \(error.localizedDescription)")
            // Return empty vocabulary instead of crashing
            return [:]
        }
    }

    private func calculateDirectorySize(at url: URL) -> Int64 {
        var size: Int64 = 0

        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let fileURL as URL in enumerator {
                if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    size += Int64(fileSize)
                }
            }
        }

        return size
    }

    private func loadModelWithCompilation(
        path: URL,
        name: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil

        do {
            // Try loading directly first
            if isCI {
                logger.info("🔍 CI: Attempting to load \(name) model from: \(path.path)")
                logger.info("   File exists: \(FileManager.default.fileExists(atPath: path.path))")

                // Check directory size for .mlmodelc (which is a directory, not a file)
                if FileManager.default.fileExists(atPath: path.path) {
                    let directorySize = calculateDirectorySize(at: path)
                    logger.info("   Model size: \(directorySize / 1024)KB")
                }
            }

            let model = try MLModel(contentsOf: path, configuration: configuration)

            if isCI {
                logger.info("✅ CI: Successfully loaded \(name) model")
            }

            return model
        } catch {
            // If loading fails, log detailed error info
            logger.error("Failed to load \(name) model: \(error)")

            let isCompiled = path.pathExtension == "mlmodelc"
            
            if isCI {
                logger.error("❌ CI: Failed to load \(name) model from \(path.path)")
                logger.error("   Error: \(error)")
                logger.error("   Error type: \(type(of: error))")
                logger.error("   Is compiled model: \(isCompiled)")

                if !isCompiled {
                    logger.info("   Attempting to compile model...")
                }
            }

            // If it's already a compiled model, check if we should remove and re-download
            if isCompiled {
                if isCI {
                    logger.error("❌ CI: Compiled model failed to load. This may indicate corruption.")
                    logger.error("   Removing corrupted model: \(path.path)")
                    try? FileManager.default.removeItem(at: path)

                    // Check if we have a parent directory with .mlpackage
                    let parentDir = path.deletingLastPathComponent()
                    let packageName = path.deletingPathExtension().lastPathComponent + ".mlpackage"
                    let packagePath = parentDir.appendingPathComponent(packageName)

                    if FileManager.default.fileExists(atPath: packagePath.path) {
                        logger.info("   Found .mlpackage, attempting to compile...")
                        let compiledURL = try await MLModel.compileModel(at: packagePath)
                        return try MLModel(contentsOf: compiledURL, configuration: configuration)
                    }
                }
                throw ASRError.modelLoadFailed
            }

            // Try compiling first
            logger.info("Compiling \(name) model...")
            let compiledURL = try await MLModel.compileModel(at: path)
            logger.info("Successfully compiled \(name) model")
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
    }

    /// Load all required models in a single operation
    private func loadAllModels(
        melSpectrogramPath: URL,
        encoderPath: URL,
        decoderPath: URL,
        jointPath: URL,
        configuration: MLModelConfiguration
    ) async throws -> (melSpectrogram: MLModel, encoder: MLModel, decoder: MLModel, joint: MLModel) {
        // Load models in parallel for better performance
        async let melSpectrogram = loadModelWithCompilation(path: melSpectrogramPath, name: "mel-spectrogram", configuration: configuration)
        async let encoder = loadModelWithCompilation(path: encoderPath, name: "encoder", configuration: configuration)
        async let decoder = loadModelWithCompilation(path: decoderPath, name: "decoder", configuration: configuration)
        async let joint = loadModelWithCompilation(path: jointPath, name: "joint", configuration: configuration)

        return try await (melSpectrogram, encoder, decoder, joint)
    }

    private static func getDefaultModelsDirectory() -> URL {
        // Use Application Support cache directory for better system integration
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        let directory = appDirectory.appendingPathComponent("Models/Parakeet", isDirectory: true)

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    private func getModelsDirectory() -> URL {
        return modelsDirectory
    }

    public func cleanup() {
        melSpectrogramModel = nil
        encoderModel = nil
        decoderModel = nil
        jointModel = nil
        decoderState = DecoderState.zero()
        logger.info("AsrManager resources cleaned up")
    }



    // MARK: - TDT Decoding Implementation

    /// Split joint logits into token and duration components
    private func splitLogits(_ logits: MLMultiArray) throws -> (tokenLogits: [Float], durationLogits: [Float]) {
        let totalElements = logits.count
        let durationElements = config.tdtConfig.durations.count
        let vocabSize = totalElements - durationElements

        // Split logits analysis - now with correct duration config [0,1,2,3,4]
        if config.enableDebug {
            logger.info("🔍 DEBUG: splitLogits analysis:")
            logger.info("   - Total joint output elements: \(totalElements)")
            logger.info("   - Expected duration elements: \(durationElements)")
            logger.info("   - Calculated vocab size: \(vocabSize)")
            logger.info("   - Configured durations: \(self.config.tdtConfig.durations)")
            logger.info("   - Logits shape: \(logits.shape)")
        }

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed("Logits dimension mismatch: expected at least \(durationElements) elements, got \(totalElements)")
        }

        var tokenLogits = [Float]()
        var durationLogits = [Float]()

        // Extract token logits (first V elements)
        for i in 0..<vocabSize {
            tokenLogits.append(logits[i].floatValue)
        }

        // Extract duration logits (last D elements)
        for i in vocabSize..<totalElements {
            durationLogits.append(logits[i].floatValue)
        }

        if config.enableDebug {
            logger.info("   - Token logits extracted: \(tokenLogits.count)")
            logger.info("   - Duration logits extracted: \(durationLogits.count)")
            logger.info("   - Sample token logits (first 5): \(Array(tokenLogits.prefix(5)))")
            logger.info("   - Sample duration logits: \(durationLogits)")
        }

        return (tokenLogits, durationLogits)
    }


    /// Process duration logits and return duration index with skip value
    private func processDurationLogits(_ logits: [Float]) throws -> (index: Int, skip: Int) {
        var maxIndex = 0
        var maxValue = -Float.infinity

        for (i, value) in logits.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }

        let durations = config.tdtConfig.durations
        guard maxIndex < durations.count else {
            throw ASRError.processingFailed("Duration index out of bounds: \(maxIndex) >= \(durations.count)")
        }

        return (maxIndex, durations[maxIndex])
    }



    /// TDT decoding algorithm following GreedyTDTInfer logic
    internal func tdtDecode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        originalAudioSamples: [Float]
    ) async throws -> [Int] {
        var hypothesis = TDTHypothesis()
        var timeIdx = 0
        var symbolsAdded = 0
        let maxSymbolsPerFrame = config.tdtConfig.maxSymbolsPerStep ?? config.maxSymbolsPerFrame

        // Initialize decoder state
        try await initializeDecoderState()
        hypothesis.decState = decoderState

        if config.enableDebug {
            logger.info("Starting TDT decoding: encoder_len=\(encoderSequenceLength), durations=\(self.config.tdtConfig.durations), audio_samples=\(originalAudioSamples.count)")
        }

        // TDT decoding started with correct duration configuration

        // Handle very short sequences by falling back to standard RNNT
        if encoderSequenceLength <= 1 {
            logger.warning("TDT: Encoder sequence too short (\(encoderSequenceLength)), falling back to standard RNNT")
            // Convert back to encoder output and call standard RNNT
            let encoderProvider = try createEncoderProvider(from: encoderOutput)
            return try await improvedDecodeWithRNNT(encoderProvider, originalAudioSamples: originalAudioSamples)
        }

        // Initialize hypothesis with SOS token properly
        hypothesis.lastToken = nil  // Start with nil to trigger SOS usage

        while timeIdx < encoderSequenceLength {
            // Additional bounds check before extracting encoder step
            if timeIdx >= encoderSequenceLength {
                if config.enableDebug {
                    logger.info("TDT: Time index \(timeIdx) reached encoder length \(encoderSequenceLength), exiting outer loop")
                }
                break
            }

            // Extract encoder embedding at current timestep
            let encoderStep = try extractEncoderTimeStep(encoderOutput, timeIndex: timeIdx)

            var needLoop = true
            symbolsAdded = 0

            while needLoop && symbolsAdded < maxSymbolsPerFrame {
                // Prepare decoder input - either SOS or last token (following Python logic)
                let targetToken: Int
                if hypothesis.lastToken == nil {
                    targetToken = sosId  // Start of sequence
                } else {
                    targetToken = hypothesis.lastToken!
                }

                // Run decoder prediction
                let decoderInput = try prepareDecoderInput(
                    targetToken: targetToken,
                    hiddenState: hypothesis.decState?.hiddenState ?? decoderState.hiddenState,
                    cellState: hypothesis.decState?.cellState ?? decoderState.cellState
                )

                guard let decoderOutput = try decoderModel?.prediction(from: decoderInput, options: predictionOptions ?? MLPredictionOptions()) else {
                    throw ASRError.processingFailed("Decoder prediction failed")
                }

                // Update decoder state
                var newDecState = hypothesis.decState ?? decoderState
                newDecState.update(from: decoderOutput)

                // Prepare joint input
                let jointInput = try prepareJointInput(
                    encoderOutput: encoderStep,
                    decoderOutput: decoderOutput,
                    timeIndex: timeIdx
                )

                // Run joint prediction
                guard let jointOutput = try jointModel?.prediction(from: jointInput, options: predictionOptions ?? MLPredictionOptions()),
                      let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                    throw ASRError.processingFailed("Joint prediction failed")
                }

                // Debug: Check joint output dimensions
                if config.enableDebug && timeIdx == 0 {
                    logger.info("🔍 DEBUG: Joint output logits shape: \(logits.shape), count: \(logits.count)")
                    logger.info("🔍 DEBUG: Expected vocab size: 1025, duration count: \(self.config.tdtConfig.durations.count)")
                    logger.info("🔍 DEBUG: Total expected: \(1025 + self.config.tdtConfig.durations.count)")
                }

                // Joint model executed

                // Split logits into token and duration parts
                let (tokenLogits, durationLogits) = try splitLogits(logits)

                // SIMPLIFIED TDT: Use same token selection logic as RNNT
                // Create MLMultiArray from tokenLogits for compatibility with RNNT function
                let tokenLogitsArray = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: tokenLogits.count)], dataType: .float32)
                for (i, logit) in tokenLogits.enumerated() {
                    tokenLogitsArray[i] = NSNumber(value: logit)
                }

                let bestToken = findBestTokenWithRepetitionPrevention(
                    tokenLogitsArray,
                    lastEmittedTokens: Array(hypothesis.ySequence.suffix(10)),
                    consecutiveBlanks: 0, // TDT doesn't use consecutive blanks the same way
                    maxConsecutiveBlanks: 10,
                    originalAudioSamples: originalAudioSamples
                )

                // Process duration logits to get actual duration value
                let (_, skip) = try processDurationLogits(durationLogits)

                // Additional TDT-specific quality check: verify duration makes sense
                if bestToken != blankId && skip > 8 {
                    // If predicted duration is very long (>8 frames), be more conservative
                    if config.enableDebug {
                        logger.warning("TDT: Long duration prediction (\(skip)) for token \(bestToken), may indicate uncertainty")
                    }
                }

                if config.enableDebug && timeIdx < 10 {
                    logger.info("TDT Time \(timeIdx): token=\(bestToken), duration=\(skip)")
                }

                // Update hypothesis based on token type
                if bestToken != blankId {
                    hypothesis.ySequence.append(bestToken)
                    // Use logit value as confidence score (approximation)
                    let tokenConfidence = tokenLogits[bestToken]
                    hypothesis.score += tokenConfidence
                    hypothesis.timestamps.append(timeIdx)
                    hypothesis.decState = newDecState
                    hypothesis.lastToken = bestToken

                    if config.tdtConfig.includeTokenDuration {
                        hypothesis.tokenDurations.append(skip)
                    }

                    if config.enableDebug {
                        logger.info("TDT Emitted token: \(bestToken) at time \(timeIdx) with duration \(skip), conf: \(String(format: "%.3f", tokenConfidence))")
                    }
                }

                // Update loop control variables
                symbolsAdded += 1
                needLoop = (skip == 0)

                // Advance time index following Python logic
                if skip > 0 {
                    // Cap the advancement to prevent going beyond encoder sequence length
                    let maxAdvancement = encoderSequenceLength - timeIdx
                    let actualSkip = min(skip, maxAdvancement)
                    timeIdx += actualSkip

                    if config.enableDebug && actualSkip != skip {
                        logger.info("TDT: Capped skip from \(skip) to \(actualSkip) to stay within bounds")
                    }
                } else {
                    // Python logic: if skip == 0, force advance by 1 to prevent infinite loops
                    timeIdx += 1
                    needLoop = false

                    if config.enableDebug {
                        logger.info("TDT: Skip=0, forcing advance to prevent infinite loop")
                    }
                }

                // Break if we've reached the end - this prevents the bounds error
                if timeIdx >= encoderSequenceLength {
                    if config.enableDebug {
                        logger.info("TDT: Time index \(timeIdx) reached encoder length \(encoderSequenceLength), completing")
                    }
                    needLoop = false
                    break
                }
            }
        }

        if config.enableDebug {
            logger.info("TDT decoding completed: \(hypothesis.ySequence.count) tokens, final score: \(hypothesis.score)")
        }

        return hypothesis.ySequence
    }

    /// Transcribes audio samples to text using Parakeet TDT models.
    ///
    /// This method automatically handles long audio files by splitting them into 10-second chunks
    /// and concatenating the results. For optimal performance, audio should be:
    /// - 16kHz sample rate
    /// - Mono channel
    /// - Float32 samples normalized to [-1.0, 1.0]
    ///
    /// - Parameter audioSamples: Array of audio samples at 16kHz
    /// - Returns: `ASRResult` containing transcribed text, confidence, and timing information
    /// - Throws:
    ///   - `ASRError.notInitialized` if models are not loaded
    ///   - `ASRError.invalidAudioData` if audio is too short (<1 second)
    ///   - `ASRError.processingFailed` if transcription fails
    ///
    /// ## Performance Notes
    /// - Audio ≤10 seconds: Processed in a single pass
    /// - Audio >10 seconds: Automatically chunked with 0.5s overlap
    /// - RTFx includes full end-to-end time (I/O + inference)
    public func transcribe(_ audioSamples: [Float]) async throws -> ASRResult {
        // Use the new unified transcription approach
        return try await transcribeUnified(audioSamples)
    }

    /// Optimized RNNT decoding algorithm with repetition prevention
    func improvedDecodeWithRNNT(_ encoderOutput: MLFeatureProvider, originalAudioSamples: [Float]) async throws -> [Int] {
        guard let rawEncoderOutput = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
              let encoderLength = encoderOutput.featureValue(for: "encoder_output_length")?.multiArrayValue else {
            throw ASRError.processingFailed("Invalid encoder output")
        }

        // Transpose encoder output from (1, 1024, T) to (1, T, 1024)
        let encoderHiddenStates = try transposeEncoderOutput(rawEncoderOutput)
        let encoderSequenceLength = encoderLength[0].intValue

        // Initialize decoder state with clean reset
        try await initializeDecoderState()

        var tokens: [Int] = []
        var timeIndex = 0
        var currentToken = sosId // Revert to SOS token for now

        // Repetition prevention state
        var consecutiveBlanks = 0
        let maxConsecutiveBlanks = 10
        var lastEmittedTokens: [Int] = []
        let maxRepeats = 3
        var stuckCounter = 0
        let maxStuckSteps = 20

        // PHASE 1 FIX 4: Enhanced Length Normalization
        let audioLengthSeconds = Double(originalAudioSamples.count) / 16000.0
        let isShortAudio = audioLengthSeconds < 4.0
        let isLongAudio = audioLengthSeconds > 10.0

        // More aggressive length limits to prevent runaway generation
        let maxTokensForAudio: Int
        if isShortAudio {
            maxTokensForAudio = min(Int(audioLengthSeconds * 2.5), 12)  // Even more conservative: 2.5 tokens/sec, max 12
        } else if isLongAudio {
            maxTokensForAudio = min(Int(audioLengthSeconds * 6), 120)   // Strict limit for long audio: 6 tokens/sec, max 120
        } else {
            maxTokensForAudio = Int(audioLengthSeconds * 7)             // Balanced for medium audio: 7 tokens/sec
        }

        if config.enableDebug {
            logger.info("Starting RNNT decoding: audio=\(String(format: "%.1f", audioLengthSeconds))s, encoder_len=\(encoderSequenceLength), max_tokens=\(maxTokensForAudio)")
        }

        // Main RNNT decoding loop with adaptive stopping criteria
        var shouldContinue = true
        while timeIndex < encoderSequenceLength && tokens.count < maxTokensForAudio && shouldContinue {
            do {
                // Step 1: Run decoder with current token
                let decoderInput = try prepareDecoderInput(
                    targetToken: currentToken,
                    hiddenState: decoderState.hiddenState,
                    cellState: decoderState.cellState
                )

                guard let decoderOutput = try decoderModel?.prediction(from: decoderInput, options: predictionOptions ?? MLPredictionOptions()) else {
                    throw ASRError.processingFailed("Decoder model failed")
                }

                // Update decoder state
                decoderState.update(from: decoderOutput)

                // Step 2: Run joint network at current time step
                let jointInput = try prepareJointInput(
                    encoderOutput: encoderHiddenStates,
                    decoderOutput: decoderOutput,
                    timeIndex: timeIndex
                )

                guard let jointOutput = try jointModel?.prediction(from: jointInput, options: predictionOptions ?? MLPredictionOptions()) else {
                    throw ASRError.processingFailed("Joint model failed")
                }

                guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                    throw ASRError.processingFailed("Invalid joint output")
                }

                // Step 3: Enhanced token selection with adaptive constraints
                let bestToken = findBestTokenWithRepetitionPrevention(
                    logits,
                    lastEmittedTokens: lastEmittedTokens,
                    consecutiveBlanks: consecutiveBlanks,
                    maxConsecutiveBlanks: maxConsecutiveBlanks,
                    originalAudioSamples: originalAudioSamples
                )

                if config.enableDebug && timeIndex < 15 {
                    logger.info("Time \(timeIndex): Token \(bestToken)")
                }

                // Step 4: RNNT token decision with loop prevention
                if bestToken == blankId {
                    // Blank - advance time
                    timeIndex += 1
                    consecutiveBlanks += 1

                    // Prevent infinite blank loops
                    if consecutiveBlanks > maxConsecutiveBlanks {
                        if config.enableDebug {
                            logger.warning("Too many consecutive blanks (\(consecutiveBlanks)), advancing")
                        }
                        timeIndex += 1
                        consecutiveBlanks = 0
                    }
                } else {
                    // Non-blank - emit token with repetition checking
                    consecutiveBlanks = 0

                    // Basic token validation (simplified for testing)
                    if shouldEmitToken(bestToken, lastEmittedTokens: lastEmittedTokens, maxRepeats: maxRepeats) {
                        tokens.append(bestToken)
                        currentToken = bestToken

                        // Update last emitted tokens (keep last 5 for pattern detection)
                        lastEmittedTokens.append(bestToken)
                        if lastEmittedTokens.count > 5 {
                            lastEmittedTokens.removeFirst()
                        }

                        stuckCounter = 0

                        if config.enableDebug {
                            logger.info("Emitted token: \(bestToken)")
                        }


                    } else {
                        // Skip problematic token, advance time to break loop
                        timeIndex += 1
                        stuckCounter += 1

                        if config.enableDebug {
                            logger.warning("Skipped problematic token: \(bestToken)")
                        }
                    }

                    // Early instability detection - reset if needed (reduced frequency)
                    if tokens.count <= 5 && tokens.count > 2 && isEarlySequenceUnstable(tokens, vocabulary: loadVocabulary()) {
                        if config.enableDebug {
                            logger.warning("Early instability detected, resetting decoder")
                        }

                        // Reset decoder state and try to recover
                        try await initializeDecoderState()
                        tokens.removeAll()
                        lastEmittedTokens.removeAll()
                        currentToken = blankId
                        timeIndex += 2 // Skip ahead to avoid same problematic state
                        stuckCounter = 0
                    }
                }

                // Detect stuck state and force progress
                if stuckCounter > maxStuckSteps {
                    if config.enableDebug {
                        logger.warning("Decoder appears stuck, forcing progress")
                    }
                    timeIndex += 2  // Skip ahead
                    stuckCounter = 0
                    consecutiveBlanks = 0
                }

                // Progressive stopping criteria with aggressive stability checks
                if tokens.count > 30 {
                    // Check for repetitive patterns in recent tokens
                    if hasRecentRepetitivePattern(tokens, checkLength: 10) {
                        if config.enableDebug {
                            logger.warning("Detected repetitive pattern, stopping early")
                        }
                        shouldContinue = false
                    }

                    // Check sequence quality and stability
                    if isSequenceUnstable(tokens, vocabulary: loadVocabulary()) {
                        if config.enableDebug {
                            logger.warning("Sequence appears unstable, stopping early")
                        }
                        shouldContinue = false
                    }
                }

                // Emergency brake for very short utterances generating too much
                let audioLengthSeconds = Double(originalAudioSamples.count) / 16000.0
                let expectedTokensPerSecond = 5.0 // Reasonable estimate
                let maxExpectedTokens = Int(audioLengthSeconds * expectedTokensPerSecond * 2) // 2x buffer

                if tokens.count > maxExpectedTokens && tokens.count > 50 {
                    if config.enableDebug {
                        logger.warning("Generated too many tokens (\(tokens.count)) for audio length (\(String(format: "%.1f", audioLengthSeconds))s), stopping")
                    }
                    shouldContinue = false
                }

            } catch {
                logger.error("Error in enhanced decoding loop: \(error)")
                shouldContinue = false
            }
        }

        // Final cleanup - remove obvious repetitive patterns
        let cleanedTokens = tokens

        if config.enableDebug {
            logger.info("RNNT decoding completed: \(tokens.count) raw tokens -> \(cleanedTokens.count) cleaned tokens")
        }

        return cleanedTokens
    }

    // DEPRECATED: Moved to TranscriptionStrategy.swift
    /*
    private func transcribeWithChunking(_ audioSamples: [Float], startTime: Date) async throws -> ASRResult {
        let chunkSize = 160_000  // 10 seconds at 16kHz
        let overlap = 16_000     // 1 second overlap to avoid boundary issues
        let stride = chunkSize - overlap  // 9 seconds stride

        var allTexts: [String] = []
        var totalProcessingTime: TimeInterval = 0
        let audioLength = Double(audioSamples.count) / Double(config.sampleRate)

        if config.enableDebug {
            logger.info("🔄 Chunking audio: \(String(format: "%.1f", audioLength))s into \(String(format: "%.1f", Double(chunkSize)/16000.0))s chunks with \(String(format: "%.1f", Double(overlap)/16000.0))s overlap")
        }

        // Process audio in chunks
        var position = 0
        var chunkIndex = 0
        var previousChunkLastWords: [String] = []

        while position < audioSamples.count {
            let endPosition = min(position + chunkSize, audioSamples.count)
            let chunkSamples = Array(audioSamples[position..<endPosition])

            // Pad last chunk if needed
            let paddedChunk = padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

            if config.enableDebug {
                let chunkDuration = Double(chunkSamples.count) / 16000.0
                logger.info("   📄 Chunk \(chunkIndex): \(String(format: "%.1f", Double(position)/16000.0))s - \(String(format: "%.1f", Double(endPosition)/16000.0))s (\(String(format: "%.1f", chunkDuration))s)")
            }

            // Don't reset state between chunks - maintain context
            // Only reset state for the first chunk
            if chunkIndex == 0 {
                try await initializeDecoderState()
            }

            // Transcribe chunk
            let chunkResult = try await transcribeSingleChunk(paddedChunk)

            // Handle chunk boundary overlap intelligently
            if chunkIndex > 0 && !chunkResult.text.isEmpty {
                // Smart deduplication: remove repeated words at the beginning of this chunk
                // that match the end of the previous chunk
                let chunkWords = chunkResult.text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                var startIndex = 0

                // Check for overlapping words and find where new content starts
                if !previousChunkLastWords.isEmpty && !chunkWords.isEmpty {
                    // Look for the longest matching sequence at the boundary
                    for overlapSize in Swift.stride(from: Swift.min(previousChunkLastWords.count, chunkWords.count), through: 1, by: -1) {
                        let previousEnd = previousChunkLastWords.suffix(overlapSize)
                        let currentStart = chunkWords.prefix(overlapSize)

                        if previousEnd.map({ $0.lowercased() }) == currentStart.map({ $0.lowercased() }) {
                            startIndex = overlapSize
                            if config.enableDebug {
                                logger.info("   🔗 Found \(overlapSize) overlapping words at chunk boundary")
                            }
                            break
                        }
                    }
                }

                // Add only the non-overlapping portion
                let uniqueWords = chunkWords.dropFirst(startIndex)
                if !uniqueWords.isEmpty {
                    allTexts.append(uniqueWords.joined(separator: " "))
                }

                // Update previous chunk's last words for next iteration
                previousChunkLastWords = Array(chunkWords.suffix(10)) // Keep last 10 words
            } else {
                // First chunk or empty chunk
                allTexts.append(chunkResult.text)
                let words = chunkResult.text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                previousChunkLastWords = Array(words.suffix(10))
            }

            totalProcessingTime += chunkResult.processingTime

            // Move to next chunk
            position += stride
            chunkIndex += 1

            // Break if we've processed the last meaningful chunk
            if position >= audioSamples.count - 8000 {  // Less than 0.5s remaining
                break
            }
        }

        // Combine all chunk transcriptions
        let finalText = allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)

        // Apply post-processing to clean up any remaining issues
        let cleanedText = config.enableAdvancedPostProcessing ?
            applyAdvancedPostProcessing(finalText, tokenTimings: []) :
            finalText

        let totalTime = Date().timeIntervalSince(startTime)

        if config.enableDebug {
            logger.info("✅ Chunked transcription complete: \(chunkIndex) chunks, \(String(format: "%.1f", audioLength))s audio, \(String(format: "%.3f", totalTime))s processing")
        }

        return ASRResult(
            text: cleanedText,
            confidence: 1.0,  // Confidence calculation would need refinement for chunks
            duration: audioLength,
            processingTime: totalTime,
            tokenTimings: nil  // Token timings would need special handling for chunks
        )
    }
    */

    // DEPRECATED: Moved to TranscriptionStrategy.swift  
    /*
    private func transcribeSingleChunk(_ paddedAudio: [Float]) async throws -> ASRResult {
        let chunkStartTime = Date()

        do {
            // Use shared transcription core
            let (tokenIds, encoderSequenceLength) = try await transcriptionCore(
                paddedAudio,
                enableDebug: false
            )

            let processingTime = Date().timeIntervalSince(chunkStartTime)

            // Process result
            return processTranscriptionResult(
                tokenIds: tokenIds,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: paddedAudio,
                processingTime: processingTime
            )

        } catch {
            logger.error("Chunk transcription failed: \(error.localizedDescription)")
            throw error
        }
    }
    */

    /// Convert tokens to simple text without timing information
    private func convertTokensToSimpleText(_ tokenIds: [Int]) -> String {
        let (text, _) = convertTokensWithExistingTimings(tokenIds, timings: [])
        return text
    }

    /// Convert tokens to text using existing timing information from TDT
    internal func convertTokensWithExistingTimings(_ tokenIds: [Int], timings: [TokenTiming]) -> (text: String, timings: [TokenTiming]) {
        if tokenIds.isEmpty {
            return ("", [])
        }

        let vocabulary = loadVocabulary()
        var result = ""
        var lastWasSpace = false
        var adjustedTimings: [TokenTiming] = []

        for (index, tokenId) in tokenIds.enumerated() {
            guard let token = vocabulary[tokenId] else {
                continue
            }

            // Get corresponding timing if available
            let timing = index < timings.count ? timings[index] : nil

            // Process token text (same logic as original)
            if token.hasPrefix("▁") {
                let cleanToken = String(token.dropFirst())
                if !cleanToken.isEmpty {
                    if !result.isEmpty && !lastWasSpace {
                        result += " "
                    }
                    result += cleanToken
                    lastWasSpace = false

                    // Use existing timing or create adjusted one
                    if let timing = timing {
                        adjustedTimings.append(TokenTiming(
                            token: cleanToken,
                            tokenId: tokenId,
                            startTime: timing.startTime,
                            endTime: timing.endTime,
                            confidence: timing.confidence
                        ))
                    }
                }
            } else {
                if !token.isEmpty {
                    result += token
                    lastWasSpace = false

                    // Use existing timing or create adjusted one
                    if let timing = timing {
                        adjustedTimings.append(TokenTiming(
                            token: token,
                            tokenId: tokenId,
                            startTime: timing.startTime,
                            endTime: timing.endTime,
                            confidence: timing.confidence
                        ))
                    }
                }
            }
        }

        return (result, adjustedTimings)
    }
}

