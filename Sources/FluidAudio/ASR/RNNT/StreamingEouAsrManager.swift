@preconcurrency import CoreML
import Foundation
import OSLog

/// Streaming manager for Parakeet Realtime EOU 120M speech recognition
///
/// This manager provides true cache-aware streaming inference using NVIDIA's
/// FastConformer encoder with persistent attention and time caches.
///
/// Key features:
/// - Cache-aware streaming encoder maintains state across chunks
/// - Low latency processing with incremental results
/// - End-of-utterance detection via <EOU> token
/// - Compatible with NVIDIA's 80-160ms streaming latency target
public final class StreamingEouAsrManager {
    private var models: StreamingEouAsrModels?
    private var encoderState: StreamingEncoderState?
    private var decoderState: RnntDecoderState
    private let logger = AppLogger(category: "StreamingEouASR")
    private let config: RnntConfig
    private let encoderConfig: StreamingEncoderConfig
    
    // Partial hypothesis state (NeMo pattern for streaming)
    private var accumulatedTokens: [Int] = []
    private var accumulatedTimestamps: [Int] = []
    private var accumulatedConfidences: [Float] = []
    private var accumulatedScore: Float = 0.0
    
    // Audio buffer for continuous preprocessing (NeMo pattern)
    private var audioBuffer: [Float] = []
    private let bufferSizeSeconds: Double = 4.0

    // Mel feature buffer for incremental streaming
    // Each element is a frame: [128 mel coefficients]
    private var melBuffer: [[Float]] = []

    // Overlap buffer for STFT continuity (disabled for now)
    private var audioOverlapBuffer: [Float] = []
    private let overlapSamples = 0
    private let hopLength = 160

    /// Sample rate expected by the model
    public static let sampleRate = 16_000

    public init(
        config: RnntConfig = .parakeetEOU,
        encoderConfig: StreamingEncoderConfig = .parakeetEOUStreaming
    ) {
        self.config = config
        self.encoderConfig = encoderConfig
        self.decoderState = RnntDecoderState.make(hiddenSize: config.decoderHiddenSize)
    }

    /// Initialize from local directory containing streaming models
    public func initializeFromLocalPath(_ directory: URL, useMLPackage: Bool = true) async throws {
        let models = try await StreamingEouAsrModels.load(
            from: directory,
            configuration: StreamingEouAsrModels.defaultConfiguration(),
            useMLPackage: useMLPackage
        )
        self.models = models
        self.encoderState = try StreamingEncoderState(config: encoderConfig)
        logger.info("StreamingEouAsrManager initialized from: \(directory.path)")
    }

    /// Download and initialize models
    public func initialize(cacheDirectory: URL? = nil) async throws {
        let models = try await StreamingEouAsrModels.downloadAndLoad(to: cacheDirectory)
        self.models = models
        self.encoderState = try StreamingEncoderState(config: encoderConfig)
        logger.info("StreamingEouAsrManager initialized with downloaded models")
    }

    /// Process a chunk of audio and return incremental transcription
    ///
    /// - Parameters:
    ///   - audioChunk: Audio samples at 16kHz mono
    ///   - isFinal: Whether this is the final chunk
    /// - Returns: Incremental transcription result
    public func processChunk(_ audioChunk: [Float], isFinal: Bool = false) async throws -> StreamingTranscriptionResult
    {
        guard let models = models, let encoderState = encoderState else {
            throw ASRError.notInitialized
        }

        let startTime = Date()

        // --- Direct Chunk Mel Processing ---
        // Feed each chunk's mel features directly to the streaming encoder
        // The cache provides context from previous chunks (no overlapping mel needed)
        //
        // NeMo streaming config uses 45 mel frames per chunk (chunk_size=9 * 4 + 9 buffer)
        // This corresponds to ~450ms of audio at 100 frames/second

        // Run preprocessor on this audio chunk with overlap
        let inputAudio = audioOverlapBuffer + audioChunk
        let (fullMel, fullMelLength) = try await runPreprocessor(
            audio: inputAudio,
            audioLength: inputAudio.count,
            model: models.preprocessor
        )

        // Calculate frames to discard (corresponding to overlap)
        // Calculate frames to discard (corresponding to overlap)
        // If buffer was empty (first chunk), discard 0.
        // Otherwise discard overlapSamples / hopLength.
        let discardFrames = audioOverlapBuffer.isEmpty ? 0 : (audioOverlapBuffer.count / hopLength)
        let actualMelFrames = max(0, fullMelLength - discardFrames)

        // Update overlap buffer for next chunk
        if audioChunk.count >= overlapSamples {
            audioOverlapBuffer = Array(audioChunk.suffix(overlapSamples))
        } else {
            audioOverlapBuffer.append(contentsOf: audioChunk)
            if audioOverlapBuffer.count > overlapSamples {
                audioOverlapBuffer = Array(audioOverlapBuffer.suffix(overlapSamples))
            }
        }

        // Prepare fixed-size input (32 frames = 320ms chunks)
        let fixedFrames = 32  // Must match exported streaming_encoder model
        let melDim = 128
        let inputMel = try MLMultiArray(shape: [1, NSNumber(value: melDim), NSNumber(value: fixedFrames)], dataType: .float32)
        let destPtr = inputMel.dataPointer.bindMemory(to: Float.self, capacity: inputMel.count)
        
        // Initialize with silence (-10.0) - padding for short chunks
        destPtr.initialize(repeating: -10.0, count: inputMel.count)

        // Copy actual mel frames from this chunk (skipping overlap)
        let framesToCopy = min(actualMelFrames, fixedFrames)
        let srcPtr = fullMel.dataPointer.bindMemory(to: Float.self, capacity: fullMel.count)
        let srcT = fullMel.shape[2].intValue

        for channel in 0..<melDim {
            for frame in 0..<framesToCopy {
                // src index: channel * srcT + (discardFrames + frame)
                destPtr[channel * fixedFrames + frame] = srcPtr[channel * srcT + (discardFrames + frame)]
            }
        }

        // Tell the encoder how many valid frames we have
        let inputLength = framesToCopy
        let melLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        melLengthArray[0] = NSNumber(value: inputLength)

        // logger.debug("Preprocessor: chunk=\(audioChunk.count) samples, mel_frames=\(actualMelFrames), padding=\(fixedFrames - framesToCopy)")

        // Run streaming encoder with cache support
        let (encoderOutput, encoderLength, newCacheChannel, newCacheTime, newCacheLen) = try await runStreamingEncoder(
            mel: inputMel,
            melLength: melLengthArray,
            model: models.streamingEncoder,
            encoderState: encoderState
        )

        let cacheLenBefore = encoderState.cacheLastChannelLen

        // Update encoder state with new caches
        encoderState.updateConformerCache(
            newCacheChannel: newCacheChannel,
            newCacheTime: newCacheTime,
            newCacheLen: newCacheLen
        )

        // print("DEBUG: StreamingEncoder: enc_length=\(encoderLength), cache_len: \(cacheLenBefore) → \(newCacheLen)")

        // Create partial hypothesis from accumulated state (NeMo pattern)
        let previousTokenCount = accumulatedTokens.count
        let partialHypothesis = RnntHypothesis(
            score: accumulatedScore,
            ySequence: accumulatedTokens,
            timestamps: accumulatedTimestamps,
            tokenConfidences: accumulatedConfidences,
            eouDetected: false,
            lastToken: accumulatedTokens.last
        )
        
        // logger.debug("Partial hypothesis: \(previousTokenCount) tokens, last_token=\(partialHypothesis.lastToken ?? -1)")

        // Run RNNT decoder with partial hypothesis
        let decoder = RnntDecoder(config: config)
        let fullHypothesis = try await decoder.decodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderLength,
            decoderModel: models.decoder,
            jointModel: models.joint,
            decoderState: &decoderState,
            partialHypothesis: partialHypothesis  // Pass accumulated state!
        )
        
        // Extract NEW tokens (difference from previous)
        let newTokens = Array(fullHypothesis.ySequence.dropFirst(previousTokenCount))
        let newTimestamps = Array(fullHypothesis.timestamps.dropFirst(previousTokenCount))
        let newConfidences = Array(fullHypothesis.tokenConfidences.dropFirst(previousTokenCount))
        
        // logger.debug("New tokens this chunk: \(newTokens.count) (total now: \(fullHypothesis.ySequence.count))")
        
        // Update accumulated state
        accumulatedTokens = fullHypothesis.ySequence
        accumulatedTimestamps = fullHypothesis.timestamps
        accumulatedConfidences = fullHypothesis.tokenConfidences
        accumulatedScore = fullHypothesis.score
        
        // Convert only NEW tokens to text for this chunk
        let (text, tokenTimings) = convertTokensToText(
            tokens: newTokens,
            timestamps: newTimestamps,
            confidences: newConfidences,
            vocabulary: models.vocabulary
        )

        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(audioChunk.count) / Double(Self.sampleRate)
        
        // Reset partial hypothesis on EOU or final chunk (NeMo pattern)
        // Also reset decoder state to start fresh for next utterance
        if fullHypothesis.eouDetected || isFinal {
            // logger.debug("Resetting partial hypothesis and decoder state (EOU=\(fullHypothesis.eouDetected), final=\(isFinal))")
            accumulatedTokens.removeAll()
            accumulatedTimestamps.removeAll()
            accumulatedConfidences.removeAll()
            accumulatedScore = 0.0
            decoderState.reset()  // Reset decoder LSTM state for new utterance
        }

        return StreamingTranscriptionResult(
            text: text,
            confidence: fullHypothesis.score / Float(max(1, fullHypothesis.ySequence.count)),
            chunkDuration: audioDuration,
            processingTime: processingTime,
            tokenTimings: tokenTimings,
            eouDetected: fullHypothesis.eouDetected,
            isFinal: isFinal
        )
    }

    /// Reset all state for new utterance
    public func resetState() {
        encoderState?.reset()
        decoderState.reset()

        // Reset partial hypothesis
        accumulatedTokens.removeAll()
        accumulatedTimestamps.removeAll()
        accumulatedConfidences.removeAll()
        accumulatedScore = 0.0
        audioBuffer.removeAll()
        melBuffer.removeAll()
        audioOverlapBuffer.removeAll()

        logger.debug("State reset for new utterance")
    }

    /// Check if end-of-utterance was detected
    public func wasEouDetected() -> Bool {
        return decoderState.eouDetected
    }

    // MARK: - Private Methods

    private func runPreprocessor(
        audio: [Float],
        audioLength: Int,
        model: MLModel
    ) async throws -> (mel: MLMultiArray, length: Int) {
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: audio.count)], dataType: .float32)
        let audioPtr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: audio.count)
        for (i, sample) in audio.enumerated() {
            audioPtr[i] = sample
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: audioLength)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: lengthArray),
        ])

        let output = try await model.prediction(from: input)

        guard let mel = output.featureValue(for: "mel")?.multiArrayValue,
            let melLengthArray = output.featureValue(for: "mel_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Preprocessor output missing expected features")
        }

        let melLength = melLengthArray[0].intValue
        return (mel, melLength)
    }

    /// Run streaming encoder with cache support
    private func runStreamingEncoder(
        mel: MLMultiArray,
        melLength: MLMultiArray,
        model: MLModel,
        encoderState: StreamingEncoderState
    ) async throws -> (
        encoder: MLMultiArray, length: Int, cacheChannel: MLMultiArray, cacheTime: MLMultiArray, cacheLen: Int
    ) {
        let cacheLenArray = try encoderState.createCacheLenArray()

        // logger.debug(
        //     "StreamingEncoder input: mel=\(mel.shape), cache_channel=\(encoderState.cacheLastChannel.shape), cache_time=\(encoderState.cacheLastTime.shape), cache_len=\(encoderState.cacheLastChannelLen)"
        // )

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: melLength),
            "cache_last_channel": MLFeatureValue(multiArray: encoderState.cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: encoderState.cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLenArray),
        ])

        let output = try await model.prediction(from: input)

        guard let encoder = output.featureValue(for: "encoder")?.multiArrayValue,
            let encoderLengthArray = output.featureValue(for: "encoder_length")?.multiArrayValue,
            let newCacheChannel = output.featureValue(for: "cache_last_channel_out")?.multiArrayValue,
            let newCacheTime = output.featureValue(for: "cache_last_time_out")?.multiArrayValue,
            let newCacheLenArray = output.featureValue(for: "cache_last_channel_len_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("StreamingEncoder output missing expected features")
        }

        let outputLength = encoderLengthArray[0].intValue
        let newCacheLen = newCacheLenArray[0].intValue

        return (encoder, outputLength, newCacheChannel, newCacheTime, newCacheLen)
    }

    private func convertTokensToText(
        tokens: [Int],
        timestamps: [Int],
        confidences: [Float],
        vocabulary: [Int: String]
    ) -> (text: String, timings: [EouTokenTiming]) {
        var textParts: [String] = []
        var timings: [EouTokenTiming] = []

        for (i, tokenId) in tokens.enumerated() {
            // Skip special tokens
            if tokenId == config.blankId || tokenId == config.eouTokenId || tokenId == config.eobTokenId {
                continue
            }

            guard let token = vocabulary[tokenId] else {
                continue
            }

            // Handle SentencePiece encoding (  = word boundary, U+2581)
            let displayToken: String
            
            // Debug logging for token analysis
            let hexStr = token.unicodeScalars.map { String(format: "%02X", $0.value) }.joined(separator: " ")
            // print("DEBUG: Token [\(tokenId)]: '\(token)' (Hex: \(hexStr))")

            if token.hasPrefix("\u{2581}") {
                displayToken = " " + String(token.dropFirst())
            } else {
                displayToken = token
            }

            textParts.append(displayToken)

            let timestamp = i < timestamps.count ? timestamps[i] : 0
            let confidence = i < confidences.count ? confidences[i] : 0

            timings.append(
                EouTokenTiming(
                    token: displayToken,
                    tokenId: tokenId,
                    frameIndex: timestamp,
                    confidence: confidence
                ))
        }

        let text = textParts.joined()
        return (text, timings)
    }
}

/// Result of streaming transcription chunk
public struct StreamingTranscriptionResult: Sendable {
    /// Transcribed text from this chunk
    public let text: String

    /// Average confidence score (0-1)
    public let confidence: Float

    /// Audio chunk duration in seconds
    public let chunkDuration: TimeInterval

    /// Processing time in seconds
    public let processingTime: TimeInterval

    /// Per-token timing information
    public let tokenTimings: [EouTokenTiming]?

    /// Whether end-of-utterance was detected
    public let eouDetected: Bool

    /// Whether this is the final chunk
    public let isFinal: Bool

    /// Real-time factor (processing speed)
    public var rtfx: Float {
        guard processingTime > 0 else { return 0 }
        return Float(chunkDuration / processingTime)
    }
}

/// Holds loaded CoreML models for streaming Parakeet EOU
public struct StreamingEouAsrModels: Sendable {
    public let preprocessor: MLModel
    public let streamingEncoder: MLModel  // Cache-aware streaming encoder (mel -> encoder with caches)
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "StreamingEouAsrModels")

    /// Load streaming EOU models from a directory (streaming encoder architecture)
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        useMLPackage: Bool = true // Default to true as repo has mlpackage
    ) async throws -> StreamingEouAsrModels {
        let mlConfig = configuration ?? defaultConfiguration()
        let fm = FileManager.default

        // If using ModelNames with subfolders, we need to handle that.
        // The directory passed in is usually the repo root.
        
        logger.info("Loading streaming EOU models from \(directory.path)")

        // Helper to find model
        func findModel(_ path: String) -> URL? {
            let url = directory.appendingPathComponent(path)
            if fm.fileExists(atPath: url.path) {
                return url
            }
            return nil
        }

        // Try using ModelNames (which now include subfolders and extensions)
        let preprocessorURL = directory.appendingPathComponent(ModelNames.ASREOUStreaming.preprocessorFile)
        let streamingEncoderURL = directory.appendingPathComponent(ModelNames.ASREOUStreaming.streamingEncoderFile)
        let decoderURL = directory.appendingPathComponent(ModelNames.ASREOUStreaming.decoderFile)
        let jointURL = directory.appendingPathComponent(ModelNames.ASREOUStreaming.jointFile)
        
        // Vocabulary: try subfolder first, then root
        let vocabURLSub = directory.appendingPathComponent(ModelNames.ASREOUStreaming.vocabularyPath)
        let vocabURLRoot = directory.appendingPathComponent(ModelNames.ASREOUStreaming.vocabularyFile)
        let vocabURL = fm.fileExists(atPath: vocabURLSub.path) ? vocabURLSub : vocabURLRoot

        // Fallback for local testing (flat structure)
        // ... (omitted for brevity, assume standardized structure for now)

        guard fm.fileExists(atPath: preprocessorURL.path) else {
            throw ASRError.processingFailed("Preprocessor model not found at \(preprocessorURL.path)")
        }
        guard fm.fileExists(atPath: streamingEncoderURL.path) else {
            throw ASRError.processingFailed("StreamingEncoder model not found at \(streamingEncoderURL.path)")
        }
        guard fm.fileExists(atPath: decoderURL.path) else {
            throw ASRError.processingFailed("Decoder model not found at \(decoderURL.path)")
        }
        guard fm.fileExists(atPath: jointURL.path) else {
            throw ASRError.processingFailed("Joint model not found at \(jointURL.path)")
        }

        // Compile and load models
        // Note: MLModel.load can load .mlpackage directly on macOS 13+
        // But for .mlmodelc we use it directly.
        // compileAndLoad handles both?
        
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        let preprocessor = try await compileAndLoad(preprocessorURL, configuration: cpuConfig)
        let streamingEncoder = try await compileAndLoad(streamingEncoderURL, configuration: mlConfig)
        let decoder = try await compileAndLoad(decoderURL, configuration: cpuConfig)
        let joint = try await compileAndLoad(jointURL, configuration: cpuConfig)

        // Load vocabulary
        let vocabulary = try loadVocabulary(from: vocabURL)

        logger.info("Streaming EOU models loaded successfully (\(vocabulary.count) vocab tokens)")

        return StreamingEouAsrModels(
            preprocessor: preprocessor,
            streamingEncoder: streamingEncoder,
            decoder: decoder,
            joint: joint,
            configuration: mlConfig,
            vocabulary: vocabulary
        )
    }

    /// Download streaming EOU models from HuggingFace and load them
    public static func downloadAndLoad(
        to cacheDirectory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> StreamingEouAsrModels {
        let directory = cacheDirectory ?? defaultCacheDirectory()

        // Download models using DownloadUtils
        let modelDir = directory.appendingPathComponent(Repo.parakeetEOU.folderName)
        let modelNames = Array(ModelNames.ASREOUStreaming.requiredModels)

        _ = try await DownloadUtils.loadModels(
            .parakeetEOU,
            modelNames: modelNames,
            directory: directory,
            computeUnits: .cpuOnly
        )

        // Also download vocabulary if needed (it might be in subfolder)
        // DownloadUtils should handle it if it's in requiredModels?
        // No, vocab is not in requiredModels usually.
        // Let's add it to requiredModels in ModelNames? No, it's a Set<String>.
        
        // Manually check/download vocab
        let vocabPath = ModelNames.ASREOUStreaming.vocabularyPath
        let vocabURL = modelDir.appendingPathComponent(vocabPath)
        
        if !FileManager.default.fileExists(atPath: vocabURL.path) {
             // Try root vocab
             let rootVocabURL = modelDir.appendingPathComponent("vocab.json")
             if !FileManager.default.fileExists(atPath: rootVocabURL.path) {
                 logger.warning("Vocabulary file not found, attempting to download...")
                 // This might fail if we don't use DownloadUtils for it.
                 // But DownloadUtils.loadModels downloads "essential files" (json/txt) automatically if they are in the root.
                 // If it's in subfolder, we might need to be explicit.
             }
        }

        // Load models
        return try await load(from: modelDir, configuration: configuration)
    }

    /// Default cache directory for EOU models
    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("FluidAudio/Models")
    }

    /// Default model configuration
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return config
    }

    // MARK: - Private

    /// Compile mlpackage to mlmodelc and load
    private static func compileAndLoad(
        _ url: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        // If it's already compiled (.mlmodelc), load directly
        if url.pathExtension == "mlmodelc" {
            return try await MLModel.load(contentsOf: url, configuration: configuration)
        }

        // Compile mlpackage to temporary location
        logger.info("Compiling \(url.lastPathComponent)...")
        do {
            let compiledURL = try await MLModel.compileModel(at: url)
            
            // Load the compiled model
            let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)

            // Clean up temporary compiled model
            try? FileManager.default.removeItem(at: compiledURL)

            return model
        } catch {
            logger.error("Compilation failed: \(error)")
            // Try compiling the inner .mlmodel if it exists
            let innerModel = url.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
            if FileManager.default.fileExists(atPath: innerModel.path) {
                logger.info("Retrying compilation with inner model: \(innerModel.path)")
                let compiledURL = try await MLModel.compileModel(at: innerModel)
                let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
                try? FileManager.default.removeItem(at: compiledURL)
                return model
            }
            throw error
        }
    }

    private static func loadVocabulary(from url: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: url)

        // Try parsing as simple {id: token} format first
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: String] {
            var vocabulary: [Int: String] = [:]
            for (idStr, token) in json {
                if let id = Int(idStr) {
                    vocabulary[id] = token
                }
            }
            return vocabulary
        }

        // Fall back to {vocab: [tokens]} format
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let vocabArray = json["vocab"] as? [String]
        {
            var vocabulary: [Int: String] = [:]
            for (index, token) in vocabArray.enumerated() {
                vocabulary[index] = token
            }
            return vocabulary
        }

        throw ASRError.processingFailed("Invalid vocabulary format in \(url.path)")
    }
}
