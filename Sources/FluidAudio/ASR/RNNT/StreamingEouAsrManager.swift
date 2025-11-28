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

        // --- Audio Buffering (NeMo Pattern) ---
        // Append new chunk to buffer
        audioBuffer.append(contentsOf: audioChunk)
        
        // Maintain buffer size (4s)
        let maxSamples = Int(Double(StreamingEouAsrManager.sampleRate) * bufferSizeSeconds)
        if audioBuffer.count > maxSamples {
            audioBuffer.removeFirst(audioBuffer.count - maxSamples)
        }
        
        // Run preprocessor on the ENTIRE buffer to get continuous features
        let (melFeatures, melLength) = try await runPreprocessor(
            audio: audioBuffer,
            audioLength: audioBuffer.count,
            model: models.preprocessor
        )
        
        // Extract features corresponding to the NEW chunk (last 129 frames for 1280ms)
        // NeMo logic: feature_chunk_len = int(chunk_size / stride)
        // For 1280ms chunk, stride 10ms -> 128 frames.
        // CoreML model expects 129 frames.
        let fixedFrames = 129
        let totalFrames = melFeatures.shape[2].intValue
        let inputMel = try MLMultiArray(shape: [1, 128, NSNumber(value: fixedFrames)], dataType: .float32)
        
        // Initialize with silence (-10.0)
        let count = inputMel.count
        let destPtr = inputMel.dataPointer.bindMemory(to: Float.self, capacity: count)
        destPtr.initialize(repeating: -10.0, count: count)
        
        if totalFrames >= fixedFrames {
            // Slicing: Copy last fixedFrames from each of the 128 channels
            // Assuming default C-contiguous layout [1, 128, T] -> T is inner dimension
            let srcPtr = melFeatures.dataPointer.bindMemory(to: Float.self, capacity: melFeatures.count)
            let startFrame = totalFrames - fixedFrames
            
            for channel in 0..<128 {
                // Source: channel * totalFrames + startFrame
                let srcOffset = channel * totalFrames + startFrame
                // Dest: channel * fixedFrames
                let destOffset = channel * fixedFrames
                
                // Copy fixedFrames floats
                (destPtr + destOffset).update(from: (srcPtr + srcOffset), count: fixedFrames)
            }
        } else {
            // Padding: Copy all totalFrames to the END of inputMel (or beginning?)
            // If we are at start of stream, we have [0...totalFrames].
            // We should put them at [0...totalFrames] and pad the rest?
            // NeMo padding usually pads at the end.
            
            let srcPtr = melFeatures.dataPointer.bindMemory(to: Float.self, capacity: melFeatures.count)
            
            for channel in 0..<128 {
                let srcOffset = channel * totalFrames
                let destOffset = channel * fixedFrames
                
                // Copy totalFrames floats
                (destPtr + destOffset).update(from: (srcPtr + srcOffset), count: totalFrames)
            }
        }
        
        // Create length array (always 129 if we padded/sliced to 129)
        // But encoder might need actual length if padded?
        // If we padded, actual length is totalFrames.
        // If we sliced, actual length is fixedFrames.
        let inputLength = min(totalFrames, fixedFrames)
        let melLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        melLengthArray[0] = NSNumber(value: inputLength)

        logger.debug("Preprocessor (Buffered): buffer=\(audioBuffer.count) samples, mel_total=\(totalFrames), extracted=\(fixedFrames)")

        // Run streaming encoder with cache state
        let (encoderOutput, encoderLength, newCacheChannel, newCacheTime, newCacheLen) = try await runStreamingEncoder(
            mel: inputMel,
            melLength: melLengthArray,
            model: models.encoder,
            encoderState: encoderState
        )

        // Update encoder cache state
        encoderState.updateCache(
            newCacheChannel: newCacheChannel,
            newCacheTime: newCacheTime,
            newCacheLen: newCacheLen
        )

        logger.debug("Encoder: output shape=\(encoderOutput.shape), length=\(encoderLength), cacheLen=\(newCacheLen)")

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
        
        logger.debug("Partial hypothesis: \(previousTokenCount) tokens, last_token=\(partialHypothesis.lastToken ?? -1)")

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
        
        logger.debug("New tokens this chunk: \(newTokens.count) (total now: \(fullHypothesis.ySequence.count))")
        
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
        if fullHypothesis.eouDetected || isFinal {
            logger.debug("Resetting partial hypothesis (EOU=\(fullHypothesis.eouDetected), final=\(isFinal))")
            accumulatedTokens.removeAll()
            accumulatedTimestamps.removeAll()
            accumulatedConfidences.removeAll()
            accumulatedScore = 0.0
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

    private func runStreamingEncoder(
        mel: MLMultiArray,
        melLength: MLMultiArray,
        model: MLModel,
        encoderState: StreamingEncoderState
    ) async throws -> (
        encoder: MLMultiArray, length: Int, cacheChannel: MLMultiArray, cacheTime: MLMultiArray, cacheLen: Int
    ) {
        // melLength is already MLMultiArray
        
        let cacheLenArray = try encoderState.createCacheLenArray()

        // Padding/Slicing is now handled in processChunk
        let inputMel = mel

        logger.debug(
            "Encoder input shapes: mel=\(inputMel.shape), cache_channel=\(encoderState.cacheLastChannel.shape), cache_time=\(encoderState.cacheLastTime.shape), cache_len=\(encoderState.cacheLastChannelLen)"
        )

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: inputMel),
            "mel_length": MLFeatureValue(multiArray: melLength),
            "cache_last_channel": MLFeatureValue(multiArray: encoderState.cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: encoderState.cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLenArray)
        ])

        let output = try await model.prediction(from: input)

        guard let encoder = output.featureValue(for: "encoder")?.multiArrayValue,
            let encoderLengthArray = output.featureValue(for: "encoder_length")?.multiArrayValue,
            let newCacheChannel = output.featureValue(for: "cache_last_channel_out")?.multiArrayValue,
            let newCacheTime = output.featureValue(for: "cache_last_time_out")?.multiArrayValue,
            let newCacheLenArray = output.featureValue(for: "cache_last_channel_len_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Streaming encoder output missing expected features")
        }

        let encoderLength = encoderLengthArray[0].intValue
        let newCacheLen = newCacheLenArray[0].intValue

        return (encoder, encoderLength, newCacheChannel, newCacheTime, newCacheLen)
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

            // Handle SentencePiece encoding (▁ = word boundary)
            let displayToken: String
            if token.hasPrefix("▁") {
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

        let text = textParts.joined().trimmingCharacters(in: .whitespaces)
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
    public let encoder: MLModel  // Streaming encoder with cache I/O
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "StreamingEouAsrModels")

    /// Load streaming EOU models from a directory
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        useMLPackage: Bool = true
    ) async throws -> StreamingEouAsrModels {
        let mlConfig = configuration ?? defaultConfiguration()

        let ext = useMLPackage ? ".mlpackage" : ".mlmodelc"
        logger.info("Loading streaming EOU models from \(directory.path) (extension: \(ext))")

        // Streaming model names
        let preprocessorURL = directory.appendingPathComponent("parakeet_eou_streaming_preprocessor" + ext)
        let encoderURL = directory.appendingPathComponent("parakeet_eou_streaming_encoder" + ext)
        let decoderURL = directory.appendingPathComponent("parakeet_eou_streaming_decoder" + ext)
        let jointURL = directory.appendingPathComponent("parakeet_eou_streaming_joint_decision" + ext)
        let vocabURL = directory.appendingPathComponent("vocab.json")

        // Check files exist
        let fm = FileManager.default
        for (name, url) in [
            ("Preprocessor", preprocessorURL),
            ("Encoder", encoderURL),
            ("Decoder", decoderURL),
            ("Joint", jointURL),
        ] {
            guard fm.fileExists(atPath: url.path) else {
                throw ASRError.processingFailed("\(name) model not found at \(url.path)")
            }
        }

        // Compile and load models (mlpackage -> mlmodelc)
        let preprocessor = try await compileAndLoad(preprocessorURL, configuration: mlConfig)
        let encoder = try await compileAndLoad(encoderURL, configuration: mlConfig)
        let decoder = try await compileAndLoad(decoderURL, configuration: mlConfig)
        let joint = try await compileAndLoad(jointURL, configuration: mlConfig)

        // Load vocabulary
        let vocabulary = try loadVocabulary(from: vocabURL)

        logger.info("Streaming EOU models loaded successfully (\(vocabulary.count) vocab tokens)")

        return StreamingEouAsrModels(
            preprocessor: preprocessor,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            configuration: mlConfig,
            vocabulary: vocabulary
        )
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
        let compiledURL = try await MLModel.compileModel(at: url)

        // Load the compiled model
        let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)

        // Clean up temporary compiled model
        try? FileManager.default.removeItem(at: compiledURL)

        return model
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
