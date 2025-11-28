@preconcurrency import CoreML
import Foundation
import OSLog

/// Batch manager for Parakeet Realtime EOU 120M speech recognition
///
/// This manager provides batch (non-streaming) inference using NVIDIA's
/// FastConformer encoder. It processes audio in fixed-size chunks
/// of up to 15 seconds (1501 mel frames).
///
/// Key features:
/// - Batch processing with padding to fixed encoder size
/// - End-of-utterance detection via <EOU> token
/// - Works with the original exported CoreML models
public final class BatchEouAsrManager {
    private var models: BatchEouAsrModels?
    private var decoderState: RnntDecoderState
    private let logger = AppLogger(category: "BatchEouASR")
    private let config: RnntConfig

    /// Sample rate expected by the model
    public static let sampleRate = 16_000

    public init(config: RnntConfig = .parakeetEOU) {
        self.config = config
        self.decoderState = RnntDecoderState.make(hiddenSize: config.decoderHiddenSize)
    }

    /// Initialize from a local directory containing models
    public func initializeFromLocalPath(_ directory: URL, useMLPackage: Bool = false) async throws {
        self.models = try await BatchEouAsrModels.load(from: directory, useMLPackage: useMLPackage)
        logger.info("BatchEouAsrManager initialized from: \(directory.path)")
    }

    // Max audio samples for batch processing (128 mel frames * 160 hop = 20480 samples = 1.28 seconds)
    // This matches the re-exported 1.28s model
    public static let maxAudioSamples = 20_480

    /// Transcribe audio samples
    ///
    /// - Parameters:
    ///   - audioSamples: Audio samples at 16kHz mono
    /// - Returns: Transcription result
    public func transcribe(_ audioSamples: [Float]) async throws -> BatchTranscriptionResult {
        guard let models = models else {
            throw ASRError.notInitialized
        }

        let startTime = Date()
        
        // Process in chunks if audio is longer than maxAudioSamples
        var allText = ""
        var totalConfidence: Float = 0
        var chunkCount = 0
        var allTimings: [EouTokenTiming] = []
        var anyEouDetected = false
        
        // Instantiate decoder helper
        let decoder = RnntDecoder(config: config)
        
        var offset = 0
        while offset < audioSamples.count {
            // Get chunk
            let end = min(offset + Self.maxAudioSamples, audioSamples.count)
            let chunk = Array(audioSamples[offset..<end])
            
            // Pad chunk if needed (always pad to maxAudioSamples for consistent input size)
            var paddedAudio = chunk
            if paddedAudio.count < Self.maxAudioSamples {
                paddedAudio.append(contentsOf: [Float](repeating: 0, count: Self.maxAudioSamples - paddedAudio.count))
            }
            
            // Run preprocessor
            let (melFeatures, melLength) = try await runPreprocessor(
                audio: paddedAudio,
                audioLength: paddedAudio.count,
                model: models.preprocessor
            )
            
            // Run encoder
            let encoderOutput: MLMultiArray
            let encoderLength: Int
            
            if let encoder = models.encoder {
                 let result = try await runBatchEncoder(
                    mel: melFeatures,
                    melLength: melLength,
                    model: encoder
                )
                encoderOutput = result.encoder
                encoderLength = result.length
            } else if let preEncode = models.preEncode, let conformer = models.conformer {
                 let result = try await runSplitEncoder(
                    mel: melFeatures,
                    melLength: melLength,
                    preEncode: preEncode,
                    conformer: conformer
                )
                encoderOutput = result.encoder
                encoderLength = result.length
            } else {
                throw ASRError.processingFailed("No encoder model available")
            }
            
            // Calculate valid encoder length based on actual audio content in this chunk
            // (Similar logic to before, but for this chunk)
            let chunkAudioLen = chunk.count
            let winLength = 400
            let hopLength = 160
            let correctMelLength = (chunkAudioLen - winLength) / hopLength + 1
            let validEncoderLength = Int(ceil(Double(correctMelLength) / 8.0))
            let finalEncoderLength = min(encoderLength, validEncoderLength)
            
            // Do NOT reset decoder state for each chunk.
            // We want to maintain LSTM context across chunks for better accuracy.
            // decoderState.reset()
            
            // Run decoder
            var decoderStateCopy = decoderState
            let hypothesis = try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: finalEncoderLength,
                decoderModel: models.decoder,
                jointModel: models.joint,
                decoderState: &decoderStateCopy
            )
            
            // Convert tokens
            let (text, tokenTimings) = convertTokensToText(
                tokens: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                vocabulary: models.vocabulary
            )
            
            if !text.isEmpty {
                if !allText.isEmpty {
                    allText += " "
                }
                allText += text
            }
            
            totalConfidence += hypothesis.score / Float(max(1, hypothesis.ySequence.count))
            chunkCount += 1
            
            // Adjust timings offset
            // Frame index needs to be offset by previous chunks
            // But frame index is relative to encoder output.
            // We can't easily merge frame indices without knowing total frames.
            // For now, just append them (indices will reset for each chunk).
            allTimings.append(contentsOf: tokenTimings)
            
            if hypothesis.eouDetected {
                anyEouDetected = true
            }
            
            offset += Self.maxAudioSamples
        }
        
        let avgConfidence = chunkCount > 0 ? totalConfidence / Float(chunkCount) : 0.0
        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(audioSamples.count) / Double(Self.sampleRate)
        
        return BatchTranscriptionResult(
            text: allText,
            confidence: avgConfidence,
            audioDuration: audioDuration,
            processingTime: processingTime,
            tokenTimings: allTimings,
            eouDetected: anyEouDetected
        )
    }

    /// Reset decoder state for new utterance
    public func resetDecoderState() {
        decoderState.reset()
        logger.debug("Decoder state reset")
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

    private func runBatchEncoder(
        mel: MLMultiArray,
        melLength: Int,
        model: MLModel
    ) async throws -> (encoder: MLMultiArray, length: Int, frameTimes: MLMultiArray?) {
        let melLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        melLengthArray[0] = NSNumber(value: melLength)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: melLengthArray),
        ])

        let output = try await model.prediction(from: input)

        guard let encoder = output.featureValue(for: "encoder")?.multiArrayValue,
            let encoderLengthArray = output.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Encoder output missing expected features")
        }

        let encoderLength = encoderLengthArray[0].intValue
        let frameTimes = output.featureValue(for: "frame_times")?.multiArrayValue

        return (encoder, encoderLength, frameTimes)
    }

    private func runSplitEncoder(
        mel: MLMultiArray,
        melLength: Int,
        preEncode: MLModel,
        conformer: MLModel
    ) async throws -> (encoder: MLMultiArray, length: Int) {
        // Constants from metadata
        // Diagnostic mode: Single large chunk (128 frames = 1.28s)
        let chunkInputSize = 128
        let melDim = 128
        let hiddenDim = 512
        
        var preEncodedChunks: [MLMultiArray] = []
        var totalPreEncodedLength = 0
        
        // Iterate through mel in chunks of 1500 (effectively one chunk for short audio)
        // Iterate through mel in chunks of 1500 (effectively one chunk for short audio)
        // CRITICAL FIX: The model is fixed to 1500 frames. If melLength is 1501 (due to windowing),
        // the loop would run twice, producing 378 frames instead of 189.
        // We force it to run only once for the first 1500 frames.
        var chunkIndex = 0
        while chunkIndex == 0 { // Force single chunk
            let startFrame = chunkIndex * chunkInputSize
            
            // Create chunk input [1, 128, 1500] (Channel-Major)
            let chunkInput = try MLMultiArray(shape: [1, NSNumber(value: melDim), NSNumber(value: chunkInputSize)], dataType: .float32)
            
            // Fill chunk
            for t in 0..<chunkInputSize {
                let globalT = startFrame + t
                
                for f in 0..<melDim {
                    let value: Float
                    if globalT < melLength {
                        // Read from mel [0, f, globalT]
                        value = mel[[0, NSNumber(value: f), NSNumber(value: globalT)]].floatValue
                    } else {
                        value = 0.0
                    }
                    
                    // Write to chunk [0, f, t]
                    chunkInput[[0, NSNumber(value: f), NSNumber(value: t)]] = NSNumber(value: value)
                }
            }
            
            // Run preEncode
            let melLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
            melLengthArray[0] = NSNumber(value: chunkInputSize)
            
            let preInput = try MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: chunkInput),
                "mel_length": MLFeatureValue(multiArray: melLengthArray),
            ])
            
            let preOutput = try await preEncode.prediction(from: preInput)
            
            guard let chunkOutput = preOutput.featureValue(for: "pre_encoded")?.multiArrayValue
            else {
                throw ASRError.processingFailed("PreEncode output missing expected features")
            }
            
            // Debug stats
            let ptr = chunkOutput.dataPointer.bindMemory(to: Float.self, capacity: chunkOutput.count)
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            var sumVal: Float = 0
            for i in 0..<chunkOutput.count {
                let v = ptr[i]
                minVal = min(minVal, v)
                maxVal = max(maxVal, v)
                sumVal += v
            }
            let meanVal = sumVal / Float(chunkOutput.count)
            logger.debug("PreEncode chunk \(chunkIndex) stats: min=\(minVal), max=\(maxVal), mean=\(meanVal)")
            
            // Accumulate output
            preEncodedChunks.append(chunkOutput)
            totalPreEncodedLength += chunkOutput.shape[1].intValue
            
            chunkIndex += 1
        }
        
        // 2. Concatenate chunks
        // Result shape [1, totalPreEncodedLength, 512]
        let concatenated = try MLMultiArray(shape: [1, NSNumber(value: totalPreEncodedLength), NSNumber(value: hiddenDim)], dataType: .float32)
        
        var offset = 0
        for chunk in preEncodedChunks {
            let chunkLen = chunk.shape[1].intValue
            let ptr = UnsafeMutableBufferPointer(start: concatenated.dataPointer.assumingMemoryBound(to: Float.self), count: concatenated.count)
            let chunkPtr = UnsafeBufferPointer(start: chunk.dataPointer.assumingMemoryBound(to: Float.self), count: chunk.count)
            
            // Copy chunk data
            let startIdx = offset * hiddenDim
            let count = chunkLen * hiddenDim
            
            for i in 0..<count {
                ptr[startIdx + i] = chunkPtr[i]
            }
            
            offset += chunkLen
        }
        
        // 3. Run Conformer
        let confLenArray = try MLMultiArray(shape: [1], dataType: .int32)
        confLenArray[0] = NSNumber(value: totalPreEncodedLength)
        
        let confInput = try MLDictionaryFeatureProvider(dictionary: [
            "pre_encoded": MLFeatureValue(multiArray: concatenated),
            "pre_encoded_length": MLFeatureValue(multiArray: confLenArray),
        ])

        let confOutput = try await conformer.prediction(from: confInput)

        guard let encoder = confOutput.featureValue(for: "encoder")?.multiArrayValue,
              let encoderLengthArray = confOutput.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Conformer output missing expected features")
        }

        let encoderLength = encoderLengthArray[0].intValue
        return (encoder, encoderLength)
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

/// Result of batch transcription
public struct BatchTranscriptionResult: Sendable {
    /// Transcribed text
    public let text: String

    /// Average confidence score (0-1)
    public let confidence: Float

    /// Audio duration in seconds
    public let audioDuration: TimeInterval

    /// Processing time in seconds
    public let processingTime: TimeInterval

    /// Per-token timing information
    public let tokenTimings: [EouTokenTiming]?

    /// Whether end-of-utterance was detected
    public let eouDetected: Bool

    /// Real-time factor (processing speed)
    public var rtfx: Float {
        guard processingTime > 0 else { return 0 }
        return Float(audioDuration / processingTime)
    }
}

/// Holds loaded CoreML models for batch Parakeet EOU
public struct BatchEouAsrModels: Sendable {
    public let preprocessor: MLModel
    public let encoder: MLModel?
    public let preEncode: MLModel?
    public let conformer: MLModel?
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "BatchEouAsrModels")

    /// Load batch EOU models from a directory
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        useMLPackage: Bool = false
    ) async throws -> BatchEouAsrModels {
        let mlConfig = configuration ?? defaultConfiguration()

        let ext = useMLPackage ? ".mlpackage" : ".mlmodelc"
        logger.info("Loading batch EOU models from \(directory.path) (extension: \(ext))")

        // Batch model names (original export)
        let preprocessorURL = directory.appendingPathComponent("preprocessor" + ext)
        let encoderURL = directory.appendingPathComponent("encoder" + ext)
        let preEncodeURL = directory.appendingPathComponent("pre_encode" + ext)
        let conformerURL = directory.appendingPathComponent("conformer_batch" + ext)
        let decoderURL = directory.appendingPathComponent("decoder" + ext)
        let jointURL = directory.appendingPathComponent("joint_decision" + ext)
        let vocabURL = directory.appendingPathComponent("vocab.json")

        // Check files exist
        let fm = FileManager.default
        var hasSingleEncoder = fm.fileExists(atPath: encoderURL.path)
        var hasSplitEncoder = fm.fileExists(atPath: preEncodeURL.path) && fm.fileExists(atPath: conformerURL.path)

        if !hasSingleEncoder && !hasSplitEncoder {
             throw ASRError.processingFailed("Encoder model(s) not found at \(directory.path)")
        }

        for (name, url) in [
            ("Preprocessor", preprocessorURL),
            ("Decoder", decoderURL),
            ("Joint", jointURL),
        ] {
            guard fm.fileExists(atPath: url.path) else {
                throw ASRError.processingFailed("\(name) model not found at \(url.path)")
            }
        }

        // Compile and load models
        // Preprocessor often has dynamic shapes, safer on CPU
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        let preprocessor = try await compileAndLoad(preprocessorURL, configuration: cpuConfig)
        
        var encoder: MLModel?
        var preEncode: MLModel?
        var conformer: MLModel?

        if hasSingleEncoder {
            encoder = try await compileAndLoad(encoderURL, configuration: mlConfig)
        } else {
            preEncode = try await compileAndLoad(preEncodeURL, configuration: mlConfig)
            conformer = try await compileAndLoad(conformerURL, configuration: mlConfig)
        }

        let decoder = try await compileAndLoad(decoderURL, configuration: cpuConfig)
        let joint = try await compileAndLoad(jointURL, configuration: cpuConfig)

        // Load vocabulary
        let vocabulary = try loadVocabulary(from: vocabURL)

        logger.info("Batch EOU models loaded successfully (\(vocabulary.count) vocab tokens)")

        return BatchEouAsrModels(
            preprocessor: preprocessor,
            encoder: encoder,
            preEncode: preEncode,
            conformer: conformer,
            decoder: decoder,
            joint: joint,
            configuration: mlConfig,
            vocabulary: vocabulary
        )
    }

    /// Default model configuration
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return config
    }

    // MARK: - Private

    private static func compileAndLoad(
        _ url: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        if url.pathExtension == "mlmodelc" {
            return try await MLModel.load(contentsOf: url, configuration: configuration)
        }

        logger.info("Compiling \(url.lastPathComponent)...")
        let compiledURL = try await MLModel.compileModel(at: url)
        let model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        try? FileManager.default.removeItem(at: compiledURL)
        return model
    }

    private static func loadVocabulary(from url: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: url)

        if let json = try JSONSerialization.jsonObject(with: data) as? [String: String] {
            var vocabulary: [Int: String] = [:]
            for (idStr, token) in json {
                if let id = Int(idStr) {
                    vocabulary[id] = token
                }
            }
            return vocabulary
        }

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
