@preconcurrency import CoreML
import Foundation
import OSLog

/// Manager for Parakeet Realtime EOU 120M speech recognition
///
/// This manager provides speech-to-text transcription using NVIDIA's Parakeet
/// Realtime EOU model, which is optimized for low-latency streaming and includes
/// end-of-utterance detection.
///
/// Key features:
/// - RNNT (Recurrent Neural Network Transducer) architecture
/// - 120M parameters (smaller and faster than TDT 0.6B)
/// - End-of-utterance detection via <EOU> token
/// - 80-160ms streaming latency
public final class EouAsrManager {
    private var models: EouAsrModels?
    private var decoderState: RnntDecoderState
    private let logger = AppLogger(category: "EouASR")
    private let config: RnntConfig

    /// Maximum audio samples per chunk (15 seconds at 16kHz)
    /// Model was exported with this fixed input size
    public static let maxAudioSamples = 240_000

    /// Sample rate expected by the model
    public static let sampleRate = 16_000

    public init(config: RnntConfig = .parakeetEOU) {
        self.config = config
        self.decoderState = RnntDecoderState.make(hiddenSize: config.decoderHiddenSize)
    }

    /// Initialize with pre-loaded models
    public func initialize(models: EouAsrModels) {
        self.models = models
        logger.info("EouAsrManager initialized with pre-loaded models")
    }

    /// Download and initialize models
    public func initialize(cacheDirectory: URL? = nil) async throws {
        let models = try await EouAsrModels.downloadAndLoad(to: cacheDirectory)
        self.models = models
        logger.info("EouAsrManager initialized with downloaded models")
    }

    /// Initialize from local directory (for debugging/testing)
    public func initializeFromLocalPath(_ directory: URL, useMLPackage: Bool = false) async throws {
        let models = try await EouAsrModels.load(
            from: directory,
            configuration: EouAsrModels.defaultConfiguration(),
            useMLPackage: useMLPackage
        )
        self.models = models
        logger.info("EouAsrManager initialized from local path: \(directory.path)")
    }

    /// Transcribe audio samples
    ///
    /// - Parameter audioSamples: Audio samples at 16kHz mono
    /// - Returns: Transcription result with text and metadata
    public func transcribe(_ audioSamples: [Float]) async throws -> EouTranscriptionResult {
        guard let models = models else {
            throw ASRError.notInitialized
        }

        let startTime = Date()

        // For long audio, process in chunks
        if audioSamples.count > Self.maxAudioSamples {
            return try await transcribeChunked(audioSamples, models: models, startTime: startTime)
        }

        // Pad audio to max length
        var paddedAudio = audioSamples
        if paddedAudio.count < Self.maxAudioSamples {
            paddedAudio.append(contentsOf: [Float](repeating: 0, count: Self.maxAudioSamples - paddedAudio.count))
        }

        // Run preprocessor
        let (melFeatures, melLength) = try await runPreprocessor(
            audio: paddedAudio,
            audioLength: audioSamples.count,
            model: models.preprocessor
        )

        // Run encoder
        let (encoderOutput, encoderLength) = try await runEncoder(
            mel: melFeatures,
            melLength: melLength,
            model: models.encoder
        )

        // Reset decoder state for fresh transcription
        decoderState.reset()

        // Run RNNT decoder
        let decoder = RnntDecoder(config: config)
        let hypothesis = try await decoder.decodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: encoderLength,
            decoderModel: models.decoder,
            jointModel: models.joint,
            decoderState: &decoderState
        )

        // Convert tokens to text
        let (text, tokenTimings) = convertTokensToText(
            tokens: hypothesis.ySequence,
            timestamps: hypothesis.timestamps,
            confidences: hypothesis.tokenConfidences,
            vocabulary: models.vocabulary
        )

        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(audioSamples.count) / Double(Self.sampleRate)

        return EouTranscriptionResult(
            text: text,
            confidence: hypothesis.score / Float(max(1, hypothesis.ySequence.count)),
            duration: audioDuration,
            processingTime: processingTime,
            tokenTimings: tokenTimings,
            eouDetected: hypothesis.eouDetected
        )
    }

    /// Transcribe long audio by processing in chunks
    private func transcribeChunked(
        _ audioSamples: [Float],
        models: EouAsrModels,
        startTime: Date
    ) async throws -> EouTranscriptionResult {
        let chunkSize = Self.maxAudioSamples
        let overlap = 16000  // 1 second overlap for continuity
        let stride = chunkSize - overlap

        var allText = ""
        var allTimings: [EouTokenTiming] = []
        var totalConfidence: Float = 0
        var chunkCount = 0
        var eouDetected = false

        var offset = 0
        while offset < audioSamples.count {
            let endIdx = min(offset + chunkSize, audioSamples.count)
            let chunkSamples = Array(audioSamples[offset..<endIdx])

            // Pad chunk if needed
            var paddedChunk = chunkSamples
            if paddedChunk.count < chunkSize {
                paddedChunk.append(contentsOf: [Float](repeating: 0, count: chunkSize - paddedChunk.count))
            }

            // Process chunk
            let (melFeatures, melLength) = try await runPreprocessor(
                audio: paddedChunk,
                audioLength: chunkSamples.count,
                model: models.preprocessor
            )

            let (encoderOutput, encoderLength) = try await runEncoder(
                mel: melFeatures,
                melLength: melLength,
                model: models.encoder
            )

            // Reset decoder for each chunk (stateless)
            decoderState.reset()

            let decoder = RnntDecoder(config: config)
            let hypothesis = try await decoder.decodeWithTimings(
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderLength,
                decoderModel: models.decoder,
                jointModel: models.joint,
                decoderState: &decoderState
            )

            let (text, tokenTimings) = convertTokensToText(
                tokens: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                vocabulary: models.vocabulary
            )

            // Adjust timings for chunk offset (frame offset = samples / samples_per_frame)
            // Each frame is ~80ms = 1280 samples at 16kHz
            let frameOffset = offset / 1280
            for timing in tokenTimings {
                allTimings.append(EouTokenTiming(
                    token: timing.token,
                    tokenId: timing.tokenId,
                    frameIndex: timing.frameIndex + frameOffset,
                    confidence: timing.confidence
                ))
            }

            if !text.isEmpty {
                if !allText.isEmpty {
                    allText += " "
                }
                allText += text
            }

            totalConfidence += hypothesis.score
            chunkCount += 1
            if hypothesis.eouDetected {
                eouDetected = true
            }

            offset += stride
        }

        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(audioSamples.count) / Double(Self.sampleRate)

        return EouTranscriptionResult(
            text: allText,
            confidence: chunkCount > 0 ? totalConfidence / Float(chunkCount) : 0,
            duration: audioDuration,
            processingTime: processingTime,
            tokenTimings: allTimings.isEmpty ? nil : allTimings,
            eouDetected: eouDetected
        )
    }

    /// Reset decoder state for new utterance
    public func resetState() {
        decoderState.reset()
    }

    /// Check if end-of-utterance was detected in last transcription
    public func wasEouDetected() -> Bool {
        return decoderState.eouDetected
    }

    // MARK: - Private Methods

    private func runPreprocessor(
        audio: [Float],
        audioLength: Int,
        model: MLModel
    ) async throws -> (mel: MLMultiArray, length: Int) {
        // Create input arrays
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

    private func runEncoder(
        mel: MLMultiArray,
        melLength: Int,
        model: MLModel
    ) async throws -> (encoder: MLMultiArray, length: Int) {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: melLength)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ])

        let output = try await model.prediction(from: input)

        guard let encoder = output.featureValue(for: "encoder")?.multiArrayValue,
            let encoderLengthArray = output.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Encoder output missing expected features")
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

/// Result of EOU transcription
public struct EouTranscriptionResult: Sendable {
    /// Transcribed text
    public let text: String

    /// Average confidence score (0-1)
    public let confidence: Float

    /// Audio duration in seconds
    public let duration: TimeInterval

    /// Processing time in seconds
    public let processingTime: TimeInterval

    /// Per-token timing information
    public let tokenTimings: [EouTokenTiming]?

    /// Whether end-of-utterance was detected
    public let eouDetected: Bool

    /// Real-time factor (processing speed)
    public var rtfx: Float {
        guard processingTime > 0 else { return 0 }
        return Float(duration / processingTime)
    }
}

/// Token timing information
public struct EouTokenTiming: Sendable {
    /// Decoded token string
    public let token: String

    /// Token ID in vocabulary
    public let tokenId: Int

    /// Encoder frame index when token was emitted
    public let frameIndex: Int

    /// Confidence score for this token
    public let confidence: Float

    /// Time in seconds (frame * 0.08s per frame)
    public var timeSeconds: Double {
        Double(frameIndex) * 0.08
    }
}
