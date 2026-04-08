@preconcurrency import CoreML
import Foundation
import AVFoundation

/// Manager for Parakeet TDT ja (Japanese) transcription using Token-and-Duration Transducer decoding
///
/// This manager handles the full pipeline for Japanese TDT transcription:
/// 1. Preprocessor: Audio → Mel spectrogram
/// 2. Encoder: Mel → Encoder features
/// 3. TDT Decoder: Token prediction with duration modeling
/// 4. Joint Network: Combines encoder and decoder for predictions
public actor TdtJaManager {

    private let models: TdtJaModels
    private let maxAudioSamples: Int
    private let sampleRate: Int
    private let config: ASRConfig
    private let tdtDecoder: TdtDecoderV3

    // Decoder state for maintaining LSTM context
    private var decoderState: TdtDecoderState

    private static let logger = AppLogger(category: "TdtJaManager")

    /// Initialize with pre-loaded models
    public init(
        models: TdtJaModels,
        maxAudioSamples: Int = 240_000,
        sampleRate: Int = 16_000
    ) {
        self.models = models
        self.maxAudioSamples = maxAudioSamples
        self.sampleRate = sampleRate

        // Configure for Japanese TDT (1024 hidden size, 3072 vocab/blank)
        let tdtConfig = TdtConfig(blankId: 3072)
        self.config = ASRConfig(
            sampleRate: 16_000,
            tdtConfig: tdtConfig,
            encoderHiddenSize: 1024
        )
        self.tdtDecoder = TdtDecoderV3(config: config)
        self.decoderState = TdtDecoderState.make(decoderLayers: 2)  // Japanese uses 2-layer LSTM
    }

    /// Convenience initializer that loads models from default cache directory
    public static func load(
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> TdtJaManager {
        let models = try await TdtJaModels.downloadAndLoad(
            configuration: configuration,
            progressHandler: progressHandler
        )
        return TdtJaManager(models: models)
    }

    /// Transcribe audio to text using TDT decoding
    ///
    /// - Parameters:
    ///   - audio: Audio samples (mono, 16kHz)
    ///   - audioLength: Optional audio length (if nil, uses audio.count)
    /// - Returns: Transcribed Japanese text
    public func transcribe(
        audio: [Float],
        audioLength: Int? = nil
    ) async throws -> String {
        let actualLength = audioLength ?? audio.count

        // Pad or truncate audio to maxAudioSamples based on actual array size
        var processedAudio = audio
        if audio.count < maxAudioSamples {
            processedAudio.append(contentsOf: [Float](repeating: 0, count: maxAudioSamples - audio.count))
        } else if audio.count > maxAudioSamples {
            processedAudio = Array(audio.prefix(maxAudioSamples))
        }

        // Step 1: Preprocessor (audio → mel spectrogram)
        let audioArray = try MLMultiArray(shape: [1, maxAudioSamples as NSNumber], dataType: .float32)
        for (i, sample) in processedAudio.enumerated() where i < maxAudioSamples {
            audioArray[i] = NSNumber(value: sample)
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: min(actualLength, maxAudioSamples))

        let preprocessorInput = try createFeatureProvider(features: [
            ("audio_signal", audioArray),
            ("length", lengthArray),  // CTC preprocessor uses "length" not "audio_length"
        ])

        let preprocessorOutput = try await models.preprocessor.prediction(from: preprocessorInput)
        guard
            let melOutput = preprocessorOutput.featureValue(for: "mel_features")?.multiArrayValue,  // CTC outputs "mel_features"
            let melLengthOutput = preprocessorOutput.featureValue(for: "mel_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Failed to get preprocessor output")
        }

        // Step 2: Encoder (mel → encoder features)
        let encoderInput = try createFeatureProvider(features: [
            ("mel_features", melOutput),  // CTC encoder expects "mel_features"
            ("mel_length", melLengthOutput),
        ])

        let encoderOutput = try await models.encoder.prediction(from: encoderInput)
        guard
            let encoderFeatures = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,  // CTC outputs "encoder_output"
            let encoderLengthOutput = encoderOutput.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Failed to get encoder output")
        }

        let encoderLength = encoderLengthOutput[0].intValue

        // Step 3: TDT Decoding (encoder features → tokens)
        // Validate joint model is present (required for TDT)
        guard let jointModel = models.joint else {
            throw ASRError.processingFailed("TDT models require a joint model")
        }

        // Extract decoder and state to local variables for inout passing
        var localDecoderState = decoderState
        let localTdtDecoder = tdtDecoder
        let hypothesis = try await localTdtDecoder.decodeWithTimings(
            encoderOutput: encoderFeatures,
            encoderSequenceLength: encoderLength,
            actualAudioFrames: encoderLength,
            decoderModel: models.decoder,
            jointModel: jointModel,
            decoderState: &localDecoderState,
            contextFrameAdjustment: 0,
            isLastChunk: true,
            globalFrameOffset: 0
        )

        // Step 4: Convert tokens to text
        Self.logger.info("Decoded tokens: \(hypothesis.ySequence.prefix(20))")
        let text = tokensToText(hypothesis.ySequence)
        Self.logger.info("Final text: '\(text)'")

        // Reset decoder state for next transcription
        decoderState = TdtDecoderState.make(decoderLayers: 2)

        return text
    }

    /// Reset the decoder state (clears LSTM context)
    public func resetDecoderState() {
        decoderState = TdtDecoderState.make(decoderLayers: 2)
    }

    // MARK: - Helper Methods

    private func createFeatureProvider(
        features: [(name: String, array: MLMultiArray)]
    ) throws -> MLFeatureProvider {
        var featureDict: [String: MLFeatureValue] = [:]
        for (name, array) in features {
            featureDict[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: featureDict)
    }

    private func createScalarArray(
        value: Int,
        shape: [NSNumber] = [1],
        dataType: MLMultiArrayDataType = .int32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }

    /// Convert token IDs to Japanese text
    private func tokensToText(_ tokens: [Int]) -> String {
        var pieces: [String] = []
        for tokenId in tokens {
            if tokenId == models.blankId {
                continue  // Skip blank tokens
            }
            if let piece = models.vocabulary[tokenId] {
                pieces.append(piece)
            }
        }

        // Join SentencePiece tokens and clean up
        let rawText = pieces.joined()

        // Replace SentencePiece underscore with space
        var text = rawText.replacingOccurrences(of: "▁", with: " ")

        // Remove leading/trailing whitespace
        text = text.trimmingCharacters(in: .whitespaces)

        return text
    }

    /// Convenience method to transcribe from audio file URL
    ///
    /// - Parameter audioURL: URL to audio file (wav, mp3, etc.)
    /// - Returns: Transcribed Japanese text
    public func transcribe(audioURL: URL) async throws -> String {
        let audio = try AudioConverter().resampleAudioFile(audioURL)
        return try await transcribe(audio: audio)
    }
}
