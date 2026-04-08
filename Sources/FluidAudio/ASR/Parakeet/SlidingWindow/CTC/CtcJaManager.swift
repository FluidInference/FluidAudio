@preconcurrency import CoreML
import Foundation

/// Manager for Parakeet CTC ja (Japanese) transcription
///
/// This manager handles the full pipeline for Japanese CTC transcription:
/// 1. Preprocessor: Audio → Mel spectrogram
/// 2. Encoder: Mel → Encoder features
/// 3. CTC Decoder: Encoder features → CTC logits
/// 4. Greedy CTC decoding: Logits → Text
public actor CtcJaManager {

    private let models: CtcJaModels
    private let maxAudioSamples: Int
    private let sampleRate: Int

    private static let logger = AppLogger(category: "CtcJaManager")

    /// Initialize with pre-loaded models
    public init(models: CtcJaModels, maxAudioSamples: Int = 240_000, sampleRate: Int = 16_000) {
        self.models = models
        self.maxAudioSamples = maxAudioSamples
        self.sampleRate = sampleRate
    }

    /// Convenience initializer that loads models from default cache directory
    public static func load(
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcJaManager {
        let models = try await CtcJaModels.downloadAndLoad(
            configuration: configuration,
            progressHandler: progressHandler
        )
        return CtcJaManager(models: models)
    }

    /// Transcribe audio to text using CTC decoding
    ///
    /// - Parameters:
    ///   - audio: Audio samples (mono, 16kHz)
    ///   - audioLength: Optional audio length (if nil, uses audio.count)
    /// - Returns: Transcribed Japanese text
    public func transcribe(
        audio: [Float],
        audioLength: Int? = nil
    ) throws -> String {
        let actualLength = audioLength ?? audio.count

        // Pad or truncate audio to maxAudioSamples
        let paddedAudio = padOrTruncateAudio(audio, targetLength: maxAudioSamples)

        // Step 1: Preprocessor (audio → mel spectrogram)
        let melOutput = try runPreprocessor(audio: paddedAudio, audioLength: actualLength)

        // Step 2: Encoder (mel → encoder features)
        let encoderOutput = try runEncoder(mel: melOutput.mel, melLength: melOutput.melLength)

        // Step 3: CTC Decoder (encoder features → CTC logits)
        let ctcLogits = try runCtcDecoder(encoderOutput: encoderOutput)

        // Step 4: CTC decoding (logits → text)
        let text = greedyCtcDecode(logits: ctcLogits)

        return text
    }

    /// Transcribe audio file to text
    ///
    /// - Parameters:
    ///   - audioURL: URL to audio file (will be resampled to 16kHz mono)
    /// - Returns: Transcribed Japanese text
    public func transcribe(audioURL: URL) throws -> String {
        // Load and convert audio
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)

        return try transcribe(audio: samples)
    }

    // MARK: - Private Pipeline Methods

    private struct MelOutput {
        let mel: MLMultiArray
        let melLength: MLMultiArray
    }

    private func runPreprocessor(audio: [Float], audioLength: Int) throws -> MelOutput {
        // Create input arrays
        let audioArray = try MLMultiArray(shape: [1, maxAudioSamples as NSNumber], dataType: .float32)
        for (i, sample) in audio.enumerated() where i < maxAudioSamples {
            audioArray[i] = NSNumber(value: sample)
        }

        let audioLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        audioLengthArray[0] = NSNumber(value: min(audioLength, maxAudioSamples))

        // Run preprocessor
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "audio_signal": MLFeatureValue(multiArray: audioArray),
                "length": MLFeatureValue(multiArray: audioLengthArray),
            ]
        )
        let output = try models.preprocessor.prediction(from: input)

        guard
            let mel = output.featureValue(for: "mel_features")?.multiArrayValue,
            let melLength = output.featureValue(for: "mel_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Failed to extract mel_features or mel_length from preprocessor output")
        }

        return MelOutput(mel: mel, melLength: melLength)
    }

    private func runEncoder(mel: MLMultiArray, melLength: MLMultiArray) throws -> MLMultiArray {
        // Run encoder
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "mel_features": MLFeatureValue(multiArray: mel),
                "mel_length": MLFeatureValue(multiArray: melLength),
            ]
        )
        let output = try models.encoder.prediction(from: input)

        guard let encoderOutput = output.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw ASRError.processingFailed("Failed to extract encoder_output from encoder")
        }

        return encoderOutput
    }

    private func runCtcDecoder(encoderOutput: MLMultiArray) throws -> MLMultiArray {
        // Run CTC decoder head
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "encoder_output": MLFeatureValue(multiArray: encoderOutput)
            ]
        )
        let output = try models.decoder.prediction(from: input)

        guard let ctcLogits = output.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Failed to extract ctc_logits from decoder")
        }

        return ctcLogits
    }

    private func greedyCtcDecode(logits: MLMultiArray) -> String {
        // logits shape: [1, T, vocab_size+1] where T is time steps (188)
        // vocab_size = 3072, blank_id = 3072

        let timeSteps = logits.shape[1].intValue
        let vocabSize = logits.shape[2].intValue

        var decoded: [Int] = []
        var prevLabel: Int? = nil

        for t in 0..<timeSteps {
            // Find argmax at this time step
            var maxLogit: Float = -.infinity
            var maxLabel = 0

            for v in 0..<vocabSize {
                let logit = logits[[0, t as NSNumber, v as NSNumber]].floatValue
                if logit > maxLogit {
                    maxLogit = logit
                    maxLabel = v
                }
            }

            // CTC collapse: skip blanks and repeats
            if maxLabel != models.blankId && maxLabel != prevLabel {
                decoded.append(maxLabel)
            }
            prevLabel = maxLabel
        }

        // Convert token IDs to text
        var text = ""
        for tokenId in decoded {
            if let token = models.vocabulary[tokenId] {
                text += token
            }
        }

        // Replace SentencePiece underscores with spaces
        text = text.replacingOccurrences(of: "▁", with: " ")

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func padOrTruncateAudio(_ audio: [Float], targetLength: Int) -> [Float] {
        var result = audio
        if result.count < targetLength {
            // Pad with zeros
            result.append(contentsOf: Array(repeating: 0.0, count: targetLength - result.count))
        } else if result.count > targetLength {
            // Truncate
            result = Array(result.prefix(targetLength))
        }
        return result
    }
}
