import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereAsrManager")

// MARK: - Cohere Transcribe ASR Manager

/// Manages Cohere Transcribe CoreML inference.
///
/// Pipeline:
/// 1. Audio -> mel spectrogram -> encoder -> hidden states (1, 376, 1024)
/// 2. Decode loop with KV cache:
///    - Feed previous token + encoder_hidden_states
///    - Get logits + updated cache
///    - Sample next token
/// 3. Continue until EOS or max tokens
@available(macOS 14, iOS 17, *)
public actor CohereAsrManager {
    private var models: CohereAsrModels?
    private let melExtractor: CohereMelSpectrogram

    public init() {
        self.melExtractor = CohereMelSpectrogram()
    }

    /// Load models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await CohereAsrModels.load(from: directory, computeUnits: computeUnits)
        logger.info("Cohere Transcribe models loaded successfully")
    }

    /// Transcribe raw audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - maxNewTokens: Maximum number of tokens to generate.
    /// - Returns: Transcribed text.
    public func transcribe(
        audioSamples: [Float],
        maxNewTokens: Int = 200
    ) async throws -> String {
        guard let models = models else {
            throw CohereAsrError.generationFailed("Models not loaded")
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Step 1: Extract mel spectrogram
        let mel = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw CohereAsrError.invalidInput("Audio too short to extract mel spectrogram")
        }

        let nFrames = mel[0].count

        // Pad to 3001 frames (max length)
        let paddedMel = padMelSpectrogram(mel, targetFrames: 3001)

        // Step 2: Encode audio
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let encoderHidden = try await encodeAudio(paddedMel: paddedMel, featureLength: nFrames, models: models)
        let encodeTime = CFAbsoluteTimeGetCurrent() - encodeStart
        logger.debug("Encoder: \(String(format: "%.3f", encodeTime))s")

        // Step 3: Decode with KV cache
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let tokens = try await decode(
            encoderHidden: encoderHidden,
            maxNewTokens: maxNewTokens,
            models: models
        )
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        logger.debug("Decoder: \(String(format: "%.3f", decodeTime))s (\(tokens.count) tokens)")

        let totalTime = CFAbsoluteTimeGetCurrent() - start
        logger.info(
            "Transcribed \(String(format: "%.2f", Float(audioSamples.count) / 16000.0))s audio in \(String(format: "%.3f", totalTime))s"
        )

        // Step 4: Detokenize
        let text = convertTokensToText(tokens, vocabulary: models.vocabulary)

        return text
    }

    // MARK: - Private Helpers

    /// Pad mel spectrogram to target number of frames.
    private func padMelSpectrogram(_ mel: [[Float]], targetFrames: Int) -> [[Float]] {
        let nMels = mel.count
        let nFrames = mel[0].count

        guard nFrames < targetFrames else {
            return mel
        }

        var padded = [[Float]](repeating: [Float](repeating: 0, count: targetFrames), count: nMels)
        for m in 0..<nMels {
            for f in 0..<nFrames {
                padded[m][f] = mel[m][f]
            }
        }

        return padded
    }

    /// Encode audio mel spectrogram to hidden states.
    private func encodeAudio(
        paddedMel: [[Float]],
        featureLength: Int,
        models: CohereAsrModels
    ) async throws -> MLMultiArray {
        // Create input MLMultiArray (1, 128, 3001)
        let inputShape = [1, CohereAsrConfig.numMelBins, 3001] as [NSNumber]
        guard
            let inputFeatures = try? MLMultiArray(
                shape: inputShape,
                dataType: .float32
            )
        else {
            throw CohereAsrError.encodingFailed("Failed to create input MLMultiArray")
        }

        // Fill with mel data (shape: [1, 128, 3001])
        for m in 0..<CohereAsrConfig.numMelBins {
            for f in 0..<3001 {
                let index = [0, m, f] as [NSNumber]
                inputFeatures[index] = NSNumber(value: paddedMel[m][f])
            }
        }

        // Create feature length input
        guard let featureLengthArray = try? MLMultiArray(shape: [1], dataType: .int32) else {
            throw CohereAsrError.encodingFailed("Failed to create feature_length MLMultiArray")
        }
        featureLengthArray[0] = NSNumber(value: featureLength)

        // Run encoder
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: inputFeatures),
            "feature_length": MLFeatureValue(multiArray: featureLengthArray),
        ])

        let encoderOutput = try await models.encoder.prediction(from: encoderInput)

        guard let hiddenStates = encoderOutput.featureValue(for: "encoder_outputs")?.multiArrayValue else {
            throw CohereAsrError.encodingFailed("Failed to get encoder output")
        }

        return hiddenStates
    }

    /// Decode with KV cache.
    private func decode(
        encoderHidden: MLMultiArray,
        maxNewTokens: Int,
        models: CohereAsrModels
    ) async throws -> [Int] {
        // Initialize KV cache: (8, 8, 108, 128)
        let cacheShape =
            [
                CohereAsrConfig.numDecoderLayers,
                CohereAsrConfig.numDecoderHeads,
                CohereAsrConfig.maxSeqLen,
                CohereAsrConfig.headDim,
            ] as [NSNumber]

        guard
            let cacheK = try? MLMultiArray(shape: cacheShape, dataType: .float16),
            let cacheV = try? MLMultiArray(shape: cacheShape, dataType: .float16)
        else {
            throw CohereAsrError.decodingFailed("Failed to create KV cache arrays")
        }

        // Initialize with zeros
        let cacheSize = cacheK.count
        for i in 0..<cacheSize {
            cacheK[i] = 0
            cacheV[i] = 0
        }

        // Cross-attention mask: (1, 1, 1, 376) - all ones
        guard
            let crossAttentionMask = try? MLMultiArray(shape: [1, 1, 1, 376], dataType: .float16)
        else {
            throw CohereAsrError.decodingFailed("Failed to create cross-attention mask")
        }
        for i in 0..<376 {
            crossAttentionMask[[0, 0, 0, i] as [NSNumber]] = 1.0
        }

        var tokens = [Int]()
        var currentToken = CohereAsrConfig.SpecialTokens.startToken

        for step in 0..<maxNewTokens {
            // Create decoder input
            guard let inputId = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create input_id array")
            }
            inputId[0] = NSNumber(value: currentToken)

            guard let stepArray = try? MLMultiArray(shape: [1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create step array")
            }
            stepArray[0] = NSNumber(value: step)

            // Run decoder
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_id": MLFeatureValue(multiArray: inputId),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "cache_k": MLFeatureValue(multiArray: cacheK),
                "cache_v": MLFeatureValue(multiArray: cacheV),
                "step": MLFeatureValue(multiArray: stepArray),
                "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
            ])

            let decoderOutput = try await models.decoder.prediction(from: decoderInput)

            // Get logits and sample next token
            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.decodingFailed("Failed to get logits")
            }

            let nextToken = argmax(logits)
            tokens.append(nextToken)

            // Update cache
            guard
                let newCacheK = decoderOutput.featureValue(for: "new_cache_k")?.multiArrayValue,
                let newCacheV = decoderOutput.featureValue(for: "new_cache_v")?.multiArrayValue
            else {
                throw CohereAsrError.decodingFailed("Failed to get updated cache")
            }

            // Copy updated cache
            for i in 0..<cacheSize {
                cacheK[i] = newCacheK[i]
                cacheV[i] = newCacheV[i]
            }

            // Check for EOS
            if nextToken == CohereAsrConfig.SpecialTokens.eosToken {
                break
            }

            currentToken = nextToken
        }

        return tokens
    }

    /// Find argmax of logits array.
    private func argmax(_ logits: MLMultiArray) -> Int {
        let count = logits.count
        var maxIdx = 0
        var maxVal = logits[0].floatValue

        for i in 1..<count {
            let val = logits[i].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }

        return maxIdx
    }

    /// Convert token IDs to text using SentencePiece conventions.
    private func convertTokensToText(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        guard !tokenIds.isEmpty else { return "" }

        // Filter out special tokens and lookup each token
        let tokens = tokenIds.compactMap { tokenId -> String? in
            // Skip special tokens (IDs <= 4 or EOS)
            if tokenId <= 4 || tokenId == CohereAsrConfig.SpecialTokens.eosToken {
                return nil
            }

            guard let token = vocabulary[tokenId] else {
                return nil
            }

            // Skip control tokens (anything starting with <|)
            if token.hasPrefix("<|") {
                return nil
            }

            return token
        }.filter { !$0.isEmpty }

        // Join tokens and replace SentencePiece word boundary marker with spaces
        return tokens.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
