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
    /// - Important: The cache-external decoder only works reliably for **Spanish** (18-24% WER).
    ///   Other languages may hallucinate and produce wrong-language output (>50% WER).
    ///   For multilingual ASR, use Whisper or Qwen3 models instead.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - language: Target language for transcription. Only `.spanish` is reliable.
    ///   - maxNewTokens: Maximum number of tokens to generate.
    /// - Returns: Transcribed text.
    public func transcribe(
        audioSamples: [Float],
        language: CohereAsrConfig.Language? = .english,
        maxNewTokens: Int = 200
    ) async throws -> String {
        guard let models = models else {
            throw CohereAsrError.generationFailed("Models not loaded")
        }

        // IMPORTANT: Cache-external decoder only works reliably for Spanish
        // Other languages may hallucinate (produce wrong-language output)
        // For multilingual ASR, use Whisper or Qwen3 models instead
        if let lang = language, lang != .spanish {
            logger.warning(
                "Cache-external decoder only supports Spanish reliably. Language '\(lang.rawValue)' may produce incorrect output. Consider using Whisper or Qwen3 for multilingual ASR."
            )
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Step 1: Extract mel spectrogram
        let mel = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw CohereAsrError.invalidInput("Audio too short to extract mel spectrogram")
        }

        let nFrames = mel[0].count

        // Pad to 3500 frames (max length)
        let paddedMel = padMelSpectrogram(mel, targetFrames: 3500)

        // Step 2: Encode audio
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let encoderHidden = try await encodeAudio(paddedMel: paddedMel, featureLength: nFrames, models: models)
        let encodeTime = CFAbsoluteTimeGetCurrent() - encodeStart
        logger.debug("Encoder: \(String(format: "%.3f", encodeTime))s")

        // Step 3: Decode with KV cache
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let tokens: [Int]

        // Use cache-external decoder (stateful not supported on macOS)
        tokens = try await decodeCacheExternal(
            encoderHidden: encoderHidden,
            language: language,
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
        // Create input MLMultiArray (1, 128, 3500)
        let inputShape = [1, CohereAsrConfig.numMelBins, 3500] as [NSNumber]
        guard
            let inputFeatures = try? MLMultiArray(
                shape: inputShape,
                dataType: .float32
            )
        else {
            throw CohereAsrError.encodingFailed("Failed to create input MLMultiArray")
        }

        // Fill with mel data (shape: [1, 128, 3500])
        for m in 0..<CohereAsrConfig.numMelBins {
            for f in 0..<3500 {
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

        guard let hiddenStates = encoderOutput.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw CohereAsrError.encodingFailed("Failed to get encoder output")
        }

        return hiddenStates
    }

    /// Decode with stateful decoder (CoreML manages KV cache).
    /// NOTE: Stateful decoders are iOS-only (newState() unavailable on macOS).
    @available(iOS 18, *)
    private func decodeStateful(
        encoderHidden: MLMultiArray,
        language: CohereAsrConfig.Language?,
        maxNewTokens: Int,
        models: CohereAsrModels
    ) async throws -> [Int] {
        // Build prompt sequence for the language
        let prompt = language?.promptSequence ?? [CohereAsrConfig.SpecialTokens.startToken]
        var tokens = [Int]()

        // Cross-attention mask: (1, 1, 1, encoder_seq_len) - all ones
        let encoderSeqLen = encoderHidden.shape[1].intValue
        guard
            let crossAttentionMask = try? MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: encoderSeqLen)], dataType: .float32)
        else {
            throw CohereAsrError.decodingFailed("Failed to create cross-attention mask")
        }
        for i in 0..<encoderSeqLen {
            crossAttentionMask[[0, 0, 0, i] as [NSNumber]] = 1.0
        }

        // Initialize stateful decoder
        let state = models.decoder.newState()

        var currentToken = prompt[0]
        let effectiveMaxTokens = min(maxNewTokens + prompt.count, CohereAsrConfig.maxSeqLen)

        for totalStep in 0..<effectiveMaxTokens {
            // Use prompt tokens first, then generated tokens
            if totalStep < prompt.count {
                currentToken = prompt[totalStep]
            }

            // Create decoder inputs
            guard let inputId = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create input_id array")
            }
            inputId[0] = NSNumber(value: currentToken)

            guard let positionId = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create position_id array")
            }
            positionId[0] = NSNumber(value: totalStep)

            // Attention mask (causal): (1, 1, 1, totalStep+1) - all zeros for stateful
            guard
                let attentionMask = try? MLMultiArray(
                    shape: [1, 1, 1, NSNumber(value: totalStep + 1)], dataType: .float32)
            else {
                throw CohereAsrError.decodingFailed("Failed to create attention mask")
            }
            for i in 0..<(totalStep + 1) {
                attentionMask[[0, 0, 0, i] as [NSNumber]] = 0
            }

            // Build decoder input dictionary (no cache inputs - managed by CoreML state)
            let inputDict: [String: MLFeatureValue] = [
                "input_id": MLFeatureValue(multiArray: inputId),
                "position_ids": MLFeatureValue(multiArray: positionId),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
            ]

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let decoderOutput = try await models.decoder.prediction(from: decoderInput, using: state)

            // Get logits and sample next token
            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.decodingFailed("Failed to get logits")
            }

            let nextToken = argmax(logits)

            // Only collect tokens after the prompt
            if totalStep >= prompt.count - 1 {
                // Check for EOS before adding to tokens
                if nextToken == CohereAsrConfig.SpecialTokens.eosToken {
                    break
                }

                tokens.append(nextToken)
            }

            // Use the predicted next token (unless we're still feeding the prompt)
            if totalStep < prompt.count - 1 {
                // Still feeding prompt, nextToken is ignored
                currentToken = prompt[totalStep + 1]
            } else {
                currentToken = nextToken
            }
        }

        return tokens
    }

    /// Decode with cache-external KV cache (Parakeet pattern).
    private func decodeCacheExternal(
        encoderHidden: MLMultiArray,
        language: CohereAsrConfig.Language?,
        maxNewTokens: Int,
        models: CohereAsrModels
    ) async throws -> [Int] {
        // Initialize 16 separate KV cache arrays (8 layers × 2)
        // Each cache shape: (1, 8, 108, 128) = [batch, heads, seq_len, head_dim]
        let cacheShape =
            [1, CohereAsrConfig.numDecoderHeads, CohereAsrConfig.maxSeqLen, CohereAsrConfig.headDim] as [NSNumber]

        var kCaches: [MLMultiArray] = []
        var vCaches: [MLMultiArray] = []

        for _ in 0..<CohereAsrConfig.numDecoderLayers {
            guard
                let kCache = try? MLMultiArray(shape: cacheShape, dataType: .float32),
                let vCache = try? MLMultiArray(shape: cacheShape, dataType: .float32)
            else {
                throw CohereAsrError.decodingFailed("Failed to create KV cache arrays")
            }

            // Initialize with zeros
            for i in 0..<kCache.count {
                kCache[i] = 0
                vCache[i] = 0
            }

            kCaches.append(kCache)
            vCaches.append(vCache)
        }

        // Cross-attention mask: (1, 1, 1, encoder_seq_len) - all ones
        // Get encoder sequence length from hidden states shape (1, seq_len, hidden_dim)
        let encoderSeqLen = encoderHidden.shape[1].intValue
        guard
            let crossAttentionMask = try? MLMultiArray(
                shape: [1, 1, 1, NSNumber(value: encoderSeqLen)], dataType: .float32)
        else {
            throw CohereAsrError.decodingFailed("Failed to create cross-attention mask")
        }
        for i in 0..<encoderSeqLen {
            crossAttentionMask[[0, 0, 0, i] as [NSNumber]] = 1.0
        }

        // Build prompt sequence for the language
        let prompt = language?.promptSequence ?? [CohereAsrConfig.SpecialTokens.startToken]
        var tokens = [Int]()
        var currentToken = prompt[0]

        // Bound by KV cache size to prevent out-of-bounds access
        let effectiveMaxTokens = min(maxNewTokens + prompt.count, CohereAsrConfig.maxSeqLen)

        for totalStep in 0..<effectiveMaxTokens {
            // Use prompt tokens first, then generated tokens
            if totalStep < prompt.count {
                currentToken = prompt[totalStep]
            }
            // Create decoder inputs
            guard let inputId = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create input_id array")
            }
            inputId[0] = NSNumber(value: currentToken)

            guard let positionId = try? MLMultiArray(shape: [1, 1], dataType: .int32) else {
                throw CohereAsrError.decodingFailed("Failed to create position_id array")
            }
            positionId[0] = NSNumber(value: totalStep)

            // Attention mask (causal): (1, 1, 1, totalStep+1) - all zeros for cache-external
            guard
                let attentionMask = try? MLMultiArray(
                    shape: [1, 1, 1, NSNumber(value: totalStep + 1)], dataType: .float32)
            else {
                throw CohereAsrError.decodingFailed("Failed to create attention mask")
            }
            for i in 0..<(totalStep + 1) {
                attentionMask[[0, 0, 0, i] as [NSNumber]] = 0
            }

            // Build decoder input dictionary
            var inputDict: [String: MLFeatureValue] = [
                "input_id": MLFeatureValue(multiArray: inputId),
                "position_id": MLFeatureValue(multiArray: positionId),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
            ]

            // Add all cache inputs
            for i in 0..<CohereAsrConfig.numDecoderLayers {
                inputDict["k_cache_\(i)"] = MLFeatureValue(multiArray: kCaches[i])
                inputDict["v_cache_\(i)"] = MLFeatureValue(multiArray: vCaches[i])
            }

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: inputDict)
            let decoderOutput = try await models.decoder.prediction(from: decoderInput)

            // Get logits and sample next token
            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.decodingFailed("Failed to get logits")
            }

            let nextToken = argmax(logits)

            // Only collect tokens after the prompt
            if totalStep >= prompt.count - 1 {
                // Check for EOS before adding to tokens
                if nextToken == CohereAsrConfig.SpecialTokens.eosToken {
                    break
                }

                tokens.append(nextToken)
            }

            // Update caches from outputs
            for i in 0..<CohereAsrConfig.numDecoderLayers {
                guard
                    let kCacheOut = decoderOutput.featureValue(for: "k_cache_\(i)_out")?.multiArrayValue,
                    let vCacheOut = decoderOutput.featureValue(for: "v_cache_\(i)_out")?.multiArrayValue
                else {
                    throw CohereAsrError.decodingFailed("Failed to get updated cache for layer \(i)")
                }
                kCaches[i] = kCacheOut
                vCaches[i] = vCacheOut
            }

            // Use the predicted next token (unless we're still feeding the prompt)
            if totalStep < prompt.count - 1 {
                // Still feeding prompt, nextToken is ignored
                currentToken = prompt[totalStep + 1]
            } else {
                currentToken = nextToken
            }
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
