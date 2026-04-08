import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereStatelessManager")

/// Cohere Transcribe ASR with stateless decoder (O(n²) but simple).
///
/// This decoder reprocesses ALL tokens at each step (no cache).
/// - Pros: Simple, works on macOS 14, no cache bugs, can compile to .mlmodelc
/// - Cons: O(n²) complexity
/// - Verdict: For 108 token limit, this is totally acceptable!
@available(macOS 14, iOS 17, *)
public actor CohereStatelessManager {
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var vocabulary: [String] = []
    private let melExtractor: CohereMelSpectrogram

    // Constants
    private let maxSeqLen = 108
    private let startTokenId = 4
    private let eosTokenId = 3  // <|endoftext|> - verified from model.generation_config.eos_token_id

    public init() {
        self.melExtractor = CohereMelSpectrogram()
    }

    /// Load models from directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Load encoder
        let encoderURL = directory.appendingPathComponent("encoder.mlpackage")
        encoder = try await MLModel.load(contentsOf: encoderURL, configuration: config)

        // Load stateless decoder
        let decoderURL = directory.appendingPathComponent("decoder_stateless.mlpackage")
        decoder = try await MLModel.load(contentsOf: decoderURL, configuration: config)

        // Load vocabulary
        let vocabURL = directory.appendingPathComponent("vocabulary.txt")
        let vocabText = try String(contentsOf: vocabURL, encoding: .utf8)
        vocabulary = vocabText.components(separatedBy: .newlines)

        logger.info("Cohere Stateless models loaded")
    }

    /// Transcribe audio (stateless decoder).
    public func transcribe(audioSamples: [Float], maxNewTokens: Int = 200) async throws -> String {
        guard let encoder = encoder, let decoder = decoder else {
            throw CohereAsrError.generationFailed("Models not loaded")
        }

        let start = CFAbsoluteTimeGetCurrent()

        // 1. Extract mel spectrogram
        let mel = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw CohereAsrError.invalidInput("Audio too short")
        }

        let nFrames = mel[0].count
        let paddedMel = padMelSpectrogram(mel, targetFrames: 3500)

        // 2. Encode audio
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let encoderHidden = try await encodeAudio(
            paddedMel: paddedMel,
            featureLength: nFrames,
            encoder: encoder
        )
        let encodeTime = CFAbsoluteTimeGetCurrent() - encodeStart
        logger.debug("Encoder: \(String(format: "%.3f", encodeTime))s")

        // 3. Decode (stateless - much simpler!)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let tokens = try await decodeStateless(
            encoderHidden: encoderHidden,
            maxNewTokens: min(maxNewTokens, maxSeqLen),
            decoder: decoder
        )
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        logger.debug("Decoder: \(String(format: "%.3f", decodeTime))s (\(tokens.count) tokens)")

        let totalTime = CFAbsoluteTimeGetCurrent() - start
        logger.info("Transcribed in \(String(format: "%.3f", totalTime))s")

        // 4. Detokenize
        let text = detokenize(tokens)
        return text
    }

    // MARK: - Encoding

    private func padMelSpectrogram(_ mel: [[Float]], targetFrames: Int) -> [[Float]] {
        let nMels = mel.count
        let nFrames = mel[0].count
        guard nFrames < targetFrames else { return mel }

        var padded = [[Float]](repeating: [Float](repeating: 0, count: targetFrames), count: nMels)
        for m in 0..<nMels {
            for f in 0..<nFrames {
                padded[m][f] = mel[m][f]
            }
        }
        return padded
    }

    private func encodeAudio(
        paddedMel: [[Float]],
        featureLength: Int,
        encoder: MLModel
    ) async throws -> MLMultiArray {
        let inputShape = [1, CohereAsrConfig.numMelBins, 3500] as [NSNumber]
        let inputFeatures = try MLMultiArray(shape: inputShape, dataType: .float32)

        for m in 0..<CohereAsrConfig.numMelBins {
            for f in 0..<3500 {
                inputFeatures[[0, m, f] as [NSNumber]] = NSNumber(value: paddedMel[m][f])
            }
        }

        let featureLengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        featureLengthArray[0] = NSNumber(value: featureLength)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: inputFeatures),
            "feature_length": MLFeatureValue(multiArray: featureLengthArray),
        ])

        let encoderOutput = try await encoder.prediction(from: encoderInput)

        guard let hiddenStates = encoderOutput.featureValue(for: "encoder_outputs")?.multiArrayValue else {
            throw CohereAsrError.encodingFailed("Missing encoder output")
        }

        return hiddenStates
    }

    // MARK: - Stateless Decoding

    /// Decode with stateless approach (reprocess all tokens each step).
    ///
    /// This is O(n²) but for 108 tokens it's totally fine on ANE.
    /// Much simpler than cache management!
    private func decodeStateless(
        encoderHidden: MLMultiArray,
        maxNewTokens: Int,
        decoder: MLModel
    ) async throws -> [Int] {
        // Create cross-attention mask (all ones - attend to all encoder positions)
        let encoderSeqLen = encoderHidden.shape[1].intValue
        let crossMask = try createCrossAttentionMask(encoderSeqLen: encoderSeqLen)

        var tokenIds: [Int] = [startTokenId]  // Start with start token

        // Autoregressive generation
        for step in 0..<maxNewTokens {
            // Create input_ids with ALL tokens so far [1, seq_len]
            let currentSeqLen = tokenIds.count
            let inputIds = try MLMultiArray(shape: [1, NSNumber(value: currentSeqLen)], dataType: .int32)

            for (i, tokenId) in tokenIds.enumerated() {
                inputIds[[0, i] as [NSNumber]] = NSNumber(value: tokenId)
            }

            // Run decoder (processes ALL tokens, returns logits for ALL positions)
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                "cross_attention_mask": MLFeatureValue(multiArray: crossMask),
            ])

            let decoderOutput = try await decoder.prediction(from: decoderInput)

            guard let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw CohereAsrError.decodingFailed("Missing logits")
            }

            // Extract logits for the LAST token position
            // logits shape: [1, seq_len, vocab_size]
            // We want logits[0, -1, :]
            let nextToken = extractLastTokenLogits(logits, seqLen: currentSeqLen)

            // Check for EOS
            if nextToken == eosTokenId {
                break
            }

            tokenIds.append(nextToken)

            // Safety check
            if tokenIds.count >= self.maxSeqLen {
                logger.warning("Hit max sequence length \(self.maxSeqLen)")
                break
            }
        }

        // Remove start token
        return Array(tokenIds.dropFirst())
    }

    /// Extract logits for the last token and sample greedily.
    private func extractLastTokenLogits(_ logits: MLMultiArray, seqLen: Int) -> Int {
        // logits shape: [1, seq_len, vocab_size]
        let vocabSize = logits.shape[2].intValue
        let lastTokenOffset = (seqLen - 1) * vocabSize

        var maxVal: Float = -Float.infinity
        var maxIdx = 0

        for v in 0..<vocabSize {
            let idx = [0, seqLen - 1, v] as [NSNumber]
            let val = logits[idx].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = v
            }
        }

        return maxIdx
    }

    private func createCrossAttentionMask(encoderSeqLen: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: encoderSeqLen)], dataType: .float32)
        for i in 0..<encoderSeqLen {
            mask[[0, 0, 0, i] as [NSNumber]] = 1.0
        }
        return mask
    }

    private func detokenize(_ tokenIds: [Int]) -> String {
        tokenIds
            .compactMap { id -> String? in
                guard id > 4 && id != eosTokenId && id < vocabulary.count else { return nil }
                let token = vocabulary[id]
                guard !token.hasPrefix("<|") else { return nil }
                return token
            }
            .joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
