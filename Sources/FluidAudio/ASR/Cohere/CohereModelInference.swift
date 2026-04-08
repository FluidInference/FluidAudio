import Accelerate
@preconcurrency import CoreML
import Foundation

/// Model inference operations for Cohere Transcribe decoding.
///
/// Encapsulates execution of the cache-external decoder following Parakeet's pattern.
/// The decoder receives KV cache as inputs and returns updated cache as outputs.
internal struct CohereModelInference: Sendable {
    private let predictionOptions: MLPredictionOptions
    private let numHeads: Int
    private let headDim: Int
    private let maxSeqLen: Int

    init(numHeads: Int = 8, headDim: Int = 128, maxSeqLen: Int = 108) {
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
        self.numHeads = numHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen
    }

    /// Execute decoder with cache-external pattern (Parakeet approach).
    ///
    /// - Parameters:
    ///   - tokenId: Current token ID to decode
    ///   - positionId: Current position index
    ///   - encoderHiddenStates: Encoder output [1, enc_len, 1024]
    ///   - crossAttentionMask: Encoder attention mask [1, 1, 1, enc_len]
    ///   - state: Current decoder state (contains KV caches)
    ///   - model: Decoder MLModel
    ///   - inputId: Pre-allocated array for token input
    ///   - posId: Pre-allocated array for position input
    ///   - attentionMask: Pre-allocated attention mask (will be resized each step)
    ///
    /// - Returns: Tuple of (logits, updated state)
    func runDecoder(
        tokenId: Int,
        positionId: Int,
        encoderHiddenStates: MLMultiArray,
        crossAttentionMask: MLMultiArray,
        state: CohereDecoderState,
        model: MLModel,
        inputId: MLMultiArray,
        posId: MLMultiArray,
        attentionMask: MLMultiArray
    ) throws -> (logits: MLMultiArray, newState: CohereDecoderState) {

        // Set input token and position
        inputId[0] = NSNumber(value: tokenId)
        posId[0] = NSNumber(value: positionId)

        // Update attention mask size based on current sequence length
        // attention_mask grows: [1,1,1,1] -> [1,1,1,2] -> [1,1,1,3] ...
        let currentSeqLen = state.pastKvLen + 1
        let updatedAttentionMask = try createAttentionMask(seqLen: currentSeqLen)

        // Build input dictionary
        var inputDict: [String: MLFeatureValue] = [
            "input_id": MLFeatureValue(multiArray: inputId),
            "position_id": MLFeatureValue(multiArray: posId),
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
            "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
            "attention_mask": MLFeatureValue(multiArray: updatedAttentionMask),
        ]

        // Add all K/V caches as inputs
        for i in 0..<state.kCaches.count {
            inputDict["k_cache_\(i)"] = MLFeatureValue(multiArray: state.kCaches[i])
            inputDict["v_cache_\(i)"] = MLFeatureValue(multiArray: state.vCaches[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)

        // Run decoder
        // Note: We could use outputBackings to avoid allocations, but
        // CoreML needs to create new cache tensors anyway for the output
        let output = try model.prediction(from: input, options: predictionOptions)

        // Extract logits
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing logits from decoder output")
        }

        // Update state with new caches
        var newState = state
        newState.updateFromOutput(output)
        newState.lastToken = tokenId

        return (logits, newState)
    }

    /// Create causal attention mask for decoder self-attention.
    ///
    /// The mask is [1, 1, 1, seqLen] with:
    /// - 0.0 for positions 0..<seqLen (valid, can attend)
    /// - -inf for positions seqLen..<maxSeqLen (invalid, masked out)
    ///
    /// - Parameter seqLen: Current sequence length (1, 2, 3, ...)
    /// - Returns: Attention mask array [1, 1, 1, maxSeqLen]
    private func createAttentionMask(seqLen: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: maxSeqLen)],
            dataType: .float32
        )

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: maxSeqLen)

        // Set valid positions to 0.0, invalid to -inf
        for i in 0..<maxSeqLen {
            ptr[i] = (i < seqLen) ? 0.0 : -Float.infinity
        }

        return mask
    }

    /// Extract logits and sample next token using greedy decoding.
    ///
    /// - Parameter logits: Logits array [1, vocab_size] or [vocab_size]
    /// - Returns: Token ID with highest probability
    func greedySample(logits: MLMultiArray) -> Int {
        let vocabSize = logits.count
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)

        var maxVal: Float = -Float.infinity
        var maxIdx: Int = 0

        for i in 0..<vocabSize {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = i
            }
        }

        return maxIdx
    }
}
