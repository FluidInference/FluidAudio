import AVFoundation
import Accelerate
import CoreML
import Foundation
import OSLog

public final class CanaryManager {

    private let logger = AppLogger(category: "CanaryManager")
    
    private var models: CanaryModels?
    private var preprocessorModel: MLModel?
    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    
    // Projection weights
    private var projectionWeights: [Float] = []
    private var projectionBias: [Float] = []
    
    // Audio buffering
    private var buffer: [Float] = []
    
    // Constants (16kHz)
    private let sampleRate: Double = 16000.0
    // 14s window: 10s left + 2s chunk + 2s right
    private let chunkSamples: Int = 32000      // 2.0s
    private let leftContextSamples: Int = 160000 // 10.0s
    private let rightContextSamples: Int = 32000 // 2.0s
    
    // Special Tokens (Canary-1B v2)
    private let PAD_ID = 0
    private let BOS_ID = 16053
    private let EOS_ID = 3 // <|endoftext|>
    private let START_CONTEXT_ID = 7
    private let START_TRANSCRIPT_ID = 4
    private let EMO_UNDEFINED_ID = 16
    private let PNC_ID = 5
    private let NO_ITN_ID = 9
    private let NO_TIMESTAMP_ID = 11
    private let NO_DIARIZE_ID = 13
    private let LANG_EN_ID = 64 // <|en|>
    
    public init() {}
    
    public func initialize(models: CanaryModels) {
        self.models = models
        self.preprocessorModel = models.preprocessor
        self.encoderModel = models.encoder
        self.decoderModel = models.decoder
        self.projectionWeights = models.projectionWeights
        self.projectionBias = models.projectionBias
        
        // Initialize buffer with padding for left context
        self.buffer = Array(repeating: 0.0, count: leftContextSamples)
        
        logger.info("CanaryManager initialized with models and projection weights")
        logger.info("Projection weights count: \(self.projectionWeights.count)")
        logger.info("Projection bias count: \(self.projectionBias.count)")
    }
    
    private var lastWindowText: String = ""
    
    /// Reset the manager state (clear buffer).
    public func reset() {
        self.buffer = Array(repeating: 0.0, count: leftContextSamples)
        self.lastWindowText = ""
    }
    
    /// Process a streaming chunk of audio.
    /// - Parameter chunk: Audio samples (Float32, 16kHz).
    /// - Returns: Transcribed text for the processed chunk.
    public func processStreamingChunk(_ chunk: [Float]) async throws -> String {
        buffer.append(contentsOf: chunk)
        
        let windowSize = leftContextSamples + chunkSamples + rightContextSamples
        
        // Check if we have enough data for a full window
        if buffer.count >= windowSize {
            // Extract window (14s)
            let window = Array(buffer[0..<windowSize])
            
            // Process window
            let currentWindowText = try await transcribeWindow(window)
            
            // Robust overlap detection
            let lastNorm = NormalizedText(lastWindowText)
            let currNorm = NormalizedText(currentWindowText)
            
            var overlapLength = 0
            let maxOverlap = min(lastNorm.text.count, currNorm.text.count)
            
            for k in stride(from: maxOverlap, through: 1, by: -1) {
                if lastNorm.text.hasSuffix(String(currNorm.text.prefix(k))) {
                    overlapLength = k
                    break
                }
            }
            
            var newText = ""
            if overlapLength > 0 {
                // Find the end index in the original string
                // The last matched character in normalized text is at index overlapLength - 1
                if overlapLength <= currNorm.originalIndices.count {
                     let lastMatchedIndex = currNorm.originalIndices[overlapLength - 1]
                     // We want to start AFTER this character
                     let cutIndex = currentWindowText.index(after: lastMatchedIndex)
                     newText = String(currentWindowText[cutIndex...])
                } else {
                    // Should not happen if logic is correct
                    newText = ""
                }
            } else {
                newText = currentWindowText
            }
            
            // Clean up leading punctuation from new text if any
            newText = newText.trimmingCharacters(in: .punctuationCharacters.union(.whitespaces))
            if !newText.isEmpty {
                 newText = " " + newText
            }
            
            // Update lastWindowText
            lastWindowText = currentWindowText
            
            // Shift buffer
            buffer.removeFirst(chunkSamples)
            
            return newText
        }
        
        return ""
    }
    
    private struct NormalizedText {
        let text: String
        let originalIndices: [String.Index]
        
        init(_ original: String) {
            var text = ""
            var indices: [String.Index] = []
            
            for (i, char) in original.enumerated() {
                if char.isLetter || char.isNumber {
                    text.append(char.lowercased())
                    indices.append(original.index(original.startIndex, offsetBy: i))
                }
            }
            self.text = text
            self.originalIndices = indices
        }
    }
    
    private func transcribeWindow(_ audioSamples: [Float]) async throws -> String {
        guard let preprocessor = preprocessorModel,
              let encoder = encoderModel,
              let decoder = decoderModel else {
            throw ASRError.notInitialized
        }
        
        // 1. Preprocess (Audio -> Mel Spectrogram)
        // Input: "audio_signal" (1, L), "length" (1)
        let features = try preparePreprocessorInput(audioSamples)
        let preprocessorOutput = try await preprocessor.prediction(from: features)
        
        guard let audioFeatures = preprocessorOutput.featureValue(for: "audio_features")?.multiArrayValue,
              let audioFeaturesLength = preprocessorOutput.featureValue(for: "audio_features_length")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing preprocessor output")
        }
        
        // 2. Encoder
        // Input: "audio_features" (1, 80, T), "audio_lengths" (1)
        // Note: Conversion script used "audio_lengths" for encoder input name
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_features": MLFeatureValue(multiArray: audioFeatures),
            "audio_lengths": MLFeatureValue(multiArray: audioFeaturesLength)
        ])
        
        let encoderOutput = try await encoder.prediction(from: encoderInput)
        
        guard let encodedFeatures = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue else {
             throw ASRError.processingFailed("Missing encoder_output")
        }
        
        // Transpose encoded features from [1, D, T] to [1, T, D]
        // D=1024, T=176
        let transposedFeatures = try transposeEncodedFeatures(encodedFeatures)
        
        // 3. Decoder (Greedy Search)
        // Prompt: <|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        var currentTokens: [Int] = [
            START_CONTEXT_ID,
            START_TRANSCRIPT_ID,
            EMO_UNDEFINED_ID,
            LANG_EN_ID, // Source Lang
            LANG_EN_ID, // Target Lang
            PNC_ID,
            NO_ITN_ID,
            NO_TIMESTAMP_ID,
            NO_DIARIZE_ID
        ]
        
        // Encoder Mask (all ones for now, assuming full length valid)
        // Shape: [1, T_enc]
        let encSeqLen = transposedFeatures.shape[1].intValue // [1, T, D] -> T
        let encoderMask = try MLMultiArray(shape: [1, NSNumber(value: encSeqLen)], dataType: .int32)
        for i in 0..<encSeqLen {
            encoderMask[i] = 1
        }
        
        let maxTokens = 50 // Limit for 2s chunk
        let decoderSeqLen = 128 // Fixed input length for CoreML model
        
        for step in 0..<maxTokens {
            // Prepare decoder input
            // Pad input_ids and decoder_mask to decoderSeqLen
            var paddedTokens = currentTokens
            var paddedMask = Array(repeating: 1, count: currentTokens.count)
            
            if paddedTokens.count < decoderSeqLen {
                let padding = Array(repeating: PAD_ID, count: decoderSeqLen - paddedTokens.count)
                paddedTokens.append(contentsOf: padding)
                paddedMask.append(contentsOf: Array(repeating: 0, count: padding.count))
            } else if paddedTokens.count > decoderSeqLen {
                // Truncate or handle overflow?
                // For now, just truncate (though this shouldn't happen with short chunks)
                paddedTokens = Array(paddedTokens.prefix(decoderSeqLen))
                paddedMask = Array(paddedMask.prefix(decoderSeqLen))
            }
            
            let inputIdsArray = try createIntArray(paddedTokens)
            let decoderMaskArray = try createIntArray(paddedMask)
            
            let decoderInputDict: [String: MLFeatureValue] = [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "decoder_mask": MLFeatureValue(multiArray: decoderMaskArray),
                "encoder_embeddings": MLFeatureValue(multiArray: transposedFeatures),
                "encoder_mask": MLFeatureValue(multiArray: encoderMask)
            ]
            
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: decoderInputDict)
            let decoderResult = try await decoder.prediction(from: decoderInput)
            
            // Output: "hidden_states" [1, Seq, 1024]
            guard let hiddenStates = decoderResult.featureValue(for: "hidden_states")?.multiArrayValue else {
                break
            }
            
            // Get last hidden state
            let seqLen = hiddenStates.shape[1].intValue
            let hiddenDim = hiddenStates.shape[2].intValue // 1024
            
            // Safe extraction using subscripts
            var lastHiddenState = [Float](repeating: 0, count: hiddenDim)
            // We want the hidden state corresponding to the last VALID token
            // currentTokens.count is the number of valid tokens (including prompt)
            // So index is currentTokens.count - 1
            let seqIndex = currentTokens.count - 1
            
            for i in 0..<hiddenDim {
                // hiddenStates: [1, Seq, Dim]
                let val = hiddenStates[[0, NSNumber(value: seqIndex), NSNumber(value: i)] as [NSNumber]].floatValue
                lastHiddenState[i] = val
            }
            
            // Project to Logits
            // logits = hidden (1x1024) @ weights.T (1024x16384) + bias
            // Or: logits = weights (16384x1024) * hidden (1024x1) + bias
            let vocabSize = 16384
            var logits = [Float](repeating: 0, count: vocabSize)
            
            // cblas_sgemv: y = alpha * A * x + beta * y
            // A = weights (RowMajor, 16384 x 1024)
            // x = lastHiddenState
            // y = logits (initialized with bias)
            
            // Copy bias to logits first
            logits = projectionBias
            
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                Int32(vocabSize),
                Int32(hiddenDim),
                1.0,
                projectionWeights,
                Int32(hiddenDim),
                lastHiddenState,
                1,
                1.0, // beta=1.0 adds to existing bias
                &logits,
                1
            )
            
            // Argmax
            var maxLogit: Float = -Float.infinity
            var bestToken = 0
            
            for i in 0..<vocabSize {
                if logits[i] > maxLogit {
                    maxLogit = logits[i]
                    bestToken = i
                }
            }
            
            if bestToken == EOS_ID {
                break
            }
            
            currentTokens.append(bestToken)
        }
        
        // Decode tokens to text
        // Filter out prompt tokens and special tokens
        let promptLen = 9 // Number of initial tokens
        let newTokens = currentTokens.dropFirst(promptLen)
        
        // Simple decoding using vocabulary map
        // TODO: Use BPE decoder for proper subword merging
        let decoded = newTokens
            .filter { $0 != PAD_ID && $0 != EOS_ID }
            .compactMap { models?.vocabulary[$0] }
            .joined()
            .replacingOccurrences(of: " ", with: " ") // Handle SentencePiece underscore
        
        return decoded
    }
    
    // MARK: - Helpers
    
    private func createScalarArray(
        value: Int, shape: [NSNumber] = [1], dataType: MLMultiArrayDataType = .int32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dataType)
        array[0] = NSNumber(value: value)
        return array
    }
    
    private func createIntArray(_ values: [Int]) throws -> MLMultiArray {
        let shape = [1, NSNumber(value: values.count)]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        for (i, val) in values.enumerated() {
            array[i] = NSNumber(value: val)
        }
        return array
    }
    
    private func preparePreprocessorInput(_ audioSamples: [Float]) throws -> MLFeatureProvider {
        let audioLength = audioSamples.count
        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: audioLength)], dataType: .float32)
        
        audioSamples.withUnsafeBufferPointer { buffer in
            let destPtr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: audioLength)
            memcpy(destPtr, buffer.baseAddress!, audioLength * MemoryLayout<Float>.stride)
        }
        
        let lengthArray = try createScalarArray(value: audioLength)
        
        return try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "length": MLFeatureValue(multiArray: lengthArray), // Note: "length" not "audio_length" based on script
        ])
    }
    
    private func transposeEncodedFeatures(_ features: MLMultiArray) throws -> MLMultiArray {
        // features: [1, D, T]
        let D = features.shape[1].intValue // 1024
        let T = features.shape[2].intValue // 176
        
        let transposed = try MLMultiArray(shape: [1, NSNumber(value: T), NSNumber(value: D)], dataType: .float32)
        
        // Manual transpose using subscripts (slow but safe)
        for d in 0..<D {
            for t in 0..<T {
                let val = features[[0, NSNumber(value: d), NSNumber(value: t)] as [NSNumber]].floatValue
                transposed[[0, NSNumber(value: t), NSNumber(value: d)] as [NSNumber]] = NSNumber(value: val)
            }
        }
        
        return transposed
    }
}
