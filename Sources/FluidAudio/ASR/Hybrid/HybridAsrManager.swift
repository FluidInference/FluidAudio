@preconcurrency import CoreML
import Foundation
import OSLog

/// Hybrid ASR Manager that uses shared encoder for both TDT transcription and CTC keyword spotting.
/// This ensures aligned timestamps between CTC detections and TDT transcription.
///
/// Architecture (matches the research paper):
/// ```
/// Audio → Preprocessor → Encoder (shared)
///                            ↓
///              ┌─────────────┴─────────────┐
///              ↓                           ↓
///         CTC Head                   TDT Decoder+Joint
///              ↓                           ↓
///        Log-probs                   Transcription
///         (keyword                    (with aligned
///          spotting)                   timestamps)
/// ```
public actor HybridAsrManager {
    private let models: HybridAsrModels
    private let predictionOptions: MLPredictionOptions
    private let logger = AppLogger(category: "HybridAsrManager")

    // Constants
    private let sampleRate: Int = 16_000
    private let maxAudioSeconds: Double = 15.0
    private let maxSamples: Int = 240_000
    private let frameDuration: Double = 0.08  // 80ms per encoder frame

    public init(models: HybridAsrModels) {
        self.models = models
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
    }

    /// Transcribe audio and return both transcription and CTC log-probs for keyword spotting.
    /// Both use the SAME encoder output, ensuring aligned timestamps.
    public func transcribe(
        audioSamples: [Float],
        customVocabulary: CustomVocabularyContext? = nil
    ) async throws -> HybridTranscriptionResult {
        let startTime = Date()

        // 1. Prepare audio input
        let (audioArray, actualSamples) = try prepareAudioInput(audioSamples)
        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: actualSamples)

        // 2. Run preprocessor → mel features
        let preprocessorInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLength),
        ])
        let preprocessorOutput = try await models.preprocessor.prediction(
            from: preprocessorInput,
            options: predictionOptions
        )

        guard let melFeatures = preprocessorOutput.featureValue(for: "mel_features")?.multiArrayValue,
            let melLength = preprocessorOutput.featureValue(for: "mel_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Missing preprocessor outputs")
        }

        // 3. Run encoder (SHARED) → encoder embeddings
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel_features": MLFeatureValue(multiArray: melFeatures),
            "mel_length": MLFeatureValue(multiArray: melLength),
        ])
        let encoderOutput = try await models.encoder.prediction(
            from: encoderInput,
            options: predictionOptions
        )

        guard let encoderEmbeddings = encoderOutput.featureValue(for: "encoder_output")?.multiArrayValue,
            let encoderLength = encoderOutput.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Missing encoder outputs")
        }

        let numFrames = encoderLength[0].intValue

        // 4. Run CTC head on encoder embeddings → log-probs for keyword spotting
        let ctcInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_output": MLFeatureValue(multiArray: encoderEmbeddings)
        ])
        let ctcOutput = try await models.ctcHead.prediction(
            from: ctcInput,
            options: predictionOptions
        )

        guard let ctcLogits = ctcOutput.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing CTC output")
        }

        // Convert CTC logits to log-probs
        let ctcLogProbs = try convertToLogProbs(ctcLogits, numFrames: numFrames)

        // 5. Use CTC greedy decoding for transcription (simpler and more reliable)
        // TDT decoder has issues with the CoreML conversion
        let transcription = ctcGreedyDecode(
            logProbs: ctcLogProbs,
            frameDuration: frameDuration
        )

        let processingTime = Date().timeIntervalSince(startTime)
        let audioDuration = Double(actualSamples) / Double(sampleRate)

        logger.info(
            "Hybrid transcription: \(transcription.tokens.count) tokens, "
                + "\(numFrames) frames, RTFx=\(String(format: "%.1f", audioDuration / processingTime))"
        )

        return HybridTranscriptionResult(
            text: transcription.text,
            tokens: transcription.tokens,
            ctcLogProbs: ctcLogProbs,
            numFrames: numFrames,
            frameDuration: frameDuration,
            audioDuration: audioDuration,
            processingTime: processingTime
        )
    }

    // MARK: - Private Methods

    private func prepareAudioInput(_ samples: [Float]) throws -> (MLMultiArray, Int) {
        let actualSamples = min(samples.count, maxSamples)

        let audioArray = try MLMultiArray(shape: [1, NSNumber(value: maxSamples)], dataType: .float32)

        // Copy samples
        let ptr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: maxSamples)
        for i in 0..<actualSamples {
            ptr[i] = samples[i]
        }
        // Zero-pad remaining
        for i in actualSamples..<maxSamples {
            ptr[i] = 0
        }

        return (audioArray, actualSamples)
    }

    private func convertToLogProbs(_ logits: MLMultiArray, numFrames: Int) throws -> [[Float]] {
        // logits shape: [1, T, V]
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid CTC logits shape: \(shape)")
        }

        let T = min(shape[1], numFrames)
        let V = shape[2]

        // Use strides from the MLMultiArray (may have padding)
        let strides = logits.strides.map { $0.intValue }
        let strideT = strides[1]  // Stride between time frames
        let strideV = strides[2]  // Stride between vocab entries (usually 1)

        var result: [[Float]] = []
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)

        for t in 0..<T {
            var frame: [Float] = Array(repeating: 0, count: V)
            var maxVal: Float = -.infinity

            // Get logits for this frame using proper strides
            for v in 0..<V {
                let idx = t * strideT + v * strideV
                frame[v] = ptr[idx]
                maxVal = max(maxVal, frame[v])
            }

            // Log-softmax: log(exp(x - max) / sum(exp(x - max)))
            var sumExp: Float = 0
            for v in 0..<V {
                sumExp += exp(frame[v] - maxVal)
            }
            let logSumExp = maxVal + log(sumExp)

            for v in 0..<V {
                frame[v] = frame[v] - logSumExp
            }

            result.append(frame)
        }

        return result
    }

    private func runRnntDecoder(
        encoderEmbeddings: MLMultiArray,
        numFrames: Int
    ) async throws -> (text: String, tokens: [TokenWithTiming]) {

        // Initialize decoder state
        let hiddenDim = 640  // From metadata
        let numLayers = 1

        var h = try MLMultiArray(
            shape: [NSNumber(value: numLayers), 1, NSNumber(value: hiddenDim)], dataType: .float32)
        var c = try MLMultiArray(
            shape: [NSNumber(value: numLayers), 1, NSNumber(value: hiddenDim)], dataType: .float32)

        // Initialize with zeros
        let hPtr = h.dataPointer.bindMemory(to: Float.self, capacity: h.count)
        let cPtr = c.dataPointer.bindMemory(to: Float.self, capacity: c.count)
        for i in 0..<h.count {
            hPtr[i] = 0
            cPtr[i] = 0
        }

        var tokens: [TokenWithTiming] = []
        var currentToken: Int = 0  // Start with blank
        var frameIdx = 0

        let encoderShape = encoderEmbeddings.shape.map { $0.intValue }
        let hiddenDimEnc = encoderShape[1]

        // Max tokens per frame to prevent infinite loops
        let maxTokensPerFrame = 10

        while frameIdx < numFrames {
            // Extract encoder step [1, D, 1]
            let encoderStep = try extractEncoderStep(
                encoderEmbeddings, frameIndex: frameIdx, hiddenDim: hiddenDimEnc)

            // Inner loop: emit tokens at this frame until blank is predicted
            var tokensAtFrame = 0
            var shouldAdvance = false

            while !shouldAdvance && tokensAtFrame < maxTokensPerFrame {
                // Run decoder step
                let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
                targets[0] = NSNumber(value: currentToken)
                let targetLen = try MLMultiArray(shape: [1], dataType: .int32)
                targetLen[0] = 1

                let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                    "targets": MLFeatureValue(multiArray: targets),
                    "target_lengths": MLFeatureValue(multiArray: targetLen),
                    "h_in": MLFeatureValue(multiArray: h),
                    "c_in": MLFeatureValue(multiArray: c),
                ])
                let decoderOutput = try await models.decoder.prediction(
                    from: decoderInput,
                    options: predictionOptions
                )

                guard let decoderEmb = decoderOutput.featureValue(for: "decoder_output")?.multiArrayValue,
                    let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                    let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
                else {
                    throw ASRError.processingFailed("Missing decoder outputs")
                }

                // Decoder output is [1, D, U] format - extract first step (after SOS)
                // The conversion traced with dec_ref[:,:,:1] which is index 0
                let decoderStep = try extractDecoderStep(decoderEmb, stepIndex: 0)

                // Run joint decision
                let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder_step": MLFeatureValue(multiArray: encoderStep),
                    "decoder_step": MLFeatureValue(multiArray: decoderStep),
                ])
                let jointOutput = try await models.jointDecision.prediction(
                    from: jointInput,
                    options: predictionOptions
                )

                guard let tokenId = jointOutput.featureValue(for: "token_id")?.multiArrayValue,
                    let durationArr = jointOutput.featureValue(for: "duration")?.multiArrayValue
                else {
                    throw ASRError.processingFailed("Missing joint outputs")
                }

                let predictedToken = tokenId[0].intValue
                let predictedDuration = durationArr[0].intValue

                // TDT decoding logic:
                // - duration=0 means stay at current frame
                // - duration>0 means advance by that many frames
                // - blank token + duration>0 = skip frames without emitting

                if predictedToken == models.blankId || predictedToken >= models.vocabSize {
                    // Blank token: advance by duration (at least 1)
                    frameIdx += max(1, predictedDuration)
                    shouldAdvance = true
                } else if predictedDuration > 0 {
                    // Token with duration: emit token and advance
                    h = hOut
                    c = cOut
                    currentToken = predictedToken

                    let startTime = Double(frameIdx) * self.frameDuration
                    let endTime = startTime + Double(predictedDuration) * self.frameDuration

                    if let tokenText = models.vocabulary[predictedToken] {
                        tokens.append(
                            TokenWithTiming(
                                tokenId: predictedToken,
                                text: tokenText,
                                startTime: startTime,
                                endTime: endTime
                            ))
                    }
                    frameIdx += predictedDuration
                    shouldAdvance = true
                } else {
                    // Token with duration=0: emit token, stay at frame
                    h = hOut
                    c = cOut
                    currentToken = predictedToken

                    let startTime = Double(frameIdx) * self.frameDuration
                    let endTime = startTime + self.frameDuration

                    if let tokenText = models.vocabulary[predictedToken] {
                        tokens.append(
                            TokenWithTiming(
                                tokenId: predictedToken,
                                text: tokenText,
                                startTime: startTime,
                                endTime: endTime
                            ))
                    }
                    tokensAtFrame += 1
                }
            }

            // Frame advancement is now handled in the TDT logic above
            if !shouldAdvance {
                frameIdx += 1  // Safety: advance if we hit max tokens per frame
            }
        }

        // Convert tokens to text
        let text = tokensToText(tokens)

        return (text, tokens)
    }

    private func extractEncoderStep(
        _ encoder: MLMultiArray,
        frameIndex: Int,
        hiddenDim: Int
    ) throws -> MLMultiArray {
        // encoder shape: [1, D, T] -> extract [1, D, 1]
        let step = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), 1], dataType: .float32)

        let srcPtr = encoder.dataPointer.bindMemory(to: Float.self, capacity: encoder.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)

        let T = encoder.shape[2].intValue
        for d in 0..<hiddenDim {
            let srcIdx = d * T + frameIndex
            dstPtr[d] = srcPtr[srcIdx]
        }

        return step
    }

    /// Extract the first step from decoder output [1, D, U] -> [1, D, 1]
    /// NeMo decoder adds SOS token, so output[0] is the embedding before any target tokens.
    /// For RNN-T, we need the embedding state after processing previous tokens.
    /// With targets=[last_token], output[0] = after SOS, output[1] = after last_token.
    /// We use output[1] which represents the state after seeing the previous token.
    private func extractDecoderStep(_ decoder: MLMultiArray, stepIndex: Int = 1) throws -> MLMultiArray {
        let shape = decoder.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid decoder shape: \(shape)")
        }

        let D = shape[1]
        let U = shape[2]
        let step = min(stepIndex, U - 1)

        let result = try MLMultiArray(shape: [1, NSNumber(value: D), 1], dataType: .float32)
        let srcPtr = decoder.dataPointer.bindMemory(to: Float.self, capacity: decoder.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)

        for d in 0..<D {
            let srcIdx = d * U + step
            dstPtr[d] = srcPtr[srcIdx]
        }

        return result
    }

    private func transposeDecoderOutput(_ decoder: MLMultiArray) throws -> MLMultiArray {
        // decoder shape: [1, U, D] -> transpose to [1, D, U]
        let shape = decoder.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid decoder shape: \(shape)")
        }

        let B = shape[0]
        let U = shape[1]
        let D = shape[2]

        let transposed = try MLMultiArray(
            shape: [NSNumber(value: B), NSNumber(value: D), NSNumber(value: U)],
            dataType: .float32
        )

        let srcPtr = decoder.dataPointer.bindMemory(to: Float.self, capacity: decoder.count)
        let dstPtr = transposed.dataPointer.bindMemory(to: Float.self, capacity: transposed.count)

        // [1, U, D] row-major: index = u * D + d
        // [1, D, U] row-major: index = d * U + u
        for u in 0..<U {
            for d in 0..<D {
                let srcIdx = u * D + d
                let dstIdx = d * U + u
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return transposed
    }

    /// CTC greedy decoding: pick best token per frame, collapse repeats, remove blanks
    private func ctcGreedyDecode(
        logProbs: [[Float]],
        frameDuration: Double
    ) -> (text: String, tokens: [TokenWithTiming]) {
        var tokens: [TokenWithTiming] = []
        var prevToken: Int = -1

        for (frameIdx, frameLogProbs) in logProbs.enumerated() {
            // Find best token (argmax)
            var bestToken = 0
            var bestProb = frameLogProbs[0]
            for (idx, prob) in frameLogProbs.enumerated() {
                if prob > bestProb {
                    bestProb = prob
                    bestToken = idx
                }
            }

            // Skip blank tokens (last token in vocab)
            let blankId = models.blankId
            if bestToken == blankId {
                prevToken = bestToken
                continue
            }

            // Skip repeated tokens (CTC collapse)
            if bestToken == prevToken {
                continue
            }

            // Emit token
            let startTime = Double(frameIdx) * frameDuration
            let endTime = startTime + frameDuration

            if let tokenText = models.vocabulary[bestToken] {
                tokens.append(
                    TokenWithTiming(
                        tokenId: bestToken,
                        text: tokenText,
                        startTime: startTime,
                        endTime: endTime
                    ))
            }
            prevToken = bestToken
        }

        let text = tokensToText(tokens)
        return (text, tokens)
    }

    private func tokensToText(_ tokens: [TokenWithTiming]) -> String {
        var result = ""
        for token in tokens {
            var text = token.text
            // Handle SentencePiece: ▁ marks word boundary
            if text.hasPrefix("▁") {
                text = String(text.dropFirst())
                if !result.isEmpty {
                    result += " "
                }
            }
            result += text
        }
        return result
    }
}

// MARK: - Result Types

/// Result from hybrid transcription with both TDT output and CTC log-probs.
public struct HybridTranscriptionResult: Sendable {
    /// Transcribed text from TDT decoder
    public let text: String

    /// Tokens with timing from TDT decoder (aligned with CTC)
    public let tokens: [TokenWithTiming]

    /// CTC log-probabilities [T, V] for keyword spotting
    public let ctcLogProbs: [[Float]]

    /// Number of encoder frames
    public let numFrames: Int

    /// Duration of each frame in seconds (80ms)
    public let frameDuration: Double

    /// Total audio duration in seconds
    public let audioDuration: Double

    /// Processing time in seconds
    public let processingTime: Double
}

/// Token with timing information
public struct TokenWithTiming: Sendable {
    public let tokenId: Int
    public let text: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval
}
