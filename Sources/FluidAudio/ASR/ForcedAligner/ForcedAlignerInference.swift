import Accelerate
@preconcurrency import CoreML
import CoreMLPredictionWrapper
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "ForcedAlignerInference")

/// Inference pipeline for the Qwen3-ForcedAligner CoreML models.
///
/// Replicates the Python `run_coreml_inference.py` pipeline:
/// 1. Compute mel spectrogram
/// 2. Run audio encoder (chunked 100-frame windows)
/// 3. Tokenize text with timestamp delimiters
/// 4. Run embedding model on input_ids
/// 5. Merge audio features into text embedding positions
/// 6. Pad to PREFILL_SEQ_LEN=1024
/// 7. Compute MRoPE cos/sin
/// 8. Run decoder+lm_head (single pass, NAR)
/// 9. Argmax → timestamps → word alignments
@available(macOS 14, iOS 17, *)
struct ForcedAlignerInference {

    private let models: ForcedAlignerModels
    private let melExtractor: ForcedAlignerMelSpectrogram
    private let mrope: ForcedAlignerMRoPE
    private let tokenizer: ForcedAlignerTokenizer

    init(models: ForcedAlignerModels, tokenizer: ForcedAlignerTokenizer) {
        self.models = models
        self.melExtractor = ForcedAlignerMelSpectrogram()
        self.mrope = ForcedAlignerMRoPE()
        self.tokenizer = tokenizer
    }

    /// Run the full forced alignment pipeline.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - text: Transcript text to align.
    /// - Returns: Per-word timestamp alignments.
    func align(audioSamples: [Float], text: String) throws -> [WordAlignment] {
        // Step 1: Compute mel spectrogram (Slaney scale, padded to 30s internally)
        let (mel, melFrames) = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw ForcedAlignerError.alignmentFailed("Audio too short to extract mel spectrogram")
        }
        logger.debug("Mel spectrogram: \(ForcedAlignerConfig.numMelBins) x \(melFrames)")

        // Step 2: Run audio encoder on 100-frame chunks
        let audioFeatures = try encodeAudio(mel: mel)
        let numAudioFrames = audioFeatures.count
        logger.debug("Audio features: \(numAudioFrames) x \(ForcedAlignerConfig.encoderOutputDim)")

        // Step 3: Tokenize text with timestamp delimiters
        let (words, inputIds) = tokenizer.tokenize(text: text, numAudioFrames: numAudioFrames)
        logger.debug("Input IDs: \(inputIds.count), Words: \(words.count)")

        // Step 4: Run embedding model on input_ids
        let textEmbeddings = try runEmbedding(inputIds: inputIds)

        // Step 5: Merge audio features into audio_pad positions
        var merged = textEmbeddings
        var audioIdx = 0
        for i in 0..<inputIds.count {
            if inputIds[i] == ForcedAlignerConfig.audioPadTokenId, audioIdx < audioFeatures.count {
                merged[i] = audioFeatures[audioIdx]
                audioIdx += 1
            }
        }

        let seqLen = merged.count
        logger.debug("Merged sequence: \(seqLen)")

        guard seqLen <= ForcedAlignerConfig.prefillSeqLen else {
            throw ForcedAlignerError.alignmentFailed(
                "Sequence length \(seqLen) exceeds max \(ForcedAlignerConfig.prefillSeqLen)"
            )
        }

        // Step 6: Pad to PREFILL_SEQ_LEN=1024
        let padded = padSequence(merged, to: ForcedAlignerConfig.prefillSeqLen)

        // Step 7: Compute MRoPE cos/sin (padded positions repeat last valid position)
        let (cos, sin) = mrope.compute(
            totalLen: ForcedAlignerConfig.prefillSeqLen,
            contentLen: seqLen
        )

        // Step 8: Run decoder+lm_head
        let logits = try runDecoderWithLmHead(
            hiddenStates: padded,
            cos: cos,
            sin: sin
        )

        // Step 9: Argmax on logits for actual sequence positions
        let outputIds = argmax(logits: logits, seqLen: seqLen)

        // Step 10: Extract timestamps at <timestamp> positions
        var timestampValues: [Int] = []
        for i in 0..<seqLen {
            if inputIds[i] == ForcedAlignerConfig.timestampTokenId {
                timestampValues.append(outputIds[i] * ForcedAlignerConfig.timestampSegmentTimeMs)
            }
        }

        // Step 11: Fix monotonicity via LIS
        let fixedTimestamps = fixTimestamps(timestampValues)

        // Step 12: Parse into word-level alignments
        var alignments: [WordAlignment] = []
        for i in 0..<words.count {
            let startIdx = i * 2
            let endIdx = i * 2 + 1
            guard endIdx < fixedTimestamps.count else { break }
            alignments.append(
                WordAlignment(
                    word: words[i],
                    startMs: Double(fixedTimestamps[startIdx]),
                    endMs: Double(fixedTimestamps[endIdx])
                ))
        }

        return alignments
    }

    // MARK: - Audio Encoding

    private func encodeAudio(mel: [[Float]]) throws -> [[Float]] {
        let numFrames = mel.first?.count ?? 0
        let windowSize = ForcedAlignerConfig.melWindowSize
        var allFeatures: [[Float]] = []

        // Collect all mel inputs and expected output counts
        var melInputs: [MLDictionaryFeatureProvider] = []
        var expectedOuts: [Int] = []

        var offset = 0
        while offset < numFrames {
            let end = min(offset + windowSize, numFrames)
            let actualChunkLen = end - offset

            var expectedOut = actualChunkLen
            for _ in 0..<3 {
                expectedOut = (expectedOut - 1) / 2 + 1
            }

            let melInput = try createMelInput(
                mel: mel, offset: offset, chunkLen: actualChunkLen, padTo: windowSize
            )
            melInputs.append(melInput)
            expectedOuts.append(expectedOut)

            offset += windowSize
        }

        // Use CoreML native batch API: MLModel.predictions(fromBatch:)
        let batchProvider = MLArrayBatchProvider(array: melInputs)
        let batchResults = try models.audioEncoder.predictions(fromBatch: batchProvider)

        // Extract features from batch results
        for i in 0..<batchResults.count {
            let prediction = batchResults.features(at: i)
            guard let features = prediction.featureValue(for: "audio_features")?.multiArrayValue
            else {
                throw ForcedAlignerError.encoderFailed("No audio_features output")
            }

            let featStride: Int
            if features.shape.count == 3 {
                featStride = features.strides[1].intValue
            } else {
                featStride = features.strides[0].intValue
            }

            let featPtr = features.dataPointer.bindMemory(
                to: Float.self, capacity: features.count
            )
            for f in 0..<expectedOuts[i] {
                var vec = [Float](repeating: 0.0, count: ForcedAlignerConfig.encoderOutputDim)
                let base = f * featStride
                for d in 0..<ForcedAlignerConfig.encoderOutputDim {
                    vec[d] = featPtr[base + d]
                }
                allFeatures.append(vec)
            }
        }

        return allFeatures
    }

    private func createMelInput(
        mel: [[Float]],
        offset: Int,
        chunkLen: Int,
        padTo: Int
    ) throws -> MLDictionaryFeatureProvider {
        let shape: [NSNumber] = [
            1, NSNumber(value: ForcedAlignerConfig.numMelBins), NSNumber(value: padTo),
        ]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        ptr.initialize(repeating: 0.0, count: array.count)

        for bin in 0..<ForcedAlignerConfig.numMelBins {
            for t in 0..<chunkLen {
                let srcIdx = offset + t
                if srcIdx < mel[bin].count {
                    let dstIdx = bin * padTo + t
                    ptr[dstIdx] = mel[bin][srcIdx]
                }
            }
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "mel_input": MLFeatureValue(multiArray: array)
        ])
    }

    // MARK: - Embedding

    private func runEmbedding(inputIds: [Int]) throws -> [[Float]] {
        let seqLen = inputIds.count
        let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
        for i in 0..<seqLen {
            ptr[i] = Int32(inputIds[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: array)
        ])

        let prediction = try CoreMLPredictionWrapper.predict(with: models.embedding, input: input)
        guard let embeddings = prediction.featureValue(for: "embeddings")?.multiArrayValue else {
            throw ForcedAlignerError.decoderFailed("No embeddings output")
        }

        // Parse [1, seqLen, 1024] output using strides
        var result: [[Float]] = []
        let embStride = embeddings.strides[1].intValue
        let embPtr = embeddings.dataPointer.bindMemory(
            to: Float.self,
            capacity: embeddings.strides[0].intValue
        )
        for i in 0..<seqLen {
            let base = i * embStride
            var vec = [Float](repeating: 0.0, count: ForcedAlignerConfig.hiddenSize)
            for d in 0..<ForcedAlignerConfig.hiddenSize {
                vec[d] = embPtr[base + d]
            }
            result.append(vec)
        }

        return result
    }

    // MARK: - Decoder + LM Head

    private func runDecoderWithLmHead(
        hiddenStates: [[Float]],
        cos: [Float],
        sin: [Float]
    ) throws -> [[Float]] {
        let seqLen = ForcedAlignerConfig.prefillSeqLen
        let hidden = ForcedAlignerConfig.hiddenSize
        let headDim = ForcedAlignerConfig.headDim

        // Create hidden_states [1, seqLen, hidden]
        let hiddenShape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: hidden)]
        let hiddenArray = try MLMultiArray(shape: hiddenShape, dataType: .float32)
        let hiddenPtr = hiddenArray.dataPointer.bindMemory(to: Float.self, capacity: seqLen * hidden)
        for i in 0..<seqLen {
            let offset = i * hidden
            for d in 0..<hidden {
                hiddenPtr[offset + d] = hiddenStates[i][d]
            }
        }

        // Create position_cos [1, seqLen, headDim]
        let posShape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: headDim)]
        let cosArray = try MLMultiArray(shape: posShape, dataType: .float32)
        let cosPtr = cosArray.dataPointer.bindMemory(to: Float.self, capacity: cos.count)
        for i in 0..<cos.count {
            cosPtr[i] = cos[i]
        }

        // Create position_sin [1, seqLen, headDim]
        let sinArray = try MLMultiArray(shape: posShape, dataType: .float32)
        let sinPtr = sinArray.dataPointer.bindMemory(to: Float.self, capacity: sin.count)
        for i in 0..<sin.count {
            sinPtr[i] = sin[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenArray),
            "position_cos": MLFeatureValue(multiArray: cosArray),
            "position_sin": MLFeatureValue(multiArray: sinArray),
        ])

        let prediction = try CoreMLPredictionWrapper.predict(with: models.decoderWithLmHead, input: input)
        guard let logits = prediction.featureValue(for: "logits")?.multiArrayValue else {
            throw ForcedAlignerError.decoderFailed("No logits output from decoder+lm_head")
        }

        // Parse [1, seqLen, lmHeadOutputDim] logits using strides
        let vocabDim = ForcedAlignerConfig.lmHeadOutputDim
        let logitsStride = logits.strides[1].intValue
        let logitsPtr = logits.dataPointer.bindMemory(
            to: Float.self, capacity: logits.strides[0].intValue
        )

        var result: [[Float]] = []
        for i in 0..<seqLen {
            let offset = i * logitsStride
            var row = [Float](repeating: 0.0, count: vocabDim)
            for d in 0..<vocabDim {
                row[d] = logitsPtr[offset + d]
            }
            result.append(row)
        }

        return result
    }

    // MARK: - Argmax

    private func argmax(logits: [[Float]], seqLen: Int) -> [Int] {
        let vocabDim = ForcedAlignerConfig.lmHeadOutputDim
        var result = [Int](repeating: 0, count: seqLen)

        for i in 0..<seqLen {
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            logits[i].withUnsafeBufferPointer { buf in
                vDSP_maxvi(buf.baseAddress!, 1, &maxVal, &maxIdx, vDSP_Length(vocabDim))
            }
            result[i] = Int(maxIdx)
        }

        return result
    }

    // MARK: - Sequence Padding

    private func padSequence(_ sequence: [[Float]], to length: Int) -> [[Float]] {
        guard sequence.count < length else { return Array(sequence.prefix(length)) }
        let hiddenSize = ForcedAlignerConfig.hiddenSize
        let padding = [[Float]](
            repeating: [Float](repeating: 0.0, count: hiddenSize),
            count: length - sequence.count
        )
        return sequence + padding
    }

    // MARK: - Timestamp Fix (LIS-based)

    /// Fix non-monotonic timestamps using Longest Increasing Subsequence.
    ///
    /// Replicates `Qwen3ForceAlignProcessor.fix_timestamp()` from the Python reference.
    func fixTimestamps(_ data: [Int]) -> [Int] {
        let n = data.count
        guard n > 0 else { return data }

        // LIS with parent tracking
        var dp = [Int](repeating: 1, count: n)
        var parent = [Int](repeating: -1, count: n)

        for i in 1..<n {
            for j in 0..<i {
                if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                    dp[i] = dp[j] + 1
                    parent[i] = j
                }
            }
        }

        let maxLength = dp.max() ?? 0
        guard let maxIdx = dp.firstIndex(of: maxLength) else { return data }

        // Backtrack to find LIS indices
        var lisIndices: [Int] = []
        var idx = maxIdx
        while idx != -1 {
            lisIndices.append(idx)
            idx = parent[idx]
        }
        lisIndices.reverse()

        var isNormal = [Bool](repeating: false, count: n)
        for idx in lisIndices {
            isNormal[idx] = true
        }

        var result = data
        var i = 0

        while i < n {
            guard !isNormal[i] else {
                i += 1
                continue
            }

            var j = i
            while j < n && !isNormal[j] {
                j += 1
            }

            let anomalyCount = j - i

            // Find nearest normal values on left and right
            let leftVal: Int? = {
                for k in stride(from: i - 1, through: 0, by: -1) where isNormal[k] {
                    return result[k]
                }
                return nil
            }()

            let rightVal: Int? = {
                for k in j..<n where isNormal[k] {
                    return result[k]
                }
                return nil
            }()

            if anomalyCount <= 2 {
                for k in i..<j {
                    if leftVal == nil {
                        result[k] = rightVal ?? 0
                    } else if rightVal == nil {
                        result[k] = leftVal ?? 0
                    } else {
                        result[k] = (k - (i - 1)) <= (j - k) ? leftVal! : rightVal!
                    }
                }
            } else {
                if let left = leftVal, let right = rightVal {
                    let step = Double(right - left) / Double(anomalyCount + 1)
                    for k in i..<j {
                        result[k] = left + Int(step * Double(k - i + 1))
                    }
                } else if let left = leftVal {
                    for k in i..<j {
                        result[k] = left
                    }
                } else if let right = rightVal {
                    for k in i..<j {
                        result[k] = right
                    }
                }
            }

            i = j
        }

        return result
    }
}
