import Accelerate
import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrManagerFull")

// MARK: - Qwen3-ASR Manager (2-model pipeline)

/// Manages Qwen3-ASR CoreML inference using the optimized 2-model pipeline.
///
/// This variant eliminates the embedding CoreML model by doing embedding lookup
/// directly in Swift. Reduces CoreML calls from 3 to 2 per token decode step.
///
/// Pipeline:
/// 1. Audio -> mel spectrogram -> audio encoder -> audio features
/// 2. Build prompt tokens -> Swift-side embedding lookup -> merge audio features
/// 3. Prefill through decoder -> first token
/// 4. Decode loop: Swift embedding -> decoder -> next token
@available(macOS 15, iOS 18, *)
public final class Qwen3AsrManagerFull {
    private var models: Qwen3AsrModelsFull?
    private let config: Qwen3AsrConfig
    private let rope: Qwen3RoPE

    public init(config: Qwen3AsrConfig = .default) {
        self.config = config
        self.rope = Qwen3RoPE(config: config)
    }

    /// Load all models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await Qwen3AsrModelsFull.load(from: directory, computeUnits: computeUnits)
        logger.info("Qwen3-ASR models (2-model pipeline) loaded successfully")
    }

    /// Transcribe raw audio samples.
    public func transcribe(
        audioSamples: [Float],
        language: String? = nil,
        maxNewTokens: Int = 512
    ) async throws -> String {
        let melExtractor = WhisperMelSpectrogram()
        let mel = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw Qwen3AsrError.generationFailed("Audio too short to extract mel spectrogram")
        }
        return try await transcribe(melSpectrogram: mel, language: language, maxNewTokens: maxNewTokens)
    }

    /// Transcribe from a pre-computed mel spectrogram.
    public func transcribe(
        melSpectrogram: [[Float]],
        language: String? = nil,
        maxNewTokens: Int = 512
    ) async throws -> String {
        guard let models = models else {
            throw Qwen3AsrError.generationFailed("Models not loaded")
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Step 1: Encode audio
        let t1 = CFAbsoluteTimeGetCurrent()
        let audioFeatures = try await encodeAudio(melSpectrogram: melSpectrogram, models: models)
        let numAudioFrames = audioFeatures.count
        let audioEncodeTime = CFAbsoluteTimeGetCurrent() - t1

        // Step 2: Build chat template with audio tokens
        let promptTokens = buildPromptTokens(numAudioFrames: numAudioFrames, language: language)

        // Step 3: Swift-side embedding + audio merge
        let t3 = CFAbsoluteTimeGetCurrent()
        let initialEmbeddings = embedAndMerge(
            promptTokens: promptTokens,
            audioFeatures: audioFeatures,
            models: models
        )
        let embedTime = CFAbsoluteTimeGetCurrent() - t3

        // Step 4: Autoregressive generation
        let t4 = CFAbsoluteTimeGetCurrent()
        let generatedTokenIds = try await generate(
            initialEmbeddings: initialEmbeddings,
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            models: models
        )
        let generateTime = CFAbsoluteTimeGetCurrent() - t4

        // Step 5: Decode tokens to text
        let text = decodeTokens(generatedTokenIds, vocabulary: models.vocabulary)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print(
            "[Qwen3-Full] Timing: audio=\(String(format: "%.2f", audioEncodeTime))s embed=\(String(format: "%.2f", embedTime))s gen=\(String(format: "%.2f", generateTime))s total=\(String(format: "%.2f", elapsed))s prompt=\(promptTokens.count) decoded=\(generatedTokenIds.count)"
        )

        return text
    }

    // MARK: - Audio Encoding

    private func encodeAudio(
        melSpectrogram: [[Float]],
        models: Qwen3AsrModelsFull
    ) async throws -> [[Float]] {
        let windowSize = config.melWindowSize
        let numFrames = melSpectrogram.first?.count ?? 0

        var allFeatures: [[Float]] = []
        var offset = 0

        while offset < numFrames {
            let end = min(offset + windowSize, numFrames)
            let currentWindowSize = end - offset

            let melInput = try createMelInput(
                melSpectrogram: melSpectrogram,
                offset: offset,
                windowSize: currentWindowSize,
                padTo: windowSize
            )

            let prediction = try await models.audioEncoder.prediction(from: melInput)
            guard let features = prediction.featureValue(for: "audio_features")?.multiArrayValue else {
                throw Qwen3AsrError.encoderFailed("No audio_features output")
            }

            let numOutputFrames: Int
            if currentWindowSize == windowSize {
                numOutputFrames = config.outputFramesPerWindow
            } else {
                numOutputFrames = (currentWindowSize + config.convDownsampleFactor - 1) / config.convDownsampleFactor
            }

            for f in 0..<numOutputFrames {
                var vec = [Float](repeating: 0.0, count: config.encoderOutputDim)
                for d in 0..<config.encoderOutputDim {
                    let idx = f * config.encoderOutputDim + d
                    vec[d] = features[idx].floatValue
                }
                allFeatures.append(vec)
            }

            offset += windowSize
        }

        return allFeatures
    }

    private func createMelInput(
        melSpectrogram: [[Float]],
        offset: Int,
        windowSize: Int,
        padTo: Int
    ) throws -> MLDictionaryFeatureProvider {
        let shape: [NSNumber] = [1, NSNumber(value: config.numMelBins), NSNumber(value: padTo)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)

        ptr.initialize(repeating: 0.0, count: array.count)

        for bin in 0..<config.numMelBins {
            for t in 0..<windowSize {
                let srcIdx = offset + t
                if srcIdx < (melSpectrogram[bin].count) {
                    let dstIdx = bin * padTo + t
                    ptr[dstIdx] = melSpectrogram[bin][srcIdx]
                }
            }
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "mel_input": MLFeatureValue(multiArray: array)
        ])
    }

    // MARK: - Token Building

    private func buildPromptTokens(numAudioFrames: Int, language: String?) -> [Int32] {
        var tokens: [Int32] = []

        // System message (empty)
        tokens.append(Int32(config.imStartTokenId))
        tokens.append(Int32(config.systemTokenId))
        tokens.append(Int32(config.newlineTokenId))
        tokens.append(Int32(config.imEndTokenId))
        tokens.append(Int32(config.newlineTokenId))

        // User message with audio
        tokens.append(Int32(config.imStartTokenId))
        tokens.append(Int32(config.userTokenId))
        tokens.append(Int32(config.newlineTokenId))
        tokens.append(Int32(config.audioStartTokenId))
        for _ in 0..<numAudioFrames {
            tokens.append(Int32(config.audioTokenId))
        }
        tokens.append(Int32(config.audioEndTokenId))
        tokens.append(Int32(config.imEndTokenId))
        tokens.append(Int32(config.newlineTokenId))

        // Assistant start
        tokens.append(Int32(config.imStartTokenId))
        tokens.append(Int32(config.assistantTokenId))
        tokens.append(Int32(config.newlineTokenId))

        return tokens
    }

    // MARK: - Swift-side Embedding & Audio Merge

    private func embedAndMerge(
        promptTokens: [Int32],
        audioFeatures: [[Float]],
        models: Qwen3AsrModelsFull
    ) -> [[Float]] {
        // Swift-side embedding lookup (no CoreML call!)
        var embeddings = models.embeddingWeights.embeddings(for: promptTokens)

        // Replace audio_token positions with audio features
        var audioIdx = 0
        for i in 0..<promptTokens.count {
            if promptTokens[i] == Int32(config.audioTokenId), audioIdx < audioFeatures.count {
                embeddings[i] = audioFeatures[audioIdx]
                audioIdx += 1
            }
        }

        return embeddings
    }

    // MARK: - Autoregressive Generation

    private func generate(
        initialEmbeddings: [[Float]],
        promptLength: Int,
        maxNewTokens: Int,
        models: Qwen3AsrModelsFull
    ) async throws -> [Int] {
        let state = models.decoderStateful.makeState()
        var generatedTokens: [Int] = []
        var currentPosition = 0

        guard promptLength > 0 else {
            throw Qwen3AsrError.generationFailed("Empty prompt")
        }

        let effectiveMaxNew = min(maxNewTokens, config.maxCacheSeqLen - promptLength)
        guard effectiveMaxNew > 0 else {
            throw Qwen3AsrError.generationFailed(
                "Prompt length \(promptLength) exceeds cache capacity \(config.maxCacheSeqLen)"
            )
        }

        // ---- Prefill ----
        let prefillStart = CFAbsoluteTimeGetCurrent()

        let (prefillCos, prefillSin) = rope.computeRange(startPosition: 0, count: promptLength)
        let hiddenArray = try createBatchedHiddenArray(
            embeddings: Array(initialEmbeddings[0..<promptLength])
        )
        let cosArray = try createBatchedPositionArray(values: prefillCos, seqLen: promptLength)
        let sinArray = try createBatchedPositionArray(values: prefillSin, seqLen: promptLength)
        let prefillMask = try createPrefillMask(seqLen: promptLength)

        let prefillLogits = try runStatefulDecoder(
            hiddenStates: hiddenArray,
            positionCos: cosArray,
            positionSin: sinArray,
            mask: prefillMask,
            state: state,
            models: models
        )

        currentPosition = promptLength

        // Preallocate decode buffers
        let decHiddenArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: config.hiddenSize)], dataType: .float32
        )
        let decHiddenPtr = decHiddenArray.dataPointer.bindMemory(
            to: Float.self, capacity: config.hiddenSize
        )
        let decodeCosArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: config.headDim)], dataType: .float32
        )
        let decodeCosPtr = decodeCosArray.dataPointer.bindMemory(
            to: Float.self, capacity: config.headDim
        )
        let decodeSinArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: config.headDim)], dataType: .float32
        )
        let decodeSinPtr = decodeSinArray.dataPointer.bindMemory(
            to: Float.self, capacity: config.headDim
        )

        // Get first token from prefill logits
        let firstTokenId = try argmaxFromLogits(prefillLogits, generatedTokens: generatedTokens)
        if !config.eosTokenIds.contains(firstTokenId) {
            generatedTokens.append(firstTokenId)
        }

        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print(
            "[Qwen3-Full]   Prefill: \(String(format: "%.3f", prefillTime))s for \(promptLength) tokens"
        )

        // ---- Decode ----
        if config.eosTokenIds.contains(firstTokenId) {
            return generatedTokens
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        var decoderTotal = 0.0

        for _ in 1..<effectiveMaxNew {
            guard let lastTokenId = generatedTokens.last else { break }

            // Swift-side embedding lookup (no CoreML call!)
            let t1 = CFAbsoluteTimeGetCurrent()
            let nextEmbedding = models.embeddingWeights.embedding(for: lastTokenId)

            nextEmbedding.withUnsafeBufferPointer { src in
                _ = memcpy(decHiddenPtr, src.baseAddress!, config.hiddenSize * MemoryLayout<Float>.size)
            }
            rope.fill(position: currentPosition, cosPtr: decodeCosPtr, sinPtr: decodeSinPtr)
            let endStep = currentPosition + 1
            let mask = try createDecodeMask(endStep: endStep)

            let logits = try runStatefulDecoder(
                hiddenStates: decHiddenArray,
                positionCos: decodeCosArray,
                positionSin: decodeSinArray,
                mask: mask,
                state: state,
                models: models
            )

            currentPosition += 1
            decoderTotal += CFAbsoluteTimeGetCurrent() - t1

            let tokenId = try argmaxFromLogits(logits, generatedTokens: generatedTokens)

            if config.eosTokenIds.contains(tokenId) {
                break
            }

            generatedTokens.append(tokenId)
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let perToken = generatedTokens.isEmpty ? 0.0 : decodeTime / Double(generatedTokens.count)
        print(
            "[Qwen3-Full]   Decode: \(String(format: "%.3f", decodeTime))s for \(generatedTokens.count) tokens (\(String(format: "%.1f", perToken * 1000))ms/tok)"
        )
        return generatedTokens
    }

    // MARK: - Stateful Decoder

    private func runStatefulDecoder(
        hiddenStates: MLMultiArray,
        positionCos: MLMultiArray,
        positionSin: MLMultiArray,
        mask: MLMultiArray,
        state: MLState,
        models: Qwen3AsrModelsFull
    ) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStates),
            "position_cos": MLFeatureValue(multiArray: positionCos),
            "position_sin": MLFeatureValue(multiArray: positionSin),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])

        let output = try models.decoderStateful.prediction(from: input, using: state)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw Qwen3AsrError.decoderFailed("Missing logits from stateful decoder")
        }

        return logits
    }

    // MARK: - Argmax

    private static let repetitionPenalty: Float = 1.0

    private func argmaxFromLogits(
        _ logits: MLMultiArray,
        generatedTokens: [Int]
    ) throws -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: config.vocabSize)

        if Self.repetitionPenalty != 1.0 {
            let seen = Set(generatedTokens)
            for tokenId in seen {
                guard tokenId >= 0, tokenId < config.vocabSize else { continue }
                let logit = ptr[tokenId]
                if logit > 0 {
                    ptr[tokenId] = logit / Self.repetitionPenalty
                } else {
                    ptr[tokenId] = logit * Self.repetitionPenalty
                }
            }
        }

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(config.vocabSize))
        return Int(maxIdx)
    }

    // MARK: - Text Decoding

    private static let asrTextTokenId = 151_704

    private static let bpeUnicodeToByte: [UInt32: UInt8] = {
        var printable = [Int]()
        printable.append(contentsOf: 33...126)
        printable.append(contentsOf: 161...172)
        printable.append(contentsOf: 174...255)
        let printableSet = Set(printable)

        var mapping = [UInt32: UInt8]()
        for b in printable {
            mapping[UInt32(b)] = UInt8(b)
        }
        var extra: UInt32 = 256
        for b in 0...255 {
            if !printableSet.contains(b) {
                mapping[extra] = UInt8(b)
                extra += 1
            }
        }
        return mapping
    }()

    private func decodeTokens(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        var startIdx = 0
        if let asrIdx = tokenIds.firstIndex(of: Self.asrTextTokenId) {
            startIdx = asrIdx + 1
        }
        let transcriptionTokens = Array(tokenIds[startIdx...])

        var pieces: [String] = []
        for id in transcriptionTokens {
            if let piece = vocabulary[id] {
                pieces.append(piece)
            }
        }
        let raw = pieces.joined()

        var bytes = [UInt8]()
        for scalar in raw.unicodeScalars {
            if let byte = Self.bpeUnicodeToByte[scalar.value] {
                bytes.append(byte)
            }
        }

        let decoded = String(bytes: bytes, encoding: .utf8) ?? String(raw.filter { $0.isASCII })
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - MLMultiArray Helpers

    private func createPrefillMask(seqLen: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: seqLen), NSNumber(value: seqLen)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: seqLen * seqLen)
        for i in 0..<seqLen {
            for j in 0..<seqLen {
                ptr[i * seqLen + j] = j > i ? Float(-1e9) : 0.0
            }
        }
        return array
    }

    private func createDecodeMask(endStep: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: endStep)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: endStep)
        for i in 0..<endStep {
            ptr[i] = 0.0
        }
        return array
    }

    private func createBatchedHiddenArray(embeddings: [[Float]]) throws -> MLMultiArray {
        let seqLen = embeddings.count
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: config.hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let totalCount = seqLen * config.hiddenSize
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
        for i in 0..<seqLen {
            let offset = i * config.hiddenSize
            let emb = embeddings[i]
            for j in 0..<config.hiddenSize {
                ptr[offset + j] = emb[j]
            }
        }
        return array
    }

    private func createBatchedPositionArray(values: [Float], seqLen: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: config.headDim)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count {
            ptr[i] = values[i]
        }
        return array
    }
}
