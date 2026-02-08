import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrManager")

// MARK: - Qwen3-ASR Manager (2-model pipeline)

/// Manages Qwen3-ASR CoreML inference using the optimized 2-model pipeline.
///
/// - Warning: **Beta** - This API is experimental and subject to change.
///
/// This uses Swift-side embedding lookup from a preloaded weight matrix,
/// eliminating the embedding CoreML model. Reduces CoreML calls from 3 to 2 per token.
///
/// Pipeline:
/// 1. Audio -> mel spectrogram -> audio encoder -> audio features
/// 2. Build prompt tokens -> Swift-side embedding lookup -> merge audio features
/// 3. Prefill through decoder -> first token
/// 4. Decode loop: Swift embedding -> decoder -> next token
@available(macOS 15, iOS 18, *)
public actor Qwen3AsrManager {
    private var models: Qwen3AsrModels?
    private let rope: Qwen3RoPE
    private let melExtractor: WhisperMelSpectrogram

    public init() {
        self.rope = Qwen3RoPE()
        self.melExtractor = WhisperMelSpectrogram()
    }

    /// Load all models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await Qwen3AsrModels.load(from: directory, computeUnits: computeUnits)
        logger.info("Qwen3-ASR models (2-model pipeline) loaded successfully")
    }

    /// Transcribe raw audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - language: Optional language hint (ISO code like "en", "zh", or English name like "English").
    ///               Pass nil for automatic language detection.
    ///   - maxNewTokens: Maximum number of tokens to generate.
    /// - Returns: Transcribed text.
    public func transcribe(
        audioSamples: [Float],
        language: String? = nil,
        maxNewTokens: Int = 512
    ) async throws -> String {
        let mel = melExtractor.compute(audio: audioSamples)
        guard !mel.isEmpty else {
            throw Qwen3AsrError.generationFailed("Audio too short to extract mel spectrogram")
        }
        return try await transcribe(melSpectrogram: mel, language: language, maxNewTokens: maxNewTokens)
    }

    /// Transcribe raw audio samples with typed language.
    public func transcribe(
        audioSamples: [Float],
        language: Qwen3AsrConfig.Language?,
        maxNewTokens: Int = 512
    ) async throws -> String {
        try await transcribe(
            audioSamples: audioSamples,
            language: language?.englishName,
            maxNewTokens: maxNewTokens
        )
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

        // Resolve language
        let resolvedLanguage: Qwen3AsrConfig.Language?
        if let lang = language {
            resolvedLanguage = Qwen3AsrConfig.Language(from: lang)
            if resolvedLanguage == nil {
                logger.warning("Unknown language '\(lang)', using automatic detection")
            }
        } else {
            resolvedLanguage = nil
        }

        // Step 1: Encode audio
        let t1 = CFAbsoluteTimeGetCurrent()
        let audioFeatures = try encodeAudio(melSpectrogram: melSpectrogram, models: models)
        let numAudioFrames = audioFeatures.count
        let audioEncodeTime = CFAbsoluteTimeGetCurrent() - t1

        // Step 2: Build chat template with audio tokens
        let promptTokens = buildPromptTokens(numAudioFrames: numAudioFrames, language: resolvedLanguage)

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
        let generatedTokenIds = try generate(
            initialEmbeddings: initialEmbeddings,
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            models: models
        )
        let generateTime = CFAbsoluteTimeGetCurrent() - t4

        // Step 5: Decode tokens to text
        let text = decodeTokens(generatedTokenIds, vocabulary: models.vocabulary)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.debug(
            "Timing: audio=\(String(format: "%.2f", audioEncodeTime))s embed=\(String(format: "%.2f", embedTime))s gen=\(String(format: "%.2f", generateTime))s total=\(String(format: "%.2f", elapsed))s prompt=\(promptTokens.count) decoded=\(generatedTokenIds.count)"
        )

        return text
    }

    // MARK: - Audio Encoding

    private func encodeAudio(
        melSpectrogram: [[Float]],
        models: Qwen3AsrModels
    ) throws -> [[Float]] {
        let windowSize = Qwen3AsrConfig.melWindowSize
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

            let prediction = try models.audioEncoder.prediction(from: melInput)
            guard let features = prediction.featureValue(for: "audio_features")?.multiArrayValue else {
                throw Qwen3AsrError.encoderFailed("No audio_features output")
            }

            let numOutputFrames: Int
            if currentWindowSize == windowSize {
                numOutputFrames = Qwen3AsrConfig.outputFramesPerWindow
            } else {
                numOutputFrames =
                    (currentWindowSize + Qwen3AsrConfig.convDownsampleFactor - 1)
                    / Qwen3AsrConfig.convDownsampleFactor
            }

            for f in 0..<numOutputFrames {
                var vec = [Float](repeating: 0.0, count: Qwen3AsrConfig.encoderOutputDim)
                for d in 0..<Qwen3AsrConfig.encoderOutputDim {
                    let idx = f * Qwen3AsrConfig.encoderOutputDim + d
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
        let shape: [NSNumber] = [1, NSNumber(value: Qwen3AsrConfig.numMelBins), NSNumber(value: padTo)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)

        ptr.initialize(repeating: 0.0, count: array.count)

        for bin in 0..<Qwen3AsrConfig.numMelBins {
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

    /// Task description token IDs for language-specific transcription.
    /// These are tokenized versions of "Transcribe the audio to {Language} text."
    private static let taskTokens: [Qwen3AsrConfig.Language: [Int32]] = [
        .english: [3246, 56541, 279, 7461, 311, 6364, 1467, 13],
        .chinese: [3246, 56541, 279, 7461, 311, 8449, 1467, 13],
        .cantonese: [3246, 56541, 279, 7461, 311, 56782, 26730, 1467, 13],
        .japanese: [3246, 56541, 279, 7461, 311, 11411, 1467, 13],
        .korean: [3246, 56541, 279, 7461, 311, 15791, 1467, 13],
        .french: [3246, 56541, 279, 7461, 311, 8620, 1467, 13],
        .german: [3246, 56541, 279, 7461, 311, 6581, 1467, 13],
        .spanish: [3246, 56541, 279, 7461, 311, 14610, 1467, 13],
        .portuguese: [3246, 56541, 279, 7461, 311, 42322, 1467, 13],
        .italian: [3246, 56541, 279, 7461, 311, 15333, 1467, 13],
        .russian: [3246, 56541, 279, 7461, 311, 10479, 1467, 13],
        .arabic: [3246, 56541, 279, 7461, 311, 17900, 1467, 13],
        .hindi: [3246, 56541, 279, 7461, 311, 43083, 1467, 13],
        .thai: [3246, 56541, 279, 7461, 311, 40764, 1467, 13],
        .vietnamese: [3246, 56541, 279, 7461, 311, 48416, 1467, 13],
        .indonesian: [3246, 56541, 279, 7461, 311, 66986, 1467, 13],
        .malay: [3246, 56541, 279, 7461, 311, 80985, 1467, 13],
        .turkish: [3246, 56541, 279, 7461, 311, 38703, 1467, 13],
        .dutch: [3246, 56541, 279, 7461, 311, 19227, 1467, 13],
        .swedish: [3246, 56541, 279, 7461, 311, 54259, 1467, 13],
        .danish: [3246, 56541, 279, 7461, 311, 39093, 1467, 13],
        .finnish: [3246, 56541, 279, 7461, 311, 56391, 1467, 13],
        .polish: [3246, 56541, 279, 7461, 311, 34827, 1467, 13],
        .czech: [3246, 56541, 279, 7461, 311, 51728, 1467, 13],
        .greek: [3246, 56541, 279, 7461, 311, 18173, 1467, 13],
        .hungarian: [3246, 56541, 279, 7461, 311, 57751, 1467, 13],
        .romanian: [3246, 56541, 279, 7461, 311, 56949, 1467, 13],
        .persian: [3246, 56541, 279, 7461, 311, 59181, 1467, 13],
        .filipino: [3246, 56541, 279, 7461, 311, 66847, 1467, 13],
        .macedonian: [3246, 56541, 279, 7461, 311, 17067, 103881, 1467, 13],
    ]

    private func buildPromptTokens(numAudioFrames: Int, language: Qwen3AsrConfig.Language?) -> [Int32] {
        var tokens: [Int32] = []

        // System message with optional task description
        tokens.append(Int32(Qwen3AsrConfig.imStartTokenId))
        tokens.append(Int32(Qwen3AsrConfig.systemTokenId))
        tokens.append(Int32(Qwen3AsrConfig.newlineTokenId))
        if let lang = language, let taskToks = Self.taskTokens[lang] {
            tokens.append(contentsOf: taskToks)
        }
        tokens.append(Int32(Qwen3AsrConfig.imEndTokenId))
        tokens.append(Int32(Qwen3AsrConfig.newlineTokenId))

        // User message with audio
        tokens.append(Int32(Qwen3AsrConfig.imStartTokenId))
        tokens.append(Int32(Qwen3AsrConfig.userTokenId))
        tokens.append(Int32(Qwen3AsrConfig.newlineTokenId))
        tokens.append(Int32(Qwen3AsrConfig.audioStartTokenId))
        for _ in 0..<numAudioFrames {
            tokens.append(Int32(Qwen3AsrConfig.audioTokenId))
        }
        tokens.append(Int32(Qwen3AsrConfig.audioEndTokenId))
        tokens.append(Int32(Qwen3AsrConfig.imEndTokenId))
        tokens.append(Int32(Qwen3AsrConfig.newlineTokenId))

        // Assistant start
        tokens.append(Int32(Qwen3AsrConfig.imStartTokenId))
        tokens.append(Int32(Qwen3AsrConfig.assistantTokenId))
        tokens.append(Int32(Qwen3AsrConfig.newlineTokenId))

        return tokens
    }

    // MARK: - Swift-side Embedding & Audio Merge

    private func embedAndMerge(
        promptTokens: [Int32],
        audioFeatures: [[Float]],
        models: Qwen3AsrModels
    ) -> [[Float]] {
        // Swift-side embedding lookup (no CoreML call!)
        var embeddings = models.embeddingWeights.embeddings(for: promptTokens)

        // Replace audio_token positions with audio features
        var audioIdx = 0
        for i in 0..<promptTokens.count {
            if promptTokens[i] == Int32(Qwen3AsrConfig.audioTokenId), audioIdx < audioFeatures.count {
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
        models: Qwen3AsrModels
    ) throws -> [Int] {
        let state = models.decoderStateful.makeState()
        var generatedTokens: [Int] = []
        var currentPosition = 0

        guard promptLength > 0 else {
            throw Qwen3AsrError.generationFailed("Empty prompt")
        }

        let effectiveMaxNew = min(maxNewTokens, Qwen3AsrConfig.maxCacheSeqLen - promptLength)
        guard effectiveMaxNew > 0 else {
            throw Qwen3AsrError.generationFailed(
                "Prompt length \(promptLength) exceeds cache capacity \(Qwen3AsrConfig.maxCacheSeqLen)"
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
            shape: [1, 1, NSNumber(value: Qwen3AsrConfig.hiddenSize)], dataType: .float32
        )
        let decHiddenPtr = decHiddenArray.dataPointer.bindMemory(
            to: Float.self, capacity: Qwen3AsrConfig.hiddenSize
        )
        let decodeCosArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: Qwen3AsrConfig.headDim)], dataType: .float32
        )
        let decodeCosPtr = decodeCosArray.dataPointer.bindMemory(
            to: Float.self, capacity: Qwen3AsrConfig.headDim
        )
        let decodeSinArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: Qwen3AsrConfig.headDim)], dataType: .float32
        )
        let decodeSinPtr = decodeSinArray.dataPointer.bindMemory(
            to: Float.self, capacity: Qwen3AsrConfig.headDim
        )

        // Get first token from prefill logits
        let firstTokenId = argmaxFromLogits(prefillLogits)
        if !Qwen3AsrConfig.eosTokenIds.contains(firstTokenId) {
            generatedTokens.append(firstTokenId)
        }

        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        logger.debug("Prefill: \(String(format: "%.3f", prefillTime))s for \(promptLength) tokens")

        // ---- Decode ----
        if Qwen3AsrConfig.eosTokenIds.contains(firstTokenId) {
            return generatedTokens
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()

        for _ in 1..<effectiveMaxNew {
            guard let lastTokenId = generatedTokens.last else { break }

            // Swift-side embedding lookup (no CoreML call!)
            let nextEmbedding = models.embeddingWeights.embedding(for: lastTokenId)

            nextEmbedding.withUnsafeBufferPointer { src in
                _ = memcpy(decHiddenPtr, src.baseAddress!, Qwen3AsrConfig.hiddenSize * MemoryLayout<Float>.size)
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

            let tokenId = argmaxFromLogits(logits)

            if Qwen3AsrConfig.eosTokenIds.contains(tokenId) {
                break
            }

            generatedTokens.append(tokenId)
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let perToken = generatedTokens.isEmpty ? 0.0 : decodeTime / Double(generatedTokens.count)
        logger.debug(
            "Decode: \(String(format: "%.3f", decodeTime))s for \(generatedTokens.count) tokens (\(String(format: "%.1f", perToken * 1000))ms/tok)"
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
        models: Qwen3AsrModels
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

    private func argmaxFromLogits(_ logits: MLMultiArray) -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: Qwen3AsrConfig.vocabSize)
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(Qwen3AsrConfig.vocabSize))
        return Int(maxIdx)
    }

    // MARK: - Text Decoding

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
        if let asrIdx = tokenIds.firstIndex(of: Qwen3AsrConfig.asrTextTokenId) {
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
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: Qwen3AsrConfig.hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let totalCount = seqLen * Qwen3AsrConfig.hiddenSize
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
        for i in 0..<seqLen {
            let offset = i * Qwen3AsrConfig.hiddenSize
            let emb = embeddings[i]
            for j in 0..<Qwen3AsrConfig.hiddenSize {
                ptr[offset + j] = emb[j]
            }
        }
        return array
    }

    private func createBatchedPositionArray(values: [Float], seqLen: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: Qwen3AsrConfig.headDim)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count {
            ptr[i] = values[i]
        }
        return array
    }
}
