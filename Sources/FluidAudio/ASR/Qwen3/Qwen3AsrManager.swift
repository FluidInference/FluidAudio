import Accelerate
import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrManager")

// MARK: - Qwen3-ASR Manager

/// Manages Qwen3-ASR CoreML inference for on-device speech recognition.
///
/// Pipeline:
/// 1. Audio -> Whisper mel spectrogram (WhisperMelSpectrogram, or pre-computed)
/// 2. Mel -> audio encoder -> audio features [1, T', 1024]
/// 3. Build chat template: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*T'<|audio_end|><|im_end|>\n<|im_start|>assistant\n
/// 4. Embed tokens -> replace audio_pad positions with encoder features
/// 5. Run through 28 decoder layers autoregressively with KV-cache
/// 6. LM head -> logits -> argmax -> next token
/// 7. Repeat until EOS or max length
///
/// This is an LLM-based ASR model, not an encoder-decoder-joiner like Parakeet TDT.
/// The decoder is autoregressive and generates text token-by-token.
@available(macOS 15, iOS 18, *)
public final class Qwen3AsrManager {
    private var models: Qwen3AsrModels?
    private let config: Qwen3AsrConfig
    private let rope: Qwen3RoPE
    private var embeddingCache: [Int32: [Float]] = [:]

    public init(config: Qwen3AsrConfig = .default) {
        self.config = config
        self.rope = Qwen3RoPE(config: config)
    }

    /// Load all CoreML models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await Qwen3AsrModels.load(from: directory, computeUnits: computeUnits)
        embeddingCache = [:]
        logger.info("Qwen3-ASR models loaded successfully")
    }

    /// Transcribe raw audio samples.
    ///
    /// Computes Whisper-compatible mel spectrogram internally, then runs the full pipeline.
    ///
    /// - Parameters:
    ///   - audioSamples: Raw audio samples at 16kHz, mono, Float32.
    ///   - language: Optional language hint (e.g., "en", "zh", "ja")
    ///   - maxNewTokens: Maximum number of tokens to generate
    /// - Returns: Transcribed text
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
    ///
    /// - Parameters:
    ///   - melSpectrogram: Mel spectrogram of shape [128, T] (128 mel bins, T frames)
    ///   - language: Optional language hint (e.g., "en", "zh", "ja")
    ///   - maxNewTokens: Maximum number of tokens to generate
    /// - Returns: Transcribed text
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

        // Step 3: Embed tokens and merge audio features
        let t3 = CFAbsoluteTimeGetCurrent()
        let initialEmbeddings = try await embedAndMerge(
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
            "[Qwen3] Timing: audio=\(String(format: "%.2f", audioEncodeTime))s embed=\(String(format: "%.2f", embedTime))s gen=\(String(format: "%.2f", generateTime))s total=\(String(format: "%.2f", elapsed))s prompt=\(promptTokens.count) decoded=\(generatedTokenIds.count)"
        )

        return text
    }

    // MARK: - Audio Encoding

    private func encodeAudio(
        melSpectrogram: [[Float]],
        models: Qwen3AsrModels
    ) async throws -> [[Float]] {
        // melSpectrogram: [128][T] -> we need to chunk into windows of 100 frames
        let windowSize = config.melWindowSize
        let numFrames = melSpectrogram.first?.count ?? 0

        var allFeatures: [[Float]] = []

        // Process each window
        var offset = 0
        while offset < numFrames {
            let end = min(offset + windowSize, numFrames)
            let currentWindowSize = end - offset

            // Create mel input: [1, 128, windowSize]
            let melInput = try createMelInput(
                melSpectrogram: melSpectrogram,
                offset: offset,
                windowSize: currentWindowSize,
                padTo: windowSize
            )

            // Run audio encoder
            let prediction = try await models.audioEncoder.prediction(from: melInput)
            guard let features = prediction.featureValue(for: "audio_features")?.multiArrayValue else {
                throw Qwen3AsrError.encoderFailed("No audio_features output")
            }

            // Extract features: [1, T', 1024] -> [[Float]] of T' vectors
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

        // Zero-fill first (handles padding)
        ptr.initialize(repeating: 0.0, count: array.count)

        // Copy mel data: [1, 128, T]
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

        // Full chat template:
        // <|im_start|>system\n<|im_end|>\n
        // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
        // <|im_start|>assistant\n

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

    // MARK: - Embedding & Audio Merge

    private func embedAndMerge(
        promptTokens: [Int32],
        audioFeatures: [[Float]],
        models: Qwen3AsrModels
    ) async throws -> [[Float]] {
        let seqLen = promptTokens.count

        // Create input_ids: [1, seqLen]
        let shape: [NSNumber] = [1, NSNumber(value: seqLen)]
        let idsArray = try MLMultiArray(shape: shape, dataType: .int32)
        let idsPtr = idsArray.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
        for i in 0..<seqLen {
            idsPtr[i] = promptTokens[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray)
        ])

        let prediction = try await models.embedding.prediction(from: input)
        guard let embeddingsArray = prediction.featureValue(for: "embeddings")?.multiArrayValue else {
            throw Qwen3AsrError.decoderFailed("No embeddings output")
        }

        // Extract embeddings: [1, seqLen, 1024]
        var embeddings = [[Float]](repeating: [Float](repeating: 0.0, count: config.hiddenSize), count: seqLen)
        let embPtr = embeddingsArray.dataPointer.bindMemory(to: Float.self, capacity: embeddingsArray.count)
        for i in 0..<seqLen {
            for d in 0..<config.hiddenSize {
                embeddings[i][d] = embPtr[i * config.hiddenSize + d]
            }
        }

        // Replace audio_token positions with audio features (masked_scatter equivalent)
        var audioIdx = 0
        for i in 0..<seqLen {
            if promptTokens[i] == Int32(config.audioTokenId), audioIdx < audioFeatures.count {
                embeddings[i] = audioFeatures[audioIdx]
                audioIdx += 1
            }
        }

        return embeddings
    }

    // MARK: - Autoregressive Generation (Stateful Decoder)

    private func generate(
        initialEmbeddings: [[Float]],
        promptLength: Int,
        maxNewTokens: Int,
        models: Qwen3AsrModels
    ) async throws -> [Int] {
        // Create fresh KV cache state for this transcription
        let state = models.decoderStateful.makeState()
        var generatedTokens: [Int] = []
        var currentPosition = 0

        guard promptLength > 0 else {
            throw Qwen3AsrError.generationFailed("Empty prompt")
        }

        // Clamp generation to cache capacity
        let effectiveMaxNew = min(maxNewTokens, config.maxCacheSeqLen - promptLength)
        guard effectiveMaxNew > 0 else {
            throw Qwen3AsrError.generationFailed(
                "Prompt length \(promptLength) exceeds cache capacity \(config.maxCacheSeqLen)"
            )
        }

        // ---- Prefill: process all prompt tokens in a single decoder call ----
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

        // Preallocate buffers for decode loop (avoids per-token MLMultiArray allocation)
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

        // Get first token from prefill logits (already [1, 1, vocabSize])
        let firstTokenId = try argmaxFromLogits(prefillLogits, generatedTokens: generatedTokens)
        if !config.eosTokenIds.contains(firstTokenId) {
            generatedTokens.append(firstTokenId)
        }

        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print(
            "[Qwen3]   Prefill: \(String(format: "%.3f", prefillTime))s for \(promptLength) tokens (stateful)"
        )

        // ---- Decode: generate tokens one at a time ----
        // Skip decode loop if first token was EOS
        if config.eosTokenIds.contains(firstTokenId) {
            return generatedTokens
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        var embedTotal = 0.0
        var decoderTotal = 0.0

        for _ in 1..<effectiveMaxNew {
            // Get last generated token
            guard let lastTokenId = generatedTokens.last else { break }

            // Embed the token
            let t1 = CFAbsoluteTimeGetCurrent()
            let nextEmbedding = try await embedSingleToken(tokenId: Int32(lastTokenId), models: models)
            embedTotal += CFAbsoluteTimeGetCurrent() - t1

            // Run stateful decoder step (returns logits directly)
            let t2 = CFAbsoluteTimeGetCurrent()
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
            decoderTotal += CFAbsoluteTimeGetCurrent() - t2

            // Argmax on logits to get next token
            let tokenId = try argmaxFromLogits(logits, generatedTokens: generatedTokens)

            if config.eosTokenIds.contains(tokenId) {
                break
            }

            generatedTokens.append(tokenId)
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let perToken = generatedTokens.isEmpty ? 0.0 : decodeTime / Double(generatedTokens.count)
        let n = max(generatedTokens.count, 1)
        print(
            "[Qwen3]   Decode: \(String(format: "%.3f", decodeTime))s for \(generatedTokens.count) tokens (\(String(format: "%.1f", perToken * 1000))ms/tok)"
        )
        print(
            "[Qwen3]   Breakdown: embed=\(String(format: "%.3f", embedTotal))s(\(String(format: "%.1f", embedTotal / Double(n) * 1000))ms) decoder=\(String(format: "%.3f", decoderTotal))s(\(String(format: "%.1f", decoderTotal / Double(n) * 1000))ms)"
        )
        return generatedTokens
    }

    // MARK: - Stateful Decoder (Fused with lmHead)

    /// Run the stateful decoder for one step (prefill or decode).
    ///
    /// The KV cache lives inside the model as persistent state buffers (fp16).
    /// Each call updates the cache in-place via slice writes — no data is copied in or out.
    /// Returns logits [1, 1, vocabSize] for the last position (lmHead is fused into decoder).
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

    // MARK: - Argmax from Logits

    /// Repetition penalty factor applied to logits of previously generated tokens.
    /// Values > 1.0 discourage repetition. 1.2 is a common default.
    private static let repetitionPenalty: Float = 1.0

    /// Extract the most likely token ID from logits [1, 1, vocabSize].
    private func argmaxFromLogits(
        _ logits: MLMultiArray,
        generatedTokens: [Int]
    ) throws -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: config.vocabSize)

        // Apply repetition penalty: divide positive logits / multiply negative logits
        // for any token that already appeared in the generated sequence
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

        // vDSP vectorized argmax over 151K vocab
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(config.vocabSize))
        return Int(maxIdx)
    }

    // MARK: - Token Embedding (single token)

    private func embedSingleToken(tokenId: Int32, models: Qwen3AsrModels) async throws -> [Float] {
        if let cached = embeddingCache[tokenId] {
            return cached
        }

        let shape: [NSNumber] = [1, 1]
        let idsArray = try MLMultiArray(shape: shape, dataType: .int32)
        let ptr = idsArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
        ptr[0] = tokenId

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray)
        ])

        let output = try await models.embedding.prediction(from: input)
        guard let emb = output.featureValue(for: "embeddings")?.multiArrayValue else {
            throw Qwen3AsrError.decoderFailed("No embedding output")
        }

        let embPtr = emb.dataPointer.bindMemory(to: Float.self, capacity: config.hiddenSize)
        let result = Array(UnsafeBufferPointer(start: embPtr, count: config.hiddenSize))
        embeddingCache[tokenId] = result
        return result
    }

    // MARK: - Text Decoding

    /// Special token ID for `<asr_text>` — marks the start of actual transcription.
    /// Everything before this token is the language prefix (e.g., "language English").
    private static let asrTextTokenId = 151_704

    /// Reverse mapping from GPT-2 byte-level BPE Unicode characters back to byte values.
    /// GPT-2/Qwen tokenizers encode each byte as a visible Unicode character.
    /// Printable bytes (33-126, 161-172, 174-255) map to themselves.
    /// Non-printable bytes map to U+0100+ in order.
    private static let bpeUnicodeToByte: [UInt32: UInt8] = {
        // Printable byte ranges (identity mapping)
        var printable = [Int]()
        printable.append(contentsOf: 33...126)  // ! to ~
        printable.append(contentsOf: 161...172)  // ¡ to ¬
        printable.append(contentsOf: 174...255)  // ® to ÿ
        let printableSet = Set(printable)

        var mapping = [UInt32: UInt8]()

        // Printable: Unicode codepoint == byte value
        for b in printable {
            mapping[UInt32(b)] = UInt8(b)
        }

        // Non-printable: mapped to U+0100, U+0101, ... in order
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
        // Strip prefix: skip all tokens up to and including <asr_text> (ID 151704)
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

        // Decode byte-level BPE: convert each Unicode character back to its byte value,
        // then interpret the byte sequence as UTF-8.
        var bytes = [UInt8]()
        for scalar in raw.unicodeScalars {
            if let byte = Self.bpeUnicodeToByte[scalar.value] {
                bytes.append(byte)
            }
            // Skip characters not in the BPE byte mapping (shouldn't happen)
        }

        let decoded = String(bytes: bytes, encoding: .utf8) ?? String(raw.filter { $0.isASCII })
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - MLMultiArray Helpers

    /// Create lower-triangular causal mask for prefill: [1, 1, seqLen, seqLen].
    /// Positions where j > i get -1e9 (masked), positions where j <= i get 0.0 (attend).
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

    /// Create decode mask for single-token step: [1, 1, 1, endStep].
    /// All zeros — attend to all valid cached positions.
    private func createDecodeMask(endStep: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: endStep)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: endStep)
        for i in 0..<endStep {
            ptr[i] = 0.0
        }
        return array
    }

    // MARK: - Batched MLMultiArray Helpers

    /// Create batched hidden states: [1, seqLen, hiddenSize]
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

    /// Create batched position array: [1, seqLen, headDim]
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
