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
public final class Qwen3AsrManager {
    private var models: Qwen3AsrModels?
    private let config: Qwen3AsrConfig
    private let rope: Qwen3RoPE

    public init(config: Qwen3AsrConfig = .default) {
        self.config = config
        self.rope = Qwen3RoPE(config: config)
    }

    /// Load all CoreML models from the specified directory.
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await Qwen3AsrModels.load(from: directory, computeUnits: computeUnits)
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

    // MARK: - Autoregressive Generation

    private func generate(
        initialEmbeddings: [[Float]],
        promptLength: Int,
        maxNewTokens: Int,
        models: Qwen3AsrModels
    ) async throws -> [Int] {
        var kvCache = Qwen3KVCache(config: config)
        var generatedTokens: [Int] = []

        // Prefill: process all prompt tokens in a single decoder call
        guard promptLength > 0 else {
            throw Qwen3AsrError.generationFailed("Empty prompt")
        }

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var currentHidden = try await runDecoderPrefill(
            embeddings: Array(initialEmbeddings[0..<promptLength]),
            startPosition: 0,
            kvCache: &kvCache,
            models: models
        )
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print(
            "[Qwen3]   Prefill: \(String(format: "%.3f", prefillTime))s for \(promptLength) tokens (padded to \(Self.prefillSeqLen))"
        )

        // Decode phase: generate tokens one at a time
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var lmHeadTotal = 0.0
        var embedTotal = 0.0
        var decoderTotal = 0.0
        for step in 0..<maxNewTokens {
            // LM head: hidden -> logits (with repetition penalty)
            let t1 = CFAbsoluteTimeGetCurrent()
            let tokenId = try await lmHeadArgmax(
                hiddenStates: currentHidden, generatedTokens: generatedTokens, models: models
            )
            lmHeadTotal += CFAbsoluteTimeGetCurrent() - t1

            // Check for EOS
            if config.eosTokenIds.contains(tokenId) {
                break
            }

            generatedTokens.append(tokenId)

            // Embed the new token
            let t2 = CFAbsoluteTimeGetCurrent()
            let nextEmbedding = try await embedSingleToken(tokenId: Int32(tokenId), models: models)
            embedTotal += CFAbsoluteTimeGetCurrent() - t2

            // Run decoder for next step
            let t3 = CFAbsoluteTimeGetCurrent()
            let position = promptLength + step + 1
            currentHidden = try await runDecoderStep(
                hiddenStates: nextEmbedding,
                position: position,
                kvCache: &kvCache,
                models: models
            )
            decoderTotal += CFAbsoluteTimeGetCurrent() - t3
            // KV cache sequence length updated inside runDecoderStep
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let perToken = generatedTokens.isEmpty ? 0.0 : decodeTime / Double(generatedTokens.count)
        let n = max(generatedTokens.count, 1)
        print(
            "[Qwen3]   Decode: \(String(format: "%.3f", decodeTime))s for \(generatedTokens.count) tokens (\(String(format: "%.1f", perToken * 1000))ms/tok)"
        )
        print(
            "[Qwen3]   Breakdown: lmHead=\(String(format: "%.3f", lmHeadTotal))s(\(String(format: "%.1f", lmHeadTotal / Double(n) * 1000))ms) embed=\(String(format: "%.3f", embedTotal))s(\(String(format: "%.1f", embedTotal / Double(n) * 1000))ms) decoder=\(String(format: "%.3f", decoderTotal))s(\(String(format: "%.1f", decoderTotal / Double(n) * 1000))ms)"
        )
        return generatedTokens
    }

    // MARK: - Decoder Step (single token, for autoregressive decode)

    private func runDecoderStep(
        hiddenStates: [Float],
        position: Int,
        kvCache: inout Qwen3KVCache,
        models: Qwen3AsrModels
    ) async throws -> [Float] {
        let (positionCos, positionSin) = rope.compute(position: position)
        let cosArray = try createPositionArray(values: positionCos)
        let sinArray = try createPositionArray(values: positionSin)

        let hiddenArray = try createHiddenArray(values: hiddenStates)
        let maskArray = try createCausalMask(
            cacheDim: kvCache.cacheSequenceDimension, paddingIndices: kvCache.paddingIndices
        )

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenArray),
            "k_caches": MLFeatureValue(multiArray: kvCache.kCaches),
            "v_caches": MLFeatureValue(multiArray: kvCache.vCaches),
            "position_cos": MLFeatureValue(multiArray: cosArray),
            "position_sin": MLFeatureValue(multiArray: sinArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])

        let output = try await models.decoderStack.prediction(from: input)

        guard let outputHidden = output.featureValue(for: "output_hidden")?.multiArrayValue,
            let kCachesOut = output.featureValue(for: "k_caches_out")?.multiArrayValue,
            let vCachesOut = output.featureValue(for: "v_caches_out")?.multiArrayValue
        else {
            throw Qwen3AsrError.decoderFailed("Missing outputs from decoder stack")
        }

        kvCache.update(kCachesOut: kCachesOut, vCachesOut: vCachesOut)

        // Extract hidden states as [Float]
        let ptr = outputHidden.dataPointer.bindMemory(to: Float.self, capacity: config.hiddenSize)
        return Array(UnsafeBufferPointer(start: ptr, count: config.hiddenSize))
    }

    // MARK: - Batched Prefill

    /// Fixed prefill sequence length — must match DecoderPrefillWrapper.PREFILL_SEQ_LEN.
    private static let prefillSeqLen = 512

    /// Process all prompt embeddings in a single decoder call using the fixed-shape prefill model.
    ///
    /// Pads embeddings to `prefillSeqLen`, runs one prediction call, extracts
    /// the last real position's hidden state, and trims the KV cache to discard
    /// padded entries.
    ///
    /// Falls back to sequential processing if the prompt exceeds prefillSeqLen.
    private func runDecoderPrefill(
        embeddings: [[Float]],
        startPosition: Int,
        kvCache: inout Qwen3KVCache,
        models: Qwen3AsrModels
    ) async throws -> [Float] {
        let realLen = embeddings.count

        // Fall back to sequential if prompt exceeds fixed prefill capacity
        guard realLen <= Self.prefillSeqLen else {
            logger.info("Prompt \(realLen) > prefill capacity \(Self.prefillSeqLen), falling back to sequential")
            return try await runSequentialPrefill(
                embeddings: embeddings, startPosition: startPosition,
                kvCache: &kvCache, models: models
            )
        }

        // Pad embeddings to prefillSeqLen with zeros
        var paddedEmbeddings = embeddings
        let zeroPad = [Float](repeating: 0.0, count: config.hiddenSize)
        for _ in realLen..<Self.prefillSeqLen {
            paddedEmbeddings.append(zeroPad)
        }

        // Compute RoPE for all prefillSeqLen positions
        let (positionCos, positionSin) = rope.computeRange(
            startPosition: startPosition, count: Self.prefillSeqLen
        )
        let cosArray = try createBatchedPositionArray(values: positionCos, seqLen: Self.prefillSeqLen)
        let sinArray = try createBatchedPositionArray(values: positionSin, seqLen: Self.prefillSeqLen)
        let hiddenArray = try createBatchedHiddenArray(embeddings: paddedEmbeddings)

        // Use the fixed-shape prefill model (no attention_mask input — baked in)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenArray),
            "k_caches": MLFeatureValue(multiArray: kvCache.kCaches),
            "v_caches": MLFeatureValue(multiArray: kvCache.vCaches),
            "position_cos": MLFeatureValue(multiArray: cosArray),
            "position_sin": MLFeatureValue(multiArray: sinArray),
        ])

        let output = try await models.decoderPrefill.prediction(from: input)

        guard let outputHidden = output.featureValue(for: "output_hidden")?.multiArrayValue,
            let kCachesOut = output.featureValue(for: "k_caches_out")?.multiArrayValue,
            let vCachesOut = output.featureValue(for: "v_caches_out")?.multiArrayValue
        else {
            throw Qwen3AsrError.decoderFailed("Missing outputs from decoder prefill")
        }

        // Update cache with full prefillSeqLen, then trim to discard padded entries
        kvCache.update(kCachesOut: kCachesOut, vCachesOut: vCachesOut, tokenCount: Self.prefillSeqLen)
        kvCache.trim(toSequenceLength: realLen)

        // Strip the dummy entry from the KV cache.
        // During prefill the dummy is masked with -1e9, but CoreML's masking becomes
        // numerically unstable as cache grows. Removing it before decode prevents
        // catastrophic divergence on some files.
        kvCache.stripDummy()

        // Pad the cache to HEAD_DIM (128) to skip a CoreML attention bad zone.
        // CoreML's coremltools conversion produces catastrophic errors when the KV
        // cache sequence dimension is in [112, 126]. Padding ensures the cache starts
        // at 128 and only grows from there, never entering the bad zone during decode.
        kvCache.padToMinimumLength(config.headDim)

        // Extract the last REAL position's hidden state
        let totalElements = Self.prefillSeqLen * config.hiddenSize
        let ptr = outputHidden.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
        let lastOffset = (realLen - 1) * config.hiddenSize
        return Array(UnsafeBufferPointer(start: ptr + lastOffset, count: config.hiddenSize))
    }

    /// Fallback: process prompt tokens one at a time when prompt exceeds prefill capacity.
    private func runSequentialPrefill(
        embeddings: [[Float]],
        startPosition: Int,
        kvCache: inout Qwen3KVCache,
        models: Qwen3AsrModels
    ) async throws -> [Float] {
        var currentHidden = embeddings[0]
        for i in 0..<embeddings.count {
            currentHidden = try await runDecoderStep(
                hiddenStates: embeddings[i],
                position: startPosition + i,
                kvCache: &kvCache,
                models: models
            )
        }
        // Strip dummy before decode (same as batched prefill path)
        kvCache.stripDummy()
        kvCache.padToMinimumLength(config.headDim)
        return currentHidden
    }

    // MARK: - LM Head

    /// Repetition penalty factor applied to logits of previously generated tokens.
    /// Values > 1.0 discourage repetition. 1.2 is a common default.
    private static let repetitionPenalty: Float = 1.0

    private func lmHeadArgmax(
        hiddenStates: [Float],
        generatedTokens: [Int],
        models: Qwen3AsrModels
    ) async throws -> Int {
        let hiddenArray = try createHiddenArray(values: hiddenStates)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenArray)
        ])

        let output = try await models.lmHead.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw Qwen3AsrError.generationFailed("No logits output")
        }

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

        // Argmax over vocab dimension
        var maxVal: Float = -Float.infinity
        var maxIdx = 0
        for i in 0..<config.vocabSize {
            let val = ptr[i]
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
        return maxIdx
    }

    // MARK: - Token Embedding (single token)

    private func embedSingleToken(tokenId: Int32, models: Qwen3AsrModels) async throws -> [Float] {
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
        return Array(UnsafeBufferPointer(start: embPtr, count: config.hiddenSize))
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

    /// Create attention mask for single-token decode: [1, 1, 1, cacheDim+1].
    ///
    /// All positions are 0.0 (attend) except padding positions which are -1e9 (ignore).
    /// Padding positions exist when the KV cache was padded to skip a CoreML bad zone.
    private func createCausalMask(cacheDim: Int, paddingIndices: Range<Int>?) throws -> MLMultiArray {
        let totalDim = cacheDim + 1  // cache entries + 1 new token
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: totalDim)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: totalDim)
        if let padding = paddingIndices {
            for i in 0..<totalDim {
                ptr[i] = padding.contains(i) ? -1e9 : 0.0
            }
        } else {
            for i in 0..<totalDim {
                ptr[i] = 0.0
            }
        }
        return array
    }

    private func createHiddenArray(values: [Float]) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: config.hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count {
            ptr[i] = values[i]
        }
        return array
    }

    private func createPositionArray(values: [Float]) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: config.headDim)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count {
            ptr[i] = values[i]
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
