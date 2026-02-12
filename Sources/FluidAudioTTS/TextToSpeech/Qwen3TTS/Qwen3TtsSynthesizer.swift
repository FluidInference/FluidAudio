@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

/// Qwen3-TTS language model synthesizer.
///
/// Pipeline (V10 decode + KV-cached code predictor):
/// 1. Prefill: Process text context → logits, kv_cache, past_hidden
/// 2. Decode Loop:
///    a. CB0 = argmax(logits) with suppression mask
///    b. Code Predictor (CP prefill + 14 CP decode steps) → CB1-15 (temperature + top-k sampling)
///    c. V10 Decode(CB0, CB1-15, pad_embed, kv_cache, position) → new logits, kv_cache, past_hidden
/// 3. Audio Decoder: All 16 codebooks → waveform
///
/// IMPORTANT: CB1-15 MUST use temperature sampling (not greedy). Greedy code predictor
/// produces silent/broken audio due to codebook pattern interference.
public struct Qwen3TtsSynthesizer {

    static let logger = AppLogger(category: "Qwen3TtsSynthesizer")

    private enum Context {
        @TaskLocal static var modelStore: Qwen3TtsModelStore?
    }

    static func withModelStore<T>(
        _ store: Qwen3TtsModelStore,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelStore.withValue(store) {
            try await operation()
        }
    }

    static func currentModelStore() throws -> Qwen3TtsModelStore {
        guard let store = Context.modelStore else {
            throw TTSError.processingFailed(
                "Qwen3TtsSynthesizer requires a model store context.")
        }
        return store
    }

    // MARK: - Public Types

    /// Result of a Qwen3-TTS synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data (24kHz).
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of codec tokens generated.
        public let tokenCount: Int
    }

    // MARK: - Public API

    /// Synthesize audio from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - tokenIds: Pre-tokenized text IDs (if nil, text must be tokenized externally).
    ///   - useSpeaker: Whether to use speaker embedding (default: true).
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        tokenIds: [Int]? = nil,
        useSpeaker: Bool = true
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()

        logger.info("Qwen3-TTS synthesizing: '\(text)'")

        // Get token IDs (either provided or placeholder)
        let textTokens: [Int]
        if let provided = tokenIds {
            textTokens = provided
        } else {
            throw TTSError.processingFailed(
                "Qwen3-TTS requires pre-tokenized input. Please provide tokenIds.")
        }

        // 1. Run prefill
        let prefillStart = Date()
        let (firstLogits, kvCache, pastHidden, padEmbed) = try await runPrefill(
            textTokens: textTokens,
            useSpeaker: useSpeaker,
            store: store
        )
        let prefillTime = Date().timeIntervalSince(prefillStart)
        logger.info("Prefill completed in \(String(format: "%.2f", prefillTime))s")

        // 2. Sample first token from prefill logits
        let firstToken = sampleToken(from: firstLogits)
        logger.info("First token: \(firstToken)")

        // 3. Run greedy decode loop to generate all 16 codebooks per step
        let decodeStart = Date()
        let actualPrefillLen = textTokens.count + 11  // role(3) + text + think(7) + speaker(1)
        let allCodebooks = try await runDecodeLoop(
            firstToken: firstToken,
            kvCache: kvCache,
            pastHidden: pastHidden,
            padEmbed: padEmbed,
            startPosition: actualPrefillLen,
            store: store
        )
        let decodeTime = Date().timeIntervalSince(decodeStart)
        logger.info(
            "Generated \(allCodebooks.count) frames (16 codebooks each) in \(String(format: "%.2f", decodeTime))s"
        )

        // 4. Run audio decoder
        let decoderStart = Date()
        let audioSamples = try await runAudioDecoder(
            allCodebooks: allCodebooks,
            store: store
        )
        let decoderTime = Date().timeIntervalSince(decoderStart)
        logger.info("Audio decoder completed in \(String(format: "%.2f", decoderTime))s")

        // 5. Trim audio to actual content length and strip leading/trailing silence
        // The audio decoder outputs fixed-length audio (maxCodecTokens frames),
        // but actual content may be shorter if EOS was hit.
        // Each codec frame = sampleRate / 12.5 = 1920 samples.
        let samplesPerFrame = Qwen3TtsConstants.audioSampleRate / 125 * 10  // 1920
        let expectedSamples = allCodebooks.count * samplesPerFrame
        let frameTrimmed: [Float]
        if expectedSamples < audioSamples.count {
            frameTrimmed = Array(audioSamples.prefix(expectedSamples))
            logger.info("Trimmed audio from \(audioSamples.count) to \(expectedSamples) samples")
        } else {
            frameTrimmed = audioSamples
        }

        // Strip leading/trailing silence (sampling can produce silent codec frames)
        var trimmedSamples = trimSilence(frameTrimmed, sampleRate: Qwen3TtsConstants.audioSampleRate)

        // 6. Apply audio post-processing to reduce noise/artifacts
        AudioPostProcessor.applyTtsPostProcessing(
            &trimmedSamples,
            sampleRate: Float(Qwen3TtsConstants.audioSampleRate),
            deEssAmount: -4.0,
            smoothing: true
        )

        // 7. Encode as WAV
        let audioData = try AudioWAV.data(
            from: trimmedSamples,
            sampleRate: Double(Qwen3TtsConstants.audioSampleRate)
        )

        let duration = Double(trimmedSamples.count) / Double(Qwen3TtsConstants.audioSampleRate)
        logger.info("Audio duration: \(String(format: "%.2f", duration))s")

        return SynthesisResult(
            audio: audioData,
            samples: trimmedSamples,
            tokenCount: allCodebooks.count
        )
    }

    // MARK: - Pipeline Steps

    /// Run the LM prefill model.
    private static func runPrefill(
        textTokens: [Int],
        useSpeaker: Bool,
        store: Qwen3TtsModelStore
    ) async throws -> (logits: [Float], kvCache: MLMultiArray, pastHidden: MLMultiArray, padEmbed: MLMultiArray) {
        let model = try await store.prefill()

        // Create role_ids [1, 3]
        let roleIds = try createRoleIds()

        // Create text_ids [1, 128] and text_length [1]
        let (textIds, textLength) = try createTextInputs(tokens: textTokens)

        // Load embeddings (tts_bos, tts_pad, tts_eos)
        let (bosEmbed, padEmbed, eosEmbed) = try await loadTtsEmbeddings(store: store)

        // Create speaker embedding [1, 1024]
        let speakerEmbed = try await createSpeakerEmbedding(useSpeaker: useSpeaker, store: store)

        // Create feature provider
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "role_ids": roleIds,
            "text_ids": textIds,
            "text_length": textLength,
            "tts_bos_embed": bosEmbed,
            "tts_pad_embed": padEmbed,
            "tts_eos_embed": eosEmbed,
            "speaker_embed": speakerEmbed,
        ])

        // Run prediction
        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        // Extract outputs
        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue,
            let kvCacheRaw = output.featureValue(for: "kv_cache")?.multiArrayValue,
            let pastHiddenArray = output.featureValue(for: "past_hidden")?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing prefill outputs")
        }

        // CRITICAL: Trim KV cache to actual length.
        // Prefill pads to maxTextLength+11 but only textLen+11 positions are valid.
        // The decode model attends to ALL KV entries, so garbage entries corrupt output.
        let actualLen = min(textTokens.count, Qwen3TtsConstants.maxTextLength) + 11
        let kvCacheArray = try trimKvCache(kvCacheRaw, toLength: actualLen)

        // Convert logits to [Float]
        let logits = extractFloatArray(from: logitsArray)

        return (logits, kvCacheArray, pastHiddenArray, padEmbed)
    }

    /// Run the decode loop using V10 LM decode + KV-cached code predictor.
    ///
    /// For each step:
    /// 1. Run code predictor (CP prefill + 14 CP decode) to get CB1-15 from past_hidden + CB0
    /// 2. Run V10 LM decode with all 16 codebook IDs to get next logits + past_hidden
    private static func runDecodeLoop(
        firstToken: Int,
        kvCache: MLMultiArray,
        pastHidden: MLMultiArray,
        padEmbed: MLMultiArray,
        startPosition: Int,
        store: Qwen3TtsModelStore
    ) async throws -> [[Int]] {
        let lmModel = try await store.decode()
        let maxTokens = Qwen3TtsConstants.maxCodecTokens
        let eosToken = Qwen3TtsConstants.codecEosTokenId

        var allCodebooks: [[Int]] = []
        var currentKvCache = kvCache
        var currentPastHidden = pastHidden
        var currentCb0 = firstToken
        var position = startPosition

        while allCodebooks.count < maxTokens {
            // Step 1: Run code predictor to get CB1-15
            let cb1_15 = try await runCodePredictor(
                pastHidden: currentPastHidden,
                cb0Token: currentCb0,
                store: store
            )

            // Build full frame: [CB0, CB1, ..., CB15]
            var frame = [currentCb0]
            frame.append(contentsOf: cb1_15)
            allCodebooks.append(frame)

            // Step 2: Run V10 LM decode with all 16 codebook IDs
            let cb0Input = try createTokenInput(token: currentCb0)
            let cb1_15Input = try createCb1_15Input(tokens: cb1_15)
            let positionInput = try createPositionInput(position: position)

            let features = try MLDictionaryFeatureProvider(dictionary: [
                "cb0_id": cb0Input,
                "cb1_15_ids": cb1_15Input,
                "trailing_text_embed": padEmbed,
                "kv_cache": currentKvCache,
                "position": positionInput,
            ])

            let output = try await lmModel.compatPrediction(
                from: features, options: MLPredictionOptions())

            guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue,
                let newKvCache = output.featureValue(for: "new_kv_cache")?.multiArrayValue,
                let newPastHidden = output.featureValue(for: "past_hidden")?.multiArrayValue
            else {
                throw TTSError.processingFailed("Missing V10 decode outputs")
            }

            // Sample next CB0 token from logits with repetition penalty
            var logits = extractFloatArray(from: logitsArray)

            // Apply repetition penalty (matching PyTorch default of 1.3)
            let repetitionPenalty: Float = 1.3
            let recentTokens = allCodebooks.suffix(20).map { $0[0] }  // Last 20 CB0 tokens
            for token in recentTokens {
                if token < logits.count && logits[token] > 0 {
                    logits[token] /= repetitionPenalty
                } else if token < logits.count {
                    logits[token] *= repetitionPenalty
                }
            }

            let nextCb0 = sampleToken(from: logits)

            // Update state
            currentKvCache = newKvCache
            currentPastHidden = newPastHidden
            currentCb0 = nextCb0
            position += 1

            // Check for EOS
            if nextCb0 == eosToken {
                logger.info("EOS at frame \(allCodebooks.count)")
                break
            }
        }

        return allCodebooks
    }

    /// Run the code predictor to generate CB1-15 from past_hidden and CB0 token.
    ///
    /// Uses CP Prefill (2 tokens: past_hidden + cb0_embed) then 14 CP Decode steps.
    private static func runCodePredictor(
        pastHidden: MLMultiArray,
        cb0Token: Int,
        store: Qwen3TtsModelStore
    ) async throws -> [Int] {
        let cpPrefillModel = try await store.cpPrefill()
        let cpDecodeModel = try await store.cpDecode()
        let cpEmbeddings = try await store.getCpEmbeddings()

        // CP Prefill: past_hidden + cb0_token → all_logits + kv_cache
        let cb0Input = try createTokenInput(token: cb0Token)

        let prefillFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "past_hidden": pastHidden,
            "cb0_token": cb0Input,
        ])
        let prefillOutput = try await cpPrefillModel.compatPrediction(
            from: prefillFeatures, options: MLPredictionOptions())

        guard let allLogits = prefillOutput.featureValue(for: "all_logits")?.multiArrayValue,
            let cpKvCache = prefillOutput.featureValue(for: "kv_cache")?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing CP prefill outputs")
        }

        // Extract CB1 from logits[0] using sampling (required for correct audio)
        let cb1 = sampleFromSlice(allLogits, sliceIndex: 0)
        var tokens = [cb1]
        var currentCpKv = cpKvCache

        // CP Decode steps: generate CB2-CB15
        for step in 1..<15 {
            // Look up embedding for the last generated token
            let embedArray = try createEmbeddingFromTable(
                cpEmbeddings: cpEmbeddings,
                tableIndex: step - 1,
                tokenId: tokens.last!
            )

            let posInput = try createPositionInput(position: step + 1)
            let decodeFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "input_embed": embedArray,
                "kv_cache": currentCpKv,
                "position": posInput,
            ])

            let decodeOutput = try await cpDecodeModel.compatPrediction(
                from: decodeFeatures, options: MLPredictionOptions())

            guard let decodeLogits = decodeOutput.featureValue(for: "all_logits")?.multiArrayValue,
                let newCpKv = decodeOutput.featureValue(for: "new_kv_cache")?.multiArrayValue
            else {
                throw TTSError.processingFailed("Missing CP decode outputs")
            }

            // Sample logits[step] for current codebook (each slice is a different LM head)
            let nextToken = sampleFromSlice(decodeLogits, sliceIndex: step)
            tokens.append(nextToken)
            currentCpKv = newCpKv
        }

        return tokens
    }

    /// Run the audio decoder model.
    ///
    /// Note: Audio decoder expects exactly 125 tokens. Uses fixed-length codes tensor.
    private static func runAudioDecoder(
        allCodebooks: [[Int]],
        store: Qwen3TtsModelStore
    ) async throws -> [Float] {
        let model = try await store.audioDecoder()
        let fixedLen = Qwen3TtsConstants.maxCodecTokens  // 125
        let actualLen = allCodebooks.count

        // Build codes tensor [1, 16, 125]
        let codes = try MLMultiArray(
            shape: [1, 16, NSNumber(value: fixedLen)],
            dataType: .int32
        )
        let codesPtr = codes.dataPointer.bindMemory(to: Int32.self, capacity: 16 * fixedLen)

        // Initialize all to pad token
        let padToken = Int32(Qwen3TtsConstants.codecPadTokenId)
        for i in 0..<(16 * fixedLen) {
            codesPtr[i] = padToken
        }

        // Fill all 16 codebooks from the per-frame data
        // allCodebooks[t] is the 16 codebook values for frame t
        for t in 0..<min(actualLen, fixedLen) {
            let frame = allCodebooks[t]
            for cb in 0..<min(frame.count, 16) {
                codesPtr[cb * fixedLen + t] = Int32(frame[cb])
            }
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "codes": codes
        ])

        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing audio decoder output")
        }

        // Extract all audio samples - the model handles the actual length internally
        return extractFloatArray(from: audioArray)
    }

    // MARK: - Input Creation Helpers

    /// Create role_ids input [1, 3].
    private static func createRoleIds() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 3], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: 3)
        for (i, token) in Qwen3TtsConstants.rolePrefixTokens.enumerated() {
            ptr[i] = Int32(token)
        }
        return array
    }

    /// Create text_ids [1, 128] and text_length [1] inputs.
    private static func createTextInputs(tokens: [Int]) throws -> (MLMultiArray, MLMultiArray) {
        let maxLen = Qwen3TtsConstants.maxTextLength
        let actualLen = min(tokens.count, maxLen)

        // text_ids [1, 128]
        let textIds = try MLMultiArray(shape: [1, NSNumber(value: maxLen)], dataType: .int32)
        let textPtr = textIds.dataPointer.bindMemory(to: Int32.self, capacity: maxLen)
        for i in 0..<maxLen {
            textPtr[i] = i < actualLen ? Int32(tokens[i]) : 0
        }

        // text_length [1]
        let textLength = try MLMultiArray(shape: [1], dataType: .int32)
        textLength[0] = NSNumber(value: actualLen)

        return (textIds, textLength)
    }

    /// Create token_id input [1, 1].
    private static func createTokenInput(token: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 1], dataType: .int32)
        array[0] = NSNumber(value: token)
        return array
    }

    /// Create position input [1].
    private static func createPositionInput(position: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: position)
        return array
    }

    /// Create CB1-15 input [1, 15].
    private static func createCb1_15Input(tokens: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 15], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: 15)
        for (i, token) in tokens.prefix(15).enumerated() {
            ptr[i] = Int32(token)
        }
        return array
    }

    /// Create embedding array [1, 1, 1024] from CP embedding table lookup.
    private static func createEmbeddingFromTable(
        cpEmbeddings: [[[Float]]],
        tableIndex: Int,
        tokenId: Int
    ) throws -> MLMultiArray {
        let dim = Qwen3TtsConstants.hiddenSize
        let array = try MLMultiArray(shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)

        let embedding = cpEmbeddings[tableIndex][tokenId]
        for i in 0..<dim {
            ptr[i] = embedding[i]
        }
        return array
    }

    /// Load TTS special token embeddings.
    private static func loadTtsEmbeddings(
        store: Qwen3TtsModelStore
    ) async throws -> (bos: MLMultiArray, pad: MLMultiArray, eos: MLMultiArray) {
        let dim = Qwen3TtsConstants.hiddenSize

        // Check if embeddings were loaded from files
        if let embeddings = await store.getTtsEmbeddings() {
            let bosEmbed = try createEmbeddingArray(from: embeddings.bos)
            let padEmbed = try createEmbeddingArray(from: embeddings.pad)
            let eosEmbed = try createEmbeddingArray(from: embeddings.eos)
            return (bosEmbed, padEmbed, eosEmbed)
        }

        // Create placeholder embeddings (zeros)
        logger.warning("TTS embeddings not loaded - using zeros (synthesis may fail)")

        let bosEmbed = try MLMultiArray(shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let padEmbed = try MLMultiArray(shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let eosEmbed = try MLMultiArray(shape: [1, 1, NSNumber(value: dim)], dataType: .float32)

        return (bosEmbed, padEmbed, eosEmbed)
    }

    /// Create MLMultiArray [1, 1, dim] from Float array.
    private static func createEmbeddingArray(from data: [Float]) throws -> MLMultiArray {
        let dim = data.count
        let array = try MLMultiArray(shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        for (i, value) in data.enumerated() {
            ptr[i] = value
        }
        return array
    }

    /// Create speaker embedding input [1, 1024].
    private static func createSpeakerEmbedding(
        useSpeaker: Bool,
        store: Qwen3TtsModelStore
    ) async throws -> MLMultiArray {
        let dim = Qwen3TtsConstants.hiddenSize
        let array = try MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)

        if useSpeaker, let speaker = await store.speaker() {
            for (i, value) in speaker.enumerated() where i < dim {
                ptr[i] = value
            }
        } else {
            for i in 0..<dim {
                ptr[i] = 0
            }
        }

        return array
    }

    // MARK: - Sampling

    /// Get argmax from a slice of a multi-dimensional logits array.
    ///
    /// The all_logits array from code predictor has shape [15, 1, 2048].
    /// We pick logits[sliceIndex] (shape [1, 2048]) and return argmax.
    private static func argmaxFromSlice(_ allLogits: MLMultiArray, sliceIndex: Int) -> Int {
        // all_logits shape: [15, 1, 2048]
        // For slice i: offset = i * 1 * 2048
        let vocabSize = 2048
        let sliceOffset = sliceIndex * vocabSize

        var maxIdx = 0
        var maxVal: Float = -.infinity

        let count = allLogits.count
        let rawPtr = allLogits.dataPointer.bindMemory(to: Float.self, capacity: count)

        for i in 0..<vocabSize {
            let idx = sliceOffset + i
            guard idx < count else { break }
            let val = rawPtr[idx]
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }

        return maxIdx
    }

    /// Sample from a slice of code predictor logits with temperature + top-k.
    ///
    /// The all_logits array has shape [15, 1, 2048]. We pick logits[sliceIndex],
    /// apply temperature scaling and top-k filtering, then sample from the distribution.
    /// This is required for the code predictor — greedy decoding produces silent audio.
    private static func sampleFromSlice(
        _ allLogits: MLMultiArray,
        sliceIndex: Int,
        temperature: Float = Qwen3TtsConstants.temperature,
        topK: Int = Qwen3TtsConstants.topK
    ) -> Int {
        let vocabSize = 2048
        let sliceOffset = sliceIndex * vocabSize
        let count = allLogits.count
        let rawPtr = allLogits.dataPointer.bindMemory(to: Float.self, capacity: count)

        // Extract logits for this slice and apply temperature
        var logits = [Float](repeating: -.infinity, count: vocabSize)
        for i in 0..<vocabSize {
            let idx = sliceOffset + i
            guard idx < count else { break }
            logits[i] = rawPtr[idx] / temperature
        }

        // Top-k filtering: keep only the top-k values
        if topK > 0 && topK < vocabSize {
            // Find the k-th largest value
            var sorted = logits
            sorted.sort(by: >)
            let threshold = sorted[topK - 1]
            for i in 0..<vocabSize where logits[i] < threshold {
                logits[i] = -.infinity
            }
        }

        // Softmax
        let maxLogit = logits.max() ?? 0
        var expSum: Float = 0
        var expLogits = [Float](repeating: 0, count: vocabSize)
        for i in 0..<vocabSize {
            let e = exp(logits[i] - maxLogit)
            expLogits[i] = e
            expSum += e
        }

        // Multinomial sampling
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<vocabSize {
            cumulative += expLogits[i] / expSum
            if cumulative >= r {
                return i
            }
        }

        return vocabSize - 1  // Fallback
    }

    /// Sample CB0 token from logits with suppression mask and temperature sampling.
    ///
    /// Uses temperature + top-k sampling (matching PyTorch default: do_sample=True,
    /// temperature=0.9, top_k=50). Greedy argmax never produces EOS because a codec
    /// token always has higher logit than EOS — sampling allows natural EOS generation.
    private static func sampleToken(from logits: [Float]) -> Int {
        let eosToken = Qwen3TtsConstants.codecEosTokenId
        let temperature = Qwen3TtsConstants.temperature
        let topK = Qwen3TtsConstants.topK

        // Apply suppression: only allow tokens 0-2047 and EOS token
        var masked = [Float](repeating: -.infinity, count: logits.count)
        for i in 0..<min(2048, logits.count) {
            masked[i] = logits[i]
        }
        if eosToken < logits.count {
            masked[eosToken] = logits[eosToken]
        }

        // Apply temperature
        let allowedCount = min(2048, logits.count) + 1  // 2048 codec tokens + EOS
        for i in 0..<masked.count {
            if masked[i] > -.infinity {
                masked[i] /= temperature
            }
        }

        // Top-k filtering
        if topK > 0 && topK < allowedCount {
            // Find top-k threshold
            var sortable = masked.filter { $0 > -.infinity }
            sortable.sort(by: >)
            if sortable.count > topK {
                let threshold = sortable[topK - 1]
                for i in 0..<masked.count where masked[i] < threshold {
                    masked[i] = -.infinity
                }
            }
        }

        // Softmax
        let maxLogit = masked.max() ?? 0
        var expSum: Float = 0
        var exps = [Float](repeating: 0, count: masked.count)
        for i in 0..<masked.count {
            if masked[i] > -.infinity {
                let e = exp(masked[i] - maxLogit)
                exps[i] = e
                expSum += e
            }
        }

        // Multinomial sample
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<exps.count {
            guard exps[i] > 0 else { continue }
            cumulative += exps[i] / expSum
            if cumulative >= r {
                return i
            }
        }

        return 0  // Fallback
    }

    // MARK: - Audio Post-Processing

    /// Trim leading and trailing silence from audio samples.
    ///
    /// Temperature sampling can produce initial codec frames that decode to near-silence.
    /// This trims those regions with a small padding to preserve natural onset.
    private static func trimSilence(
        _ samples: [Float],
        sampleRate: Int,
        threshold: Float = 0.02,
        windowMs: Int = 20,
        padMs: Int = 50
    ) -> [Float] {
        let windowSize = sampleRate * windowMs / 1000
        let padSize = sampleRate * padMs / 1000
        guard samples.count > windowSize else { return samples }

        // Find first non-silent window
        var start = 0
        for i in stride(from: 0, to: samples.count - windowSize, by: windowSize) {
            var sum: Float = 0
            for j in i..<(i + windowSize) {
                sum += samples[j] * samples[j]
            }
            let rms = (sum / Float(windowSize)).squareRoot()
            if rms > threshold {
                start = max(0, i - padSize)
                break
            }
        }

        // Find last non-silent window (use larger window to skip tiny blips)
        let bigWindow = sampleRate / 5  // 200ms
        var end = samples.count
        for i in stride(from: samples.count - bigWindow, through: 0, by: -windowSize) {
            let windowEnd = min(i + bigWindow, samples.count)
            var sum: Float = 0
            for j in i..<windowEnd {
                sum += samples[j] * samples[j]
            }
            let rms = (sum / Float(windowEnd - i)).squareRoot()
            if rms > threshold {
                end = min(samples.count, windowEnd + padSize)
                break
            }
        }

        guard start < end else { return samples }
        return Array(samples[start..<end])
    }

    // MARK: - KV Cache

    /// Trim KV cache to actual sequence length.
    ///
    /// The prefill model pads to maxTextLength+11 but only textLen+11 positions are valid.
    /// The decode model attends to ALL KV cache entries (no causal mask for single-query),
    /// so padding entries must be removed to avoid corrupted attention.
    ///
    /// KV cache shape: [56, 1, 8, seqLen, 128] → [56, 1, 8, actualLen, 128]
    private static func trimKvCache(_ kvCache: MLMultiArray, toLength actualLen: Int) throws -> MLMultiArray {
        // kvCache shape: [kvEntries, batch, kvHeads, seqLen, headDim]
        let kvEntries = kvCache.shape[0].intValue  // 56
        let batch = kvCache.shape[1].intValue  // 1
        let kvHeads = kvCache.shape[2].intValue  // 8
        let fullLen = kvCache.shape[3].intValue  // maxTextLen + 11
        let headDim = kvCache.shape[4].intValue  // 128

        guard actualLen <= fullLen else {
            throw TTSError.processingFailed(
                "actualLen \(actualLen) exceeds KV cache length \(fullLen)")
        }

        // Already the right size? Return as-is.
        guard actualLen < fullLen else { return kvCache }

        let trimmed = try MLMultiArray(
            shape: [
                NSNumber(value: kvEntries), NSNumber(value: batch),
                NSNumber(value: kvHeads), NSNumber(value: actualLen),
                NSNumber(value: headDim),
            ],
            dataType: .float32
        )

        let srcPtr = kvCache.dataPointer.bindMemory(
            to: Float.self, capacity: kvCache.count)
        let dstPtr = trimmed.dataPointer.bindMemory(
            to: Float.self, capacity: trimmed.count)

        // Copy entry by entry: [e][b][h][0..<actualLen][d]
        for e in 0..<kvEntries {
            for b in 0..<batch {
                for h in 0..<kvHeads {
                    let srcBase = ((e * batch + b) * kvHeads + h) * fullLen * headDim
                    let dstBase = ((e * batch + b) * kvHeads + h) * actualLen * headDim
                    let copyCount = actualLen * headDim
                    for i in 0..<copyCount {
                        dstPtr[dstBase + i] = srcPtr[srcBase + i]
                    }
                }
            }
        }

        return trimmed
    }

    // MARK: - Array Extraction

    /// Extract Float array from MLMultiArray.
    private static func extractFloatArray(from array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)

        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                result[i] = ptr[i]
            }
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count {
                result[i] = Float(Float16(bitPattern: ptr[i]))
            }
        default:
            logger.warning("Unexpected array data type: \(array.dataType.rawValue)")
        }

        return result
    }

    /// Extract Int32 array from MLMultiArray.
    private static func extractInt32Array(from array: MLMultiArray) -> [Int] {
        let count = array.count
        var result = [Int](repeating: 0, count: count)

        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            result[i] = Int(ptr[i])
        }

        return result
    }
}
