@preconcurrency import CoreML
import Foundation
import OSLog

/// Qwen3-TTS 6-model CoreML synthesizer.
///
/// Pipeline (Argmax-style, matching `inference.py`):
/// 1. Build prefill embeddings: TextProjector(text) + CodeEmbedder(codec) per position
/// 2. CodeDecoder prefill: feed each embedding one at a time with KV cache
/// 3. Autoregressive decode loop:
///    a. MultiCodeDecoder: hidden_states + CB0 → CB1-CB15
///    b. Sum all 16 codec embeddings + tts_pad → CodeDecoder step → next CB0
/// 4. SpeechDecoder: all codec frames → audio waveform
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
    ///   - tokenIds: Pre-tokenized text IDs.
    ///   - useSpeaker: Whether to use speaker embedding (default: true).
    ///   - language: Language for synthesis (default: "english").
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        tokenIds: [Int]? = nil,
        useSpeaker: Bool = true,
        language: String = Qwen3TtsConstants.defaultLanguage
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()

        logger.info("Qwen3-TTS synthesizing: '\(text)'")

        guard let textTokens = tokenIds else {
            throw TTSError.processingFailed(
                "Qwen3-TTS requires pre-tokenized input. Please provide tokenIds.")
        }

        // 1. Build prefill embeddings
        let prefillStart = Date()
        let prefillEmbeds = try await buildPrefillEmbeddings(
            textTokens: textTokens,
            useSpeaker: useSpeaker,
            language: language,
            store: store
        )
        let prefillBuildTime = Date().timeIntervalSince(prefillStart)
        logger.info("Built \(prefillEmbeds.count) prefill embeddings in \(String(format: "%.2f", prefillBuildTime))s")

        // 2. CodeDecoder prefill
        let cdPrefillStart = Date()
        var cdState = CodeDecoderKVState()
        var lastOutput: CodeDecoderOutput!

        for emb in prefillEmbeds {
            lastOutput = try await runCodeDecoderStep(
                inputEmbeds: emb, state: &cdState, store: store)
        }
        let cdPrefillTime = Date().timeIntervalSince(cdPrefillStart)
        logger.info(
            "CodeDecoder prefill: \(prefillEmbeds.count) positions in \(String(format: "%.2f", cdPrefillTime))s"
        )

        // 3. Sample first CB0 from prefill logits
        var logits = extractFloatArray(from: lastOutput.logits)

        suppressControlTokens(&logits)
        suppressEos(&logits)  // min_new_tokens: suppress EOS for step 0
        let firstCb0 = sampleTopK(logits: &logits)
        var generatedCb0s: [Int] = [firstCb0]

        logger.info("First CB0: \(firstCb0)")

        // 4. Autoregressive decode loop
        let decodeStart = Date()
        var allFrames: [[Int]] = []
        var currentCb0 = firstCb0
        var currentHidden = lastOutput.hiddenStates

        // Cache tts_pad embedding for decode loop
        let textProjector = try await store.textProjector()
        let ttsPadEmbed = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsPadTokenId)
        let codeEmbedder = try await store.codeEmbedder()
        let multiCodeEmbedder = try await store.multiCodeEmbedder()

        for step in 0..<Qwen3TtsConstants.maxCodecTokens {
            // MultiCodeDecoder: hidden + CB0 → CB1-CB15
            let cb1to15 = try await runMultiCodeDecoder(
                hiddenStates: currentHidden,
                cb0Token: currentCb0,
                codeEmbedder: codeEmbedder,
                multiCodeEmbedder: multiCodeEmbedder,
                store: store
            )

            let frame = [currentCb0] + cb1to15
            allFrames.append(frame)

            // Build decode input: sum(all 16 codec embeddings) + tts_pad
            let cb0Embed = try runCodeEmbedder(codeEmbedder, tokenId: currentCb0)
            var codecSum = extractFloatArray(from: cb0Embed)

            for cbIdx in 0..<15 {
                let linIdx = cbIdx * Qwen3TtsConstants.codecVocabSize + cb1to15[cbIdx]
                let cbEmbed = try runMultiCodeEmbedder(multiCodeEmbedder, linearizedId: linIdx)
                let cbFloats = extractFloatArray(from: cbEmbed)
                for i in 0..<codecSum.count {
                    codecSum[i] += cbFloats[i]
                }
            }

            // Add tts_pad overlay
            let padFloats = extractFloatArray(from: ttsPadEmbed)
            for i in 0..<codecSum.count {
                codecSum[i] += padFloats[i]
            }

            // Create input_embeds MLMultiArray [1, 1024, 1, 1]
            let decodeInput = try createEmbedding(from: codecSum)

            // CodeDecoder step
            let cdOutput = try await runCodeDecoderStep(
                inputEmbeds: decodeInput, state: &cdState, store: store)
            currentHidden = cdOutput.hiddenStates

            // Sample next CB0
            var nextLogits = extractFloatArray(from: cdOutput.logits)
            suppressControlTokens(&nextLogits)
            if step >= 1 {
                // Allow EOS after min_new_tokens=2 (step 0 was first token, step 1 is second)
            } else {
                suppressEos(&nextLogits)
            }
            applyRepetitionPenalty(&nextLogits, generatedIds: generatedCb0s)
            let nextCb0 = sampleTopK(logits: &nextLogits)

            if nextCb0 == Qwen3TtsConstants.codecEosId {
                logger.info("EOS at step \(step + 1)")
                break
            }

            if cdState.position >= Qwen3TtsConstants.cdKvLen - 1 {
                logger.info("KV cache full at step \(step + 1)")
                break
            }

            generatedCb0s.append(nextCb0)
            currentCb0 = nextCb0
        }

        let decodeTime = Date().timeIntervalSince(decodeStart)
        let fps = Double(allFrames.count) / max(decodeTime, 0.001)
        logger.info(
            "Decoded \(allFrames.count) frames in \(String(format: "%.2f", decodeTime))s"
                + " (\(String(format: "%.1f", fps)) frames/s)"
        )

        // 5. SpeechDecoder: codes → audio
        let speechStart = Date()
        let audioSamples = try await runSpeechDecoder(
            allFrames: allFrames, store: store)
        let speechTime = Date().timeIntervalSince(speechStart)
        logger.info("SpeechDecoder: \(String(format: "%.2f", speechTime))s")

        // 6. Trim to actual frame count
        let expectedSamples = allFrames.count * Qwen3TtsConstants.samplesPerFrame
        let frameTrimmed: [Float]
        if expectedSamples < audioSamples.count {
            frameTrimmed = Array(audioSamples.prefix(expectedSamples))
        } else {
            frameTrimmed = audioSamples
        }

        // Strip leading/trailing silence
        let trimmedSamples = trimSilence(
            frameTrimmed, sampleRate: Qwen3TtsConstants.audioSampleRate)

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
            tokenCount: allFrames.count
        )
    }

    // MARK: - Prefill Embedding Construction

    /// Build dual-embedding prefill sequence matching inference.py.
    ///
    /// Layout: role(3) + control(4) + speaker?(0-1) + bos(1) + text(N) + eos(1) + final(1)
    private static func buildPrefillEmbeddings(
        textTokens: [Int],
        useSpeaker: Bool,
        language: String,
        store: Qwen3TtsModelStore
    ) async throws -> [MLMultiArray] {
        let textProjector = try await store.textProjector()
        let codeEmbedder = try await store.codeEmbedder()

        var embeds: [MLMultiArray] = []

        // [0:3] Role: text_proj only (no codec overlay)
        for tokenId in Qwen3TtsConstants.rolePrefixTokens {
            embeds.append(try runTextProjector(textProjector, tokenId: tokenId))
        }

        // Cache tts_pad, tts_bos, tts_eos embeddings
        let ttsPad = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsPadTokenId)
        let ttsBos = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsBosTokenId)
        let ttsEos = try runTextProjector(textProjector, tokenId: Qwen3TtsConstants.ttsEosTokenId)

        // [3:7] Control: tts_pad + codec_emb([think, think_bos, lang, think_eos])
        let langId =
            Qwen3TtsConstants.languageIds[language] ?? Qwen3TtsConstants.languageIds["english"]!
        let codecCtrlTokens = [
            Qwen3TtsConstants.codecThinkId,
            Qwen3TtsConstants.codecThinkBosId,
            langId,
            Qwen3TtsConstants.codecThinkEosId,
        ]
        for ctok in codecCtrlTokens {
            let codecEmb = try runCodeEmbedder(codeEmbedder, tokenId: ctok)
            embeds.append(try addEmbeddings(ttsPad, codecEmb))
        }

        // [7] Optional speaker embedding
        if useSpeaker, let speakerData = await store.speaker() {
            let speakerEmbed = try createEmbedding(from: speakerData)
            embeds.append(try addEmbeddings(ttsPad, speakerEmbed))
        }

        // Control: tts_bos + codec_emb(codec_pad)
        let codecPadEmb = try runCodeEmbedder(codeEmbedder, tokenId: Qwen3TtsConstants.codecPadId)
        embeds.append(try addEmbeddings(ttsBos, codecPadEmb))

        // Text: text_proj(token) + codec_emb(codec_pad) for each token
        for tokenId in textTokens {
            let textEmb = try runTextProjector(textProjector, tokenId: tokenId)
            embeds.append(try addEmbeddings(textEmb, codecPadEmb))
        }

        // EOS: text_proj(tts_eos) + codec_emb(codec_pad)
        embeds.append(try addEmbeddings(ttsEos, codecPadEmb))

        // Final: tts_pad + codec_emb(codec_bos)
        let codecBosEmb = try runCodeEmbedder(
            codeEmbedder, tokenId: Qwen3TtsConstants.codecBosId)
        embeds.append(try addEmbeddings(ttsPad, codecBosEmb))

        return embeds
    }

    // MARK: - CodeDecoder

    /// KV cache state for the CodeDecoder (28-layer transformer).
    private struct CodeDecoderKVState {
        var keyCache: MLMultiArray
        var valueCache: MLMultiArray
        var position: Int = 0

        init() {
            // [1, 28672, 1, 256] float16
            let shape: [NSNumber] = [
                1, NSNumber(value: Qwen3TtsConstants.cdKvDim), 1,
                NSNumber(value: Qwen3TtsConstants.cdKvLen),
            ]
            keyCache = try! MLMultiArray(shape: shape, dataType: .float16)
            valueCache = try! MLMultiArray(shape: shape, dataType: .float16)
        }
    }

    private struct CodeDecoderOutput {
        let logits: MLMultiArray
        let hiddenStates: MLMultiArray
    }

    /// Run a single CodeDecoder step (prefill or decode).
    private static func runCodeDecoderStep(
        inputEmbeds: MLMultiArray,
        state: inout CodeDecoderKVState,
        store: Qwen3TtsModelStore
    ) async throws -> CodeDecoderOutput {
        let model = try await store.codeDecoder()
        let pos = state.position
        let kvLen = Qwen3TtsConstants.cdKvLen

        // key_padding_mask [1, 256] float16: 0..pos = 0.0, rest = -10000.0
        let keyMask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            keyMask[i] = NSNumber(value: i <= pos ? Float(0.0) : Float(-10000.0))
        }

        // kv_cache_update_mask [1, 256] float16: only pos = 1.0
        let updateMask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            updateMask[i] = NSNumber(value: i == pos ? Float(1.0) : Float(0.0))
        }

        let cacheLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLenArr[0] = NSNumber(value: pos)

        // Cast input_embeds to float16
        let f16Input = try toFloat16(inputEmbeds)

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Input,
            "cache_length": cacheLenArr,
            "key_padding_mask": keyMask,
            "kv_cache_update_mask": updateMask,
            "key_cache": state.keyCache,
            "value_cache": state.valueCache,
        ])

        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        guard let newKeyCache = output.featureValue(for: "new_key_cache")?.multiArrayValue,
            let newValueCache = output.featureValue(for: "new_value_cache")?.multiArrayValue,
            let hiddenStates = output.featureValue(for: "hidden_states")?.multiArrayValue,
            let logits = output.featureValue(for: "logits")?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing CodeDecoder outputs")
        }

        state.keyCache = newKeyCache
        state.valueCache = newValueCache
        state.position += 1

        return CodeDecoderOutput(logits: logits, hiddenStates: hiddenStates)
    }

    // MARK: - MultiCodeDecoder

    /// Run MultiCodeDecoder to generate CB1-CB15 from hidden_states + CB0.
    private static func runMultiCodeDecoder(
        hiddenStates: MLMultiArray,
        cb0Token: Int,
        codeEmbedder: MLModel,
        multiCodeEmbedder: MLModel,
        store: Qwen3TtsModelStore
    ) async throws -> [Int] {
        let model = try await store.multiCodeDecoder()
        let kvLen = Qwen3TtsConstants.mcdKvLen

        // Get properly-strided KV caches from the model via a warmup prediction.
        // CoreML models require specific non-contiguous stride layouts that match
        // their compiled internal representation. Creating MLMultiArrays with standard
        // contiguous strides causes NaN outputs because the model reads wrong memory offsets.
        var (mcdKey, mcdVal) = try await getModelStridedKVCaches(
            model: model, kvLen: kvLen)

        // Position 0: feed hidden_states
        let (mask0, umask0) = try makeMcdMasks(pos: 0, kvLen: kvLen)
        let cacheLen0 = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen0[0] = NSNumber(value: 0)

        let f16Hidden = try toFloat16(hiddenStates)
        let feat0 = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Hidden,
            "cache_length": cacheLen0,
            "key_cache": mcdKey,
            "value_cache": mcdVal,
            "key_padding_mask": mask0,
            "kv_cache_update_mask": umask0,
        ])
        let out0 = try await model.compatPrediction(from: feat0, options: MLPredictionOptions())
        mcdKey = out0.featureValue(for: "new_key_cache")!.multiArrayValue!
        mcdVal = out0.featureValue(for: "new_value_cache")!.multiArrayValue!

        // Position 1: feed CB0 embedding → lm_head[0] → CB1
        let cb0Emb = try runCodeEmbedder(codeEmbedder, tokenId: cb0Token)
        let (mask1, umask1) = try makeMcdMasks(pos: 1, kvLen: kvLen)
        let cacheLen1 = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen1[0] = NSNumber(value: 1)

        let f16Cb0 = try toFloat16(cb0Emb)
        let feat1 = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": f16Cb0,
            "cache_length": cacheLen1,
            "key_cache": mcdKey,
            "value_cache": mcdVal,
            "key_padding_mask": mask1,
            "kv_cache_update_mask": umask1,
        ])
        let out1 = try await model.compatPrediction(from: feat1, options: MLPredictionOptions())
        mcdKey = out1.featureValue(for: "new_key_cache")!.multiArrayValue!
        mcdVal = out1.featureValue(for: "new_value_cache")!.multiArrayValue!

        // CB1 from lm_head[0]
        let allLogits1 = out1.featureValue(for: "all_logits")!.multiArrayValue!
        var cb1Logits = extractSliceLogits(allLogits1, sliceIndex: 0)

        let cb1 = sampleTopK(logits: &cb1Logits)
        var cbTokens = [cb1]

        // Positions 2-15: autoregressive decode for CB2-CB15
        for cbStep in 1..<15 {
            let prevCb = cbTokens.last!
            let linIdx = (cbStep - 1) * Qwen3TtsConstants.codecVocabSize + prevCb
            let cbEmb = try runMultiCodeEmbedder(multiCodeEmbedder, linearizedId: linIdx)

            let mcdPos = cbStep + 1
            let (mask, umask) = try makeMcdMasks(pos: mcdPos, kvLen: kvLen)
            let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
            cacheLen[0] = NSNumber(value: mcdPos)

            let f16Emb = try toFloat16(cbEmb)
            let feat = try MLDictionaryFeatureProvider(dictionary: [
                "input_embeds": f16Emb,
                "cache_length": cacheLen,
                "key_cache": mcdKey,
                "value_cache": mcdVal,
                "key_padding_mask": mask,
                "kv_cache_update_mask": umask,
            ])
            let out = try await model.compatPrediction(from: feat, options: MLPredictionOptions())
            mcdKey = out.featureValue(for: "new_key_cache")!.multiArrayValue!
            mcdVal = out.featureValue(for: "new_value_cache")!.multiArrayValue!

            let allLogits = out.featureValue(for: "all_logits")!.multiArrayValue!
            var cbLogits = extractSliceLogits(allLogits, sliceIndex: cbStep)
            cbTokens.append(sampleTopK(logits: &cbLogits))
        }

        return cbTokens
    }

    /// Create key_padding_mask and kv_cache_update_mask for MultiCodeDecoder.
    private static func makeMcdMasks(
        pos: Int, kvLen: Int
    ) throws -> (MLMultiArray, MLMultiArray) {
        let mask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        let umask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)

        for i in 0..<kvLen {
            mask[i] = NSNumber(value: i <= pos ? Float(0.0) : Float(-10000.0))
            umask[i] = NSNumber(value: i == pos ? Float(1.0) : Float(0.0))
        }

        return (mask, umask)
    }

    /// Get zero-initialized KV caches with the model's expected stride layout.
    ///
    /// CoreML compiled models use specific non-contiguous memory layouts.
    /// The only reliable way to get properly-strided arrays is to run a
    /// prediction and use the output KV caches, then zero them for reuse.
    private static func getModelStridedKVCaches(
        model: MLModel, kvLen: Int
    ) async throws -> (MLMultiArray, MLMultiArray) {
        // Create minimal inputs for a warmup prediction
        let kvDim = Qwen3TtsConstants.mcdKvDim
        let shape: [NSNumber] = [1, NSNumber(value: kvDim), 1, NSNumber(value: kvLen)]

        // Use zero inputs — the output stride layout is what matters
        let dummyInput = try MLMultiArray(shape: [1, 1024, 1, 1], dataType: .float16)
        let dummyKey = try MLMultiArray(shape: shape, dataType: .float16)
        let dummyVal = try MLMultiArray(shape: shape, dataType: .float16)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        for i in 0..<kvLen {
            mask[i] = NSNumber(value: Float(-10000.0))
        }
        let umask = try MLMultiArray(shape: [1, NSNumber(value: kvLen)], dataType: .float16)
        let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)

        let feat = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": dummyInput,
            "cache_length": cacheLen,
            "key_cache": dummyKey,
            "value_cache": dummyVal,
            "key_padding_mask": mask,
            "kv_cache_update_mask": umask,
        ])

        let out = try await model.compatPrediction(from: feat, options: MLPredictionOptions())
        let outKey = out.featureValue(for: "new_key_cache")!.multiArrayValue!
        let outVal = out.featureValue(for: "new_value_cache")!.multiArrayValue!

        // Zero the caches while preserving their stride layout
        for i in 0..<outKey.count { outKey[i] = NSNumber(value: Float(0.0)) }
        for i in 0..<outVal.count { outVal[i] = NSNumber(value: Float(0.0)) }

        return (outKey, outVal)
    }

    // MARK: - SpeechDecoder

    /// Run the SpeechDecoder on all codec frames.
    private static func runSpeechDecoder(
        allFrames: [[Int]],
        store: Qwen3TtsModelStore
    ) async throws -> [Float] {
        let model = try await store.speechDecoder()
        let fixedLen = Qwen3TtsConstants.speechDecoderFrames  // 125
        let numCb = Qwen3TtsConstants.numCodebooks  // 16

        // Build codes tensor [1, 16, 125] int32
        let codes = try MLMultiArray(
            shape: [1, NSNumber(value: numCb), NSNumber(value: fixedLen)],
            dataType: .int32
        )

        // Initialize to zero (pad) using subscript for stride safety
        for i in 0..<(numCb * fixedLen) {
            codes[i] = NSNumber(value: Int32(0))
        }

        // Fill: codes[0, cb, t] = allFrames[t][cb]
        for t in 0..<min(allFrames.count, fixedLen) {
            let frame = allFrames[t]
            for cb in 0..<min(frame.count, numCb) {
                codes[cb * fixedLen + t] = NSNumber(value: Int32(frame[cb]))
            }
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": codes
        ])

        let output = try await model.compatPrediction(from: features, options: MLPredictionOptions())

        guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing SpeechDecoder output")
        }

        return extractFloatArray(from: audioArray)
    }

    // MARK: - Model Runners

    /// TextProjector: text_token → embedding [1, 1024, 1, 1].
    private static func runTextProjector(_ model: MLModel, tokenId: Int) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: tokenId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing TextProjector output")
        }
        return embeds
    }

    /// CodeEmbedder: codec_token → embedding [1, 1024, 1, 1].
    private static func runCodeEmbedder(_ model: MLModel, tokenId: Int) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: tokenId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing CodeEmbedder output")
        }
        return embeds
    }

    /// MultiCodeEmbedder: linearized CB index → embedding [1, 1024, 1, 1].
    private static func runMultiCodeEmbedder(
        _ model: MLModel, linearizedId: Int
    ) throws -> MLMultiArray {
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = NSNumber(value: linearizedId)

        let features = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let output = try model.prediction(from: features, options: MLPredictionOptions())

        guard let embeds = output.featureValue(for: "input_embeds")?.multiArrayValue else {
            throw TTSError.processingFailed("Missing MultiCodeEmbedder output")
        }
        return embeds
    }

    // MARK: - Sampling

    /// Suppress control tokens [2048, 3072) except EOS (2150).
    private static func suppressControlTokens(_ logits: inout [Float]) {
        let eosToken = Qwen3TtsConstants.codecEosId
        let vocabSize = Qwen3TtsConstants.codecVocabSize

        // Save EOS logit before suppression
        let eosLogit = eosToken < logits.count ? logits[eosToken] : -Float.infinity

        // Suppress [2048, 3072)
        for i in vocabSize..<min(3072, logits.count) {
            logits[i] = -.infinity
        }

        // Restore EOS
        if eosToken < logits.count {
            logits[eosToken] = eosLogit
        }
    }

    /// Suppress EOS token (for min_new_tokens enforcement).
    private static func suppressEos(_ logits: inout [Float]) {
        let eosToken = Qwen3TtsConstants.codecEosId
        if eosToken < logits.count {
            logits[eosToken] = -.infinity
        }
    }

    /// Apply repetition penalty to already-generated tokens.
    private static func applyRepetitionPenalty(
        _ logits: inout [Float], generatedIds: [Int]
    ) {
        let penalty = Qwen3TtsConstants.repetitionPenalty
        guard penalty != 1.0 else { return }

        let seen = Set(generatedIds)
        for tokenId in seen {
            guard tokenId < logits.count else { continue }
            if logits[tokenId] > 0 {
                logits[tokenId] /= penalty
            } else {
                logits[tokenId] *= penalty
            }
        }
    }

    /// Sample from logits with temperature + top-k.
    private static func sampleTopK(
        logits: inout [Float],
        temperature: Float = Qwen3TtsConstants.temperature,
        topK: Int = Qwen3TtsConstants.topK
    ) -> Int {
        let count = logits.count
        guard count > 0 else { return 0 }

        // Apply temperature
        for i in 0..<count {
            logits[i] /= temperature
        }

        // Top-k filtering
        if topK > 0 && topK < count {
            var sorted = logits
            sorted.sort(by: >)
            let threshold = sorted[topK - 1]
            for i in 0..<count where logits[i] < threshold {
                logits[i] = -.infinity
            }
        }

        // Softmax
        let maxLogit = logits.max() ?? 0
        var expSum: Float = 0
        var expLogits = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let e = exp(logits[i] - maxLogit)
            expLogits[i] = e
            expSum += e
        }

        // Multinomial sampling
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for i in 0..<count {
            cumulative += expLogits[i] / expSum
            if cumulative >= r {
                return i
            }
        }

        return count - 1
    }

    /// Extract logits for a specific lm_head slice from all_logits.
    ///
    /// all_logits shape from MultiCodeDecoder: [1, 15, 2048].
    /// We extract [0, sliceIndex, :] and return as [Float].
    private static func extractSliceLogits(
        _ allLogits: MLMultiArray, sliceIndex: Int
    ) -> [Float] {
        let vocabSize = Qwen3TtsConstants.codecVocabSize
        let offset = sliceIndex * vocabSize

        var result = [Float](repeating: 0, count: vocabSize)
        for i in 0..<vocabSize {
            result[i] = allLogits[offset + i].floatValue
        }
        return result
    }

    // MARK: - Audio Post-Processing

    /// Trim leading and trailing silence from audio samples.
    private static func trimSilence(
        _ samples: [Float],
        sampleRate: Int,
        threshold: Float = 0.005,
        windowMs: Int = 10,
        padMs: Int = 20
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

        // Find last non-silent window
        let bigWindow = sampleRate / 5
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

    // MARK: - MLMultiArray Helpers

    /// Extract Float array from MLMultiArray using subscript access (stride-safe).
    private static func extractFloatArray(from array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = array[i].floatValue
        }
        return result
    }

    /// Create [1, 1024, 1, 1] float32 embedding from Float array.
    private static func createEmbedding(from data: [Float]) throws -> MLMultiArray {
        let dim = data.count
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: dim), 1, 1], dataType: .float32)
        for (i, value) in data.enumerated() {
            array[i] = NSNumber(value: value)
        }
        return array
    }

    /// Add two embedding MLMultiArrays element-wise.
    private static func addEmbeddings(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let count = a.count
        let result = try MLMultiArray(shape: a.shape, dataType: .float32)
        for i in 0..<count {
            result[i] = NSNumber(value: a[i].floatValue + b[i].floatValue)
        }
        return result
    }

    /// Convert MLMultiArray to float16, preserving stride layout.
    ///
    /// If already float16, returns as-is. CoreML models expect their own output
    /// stride layout, so we must not make non-contiguous arrays contiguous.
    private static func toFloat16(_ array: MLMultiArray) throws -> MLMultiArray {
        if array.dataType == .float16 { return array }
        let count = array.count
        let result = try MLMultiArray(shape: array.shape, dataType: .float16)
        for i in 0..<count {
            result[i] = array[i]
        }
        return result
    }

}
