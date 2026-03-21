@preconcurrency import CoreML
import Foundation
import OSLog

/// KittenTTS single-shot synthesizer.
///
/// Handles phonemization (via Kokoro's G2P pipeline), tokenization to
/// KittenTTS vocab indices, CoreML inference, and audio extraction.
///
/// Pipeline: text → phonemes (Kokoro G2P) → KittenTTS tokens → CoreML → audio → WAV
public struct KittenTtsSynthesizer {

    static let logger = AppLogger(category: "KittenTtsSynthesizer")

    private enum Context {
        @TaskLocal static var modelStore: KittenTtsModelStore?
    }

    static func withModelStore<T>(
        _ store: KittenTtsModelStore,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelStore.withValue(store) {
            try await operation()
        }
    }

    static func currentModelStore() throws -> KittenTtsModelStore {
        guard let store = Context.modelStore else {
            throw KittenTTSError.processingFailed(
                "KittenTtsSynthesizer requires a model store context.")
        }
        return store
    }

    // MARK: - Public Result Type

    /// Result of a KittenTTS synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data at 24kHz.
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of valid audio samples.
        public let sampleCount: Int
    }

    // MARK: - Vocabulary Mapping

    /// Pre-built scalar-to-index map for the KittenTTS vocabulary.
    ///
    /// Uses `Unicode.Scalar` keys rather than `Character` because the
    /// vocab contains U+0329 (COMBINING VERTICAL LINE BELOW), which
    /// Swift merges with the preceding scalar when viewed as Characters.
    private static let scalarToIndex: [Unicode.Scalar: Int32] = {
        var map: [Unicode.Scalar: Int32] = [:]
        for (index, scalar) in KittenTtsConstants.vocabScalars.enumerated() {
            map[scalar] = Int32(index)
        }
        return map
    }()

    /// Convert IPA phoneme strings to KittenTTS token IDs.
    ///
    /// Each Unicode scalar in each phoneme string is individually mapped
    /// to its position in the KittenTTS vocabulary. Scalars not in the
    /// vocabulary are dropped. The result is wrapped with BOS (0) and EOS (0) tokens.
    ///
    /// - Parameter ipaPhonemes: Array of IPA phoneme strings from G2P.
    /// - Returns: Array of Int32 token IDs including BOS and EOS.
    public static func tokenize(_ ipaPhonemes: [String]) -> [Int32] {
        var ids: [Int32] = [KittenTtsConstants.padTokenId]  // BOS

        for phoneme in ipaPhonemes {
            for scalar in phoneme.unicodeScalars {
                if let id = scalarToIndex[scalar] {
                    if id != KittenTtsConstants.padTokenId {
                        ids.append(id)
                    }
                }
                // Scalars not in vocab are silently dropped
            }
        }

        ids.append(KittenTtsConstants.padTokenId)  // EOS
        return ids
    }

    // MARK: - Synthesis

    /// Synthesize text to WAV audio data.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (e.g., "expr-voice-3-f").
    ///   - speed: Speech speed multiplier (Mini only, 1.0 = normal).
    ///   - deEss: Whether to apply de-essing post-processing.
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        voice: String = KittenTtsConstants.defaultVoice,
        speed: Float = 1.0,
        deEss: Bool = true
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()

        logger.info("KittenTTS synthesizing: '\(text)'")

        // 1. Phonemize using Kokoro's G2P pipeline
        let phonemes = try await phonemize(text: text)
        logger.info("Phonemized to \(phonemes.count) IPA tokens")

        // 2. Tokenize to KittenTTS vocab indices
        let tokenIds = tokenize(phonemes)
        let realTokenCount = tokenIds.count
        logger.info("Tokenized to \(realTokenCount) token IDs")

        // 3. Select appropriate model based on token count
        let (model, modelVariant) = try await store.model(for: realTokenCount)
        let maxTokens = modelVariant.maxTokens
        logger.info("Using \(modelVariant == .fiveSecond ? "5s" : "10s") model (max \(maxTokens) tokens)")

        // 4. Load voice embedding
        let voiceFloats = try await store.voiceData(for: voice)
        let variant = await store.variant

        // 5. Run inference
        let inferenceStart = Date()
        let output: MLFeatureProvider
        if variant == .nano {
            output = try runNanoInference(
                model: model,
                tokenIds: tokenIds,
                maxTokens: maxTokens,
                voiceFloats: voiceFloats,
                modelVariant: modelVariant
            )
        } else {
            output = try runMiniInference(
                model: model,
                tokenIds: tokenIds,
                maxTokens: maxTokens,
                voiceFloats: voiceFloats,
                realTokenCount: realTokenCount,
                speed: speed
            )
        }
        let inferenceElapsed = Date().timeIntervalSince(inferenceStart)
        logger.info("Inference completed in \(String(format: "%.2f", inferenceElapsed))s")

        // 6. Extract audio
        var samples = try extractAudio(from: output)
        logger.info("Extracted \(samples.count) audio samples")

        // 7. Post-processing
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &samples,
                sampleRate: Float(KittenTtsConstants.audioSampleRate),
                deEssAmount: -3.0,
                smoothing: false
            )
        }

        // 8. Encode WAV
        let audioData = try AudioWAV.data(
            from: samples,
            sampleRate: Double(KittenTtsConstants.audioSampleRate)
        )

        let duration = Double(samples.count) / Double(KittenTtsConstants.audioSampleRate)
        logger.info("Audio duration: \(String(format: "%.2f", duration))s")

        return SynthesisResult(
            audio: audioData,
            samples: samples,
            sampleCount: samples.count
        )
    }

    // MARK: - Phonemization

    /// Phonemize text using Kokoro's G2P pipeline.
    ///
    /// Reuses the existing lexicon and G2P model infrastructure.
    private static func phonemize(text: String) async throws -> [String] {
        // Load the Kokoro lexicon/G2P models if not already loaded
        try await KokoroSynthesizer.loadSimplePhonemeDictionary()
        let lexicons = await KokoroSynthesizer.lexiconCache.lexicons()
        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        let allowedPhonemes = Set(vocabulary.keys)

        // Chunk the text into phonemes using Kokoro's pipeline
        let chunks = try await KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lexicons.word,
            caseSensitiveLexicon: lexicons.caseSensitive,
            customLexicon: nil,
            targetTokens: 500,
            hasLanguageToken: false,
            allowedPhonemes: allowedPhonemes,
            phoneticOverrides: [],
            multilingualLanguage: nil
        )

        // Flatten all chunk phonemes
        var allPhonemes: [String] = []
        for chunk in chunks {
            allPhonemes.append(contentsOf: chunk.phonemes)
        }
        return allPhonemes
    }

    // MARK: - Nano Inference

    /// Run KittenTTS Nano inference.
    ///
    /// Nano inputs: input_ids, attention_mask, ref_s, random_phases, source_noise
    private static func runNanoInference(
        model: MLModel,
        tokenIds: [Int32],
        maxTokens: Int,
        voiceFloats: [Float],
        modelVariant: ModelNames.KittenTTS.Variant
    ) throws -> MLFeatureProvider {
        let n = maxTokens
        let t =
            modelVariant == .fiveSecond
            ? KittenTtsConstants.nano5sMaxSamples
            : KittenTtsConstants.nano10sMaxSamples
        let harmonics = KittenTtsConstants.nanoHarmonics

        // input_ids [1, N]
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        let inputIdsPtr = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: n)
        for i in 0..<n {
            inputIdsPtr[i] = i < tokenIds.count ? tokenIds[i] : KittenTtsConstants.padTokenId
        }

        // attention_mask [1, N]
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        let maskPtr = attentionMask.dataPointer.bindMemory(to: Int32.self, capacity: n)
        for i in 0..<n {
            maskPtr[i] = i < tokenIds.count ? 1 : 0
        }

        // ref_s [1, 256]
        let refS = try MLMultiArray(
            shape: [1, NSNumber(value: KittenTtsConstants.nanoVoiceDim)], dataType: .float32)
        let refSPtr = refS.dataPointer.bindMemory(
            to: Float.self, capacity: KittenTtsConstants.nanoVoiceDim)
        for i in 0..<KittenTtsConstants.nanoVoiceDim {
            refSPtr[i] = voiceFloats[i]
        }

        // random_phases [1, 9]
        let randomPhases = try MLMultiArray(
            shape: [1, NSNumber(value: harmonics)], dataType: .float32)
        let phasesPtr = randomPhases.dataPointer.bindMemory(to: Float.self, capacity: harmonics)
        for i in 0..<harmonics {
            phasesPtr[i] = Float.random(in: -Float.pi...Float.pi)
        }

        // source_noise [1, T, 9]
        let sourceNoise = try MLMultiArray(
            shape: [1, NSNumber(value: t), NSNumber(value: harmonics)], dataType: .float32)
        let noisePtr = sourceNoise.dataPointer.bindMemory(
            to: Float.self, capacity: t * harmonics)
        for i in 0..<(t * harmonics) {
            noisePtr[i] = gaussianRandom()
        }

        let featureDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "ref_s": MLFeatureValue(multiArray: refS),
            "random_phases": MLFeatureValue(multiArray: randomPhases),
            "source_noise": MLFeatureValue(multiArray: sourceNoise),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
        return try model.prediction(from: provider)
    }

    // MARK: - Mini Inference

    /// Run KittenTTS Mini inference.
    ///
    /// Mini inputs: input_ids, attention_mask, style, speed
    private static func runMiniInference(
        model: MLModel,
        tokenIds: [Int32],
        maxTokens: Int,
        voiceFloats: [Float],
        realTokenCount: Int,
        speed: Float
    ) throws -> MLFeatureProvider {
        let n = maxTokens
        let dim = KittenTtsConstants.miniVoiceDim

        // input_ids [1, N]
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        let inputIdsPtr = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: n)
        for i in 0..<n {
            inputIdsPtr[i] = i < tokenIds.count ? tokenIds[i] : KittenTtsConstants.padTokenId
        }

        // attention_mask [1, N]
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        let maskPtr = attentionMask.dataPointer.bindMemory(to: Int32.self, capacity: n)
        for i in 0..<n {
            maskPtr[i] = i < tokenIds.count ? 1 : 0
        }

        // style [1, 256] — select row from voice matrix based on token count
        let rowIndex = min(realTokenCount, KittenTtsConstants.miniVoiceRows - 1)
        let style = try MLMultiArray(
            shape: [1, NSNumber(value: dim)], dataType: .float32)
        let stylePtr = style.dataPointer.bindMemory(to: Float.self, capacity: dim)
        let rowOffset = rowIndex * dim
        for i in 0..<dim {
            stylePtr[i] = voiceFloats[rowOffset + i]
        }

        // speed [1]
        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)
        speedArray[0] = NSNumber(value: speed)

        let featureDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "style": MLFeatureValue(multiArray: style),
            "speed": MLFeatureValue(multiArray: speedArray),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
        return try model.prediction(from: provider)
    }

    // MARK: - Audio Extraction

    /// Extract audio samples from model output.
    ///
    /// Output contains:
    /// - `audio` [1, 1, T+20]: raw waveform (zeroed past valid length)
    /// - `audio_length_samples` [1]: number of valid samples
    private static func extractAudio(from output: MLFeatureProvider) throws -> [Float] {
        guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
            throw KittenTTSError.processingFailed("Missing 'audio' output from model")
        }
        guard let lengthArray = output.featureValue(for: "audio_length_samples")?.multiArrayValue
        else {
            throw KittenTTSError.processingFailed(
                "Missing 'audio_length_samples' output from model")
        }

        let validLength = Int(truncating: lengthArray[0])
        let totalLength = audioArray.count
        let sampleCount = min(validLength, totalLength)

        let audioPtr = audioArray.dataPointer.bindMemory(to: Float.self, capacity: totalLength)
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            samples[i] = audioPtr[i]
        }

        return samples
    }

    // MARK: - Utilities

    /// Generate a random sample from a standard normal distribution (Box-Muller transform).
    private static func gaussianRandom() -> Float {
        let u1 = max(Float.random(in: 0..<1), Float.leastNormalMagnitude)
        let u2 = Float.random(in: 0..<1)
        return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    }
}
