import CoreML
import Foundation

/// Thread-safe CoreML-based multilingual grapheme-to-phoneme converter.
///
/// Uses the CharsiuG2P ByT5 encoder-decoder model to convert words in multiple
/// languages to IPA phonemes. The model uses byte-level tokenization (no vocab
/// file required).
public actor MultilingualG2PModel {

    public static let shared = MultilingualG2PModel()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "MultilingualG2PModel")

    // ByT5 special token IDs
    private let padTokenId: Int32 = 0
    private let eosTokenId: Int32 = 1

    // Byte offset: byte value b maps to token ID b + 3
    private let byteOffset: Int32 = 3

    private let maxDecodeSteps = 128

    // CoreML models (lazy-loaded)
    private var encoder: MLModel?
    private var decoder: MLModel?

    private init() {}

    /// Convert a word to IPA phonemes using the multilingual G2P model.
    ///
    /// - Parameters:
    ///   - word: The word to convert.
    ///   - language: The target language for phonemization.
    /// - Returns: An array of IPA phoneme strings, or `nil` if the model is
    ///   unavailable (e.g. in CI).
    public func phonemize(word: String, language: MultilingualG2PLanguage) throws -> [String]? {
        do {
            try loadIfNeeded()
        } catch {
            if ProcessInfo.processInfo.environment["CI"] != nil {
                logger.warning(
                    "Multilingual G2P unavailable in CI, returning nil for word: \(word)")
                return nil
            }
            throw error
        }

        guard let encoder, let decoder else { return nil }

        // Build input: "<lang-code>: word" encoded as UTF-8 bytes → token IDs
        let inputText = "\(language.prefix)\(word)"
        let inputBytes = Array(inputText.utf8)
        let inputIds = inputBytes.map { Int32($0) + byteOffset }

        let encLen = inputIds.count

        // Encoder input arrays
        let encoderInputIds = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        for i in 0..<encLen {
            encoderInputIds[[0, i] as [NSNumber]] = NSNumber(value: inputIds[i])
            attentionMask[[0, i] as [NSNumber]] = NSNumber(value: Int32(1))
        }

        // Run encoder
        let encoderProvider = try MLDictionaryFeatureProvider(
            dictionary: [
                "input_ids": MLFeatureValue(multiArray: encoderInputIds),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
            ]
        )
        guard let encoderOutput = try? encoder.prediction(from: encoderProvider),
            let encoderHidden = encoderOutput.featureValue(for: "last_hidden_state")?.multiArrayValue
        else {
            throw MultilingualG2PError.encoderPredictionFailed
        }

        // Greedy autoregressive decode
        var outputTokens: [Int32] = []
        var decoderIds: [Int32] = [padTokenId]  // decoder start token

        for _ in 0..<maxDecodeSteps {
            let decLen = decoderIds.count

            let decInput = try MLMultiArray(
                shape: [1, NSNumber(value: decLen)], dataType: .int32)
            for i in 0..<decLen {
                decInput[[0, i] as [NSNumber]] = NSNumber(value: decoderIds[i])
            }

            let decoderProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "decoder_input_ids": MLFeatureValue(multiArray: decInput),
                    "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                    "encoder_attention_mask": MLFeatureValue(multiArray: attentionMask),
                ]
            )

            guard let decoderOutput = try? decoder.prediction(from: decoderProvider),
                let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue
            else {
                throw MultilingualG2PError.decoderPredictionFailed
            }

            // Argmax over last position
            let vocabSize = logits.shape.last!.intValue
            let lastPos = decLen - 1
            var bestId: Int32 = 0
            var bestVal: Float = -.infinity
            for v in 0..<vocabSize {
                let val = logits[[0, lastPos, v] as [NSNumber]].floatValue
                if val > bestVal {
                    bestVal = val
                    bestId = Int32(v)
                }
            }

            if bestId == eosTokenId { break }

            outputTokens.append(bestId)
            decoderIds = [padTokenId] + outputTokens
        }

        // Decode token IDs back to UTF-8 string
        let outputBytes = outputTokens.compactMap { tokenId -> UInt8? in
            let byteVal = tokenId - byteOffset
            guard byteVal >= 0, byteVal <= 255 else { return nil }
            return UInt8(byteVal)
        }

        guard let ipaString = String(bytes: outputBytes, encoding: .utf8), !ipaString.isEmpty else {
            return nil
        }

        // Split IPA string into individual phoneme characters
        return ipaString.map { String($0) }.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    /// Verifies that CoreML models can be loaded.
    public func ensureModelsAvailable() throws {
        try loadIfNeeded()
    }

    // MARK: - Private

    private func loadIfNeeded() throws {
        if encoder != nil && decoder != nil { return }

        let modelsDir = try TtsModels.cacheDirectoryURL()
            .appendingPathComponent("Models")
            .appendingPathComponent(Repo.multilingualG2p.folderName)

        let encoderURL = modelsDir.appendingPathComponent(ModelNames.MultilingualG2P.encoderFile)
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw MultilingualG2PError.modelLoadFailed(
                "\(ModelNames.MultilingualG2P.encoderFile) not found at \(encoderURL.path)")
        }

        let decoderURL = modelsDir.appendingPathComponent(ModelNames.MultilingualG2P.decoderFile)
        guard FileManager.default.fileExists(atPath: decoderURL.path) else {
            throw MultilingualG2PError.modelLoadFailed(
                "\(ModelNames.MultilingualG2P.decoderFile) not found at \(decoderURL.path)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        encoder = try MLModel(contentsOf: encoderURL, configuration: config)
        decoder = try MLModel(contentsOf: decoderURL, configuration: config)

        logger.info("Loaded multilingual G2P CoreML models from \(modelsDir.path)")
    }
}
