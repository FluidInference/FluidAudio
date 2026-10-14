#if TTS
import CoreML
import Foundation
import os

/// Thread-safe CoreML-based grapheme-to-phoneme converter.
/// Uses a small BART encoder-decoder model to convert English words to IPA phonemes.
///
/// Assets load from `<models root>/kokoro/`, where the root is the
/// `directory` the instance was created with, or the shared TTS cache
/// (`TtsCacheDirectory`) when `nil`. Use ``shared(directory:)`` so every
/// consumer of one root (KokoroAne, StyleTTS2) reuses a single loaded model.
actor G2PModel {
    enum G2PModelError: Error, LocalizedError {
        case vocabLoadFailed(String)
        case modelLoadFailed(String)
        case encoderPredictionFailed
        case decoderPredictionFailed

        var errorDescription: String? {
            switch self {
            case .vocabLoadFailed(let detail):
                return "Failed to load G2P \(ModelNames.G2P.vocabularyFile): \(detail)"
            case .modelLoadFailed(let detail):
                return "Failed to load G2P CoreML model: \(detail)"
            case .encoderPredictionFailed:
                return "G2P encoder prediction failed."
            case .decoderPredictionFailed:
                return "G2P decoder prediction failed."
            }
        }
    }

    /// Instance bound to the default TTS cache root.
    static var shared: G2PModel { shared(directory: nil) }

    private struct WeakModel {
        weak var model: G2PModel?
    }

    /// One live instance per models root, so consumers of a root share one
    /// loaded model while any of them holds it; a root nobody references
    /// anymore releases its CoreML models.
    private static let registry = OSAllocatedUnfairLock<[String: WeakModel]>(initialState: [:])

    /// The instance that loads from `<directory>/kokoro/` (`nil` = default
    /// TTS cache root, as `TtsCacheDirectory.ensure()/Models`). Hold the
    /// returned instance; it is not retained by the registry.
    static func shared(directory: URL?) -> G2PModel {
        let key = registryKey(directory: directory)
        return registry.withLock { registry in
            if let existing = registry[key]?.model { return existing }
            registry = registry.filter { $0.value.model != nil }
            let model = G2PModel(directory: directory)
            registry[key] = WeakModel(model: model)
            return model
        }
    }

    /// Resolved models-root path, so `nil`, the explicit default root and a
    /// symlinked spelling of one location map to the same instance.
    private static func registryKey(directory: URL?) -> String {
        guard let root = try? directory ?? TtsCacheDirectory.ensure().appendingPathComponent("Models") else {
            return ""
        }
        return root.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "G2PModel")

    /// Models root passed at creation; `nil` resolves to the default cache.
    nonisolated let modelsDirectory: URL?

    // Vocab tables (loaded once)
    private var graphemeToId: [Character: Int]?
    private var idToPhoneme: [Int: String]?
    private var bosTokenId: Int = 1
    private var eosTokenId: Int = 2
    private var unkTokenId: Int = 3

    // CoreML models (lazy-loaded)
    private var encoder: MLModel?
    private var decoder: MLModel?

    init(directory: URL? = nil) {
        self.modelsDirectory = directory
    }

    /// `<models root>/kokoro/`: where the vocab and the encoder/decoder
    /// bundles are read from.
    nonisolated func kokoroDirectory() throws -> URL {
        let root = try modelsDirectory ?? TtsCacheDirectory.ensure().appendingPathComponent("Models")
        return root.appendingPathComponent(Repo.kokoro.folderName)
    }

    func phonemize(word: String) throws -> [String]? {
        do {
            try loadIfNeeded()
        } catch {
            if ProcessInfo.processInfo.environment["CI"] != nil {
                logger.warning("G2P unavailable in CI, returning nil for word: \(word)")
                return nil
            }
            throw error
        }

        guard let graphemeToId, let idToPhoneme, let encoder, let decoder else {
            return nil
        }

        // Encode: [BOS] + grapheme IDs + [EOS]
        var inputIds: [Int32] = [Int32(bosTokenId)]
        for ch in word {
            inputIds.append(Int32(graphemeToId[ch] ?? unkTokenId))
        }
        inputIds.append(Int32(eosTokenId))

        let encLen = inputIds.count

        // Build encoder input MLMultiArray
        let encoderInput = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        for i in 0..<encLen {
            encoderInput[[0, i] as [NSNumber]] = NSNumber(value: inputIds[i])
        }

        // Run encoder
        let encoderProvider = try MLDictionaryFeatureProvider(
            dictionary: ["input_ids": MLFeatureValue(multiArray: encoderInput)]
        )
        guard let encoderOutput = try? encoder.prediction(from: encoderProvider),
            let encoderHidden = encoderOutput.featureValue(for: "encoder_hidden_states")?.multiArrayValue
        else {
            throw G2PModelError.encoderPredictionFailed
        }

        // Greedy decode loop
        let maxSteps = 64
        var decoderIds: [Int32] = [Int32(bosTokenId)]

        for _ in 0..<maxSteps {
            let decLen = decoderIds.count

            // decoder_input_ids
            let decInput = try MLMultiArray(shape: [1, NSNumber(value: decLen)], dataType: .int32)
            for i in 0..<decLen {
                decInput[[0, i] as [NSNumber]] = NSNumber(value: decoderIds[i])
            }

            // position_ids (BART offset = 2)
            let posIds = try MLMultiArray(shape: [1, NSNumber(value: decLen)], dataType: .int32)
            for i in 0..<decLen {
                posIds[[0, i] as [NSNumber]] = NSNumber(value: Int32(i + 2))
            }

            // causal_mask: upper triangular with -1e4
            let mask = try MLMultiArray(
                shape: [1, NSNumber(value: decLen), NSNumber(value: decLen)], dataType: .float32)
            for i in 0..<decLen {
                for j in 0..<decLen {
                    let val: Float = j > i ? -1e4 : 0
                    mask[[0, i, j] as [NSNumber]] = NSNumber(value: val)
                }
            }

            let decoderProvider = try MLDictionaryFeatureProvider(
                dictionary: [
                    "decoder_input_ids": MLFeatureValue(multiArray: decInput),
                    "encoder_hidden_states": MLFeatureValue(multiArray: encoderHidden),
                    "position_ids": MLFeatureValue(multiArray: posIds),
                    "causal_mask": MLFeatureValue(multiArray: mask),
                ]
            )

            guard let decoderOutput = try? decoder.prediction(from: decoderProvider),
                let logits = decoderOutput.featureValue(for: "logits")?.multiArrayValue
            else {
                throw G2PModelError.decoderPredictionFailed
            }

            // Argmax of last position's logits
            let vocabSize = logits.shape.last!.intValue
            let lastPos = decLen - 1
            var bestId = 0
            var bestVal: Float = -Float.infinity
            for v in 0..<vocabSize {
                let val = logits[[0, lastPos, v] as [NSNumber]].floatValue
                if val > bestVal {
                    bestVal = val
                    bestId = v
                }
            }

            if bestId == eosTokenId { break }
            decoderIds.append(Int32(bestId))
        }

        // Convert token IDs to phoneme string, skipping special tokens
        let specialTokens: Set<Int> = [0, bosTokenId, eosTokenId, unkTokenId]
        var phonemes: [String] = []
        for id in decoderIds {
            let intId = Int(id)
            if specialTokens.contains(intId) { continue }
            if let ph = idToPhoneme[intId] {
                phonemes.append(ph)
            }
        }

        return phonemes.isEmpty ? nil : phonemes
    }

    /// Verifies that CoreML models and vocab can be loaded.
    func ensureModelsAvailable() throws {
        try loadIfNeeded()
    }

    // MARK: - Private

    private func loadIfNeeded() throws {
        if graphemeToId != nil && encoder != nil && decoder != nil { return }

        let kokoroDir = try kokoroDirectory()

        // Load g2p_vocab.json from cache directory
        let vocabURL = kokoroDir.appendingPathComponent(ModelNames.G2P.vocabularyFile)
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw G2PModelError.vocabLoadFailed("\(ModelNames.G2P.vocabularyFile) not found at \(vocabURL.path)")
        }

        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocab = try JSONSerialization.jsonObject(with: vocabData) as? [String: Any],
            let g2id = vocab["grapheme_to_id"] as? [String: Int],
            let id2ph = vocab["id_to_phoneme"] as? [String: String]
        else {
            throw G2PModelError.vocabLoadFailed("invalid JSON structure")
        }

        var gMap: [Character: Int] = [:]
        for (key, val) in g2id {
            if let ch = key.first, key.count == 1 {
                gMap[ch] = val
            }
        }
        graphemeToId = gMap

        var pMap: [Int: String] = [:]
        for (key, val) in id2ph {
            if let intKey = Int(key) {
                pMap[intKey] = val
            }
        }
        idToPhoneme = pMap

        if let bos = vocab["bos_token_id"] as? Int { bosTokenId = bos }
        if let eos = vocab["eos_token_id"] as? Int { eosTokenId = eos }
        if let unk = vocab["unk_token_id"] as? Int { unkTokenId = unk }

        logger.info("Loaded G2P vocab (\(gMap.count) graphemes, \(pMap.count) phonemes)")

        // Load CoreML models from cache directory
        let encoderURL = kokoroDir.appendingPathComponent(ModelNames.G2P.encoderFile)
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw G2PModelError.modelLoadFailed("\(ModelNames.G2P.encoderFile) not found at \(encoderURL.path)")
        }
        let decoderURL = kokoroDir.appendingPathComponent(ModelNames.G2P.decoderFile)
        guard FileManager.default.fileExists(atPath: decoderURL.path) else {
            throw G2PModelError.modelLoadFailed("\(ModelNames.G2P.decoderFile) not found at \(decoderURL.path)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        encoder = try MLModel(contentsOf: encoderURL, configuration: config)
        decoder = try MLModel(contentsOf: decoderURL, configuration: config)

        logger.info("Loaded G2P CoreML models")
    }
}
#endif
