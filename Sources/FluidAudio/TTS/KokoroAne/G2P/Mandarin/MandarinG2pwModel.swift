#if TTS
@preconcurrency import CoreML
import Foundation

/// Async wrapper around the int8 g2pW CoreML model that fixes
/// polyphone pronunciations using sentence context.
///
/// The model is the upstream g2pW ONNX graph (BERT-base + conditional
/// softmax head) converted to CoreML. Per query it takes the tokenized
/// sentence, the token position of one target Hanzi, that Hanzi's index
/// in the sorted polyphone inventory (`char_ids`) and a 0/1 mask over
/// the global label set (`phoneme_mask`), and returns masked
/// probabilities over the labels. Input preparation mirrors
/// `prepare_onnx_input` in PaddleSpeech's `g2pw/dataset.py`.
///
/// Inference is invoked one target at a time — batching is left for a
/// future iteration once the round-trip CER benchmark shows a
/// throughput-shaped bottleneck. For typical sentence lengths (≤ 60
/// chars, ≤ 4 polyphones) this is comfortably below the speech budget.
///
/// Construction is deliberately tolerant of a missing model: callers
/// can wire the pipeline with `g2pw == nil` and the segmenter falls
/// back to the pinyin-dict path, exactly as it does today.
public actor MandarinG2pwModel {

    private let model: MLModel
    private let tokenizer: MandarinBertTokenizer
    private let catalog: MandarinPolyphoneCatalog
    private let maxLength: Int
    private static let logger = AppLogger(category: "MandarinG2pwModel")

    /// Names of the model's input/output features. These match the
    /// converted CoreML signature shipped at
    /// `kokoro-82m-coreml/ANE-zh/g2pw/g2pw.mlmodelc`.
    enum Feature: String, CaseIterable {
        case inputIds = "input_ids"
        case attentionMask = "attention_mask"
        case tokenTypeIds = "token_type_ids"
        case phonemeMask = "phoneme_mask"
        case charIds = "char_ids"
        case positionIds = "position_ids"
        case probs

        static var inputs: [Feature] {
            allCases.filter { $0 != .probs }
        }
    }

    public enum InferenceError: Swift.Error, LocalizedError {
        case missingOutput(String)
        case shapeMismatch(String)
        case signatureMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .missingOutput(let name):
                return "g2pW model output '\(name)' is missing"
            case .shapeMismatch(let what):
                return "g2pW shape mismatch: \(what)"
            case .signatureMismatch(let what):
                return "g2pW model signature mismatch: \(what)"
            }
        }
    }

    /// Wrap a loaded model. Throws when the model's feature names or
    /// label dimension don't line up with `catalog`, so a stale bundle
    /// is rejected at load time instead of failing on every query. The
    /// sequence length is read from the model's fixed `input_ids` shape;
    /// `maxLength` only applies when the bundle leaves it flexible.
    public init(
        model: MLModel,
        tokenizer: MandarinBertTokenizer,
        catalog: MandarinPolyphoneCatalog,
        maxLength: Int = MandarinBertTokenizer.defaultMaxLength
    ) throws {
        let modelLength = try Self.validate(model: model, labelCount: catalog.labels.count)
        self.model = model
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.maxLength = modelLength ?? maxLength
    }

    /// Check the feature signature; returns the fixed `input_ids`
    /// length when the model declares one.
    private static func validate(model: MLModel, labelCount: Int) throws -> Int? {
        let description = model.modelDescription
        for feature in Feature.inputs {
            guard description.inputDescriptionsByName[feature.rawValue] != nil else {
                throw InferenceError.signatureMismatch("missing input '\(feature.rawValue)'")
            }
        }
        guard let probs = description.outputDescriptionsByName[Feature.probs.rawValue] else {
            throw InferenceError.signatureMismatch("missing output '\(Feature.probs.rawValue)'")
        }
        if let shape = probs.multiArrayConstraint?.shape, let last = shape.last?.intValue,
            last != labelCount
        {
            throw InferenceError.signatureMismatch(
                "model emits \(last) labels but POLYPHONIC_CHARS.txt defines \(labelCount)")
        }
        let inputIds = description.inputDescriptionsByName[Feature.inputIds.rawValue]
        return inputIds?.multiArrayConstraint?.shape.last?.intValue
    }

    /// Build the 0/1 `phoneme_mask` row: 1 at every label index the
    /// target char may take, 0 elsewhere. Out-of-range candidates are
    /// ignored.
    static func phonemeMask(candidates: [Int], labelCount: Int) -> [Float] {
        var mask = [Float](repeating: 0, count: labelCount)
        for idx in candidates where idx >= 0 && idx < labelCount {
            mask[idx] = 1
        }
        return mask
    }

    /// Disambiguate `targets` (positions inside `chars`) using the
    /// supplied sentence context. Returns a sparse map from target
    /// position to the predicted bopomofo label (digit-tone form as
    /// stored in the catalog).
    ///
    /// Targets that aren't polyphonic (per the catalog) are silently
    /// dropped — the caller should already have filtered them, but a
    /// belt-and-braces check keeps the contract honest.
    ///
    /// `dictReadings` gives, per target position, the readings the
    /// caller's dictionary lists, default first. Such a target returns
    /// only a label spelling one of them (see ``MainlandBins``), and is
    /// dropped — the caller keeps its default — when that default wins,
    /// when the inventory does not cover it, or when the scores are not
    /// finite.
    public func disambiguate(
        chars: [Character],
        targets: [Int],
        dictReadings: [Int: [MandarinPinyinNormalizer.Syllable]] = [:]
    ) async throws -> [Int: String] {
        guard !targets.isEmpty else { return [:] }

        let encoded = tokenizer.encode(chars: chars, maxLength: maxLength)
        // Sentence-level inputs are shared by every target.
        let length = encoded.inputIds.count
        let sentence = SentenceInputs(
            inputIds: try Self.makeArray(encoded.inputIds, shape: [1, length], dataType: .int32),
            attentionMask: try Self.makeArray(
                encoded.attentionMask, shape: [1, length], dataType: .int32),
            tokenTypeIds: try Self.makeArray(
                encoded.tokenTypeIds, shape: [1, length], dataType: .int32))
        var output: [Int: String] = [:]
        for charIdx in targets {
            guard charIdx >= 0, charIdx < chars.count else { continue }
            guard charIdx < encoded.tokenPositionForChar.count else {
                // Truncated past this target — fall back silently.
                continue
            }
            let ch = chars[charIdx]
            guard let candidates = catalog.candidates(for: ch),
                !candidates.isEmpty,
                let charId = catalog.charIndex[ch]
            else { continue }
            var bins: MainlandBins?
            if let readings = dictReadings[charIdx] {
                guard let mainland = MainlandBins(char: ch, readings: readings, catalog: catalog) else { continue }
                bins = mainland
            }

            let tokenPos = encoded.tokenPositionForChar[charIdx]
            let probs = try await runOne(
                sentence: sentence,
                phonemeMask: Self.phonemeMask(
                    candidates: candidates, labelCount: catalog.labels.count),
                charId: charId,
                targetPosition: tokenPos
            )
            if let bins {
                if let pick = bins.pick(probs) { output[charIdx] = pick }
                continue
            }
            // Pick the argmax over the candidate subset only — the model
            // already applies the mask, this just guards against a
            // bundle that was exported without it.
            var bestIdx = candidates[0]
            var bestProb = -Float.infinity
            for cand in candidates {
                guard cand >= 0, cand < probs.count else { continue }
                if probs[cand] > bestProb {
                    bestProb = probs[cand]
                    bestIdx = cand
                }
            }
            if let bopo = catalog.bopomofo(forLabel: bestIdx) {
                output[charIdx] = bopo
            }
        }
        return output
    }

    /// Comparable form of one reading: its v1.1-zh vocab encoding, so a
    /// catalog label (`ㄏㄨㄛ5`) and a dict reading (`huo`) that spell the
    /// same syllable compare equal.
    static func readingKey(_ syllable: MandarinPinyinNormalizer.Syllable) -> String? {
        MandarinBopomofoMap.encode(syllable: syllable.base, tone: syllable.tone)
    }

    /// Catalog labels that are the Taiwan reading of a sense the mainland
    /// dict reads with a reading other than its default.
    static let taiwanReadings: [Character: [String: String]] = [
        "得": ["ㄉㄞ3": "ㄉㄟ3"]  // dǎi, "must": mainland děi
    ]

    /// Which mainland dict reading each catalog label of a character counts
    /// toward. The inventory is Taiwan-standard: a label spelling a dict
    /// reading (or its neutral-tone form) counts for itself; one the dict
    /// lacks is mostly the Taiwan reading of a sense the dict reads by
    /// default (`和` hàn is mainland hé, `削` xuè is xuē) and counts toward
    /// the default, unless ``taiwanReadings`` says otherwise.
    struct MainlandBins {
        /// Catalog label index → reading key it counts toward.
        let bin: [Int: String]
        /// Reading key → the label to return for it.
        let label: [String: String]
        let defaultKey: String

        /// `nil` when no label spells the default reading in any tone
        /// (the inventory has a different character: Traditional 吨 is not
        /// the simplified 吨, "ton") or when every label counts toward the
        /// default, so g2pW has nothing to choose.
        init?(char: Character, readings: [MandarinPinyinNormalizer.Syllable], catalog: MandarinPolyphoneCatalog) {
            guard let first = readings.first, let defaultKey = MandarinG2pwModel.readingKey(first),
                let candidates = catalog.candidates(for: char)
            else { return nil }
            let neutral = readings.map { MandarinPinyinNormalizer.Syllable(base: $0.base, tone: 5) }
            let dictKeys = Set((readings + neutral).compactMap(MandarinG2pwModel.readingKey))
            var bin: [Int: String] = [:]
            var label: [String: String] = [:]
            var coversDefault = false
            for index in candidates {
                guard let text = catalog.bopomofo(forLabel: index) else { continue }
                let syllable = MandarinBopomofoMap.decode(bopomofo: text)
                let key = syllable.flatMap(MandarinG2pwModel.readingKey)
                if let key, dictKeys.contains(key) {
                    bin[index] = key
                    label[key] = text
                } else if let alias = MandarinG2pwModel.taiwanReadings[char]?[text],
                    let aliasKey = MandarinBopomofoMap.decode(bopomofo: alias).flatMap(MandarinG2pwModel.readingKey)
                {
                    bin[index] = aliasKey
                    label[aliasKey] = label[aliasKey] ?? alias
                } else {
                    bin[index] = defaultKey
                }
                if key == defaultKey || syllable?.base == first.base { coversDefault = true }
            }
            guard coversDefault, Set(bin.values).count >= 2 else { return nil }
            self.bin = bin
            self.label = label
            self.defaultKey = defaultKey
        }

        /// The label of the reading with the most probability, or `nil`
        /// when that is the default (the caller already has it) or no
        /// score is finite.
        func pick(_ probs: [Float]) -> String? {
            var score: [String: Float] = [:]
            for (index, key) in bin where index >= 0 && index < probs.count && probs[index].isFinite {
                score[key, default: 0] += probs[index]
            }
            guard let best = score.max(by: { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) })
            else { return nil }
            return best.key == defaultKey ? nil : label[best.key]
        }
    }

    private struct SentenceInputs {
        let inputIds: MLMultiArray
        let attentionMask: MLMultiArray
        let tokenTypeIds: MLMultiArray
    }

    /// Pack the per-target inputs, run the model, and pull the
    /// probability row out as a `[Float]`.
    private func runOne(
        sentence: SentenceInputs,
        phonemeMask: [Float],
        charId: Int,
        targetPosition: Int
    ) async throws -> [Float] {
        let maskArr = try Self.makeArray(
            phonemeMask, shape: [1, phonemeMask.count], dataType: .float32)
        let charArr = try Self.makeArray([Int32(charId)], shape: [1], dataType: .int32)
        let positionArr = try Self.makeArray(
            [Int32(targetPosition)], shape: [1], dataType: .int32)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds.rawValue: sentence.inputIds,
            Feature.attentionMask.rawValue: sentence.attentionMask,
            Feature.tokenTypeIds.rawValue: sentence.tokenTypeIds,
            Feature.phonemeMask.rawValue: maskArr,
            Feature.charIds.rawValue: charArr,
            Feature.positionIds.rawValue: positionArr,
        ])
        let result = try await model.prediction(from: provider)
        guard let probsArr = result.featureValue(for: Feature.probs.rawValue)?.multiArrayValue
        else {
            throw InferenceError.missingOutput(Feature.probs.rawValue)
        }
        return Self.flatten(probsArr)
    }

    /// Dense copy of `values` into a fresh MLMultiArray of `shape`.
    /// `T` must match `dataType` (`Int32` ↔ `.int32`, `Float` ↔ `.float32`).
    private static func makeArray<T>(
        _ values: [T], shape: [Int], dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw InferenceError.shapeMismatch(
                "expected \(count) elements for shape \(shape), got \(values.count)")
        }
        let arr = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        guard count > 0 else { return arr }
        let ptr = arr.dataPointer.bindMemory(to: T.self, capacity: count)
        values.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            ptr.update(from: base, count: count)
        }
        return arr
    }

    private static func flatten(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = Array(repeating: Float(0), count: count)
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { out[i] = ptr[i] }
        case .float16:
            // Fallback: walk indexes via subscripting. This keeps the
            // wrapper independent of the float16 SIMD path that varies
            // by deployment target.
            for i in 0..<count { out[i] = array[i].floatValue }
        case .double:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(ptr[i]) }
        default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }
}
#endif
