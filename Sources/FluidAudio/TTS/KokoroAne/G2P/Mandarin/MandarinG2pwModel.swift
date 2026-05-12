@preconcurrency import CoreML
import Foundation

/// Async wrapper around the int8 g2pW CoreML model that fixes
/// polyphone pronunciations using sentence context.
///
/// The model is a BERT-base classifier that, given a tokenized sentence
/// + the position of a single target Hanzi, outputs softmax logits over
/// the global polyphone label set (~700 classes). The runtime applies
/// the per-target phoneme mask from `MandarinPolyphoneCatalog` so the
/// argmax can only land on a pronunciation valid for that character.
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
    private enum Feature: String {
        case inputIds = "input_ids"
        case attentionMask = "attention_mask"
        case tokenTypeIds = "token_type_ids"
        case targetPosition = "target_position"
        case logits
    }

    public enum InferenceError: Swift.Error, LocalizedError {
        case missingOutput(String)
        case shapeMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .missingOutput(let name):
                return "g2pW model output '\(name)' is missing"
            case .shapeMismatch(let what):
                return "g2pW shape mismatch: \(what)"
            }
        }
    }

    public init(
        model: MLModel,
        tokenizer: MandarinBertTokenizer,
        catalog: MandarinPolyphoneCatalog,
        maxLength: Int = MandarinBertTokenizer.defaultMaxLength
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.catalog = catalog
        self.maxLength = maxLength
    }

    /// Disambiguate `targets` (positions inside `chars`) using the
    /// supplied sentence context. Returns a sparse map from target
    /// position to the predicted bopomofo string.
    ///
    /// Targets that aren't polyphonic (per the catalog) are silently
    /// dropped — the caller should already have filtered them, but a
    /// belt-and-braces check keeps the contract honest.
    public func disambiguate(
        chars: [Character],
        targets: [Int]
    ) async throws -> [Int: String] {
        guard !targets.isEmpty else { return [:] }

        let encoded = tokenizer.encode(chars: chars, maxLength: maxLength)
        var output: [Int: String] = [:]
        for charIdx in targets {
            guard charIdx >= 0, charIdx < chars.count else { continue }
            guard charIdx < encoded.tokenPositionForChar.count else {
                // Truncated past this target — fall back silently.
                continue
            }
            let ch = chars[charIdx]
            guard let candidates = catalog.candidates(for: ch),
                !candidates.isEmpty
            else { continue }

            let tokenPos = encoded.tokenPositionForChar[charIdx]
            let logits = try await runOne(
                inputIds: encoded.inputIds,
                attentionMask: encoded.attentionMask,
                tokenTypeIds: encoded.tokenTypeIds,
                targetPosition: tokenPos
            )
            // Pick the argmax over the candidate subset only — the rest
            // of the softmax is masked out as if `-inf`.
            var bestIdx = candidates[0]
            var bestProb = -Float.infinity
            for cand in candidates {
                guard cand >= 0, cand < logits.count else { continue }
                if logits[cand] > bestProb {
                    bestProb = logits[cand]
                    bestIdx = cand
                }
            }
            if let bopo = catalog.bopomofo(forLabel: bestIdx) {
                output[charIdx] = bopo
            }
        }
        return output
    }

    /// Pack the inputs into MLMultiArrays, run the model, and pull the
    /// (target_position-row of the) logits out as a `[Float]`.
    private func runOne(
        inputIds: [Int32],
        attentionMask: [Int32],
        tokenTypeIds: [Int32],
        targetPosition: Int
    ) async throws -> [Float] {
        let length = inputIds.count
        let inputIdsArr = try makeInt32Array(inputIds, shape: [1, length])
        let attentionArr = try makeInt32Array(attentionMask, shape: [1, length])
        let tokenTypeArr = try makeInt32Array(tokenTypeIds, shape: [1, length])
        let positionArr = try makeInt32Array([Int32(targetPosition)], shape: [1])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds.rawValue: inputIdsArr,
            Feature.attentionMask.rawValue: attentionArr,
            Feature.tokenTypeIds.rawValue: tokenTypeArr,
            Feature.targetPosition.rawValue: positionArr,
        ])
        let result = try await model.prediction(from: provider)
        guard let logitsArr = result.featureValue(for: Feature.logits.rawValue)?.multiArrayValue
        else {
            throw InferenceError.missingOutput(Feature.logits.rawValue)
        }
        return Self.flatten(logitsArr)
    }

    private func makeInt32Array(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw InferenceError.shapeMismatch(
                "expected \(count) elements for shape \(shape), got \(values.count)")
        }
        // Direct memcpy via the typed pointer; faster than per-index
        // assignment for the 512-element common case.
        let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
        values.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: count)
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
