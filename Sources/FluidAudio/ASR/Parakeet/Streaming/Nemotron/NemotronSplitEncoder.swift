@preconcurrency import CoreML
import Foundation

/// The multilingual streaming encoder: the shipped single model, or the
/// opt-in ``NemotronSplitEncoder`` chain behind the same inputs and outputs.
enum NemotronMultilingualEncoder: Sendable {
    case single(MLModel)
    case split(NemotronSplitEncoder)

    /// The single model, for paths that need `MLModel` itself (stateful
    /// caches, output backings). `nil` for the split encoder.
    var singleModel: MLModel? {
        if case .single(let model) = self { return model }
        return nil
    }

    /// A fresh zero-initialized `MLState` when the single model keeps its
    /// caches as state (macOS 15 / iOS 18+). `nil` otherwise, including for
    /// the split encoder, so assigning it also clears a stale state.
    func makeStateIfStateful() -> Any? {
        guard #available(macOS 15, iOS 18, tvOS 18, *), case .single(let model) = self,
            !model.modelDescription.stateDescriptionsByName.isEmpty
        else { return nil }
        return model.makeState()
    }

    func prediction(from input: MLFeatureProvider) async throws -> MLFeatureProvider {
        switch self {
        case .single(let model): return try await model.prediction(from: input)
        case .split(let chain): return try await chain.prediction(from: input)
        }
    }

    func prediction(from input: MLFeatureProvider, options: MLPredictionOptions) async throws -> MLFeatureProvider {
        switch self {
        case .single(let model): return try await model.prediction(from: input, options: options)
        case .split(let chain): return try await chain.prediction(from: input)
        }
    }
}

/// Opt-in split of the 2240 ms multilingual encoder (#1116): the conv
/// subsampling front end, layers 0-11 and layers 12-23 as three models.
///
/// On some older Neural Engines (M3 Pro) the single 2240 ms encoder fails
/// ANE register allocation, so Core ML runs all of it on the CPU (~4 min
/// first load, ~2 GB resident). The front end's activations are what does
/// not fit; split out, the pieces are expected to fit (the reporter's own
/// split did). The weights are the shipped encoder's, and on M5 Pro the
/// chained outputs are bit-identical to it. Not yet verified on M3 Pro.
///
/// Takes and returns the single encoder's tensors (`mel`, `mel_length`,
/// 24-layer `cache_channel` / `cache_time`, `cache_len`, `prompt_id` in;
/// `encoded`, `encoded_length`, `cache_*_out` out), so callers keep one
/// 24-layer cache layout. It emits no `encoder_proj`.
final class NemotronSplitEncoder: Sendable {
    static let frontendFile = "encoder_frontend.mlmodelc"
    static let lowerLayersFile = "encoder_body_0_11.mlmodelc"
    static let upperLayersFile = "encoder_body_12_23.mlmodelc"
    static let fileNames = [frontendFile, lowerLayersFile, upperLayersFile]
    /// The only tier the split is published for.
    static let chunkMs = 2240
    /// Repo folder holding the pieces. Kept outside `<model>/<tier>ms/` so
    /// default variant downloads never fetch them.
    static let subdirectory = "encoder-split/\(chunkMs)ms"

    private let frontend: MLModel
    private let lowerLayers: MLModel
    private let upperLayers: MLModel

    init(frontend: MLModel, lowerLayers: MLModel, upperLayers: MLModel) {
        self.frontend = frontend
        self.lowerLayers = lowerLayers
        self.upperLayers = upperLayers
    }

    /// `true` when every piece under `directory` is a complete compiled
    /// bundle. Rejects the `*.partial` files an interrupted download leaves,
    /// which a bare existence check would accept forever (#819).
    static func isCached(at directory: URL) -> Bool {
        ModelCache.isCacheComplete(at: directory, requiredFiles: Set(fileNames))
    }

    /// Load the three pieces from `directory` (see ``fileNames``).
    static func load(from directory: URL, configuration: MLModelConfiguration) async throws -> NemotronSplitEncoder {
        func open(_ name: String) async throws -> MLModel {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ASRError.processingFailed("Split encoder piece \(name) not found in \(directory.path)")
            }
            return try await MLModel.load(contentsOf: url, configuration: configuration)
        }
        return NemotronSplitEncoder(
            frontend: try await open(frontendFile),
            lowerLayers: try await open(lowerLayersFile),
            upperLayers: try await open(upperLayersFile))
    }

    /// Throws unless the pieces' fixed inputs match the variant's mel and
    /// cache shapes, so a mismatched model directory fails at load instead
    /// of on the first chunk.
    func validate(for config: NemotronMultilingualStreamingConfig) throws {
        func shape(_ model: MLModel, _ input: String) -> [Int] {
            model.modelDescription.inputDescriptionsByName[input]?.multiArrayConstraint?.shape.map(\.intValue) ?? []
        }
        func half(_ cacheShape: [Int]) -> [Int] {
            cacheShape.count == 4 ? [cacheShape[0], cacheShape[1] / 2, cacheShape[2], cacheShape[3]] : cacheShape
        }
        let mel = shape(frontend, "mel")
        let channel = shape(lowerLayers, "cache_channel")
        let time = shape(lowerLayers, "cache_time")
        guard mel == [1, config.melFeatures, config.totalMelFrames],
            channel == half(config.cacheChannelShape), time == half(config.cacheTimeShape),
            shape(upperLayers, "cache_channel") == channel, shape(upperLayers, "cache_time") == time
        else {
            throw ASRError.processingFailed(
                "The split encoder (mel \(mel), per-half caches \(channel) / \(time)) does not match this "
                    + "model's metadata (mel [1, \(config.melFeatures), \(config.totalMelFrames)], "
                    + "caches \(config.cacheChannelShape) / \(config.cacheTimeShape))")
        }
    }

    func prediction(from input: MLFeatureProvider) async throws -> MLFeatureProvider {
        func array(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
            guard let value = provider.featureValue(for: name)?.multiArrayValue else {
                throw ASRError.processingFailed("Split encoder: missing \(name)")
            }
            return value
        }
        let cacheChannel = try array(input, "cache_channel")
        let cacheTime = try array(input, "cache_time")
        let cacheLen = try array(input, "cache_len")
        let promptId = try array(input, "prompt_id")
        let half = cacheChannel.shape[1].intValue / 2

        let front = try await frontend.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "mel": try array(input, "mel"), "mel_length": try array(input, "mel_length"),
            ]))
        let preEncoded = try array(front, "pre_encoded")
        let preEncodedLength = try array(front, "pre_encoded_length")

        let lower = try await lowerLayers.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "pre_encoded": preEncoded, "pre_encoded_length": preEncodedLength,
                "cache_channel": try Self.layers(cacheChannel, 0..<half),
                "cache_time": try Self.layers(cacheTime, 0..<half),
                "cache_len": cacheLen,
            ]))
        let upper = try await upperLayers.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "hidden": try array(lower, "hidden"), "pre_encoded_length": preEncodedLength,
                "cache_channel": try Self.layers(cacheChannel, half..<(2 * half)),
                "cache_time": try Self.layers(cacheTime, half..<(2 * half)),
                "cache_len": cacheLen, "prompt_id": promptId,
            ]))

        let channelOut = try Self.stackLayers(
            try array(lower, "cache_channel_out"), try array(upper, "cache_channel_out"))
        let timeOut = try Self.stackLayers(try array(lower, "cache_time_out"), try array(upper, "cache_time_out"))
        return try MLDictionaryFeatureProvider(dictionary: [
            "encoded": try array(upper, "encoded"),
            "encoded_length": try array(upper, "encoded_length"),
            "cache_channel_out": channelOut,
            "cache_time_out": timeOut,
            "cache_len_out": try array(upper, "cache_len_out"),
        ])
    }

    /// Copy layers `range` (axis 1) of a `[1, L, A, B]` fp32 cache into a new contiguous array.
    static func layers(_ cache: MLMultiArray, _ range: Range<Int>) throws -> MLMultiArray {
        let shape = cache.shape.map(\.intValue)
        guard shape.count == 4, cache.dataType == .float32, range.upperBound <= shape[1] else {
            throw ASRError.processingFailed("Split encoder: unexpected cache layout \(shape)")
        }
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: range.count), cache.shape[2], cache.shape[3]], dataType: .float32)
        let rows = shape[2]
        let width = shape[3]
        let srcStrides = cache.strides.map(\.intValue)
        let dstStrides = result.strides.map(\.intValue)
        let src = cache.dataPointer.assumingMemoryBound(to: Float.self)
        let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
        // Row-wise copy that honours the source's strides (Core ML outputs may be padded).
        for (index, layer) in range.enumerated() {
            for row in 0..<rows {
                let s = src + layer * srcStrides[1] + row * srcStrides[2]
                let d = dst + index * dstStrides[1] + row * dstStrides[2]
                if srcStrides[3] == 1 {
                    d.update(from: s, count: width)
                } else {
                    for column in 0..<width { d[column] = s[column * srcStrides[3]] }
                }
            }
        }
        return result
    }

    /// Stack two `[1, L, A, B]` fp32 caches along axis 1.
    static func stackLayers(_ first: MLMultiArray, _ second: MLMultiArray) throws -> MLMultiArray {
        let a = first.shape.map(\.intValue)
        let b = second.shape.map(\.intValue)
        guard a.count == 4, b.count == 4, a[0] == b[0], a[2] == b[2], a[3] == b[3],
            first.dataType == .float32, second.dataType == .float32
        else {
            throw ASRError.processingFailed("Split encoder: cannot stack caches \(a) and \(b)")
        }
        return MLMultiArray(concatenating: [first, second], axis: 1, dataType: .float32)
    }
}
