import Foundation

/// Reuses decoded batch windows across repeated `transcribe` calls on one growing recording.
///
/// An app that re-transcribes a recording while it grows (live previews, then a final pass at Stop) can
/// pass the same context to each `AsrManager.transcribe(_:decoderState:language:reuse:)` call. A window
/// whose audio and decoding parameters are byte-identical to one decoded earlier with the same models
/// returns that earlier result instead of running inference again. Window selection, merging and gap
/// repair still run over the full recording on every call, and feature tensors, decoder state and
/// vocabulary rescoring are never reused.
///
/// Create one context per recording and drop it when the recording ends. Only audio longer than one
/// model window (about 15 s) is decoded in windows; shorter calls do not use the context. The context
/// keeps at most `maximumEntries` windows, clamped to 1...32.
public final class BatchWindowReuseContext: Sendable {
    /// Window lookups since the context was created.
    public struct Statistics: Sendable, Equatable {
        /// Lookups answered from an earlier decode.
        public let hits: Int
        /// Lookups that ran inference.
        public let misses: Int
        /// Windows currently kept.
        public let entries: Int
        /// Bytes of audio kept for exact matching.
        public let pcmBytes: Int
    }

    let cache: BatchWindowReuseCache

    public init(maximumEntries: Int = 16) {
        cache = BatchWindowReuseCache(maximumEntries: maximumEntries)
    }

    public func statistics() async -> Statistics {
        await cache.statistics()
    }
}

/// Exact batch windows decoded for one recording. No feature tensors or decoder state cross requests.
internal actor BatchWindowReuseCache {
    internal struct Key: Hashable, Sendable {
        /// The loaded encoder (`AsrManager.loadedEncoderIdentity`), so a context passed to another
        /// model set never matches.
        let model: ObjectIdentifier
        let pcm: Data
        let contextSamples: Int
        let chunkStart: Int
        let isLastChunk: Bool
        let flushPendingTokens: Bool?
        let maxModelSamples: Int
        let language: String?
        let emitTokensAfterFrame: Int?
        let initialTimeIndex: Int?

        init(
            model: ObjectIdentifier, samples: [Float], contextSamples: Int, chunkStart: Int, isLastChunk: Bool,
            flushPendingTokens: Bool?, maxModelSamples: Int, language: Language?,
            emitTokensAfterFrame: Int?, initialTimeIndex: Int?
        ) {
            self.model = model
            self.pcm = samples.withUnsafeBytes { Data($0) }
            self.contextSamples = contextSamples
            self.chunkStart = chunkStart
            self.isLastChunk = isLastChunk
            self.flushPendingTokens = flushPendingTokens
            self.maxModelSamples = maxModelSamples
            self.language = language?.rawValue
            self.emitTokensAfterFrame = emitTokensAfterFrame
            self.initialTimeIndex = initialTimeIndex
        }
    }

    internal struct Value: Sendable {
        let tokens: [Int]
        let timestamps: [Int]
        let confidences: [Float]
        let durations: [Int]
    }

    private let maximumEntries: Int
    private var values: [Key: Value] = [:]
    private var recency: [Key] = []
    private var hits = 0
    private var misses = 0

    init(maximumEntries: Int) { self.maximumEntries = max(1, min(32, maximumEntries)) }

    func value(for key: Key) -> Value? {
        guard let value = values[key] else {
            misses += 1
            return nil
        }
        hits += 1
        recency.removeAll { $0 == key }
        recency.append(key)
        return value
    }

    func store(_ value: Value, for key: Key) {
        values[key] = value
        recency.removeAll { $0 == key }
        recency.append(key)
        while recency.count > maximumEntries {
            let retired = recency.removeFirst()
            values.removeValue(forKey: retired)
        }
    }

    func statistics() -> BatchWindowReuseContext.Statistics {
        .init(hits: hits, misses: misses, entries: values.count, pcmBytes: values.keys.reduce(0) { $0 + $1.pcm.count })
    }
}
