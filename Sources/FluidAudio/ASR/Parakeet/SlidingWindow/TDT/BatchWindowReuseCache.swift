import Foundation

/// Recording-owned exact batch requests. No feature tensors or decoder state cross requests.
internal actor BatchWindowReuseCache {
    internal struct Key: Hashable, Sendable {
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
            samples: [Float], contextSamples: Int, chunkStart: Int, isLastChunk: Bool,
            flushPendingTokens: Bool?, maxModelSamples: Int, language: Language?,
            emitTokensAfterFrame: Int?, initialTimeIndex: Int?
        ) {
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
    private var active = true
    private var hits = 0
    private var misses = 0

    init(maximumEntries: Int) { self.maximumEntries = max(1, min(32, maximumEntries)) }

    func value(for key: Key) -> Value? {
        guard active else { return nil }
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
        guard active else { return }
        values[key] = value
        recency.removeAll { $0 == key }
        recency.append(key)
        while recency.count > maximumEntries {
            let retired = recency.removeFirst()
            values.removeValue(forKey: retired)
        }
    }

    func invalidate() {
        active = false
        values.removeAll()
        recency.removeAll()
    }

    func statistics() -> (hits: Int, misses: Int, entries: Int, pcmBytes: Int) {
        (hits, misses, values.count, values.keys.reduce(0) { $0 + $1.pcm.count })
    }
}

extension AsrManager {
    /// Begin an isolated experimental cache for one recording, before preview inference.
    /// Only byte-identical stateless batch windows with matching inference parameters are reused.
    /// Final window selection, merging and gap repair still run. Feature tensors, decoder state
    /// and vocabulary rescoring are never reused. The cache is bounded to at most 32 entries.
    /// Serialize recording lifecycle operations with transcription; finish or cancel outstanding
    /// requests before beginning another recording or changing models.
    public func beginBatchWindowReuseExperiment(maximumEntries: Int = 16) async {
        await endBatchWindowReuseExperiment()
        experimentalBatchWindowCache = BatchWindowReuseCache(maximumEntries: maximumEntries)
    }

    /// Detach recording-owned cached requests after completion or cancellation.
    public func endBatchWindowReuseExperiment() async {
        let cache = experimentalBatchWindowCache
        experimentalBatchWindowCache = nil
        await cache?.invalidate()
    }

    /// Return request-level reuse diagnostics; no feature or decoder state is exposed.
    public func batchWindowReuseExperimentStatistics() async -> (hits: Int, misses: Int, entries: Int, pcmBytes: Int) {
        guard let cache = experimentalBatchWindowCache else { return (0, 0, 0, 0) }
        return await cache.statistics()
    }

    internal func detachBatchWindowReuseExperiment() {
        let cache = experimentalBatchWindowCache
        experimentalBatchWindowCache = nil
        Task { await cache?.invalidate() }
    }

    internal func shareBatchWindowReuseExperiment(_ cache: BatchWindowReuseCache?) {
        experimentalBatchWindowCache = cache
    }
}
