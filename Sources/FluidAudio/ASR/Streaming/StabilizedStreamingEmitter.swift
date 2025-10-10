import Foundation

@available(macOS 13.0, iOS 16.0, *)
struct StabilizedTokenLatency: Sendable, Equatable {
    let tokenIndex: Int
    let latencyMs: Int
}

@available(macOS 13.0, iOS 16.0, *)
struct StabilizedUpdateResult: Sendable, Equatable {
    /// Newly committed token IDs (delta since last update).
    let committedTokens: [Int]
    /// If this update produced the first committed tokens, capture latency.
    let firstCommitLatencyMs: Int?
    /// Per-token stabilization latency for the committed tokens.
    let tokenLatencies: [StabilizedTokenLatency]
    /// Count of stable tokens withheld due to boundary rules.
    let withheldStableTokenCount: Int
    /// Debug records describing the state transition (empty when debug disabled).
    let debugRecords: [StabilizerDebugRecord]
}

@available(macOS 13.0, iOS 16.0, *)
struct StabilizerDebugRecord: Sendable, Equatable {
    let timestampMs: Int
    let latestHypothesis: [Int]
    let stablePrefix: [Int]
    let committed: [Int]
}

@available(macOS 13.0, iOS 16.0, *)
final class StabilizedStreamingEmitter {

    typealias TokenDecoder = (Int) -> String?

    private struct StreamState {
        var ringBuffer: [[Int]] = []
        var committed: [Int] = []
        var lastHypothesis: [Int] = []
        var firstSeen: [Int: Int] = [:]
        var startTimestampMs: Int?
        var firstCommitLatencyMs: Int?
        var debugRecords: [StabilizerDebugRecord] = []
    }

    private struct InternalConfig {
        let windowSize: Int
        let emitWordBoundaries: Bool
        let maxWaitMilliseconds: Int
        let enableDebugDump: Bool
        let tokenizerKind: StreamingStabilizerConfig.TokenizerKind
    }

    private let config: InternalConfig
    private var decodeToken: TokenDecoder
    private var states: [Int: StreamState] = [:]

    init(config: StreamingStabilizerConfig, tokenDecoder: @escaping TokenDecoder) {
        self.config = InternalConfig(
            windowSize: config.sanitizedWindowSize,
            emitWordBoundaries: config.emitWordBoundaries,
            maxWaitMilliseconds: config.sanitizedMaxWait,
            enableDebugDump: config.debugDumpEnabled,
            tokenizerKind: config.tokenizerKind
        )
        self.decodeToken = tokenDecoder
    }

    func update(uid: Int, tokenIds: [Int], nowMs: Int) -> StabilizedUpdateResult {
        var state = states[uid, default: StreamState()]
        if state.startTimestampMs == nil {
            state.startTimestampMs = nowMs
        }

        updateRingBuffer(&state, with: tokenIds)
        updateFirstSeen(&state, with: tokenIds, nowMs: nowMs)

        let stablePrefix = longestCommonPrefix(for: state.ringBuffer)
        let hasSufficientHistory = state.ringBuffer.count >= config.windowSize
        let committedCount = state.committed.count
        var commitCount = hasSufficientHistory ? stablePrefix.count : committedCount
        var withheldCount = hasSufficientHistory ? max(0, commitCount - committedCount) : 0

        if commitCount > committedCount {
            commitCount = determineCommitCount(
                currentCommitted: committedCount,
                stablePrefix: stablePrefix,
                state: state,
                nowMs: nowMs
            )
            withheldCount = max(0, stablePrefix.count - commitCount)
        }

        let newlyCommitted: [Int]
        var latencyMeasurements: [StabilizedTokenLatency] = []
        var firstCommitLatency: Int? = nil

        if commitCount > committedCount {
            newlyCommitted = Array(stablePrefix[committedCount..<commitCount])
            latencyMeasurements = recordLatencies(
                state: state,
                committedRange: committedCount..<commitCount,
                nowMs: nowMs
            )
            if state.firstCommitLatencyMs == nil {
                firstCommitLatency = nowMs
                state.firstCommitLatencyMs = nowMs
            }
        } else {
            newlyCommitted = []
        }

        if commitCount > state.committed.count {
            state.committed = Array(stablePrefix.prefix(commitCount))
        }

        state.lastHypothesis = tokenIds

        var debugRecords: [StabilizerDebugRecord] = []
        if config.enableDebugDump {
            let record = StabilizerDebugRecord(
                timestampMs: nowMs,
                latestHypothesis: tokenIds,
                stablePrefix: stablePrefix,
                committed: state.committed
            )
            state.debugRecords.append(record)
            debugRecords = [record]
        }

        states[uid] = state

        return StabilizedUpdateResult(
            committedTokens: newlyCommitted,
            firstCommitLatencyMs: firstCommitLatency,
            tokenLatencies: latencyMeasurements,
            withheldStableTokenCount: withheldCount,
            debugRecords: debugRecords
        )
    }

    func flush(uid: Int, nowMs: Int) -> StabilizedUpdateResult {
        guard var state = states[uid] else {
            return StabilizedUpdateResult(
                committedTokens: [],
                firstCommitLatencyMs: nil,
                tokenLatencies: [],
                withheldStableTokenCount: 0,
                debugRecords: []
            )
        }

        let totalCount = state.lastHypothesis.count
        let committedCount = state.committed.count
        guard totalCount > committedCount else {
            let records = config.enableDebugDump ? state.debugRecords : []
            return StabilizedUpdateResult(
                committedTokens: [],
                firstCommitLatencyMs: nil,
                tokenLatencies: [],
                withheldStableTokenCount: 0,
                debugRecords: records
            )
        }

        let newlyCommitted = Array(state.lastHypothesis[committedCount..<totalCount])
        let latencies = recordLatencies(
            state: state,
            committedRange: committedCount..<totalCount,
            nowMs: nowMs
        )

        var firstCommitLatency: Int?
        if !newlyCommitted.isEmpty, state.firstCommitLatencyMs == nil {
            firstCommitLatency = nowMs
            state.firstCommitLatencyMs = nowMs
        }

        state.committed = state.lastHypothesis

        var debugRecords: [StabilizerDebugRecord] = []
        if config.enableDebugDump {
            let record = StabilizerDebugRecord(
                timestampMs: nowMs,
                latestHypothesis: state.lastHypothesis,
                stablePrefix: state.lastHypothesis,
                committed: state.committed
            )
            state.debugRecords.append(record)
            debugRecords = [record]
        }
        states[uid] = state

        return StabilizedUpdateResult(
            committedTokens: newlyCommitted,
            firstCommitLatencyMs: firstCommitLatency,
            tokenLatencies: latencies,
            withheldStableTokenCount: 0,
            debugRecords: debugRecords
        )
    }

    func reset(uid: Int) {
        cleanupState(for: uid)
    }

    func cleanupState(for uid: Int) {
        states.removeValue(forKey: uid)
    }

    func drainDebugRecords(for uid: Int) -> [StabilizerDebugRecord] {
        guard config.enableDebugDump, var state = states[uid] else {
            return []
        }
        let records = state.debugRecords
        state.debugRecords.removeAll(keepingCapacity: false)
        states[uid] = state
        return records
    }

    func discardCommittedPrefix(uid: Int, count: Int) {
        guard count > 0, var state = states[uid] else { return }

        state.committed = dropPrefix(state.committed, by: count)
        state.lastHypothesis = dropPrefix(state.lastHypothesis, by: count)

        if !state.ringBuffer.isEmpty {
            state.ringBuffer = state.ringBuffer
                .map { dropPrefix($0, by: count) }
                .filter { !$0.isEmpty }
        }

        if !state.firstSeen.isEmpty {
            var shifted: [Int: Int] = [:]
            for (index, timestamp) in state.firstSeen {
                let newIndex = index - count
                guard newIndex >= 0 else { continue }
                shifted[newIndex] = timestamp
            }
            state.firstSeen = shifted
            if !state.lastHypothesis.isEmpty {
                state.firstSeen.keys
                    .filter { $0 >= state.lastHypothesis.count }
                    .forEach { state.firstSeen.removeValue(forKey: $0) }
            }
        }

        if config.enableDebugDump, !state.debugRecords.isEmpty {
            state.debugRecords = state.debugRecords.map { record in
                StabilizerDebugRecord(
                    timestampMs: record.timestampMs,
                    latestHypothesis: dropPrefix(record.latestHypothesis, by: count),
                    stablePrefix: dropPrefix(record.stablePrefix, by: count),
                    committed: dropPrefix(record.committed, by: count)
                )
            }
        }

        states[uid] = state
    }

    func updateTokenDecoder(_ decoder: @escaping TokenDecoder) {
        decodeToken = decoder
    }

    private func updateRingBuffer(_ state: inout StreamState, with tokens: [Int]) {
        state.ringBuffer.append(tokens)
        if state.ringBuffer.count > config.windowSize {
            state.ringBuffer.removeFirst(state.ringBuffer.count - config.windowSize)
        }
    }

    private func updateFirstSeen(_ state: inout StreamState, with tokens: [Int], nowMs: Int) {
        if state.lastHypothesis.isEmpty {
            for index in 0..<tokens.count {
                state.firstSeen[index] = nowMs
            }
        } else {
            let previous = state.lastHypothesis
            let maxCount = max(previous.count, tokens.count)
            for index in 0..<maxCount {
                let previousToken = index < previous.count ? previous[index] : nil
                let currentToken = index < tokens.count ? tokens[index] : nil
                if previousToken != currentToken {
                    if let _ = currentToken {
                        state.firstSeen[index] = nowMs
                    } else {
                        state.firstSeen.removeValue(forKey: index)
                    }
                }
            }
        }
        // Remove trailing entries if new hypothesis shorter.
        state.firstSeen.keys
            .filter { $0 >= tokens.count }
            .forEach { state.firstSeen.removeValue(forKey: $0) }
    }

    private func dropPrefix(_ array: [Int], by count: Int) -> [Int] {
        guard count > 0 else { return array }
        guard count < array.count else { return [] }
        return Array(array[count..<array.count])
    }

    private func longestCommonPrefix(for sequences: [[Int]]) -> [Int] {
        guard var prefix = sequences.first else { return [] }
        for sequence in sequences.dropFirst() {
            var matchCount = 0
            let compareCount = min(prefix.count, sequence.count)
            while matchCount < compareCount, prefix[matchCount] == sequence[matchCount] {
                matchCount += 1
            }
            if matchCount < prefix.count {
                prefix = Array(prefix.prefix(matchCount))
            }
            if prefix.isEmpty {
                break
            }
        }
        return prefix
    }

    private func determineCommitCount(
        currentCommitted: Int,
        stablePrefix: [Int],
        state: StreamState,
        nowMs: Int
    ) -> Int {
        var finalCount = stablePrefix.count
        if config.emitWordBoundaries {
            finalCount = trimToWordBoundary(
                stablePrefix: stablePrefix,
                currentlyCommitted: currentCommitted
            )
        }

        if config.maxWaitMilliseconds > 0, finalCount < stablePrefix.count {
            var target = finalCount
            for index in currentCommitted..<stablePrefix.count {
                guard let firstSeen = state.firstSeen[index] else { continue }
                if nowMs - firstSeen >= config.maxWaitMilliseconds {
                    target = max(target, index + 1)
                }
            }
            finalCount = target
        }

        return max(finalCount, currentCommitted)
    }

    private func trimToWordBoundary(
        stablePrefix: [Int],
        currentlyCommitted: Int
    ) -> Int {
        guard stablePrefix.count > currentlyCommitted else { return currentlyCommitted }

        var boundaryIndex = currentlyCommitted
        let range = currentlyCommitted..<stablePrefix.count

        for index in range {
            guard let tokenString = decodeToken(stablePrefix[index]) else { continue }
            let nextIndex = index + 1

            if isBoundaryToken(tokenString) {
                boundaryIndex = nextIndex
                continue
            }

            if nextIndex < stablePrefix.count,
                let nextToken = decodeToken(stablePrefix[nextIndex]),
                startsNewWord(nextToken)
            {
                boundaryIndex = nextIndex
            }
        }

        return boundaryIndex
    }

    private func startsNewWord(_ token: String) -> Bool {
        switch config.tokenizerKind {
        case .sentencePiece, .bytePairEncoding:
            return token.hasPrefix("▁")
        case .wordPiece:
            return !token.hasPrefix("##") && token != "'"
        }
    }

    private func isBoundaryToken(_ token: String) -> Bool {
        if token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        let punctuation: Set<Character> = [".", ",", ";", ":", "!", "?", "…", "-", "—"]
        if let last = token.last, punctuation.contains(last) {
            return true
        }
        if config.tokenizerKind == .wordPiece, token == "'" {
            return false
        }
        return false
    }

    private func recordLatencies(
        state: StreamState,
        committedRange: Range<Int>,
        nowMs: Int
    ) -> [StabilizedTokenLatency] {
        var measurements: [StabilizedTokenLatency] = []
        for index in committedRange {
            guard let firstSeen = state.firstSeen[index] else { continue }
            let latency = max(0, nowMs - firstSeen)
            measurements.append(StabilizedTokenLatency(tokenIndex: index, latencyMs: latency))
        }
        return measurements
    }
}
