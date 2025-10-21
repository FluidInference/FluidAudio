import Foundation

struct StreamingStabilizerOutput {
    let updates: [StreamingTranscriptionUpdate]
    let newlyCommittedCount: Int
    let totalCommittedCount: Int
}

struct StreamingStabilizerMetrics {
    let firstCommitLatencySeconds: TimeInterval?
}

struct StreamingStabilizerSink {
    private let config: StreamingStabilizerConfig

    private var emitter: StabilizedStreamingEmitter?
    private var tokenDecoder: StabilizedStreamingEmitter.TokenDecoder?
    private var asrManager: AsrManager?

    private var committedTokenCount: Int = 0
    private var totalTokensCommitted: Int = 0
    private var currentWithheldCount: Int = 0
    private var maxWithheldCount: Int = 0
    private var commitBatchSizes: [Int] = []
    private var tokenLatencies: [Int] = []
    private var firstCommitLatencyMs: Int?
    private var metricsFinalized: Bool = false

    private(set) var volatileTranscript: String = ""
    private(set) var confirmedTranscript: String = ""

    init(config: StreamingStabilizerConfig) {
        self.config = config
    }

    mutating func resetTranscripts() {
        volatileTranscript = ""
        confirmedTranscript = ""
    }

    mutating func refreshVocabulary(using manager: AsrManager?, logger: AppLogger) {
        guard let manager else {
            logger.error("Stabilizer vocabulary refresh requested without an ASR manager instance")
            return
        }

        asrManager = manager

        if manager.vocabulary.isEmpty {
            logger.warning("Stabilizer vocabulary refresh provided an empty vocabulary; token decoding may be degraded")
        }

        let decoder = prepareTokenDecoder(with: manager.vocabulary)
        if let emitter {
            emitter.updateTokenDecoder(decoder)
        } else {
            logger.error("Stabilizer emitter missing during vocabulary refresh; recreating emitter")
            self.emitter = StabilizedStreamingEmitter(config: config, tokenDecoder: decoder)
        }
    }

    mutating func initialize(using manager: AsrManager?, uid: Int, logger: AppLogger) {
        if manager?.vocabulary.isEmpty ?? true {
            logger.warning("Stabilizer initialized without a populated vocabulary; token decoding may be degraded")
        }
        asrManager = manager
        let decoder = prepareTokenDecoder(with: manager?.vocabulary ?? [:])
        emitter = StabilizedStreamingEmitter(
            config: config,
            tokenDecoder: decoder
        )

        emitter?.reset(uid: uid)
        committedTokenCount = 0
        totalTokensCommitted = 0
        currentWithheldCount = 0
        maxWithheldCount = 0
        commitBatchSizes.removeAll(keepingCapacity: false)
        tokenLatencies.removeAll(keepingCapacity: false)
        firstCommitLatencyMs = nil
        metricsFinalized = false
    }

    mutating func handleInitializationFailure(logger: AppLogger) {
        logger.debug("Stabilizer initialization failed; metrics collection reset")
    }

    mutating func makeUpdates(
        result: StabilizedUpdateResult,
        accumulatedTokens: [Int],
        latestTokens: [Int],
        latestTokenTimings: [TokenTiming],
        interimConfidence: Float,
        timestamp: Date
    ) -> StreamingStabilizerOutput {
        var updates: [StreamingTranscriptionUpdate] = []

        if let firstLatency = result.firstCommitLatencyMs, firstCommitLatencyMs == nil {
            firstCommitLatencyMs = firstLatency
        }

        currentWithheldCount = result.withheldStableTokenCount
        maxWithheldCount = max(maxWithheldCount, currentWithheldCount)

        if !result.tokenLatencies.isEmpty {
            for measurement in result.tokenLatencies {
                tokenLatencies.append(measurement.latencyMs)
            }
        }

        if !result.committedTokens.isEmpty {
            committedTokenCount += result.committedTokens.count
            totalTokensCommitted += result.committedTokens.count
            commitBatchSizes.append(result.committedTokens.count)

            let commitTimings: [TokenTiming]
            if !latestTokens.isEmpty,
                result.committedTokens.count <= latestTokens.count,
                latestTokens.suffix(result.committedTokens.count) == result.committedTokens,
                latestTokenTimings.count >= result.committedTokens.count
            {
                commitTimings = Array(latestTokenTimings.suffix(result.committedTokens.count))
            } else {
                commitTimings = []
            }

            let decoded = decodeTokens(
                result.committedTokens,
                candidateTimings: commitTimings
            )
            let appendedCommittedText = appendCommittedText(decoded.text)

            if let appendedCommittedText,
                !appendedCommittedText.isEmpty
            {
                let update = StreamingTranscriptionUpdate(
                    text: trimTrailingNewlinesPreservingLeadingWhitespace(appendedCommittedText),
                    isConfirmed: true,
                    confidence: 1.0,
                    timestamp: timestamp,
                    tokenIds: result.committedTokens,
                    tokenTimings: decoded.timings
                )
                updates.append(update)
            }
        }

        let pendingCount = max(0, accumulatedTokens.count - committedTokenCount)
        if pendingCount > 0 {
            let startIndex = max(0, committedTokenCount)
            if startIndex < accumulatedTokens.count {
                let pendingTokens = Array(accumulatedTokens[startIndex..<accumulatedTokens.count])
                let decoded = decodeTokens(pendingTokens, candidateTimings: [])
                let volatileText = normalizeSpacing(decoded.text)
                volatileTranscript = trimTrailingNewlinesPreservingLeadingWhitespace(volatileText)

                if !volatileTranscript.isEmpty {
                    let update = StreamingTranscriptionUpdate(
                        text: volatileTranscript,
                        isConfirmed: false,
                        confidence: interimConfidence,
                        timestamp: timestamp,
                        tokenIds: pendingTokens,
                        tokenTimings: []
                    )
                    updates.append(update)
                }
            }
        } else {
            volatileTranscript = ""
        }

        return StreamingStabilizerOutput(
            updates: updates,
            newlyCommittedCount: result.committedTokens.count,
            totalCommittedCount: totalTokensCommitted
        )
    }

    mutating func flush(uid: Int, nowMs: Int) -> StabilizedUpdateResult? {
        guard let emitter else { return nil }
        return emitter.flush(uid: uid, nowMs: nowMs)
    }

    mutating func emitterUpdate(
        uid: Int,
        accumulatedTokens: [Int],
        latestTokens: [Int],
        latestTokenTimings: [TokenTiming],
        interimConfidence: Float,
        nowMs: Int
    ) -> StreamingStabilizerOutput {
        guard let emitter else {
            return StreamingStabilizerOutput(
                updates: [],
                newlyCommittedCount: 0,
                totalCommittedCount: totalTokensCommitted
            )
        }
        let result = emitter.update(
            uid: uid,
            tokenIds: accumulatedTokens,
            nowMs: nowMs
        )
        return makeUpdates(
            result: result,
            accumulatedTokens: accumulatedTokens,
            latestTokens: latestTokens,
            latestTokenTimings: latestTokenTimings,
            interimConfidence: interimConfidence,
            timestamp: Date()
        )
    }

    mutating func finalizeAfterStreamEnd(logger: AppLogger) {
        guard !metricsFinalized else { return }
        finalizeStabilizerMetrics(logger: logger)
        metricsFinalized = true
    }

    func metricsSnapshot() -> StreamingStabilizerMetrics {
        StreamingStabilizerMetrics(
            firstCommitLatencySeconds: firstCommitLatencyMs.map { Double($0) / 1000.0 }
        )
    }

    mutating func discardCommittedTokenPrefix(_ count: Int, uid: Int) {
        guard count > 0 else { return }
        committedTokenCount = max(0, committedTokenCount - count)
        emitter?.discardCommittedPrefix(uid: uid, count: count)
    }

    mutating func cleanupState(uid: Int) {
        emitter?.cleanupState(for: uid)
    }

    private func decodeTokens(
        _ tokenIds: [Int],
        candidateTimings: [TokenTiming]
    ) -> (text: String, timings: [TokenTiming]) {
        if let manager = asrManager {
            let (text, timings) = manager.convertTokensWithExistingTimings(tokenIds, timings: candidateTimings)
            return (text, timings)
        }
        let text = decodeTokensToTextPreservingSpaces(tokenIds)
        return (text, candidateTimings)
    }

    private func decodeTokensToTextPreservingSpaces(_ tokenIds: [Int]) -> String {
        guard let decoder = tokenDecoder else { return "" }
        var pieces: [String] = []
        for tokenId in tokenIds {
            if let token = decoder(tokenId), !token.isEmpty {
                pieces.append(token)
            }
        }
        guard !pieces.isEmpty else { return "" }
        let joined = pieces.joined()
        return joined.replacingOccurrences(of: "▁", with: " ")
    }

    private mutating func appendCommittedText(_ text: String) -> String? {
        let normalized = normalizeSpacing(text)
        guard !normalized.isEmpty else { return nil }

        let appendedText: String
        if confirmedTranscript.isEmpty {
            let trimmed = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
            confirmedTranscript = trimmed
            appendedText = trimmed
        } else if confirmedTranscript.last?.isWhitespace == true {
            let trimmed = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
            confirmedTranscript.append(trimmed)
            appendedText = trimmed
        } else if normalized.first?.isPunctuation == true {
            confirmedTranscript.append(normalized)
            appendedText = normalized
        } else if normalized.first == " " {
            confirmedTranscript.append(normalized)
            appendedText = normalized
        } else {
            let trimmed = normalized.trimmingCharacters(in: .whitespaces)
            let segment = " \(trimmed)"
            confirmedTranscript.append(segment)
            appendedText = segment
        }

        return appendedText
    }

    private func trimTrailingNewlinesPreservingLeadingWhitespace(_ text: String) -> String {
        var result = text
        while result.last?.isNewline == true {
            result.removeLast()
        }
        return result
    }

    private func normalizeSpacing(_ text: String) -> String {
        var result = text.replacingOccurrences(of: "\n", with: " ")
        while result.contains("  ") {
            result = result.replacingOccurrences(of: "  ", with: " ")
        }
        return result
    }

    private mutating func prepareTokenDecoder(
        with vocabulary: [Int: String]
    )
        -> StabilizedStreamingEmitter.TokenDecoder
    {
        let vocabularySnapshot = vocabulary
        let decoder: StabilizedStreamingEmitter.TokenDecoder = { tokenId in
            vocabularySnapshot[tokenId]
        }
        tokenDecoder = decoder
        return decoder
    }

    private mutating func finalizeStabilizerMetrics(logger: AppLogger) {
        let committed = totalTokensCommitted
        let withheld = currentWithheldCount
        let withheldPeak = maxWithheldCount
        let batches = commitBatchSizes
        let latencies = tokenLatencies
        let firstLatency = firstCommitLatencyMs

        var message = "Stabilizer metrics -> committed=\(committed) withheld_active=\(withheld)"
        if withheldPeak > 0 {
            message += " withheld_peak=\(withheldPeak)"
        }
        if let firstLatency {
            message += " first_commit_ms=\(firstLatency)"
        }
        if !batches.isEmpty {
            let averageBatch = batches.reduce(0, +) / batches.count
            message += " avg_commit_batch=\(averageBatch)"
            if let minBatch = batches.min(), let maxBatch = batches.max() {
                message += " commit_batch_range=\(minBatch)-\(maxBatch)"
            }
        }
        if !latencies.isEmpty {
            let averageLatency = latencies.reduce(0, +) / latencies.count
            message += " avg_token_latency_ms=\(averageLatency)"
            if let minLatency = latencies.min(), let maxLatency = latencies.max() {
                message += " token_latency_range_ms=\(minLatency)-\(maxLatency)"
            }
            message += " token_latency_count=\(latencies.count)"
        }
        logger.info("\(message)")
    }
}
