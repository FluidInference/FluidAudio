import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class StabilizedStreamingIntegrationTests: XCTestCase {
    private let tokenizer = StabilizerTestTokenizer.shared

    func test_streaming_emits_monotonic_with_small_right_context() {
        let config = StreamingStabilizerConfig(
            windowSize: 2,
            emitWordBoundaries: true,
            maxWaitMilliseconds: 800
        )
        let sequences = incrementalSequences()
        let timestamps = [400, 480, 520, 600, 680, 760, 820]
        let result = runStream(config: config, sequences: sequences, timestamps: timestamps)

        XCTAssertFalse(result.commits.isEmpty)
        let finalText = tokenizer.decodeToText(sequences.last ?? []).trimmingCharacters(in: .whitespaces)
        for committed in result.commits {
            XCTAssertTrue(finalText.hasPrefix(committed))
        }
    }

    func test_streaming_emits_monotonic_with_large_right_context() {
        let config = StreamingStabilizerConfig(
            windowSize: 2,
            emitWordBoundaries: true,
            maxWaitMilliseconds: 1200
        )
        let sequences = incrementalSequences()
        let timestamps = [800, 900, 960, 1040, 1200, 1320, 1400]
        let result = runStream(config: config, sequences: sequences, timestamps: timestamps)
        XCTAssertFalse(result.commits.isEmpty)
        let finalText = tokenizer.decodeToText(sequences.last ?? []).trimmingCharacters(in: .whitespaces)
        for committed in result.commits {
            XCTAssertTrue(finalText.hasPrefix(committed))
        }
    }

    func test_first_commit_latency_close_to_chunk_plus_right() {
        let config = StreamingStabilizerConfig(
            windowSize: 2,
            emitWordBoundaries: true,
            maxWaitMilliseconds: 600
        )
        let sequences = incrementalSequences()
        let chunkMs = 280
        let rightMs = 170
        let timestamps = [chunkMs + rightMs, chunkMs + rightMs + 60, chunkMs + rightMs + 120, chunkMs + rightMs + 220]
        let result = runStream(
            config: config, sequences: Array(sequences.prefix(timestamps.count)), timestamps: timestamps)

        guard let latency = result.firstCommitLatency else {
            XCTFail("Expected first commit latency to be recorded")
            return
        }
        let expected = chunkMs + rightMs
        XCTAssertLessThanOrEqual(abs(latency - expected), chunkMs + rightMs)
    }

    // MARK: - Helpers

    private func runStream(
        config: StreamingStabilizerConfig,
        sequences: [[Int]],
        timestamps: [Int]
    ) -> (commits: [String], firstCommitLatency: Int?, finalTokens: [Int]) {
        precondition(sequences.count == timestamps.count)
        let emitter = StabilizedStreamingEmitter(
            config: config,
            tokenDecoder: { [unowned self] token in
                self.tokenizer.decode(token)
            })
        var committedTokens: [Int] = []
        var commits: [String] = []
        var lastCommitSnapshot: String = ""
        var firstLatency: Int?

        for (sequence, time) in zip(sequences, timestamps) {
            let result = emitter.update(uid: 0, tokenIds: sequence, nowMs: time)
            if let latency = result.firstCommitLatencyMs, firstLatency == nil {
                firstLatency = latency
            }
            if !result.committedTokens.isEmpty {
                committedTokens.append(contentsOf: result.committedTokens)
                let committedText = tokenizer.decodeToText(committedTokens)
                    .trimmingCharacters(in: .whitespaces)
                if !committedText.isEmpty, committedText != lastCommitSnapshot {
                    commits.append(committedText)
                    lastCommitSnapshot = committedText
                }
            }
        }

        return (commits, firstLatency, sequences.last ?? [])
    }

    private func incrementalSequences() -> [[Int]] {
        let base: [Int] = [1, 3, 8, 4, 9, 10, 4]
        var sequences: [[Int]] = []
        for index in 1...base.count {
            sequences.append(Array(base.prefix(index)))
        }
        return sequences
    }
}
