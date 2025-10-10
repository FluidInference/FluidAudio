import XCTest

@testable import FluidAudio

final class StreamingStabilizerSinkTests: XCTestCase {

    func testCommittedUpdatesPreserveLeadingWhitespace() {
        var sink = StreamingStabilizerSink(config: StreamingStabilizerConfig())
        let manager = AsrManager()
        manager.setVocabularyForTesting(
            [
                1: "▁hello",
                2: "▁world",
            ]
        )

        sink.initialize(using: manager, uid: 0, logger: AppLogger(category: "Test"))

        let firstResult = StabilizedUpdateResult(
            committedTokens: [1],
            firstCommitLatencyMs: nil,
            tokenLatencies: [],
            withheldStableTokenCount: 0,
            debugRecords: []
        )

        let firstOutput = sink.makeUpdates(
            result: firstResult,
            accumulatedTokens: [1],
            latestTokens: [1],
            latestTokenTimings: [],
            interimConfidence: 0.9,
            timestamp: Date()
        )

        XCTAssertEqual(firstOutput.updates.count, 1)
        XCTAssertEqual(firstOutput.newlyCommittedCount, 1)
        XCTAssertEqual(firstOutput.totalCommittedCount, 1)
        XCTAssertEqual(firstOutput.updates.first?.text, "hello")
        XCTAssertTrue(firstOutput.updates.first?.isConfirmed ?? false)
        XCTAssertEqual(sink.confirmedTranscript, "hello")

        let secondResult = StabilizedUpdateResult(
            committedTokens: [2],
            firstCommitLatencyMs: nil,
            tokenLatencies: [],
            withheldStableTokenCount: 0,
            debugRecords: []
        )

        let secondOutput = sink.makeUpdates(
            result: secondResult,
            accumulatedTokens: [1, 2],
            latestTokens: [1, 2],
            latestTokenTimings: [],
            interimConfidence: 0.95,
            timestamp: Date()
        )

        XCTAssertEqual(secondOutput.newlyCommittedCount, 1)
        XCTAssertEqual(secondOutput.totalCommittedCount, 2)

        let committedUpdate = secondOutput.updates.first { $0.isConfirmed }
        XCTAssertEqual(committedUpdate?.text, " world")
        XCTAssertEqual(sink.confirmedTranscript, "hello world")
        XCTAssertEqual(sink.volatileTranscript, "")
    }
}
