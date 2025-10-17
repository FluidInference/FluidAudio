import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class StabilizedStreamingEmitterTests: XCTestCase {
    private let tokenizer = StabilizerTestTokenizer.shared

    private func makeEmitter(
        windowSize: Int = 3,
        boundary: Bool = true,
        maxWait: Int = 800
    ) -> StabilizedStreamingEmitter {
        let config = StreamingStabilizerConfig(
            windowSize: windowSize,
            emitWordBoundaries: boundary,
            maxWaitMilliseconds: maxWait
        )
        return StabilizedStreamingEmitter(
            config: config,
            tokenDecoder: { [tokenizer] token in
                tokenizer.decode(token)
            })
    }

    func test_lcp_two_sequences_basic() {
        let emitter = makeEmitter(windowSize: 2, boundary: false)
        let first = emitter.update(uid: 0, tokenIds: [1, 3], nowMs: 0)
        XCTAssertEqual(first.committedTokens, [1, 3])
        XCTAssertEqual(first.withheldStableTokenCount, 0)

        let second = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 60)
        XCTAssertEqual(second.committedTokens, [4])
        XCTAssertEqual(second.withheldStableTokenCount, 0)
    }

    func test_lcp_multiple_sequences_noise_in_last() {
        let emitter = makeEmitter(windowSize: 3, boundary: false)
        let first = emitter.update(uid: 0, tokenIds: [1, 3, 8], nowMs: 0)
        XCTAssertEqual(first.committedTokens, [1, 3, 8])
        XCTAssertEqual(first.withheldStableTokenCount, 0)
        let second = emitter.update(uid: 0, tokenIds: [1, 3, 8, 4], nowMs: 40)
        XCTAssertTrue(second.committedTokens.isEmpty)
        let third = emitter.update(uid: 0, tokenIds: [1, 3, 19, 4], nowMs: 80)

        XCTAssertTrue(third.committedTokens.isEmpty)
        XCTAssertEqual(third.withheldStableTokenCount, 0)
    }

    func test_no_common_prefix_no_emit() {
        let emitter = makeEmitter(windowSize: 2, boundary: false)
        _ = emitter.update(uid: 0, tokenIds: [1, 3], nowMs: 0)
        let second = emitter.update(uid: 0, tokenIds: [9, 10], nowMs: 40)
        XCTAssertTrue(second.committedTokens.isEmpty)
    }

    func test_boundary_trim_blocks_midword() {
        let emitter = makeEmitter(windowSize: 2, boundary: true)
        _ = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 0)
        let second = emitter.update(uid: 0, tokenIds: [12, 13, 14], nowMs: 60)
        XCTAssertTrue(second.committedTokens.isEmpty, "Should not emit until boundary appears")

        let third = emitter.update(uid: 0, tokenIds: [12, 13, 14, 9], nowMs: 120)
        XCTAssertTrue(third.committedTokens.isEmpty, "Boundary should hold text until next word stabilizes")

        let fourth = emitter.update(uid: 0, tokenIds: [12, 13, 14, 9, 10], nowMs: 180)
        XCTAssertEqual(fourth.committedTokens, [12, 13, 14])
        XCTAssertEqual(decode(fourth.committedTokens).replacingOccurrences(of: " ", with: ""), "stabilized")
    }

    func test_max_wait_allows_midword_after_timeout() {
        let emitter = makeEmitter(windowSize: 2, boundary: true, maxWait: 50)
        _ = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 0)
        _ = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 30)
        let third = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 70)

        XCTAssertEqual(third.committedTokens, [12, 13])
        XCTAssertTrue(third.withheldStableTokenCount >= 0)
    }

    func test_flush_commits_remaining_tokens() {
        let emitter = makeEmitter(windowSize: 2, boundary: true)
        _ = emitter.update(uid: 0, tokenIds: [1, 3], nowMs: 0)
        _ = emitter.update(uid: 0, tokenIds: [1, 3, 8], nowMs: 40)
        let flush = emitter.flush(uid: 0, nowMs: 80)
        XCTAssertEqual(flush.committedTokens, [8])
    }

    func test_update_idempotent_when_no_change() {
        let emitter = makeEmitter(windowSize: 2, boundary: false)
        let first = emitter.update(uid: 0, tokenIds: [1, 3], nowMs: 0)
        XCTAssertEqual(first.committedTokens, [1, 3])
        let second = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 40)
        XCTAssertTrue(second.committedTokens.isEmpty)

        let third = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 80)
        XCTAssertEqual(third.committedTokens, [4])

        let fourth = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 120)
        XCTAssertTrue(fourth.committedTokens.isEmpty)
    }

    func test_reset_clears_state() {
        let emitter = makeEmitter(windowSize: 2, boundary: false)
        let first = emitter.update(uid: 0, tokenIds: [1, 3], nowMs: 0)
        XCTAssertEqual(first.committedTokens, [1, 3])
        _ = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 40)
        emitter.reset(uid: 0)
        let afterReset = emitter.update(uid: 0, tokenIds: [1, 3, 4], nowMs: 80)
        XCTAssertEqual(afterReset.committedTokens, [1, 3, 4], "Reset should clear commit history so tokens emit again")
    }

    func test_sentencepiece_boundary_markers_behavior() {
        let partial = [12, 13]
        let extended = [12, 13, 14]

        let boundaryEmitter = makeEmitter(windowSize: 2, boundary: true)
        let freeEmitter = makeEmitter(windowSize: 2, boundary: false)

        _ = boundaryEmitter.update(uid: 0, tokenIds: partial, nowMs: 0)
        let boundaryResult = boundaryEmitter.update(uid: 0, tokenIds: extended, nowMs: 60)

        _ = freeEmitter.update(uid: 1, tokenIds: partial, nowMs: 0)
        let freeResult = freeEmitter.update(uid: 1, tokenIds: extended, nowMs: 60)

        XCTAssertLessThanOrEqual(
            boundaryResult.committedTokens.count,
            freeResult.committedTokens.count
        )
    }

    func test_emits_are_prefix_of_final_transcript() {
        let emitter = makeEmitter(windowSize: 3, boundary: true)
        let sequences = makeNoisySequences(base: [1, 3, 8, 4, 9, 10, 4])
        var committed: [Int] = []
        let finalTranscript = decode(sequences.last ?? []).trimmingCharacters(in: .whitespaces)

        for (index, sequence) in sequences.enumerated() {
            let now = index * 70
            let result = emitter.update(uid: 0, tokenIds: sequence, nowMs: now)
            committed.append(contentsOf: result.committedTokens)
            let committedText = decode(committed).trimmingCharacters(in: .whitespaces)
            XCTAssertTrue(finalTranscript.hasPrefix(committedText))
        }
    }

    func test_never_retracts_text() {
        let emitter = makeEmitter(windowSize: 3, boundary: true)
        let sequences = makeNoisySequences(base: [1, 3, 8, 4, 9, 10, 4])
        var committed: [Int] = []
        var previous = ""

        for (idx, sequence) in sequences.enumerated() {
            let result = emitter.update(uid: 0, tokenIds: sequence, nowMs: idx * 60)
            if !result.committedTokens.isEmpty {
                committed.append(contentsOf: result.committedTokens)
                let text = decode(committed)
                XCTAssertTrue(text.hasPrefix(previous))
                previous = text
            }
        }
    }

    func test_bounded_delay_with_max_wait() {
        let emitter = makeEmitter(windowSize: 2, boundary: true, maxWait: 40)
        _ = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 0)
        _ = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 20)
        let late = emitter.update(uid: 0, tokenIds: [12, 13], nowMs: 50)
        XCTAssertEqual(late.committedTokens, [12, 13])
    }

    // MARK: - Helpers

    private func decode(_ tokens: [Int]) -> String {
        tokenizer.decodeToText(tokens)
    }

    private func makeNoisySequences(base: [Int]) -> [[Int]] {
        var sequences: [[Int]] = []
        var generator = SeededGenerator(seed: 12345)
        for index in 1...base.count {
            var current = Array(base.prefix(index))
            if index > 2, generator.next() % 3 == 0, current.count > 1 {
                current[current.count - 2] = 18  // introduce a flicker token
            }
            sequences.append(current)
        }
        return sequences
    }
}
