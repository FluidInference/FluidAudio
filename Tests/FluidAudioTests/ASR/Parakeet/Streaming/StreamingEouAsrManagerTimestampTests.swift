import XCTest

@testable import FluidAudio

final class StreamingEouAsrManagerTimestampTests: XCTestCase {

    func testTokenTimestampCalculationMs() {
        let baseFrame = 4
        let tokenFrames = [0, 1, 3]

        let timestamps = StreamingEouAsrManager.computeTokenTimestampsMs(
            baseFrame: baseFrame,
            tokenFrames: tokenFrames,
            frameDurationMs: 80
        )

        XCTAssertEqual(timestamps, [320, 400, 560])
    }

    func testTokenTimestampCalculationEmpty() {
        let timestamps = StreamingEouAsrManager.computeTokenTimestampsMs(
            baseFrame: 10,
            tokenFrames: [],
            frameDurationMs: 80
        )

        XCTAssertTrue(timestamps.isEmpty)
    }

    // MARK: - EOU Debounce (issue #827)

    private let sampleRate = 16000
    private let chunkSamples = 16000 / 1000 * 320  // 320ms chunk shift

    /// Runs a sequence of chunks through the pure debounce rule and returns the
    /// chunk index (0-based) at which EOU is confirmed, or nil if never.
    /// Each element is (hasNewTokens, eouSignal).
    private func confirmChunk(
        _ chunks: [(Bool, Bool)],
        debounceMs: Int
    ) -> Int? {
        var anchor: Int?
        var confirmed = false
        var total = 0
        for (i, chunk) in chunks.enumerated() {
            total += chunkSamples
            let decision = StreamingEouAsrManager.evaluateEouDebounce(
                hasNewTokens: chunk.0,
                eouSignal: chunk.1,
                totalSamplesProcessed: total,
                anchorSamples: anchor,
                alreadyConfirmed: confirmed,
                debounceMs: debounceMs,
                sampleRate: sampleRate
            )
            anchor = decision.anchorSamples
            if decision.confirmed {
                confirmed = true
                return i
            }
        }
        return nil
    }

    /// A single EOU signal followed by silent, non-EOU chunks still confirms once
    /// enough wall-clock time elapses — no consecutive EOU emissions required.
    func testDebounceFiresOnSilenceAfterSingleEouSignal() {
        // 1280ms debounce at 320ms chunks: anchor at t=320ms, confirm at t=1600ms.
        let chunks: [(Bool, Bool)] = [
            (false, true),  // t=320ms: EOU signal starts the timer
            (false, false),  // t=640ms: silent, no EOU token
            (false, false),  // t=960ms
            (false, false),  // t=1280ms: elapsed 960ms since anchor
            (false, false),  // t=1600ms: elapsed 1280ms -> confirm
        ]
        XCTAssertEqual(confirmChunk(chunks, debounceMs: 1280), 4)
    }

    /// The old behavior required every chunk to re-emit EOU; verify a lone blank
    /// chunk in the middle no longer resets the timer.
    func testDebounceSurvivesBlankChunk() {
        let chunks: [(Bool, Bool)] = [
            (false, true),  // start
            (false, false),  // blank - would have reset under old logic
            (false, true),
            (false, false),
            (false, false),  // t=1600ms, elapsed 1280ms -> confirm
        ]
        XCTAssertEqual(confirmChunk(chunks, debounceMs: 1280), 4)
    }

    /// Newly decoded words reset the timer (real speech is ongoing).
    func testDebounceResetOnNewTokens() {
        let chunks: [(Bool, Bool)] = [
            (false, true),  // start timer
            (false, false),
            (true, false),  // decoded a word -> reset
            (false, true),  // restart timer here (t=1280ms)
            (false, false),  // t=1600ms, elapsed 320ms
            (false, false),  // t=1920ms, elapsed 640ms
            (false, false),  // t=2240ms, elapsed 960ms
            (false, false),  // t=2560ms, elapsed 1280ms -> confirm
        ]
        XCTAssertEqual(confirmChunk(chunks, debounceMs: 1280), 7)
    }

    /// A larger value produces a proportionally longer pause (the reporter's 4s case).
    func testDebounceScalesWithValue() {
        // 4000ms at 320ms chunks: anchor at chunk 0 (t=320ms); confirm when
        // total - 320 >= 4000, i.e. total >= 4320ms -> chunk index 13 (t=4480ms).
        var chunks: [(Bool, Bool)] = [(false, true)]
        chunks.append(contentsOf: Array(repeating: (false, false), count: 20))
        XCTAssertEqual(confirmChunk(chunks, debounceMs: 4000), 13)
    }

    /// No EOU signal ever means no confirmation, regardless of silence length.
    func testDebounceNeverFiresWithoutEouSignal() {
        let chunks: [(Bool, Bool)] = Array(repeating: (false, false), count: 30)
        XCTAssertNil(confirmChunk(chunks, debounceMs: 1280))
    }
}
