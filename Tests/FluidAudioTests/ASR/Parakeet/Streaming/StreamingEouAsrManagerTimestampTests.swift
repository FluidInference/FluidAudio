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
}
