import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class SpeakerCountConstraintsTests: XCTestCase {

    // MARK: - Basic Resolution Tests

    func testResolveWithNoConstraintsReturnsDefaults() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: nil,
            maxSpeakers: nil
        )
        XCTAssertNil(result.numSpeakers)
        XCTAssertEqual(result.minSpeakers, 1)
        XCTAssertEqual(result.maxSpeakers, 100)
    }

    func testResolveWithNumSpeakersOverridesMinMax() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: 3,
            minSpeakers: 1,
            maxSpeakers: 10
        )
        XCTAssertEqual(result.numSpeakers, 3)
        XCTAssertEqual(result.minSpeakers, 3)
        XCTAssertEqual(result.maxSpeakers, 3)
    }

    func testResolveClampsSpeakerCountToEmbeddings() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 5,
            numSpeakers: nil,
            minSpeakers: 2,
            maxSpeakers: 20
        )
        XCTAssertEqual(result.minSpeakers, 2)
        XCTAssertEqual(result.maxSpeakers, 5)
    }

    /// Note: If minSpeakers > maxSpeakers, minSpeakers is clamped to maxSpeakers.
    /// This prevents crashes but may not reflect user intent.
    func testResolveEnforcesMinNotGreaterThanMax() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: 10,
            maxSpeakers: 5
        )
        XCTAssertEqual(result.minSpeakers, 5)
        XCTAssertEqual(result.maxSpeakers, 5)
    }

    // MARK: - Boundary Condition Tests (Expert Panel: Wiegers)

    func testResolveWithZeroNumSpeakersClampsToOne() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: 0,
            minSpeakers: nil,
            maxSpeakers: nil
        )
        XCTAssertEqual(result.minSpeakers, 1)
        XCTAssertEqual(result.maxSpeakers, 1)
    }

    func testResolveWithNegativeNumSpeakersClampsToOne() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: -5,
            minSpeakers: nil,
            maxSpeakers: nil
        )
        XCTAssertEqual(result.minSpeakers, 1)
        XCTAssertEqual(result.maxSpeakers, 1)
    }

    func testResolveWithZeroMinSpeakersClampsToOne() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: 0,
            maxSpeakers: 5
        )
        XCTAssertEqual(result.minSpeakers, 1)
    }

    func testResolveWithNegativeMinSpeakersClampsToOne() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: -3,
            maxSpeakers: 5
        )
        XCTAssertEqual(result.minSpeakers, 1)
    }

    // MARK: - Adjustment Tests

    func testNeedsAdjustmentWhenBelowMin() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: 5,
            maxSpeakers: 10
        )
        XCTAssertTrue(result.needsAdjustment(detectedCount: 3))
        XCTAssertEqual(result.targetCount(detectedCount: 3), 5)
    }

    func testNeedsAdjustmentWhenAboveMax() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: 2,
            maxSpeakers: 5
        )
        XCTAssertTrue(result.needsAdjustment(detectedCount: 8))
        XCTAssertEqual(result.targetCount(detectedCount: 8), 5)
    }

    func testNoAdjustmentWhenWithinBounds() {
        let result = SpeakerCountConstraints.resolve(
            numEmbeddings: 100,
            numSpeakers: nil,
            minSpeakers: 2,
            maxSpeakers: 5
        )
        XCTAssertFalse(result.needsAdjustment(detectedCount: 3))
        XCTAssertEqual(result.targetCount(detectedCount: 3), 3)
    }
}
