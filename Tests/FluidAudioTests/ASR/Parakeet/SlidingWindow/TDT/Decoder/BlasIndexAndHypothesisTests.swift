import Foundation
import XCTest

@testable import FluidAudio

final class BlasIndexAndHypothesisTests: XCTestCase {

    // MARK: - BlasIndex

    func testMakeBlasIndexValidValues() throws {
        XCTAssertEqual(try makeBlasIndex(0, label: "test"), 0)
        XCTAssertEqual(try makeBlasIndex(100, label: "test"), 100)
        XCTAssertEqual(try makeBlasIndex(Int(Int32.max), label: "test"), Int32.max)
    }

    func testMakeBlasIndexOverflowThrows() {
        let overflowValue = Int(Int32.max) + 1
        XCTAssertThrowsError(try makeBlasIndex(overflowValue, label: "overflow")) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError, got \(type(of: error))")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(
                    message.contains("overflow"),
                    "Error message should contain the label"
                )
            } else {
                XCTFail("Expected processingFailed case")
            }
        }
    }

    func testMakeBlasIndexNegativeValue() throws {
        // Negative Int32 values should work
        XCTAssertEqual(try makeBlasIndex(-1, label: "test"), -1)
        XCTAssertEqual(try makeBlasIndex(Int(Int32.min), label: "test"), Int32.min)
    }

    func testMakeBlasIndexNegativeOverflowThrows() {
        let underflowValue = Int(Int32.min) - 1
        XCTAssertThrowsError(try makeBlasIndex(underflowValue, label: "underflow"))
    }

    // MARK: - TdtHypothesis

    func testHypothesisEmptyProperties() {
        let state = TdtDecoderState.make()
        var hypothesis = TdtHypothesis(decState: state)
        hypothesis.ySequence = []
        hypothesis.timestamps = []
        hypothesis.tokenConfidences = []

        XCTAssertTrue(hypothesis.isEmpty)
        XCTAssertFalse(hypothesis.hasTokens)
        XCTAssertEqual(hypothesis.tokenCount, 0)
        XCTAssertNil(hypothesis.computedLastToken)
        XCTAssertEqual(hypothesis.maxTimestamp, 0)
    }

    func testHypothesisPopulatedProperties() {
        let state = TdtDecoderState.make()
        var hypothesis = TdtHypothesis(decState: state)
        hypothesis.ySequence = [10, 20, 30]
        hypothesis.timestamps = [1, 5, 3]
        hypothesis.tokenConfidences = [0.9, 0.8, 0.7]

        XCTAssertFalse(hypothesis.isEmpty)
        XCTAssertTrue(hypothesis.hasTokens)
        XCTAssertEqual(hypothesis.tokenCount, 3)
        XCTAssertEqual(hypothesis.computedLastToken, 30)
        XCTAssertEqual(hypothesis.maxTimestamp, 5)
    }

    func testHypothesisDestructured() {
        let state = TdtDecoderState.make()
        var hypothesis = TdtHypothesis(decState: state)
        hypothesis.ySequence = [1, 2]
        hypothesis.timestamps = [10, 20]
        hypothesis.tokenConfidences = [0.5, 0.6]

        let (tokens, timestamps, confidences) = hypothesis.destructured
        XCTAssertEqual(tokens, [1, 2])
        XCTAssertEqual(timestamps, [10, 20])
        XCTAssertEqual(confidences, [0.5, 0.6])
    }

    func testHypothesisLastTokenTracking() {
        let state = TdtDecoderState.make()
        var hypothesis = TdtHypothesis(decState: state)

        XCTAssertNil(hypothesis.lastToken)

        hypothesis.lastToken = 42
        XCTAssertEqual(hypothesis.lastToken, 42)

        hypothesis.lastToken = nil
        XCTAssertNil(hypothesis.lastToken)
    }
}
