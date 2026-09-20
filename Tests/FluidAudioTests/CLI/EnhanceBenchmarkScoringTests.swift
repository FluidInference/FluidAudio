#if os(macOS)
import XCTest

@testable import FluidAudioCLI

final class EnhanceBenchmarkScoringTests: XCTestCase {
    func testEmptyHypothesisHasZeroRecall() {
        let score = EnhanceBenchmarkCommand.score(hypothesis: [], reference: ["one", "two"], farEnd: ["three"])
        XCTAssertEqual(score.hits, 0)
        XCTAssertEqual(score.deletions, 2)
        XCTAssertEqual(score.errors, 2)
        XCTAssertEqual(score.leaked, 0)
    }

    func testLeakageSubtractsNearEndWordsAndRespectsMultiplicity() {
        let score = EnhanceBenchmarkCommand.score(
            hypothesis: ["hello", "echo", "echo"], reference: ["hello"], farEnd: ["hello", "echo"])
        XCTAssertEqual(score.hits, 1)
        XCTAssertEqual(score.insertions, 2)
        XCTAssertEqual(score.errors, 2)
        XCTAssertEqual(score.leaked, 1)
    }

    func testSubstitutionAndDeletionReduceRecallButInsertionDoesNot() {
        let score = EnhanceBenchmarkCommand.score(
            hypothesis: ["a", "x", "c", "d", "e"], reference: ["a", "b", "c", "d"], farEnd: [])
        XCTAssertEqual(score.hits, 3)
        XCTAssertEqual(score.substitutions, 1)
        XCTAssertEqual(score.insertions, 1)
        XCTAssertEqual(score.deletions, 0)
        XCTAssertEqual(score.errors, 2)
    }

    func testAlreadyNormalizedTokensAreNotNormalizedAgain() {
        let metrics = WERCalculator.calculateWordMetrics(hypothesis: ["colour"], reference: ["color"])
        XCTAssertEqual(metrics.substitutions, 1)
    }

    func testEmptyAudioFailsBeforeInference() {
        XCTAssertThrowsError(try EnhanceBenchmarkCommand.validateAudio(mic: [], reference: [], clean: [], fileID: "0"))
    }
}
#endif
