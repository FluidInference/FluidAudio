import XCTest

@testable import FluidAudio

final class HungarianAssignmentTests: XCTestCase {

    // MARK: - Square min-cost solve

    func testSolveEmptyMatrix() {
        XCTAssertTrue(HungarianAssignment.solve(costSquare: [], n: 0).isEmpty)
    }

    func testSolvePicksMinimumCostPerfectMatching() {
        // Row 0 is cheapest on col 1, row 1 on col 0, row 2 on col 2.
        let cost = [
            4, 1, 3,
            2, 0, 5,
            3, 2, 2,
        ]
        let assign = HungarianAssignment.solve(costSquare: cost, n: 3)
        XCTAssertEqual(assign, [1, 0, 2])
    }

    func testSolveResolvesGreedyConflict() {
        // Both rows prefer col 0; the optimal total forces row 1 onto col 0.
        let cost = [
            1, 2,
            0, 10,
        ]
        let assign = HungarianAssignment.solve(costSquare: cost, n: 2)
        XCTAssertEqual(assign, [1, 0])
    }

    // MARK: - Rectangular max-score assignment

    func testMaxScoreAssignmentSquare() {
        let scores: [[Double]] = [
            [0.9, 0.1],
            [0.8, 0.2],
        ]
        // Greedy would put both rows on col 0; the constraint forces row 1
        // (the weaker match) onto col 1.
        XCTAssertEqual(HungarianAssignment.maxScoreAssignment(scores: scores), [0, 1])
    }

    func testMaxScoreAssignmentMoreColumnsThanRows() {
        let scores: [[Double]] = [
            [0.1, 0.9, 0.3]
        ]
        XCTAssertEqual(HungarianAssignment.maxScoreAssignment(scores: scores), [1])
    }

    func testMaxScoreAssignmentMoreRowsThanColumnsDropsWeakestRow() {
        let scores: [[Double]] = [
            [0.9],
            [0.5],
            [0.7],
        ]
        // Only one column exists; the strongest row keeps it, the rest drop.
        XCTAssertEqual(HungarianAssignment.maxScoreAssignment(scores: scores), [0, -1, -1])
    }

    func testMaxScoreAssignmentTreatsNonFiniteAsWorst() {
        let scores: [[Double]] = [
            [Double.nan, 0.2],
            [0.6, 0.5],
        ]
        XCTAssertEqual(HungarianAssignment.maxScoreAssignment(scores: scores), [1, 0])
    }

    func testMaxScoreAssignmentEmpty() {
        XCTAssertTrue(HungarianAssignment.maxScoreAssignment(scores: []).isEmpty)
        XCTAssertEqual(HungarianAssignment.maxScoreAssignment(scores: [[], []]), [-1, -1])
    }
}
