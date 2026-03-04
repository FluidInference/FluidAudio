import Foundation
import XCTest

@testable import FluidAudio

final class VDSPOperationsTests: XCTestCase {

    // MARK: - L2 Normalize

    func testL2NormalizeProducesUnitVector() {
        let input: [Float] = [3, 4]
        let result = VDSPOperations.l2Normalize(input)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 0.6, accuracy: 1e-5, "Expected 3/5")
        XCTAssertEqual(result[1], 0.8, accuracy: 1e-5, "Expected 4/5")

        let norm = sqrt(result[0] * result[0] + result[1] * result[1])
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5, "Normalized vector should have unit norm")
    }

    func testL2NormalizeZeroVectorStaysZero() {
        let input: [Float] = [0, 0, 0]
        let result = VDSPOperations.l2Normalize(input)

        for value in result {
            XCTAssertFalse(value.isNaN, "Zero vector normalization should not produce NaN")
            XCTAssertEqual(value, 0, accuracy: 1e-6)
        }
    }

    func testL2NormalizeEmptyInput() {
        let result = VDSPOperations.l2Normalize([])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Dot Product

    func testDotProductKnownValues() {
        let result = VDSPOperations.dotProduct([1, 2, 3], [4, 5, 6])
        XCTAssertEqual(result, 32, accuracy: 1e-5, "[1,2,3]·[4,5,6] = 4+10+18 = 32")
    }

    func testDotProductOrthogonalVectorsReturnZero() {
        let result = VDSPOperations.dotProduct([1, 0], [0, 1])
        XCTAssertEqual(result, 0, accuracy: 1e-5)
    }

    // MARK: - Matrix-Vector Multiply

    func testMatrixVectorMultiplyIdentity() {
        let identity: [[Float]] = [[1, 0], [0, 1]]
        let vector: [Float] = [3, 7]
        let result = VDSPOperations.matrixVectorMultiply(matrix: identity, vector: vector)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 3, accuracy: 1e-5)
        XCTAssertEqual(result[1], 7, accuracy: 1e-5)
    }

    func testMatrixVectorMultiplyNonTrivial() {
        // [[1, 2, 3], [4, 5, 6]] * [1, 2, 3] = [14, 32]
        let matrix: [[Float]] = [[1, 2, 3], [4, 5, 6]]
        let vector: [Float] = [1, 2, 3]
        let result = VDSPOperations.matrixVectorMultiply(matrix: matrix, vector: vector)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 14, accuracy: 1e-5, "1*1 + 2*2 + 3*3 = 14")
        XCTAssertEqual(result[1], 32, accuracy: 1e-5, "4*1 + 5*2 + 6*3 = 32")
    }

    func testMatrixVectorMultiplyEmptyMatrix() {
        let result = VDSPOperations.matrixVectorMultiply(matrix: [], vector: [1, 2])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Matrix Multiply

    func testMatrixMultiply2x2() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a: [[Float]] = [[1, 2], [3, 4]]
        let b: [[Float]] = [[5, 6], [7, 8]]
        let result = VDSPOperations.matrixMultiply(a: a, b: b)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0][0], 19, accuracy: 1e-4)
        XCTAssertEqual(result[0][1], 22, accuracy: 1e-4)
        XCTAssertEqual(result[1][0], 43, accuracy: 1e-4)
        XCTAssertEqual(result[1][1], 50, accuracy: 1e-4)
    }

    func testMatrixMultiplyEmptyReturnsEmpty() {
        let result = VDSPOperations.matrixMultiply(a: [], b: [[1]])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Softmax

    func testSoftmaxSumsToOne() {
        let result = VDSPOperations.softmax([1, 2, 3])
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax output must sum to 1")
        XCTAssertTrue(result[0] < result[1] && result[1] < result[2], "Softmax preserves ordering")
    }

    func testSoftmaxUniformInput() {
        let result = VDSPOperations.softmax([0, 0, 0])
        let expected: Float = 1.0 / 3.0
        for value in result {
            XCTAssertEqual(value, expected, accuracy: 1e-5, "Equal inputs should give uniform distribution")
        }
    }

    func testSoftmaxEmpty() {
        let result = VDSPOperations.softmax([])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - LogSumExp

    func testLogSumExpMatchesNaive() {
        let input: [Float] = [1, 2, 3]
        let result = VDSPOperations.logSumExp(input)
        let naiveResult = log(exp(Float(1)) + exp(Float(2)) + exp(Float(3)))
        XCTAssertEqual(result, naiveResult, accuracy: 1e-4)
    }

    func testLogSumExpNumericalStability() {
        // With large values, naive computation would overflow. logSumExp should handle it.
        let input: [Float] = [1000, 1001, 1002]
        let result = VDSPOperations.logSumExp(input)

        // log(e^1000 + e^1001 + e^1002) = 1002 + log(e^-2 + e^-1 + 1)
        let expected = Float(1002) + log(exp(Float(-2)) + exp(Float(-1)) + 1)
        XCTAssertEqual(result, expected, accuracy: 1e-2)
        XCTAssertFalse(result.isNaN, "Should not produce NaN for large values")
        XCTAssertFalse(result.isInfinite, "Should not produce Inf for large values")
    }

    // MARK: - Sum

    func testSumFloat() {
        XCTAssertEqual(VDSPOperations.sum([1.0, 2.0, 3.0] as [Float]), 6.0, accuracy: 1e-5)
        XCTAssertEqual(VDSPOperations.sum([] as [Float]), 0)
    }

    func testSumDouble() {
        XCTAssertEqual(VDSPOperations.sum([1.0, 2.0, 3.0] as [Double]), 6.0, accuracy: 1e-10)
        XCTAssertEqual(VDSPOperations.sum([] as [Double]), 0)
    }

    // MARK: - Pairwise Euclidean Distances

    func testPairwiseEuclideanDistancesSelfIsZero() {
        let vectors: [[Float]] = [[1, 0], [0, 1]]
        let result = VDSPOperations.pairwiseEuclideanDistances(a: vectors, b: vectors)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].count, 2)
        XCTAssertEqual(result[0][0], 0, accuracy: 1e-4, "Distance to self should be 0")
        XCTAssertEqual(result[1][1], 0, accuracy: 1e-4, "Distance to self should be 0")
    }

    func testPairwiseEuclideanDistancesOrthogonal() {
        let a: [[Float]] = [[1, 0]]
        let b: [[Float]] = [[0, 1]]
        let result = VDSPOperations.pairwiseEuclideanDistances(a: a, b: b)

        let expected = sqrt(Float(2))
        XCTAssertEqual(result[0][0], expected, accuracy: 1e-4)
    }

    func testPairwiseEuclideanDistancesSymmetric() {
        let a: [[Float]] = [[1, 2, 3]]
        let b: [[Float]] = [[4, 5, 6]]

        let distAB = VDSPOperations.pairwiseEuclideanDistances(a: a, b: b)
        let distBA = VDSPOperations.pairwiseEuclideanDistances(a: b, b: a)

        XCTAssertEqual(distAB[0][0], distBA[0][0], accuracy: 1e-4, "Distance should be symmetric")
    }

    func testPairwiseEuclideanDistancesEmptyReturnsEmpty() {
        let result = VDSPOperations.pairwiseEuclideanDistances(a: [], b: [[1, 2]])
        XCTAssertTrue(result.isEmpty)
    }
}
