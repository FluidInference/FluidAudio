import XCTest

@testable import FluidAudio

final class LSEENDMatrixTests: XCTestCase {

    // MARK: - Init (validated)

    func testInitWithMatchingDimensions() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(m.rows, 2)
        XCTAssertEqual(m.columns, 3)
        XCTAssertEqual(m.values, [1, 2, 3, 4, 5, 6])
    }

    func testInitThrowsOnCountMismatch() {
        XCTAssertThrowsError(try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3])) { error in
            guard case LSEENDError.invalidMatrixShape = error else {
                return XCTFail("Expected invalidMatrixShape, got \(error)")
            }
        }
    }

    func testInitThrowsOnNegativeRows() {
        XCTAssertThrowsError(try LSEENDMatrix(rows: -1, columns: 3, values: [])) { error in
            guard case LSEENDError.invalidMatrixShape = error else {
                return XCTFail("Expected invalidMatrixShape, got \(error)")
            }
        }
    }

    func testInitThrowsOnNegativeColumns() {
        XCTAssertThrowsError(try LSEENDMatrix(rows: 2, columns: -1, values: [])) { error in
            guard case LSEENDError.invalidMatrixShape = error else {
                return XCTFail("Expected invalidMatrixShape, got \(error)")
            }
        }
    }

    func testInitWithZeroDimensions() throws {
        let m = try LSEENDMatrix(rows: 0, columns: 5, values: [])
        XCTAssertEqual(m.rows, 0)
        XCTAssertEqual(m.columns, 5)
        XCTAssertTrue(m.isEmpty)
    }

    // MARK: - Factory Methods

    func testZeros() {
        let m = LSEENDMatrix.zeros(rows: 3, columns: 2)
        XCTAssertEqual(m.rows, 3)
        XCTAssertEqual(m.columns, 2)
        XCTAssertEqual(m.values, [Float](repeating: 0, count: 6))
    }

    func testEmpty() {
        let m = LSEENDMatrix.empty(columns: 4)
        XCTAssertEqual(m.rows, 0)
        XCTAssertEqual(m.columns, 4)
        XCTAssertTrue(m.isEmpty)
    }

    // MARK: - isEmpty

    func testIsEmptyZeroRows() {
        XCTAssertTrue(LSEENDMatrix.empty(columns: 3).isEmpty)
    }

    func testIsEmptyZeroColumns() {
        let m = LSEENDMatrix(validatingRows: 3, columns: 0, values: [])
        XCTAssertTrue(m.isEmpty)
    }

    func testIsEmptyFalseForPopulatedMatrix() throws {
        let m = try LSEENDMatrix(rows: 1, columns: 1, values: [42])
        XCTAssertFalse(m.isEmpty)
    }

    // MARK: - Subscript

    func testSubscriptGet() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [10, 20, 30, 40, 50, 60])
        XCTAssertEqual(m[0, 0], 10)
        XCTAssertEqual(m[0, 2], 30)
        XCTAssertEqual(m[1, 0], 40)
        XCTAssertEqual(m[1, 2], 60)
    }

    func testSubscriptSet() throws {
        var m = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        m[1, 0] = 99
        XCTAssertEqual(m[1, 0], 99)
        XCTAssertEqual(m.values, [1, 2, 99, 4])
    }

    // MARK: - row()

    func testRow() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(Array(m.row(0)), [1, 2])
        XCTAssertEqual(Array(m.row(1)), [3, 4])
        XCTAssertEqual(Array(m.row(2)), [5, 6])
    }

    // MARK: - prefixingColumns

    func testPrefixingColumns() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 4, values: [1, 2, 3, 4, 5, 6, 7, 8])
        let prefix = m.prefixingColumns(2)
        XCTAssertEqual(prefix.rows, 2)
        XCTAssertEqual(prefix.columns, 2)
        XCTAssertEqual(prefix.values, [1, 2, 5, 6])
    }

    func testPrefixingColumnsEqualToWidth() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let same = m.prefixingColumns(3)
        XCTAssertEqual(same, m)
    }

    func testPrefixingColumnsGreaterThanWidth() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let same = m.prefixingColumns(10)
        XCTAssertEqual(same, m)
    }

    func testPrefixingColumnsZero() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let empty = m.prefixingColumns(0)
        XCTAssertTrue(empty.isEmpty)
    }

    func testPrefixingColumnsOnEmptyMatrix() {
        let m = LSEENDMatrix.empty(columns: 4)
        let result = m.prefixingColumns(2)
        XCTAssertTrue(result.isEmpty)
        XCTAssertEqual(result.columns, 2)
    }

    // MARK: - rowMajorRows

    func testRowMajorRows() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 3, values: [1, 2, 3, 4, 5, 6])
        let rows = m.rowMajorRows()
        XCTAssertEqual(rows, [[1, 2, 3], [4, 5, 6]])
    }

    func testRowMajorRowsEmpty() {
        let m = LSEENDMatrix.empty(columns: 3)
        XCTAssertEqual(m.rowMajorRows(), [])
    }

    // MARK: - appendingRows

    func testAppendingRows() throws {
        let a = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let b = try LSEENDMatrix(rows: 1, columns: 2, values: [5, 6])
        let result = a.appendingRows(b)
        XCTAssertEqual(result.rows, 3)
        XCTAssertEqual(result.columns, 2)
        XCTAssertEqual(result.values, [1, 2, 3, 4, 5, 6])
    }

    func testAppendingRowsToEmpty() throws {
        let a = LSEENDMatrix.empty(columns: 2)
        let b = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        XCTAssertEqual(a.appendingRows(b), b)
    }

    func testAppendingEmptyRows() throws {
        let a = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let b = LSEENDMatrix.empty(columns: 2)
        XCTAssertEqual(a.appendingRows(b), a)
    }

    // MARK: - droppingFirstRows

    func testDroppingFirstRows() throws {
        let m = try LSEENDMatrix(rows: 4, columns: 2, values: [1, 2, 3, 4, 5, 6, 7, 8])
        let dropped = m.droppingFirstRows(2)
        XCTAssertEqual(dropped.rows, 2)
        XCTAssertEqual(dropped.columns, 2)
        XCTAssertEqual(dropped.values, [5, 6, 7, 8])
    }

    func testDroppingAllRows() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let dropped = m.droppingFirstRows(3)
        XCTAssertEqual(dropped.rows, 0)
        XCTAssertTrue(dropped.isEmpty)
    }

    func testDroppingMoreThanTotalRows() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let dropped = m.droppingFirstRows(100)
        XCTAssertEqual(dropped.rows, 0)
        XCTAssertTrue(dropped.isEmpty)
    }

    func testDroppingZeroRows() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let same = m.droppingFirstRows(0)
        XCTAssertEqual(same, m)
    }

    func testDroppingNegativeCount() throws {
        let m = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let same = m.droppingFirstRows(-5)
        XCTAssertEqual(same, m)
    }

    // MARK: - slicingRows

    func testSlicingRows() throws {
        let m = try LSEENDMatrix(rows: 5, columns: 2, values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        let slice = m.slicingRows(start: 1, end: 4)
        XCTAssertEqual(slice.rows, 3)
        XCTAssertEqual(slice.values, [3, 4, 5, 6, 7, 8])
    }

    func testSlicingRowsFullRange() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let slice = m.slicingRows(start: 0, end: 3)
        XCTAssertEqual(slice, m)
    }

    func testSlicingRowsEmptyRange() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let slice = m.slicingRows(start: 2, end: 2)
        XCTAssertTrue(slice.isEmpty)
        XCTAssertEqual(slice.columns, 2)
    }

    func testSlicingRowsClampsOutOfBounds() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let slice = m.slicingRows(start: -5, end: 100)
        XCTAssertEqual(slice, m)
    }

    func testSlicingRowsInvertedRange() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let slice = m.slicingRows(start: 3, end: 1)
        XCTAssertTrue(slice.isEmpty)
    }

    // MARK: - applyingSigmoid

    func testSigmoidZero() throws {
        let m = try LSEENDMatrix(rows: 1, columns: 1, values: [0])
        let s = m.applyingSigmoid()
        XCTAssertEqual(s[0, 0], 0.5, accuracy: 1e-6)
    }

    func testSigmoidLargePositive() throws {
        let m = try LSEENDMatrix(rows: 1, columns: 1, values: [20])
        let s = m.applyingSigmoid()
        XCTAssertEqual(s[0, 0], 1.0, accuracy: 1e-5)
    }

    func testSigmoidLargeNegative() throws {
        let m = try LSEENDMatrix(rows: 1, columns: 1, values: [-20])
        let s = m.applyingSigmoid()
        XCTAssertEqual(s[0, 0], 0.0, accuracy: 1e-5)
    }

    func testSigmoidPreservesShape() throws {
        let m = try LSEENDMatrix(rows: 3, columns: 4, values: [Float](repeating: 0, count: 12))
        let s = m.applyingSigmoid()
        XCTAssertEqual(s.rows, 3)
        XCTAssertEqual(s.columns, 4)
        XCTAssertEqual(s.values.count, 12)
    }

    func testSigmoidDoesNotMutateOriginal() throws {
        let m = try LSEENDMatrix(rows: 1, columns: 2, values: [0, 0])
        _ = m.applyingSigmoid()
        XCTAssertEqual(m.values, [0, 0])
    }

    func testSigmoidOnEmpty() {
        let m = LSEENDMatrix.empty(columns: 3)
        let s = m.applyingSigmoid()
        XCTAssertTrue(s.isEmpty)
    }

    // MARK: - Equatable

    func testEqualMatrices() throws {
        let a = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let b = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        XCTAssertEqual(a, b)
    }

    func testUnequalValues() throws {
        let a = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 4])
        let b = try LSEENDMatrix(rows: 2, columns: 2, values: [1, 2, 3, 5])
        XCTAssertNotEqual(a, b)
    }

    // MARK: - Roundtrip: append then drop

    func testAppendThenDropRecoversOriginal() throws {
        let original = try LSEENDMatrix(rows: 3, columns: 2, values: [1, 2, 3, 4, 5, 6])
        let extra = try LSEENDMatrix(rows: 2, columns: 2, values: [7, 8, 9, 10])
        let combined = original.appendingRows(extra)
        let recovered = combined.slicingRows(start: 0, end: 3)
        XCTAssertEqual(recovered, original)
    }

    func testSliceThenAppendRecombines() throws {
        let m = try LSEENDMatrix(rows: 4, columns: 2, values: [1, 2, 3, 4, 5, 6, 7, 8])
        let head = m.slicingRows(start: 0, end: 2)
        let tail = m.slicingRows(start: 2, end: 4)
        let recombined = head.appendingRows(tail)
        XCTAssertEqual(recombined, m)
    }

    func testDropThenPrefixColumnsCommutes() throws {
        let m = try LSEENDMatrix(rows: 4, columns: 4, values: (0..<16).map { Float($0) })

        let dropFirst = m.droppingFirstRows(2).prefixingColumns(2)
        let prefixFirst = m.prefixingColumns(2).droppingFirstRows(2)

        XCTAssertEqual(dropFirst, prefixFirst)
    }
}
