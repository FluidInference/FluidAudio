import XCTest

@testable import FluidAudio

final class EmbedSpanWindowTests: XCTestCase {

    func testShortSpanIsRepeatedWithoutZeroPadding() {
        let window = OfflineEmbeddingExtractor.tiledWindow([1, 2, 3], length: 8)

        XCTAssertEqual(window, [1, 2, 3, 1, 2, 3, 1, 2])
    }

    func testSpanFillingTheWindowIsUnchanged() {
        let span: [Float] = [0.5, -0.5, 0.25, -0.25]

        XCTAssertEqual(OfflineEmbeddingExtractor.tiledWindow(span, length: 4), span)
    }

    func testOneSecondSpanFillsATenSecondWindow() {
        let span = (0..<16_000).map { Float($0 % 7 + 1) }

        let window = OfflineEmbeddingExtractor.tiledWindow(span, length: 160_000)

        XCTAssertEqual(window.count, 160_000)
        XCTAssertFalse(window.contains(0))
        XCTAssertEqual(Array(window[144_000...]), span)
    }

    func testSpanWithinOneWindowIsOnePiece() {
        XCTAssertEqual(OfflineEmbeddingExtractor.spanPieces(100..<260, windowLength: 160), [100..<260])
    }

    func testLongSpanIsCutIntoEqualPiecesOfAtMostOneWindow() {
        let pieces = OfflineEmbeddingExtractor.spanPieces(0..<25, windowLength: 10)

        XCTAssertEqual(pieces, [0..<8, 8..<16, 16..<25])
        XCTAssertTrue(pieces.allSatisfy { $0.count <= 10 })
    }

    func testEmptySpanHasNoPieces() {
        XCTAssertEqual(OfflineEmbeddingExtractor.spanPieces(5..<5, windowLength: 10), [])
    }

    func testSpanOfExactlyOneWindowIsOnePiece() {
        XCTAssertEqual(OfflineEmbeddingExtractor.spanPieces(0..<10, windowLength: 10), [0..<10])
    }

    func testSpanOfExactlyTwoWindowsIsTwoPieces() {
        XCTAssertEqual(OfflineEmbeddingExtractor.spanPieces(0..<20, windowLength: 10), [0..<10, 10..<20])
    }

    // MARK: - Sample ranges of caller-provided spans

    private func sampleRange(_ start: Double, _ end: Double) throws -> Range<Int> {
        try OfflineEmbeddingExtractor.sampleRange(
            startSeconds: start, endSeconds: end, sampleRate: 16_000, sampleCount: 160_000)
    }

    private func assertInvalidSpan(_ start: Double, _ end: Double, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try sampleRange(start, end), file: file, line: line) { error in
            guard case OfflineDiarizationError.invalidSpan = error else {
                return XCTFail("expected invalidSpan, got \(error)", file: file, line: line)
            }
        }
    }

    func testSpanIsConvertedToSamples() throws {
        XCTAssertEqual(try sampleRange(1, 2.5), 16_000..<40_000)
    }

    func testSpanIsClampedToTheAudio() throws {
        XCTAssertEqual(try sampleRange(-1, 20), 0..<160_000)
        XCTAssertEqual(try sampleRange(9, .greatestFiniteMagnitude), 144_000..<160_000)
    }

    func testNonFiniteSpanThrowsInsteadOfTrapping() {
        assertInvalidSpan(.nan, 1)
        assertInvalidSpan(0, .infinity)
        assertInvalidSpan(-.infinity, 1)
    }

    func testReversedSpanThrows() {
        assertInvalidSpan(3, 2)
    }

    func testSpanStartingAfterTheAudioIsOutOfRange() {
        assertInvalidSpan(10, 12)
        assertInvalidSpan(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
    }

    func testZeroLengthSpanIsEmpty() {
        XCTAssertThrowsError(try sampleRange(2, 2)) { error in
            guard case OfflineDiarizationError.processingFailed = error else {
                return XCTFail("expected processingFailed, got \(error)")
            }
        }
    }
}
