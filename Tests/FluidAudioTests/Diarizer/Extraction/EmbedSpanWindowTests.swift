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
}
