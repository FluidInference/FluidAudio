import XCTest

@testable import FluidAudio

final class TokenUtilsTests: XCTestCase {

    // MARK: - isWordBoundary

    func testSentencePieceBoundary() {
        XCTAssertTrue(isWordBoundary("▁hello"))
    }

    func testSpaceBoundary() {
        XCTAssertTrue(isWordBoundary(" hello"))
    }

    func testNoBoundary() {
        XCTAssertFalse(isWordBoundary("hello"))
    }

    func testEmptyString() {
        XCTAssertFalse(isWordBoundary(""))
    }

    func testOnlySentencePiecePrefix() {
        XCTAssertTrue(isWordBoundary("▁"))
    }

    func testOnlySpacePrefix() {
        XCTAssertTrue(isWordBoundary(" "))
    }

    // MARK: - stripWordBoundaryPrefix

    func testStripSentencePiecePrefix() {
        XCTAssertEqual(stripWordBoundaryPrefix("▁hello"), "hello")
    }

    func testStripSpacePrefix() {
        XCTAssertEqual(stripWordBoundaryPrefix(" hello"), "hello")
    }

    func testStripNoPrefixIsIdentity() {
        XCTAssertEqual(stripWordBoundaryPrefix("hello"), "hello")
    }

    func testStripEmptyString() {
        XCTAssertEqual(stripWordBoundaryPrefix(""), "")
    }

    func testStripOnlyPrefixGivesEmpty() {
        XCTAssertEqual(stripWordBoundaryPrefix("▁"), "")
    }

    func testStripDoesNotRemoveInternalMarkers() {
        // Only strips leading prefix, not internal occurrences
        XCTAssertEqual(stripWordBoundaryPrefix("he▁llo"), "he▁llo")
    }
}
