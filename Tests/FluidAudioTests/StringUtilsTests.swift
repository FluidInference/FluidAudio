import XCTest

@testable import FluidAudio

final class StringUtilsTests: XCTestCase {

    // MARK: - levenshteinDistance (String overload)

    func testIdenticalStrings() {
        XCTAssertEqual(StringUtils.levenshteinDistance("hello", "hello"), 0)
    }

    func testEmptyVsNonEmpty() {
        XCTAssertEqual(StringUtils.levenshteinDistance("", "abc"), 3)
        XCTAssertEqual(StringUtils.levenshteinDistance("abc", ""), 3)
    }

    func testBothEmpty() {
        XCTAssertEqual(StringUtils.levenshteinDistance("", ""), 0)
    }

    func testSingleCharacterSubstitution() {
        // "kitten" -> "sitten" (substitute k->s)
        XCTAssertEqual(StringUtils.levenshteinDistance("kitten", "sitten"), 1)
    }

    func testInsertion() {
        XCTAssertEqual(StringUtils.levenshteinDistance("abc", "abcd"), 1)
    }

    func testDeletion() {
        XCTAssertEqual(StringUtils.levenshteinDistance("abcd", "abc"), 1)
    }

    func testCompletelyDifferent() {
        XCTAssertEqual(StringUtils.levenshteinDistance("abc", "xyz"), 3)
    }

    func testKnownNlpExample() {
        // Classic example: kitten -> sitting = 3 edits
        // k->s (sub), e->i (sub), insert g
        XCTAssertEqual(StringUtils.levenshteinDistance("kitten", "sitting"), 3)
    }

    func testCaseSensitive() {
        // Character-level comparison is case-sensitive
        XCTAssertEqual(StringUtils.levenshteinDistance("ABC", "abc"), 3)
    }

    // MARK: - levenshteinDistance (Generic overload)

    func testGenericIntArrays() {
        XCTAssertEqual(StringUtils.levenshteinDistance([1, 2, 3], [1, 3, 3]), 1)
    }

    func testGenericEmptyArrays() {
        XCTAssertEqual(StringUtils.levenshteinDistance([Int](), [1, 2]), 2)
    }

    func testGenericIdenticalArrays() {
        XCTAssertEqual(StringUtils.levenshteinDistance([5, 10, 15], [5, 10, 15]), 0)
    }

    func testSymmetry() {
        let d1 = StringUtils.levenshteinDistance("abc", "xyz")
        let d2 = StringUtils.levenshteinDistance("xyz", "abc")
        XCTAssertEqual(d1, d2)
    }
}
