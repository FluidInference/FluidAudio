import XCTest

@testable import FluidAudio

final class BKTreeTests: XCTestCase {

    // MARK: - Helpers

    private func makeTerms(_ texts: [String]) -> [CustomVocabularyTerm] {
        texts.map { CustomVocabularyTerm(text: $0) }
    }

    // MARK: - Initialization

    func testEmptyTree() {
        let tree = BKTree(terms: [])
        XCTAssertTrue(tree.isEmpty)
        XCTAssertEqual(tree.count, 0)
    }

    func testSingleTermTree() {
        let tree = BKTree(terms: makeTerms(["nvidia"]))
        XCTAssertFalse(tree.isEmpty)
        XCTAssertEqual(tree.count, 1)
    }

    func testMultipleTermsCount() {
        let tree = BKTree(terms: makeTerms(["nvidia", "intel", "apple", "amd", "qualcomm"]))
        XCTAssertEqual(tree.count, 5)
        XCTAssertFalse(tree.isEmpty)
    }

    // MARK: - Search

    func testExactMatchSearch() {
        let tree = BKTree(terms: makeTerms(["nvidia", "intel", "apple"]))
        let results = tree.search(query: "nvidia", maxDistance: 0)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.distance, 0)
        XCTAssertEqual(results.first?.term.text, "nvidia")
    }

    func testFuzzySearchDistance1() {
        let tree = BKTree(terms: makeTerms(["nvidia"]))
        // "nvida" vs "nvidia" — edit distance 1 (missing 'i' at position 2)
        let results = tree.search(query: "nvida", maxDistance: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertLessThanOrEqual(results.first?.distance ?? 99, 1)
    }

    func testFuzzySearchDistance2() {
        let tree = BKTree(terms: makeTerms(["nvidia"]))
        // "nvda" vs "nvidia" — edit distance 2
        let results = tree.search(query: "nvda", maxDistance: 2)
        XCTAssertEqual(results.count, 1)
    }

    func testFuzzySearchNoMatch() {
        let tree = BKTree(terms: makeTerms(["nvidia"]))
        let results = tree.search(query: "intel", maxDistance: 1)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchMaxDistanceZeroMiss() {
        let tree = BKTree(terms: makeTerms(["hello"]))
        let results = tree.search(query: "helo", maxDistance: 0)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchReturnsMultipleMatches() {
        let tree = BKTree(terms: makeTerms(["cat", "bat", "hat", "car", "dog"]))
        let results = tree.search(query: "cat", maxDistance: 1)
        let matchedTexts = Set(results.map { $0.term.text })
        // "cat" (0), "bat" (1), "hat" (1), "car" (1) are all within distance 1
        XCTAssertTrue(matchedTexts.contains("cat"))
        XCTAssertTrue(matchedTexts.contains("bat"))
        XCTAssertTrue(matchedTexts.contains("hat"))
        XCTAssertTrue(matchedTexts.contains("car"))
        XCTAssertFalse(matchedTexts.contains("dog"))
    }

    func testSearchOnEmptyTree() {
        let tree = BKTree(terms: [])
        let results = tree.search(query: "anything", maxDistance: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testCaseInsensitiveSearch() {
        // BK-tree normalizes to lowercase internally
        let tree = BKTree(terms: makeTerms(["NVIDIA"]))
        let results = tree.search(query: "nvidia", maxDistance: 0)
        XCTAssertEqual(results.count, 1)
    }

    func testSearchResultFields() {
        let tree = BKTree(terms: makeTerms(["NVIDIA"]))
        let results = tree.search(query: "nvidia", maxDistance: 0)
        guard let result = results.first else {
            return XCTFail("Expected one result")
        }
        XCTAssertEqual(result.term.text, "NVIDIA")
        XCTAssertEqual(result.normalizedText, "nvidia")
        XCTAssertEqual(result.distance, 0)
    }

    // MARK: - Edge Cases

    func testSingleCharacterTerms() {
        let tree = BKTree(terms: makeTerms(["a", "b", "c"]))
        let results = tree.search(query: "a", maxDistance: 0)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.term.text, "a")
    }
}
