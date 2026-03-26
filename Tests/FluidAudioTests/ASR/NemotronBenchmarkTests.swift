#if os(macOS)
import Foundation
import XCTest

@testable import FluidAudio

/// Tests for WER calculation and text normalization logic
/// Note: NemotronBenchmark class is in FluidAudioCLI (macOS-only CLI tool)
/// These tests validate the WER algorithm that would be used in the benchmark
final class NemotronBenchmarkTests: XCTestCase {

    // MARK: - P0: WER Calculation

    func testWERPerfectMatch() {
        let (errors, words) = calculateWER(reference: "hello world", hypothesis: "hello world")

        XCTAssertEqual(errors, 0)
        XCTAssertEqual(words, 2)
    }

    func testWERSingleSubstitution() {
        let (errors, words) = calculateWER(reference: "hello world", hypothesis: "hello ward")

        XCTAssertEqual(errors, 1)  // "world" → "ward"
        XCTAssertEqual(words, 2)
    }

    func testWERSingleInsertion() {
        let (errors, words) = calculateWER(reference: "hello world", hypothesis: "hello big world")

        XCTAssertEqual(errors, 1)  // "big" inserted
        XCTAssertEqual(words, 2)
    }

    func testWERSingleDeletion() {
        let (errors, words) = calculateWER(reference: "hello big world", hypothesis: "hello world")

        XCTAssertEqual(errors, 1)  // "big" deleted
        XCTAssertEqual(words, 3)
    }

    func testWERMultipleErrors() {
        let (errors, words) = calculateWER(
            reference: "the quick brown fox",
            hypothesis: "the fast brown cat"
        )

        // "quick" → "fast" (substitution)
        // "fox" → "cat" (substitution)
        XCTAssertEqual(errors, 2)
        XCTAssertEqual(words, 4)
    }

    func testWERCompletelyDifferent() {
        let (errors, words) = calculateWER(reference: "hello world", hypothesis: "foo bar")

        XCTAssertEqual(errors, 2)  // All words different
        XCTAssertEqual(words, 2)
    }

    func testWEREmptyHypothesis() {
        let (errors, words) = calculateWER(reference: "hello world", hypothesis: "")

        XCTAssertEqual(errors, 2)  // All words deleted
        XCTAssertEqual(words, 2)
    }

    func testWEREmptyReference() {
        let (errors, words) = calculateWER(reference: "", hypothesis: "hello world")

        XCTAssertEqual(errors, 2)  // All words inserted
        XCTAssertEqual(words, 0)  // No reference words
    }

    func testWERBothEmpty() {
        let (errors, words) = calculateWER(reference: "", hypothesis: "")

        XCTAssertEqual(errors, 0)
        XCTAssertEqual(words, 0)
    }

    func testWERCaseInsensitive() {
        let (errors, words) = calculateWER(reference: "Hello World", hypothesis: "hello world")

        // Should normalize to lowercase
        XCTAssertEqual(errors, 0)
        XCTAssertEqual(words, 2)
    }

    func testWERWithPunctuation() {
        let (errors, words) = calculateWER(
            reference: "Hello, world!",
            hypothesis: "hello world"
        )

        // Punctuation should be removed
        XCTAssertEqual(errors, 0)
        XCTAssertEqual(words, 2)
    }

    func testWERWithExtraWhitespace() {
        let (errors, words) = calculateWER(
            reference: "hello   world",
            hypothesis: "hello world"
        )

        // Extra whitespace normalized
        XCTAssertEqual(errors, 0)
        XCTAssertEqual(words, 2)
    }

    func testWERComplexSentence() {
        let (errors, words) = calculateWER(
            reference: "The quick brown fox jumps over the lazy dog",
            hypothesis: "The fast brown fox jumped over a lazy dog"
        )

        // Differences:
        // "quick" → "fast" (1)
        // "jumps" → "jumped" (1)
        // "the" → "a" (1)
        XCTAssertEqual(errors, 3)
        XCTAssertEqual(words, 9)
    }

    // MARK: - P0: Text Normalization

    func testNormalizeTextLowercase() {
        let normalized = normalizeText("Hello World")
        XCTAssertEqual(normalized, "hello world")
    }

    func testNormalizeTextRemovesPunctuation() {
        let normalized = normalizeText("Hello, world! How's it going?")
        XCTAssertEqual(normalized, "hello world how s it going")
    }

    func testNormalizeTextCollapsesWhitespace() {
        let normalized = normalizeText("hello    world")
        XCTAssertEqual(normalized, "hello world")
    }

    func testNormalizeTextRemovesLeadingTrailingWhitespace() {
        let normalized = normalizeText("  hello world  ")
        XCTAssertEqual(normalized, "hello world")
    }

    func testNormalizeTextHandlesNumbers() {
        let normalized = normalizeText("I have 3 apples")
        XCTAssertEqual(normalized, "i have 3 apples")
    }

    func testNormalizeTextHandlesEmptyString() {
        let normalized = normalizeText("")
        XCTAssertEqual(normalized, "")
    }

    func testNormalizeTextHandlesOnlyPunctuation() {
        let normalized = normalizeText("!@#$%")
        XCTAssertEqual(normalized, "")
    }

    func testNormalizeTextHandlesNewlines() {
        let normalized = normalizeText("hello\nworld")
        XCTAssertEqual(normalized, "hello world")
    }

    // MARK: - Helper Methods

    private func calculateWER(reference: String, hypothesis: String) -> (errors: Int, words: Int) {
        let refWords = normalizeText(reference).split(separator: " ").map(String.init)
        let hypWords = normalizeText(hypothesis).split(separator: " ").map(String.init)

        let m = refWords.count
        let n = hypWords.count

        // Handle empty cases early
        if m == 0 && n == 0 {
            return (0, 0)
        }
        if m == 0 {
            return (n, 0)
        }
        if n == 0 {
            return (m, m)
        }

        var d = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { d[i][0] = i }
        for j in 0...n { d[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if refWords[i - 1] == hypWords[j - 1] {
                    d[i][j] = d[i - 1][j - 1]
                } else {
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
                }
            }
        }

        return (d[m][n], m)
    }

    private func normalizeText(_ text: String) -> String {
        let cleaned = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
        return cleaned.split(separator: " ").joined(separator: " ")
    }
}
#endif
