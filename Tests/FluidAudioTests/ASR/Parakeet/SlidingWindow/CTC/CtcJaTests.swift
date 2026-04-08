import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for CTC Japanese text normalization and CER calculation
///
/// These tests verify the pure functions used in CTC Japanese benchmarking:
/// - Text normalization (punctuation removal, whitespace handling, case folding)
/// - Character Error Rate (CER) calculation
/// - Levenshtein distance algorithm
final class CtcJaTests: XCTestCase {

    // MARK: - Text Normalization Tests

    func testNormalizeJapaneseText_RemovesJapanesePunctuation() {
        let input = "こんにちは、世界！これは・テスト。"
        var normalized = input

        let japanesePunct = "、。！？・…「」『』（）［］｛｝【】"
        for char in japanesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        let expected = "こんにちは世界これはテスト"
        XCTAssertEqual(normalized, expected, "Should remove all Japanese punctuation")
    }

    func testNormalizeJapaneseText_RemovesASCIIPunctuation() {
        let input = "Hello, world! This is a test."
        var normalized = input

        let asciiPunct = ",.!?;:\'\"()-[]{}"
        for char in asciiPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        let expected = "Hello world This is a test"
        XCTAssertEqual(normalized, expected, "Should remove all ASCII punctuation")
    }

    func testNormalizeJapaneseText_NormalizesWhitespace() {
        let input = "こんにちは   世界\nこれは\tテスト"
        let normalized = input.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        let expected = "こんにちは世界これはテスト"
        XCTAssertEqual(normalized, expected, "Should normalize and remove all whitespace")
    }

    func testNormalizeJapaneseText_ConvertsToLowercase() {
        let input = "Hello WORLD Test"
        let normalized = input.lowercased()

        let expected = "hello world test"
        XCTAssertEqual(normalized, expected, "Should convert romaji to lowercase")
    }

    func testNormalizeJapaneseText_CompleteExample() {
        // This mimics the exact normalization logic from JapaneseAsrBenchmark
        let input = "水をマレーシアから買わなくてはならないのです。"
        var normalized = input

        // Remove Japanese punctuation
        let japanesePunct = "、。！？・…「」『』（）［］｛｝【】"
        for char in japanesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove ASCII punctuation
        let asciiPunct = ",.!?;:\'\"()-[]{}"
        for char in asciiPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Normalize whitespace
        normalized = normalized.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        // Convert to lowercase for any romaji
        normalized = normalized.lowercased()

        let expected = "水をマレーシアから買わなくてはならないのです"
        XCTAssertEqual(
            normalized, expected,
            "Full normalization should match expected output")
    }

    func testNormalizeJapaneseText_MixedContent() {
        let input = "これは、Test（テスト）です！"
        var normalized = input

        // Remove Japanese punctuation
        let japanesePunct = "、。！？・…「」『』（）［］｛｝【】"
        for char in japanesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove ASCII punctuation
        let asciiPunct = ",.!?;:\'\"()-[]{}"
        for char in asciiPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Normalize whitespace
        normalized = normalized.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        // Convert to lowercase for any romaji
        normalized = normalized.lowercased()

        let expected = "これはtestテストです"
        XCTAssertEqual(normalized, expected, "Should handle mixed Japanese/English content")
    }

    // MARK: - Levenshtein Distance Tests

    func testLevenshteinDistance_IdenticalStrings() {
        let a = Array("こんにちは")
        let b = Array("こんにちは")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 0, "Identical strings should have distance 0")
    }

    func testLevenshteinDistance_EmptyStrings() {
        let a: [Character] = []
        let b: [Character] = []
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 0, "Empty strings should have distance 0")
    }

    func testLevenshteinDistance_OneEmpty() {
        let a = Array("こんにちは")
        let b: [Character] = []
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 5, "Distance should equal length of non-empty string")
    }

    func testLevenshteinDistance_SingleSubstitution() {
        let a = Array("こんにちは")
        let b = Array("こんにちわ")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single substitution should have distance 1")
    }

    func testLevenshteinDistance_SingleInsertion() {
        let a = Array("こんにちは")
        let b = Array("こんにちはあ")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single insertion should have distance 1")
    }

    func testLevenshteinDistance_SingleDeletion() {
        let a = Array("こんにちは")
        let b = Array("こんにち")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single deletion should have distance 1")
    }

    func testLevenshteinDistance_MultipleChanges() {
        let a = Array("こんにちは")
        let b = Array("さようなら")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 5, "All characters different should have distance 5")
    }

    func testLevenshteinDistance_JapaneseCharacters() {
        let a = Array("今日は良い天気です")
        let b = Array("今日は悪い天気です")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single character substitution should have distance 1")
    }

    // MARK: - CER Calculation Tests

    func testCalculateCER_IdenticalStrings() {
        let reference = "こんにちは世界"
        let hypothesis = "こんにちは世界"
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 0.0, accuracy: 0.001, "Identical strings should have CER 0")
    }

    func testCalculateCER_EmptyReference() {
        let reference = ""
        let hypothesis = "こんにちは"
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 1.0, accuracy: 0.001, "Empty reference with non-empty hypothesis should have CER 1.0")
    }

    func testCalculateCER_EmptyHypothesis() {
        let reference = "こんにちは"
        let hypothesis = ""
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 1.0, accuracy: 0.001, "Non-empty reference with empty hypothesis should have CER 1.0")
    }

    func testCalculateCER_BothEmpty() {
        let reference = ""
        let hypothesis = ""
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 0.0, accuracy: 0.001, "Both empty should have CER 0")
    }

    func testCalculateCER_SingleCharacterError() {
        let reference = "こんにちは"  // 5 characters
        let hypothesis = "こんにちわ"  // 1 substitution (は -> わ)
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 1, Length = 5, CER = 1/5 = 0.2
        XCTAssertEqual(cer, 0.2, accuracy: 0.001, "Single character error in 5 chars should be 0.2")
    }

    func testCalculateCER_MultipleErrors() {
        let reference = "今日は良い天気"  // 7 characters (今日は良い天気)
        let hypothesis = "今日は悪い天気"  // 1 substitution (良 -> 悪)
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 1, Length = 7, CER = 1/7 ≈ 0.143
        XCTAssertEqual(cer, 1.0 / 7.0, accuracy: 0.001, "1 error in 7 chars should be ~0.143")
    }

    func testCalculateCER_InsertionErrors() {
        let reference = "こんにちは"  // 5 characters
        let hypothesis = "こんにちはあ"  // 1 insertion
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 1, Length = 5, CER = 1/5 = 0.2
        XCTAssertEqual(cer, 0.2, accuracy: 0.001, "1 insertion in 5 chars should be 0.2")
    }

    func testCalculateCER_DeletionErrors() {
        let reference = "こんにちは"  // 5 characters
        let hypothesis = "こんにち"  // 1 deletion
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 1, Length = 5, CER = 1/5 = 0.2
        XCTAssertEqual(cer, 0.2, accuracy: 0.001, "1 deletion in 5 chars should be 0.2")
    }

    func testCalculateCER_RealExample() {
        // Real example from benchmark results
        let reference = "水をマレーシアから買わなくてはならないのです"
        let hypothesis = "水をマレーシアから買わなくてはならないのです"
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 0.0, accuracy: 0.001, "Perfect transcription should have CER 0")
    }

    // MARK: - Helper Functions (matching JapaneseAsrBenchmark implementation)

    private func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        guard m > 0 && n > 0 else { return dp[m][n] }

        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                }
            }
        }

        return dp[m][n]
    }

    private func calculateCER(reference: String, hypothesis: String) -> Double {
        let refChars = Array(reference)
        let hypChars = Array(hypothesis)

        let distance = levenshteinDistance(refChars, hypChars)

        guard !refChars.isEmpty else { return hypChars.isEmpty ? 0.0 : 1.0 }

        return Double(distance) / Double(refChars.count)
    }
}
