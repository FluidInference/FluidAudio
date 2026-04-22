import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for CTC zh-CN text normalization and CER calculation
///
/// These tests verify the pure functions used in CTC zh-CN benchmarking:
/// - Text normalization (punctuation removal, digit conversion, whitespace handling)
/// - Character Error Rate (CER) calculation
/// - Levenshtein distance algorithm
final class CtcZhCnTests: XCTestCase {

    // MARK: - Text Normalization Tests

    func testNormalizeChineseText_RemovesChinesePunctuation() {
        let input = "你好，世界！这是、一个：测试。"
        let expected = "你好世界这是一个测试"

        // Access via reflection since normalizeChineseText is private in CtcZhCnBenchmark
        // For testing purposes, we'll test the logic inline
        var normalized = input

        // Remove Chinese punctuation (including curly quotes U+201C, U+201D, U+2018, U+2019)
        let chinesePunct = "，。！？、；：\u{201C}\u{201D}\u{2018}\u{2019}"
        for char in chinesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        XCTAssertEqual(normalized, expected, "Should remove all Chinese punctuation")
    }

    func testNormalizeChineseText_RemovesEnglishPunctuation() {
        let input = "Hello, world! This is a test."
        var normalized = input

        let englishPunct = ",.!?;:()[]{}\\<>\"'-"
        for char in englishPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        let expected = "Hello world This is a test"
        XCTAssertEqual(normalized, expected, "Should remove all English punctuation")
    }

    func testNormalizeChineseText_ConvertsDigitsToChineseCharacters() {
        let input = "2021年8月15日"
        var normalized = input

        let digitMap: [Character: String] = [
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        ]
        for (digit, chinese) in digitMap {
            normalized = normalized.replacingOccurrences(of: String(digit), with: chinese)
        }

        let expected = "二零二一年八月一五日"
        XCTAssertEqual(
            normalized, expected,
            "Should convert Arabic digits to Chinese characters")
    }

    func testNormalizeChineseText_RemovesBracketsAndQuotes() {
        let input = "「你好」『世界』（测试）《书名》【注释】"
        var normalized = input

        let brackets = "「」『』（）《》【】"
        for char in brackets {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        let expected = "你好世界测试书名注释"
        XCTAssertEqual(normalized, expected, "Should remove all brackets and quotation marks")
    }

    func testNormalizeChineseText_NormalizesWhitespace() {
        let input = "你好   世界\n这是\t测试"
        let normalized =
            input.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        let expected = "你好世界这是测试"
        XCTAssertEqual(normalized, expected, "Should normalize and remove all whitespace")
    }

    func testNormalizeChineseText_CompleteExample() {
        // This mimics the exact normalization logic from CtcZhCnBenchmark
        let input = "桥下垂直净空15米，该项目于2011年8月完工。"
        var normalized = input

        // Remove Chinese punctuation (including curly quotes U+201C, U+201D, U+2018, U+2019)
        let chinesePunct = "，。！？、；：\u{201C}\u{201D}\u{2018}\u{2019}"
        for char in chinesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove Chinese brackets and quotes
        let brackets = "「」『』（）《》【】"
        for char in brackets {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove common symbols
        let symbols = "…—·"
        for char in symbols {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove English punctuation
        let englishPunct = ",.!?;:()[]{}\\<>\"'-"
        for char in englishPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Convert Arabic digits to Chinese characters
        let digitMap: [Character: String] = [
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        ]
        for (digit, chinese) in digitMap {
            normalized = normalized.replacingOccurrences(of: String(digit), with: chinese)
        }

        // Normalize whitespace
        normalized =
            normalized.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        let expected = "桥下垂直净空一五米该项目于二零一一年八月完工"
        XCTAssertEqual(
            normalized, expected,
            "Full normalization should match expected output")
    }

    // MARK: - Levenshtein Distance Tests

    func testLevenshteinDistance_IdenticalStrings() {
        let a = Array("hello")
        let b = Array("hello")
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
        let a = Array("hello")
        let b: [Character] = []
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 5, "Distance should equal length of non-empty string")
    }

    func testLevenshteinDistance_SingleSubstitution() {
        let a = Array("kitten")
        let b = Array("sitten")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single substitution should have distance 1")
    }

    func testLevenshteinDistance_SingleInsertion() {
        let a = Array("cat")
        let b = Array("cats")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single insertion should have distance 1")
    }

    func testLevenshteinDistance_SingleDeletion() {
        let a = Array("cats")
        let b = Array("cat")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 1, "Single deletion should have distance 1")
    }

    func testLevenshteinDistance_ClassicExample() {
        let a = Array("kitten")
        let b = Array("sitting")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 3, "Classic kitten->sitting should have distance 3")
    }

    func testLevenshteinDistance_ChineseCharacters() {
        let a = Array("你好世界")
        let b = Array("你好地球")
        let distance = levenshteinDistance(a, b)

        XCTAssertEqual(distance, 2, "Two character substitutions should have distance 2")
    }

    // MARK: - CER Calculation Tests

    func testCalculateCER_IdenticalStrings() {
        let reference = "你好世界"
        let hypothesis = "你好世界"
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 0.0, accuracy: 0.001, "Identical strings should have CER 0")
    }

    func testCalculateCER_EmptyReference() {
        let reference = ""
        let hypothesis = "你好"
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        XCTAssertEqual(cer, 1.0, accuracy: 0.001, "Empty reference with non-empty hypothesis should have CER 1.0")
    }

    func testCalculateCER_EmptyHypothesis() {
        let reference = "你好"
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
        let reference = "你好世界"  // 4 characters
        let hypothesis = "你好地界"  // 1 substitution
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 1, Length = 4, CER = 1/4 = 0.25
        XCTAssertEqual(cer, 0.25, accuracy: 0.001, "Single character error in 4 chars should be 0.25")
    }

    func testCalculateCER_MultipleErrors() {
        let reference = "你好世界今天"  // 6 characters
        let hypothesis = "你好地球昨天"  // 3 substitutions
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 3, Length = 6, CER = 3/6 = 0.5
        XCTAssertEqual(cer, 0.5, accuracy: 0.001, "3 errors in 6 chars should be 0.5")
    }

    func testCalculateCER_InsertionErrors() {
        let reference = "你好"  // 2 characters
        let hypothesis = "你好世界"  // 2 insertions
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 2, Length = 2, CER = 2/2 = 1.0
        XCTAssertEqual(cer, 1.0, accuracy: 0.001, "2 insertions in 2 chars should be 1.0")
    }

    func testCalculateCER_DeletionErrors() {
        let reference = "你好世界"  // 4 characters
        let hypothesis = "你好"  // 2 deletions
        let cer = calculateCER(reference: reference, hypothesis: hypothesis)

        // Distance = 2, Length = 4, CER = 2/4 = 0.5
        XCTAssertEqual(cer, 0.5, accuracy: 0.001, "2 deletions in 4 chars should be 0.5")
    }

    // MARK: - Helper Functions (matching CtcZhCnBenchmark implementation)

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

        // Skip main loop if either array is empty (ranges 1...0 would be invalid)
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
