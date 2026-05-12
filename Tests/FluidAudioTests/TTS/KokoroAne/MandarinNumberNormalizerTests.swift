import Foundation
import XCTest

@testable import FluidAudio

/// Network-free unit tests for `MandarinNumberNormalizer`. The
/// transformation is a pure function with no model dependency, so the
/// suite is exhaustive across the cardinal / decimal / percentage /
/// fraction / currency / date / time rules.
final class MandarinNumberNormalizerTests: XCTestCase {

    // MARK: - Cardinal

    func testCardinalZero() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(0), "零")
    }

    func testCardinalSingleDigit() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(1), "一")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(9), "九")
    }

    func testCardinalTen() {
        // Standalone 10 collapses to 十 (vs intra-number 一十).
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(10), "十")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(12), "十二")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(19), "十九")
    }

    func testCardinalTens() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(20), "二十")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(45), "四十五")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(99), "九十九")
    }

    func testCardinalHundred() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(100), "一百")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(101), "一百零一")
        // Intra-number tens use 一十, not the standalone 十 form.
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(110), "一百一十")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(123), "一百二十三")
    }

    func testCardinalThousand() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(1000), "一千")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(1001), "一千零一")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(1010), "一千零一十")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(2345), "二千三百四十五")
    }

    func testCardinalWan() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(10000), "一万")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(12345), "一万二千三百四十五")
        // Higher group's "10" surfaces as 十 because no group precedes it.
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(100_000), "十万")
        // Cross-group zero gap fills with 零 once.
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(100_001), "十万零一")
    }

    func testCardinalYi() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(100_000_000), "一亿")
        XCTAssertEqual(
            MandarinNumberNormalizer.cardinal(123_456_789),
            "一亿二千三百四十五万六千七百八十九")
    }

    func testCardinalNegative() {
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(-5), "负五")
        XCTAssertEqual(MandarinNumberNormalizer.cardinal(-1234), "负一千二百三十四")
    }

    // MARK: - Decimal

    func testDecimalSimple() {
        XCTAssertEqual(MandarinNumberNormalizer.decimal("3.14"), "三点一四")
    }

    func testDecimalIntegerOnly() {
        XCTAssertEqual(MandarinNumberNormalizer.decimal("42"), "四十二")
    }

    func testDecimalStripsTrailingZeros() {
        // Colloquial Mandarin drops trailing zeros from the fractional part.
        XCTAssertEqual(MandarinNumberNormalizer.decimal("5.50"), "五点五")
        XCTAssertEqual(MandarinNumberNormalizer.decimal("1.00"), "一")
    }

    func testDecimalPreservesInteriorZero() {
        XCTAssertEqual(MandarinNumberNormalizer.decimal("3.05"), "三点零五")
    }

    // MARK: - Digit-by-digit

    func testDigitString() {
        XCTAssertEqual(MandarinNumberNormalizer.digitString("2025"), "二零二五")
        XCTAssertEqual(MandarinNumberNormalizer.digitString("007"), "零零七")
    }

    // MARK: - normalize() — top-level

    func testNormalizeIntegerInline() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("我有3只猫"), "我有三只猫")
    }

    func testNormalizeMultipleIntegers() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("买了10个苹果和5个梨"),
            "买了十个苹果和五个梨")
    }

    func testNormalizeDecimal() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("圆周率是3.14"), "圆周率是三点一四")
    }

    // MARK: - Percentage

    func testNormalizePercentage() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("99%"), "百分之九十九")
    }

    func testNormalizeDecimalPercentage() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("0.5%"), "百分之零点五")
    }

    // MARK: - Fraction

    func testNormalizeFraction() {
        // Denominator first in Chinese: 1/2 → 二分之一.
        XCTAssertEqual(MandarinNumberNormalizer.normalize("1/2"), "二分之一")
        XCTAssertEqual(MandarinNumberNormalizer.normalize("3/4"), "四分之三")
    }

    // MARK: - Money

    func testNormalizeRMB() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("¥120"), "一百二十元")
        // Fullwidth ￥ also matches.
        XCTAssertEqual(MandarinNumberNormalizer.normalize("￥120"), "一百二十元")
    }

    func testNormalizeUSD() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("$5.50"), "五点五美元")
    }

    func testNormalizeEUR() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("€100"), "一百欧元")
    }

    func testNormalizeGBP() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("£25"), "二十五英镑")
    }

    // MARK: - Date

    func testNormalizeChineseDate() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("2025年5月3日"),
            "二零二五年五月三日")
    }

    func testNormalizeChineseDateHao() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("2025年5月3号"),
            "二零二五年五月三日")
    }

    func testNormalizeChineseYearMonth() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("2025年5月"),
            "二零二五年五月")
    }

    func testNormalizeIsoDate() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("2025-05-03"),
            "二零二五年五月三日")
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("2025/05/03"),
            "二零二五年五月三日")
    }

    func testNormalizeYearOnly() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("2025年"), "二零二五年")
    }

    // MARK: - Time

    func testNormalizeHourMinute() {
        XCTAssertEqual(MandarinNumberNormalizer.normalize("8:30"), "八点三十分")
    }

    func testNormalizeHourMinuteSecond() {
        XCTAssertEqual(
            MandarinNumberNormalizer.normalize("23:59:59"),
            "二十三点五十九分五十九秒")
    }

    // MARK: - Round-trip via MandarinG2P (regression check)

    func testBaselineHanziStillWorks() async throws {
        // Sanity: number normalization MUST NOT corrupt pure-Hanzi input
        // that has no numerals.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try await g2p.phonemize("你好")
        XCTAssertFalse(phon.isEmpty)
    }

    func testInlineNumberFlowsThroughG2P() async throws {
        // 5 normalises to 五 → already in dict → emits a syllable.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try await g2p.phonemize("有5个")
        XCTAssertFalse(phon.isEmpty)
    }

    // MARK: - Test fixtures

    private static func miniDict() -> MandarinPinyinDict {
        // Cover the few hanzi used by the round-trip tests above. Real
        // dictionary lives in HF assets; this fixture is intentionally
        // minimal to keep tests offline.
        let singles: [UInt32: [String]] = [
            0x4F60: ["nǐ"],  // 你
            0x597D: ["hǎo"],  // 好
            0x6709: ["yǒu"],  // 有
            0x4E94: ["wǔ"],  // 五
            0x4E2A: ["gè"],  // 个
        ]
        return MandarinPinyinDict(phrases: [:], singles: singles)
    }
}
