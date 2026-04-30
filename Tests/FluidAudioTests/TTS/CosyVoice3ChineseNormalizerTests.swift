import XCTest

@testable import FluidAudio

final class CosyVoice3ChineseNormalizerTests: XCTestCase {

    func testContainsChinese() {
        XCTAssertTrue(CosyVoice3ChineseNormalizer.containsChinese("你好"))
        XCTAssertTrue(CosyVoice3ChineseNormalizer.containsChinese("hello 世界"))
        XCTAssertFalse(CosyVoice3ChineseNormalizer.containsChinese("hello world"))
        XCTAssertFalse(CosyVoice3ChineseNormalizer.containsChinese(""))
    }

    func testReplaceBlankDropsCjkInteriorSpaces() {
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.replaceBlank("中 国"), "中国")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.replaceBlank("hello world"), "hello world")
        // Mixed: space between ASCII and CJK is dropped (one side non-ASCII).
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.replaceBlank("hi 你好"), "hi你好")
    }

    func testReplaceCornerMark() {
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.replaceCornerMark("面积 5m²"),
            "面积 5m平方")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.replaceCornerMark("体积 2m³"),
            "体积 2m立方")
    }

    func testRemoveBracket() {
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.removeBracket("你好（世界）"),
            "你好世界")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.removeBracket("【注意】请勿触摸"),
            "注意请勿触摸")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.removeBracket("a——b"),
            "a b")
    }

    func testSpellOutDigitsZh() {
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.spellOutDigitsZh("2024年"),
            "二零二四年")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.spellOutDigitsZh("abc"),
            "abc")
    }

    func testStripTrailingCommaLikes() {
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.stripTrailingCommaLikes("你好，，"),
            "你好。")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.stripTrailingCommaLikes("你好、,，"),
            "你好。")
        XCTAssertEqual(
            CosyVoice3ChineseNormalizer.stripTrailingCommaLikes("你好。"),
            "你好。")
    }

    func testNormalizeEndToEnd() {
        let input = "希望你以后能够做的比我还好用. 2024年,,"
        let out = CosyVoice3ChineseNormalizer.normalize(input)
        // Period becomes 。, trailing commas collapse to a single 。, digits
        // spelled out per-char, internal spaces between CJK stripped.
        XCTAssertEqual(out, "希望你以后能够做的比我还好用。二零二四年。")
    }

    func testIsOnlyPunctuation() {
        XCTAssertTrue(CosyVoice3ChineseNormalizer.isOnlyPunctuation(""))
        XCTAssertTrue(CosyVoice3ChineseNormalizer.isOnlyPunctuation("。，！？"))
        XCTAssertTrue(CosyVoice3ChineseNormalizer.isOnlyPunctuation(".,!?"))
        XCTAssertFalse(CosyVoice3ChineseNormalizer.isOnlyPunctuation("你好"))
        XCTAssertFalse(CosyVoice3ChineseNormalizer.isOnlyPunctuation("abc"))
    }
}
