import Foundation
import XCTest

@testable import FluidAudio

/// Network-free unit tests for `MandarinErhua` — both the in-place
/// merge primitive and the end-to-end behaviour through
/// `MandarinG2P.phonemize`.
final class MandarinErhuaTests: XCTestCase {

    // MARK: - merge() primitives

    func testMergeBasic() {
        // 这儿 (zhe4 + er5) → single zhe-erhua syllable.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "zhe", tone: 4),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5, isErhuaSuffix: true),
        ]
        MandarinErhua.merge(&s)
        XCTAssertEqual(s.count, 1)
        XCTAssertEqual(s[0].base, "zhe")
        XCTAssertEqual(s[0].tone, 4)
        XCTAssertTrue(s[0].erhua)
    }

    func testMergeMultiSyllable() {
        // 小孩儿 (xiao3 + hai2 + er5) → 2 syllables, last erhua.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "xiao", tone: 3),
            MandarinPinyinNormalizer.Syllable(base: "hai", tone: 2),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5, isErhuaSuffix: true),
        ]
        MandarinErhua.merge(&s)
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s[0].base, "xiao")
        XCTAssertFalse(s[0].erhua)
        XCTAssertEqual(s[1].base, "hai")
        XCTAssertEqual(s[1].tone, 2)
        XCTAssertTrue(s[1].erhua)
    }

    func testMergeAttachesToImmediatePredecessor() {
        // 一会儿 (yi1 + hui4 + er5): er attaches to hui, not yi.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "yi", tone: 1),
            MandarinPinyinNormalizer.Syllable(base: "hui", tone: 4),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5, isErhuaSuffix: true),
        ]
        MandarinErhua.merge(&s)
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s[0].base, "yi")
        XCTAssertFalse(s[0].erhua)
        XCTAssertEqual(s[1].base, "hui")
        XCTAssertTrue(s[1].erhua)
    }

    func testStandaloneErAtStartIsKept() {
        // 儿子 (er2 + zi5): er is at index 0 → must NOT merge.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 2, isErhuaSuffix: true),
            MandarinPinyinNormalizer.Syllable(base: "zi", tone: 5),
        ]
        let original = s
        MandarinErhua.merge(&s)
        XCTAssertEqual(s, original, "Leading er must not be merged")
    }

    func testStandaloneErChildrenWord() {
        // 儿童 (er2 + tong2): the er is leading, and even if it weren't,
        // it doesn't appear as a tail so no merge condition fires.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 2),
            MandarinPinyinNormalizer.Syllable(base: "tong", tone: 2),
        ]
        let original = s
        MandarinErhua.merge(&s)
        XCTAssertEqual(s, original)
    }

    func testEmptyAndSingleNoOp() {
        var empty: [MandarinPinyinNormalizer.Syllable] = []
        MandarinErhua.merge(&empty)
        XCTAssertTrue(empty.isEmpty)

        var single = [MandarinPinyinNormalizer.Syllable(base: "ma", tone: 1)]
        MandarinErhua.merge(&single)
        XCTAssertEqual(single.count, 1)
        XCTAssertFalse(single[0].erhua)
    }

    func testBackToBackErErLeftAlone() {
        // Pathological back-to-back er: no second-pass into the first er.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 2),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5, isErhuaSuffix: true),
        ]
        MandarinErhua.merge(&s)
        // The trailing er has prev.base == "er" → does NOT merge, even
        // though it is flagged as a suffix.
        XCTAssertEqual(s.count, 2)
        XCTAssertFalse(s[0].erhua)
        XCTAssertFalse(s[1].erhua)
    }

    func testMergeRunsBeforeSandhiFor3Plus3() {
        // hao3 + er5 + mei3 → erhua merges first → hao3-erhua + mei3
        // → 3+3 promotes to 2+3 → hao2-erhua + mei3.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "hao", tone: 3),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5, isErhuaSuffix: true),
            MandarinPinyinNormalizer.Syllable(base: "mei", tone: 3),
        ]
        MandarinErhua.merge(&s)
        MandarinToneSandhi.apply(&s)
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s[0].base, "hao")
        XCTAssertEqual(s[0].tone, 2)
        XCTAssertTrue(s[0].erhua)
        XCTAssertEqual(s[1].base, "mei")
        XCTAssertEqual(s[1].tone, 3)
    }

    func testMergeIgnoresUnflaggedEr() {
        // 十二 (shi2 + er4): 二 shares the pinyin `er` but isn't a 儿
        // suffix, so it must keep its own syllable.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "shi", tone: 2),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 4),
        ]
        let original = s
        MandarinErhua.merge(&s)
        XCTAssertEqual(s, original, "Only a flagged 儿 may fold into its predecessor")
    }

    // MARK: - isSuffix()

    func testIsSuffixWordFinalEr() {
        XCTAssertTrue(MandarinErhua.isSuffix(word: Array("这儿"), preceding: []))
        XCTAssertTrue(MandarinErhua.isSuffix(word: Array("小孩儿"), preceding: []))
        XCTAssertTrue(MandarinErhua.isSuffix(word: Array("這兒"), preceding: []))
    }

    func testIsSuffixRejectsOtherErCharacters() {
        for ch in ["二", "而", "耳", "尔", "爾"] {
            XCTAssertFalse(
                MandarinErhua.isSuffix(word: Array("十" + ch), preceding: []),
                "\(ch) is not an erhua suffix")
            XCTAssertFalse(
                MandarinErhua.isSuffix(word: Array(ch), preceding: Array("相隔")),
                "standalone \(ch) is not an erhua suffix")
        }
    }

    func testIsSuffixRejectsWordInitialEr() {
        // 儿子 / 儿童: 儿 starts the word, there is nothing in it to fold into.
        XCTAssertFalse(MandarinErhua.isSuffix(word: Array("儿子"), preceding: Array("他")))
        XCTAssertFalse(MandarinErhua.isSuffix(word: Array("儿童"), preceding: []))
    }

    func testIsSuffixStandaloneErNeedsPrecedingHanzi() {
        // The segmenter split 这 | 儿: fold into the preceding word...
        XCTAssertTrue(MandarinErhua.isSuffix(word: ["儿"], preceding: Array("这")))
        // ...but not when 儿 opens the buffer.
        XCTAssertFalse(MandarinErhua.isSuffix(word: ["儿"], preceding: []))
    }

    func testIsSuffixHonoursNotErhua() {
        // 儿 meaning "child" keeps its own syllable, as a phrase or split.
        XCTAssertFalse(MandarinErhua.isSuffix(word: Array("女儿"), preceding: []))
        XCTAssertFalse(MandarinErhua.isSuffix(word: Array("女兒"), preceding: []))
        XCTAssertFalse(MandarinErhua.isSuffix(word: ["儿"], preceding: Array("他的女")))
        XCTAssertFalse(MandarinErhua.isSuffix(word: Array("婴幼儿"), preceding: []))
        XCTAssertFalse(MandarinErhua.isSuffix(word: ["儿"], preceding: Array("红孩")))
    }

    func testIsSuffixHonoursMustErhua() {
        XCTAssertTrue(MandarinErhua.isSuffix(word: Array("媳妇儿"), preceding: []))
        XCTAssertTrue(MandarinErhua.isSuffix(word: Array("范儿"), preceding: []))
    }

    // MARK: - Bopomofo encoding

    func testEncodeAppendsErhuaSuffix() {
        // The erhua suffix sits between the final and the tone digit.
        let nonErhua = MandarinBopomofoMap.encode(syllable: "xiao", tone: 3)
        let erhua = MandarinBopomofoMap.encode(syllable: "xiao", tone: 3, erhua: true)
        XCTAssertNotNil(nonErhua)
        XCTAssertNotNil(erhua)
        // Erhua form contains an extra ㄦ before the tone digit.
        XCTAssertEqual(erhua, (nonErhua ?? "").replacingOccurrences(of: "3", with: "ㄦ3"))
    }

    func testEncodeErhuaOnSimpleFinal() {
        // hai2 + erhua → ㄏㄞㄦ2.
        XCTAssertEqual(
            MandarinBopomofoMap.encode(syllable: "hai", tone: 2, erhua: true),
            "ㄏㄞㄦ2")
    }

    // MARK: - End-to-end through MandarinG2P

    func testG2PEndToEndZher() async throws {
        // 这儿 in single-char dict → zhe + er → erhua-merged.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try await g2p.phonemize("这儿")
        XCTAssertTrue(
            phon.contains("ㄦ"),
            "expected erhua suffix in '\(phon)'")
        // Should NOT have a separate er-tone token after the main syllable.
        XCTAssertFalse(
            phon.contains("ㄦ5"),
            "erhua should be attached, not a standalone er5; got '\(phon)'")
    }

    func testG2PEndToEndStandaloneErzi() async throws {
        // 儿子 → er + zi → leading er, must NOT merge.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try await g2p.phonemize("儿子")
        // Both syllables are emitted independently. We confirm by
        // checking the leading er token survives with its own tone.
        XCTAssertTrue(
            phon.hasPrefix("ㄦ2") || phon.hasPrefix("ㄦ"),
            "leading 儿 should keep its own ㄦ token; got '\(phon)'")
    }

    func testG2PEndToEndNumberKeepsEr() async throws {
        // 十二 / 相隔二千: 二 must not be swallowed by the syllable before it.
        let g2p = MandarinG2P(dict: Self.miniDict())
        let shier = try await g2p.phonemize("十二")
        XCTAssertTrue(shier.hasSuffix("ㄦ4"), "二 should keep its own er4; got '\(shier)'")
        let gerqian = try await g2p.phonemize("相隔二千")
        XCTAssertTrue(gerqian.contains("ㄦ4"), "二 should keep its own er4; got '\(gerqian)'")
    }

    func testG2PEndToEndOtherErCharactersKept() async throws {
        let g2p = MandarinG2P(dict: Self.miniDict())
        for text in ["然而", "偶尔", "木耳"] {
            let phon = try await g2p.phonemize(text)
            let syllableCount = phon.filter(Self.isToneDigit).count
            XCTAssertEqual(syllableCount, 2, "\(text) should stay two syllables; got '\(phon)'")
        }
    }

    func testG2PEndToEndNotErhuaWordKept() async throws {
        // 女儿 reads nǚ ér, not nǚr — whether the dict has it as a phrase
        // or the segmenter splits it into single characters.
        let split = try await MandarinG2P(dict: Self.miniDict()).phonemize("女儿")
        XCTAssertEqual(split.filter(Self.isToneDigit).count, 2, "got '\(split)'")
        let phrase = try await MandarinG2P(dict: Self.miniDict(phrases: ["女儿": ["nǚ", "ér"]]))
            .phonemize("女儿")
        XCTAssertEqual(phrase.filter(Self.isToneDigit).count, 2, "got '\(phrase)'")
    }

    func testG2PEndToEndWordInitialErAfterAnotherWord() async throws {
        // 他儿子: 儿 opens the phrase 儿子, so it must not fold into 他.
        let g2p = MandarinG2P(dict: Self.miniDict(phrases: ["儿子": ["ér", "zi"]]))
        let phon = try await g2p.phonemize("他儿子")
        XCTAssertEqual(phon.filter(Self.isToneDigit).count, 3, "got '\(phon)'")
    }

    func testG2PEndToEndPhraseFinalErStillMerges() async throws {
        // 小孩儿 as a dict phrase: the word-final 儿 still folds.
        let g2p = MandarinG2P(dict: Self.miniDict(phrases: ["小孩儿": ["xiǎo", "hái", "ér"]]))
        let phon = try await g2p.phonemize("小孩儿")
        XCTAssertTrue(phon.hasSuffix("ㄏㄞㄦ2"), "expected erhua on 孩; got '\(phon)'")
    }

    // MARK: - Test fixtures

    /// One tone digit per emitted syllable. ASCII only: v1.1-zh writes some
    /// finals with Hanzi placeholders such as `十`, which Unicode counts as
    /// numbers too.
    private static func isToneDigit(_ c: Character) -> Bool {
        ("1"..."5").contains(c)
    }

    private static func miniDict(phrases: [String: [String]] = [:]) -> MandarinPinyinDict {
        let singles: [UInt32: [String]] = [
            0x8FD9: ["zhè"],  // 这
            0x513F: ["ér"],  // 儿 (default tone 2)
            0x5B50: ["zi"],  // 子 (neutral tone)
            0x5C0F: ["xiǎo"],  // 小
            0x5B69: ["hái"],  // 孩
            0x4E00: ["yī"],  // 一
            0x4F1A: ["huì"],  // 会
            // Characters that read `er` but are not the erhua suffix.
            0x4E8C: ["èr"],  // 二
            0x800C: ["ér"],  // 而
            0x5C14: ["ěr"],  // 尔
            0x8033: ["ěr"],  // 耳
            0x5341: ["shí"],  // 十
            0x76F8: ["xiāng"],  // 相
            0x9694: ["gé"],  // 隔
            0x5343: ["qiān"],  // 千
            0x7136: ["rán"],  // 然
            0x5076: ["ǒu"],  // 偶
            0x6728: ["mù"],  // 木
            0x5973: ["nǚ"],  // 女
            0x4ED6: ["tā"],  // 他
        ]
        // 儿 stays at tone 2 here: whether it folds depends on the
        // character and its word, not on the tone.
        return MandarinPinyinDict(phrases: phrases, singles: singles)
    }
}
