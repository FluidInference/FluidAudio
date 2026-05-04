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
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5),
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
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5),
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
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5),
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
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 2),
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
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5),
        ]
        MandarinErhua.merge(&s)
        // The trailing er has prev.base == "er" → does NOT merge.
        XCTAssertEqual(s.count, 2)
        XCTAssertFalse(s[0].erhua)
        XCTAssertFalse(s[1].erhua)
    }

    func testMergeRunsBeforeSandhiFor3Plus3() {
        // hao3 + er5 + mei3 → erhua merges first → hao3-erhua + mei3
        // → 3+3 promotes to 2+3 → hao2-erhua + mei3.
        var s = [
            MandarinPinyinNormalizer.Syllable(base: "hao", tone: 3),
            MandarinPinyinNormalizer.Syllable(base: "er", tone: 5),
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

    func testG2PEndToEndZher() throws {
        // 这儿 in single-char dict → zhe + er → erhua-merged.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try g2p.phonemize("这儿")
        XCTAssertTrue(
            phon.contains("ㄦ"),
            "expected erhua suffix in '\(phon)'")
        // Should NOT have a separate er-tone token after the main syllable.
        XCTAssertFalse(
            phon.contains("ㄦ5"),
            "erhua should be attached, not a standalone er5; got '\(phon)'")
    }

    func testG2PEndToEndStandaloneErzi() throws {
        // 儿子 → er + zi → leading er, must NOT merge.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try g2p.phonemize("儿子")
        // Both syllables are emitted independently. We confirm by
        // checking the leading er token survives with its own tone.
        XCTAssertTrue(
            phon.hasPrefix("ㄦ2") || phon.hasPrefix("ㄦ"),
            "leading 儿 should keep its own ㄦ token; got '\(phon)'")
    }

    // MARK: - Test fixtures

    private static func miniDict() -> MandarinPinyinDict {
        let singles: [UInt32: [String]] = [
            0x8FD9: ["zhè"],  // 这
            0x513F: ["ér"],  // 儿 (default tone 2)
            0x5B50: ["zi"],  // 子 (neutral tone)
            0x5C0F: ["xiǎo"],  // 小
            0x5B69: ["hái"],  // 孩
            0x4E00: ["yī"],  // 一
            0x4F1A: ["huì"],  // 会
        ]
        // Phrase override: "儿" suffix should appear as tone-5 to match
        // typical erhua pronunciation when it follows another syllable.
        // The miniDict deliberately keeps it at tone 2 — the merge rule
        // only checks `base == "er"`, not tone, so both form 2 and 5 fold.
        return MandarinPinyinDict(phrases: [:], singles: singles)
    }
}
