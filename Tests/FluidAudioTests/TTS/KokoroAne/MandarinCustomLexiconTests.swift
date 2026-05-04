import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the user-supplied Mandarin pronunciation override
/// (Issue #572 item 6). Covers:
///   * `MandarinPinyinNormalizer.parseDigitForm` digit-token parsing
///   * `MandarinCustomLexicon` file parser + validation throws
///   * `MandarinCustomLexicon.longestMatch` (user beats dict at equal
///     length, longest user entry wins)
///   * `MandarinG2P.phonemize` end-to-end with a non-empty lexicon
///     (sandhi runs across user/dict boundaries; `@`-bopomofo bypasses)
///   * `merged(with:)` semantics
///
/// All tests are network-free — they build tiny in-memory dicts in the
/// same style as `MandarinG2PTests`.
final class MandarinCustomLexiconTests: XCTestCase {

    // MARK: - parseDigitForm

    func testParseDigitFormBasicTones() {
        let zi4 = MandarinPinyinNormalizer.parseDigitForm("zi4")
        XCTAssertEqual(zi4, .init(base: "zi", tone: 4))

        let jie2 = MandarinPinyinNormalizer.parseDigitForm("jie2")
        XCTAssertEqual(jie2, .init(base: "jie", tone: 2))

        let de5 = MandarinPinyinNormalizer.parseDigitForm("de5")
        XCTAssertEqual(de5, .init(base: "de", tone: 5))
    }

    func testParseDigitFormUmlautCollapsesToV() {
        // Both diacritic ü and the ASCII v form collapse to "v".
        XCTAssertEqual(
            MandarinPinyinNormalizer.parseDigitForm("lü4"),
            .init(base: "lv", tone: 4))
        XCTAssertEqual(
            MandarinPinyinNormalizer.parseDigitForm("lv4"),
            .init(base: "lv", tone: 4))
    }

    func testParseDigitFormUppercaseLowercased() {
        XCTAssertEqual(
            MandarinPinyinNormalizer.parseDigitForm("ZI4"),
            .init(base: "zi", tone: 4))
    }

    func testParseDigitFormRejectsBadInput() {
        // No tone digit
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm("zi"))
        // Bare digit (empty base)
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm("4"))
        // Tone out of range
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm("zi6"))
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm("zi0"))
        // Non-letter inside base
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm("zi-4"))
        // Empty
        XCTAssertNil(MandarinPinyinNormalizer.parseDigitForm(""))
    }

    // MARK: - File parser

    func testParseHandlesCommentsAndBlankLines() throws {
        let lex = try MandarinCustomLexicon.parse(
            """
            # This is a comment
            # Another comment

            字节跳动  zi4 jie2 tiao4 dong4

            # Tail comment
            比亚迪  bi3 ya4 di2
            """)
        XCTAssertEqual(lex.count, 2)
        XCTAssertEqual(lex.maxKeyCharCount, 4)

        guard let bytedance = lex.entries["字节跳动"] else {
            XCTFail("missing 字节跳动")
            return
        }
        XCTAssertEqual(bytedance.count, 4)
        if case .syllable(let s) = bytedance[0] {
            XCTAssertEqual(s, .init(base: "zi", tone: 4))
        } else {
            XCTFail("expected syllable token")
        }
    }

    func testParseAcceptsAtBopomofoEscape() throws {
        let lex = try MandarinCustomLexicon.parse(
            """
            foo  @ㄈㄨ4
            """)
        XCTAssertEqual(lex.count, 1)
        if case .bopomofo(let s) = lex.entries["foo"]?.first {
            XCTAssertEqual(s, "ㄈㄨ4")
        } else {
            XCTFail("expected bopomofo token")
        }
    }

    func testParseRejectsZeroTokens() {
        XCTAssertThrowsError(try MandarinCustomLexicon.parse("foo")) { err in
            XCTAssertTrue(err is KokoroAneError)
        }
    }

    func testParseRejectsDuplicateWord() {
        XCTAssertThrowsError(
            try MandarinCustomLexicon.parse(
                """
                字节  zi4 jie2
                字节  bi3 ya4
                """)
        ) { err in
            XCTAssertTrue(err is KokoroAneError)
        }
    }

    func testParseRejectsInvalidPinyinToken() {
        // No tone digit in token → parseDigitForm returns nil → throw.
        XCTAssertThrowsError(
            try MandarinCustomLexicon.parse("foo  zi"))
    }

    func testParseRejectsUnencodableSyllable() {
        // parseDigitForm accepts "xq4" (well-formed token), but
        // MandarinBopomofoMap can't encode it → throw.
        XCTAssertThrowsError(
            try MandarinCustomLexicon.parse("foo  xq4"))
    }

    func testParseRejectsBareAtToken() {
        XCTAssertThrowsError(
            try MandarinCustomLexicon.parse("foo  @"))
    }

    func testParseRejectsBopomofoWithUnknownChars() {
        // Latin letters aren't in the v1.1-zh bopomofo charset.
        XCTAssertThrowsError(
            try MandarinCustomLexicon.parse("foo  @abc1"))
    }

    func testEmptyLexiconIsNoop() {
        let empty = MandarinCustomLexicon.empty
        XCTAssertTrue(empty.isEmpty)
        XCTAssertEqual(empty.count, 0)
        XCTAssertEqual(empty.maxKeyCharCount, 0)
        let chars = Array("你好")
        XCTAssertNil(empty.longestMatch(in: chars, from: 0))
    }

    // MARK: - longestMatch

    func testLongestMatchPicksLongestEntry() throws {
        let lex = try MandarinCustomLexicon(entries: [
            "字节": ["zi4", "jie2"],
            "字节跳动": ["zi4", "jie2", "tiao4", "dong4"],
        ])
        let chars = Array("字节跳动是公司")
        let hit = lex.longestMatch(in: chars, from: 0)
        XCTAssertEqual(hit?.length, 4, "should pick the 4-char entry over the 2-char one")
    }

    func testLongestMatchHonorsStartOffset() throws {
        let lex = try MandarinCustomLexicon(entries: [
            "字节": ["zi4", "jie2"]
        ])
        let chars = Array("公司字节")
        XCTAssertNil(lex.longestMatch(in: chars, from: 0))
        XCTAssertEqual(lex.longestMatch(in: chars, from: 2)?.length, 2)
    }

    // MARK: - merged

    func testMergedOtherWinsOnCollision() throws {
        let a = try MandarinCustomLexicon(entries: [
            "字节": ["zi4", "jie2"]
        ])
        let b = try MandarinCustomLexicon(entries: [
            "字节": ["zi3", "jie3"],
            "公司": ["gong1", "si1"],
        ])
        let merged = a.merged(with: b)
        XCTAssertEqual(merged.count, 2)
        if case .syllable(let s) = merged.entries["字节"]?.first {
            XCTAssertEqual(s.tone, 3, "b should win over a")
        } else {
            XCTFail("missing 字节 entry")
        }
    }

    // MARK: - MandarinG2P integration

    private static func smallDict() -> MandarinPinyinDict {
        let phrases: [String: [String]] = [
            "你好": ["nǐ", "hǎo"]
        ]
        let singles: [UInt32: [String]] = [
            // 是 shì
            0x662F: ["shì"],
            // 公 gōng
            0x516C: ["gōng"],
            // 司 sī
            0x53F8: ["sī"],
        ]
        return MandarinPinyinDict(phrases: phrases, singles: singles)
    }

    func testPhonemizeUsesUserLexiconForCustomWord() async throws {
        // 字节跳动 isn't in our tiny dict, so without a lexicon the
        // single-char fallback would also miss and produce nothing.
        // With the lexicon installed, the user reading wins.
        let lex = try MandarinCustomLexicon(entries: [
            "字节跳动": ["zi4", "jie2", "tiao4", "dong4"]
        ])
        var g2p = MandarinG2P(dict: Self.smallDict())
        g2p.customLexicon = lex
        let out = try await g2p.phonemize("字节跳动")
        // Each pinyin runs through the standard bopomofo encoder.
        XCTAssertFalse(out.isEmpty)
        // Spot-check that the first syllable's tone digit landed.
        XCTAssertTrue(out.contains("4"))
    }

    func testPhonemizeUserBeatsDictAtEqualLength() async throws {
        // Dict says 你好 → ni3 hao3 (→ sandhi → ni2 hao3).
        // Override forces ni4 hao4 (no sandhi promotion since neither
        // is tone-3).
        let lex = try MandarinCustomLexicon(entries: [
            "你好": ["ni4", "hao4"]
        ])
        var g2p = MandarinG2P(dict: Self.smallDict())
        g2p.customLexicon = lex
        let out = try await g2p.phonemize("你好")
        XCTAssertEqual(out, "ㄋㄧ4ㄏㄠ4")
    }

    func testPhonemizeSandhiRunsAcrossUserDictBoundary() async throws {
        // User word ends in tone-3, dict word starts with tone-3 → 3+3
        // sandhi must promote the user's tone-3 to tone-2 across the
        // boundary (validates that .syllables segments append to the
        // same pending buffer as .pinyin segments).
        let lex = try MandarinCustomLexicon(entries: [
            "我": ["wo3"]
        ])
        var g2p = MandarinG2P(dict: Self.smallDict())
        g2p.customLexicon = lex
        let out = try await g2p.phonemize("我你好")
        // "我" wo3 + "你" ni3 + "好" hao3 → 3+3+3 → right-to-left → 2+2+3
        XCTAssertEqual(out, "我2ㄋㄧ2ㄏㄠ3")
    }

    func testPhonemizeAtBopomofoBypassesSandhi() async throws {
        // @-bopomofo tokens are emitted verbatim and do not enter the
        // syllable buffer, so they don't participate in sandhi.
        let lex = try MandarinCustomLexicon(entries: [
            "我": ["@ㄨㄛ3"]
        ])
        var g2p = MandarinG2P(dict: Self.smallDict())
        g2p.customLexicon = lex
        let out = try await g2p.phonemize("我你好")
        // 我 → ㄨㄛ3 verbatim (no sandhi promotion). Then 你好 alone
        // → ni3 hao3 → sandhi → ni2 hao3.
        XCTAssertEqual(out, "ㄨㄛ3ㄋㄧ2ㄏㄠ3")
    }

    func testPhonemizeWithEmptyLexiconMatchesDictOnlyPath() async throws {
        let g2p = MandarinG2P(dict: Self.smallDict())
        let out = try await g2p.phonemize("你好")
        XCTAssertEqual(out, "ㄋㄧ2ㄏㄠ3")
    }
}
