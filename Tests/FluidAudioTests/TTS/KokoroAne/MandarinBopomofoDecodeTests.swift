import Foundation
import XCTest

@testable import FluidAudio

/// Network-free tests for `MandarinBopomofoMap.decode(bopomofo:)` — the
/// g2pW label → pinyin syllable step that lets a disambiguated reading
/// re-enter the dict pipeline (sandhi, erhua, vocab encoding).
final class MandarinBopomofoDecodeTests: XCTestCase {

    private func decode(_ label: String) -> (String, Int)? {
        guard let syl = MandarinBopomofoMap.decode(bopomofo: label) else { return nil }
        return (syl.base, syl.tone)
    }

    func testInitialPlusFinal() {
        XCTAssertEqual(decode("ㄒㄧㄥ2")?.0, "xing")
        XCTAssertEqual(decode("ㄒㄧㄥ2")?.1, 2)
        XCTAssertEqual(decode("ㄏㄤ2")?.0, "hang")
        XCTAssertEqual(decode("ㄉㄜ5")?.0, "de")
        XCTAssertEqual(decode("ㄉㄜ5")?.1, 5)
        XCTAssertEqual(decode("ㄔㄤ2")?.0, "chang")
    }

    func testPypinyinSpellingContractions() {
        XCTAssertEqual(decode("ㄌㄧㄡ2")?.0, "liu")
        XCTAssertEqual(decode("ㄉㄨㄟ4")?.0, "dui")
        XCTAssertEqual(decode("ㄉㄨㄣ1")?.0, "dun")
        XCTAssertEqual(decode("ㄍㄨㄥ1")?.0, "gong")
        XCTAssertEqual(decode("ㄒㄩㄥ2")?.0, "xiong")
    }

    func testUmlautSpelling() {
        // j/q/x drop the umlaut; n/l keep `v`.
        XCTAssertEqual(decode("ㄐㄩ4")?.0, "ju")
        XCTAssertEqual(decode("ㄑㄩㄝ4")?.0, "que")
        XCTAssertEqual(decode("ㄒㄩㄢ3")?.0, "xuan")
        XCTAssertEqual(decode("ㄐㄩㄣ1")?.0, "jun")
        XCTAssertEqual(decode("ㄌㄩ4")?.0, "lv")
        XCTAssertEqual(decode("ㄋㄩㄝ4")?.0, "nve")
    }

    func testBareSibilants() {
        XCTAssertEqual(decode("ㄓ1")?.0, "zhi")
        XCTAssertEqual(decode("ㄕ4")?.0, "shi")
        XCTAssertEqual(decode("ㄗ4")?.0, "zi")
        XCTAssertEqual(decode("ㄖ4")?.0, "ri")
        XCTAssertNil(decode("ㄅ1"))
    }

    func testStandaloneFinals() {
        XCTAssertEqual(decode("ㄧ4")?.0, "yi")
        XCTAssertEqual(decode("ㄧㄥ1")?.0, "ying")
        XCTAssertEqual(decode("ㄨ3")?.0, "wu")
        XCTAssertEqual(decode("ㄨㄥ1")?.0, "weng")
        XCTAssertEqual(decode("ㄩㄝ4")?.0, "yue")
        XCTAssertEqual(decode("ㄩㄢ2")?.0, "yuan")
        XCTAssertEqual(decode("ㄦ2")?.0, "er")
        XCTAssertEqual(decode("ㄞ4")?.0, "ai")
    }

    func testDiacriticLabelsAlsoDecode() {
        XCTAssertEqual(decode("ㄏㄤˊ")?.0, "hang")
        XCTAssertEqual(decode("ㄏㄤˊ")?.1, 2)
        XCTAssertEqual(decode("ㄉㄡ")?.1, 1)
    }

    func testRejectsMalformed() {
        XCTAssertNil(decode(""))
        // No tone mark at all is implicit tone 1, like the diacritic form.
        XCTAssertEqual(decode("ㄏㄤ")?.1, 1)
        XCTAssertNil(decode("2"))
        XCTAssertNil(decode("ㄏㄤ9"))
        XCTAssertNil(decode("ㄝ1"))
    }

    /// Every decoded syllable must re-encode into the v1.1-zh vocab the
    /// same way the dict path would, i.e. `encode` accepts it.
    func testDecodedSyllablesReEncode() throws {
        let cases: [(String, String)] = [
            ("ㄒㄧㄥ2", "ㄒ应2"),
            ("ㄏㄤ2", "ㄏㄤ2"),
            ("ㄓ1", "ㄓ十1"),
            ("ㄗ4", "ㄗㄭ4"),
            ("ㄐㄩ4", "ㄐㄩ4"),
            ("ㄌㄧㄡ2", "ㄌ又2"),
            ("ㄍㄨㄥ1", "ㄍ中1"),
            ("ㄨㄥ1", "瓮1"),
            ("ㄧ4", "ㄧ4"),
            ("ㄩㄢ2", "元2"),
        ]
        for (label, expected) in cases {
            let syl = try XCTUnwrap(MandarinBopomofoMap.decode(bopomofo: label), label)
            XCTAssertEqual(
                MandarinBopomofoMap.encode(syllable: syl.base, tone: syl.tone), expected, label)
        }
    }
}
