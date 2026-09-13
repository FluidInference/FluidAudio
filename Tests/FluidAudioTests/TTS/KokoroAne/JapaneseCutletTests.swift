import XCTest

@testable import FluidAudio

/// Pure pieces of the Japanese frontend: Cutlet's normalization and kana rules
/// and the number reader, checked against Misaki's outputs.
final class JapaneseCutletTests: XCTestCase {

    func testNumberReaderMatchesMisaki() {
        // misaki.num2kana.Convert(n, 'hiragana')
        XCTAssertEqual(JapaneseNumberReader.hiragana("0"), "ゼロ")
        XCTAssertEqual(JapaneseNumberReader.hiragana("7"), "なな")
        XCTAssertEqual(JapaneseNumberReader.hiragana("10"), "じゅう")
        XCTAssertEqual(JapaneseNumberReader.hiragana("15"), "じゅうご")
        XCTAssertEqual(JapaneseNumberReader.hiragana("40"), "よんじゅう")
        XCTAssertEqual(JapaneseNumberReader.hiragana("105"), "ひゃくご")
        XCTAssertEqual(JapaneseNumberReader.hiragana("300"), "さんびゃく")
        XCTAssertEqual(JapaneseNumberReader.hiragana("2024"), "にせんにじゅうよん")
        XCTAssertEqual(JapaneseNumberReader.hiragana("8000"), "はっせん")
        XCTAssertEqual(JapaneseNumberReader.hiragana("10000"), "いちまん")
        XCTAssertEqual(JapaneseNumberReader.hiragana("12300"), "いちまんにせんさんびゃく")
        XCTAssertEqual(JapaneseNumberReader.hiragana("200000000"), "におく")
        XCTAssertEqual(JapaneseNumberReader.hiragana("007"), "なな")
    }

    func testNormalizationReadsDigitRunsAndFoldsWidth() {
        XCTAssertEqual(JapaneseCutlet.normalize("2024年"), " にせんにじゅうよん年")
        XCTAssertEqual(JapaneseCutlet.normalize("ＡＢＣ"), "ABC")
        XCTAssertEqual(JapaneseCutlet.normalize("ｶﾞｷﾞ"), "ガギ")
        XCTAssertEqual(JapaneseCutlet.normalize("ㇰ"), "ク")
        XCTAssertEqual(JapaneseCutlet.normalize("3〜5"), " さんから ご")
    }

    func testKatakanaToHiragana() {
        XCTAssertEqual(JapaneseCutlet.katakanaToHiragana("キョー"), "きょー")
        XCTAssertEqual(JapaneseCutlet.katakanaToHiragana("ヴ"), "ゔ")
        XCTAssertEqual(JapaneseCutlet.katakanaToHiragana("abc"), "abc")
    }

    func testKanaTableCoversCutletDigraphsAndSymbols() {
        XCTAssertEqual(JapaneseCutlet.kanaTable["きょ"], "kʲo")
        XCTAssertEqual(JapaneseCutlet.kanaTable["し"], "ɕi")
        XCTAssertEqual(JapaneseCutlet.kanaTable["つ"], "ʦɨ")
        XCTAssertEqual(JapaneseCutlet.kanaTable["わ"], "βa")
        XCTAssertEqual(JapaneseCutlet.kanaTable["ふぁ"], "ɸa")
        XCTAssertEqual(JapaneseCutlet.kanaTable["。"], ".")
        XCTAssertEqual(JapaneseCutlet.kanaTable["「"], "“")
        XCTAssertNil(JapaneseCutlet.kanaTable["っ"], "sokuon is a rule, not a table entry")
        XCTAssertNil(JapaneseCutlet.kanaTable["ん"], "moraic nasal is a rule, not a table entry")
    }
}
