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
        XCTAssertEqual(JapaneseCutlet.normalize("1〜2〜3〜4"), " いちから にから さんから よん", "every range marker")
        XCTAssertEqual(JapaneseCutlet.normalize("〜あ"), "〜あ", "a wave dash not followed by a digit stays")
        XCTAssertEqual(JapaneseCutlet.normalize("３～５"), " さんから ご", "full-width digits mark a range too")
        XCTAssertEqual(JapaneseCutlet.normalize("パン３〜５個"), "パン さんから ご個")
    }

    func testFoldingHalfWidthForms() {
        XCTAssertEqual(JapaneseCutlet.foldingHalfWidthForms("ｶﾞｷﾞ"), "ガギ")
        XCTAssertEqual(JapaneseCutlet.foldingHalfWidthForms("ﾊﾟﾝ3～5個"), "パン3〜5個", "tilde folded, digits untouched")
        XCTAssertEqual(JapaneseCutlet.foldingHalfWidthForms("aɾʲi １２"), "aɾʲi １２", "only half-width kana folds")
        XCTAssertEqual(JapaneseCutlet.foldingHalfWidthForms("な～と"), "な～と", "a drawl tilde is not a range")
        XCTAssertEqual(JapaneseCutlet.foldingHalfWidthForms("３～５"), "３〜５", "full-width digit after the tilde")
    }

    func testPrecomputedIPAPredicate() {
        XCTAssertTrue(KokoroAneManager.looksLikePrecomputedJapaneseIPA("βa ɕi ʔ ᵝ"))
        XCTAssertTrue(KokoroAneManager.looksLikePrecomputedJapaneseIPA("aɾʲiɡatoː"))
        XCTAssertTrue(KokoroAneManager.looksLikePrecomputedJapaneseIPA("kʲoː βa joi teŋkʲi desɨ."))
        XCTAssertTrue(KokoroAneManager.looksLikePrecomputedJapaneseIPA("ɡaʔkoː e iʔta, “kᵝa” — «x»"))
        XCTAssertTrue(KokoroAneManager.looksLikePrecomputedJapaneseIPA("AI"), "ASCII words pass through as Cutlet does")
        XCTAssertFalse(KokoroAneManager.looksLikePrecomputedJapaneseIPA("1"), "digits are text")
        XCTAssertFalse(KokoroAneManager.looksLikePrecomputedJapaneseIPA("ｶﾞｷﾞ"), "half-width kana is text")
        XCTAssertFalse(KokoroAneManager.looksLikePrecomputedJapaneseIPA("ありがとう"))
        XCTAssertFalse(KokoroAneManager.looksLikePrecomputedJapaneseIPA("今日"))
        XCTAssertFalse(KokoroAneManager.looksLikePrecomputedJapaneseIPA(""))
    }

    func testTruncatedDictionaryThrowsInsteadOfTrapping() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Lexicons with a plausible header but no body, a 3-byte char.bin,
        // a 2-byte matrix: every one must throw modelNotLoaded.
        var header = Data(count: 72)
        header.replaceSubrange(24..<28, with: withUnsafeBytes(of: UInt32(8).littleEndian) { Data($0) })
        for name in [KokoroAneConstants.japaneseSystemDictionaryFile, KokoroAneConstants.japaneseUnknownDictionaryFile]
        {
            try header.write(to: directory.appendingPathComponent(name))
        }
        try Data([0, 0, 0]).write(to: directory.appendingPathComponent(KokoroAneConstants.japaneseCharCategoryFile))
        try Data([1, 0]).write(to: directory.appendingPathComponent(KokoroAneConstants.japaneseConnectionMatrixFile))
        XCTAssertThrowsError(try JapaneseMecabDictionary(directory: directory)) { error in
            guard case KokoroAneError.modelNotLoaded = error else {
                return XCTFail("expected modelNotLoaded, got \(error)")
            }
        }
        // The per-file check the downloader runs on cached assets rejects
        // each of them, and a missing file, but accepts a nonempty word list.
        for name in KokoroAneConstants.japaneseG2PFiles {
            XCTAssertThrowsError(
                try JapaneseMecabDictionary.validateAsset(named: name, at: directory.appendingPathComponent(name)),
                name)
        }
        try Data("あ\n".utf8).write(to: directory.appendingPathComponent(KokoroAneConstants.japaneseWordListFile))
        XCTAssertNoThrow(
            try JapaneseMecabDictionary.validateAsset(
                named: KokoroAneConstants.japaneseWordListFile,
                at: directory.appendingPathComponent(KokoroAneConstants.japaneseWordListFile)))
        // Structurally empty assets are unusable and must be rejected too: an
        // all-zero 72-byte lexicon, a four-byte matrix with 0×0 contexts, a
        // char.bin announcing zero categories.
        try Data(count: 72).write(to: directory.appendingPathComponent(KokoroAneConstants.japaneseSystemDictionaryFile))
        try Data(count: 4).write(to: directory.appendingPathComponent(KokoroAneConstants.japaneseConnectionMatrixFile))
        try Data(count: 4 + 0xFFFF * 4).write(
            to: directory.appendingPathComponent(KokoroAneConstants.japaneseCharCategoryFile))
        for name in [
            KokoroAneConstants.japaneseSystemDictionaryFile, KokoroAneConstants.japaneseConnectionMatrixFile,
            KokoroAneConstants.japaneseCharCategoryFile,
        ] {
            XCTAssertThrowsError(
                try JapaneseMecabDictionary.validateAsset(named: name, at: directory.appendingPathComponent(name)),
                name)
        }
        XCTAssertThrowsError(try JapaneseMecabDictionary(directory: directory))
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
