import XCTest

@testable import FluidAudio

final class MultilingualG2PTests: XCTestCase {

    // MARK: - Byte Tokenization

    func testByteTokenizationRoundtrip() {
        // ByT5 maps byte value b to token ID b + 3
        let text = "hello"
        let bytes = Array(text.utf8)
        let tokenIds = bytes.map { Int32($0) + 3 }

        // Decode back
        let decoded = tokenIds.compactMap { id -> UInt8? in
            let b = id - 3
            guard b >= 0, b <= 255 else { return nil }
            return UInt8(b)
        }
        let result = String(bytes: decoded, encoding: .utf8)
        XCTAssertEqual(result, text)
    }

    func testByteTokenizationWithUnicode() {
        // Multi-byte UTF-8 character
        let text = "<eng-us>: cafe\u{0301}"  // cafe + combining accent
        let bytes = Array(text.utf8)
        let tokenIds = bytes.map { Int32($0) + 3 }

        let decoded = tokenIds.compactMap { id -> UInt8? in
            let b = id - 3
            guard b >= 0, b <= 255 else { return nil }
            return UInt8(b)
        }
        let result = String(bytes: decoded, encoding: .utf8)
        XCTAssertEqual(result, text)
    }

    func testByteTokenizationWithJapanese() {
        let text = "<jpn>: \u{6771}\u{4EAC}"  // Tokyo in kanji
        let bytes = Array(text.utf8)
        let tokenIds = bytes.map { Int32($0) + 3 }

        let decoded = tokenIds.compactMap { id -> UInt8? in
            let b = id - 3
            guard b >= 0, b <= 255 else { return nil }
            return UInt8(b)
        }
        let result = String(bytes: decoded, encoding: .utf8)
        XCTAssertEqual(result, text)
    }

    // MARK: - Language Mapping

    func testKokoroVoiceToLanguage() {
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("af_heart"), .americanEnglish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("am_adam"), .americanEnglish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("bf_alice"), .britishEnglish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("bm_daniel"), .britishEnglish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("ef_dora"), .spanish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("em_alex"), .spanish)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("ff_siwis"), .french)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("hf_alpha"), .hindi)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("hm_omega"), .hindi)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("if_sara"), .italian)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("im_nicola"), .italian)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("jf_alpha"), .japanese)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("jm_kumo"), .japanese)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("pf_dora"), .brazilianPortuguese)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("pm_alex"), .brazilianPortuguese)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("zf_xiaobei"), .mandarinChinese)
        XCTAssertEqual(MultilingualG2PLanguage.fromKokoroVoice("zm_yunxi"), .mandarinChinese)
    }

    func testUnknownVoiceReturnsNil() {
        XCTAssertNil(MultilingualG2PLanguage.fromKokoroVoice("xx_unknown"))
        XCTAssertNil(MultilingualG2PLanguage.fromKokoroVoice(""))
        XCTAssertNil(MultilingualG2PLanguage.fromKokoroVoice("a"))
    }

    // MARK: - Language Properties

    func testCharsiuCodes() {
        XCTAssertEqual(MultilingualG2PLanguage.americanEnglish.charsiuCode, "eng-us")
        XCTAssertEqual(MultilingualG2PLanguage.britishEnglish.charsiuCode, "eng-uk")
        XCTAssertEqual(MultilingualG2PLanguage.spanish.charsiuCode, "spa")
        XCTAssertEqual(MultilingualG2PLanguage.french.charsiuCode, "fra")
        XCTAssertEqual(MultilingualG2PLanguage.hindi.charsiuCode, "hin")
        XCTAssertEqual(MultilingualG2PLanguage.italian.charsiuCode, "ita")
        XCTAssertEqual(MultilingualG2PLanguage.japanese.charsiuCode, "jpn")
        XCTAssertEqual(MultilingualG2PLanguage.brazilianPortuguese.charsiuCode, "por-bz")
        XCTAssertEqual(MultilingualG2PLanguage.mandarinChinese.charsiuCode, "cmn")
    }

    func testPrefixFormat() {
        XCTAssertEqual(MultilingualG2PLanguage.americanEnglish.prefix, "<eng-us>: ")
        XCTAssertEqual(MultilingualG2PLanguage.japanese.prefix, "<jpn>: ")
        XCTAssertEqual(MultilingualG2PLanguage.mandarinChinese.prefix, "<cmn>: ")
    }

    // MARK: - Model Names

    func testModelNamesMultilingualG2P() {
        XCTAssertEqual(ModelNames.MultilingualG2P.encoderFile, "MultilingualG2PEncoder.mlmodelc")
        XCTAssertEqual(ModelNames.MultilingualG2P.decoderFile, "MultilingualG2PDecoder.mlmodelc")
        XCTAssertEqual(
            ModelNames.MultilingualG2P.requiredModels,
            ["MultilingualG2PEncoder.mlmodelc", "MultilingualG2PDecoder.mlmodelc"])
    }

    func testRepoMultilingualG2P() {
        // Multilingual G2P models are bundled inside the kokoro repo
        XCTAssertEqual(Repo.kokoro.folderName, "kokoro")
        XCTAssertEqual(Repo.kokoro.remotePath, "FluidInference/kokoro-82m-coreml")
    }
}
