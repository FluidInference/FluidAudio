import XCTest

@testable import FluidAudio

final class JapaneseMoraMapperTests: XCTestCase {
    func testKokoroCompatibleMorasAndLongVowels() throws {
        XCTAssertEqual(
            try JapaneseMoraMapper.phonemize(["キョ", "オ", "ワ"]),
            "kʲoːβa")
        XCTAssertEqual(
            try JapaneseMoraMapper.phonemize(["ア", "リ", "ガ", "ト", "オ"]),
            "aɾʲiɡatoː")
        XCTAssertEqual(
            try JapaneseMoraMapper.phonemize(["ベ", "ン", "キョ", "オ"]),
            "beŋkʲoː")
    }

    func testContextualNasalAllophones() throws {
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン", "バ"]), "mba")
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン", "カ"]), "ŋka")
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン", "チ"]), "ɲʨi")
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン", "タ"]), "nta")
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン", "ア"]), "ɴa")
        XCTAssertEqual(try JapaneseMoraMapper.phonemize(["ン"]), "ɴ")
    }

    func testGeminateAndForeignMoras() throws {
        XCTAssertEqual(
            try JapaneseMoraMapper.phonemize(["ガ", "ッ", "コ", "オ"]),
            "ɡaʔkoː")
        XCTAssertEqual(
            try JapaneseMoraMapper.phonemize(["ファ", "イ", "ル"]),
            "ɸaiɾɯ")
    }

    func testUnknownMoraFailsLoudly() {
        XCTAssertThrowsError(try JapaneseMoraMapper.phonemize(["not-a-mora"])) { error in
            guard case KokoroAneError.inputProcessingFailed = error else {
                return XCTFail("expected inputProcessingFailed, got \(error)")
            }
        }
    }

    func testTarPathsCannotEscapeDestination() throws {
        XCTAssertEqual(
            try GzipTarExtractor.validatedRelativePath(
                "open_jtalk_dic_utf_8-1.11/sys.dic"),
            "open_jtalk_dic_utf_8-1.11/sys.dic")
        XCTAssertThrowsError(try GzipTarExtractor.validatedRelativePath("../outside"))
        XCTAssertThrowsError(try GzipTarExtractor.validatedRelativePath("/absolute/path"))
        XCTAssertThrowsError(try GzipTarExtractor.validatedRelativePath("safe/../../outside"))
        XCTAssertThrowsError(try GzipTarExtractor.validatedRelativePath("safe\\outside"))
    }
}
