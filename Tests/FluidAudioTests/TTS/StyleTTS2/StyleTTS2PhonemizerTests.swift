import XCTest

@testable import FluidAudio

/// Pins the misaki → espeak post-pass + the resolution-order contract
/// for `StyleTTS2Phonemizer`. The lexicon-loading and BART-loading paths
/// hit network/disk and are exercised by the smoke tests in CI; here we
/// only test logic that doesn't require model assets.
final class StyleTTS2PhonemizerTests: XCTestCase {

    // MARK: - Post-pass remap

    func testEspeakPostPassRewritesAffricates() {
        XCTAssertEqual(
            StyleTTS2Phonemizer.applyEspeakPostPass("ʧˈɔɪs"),
            "tʃˈɔɪs",
            "Misaki tʃ-ligature must decompose for the espeak-trained checkpoint")
        XCTAssertEqual(
            StyleTTS2Phonemizer.applyEspeakPostPass("ʤˈʌmps"),
            "dʒˈʌmps",
            "Misaki dʒ-ligature must decompose for the espeak-trained checkpoint")
    }

    func testEspeakPostPassCollapsesRhotics() {
        XCTAssertEqual(
            StyleTTS2Phonemizer.applyEspeakPostPass("ɡˈɜɹl"),
            "ɡˈɝl",
            "ɜɹ must collapse to ɝ (espeak rhotic)")
        XCTAssertEqual(
            StyleTTS2Phonemizer.applyEspeakPostPass("ˈoʊvəɹ"),
            "ˈoʊvɚ",
            "əɹ must collapse to ɚ (espeak rhotic)")
    }

    func testEspeakPostPassPassesThroughUnaffected() {
        XCTAssertEqual(
            StyleTTS2Phonemizer.applyEspeakPostPass("hˈɛloʊ wˈɝld"),
            "hˈɛloʊ wˈɝld",
            "Strings without ligatures or rhotic digraphs must pass through unchanged")
    }
}
