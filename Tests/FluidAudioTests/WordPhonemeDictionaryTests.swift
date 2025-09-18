import XCTest

@testable import FluidAudio

@available(macOS 13.0, *)
final class WordPhonemeDictionaryTests: XCTestCase {

    func testLanguageSpecificOverlayOverridesBasePronunciation() throws {
        let cacheDir = try TtsModels.cacheDirectoryURL().appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let overlayURL = cacheDir.appendingPathComponent("word_phonemes_es-ES.json")

        // Ensure a clean state
        try? FileManager.default.removeItem(at: overlayURL)
        KokoroModel.resetPhonemeDictionariesForTesting()

        try KokoroModel.loadSimplePhonemeDictionary()
        let baseDictionary = try KokoroModel.phonemeDictionary(for: nil)
        let basePronunciation = baseDictionary["color"]

        // Write overlay with a distinct pronunciation
        let overlay: [String: Any] = ["color": "kˈolɔɾ"]
        let data = try JSONSerialization.data(withJSONObject: overlay, options: [.prettyPrinted])
        try data.write(to: overlayURL)
        defer { try? FileManager.default.removeItem(at: overlayURL) }

        KokoroModel.resetPhonemeDictionariesForTesting()
        try KokoroModel.loadSimplePhonemeDictionary()
        let spanishDictionary = try KokoroModel.phonemeDictionary(for: "es-ES")
        let spanishPronunciation = spanishDictionary["color"]

        XCTAssertNotNil(basePronunciation)
        XCTAssertNotNil(spanishPronunciation)
        XCTAssertNotEqual(basePronunciation, spanishPronunciation)
    }
}
