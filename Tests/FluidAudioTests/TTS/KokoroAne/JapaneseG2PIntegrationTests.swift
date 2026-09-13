import Foundation
import XCTest

@testable import FluidAudio

/// Real OpenJTalk integration coverage. Set `FLUIDAUDIO_OPEN_JTALK_DICTIONARY`
/// to an extracted v1.11 dictionary; no mock analyzer or dictionary is used.
final class JapaneseG2PIntegrationTests: XCTestCase {
    func testContextualKanjiReadings() async throws {
        guard
            let path = ProcessInfo.processInfo.environment["FLUIDAUDIO_OPEN_JTALK_DICTIONARY"],
            !path.isEmpty
        else {
            throw XCTSkip("Set FLUIDAUDIO_OPEN_JTALK_DICTIONARY to run real OpenJTalk tests.")
        }

        let g2p = try JapaneseG2P(dictionaryURL: URL(fileURLWithPath: path))
        XCTAssertEqual(
            try await g2p.phonemize("今日は良い天気です。"),
            "kʲoːβa joi teŋkʲidesɨ.")
        XCTAssertEqual(
            try await g2p.phonemize("今日中に返します。"),
            "kʲoːʥɨːɲi kaeɕimasɨ.")
        XCTAssertEqual(
            try await g2p.phonemize("四月一日は休みです。"),
            "ɕiɡaʦɨ ʦɨitaʨiβa jasɨmʲidesɨ.")
        XCTAssertEqual(
            try await g2p.phonemize("一日中勉強しました。"),
            "iʨiɲiʨiʥɨː beŋkʲoː ɕimaɕita.")
        // The object particle を after an o-ending word stays a separate vowel
        // (Misaki: `ɲihoŋɡo o`), not a long vowel.
        XCTAssertEqual(try await g2p.phonemize("日本語を勉強しています。"), "ɲihoŋɡo o beŋkʲoː ɕite imasɨ.")
        XCTAssertEqual(try await g2p.phonemize("本を読みます。"), "hoɴ o jomʲimasɨ.")
    }
}
