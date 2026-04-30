import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for the StyleTTS2 178-token vocab loader.
///
/// Uses an inline JSON fixture so the test runs without HuggingFace
/// downloads or any CoreML models.
final class StyleTTS2VocabTests: XCTestCase {

    // A minimal but representative slice of the upstream
    // `text_cleaner_vocab.json` (pad, ASCII letter, IPA char, stress mark,
    // syllabic combining mark, space).
    private let fixtureJSON: String = #"""
        {
          "pad_token": "$",
          "num_tokens": 178,
          "symbols": ["$", " ", "a", "ɑ", "ˈ", "̩"],
          "token_to_id": {
            "$": 0,
            " ": 16,
            "a": 43,
            "ɑ": 69,
            "ˈ": 156,
            "̩": 175
          }
        }
        """#

    private func writeFixture(_ json: String, name: String = "vocab.json") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-vocab-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try json.data(using: .utf8)!.write(to: url)
        return url
    }

    // MARK: - load

    func testLoadParsesAllSingleGraphemeKeys() throws {
        let url = try writeFixture(fixtureJSON)
        let vocab = try StyleTTS2Vocab.load(from: url)

        XCTAssertEqual(vocab.padTokenId, 0)
        XCTAssertEqual(vocab.map["$"], 0)
        XCTAssertEqual(vocab.map[" "], 16)
        XCTAssertEqual(vocab.map["a"], 43)
        XCTAssertEqual(vocab.map["ɑ"], 69)
        XCTAssertEqual(vocab.map["ˈ"], 156)
    }

    func testLoadHandlesCombiningGraphemeAsSingleCharacter() throws {
        // The syllabic combining mark `̩` (U+0329) is a single Swift
        // `Character` because grapheme clusters group combining marks
        // with their base — but it appears alone in the upstream vocab.
        // Confirm we can round-trip it through the loader.
        let url = try writeFixture(fixtureJSON)
        let vocab = try StyleTTS2Vocab.load(from: url)
        let key: Character = "\u{0329}"  // Combining vertical line below
        XCTAssertEqual(vocab.map[key], 175)
    }

    func testLoadFailsForMissingFile() {
        let bogusURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).json")
        XCTAssertThrowsError(try StyleTTS2Vocab.load(from: bogusURL)) { error in
            guard case StyleTTS2Error.modelNotFound = error else {
                XCTFail("expected modelNotFound, got \(error)")
                return
            }
        }
    }

    func testLoadFailsForMalformedJSON() throws {
        let url = try writeFixture("not actually json", name: "bad.json")
        XCTAssertThrowsError(try StyleTTS2Vocab.load(from: url))
    }

    func testLoadFailsWhenTokenToIdMissing() throws {
        let bad = #"{"pad_token": "$", "symbols": []}"#
        let url = try writeFixture(bad, name: "missing-tokens.json")
        XCTAssertThrowsError(try StyleTTS2Vocab.load(from: url)) { error in
            guard case StyleTTS2Error.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testLoadFallsBackToPadIdZeroWhenPadTokenMissing() throws {
        // No `pad_token` field → loader defaults to id 0 (upstream contract).
        let json = #"""
            {
              "num_tokens": 1,
              "symbols": ["$"],
              "token_to_id": {"$": 0}
            }
            """#
        let url = try writeFixture(json, name: "no-pad.json")
        let vocab = try StyleTTS2Vocab.load(from: url)
        XCTAssertEqual(vocab.padTokenId, 0)
    }

    // MARK: - encode

    func testEncodeMapsKnownGraphemes() throws {
        let url = try writeFixture(fixtureJSON)
        let vocab = try StyleTTS2Vocab.load(from: url)
        let ids = vocab.encode("aɑˈa")
        XCTAssertEqual(ids, [43, 69, 156, 43])
    }

    func testEncodeDropsUnknownGraphemes() throws {
        // Upstream `text_utils.TextCleaner.__call__` silently skips
        // unmapped chars. We mirror that to avoid blowing up on stray
        // emoji or rare IPA the user pastes in.
        let url = try writeFixture(fixtureJSON)
        let vocab = try StyleTTS2Vocab.load(from: url)
        let ids = vocab.encode("a🙂ɑ")
        XCTAssertEqual(ids, [43, 69])
    }

    func testEncodeEmptyStringReturnsEmptyArray() throws {
        let url = try writeFixture(fixtureJSON)
        let vocab = try StyleTTS2Vocab.load(from: url)
        XCTAssertEqual(vocab.encode(""), [])
    }
}
