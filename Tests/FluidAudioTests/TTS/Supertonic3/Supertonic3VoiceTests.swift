import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the `Supertonic3Voice` enum: the 10 built-in voice styles
/// (F1-F5, M1-M5), their repo-relative paths, and case-insensitive name parsing.
final class Supertonic3VoiceTests: XCTestCase {

    func testHasTenVoices() {
        XCTAssertEqual(Supertonic3Voice.allCases.count, 10)
        XCTAssertEqual(
            Supertonic3Voice.allCases.map(\.rawValue),
            ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"])
    }

    func testDefaultIsM1() {
        XCTAssertEqual(Supertonic3Voice.default, .m1)
        XCTAssertEqual(Supertonic3Voice.default.rawValue, "M1")
    }

    func testFileNameMapping() {
        XCTAssertEqual(Supertonic3Voice.f3.fileName, "voice_styles/F3.json")
        XCTAssertEqual(Supertonic3Voice.m1.fileName, "voice_styles/M1.json")
        for v in Supertonic3Voice.allCases {
            XCTAssertEqual(v.fileName, "voice_styles/\(v.rawValue).json")
        }
    }

    func testNameParsingIsCaseInsensitive() {
        XCTAssertEqual(Supertonic3Voice(name: "F3"), .f3)
        XCTAssertEqual(Supertonic3Voice(name: "f3"), .f3)
        XCTAssertEqual(Supertonic3Voice(name: "m1"), .m1)
        XCTAssertEqual(Supertonic3Voice(name: "M5"), .m5)
    }

    func testNameParsingRejectsUnknown() {
        XCTAssertNil(Supertonic3Voice(name: "xyz"))
        XCTAssertNil(Supertonic3Voice(name: ""))
        XCTAssertNil(Supertonic3Voice(name: "F6"))
        // A Kokoro-style voice name (the CLI's default) must not parse here, so
        // the Supertonic-3 path can safely fall back to the default voice.
        XCTAssertNil(Supertonic3Voice(name: "af_heart"))
    }

    func testRawValueRoundTrip() {
        for v in Supertonic3Voice.allCases {
            XCTAssertEqual(Supertonic3Voice(rawValue: v.rawValue), v)
            XCTAssertEqual(Supertonic3Voice(name: v.rawValue.lowercased()), v)
        }
    }
}
