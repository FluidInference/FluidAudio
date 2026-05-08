import XCTest

@testable import FluidAudio

/// Tests for the `PhoneticEncoder` protocol's default implementations.
///
/// FluidAudio core does not ship a phonetic encoder; these tests use a
/// minimal in-test stub to verify the protocol's `soundsAlike` default
/// implementation. Real-world usage wires a paid Metaphone3 (or other)
/// encoder via the optional `Frameworks/` xcframework, gated by
/// Package.swift.
final class PhoneticEncoderTests: XCTestCase {

    /// Stub encoder that maps every word to its lowercased value, with
    /// an optional alternate code provided externally.
    private struct StubEncoder: PhoneticEncoder {
        let primary: [String: String]
        let alternate: [String: String]

        func encode(_ word: String) -> String {
            primary[word.lowercased()] ?? ""
        }

        func encodeAlternate(_ word: String) -> String? {
            alternate[word.lowercased()]
        }
    }

    func testIdenticalPrimaryCodesMatch() {
        let enc = StubEncoder(
            primary: ["chronicity": "KRANASAT", "crenessity": "KRANASAT"],
            alternate: [:]
        )
        XCTAssertTrue(enc.soundsAlike("chronicity", "crenessity"))
    }

    func testDifferentPrimaryCodesDoNotMatch() {
        let enc = StubEncoder(
            primary: ["chronicity": "KRANASAT", "trulicity": "TRALASAT"],
            alternate: [:]
        )
        XCTAssertFalse(enc.soundsAlike("chronicity", "trulicity"))
    }

    func testAlternateCodeMatchesPrimary() {
        let enc = StubEncoder(
            primary: ["read": "RD", "reed": "RD"],
            alternate: ["read": "READ"]
        )
        // Both have primary RD, should match on primary.
        XCTAssertTrue(enc.soundsAlike("read", "reed"))
    }

    func testAlternateOnOneSideMatchesPrimaryOnOther() {
        let enc = StubEncoder(
            primary: ["red": "RED", "read": "RD"],
            alternate: ["read": "RED"]  // alternate of "read" matches primary of "red"
        )
        XCTAssertTrue(enc.soundsAlike("red", "read"))
    }

    func testEmptyCodeNeverMatches() {
        let enc = StubEncoder(
            primary: ["foo": "", "bar": ""],
            alternate: [:]
        )
        // Two unknown words must NOT be claimed to sound alike just
        // because both encode to empty.
        XCTAssertFalse(enc.soundsAlike("foo", "bar"))
    }

    func testEmptyVsNonEmptyDoesNotMatch() {
        let enc = StubEncoder(
            primary: ["foo": "", "bar": "BAR"],
            alternate: [:]
        )
        XCTAssertFalse(enc.soundsAlike("foo", "bar"))
        XCTAssertFalse(enc.soundsAlike("bar", "foo"))
    }
}
