import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the shared letter-name initialism helpers (issue #710):
/// candidate boundaries and the lexicon-driven spell-out mechanics that
/// English TTS frontends share.
final class EnglishInitialismsTests: XCTestCase {

    // MARK: - Candidate boundaries

    func testCandidateLengthRange() {
        // 2-5 strict ASCII uppercase letters.
        XCTAssertTrue(EnglishInitialisms.isCandidate("FBI"))
        XCTAssertTrue(EnglishInitialisms.isCandidate("US"))
        XCTAssertTrue(EnglishInitialisms.isCandidate("ABCDE"))
        XCTAssertFalse(EnglishInitialisms.isCandidate("A"))
        XCTAssertFalse(EnglishInitialisms.isCandidate("ABCDEF"))
    }

    func testCandidateRejectsNonStrictAllCaps() {
        XCTAssertFalse(EnglishInitialisms.isCandidate("MP3"))  // digit
        XCTAssertFalse(EnglishInitialisms.isCandidate("iOS"))  // mixed case
        XCTAssertFalse(EnglishInitialisms.isCandidate("A-B"))  // hyphen
        XCTAssertFalse(EnglishInitialisms.isCandidate("ÉÀ"))  // non-ASCII
    }

    // MARK: - Spelling

    private let letters: [String: [String]] = [
        "F": ["ˈ", "ɛ", "f"],
        "B": ["b", "ˈ", "i"],
        "I": ["ˈ", "I"],
    ]

    func testSpellJoinsLetterNamesWithSpaces() {
        let spelled = EnglishInitialisms.spell("FBI") { letters[$0] }
        XCTAssertEqual(spelled, "ˈɛf bˈi ˈI")
    }

    func testSpellReturnsNilWhenAnyLetterMissing() {
        // `X` isn't in the map → nil so the caller can fall through.
        XCTAssertNil(EnglishInitialisms.spell("FXB") { letters[$0] })
    }

    func testSpellHonorsCustomRenderAndSeparator() {
        let spelled = EnglishInitialisms.spell(
            "FB",
            letterTokens: { letters[$0] },
            render: { "[" + $0.joined() + "]" },
            separator: "-"
        )
        XCTAssertEqual(spelled, "[ˈɛf]-[bˈi]")
    }

    func testLetterNameOverridesAreUppercaseOnly() {
        XCTAssertTrue(EnglishInitialisms.letterNameOverrides.contains("AI"))
        XCTAssertTrue(EnglishInitialisms.letterNameOverrides.contains("US"))
        XCTAssertFalse(EnglishInitialisms.letterNameOverrides.contains("ai"))
        XCTAssertFalse(EnglishInitialisms.letterNameOverrides.contains("us"))
    }
}
