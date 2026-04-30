import XCTest

@testable import FluidAudio

final class MagpieIpaOverrideTests: XCTestCase {

    func testPlainText() {
        let segments = MagpieIpaOverride.segment("Hello world")
        XCTAssertEqual(segments, [.text("Hello world")])
    }

    func testEmptyInput() {
        XCTAssertEqual(MagpieIpaOverride.segment(""), [])
    }

    func testSingleIpaRegion() {
        let segments = MagpieIpaOverride.segment("Hello | ˈ n ɛ m o ʊ | end")
        XCTAssertEqual(
            segments,
            [
                .text("Hello "),
                .ipa(tokens: ["ˈ", "n", "ɛ", "m", "o", "ʊ"]),
                .text(" end"),
            ])
    }

    func testMultipleIpaRegions() {
        let segments = MagpieIpaOverride.segment("A |x y| B |z|")
        XCTAssertEqual(
            segments,
            [
                .text("A "),
                .ipa(tokens: ["x", "y"]),
                .text(" B "),
                .ipa(tokens: ["z"]),
            ])
    }

    func testEmptyIpaRegionCollapses() {
        let segments = MagpieIpaOverride.segment("A || B")
        XCTAssertEqual(segments, [.text("A "), .text(" B")])
    }

    func testUnpairedTrailingPipeBecomesText() {
        let segments = MagpieIpaOverride.segment("A |stuck")
        XCTAssertEqual(segments, [.text("A "), .text("|stuck")])
    }

    func testConsecutiveWhitespaceCollapses() {
        let segments = MagpieIpaOverride.segment("|a   b|")
        XCTAssertEqual(segments, [.ipa(tokens: ["a", "b"])])
    }
}
