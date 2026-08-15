import XCTest

@testable import FluidAudio

/// Pins `LuxTtsTokenizer` against the EmiliaTokenizer fixtures: the parsed
/// `tokens.txt` table and exact token-id sequences for the prompt + both
/// test texts. Exact-parity gate — no tolerances.
final class LuxTtsTokenizerTests: XCTestCase {

    private func makeTokenizer() throws -> LuxTtsTokenizer {
        try LuxTtsTokenizer(tokensFileURL: LuxTtsFixtures.resourceURL("tokens.txt"))
    }

    func testTokenTableParsing() throws {
        let tokenizer = try makeTokenizer()
        XCTAssertEqual(tokenizer.vocabSize, 360)
        XCTAssertEqual(tokenizer.padId, 0)
        XCTAssertEqual(tokenizer.tokenToId["_"], 0)
        XCTAssertEqual(tokenizer.tokenToId[" "], 3)
        XCTAssertEqual(tokenizer.tokenToId["ˈ"], 120)
        // Multi-character pinyin entries survive the first-tab-only split.
        XCTAssertNotNil(tokenizer.tokenToId["zh0"])
    }

    func testPromptTokenIdsMatchFixture() throws {
        let fixtures = try LuxTtsFixtures.load()
        let tokenizer = try makeTokenizer()
        XCTAssertEqual(
            tokenizer.tokenIds(phonemes: fixtures.prompt.phonemeString),
            fixtures.prompt.tokenIds)
    }

    func testTextTokenIdsMatchFixture() throws {
        let fixtures = try LuxTtsFixtures.load()
        let tokenizer = try makeTokenizer()
        for text in fixtures.texts {
            XCTAssertEqual(
                tokenizer.tokenIds(phonemes: text.phonemeString),
                text.tokenIds,
                "token ids diverged for: \(text.text)")
        }
    }

    func testOovScalarsAreSkipped() throws {
        let tokenizer = try makeTokenizer()
        // '€' is not in tokens.txt; the surrounding tokens must be unaffected
        // (EmiliaTokenizer skips OOV entries).
        XCTAssertEqual(
            tokenizer.tokenIds(phonemes: "a€b"),
            tokenizer.tokenIds(phonemes: "ab"))
    }
}
