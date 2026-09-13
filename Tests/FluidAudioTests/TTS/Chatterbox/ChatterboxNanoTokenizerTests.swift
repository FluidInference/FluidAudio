import Foundation
import XCTest

@testable import FluidAudio

/// GPT-2 BPE parity against upstream `AutoTokenizer` for the
/// ResembleAI/chatterbox-nano tokenizer assets. Reference ids generated with
/// `transformers` 5.2.0 from the shipped `vocab.json` / `merges.txt` /
/// `added_tokens.json`.
final class ChatterboxNanoTokenizerTests: XCTestCase {

    /// Minimal fixture assets covering the reference sentences: the real
    /// tokenizer files are ~1.5 MB, so tests download nothing — instead the
    /// cache-dir copy is used when present, and tests are skipped otherwise.
    private func loadTokenizer() throws -> ChatterboxNanoTokenizer {
        let repoDir = try TtsCacheDirectory.ensure()
            .appendingPathComponent("Models")
            .appendingPathComponent(Repo.chatterboxNano.folderName)
        let vocabURL = repoDir.appendingPathComponent(ModelNames.ChatterboxNano.vocabFile)
        let mergesURL = repoDir.appendingPathComponent(ModelNames.ChatterboxNano.mergesFile)
        let addedURL = repoDir.appendingPathComponent(
            ModelNames.ChatterboxNano.addedTokensFile)
        guard FileManager.default.fileExists(atPath: vocabURL.path),
            FileManager.default.fileExists(atPath: mergesURL.path),
            FileManager.default.fileExists(atPath: addedURL.path)
        else {
            throw XCTSkip("chatterbox-nano tokenizer assets not cached locally")
        }
        return try ChatterboxNanoTokenizer(
            vocabURL: vocabURL, mergesURL: mergesURL, addedTokensURL: addedURL)
    }

    func testPlainSentenceMatchesUpstream() throws {
        let tokenizer = try loadTokenizer()
        XCTAssertEqual(
            tokenizer.encode("The quick brown fox jumps over the lazy dog near the river bank."),
            [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 1474, 262, 7850, 3331, 13])
    }

    func testParalinguisticTagSentenceMatchesUpstream() throws {
        let tokenizer = try loadTokenizer()
        // The space before "[chuckle]" becomes a standalone Ġ token (220);
        // the tag itself is the added-token id 50274.
        XCTAssertEqual(
            tokenizer.encode(
                "Hi there, Sarah here from MochaFone calling you back [chuckle], "
                    + "have you got one minute to chat about the billing issue?"),
            [
                17250, 612, 11, 10490, 994, 422, 4270, 11693, 37, 505, 4585, 345, 736,
                220, 50274, 11, 423, 345, 1392, 530, 5664, 284, 8537, 546, 262, 26297,
                2071, 30,
            ])
    }

    func testLeadingAndTrailingTagsMatchUpstream() throws {
        let tokenizer = try loadTokenizer()
        XCTAssertEqual(
            tokenizer.encode("[laugh] leading tag and trailing [sigh]"),
            [50275, 3756, 7621, 290, 25462, 220, 50268])
    }

    func testContractionsDigitsAndDoubleSpaces() throws {
        let tokenizer = try loadTokenizer()
        XCTAssertEqual(
            tokenizer.encode("It's 3.14, isn't it?  Multiple  spaces."),
            [1026, 338, 513, 13, 1415, 11, 2125, 470, 340, 30, 220, 20401, 220, 9029, 13])
    }

    func testPuncNormTurboVariant() {
        // Capitalization + trailing full stop.
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm("hello world"), "Hello world.")
        // Whitespace collapse (upstream " ".join(text.split())).
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm("Two  spaces   here."), "Two spaces here.")
        // LLM-punc replacement; ellipsis char rewritten, ASCII "..." kept
        // (the turbo variant drops the multilingual "..." rule).
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm("Wait… what: no"), "Wait, what, no.")
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm("Really..."), "Really...")
        // Comma already ends the sentence — no full stop appended.
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm("Trailing comma,"), "Trailing comma,")
        // Empty input → canned prompt.
        XCTAssertEqual(
            ChatterboxNanoTokenizer.puncNorm(""),
            "You need to add some text for me to talk.")
    }
}
