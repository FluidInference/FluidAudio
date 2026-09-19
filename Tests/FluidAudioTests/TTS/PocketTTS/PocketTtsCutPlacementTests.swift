import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsCutPlacementTests: XCTestCase {

    /// One token per word, so token counts equal word counts.
    private func makeTokenizer(words: [String]) throws -> SentencePieceTokenizer {
        let letters = "abcdefghijklmnopqrstuvwxyz".map { String($0) }
        return try SentencePieceTestModel.tokenizer(
            pieces: ["<unk>", "\u{2581}"] + words.map { "\u{2581}" + $0 } + letters)
    }

    private let numberWords = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    ]

    // MARK: - Dashes as clause boundaries

    func testSpacedEmDashEndsAClause() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("that precedes rain \u{2014} a mix of ozone"),
            ["that precedes rain \u{2014}", "a mix of ozone"])
    }

    func testUnspacedEmDashEndsAClause() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("precedes rain\u{2014}a mix"),
            ["precedes rain\u{2014}", "a mix"])
    }

    func testSpacedHyphenAndEnDashEndAClause() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("one thing - another"),
            ["one thing -", "another"])
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("one thing \u{2013} another"),
            ["one thing \u{2013}", "another"])
    }

    func testHyphenatedWordsAndRangesStayWhole() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("a well-known seventy-two degree day").count, 1)
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("see pages 10\u{2013}12 for more").count, 1)
    }

    func testBracketedAsideSplitsAtBothEdges() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("the ridge (which faces west) catches the rain"),
            ["the ridge", "(which faces west)", "catches the rain"])
    }

    func testClosingBracketKeepsTrailingPunctuation() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("the ridge (which faces west), and the valley"),
            ["the ridge", "(which faces west),", "and the valley"])
    }

    func testEllipsisEndsAClause() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("and then\u{2026} nothing happened"),
            ["and then\u{2026}", "nothing happened"])
    }

    func testSentenceStartChunkDoesNotKeepATrailingDash() {
        let (text, _) = PocketTtsSynthesizer.normalizeText(
            "the air carries a charge that precedes rain \u{2014}")
        XCTAssertEqual(text, "The air carries a charge that precedes rain.")
    }

    // MARK: - Runs of sentence punctuation

    func testRunOfSentencePunctuationEndsOneSentence() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitSentences("back from the lab... and then it failed. Did it matter?! Yes."),
            ["back from the lab...", "and then it failed.", "Did it matter?!", "Yes."])
    }

    // MARK: - Balanced word-boundary splitting

    func testOversizedTextSplitsIntoBalancedParts() throws {
        // 12 tokens against a limit of 10 needs two parts. Filling to the
        // limit gives 10 + 2; balancing gives 6 + 6.
        let tokenizer = try makeTokenizer(words: numberWords)
        let parts = PocketTtsSynthesizer.splitAtWordBoundaries(
            numberWords.joined(separator: " "), tokenizer: tokenizer, maxTokens: 10)
        XCTAssertEqual(
            parts,
            [
                "one two three four five six",
                "seven eight nine ten eleven twelve",
            ])
    }

    func testNoPartExceedsTheLimitOrStrandsAShortTail() throws {
        let tokenizer = try makeTokenizer(words: numberWords)
        let text = (numberWords + numberWords + ["one"]).joined(separator: " ")
        let parts = PocketTtsSynthesizer.splitAtWordBoundaries(
            text, tokenizer: tokenizer, maxTokens: 10)

        XCTAssertEqual(parts.count, 3)
        for part in parts {
            let tokens = tokenizer.encode(part).count
            XCTAssertLessThanOrEqual(tokens, 10, "part over the limit: \(part)")
            XCTAssertGreaterThanOrEqual(tokens, 7, "part far below its share: \(part)")
        }
        XCTAssertEqual(parts.joined(separator: " "), text)
    }

    func testTextWithinTheLimitIsNotSplit() throws {
        let tokenizer = try makeTokenizer(words: numberWords)
        let text = numberWords.prefix(8).joined(separator: " ")
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtWordBoundaries(text, tokenizer: tokenizer, maxTokens: 10),
            [text])
    }
}
