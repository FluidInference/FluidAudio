import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsCutPlacementTests: XCTestCase {

    /// The v2.1 English tokenizer (see Fixtures/README.md). Chunk sizing normalizes
    /// each candidate as synthesis does, so token counts come from the real vocabulary.
    private func makeEnglishTokenizer() throws -> SentencePieceTokenizer {
        let modelURL = try XCTUnwrap(
            Bundle.module.url(forResource: "pocket_tts_english_tokenizer", withExtension: "model"))
        return try SentencePieceTokenizer(modelData: Data(contentsOf: modelURL))
    }

    /// Chunks as `synthesizeStreaming` makes them for a voice whose conditioning
    /// ends at `voiceCachePosition` (126 for a shipped voice, 251 for a cloned one).
    private func chunks(
        _ text: String, tokenizer: SentencePieceTokenizer, voiceCachePosition: Int
    ) throws -> [String] {
        let maxTokens = try PocketTtsSynthesizer.effectiveMaxTokensPerChunk(
            requested: PocketTtsConstants.maxTokensPerChunk, voiceCachePosition: voiceCachePosition)
        return PocketTtsSynthesizer.chunkTextWithMetadata(
            text, tokenizer: tokenizer, maxTokens: maxTokens,
            preferredMaxTokens: min(PocketTtsConstants.preferredTokensPerChunk, maxTokens),
            voiceCachePosition: voiceCachePosition
        ).map(\.text)
    }

    private let dashSentence =
        "The humidity feels comfortable despite the approaching storm, and the air carries that "
        + "distinctive electric charge that precedes rain \u{2014} a mix of ozone and wet soil that makes "
        + "everything feel fresh and alive before the first drops hit the pavement."

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

    func testBracketsInsideAWordAreNotAnAside() {
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("we write f(x) for the item(s) we keep"),
            ["we write f(x) for the item(s) we keep"])
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtClauseBoundaries("the call g(a (b) c) returns"),
            ["the call g(a", "(b)", "c) returns"])
    }

    func testRegroupedClausesKeepTheirSeparators() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text =
            "alpha beta,gamma delta\u{2014}epsilon zeta, and then a much longer clause that has to "
            + "go into a chunk of its own because it keeps going for quite a while longer"
        let parts = PocketTtsSynthesizer.splitOversizedSentence(text, tokenizer: tokenizer, maxTokens: 30)

        XCTAssertGreaterThan(parts.count, 1)
        XCTAssertEqual(parts.first, "alpha beta,gamma delta\u{2014}epsilon zeta,")
        for part in parts {
            XCTAssertTrue(text.contains(part), "part is not verbatim text: \(part)")
        }
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

    func testWordSplitUsesTheFewestPartsAndBalancesThem() throws {
        // About 107 and 143 tokens against a limit of 50: both need three parts.
        // Filling each part to the limit strands a short tail on the first, and a
        // fixed per-part target could spill the second into a fourth part.
        let tokenizer = try makeEnglishTokenizer()
        let clause =
            "the scheduler kept retrying the same request over and over again until the operator "
            + "finally noticed the alert on the dashboard late at night"
        for clauses in [3, 4] {
            let text = Array(repeating: clause, count: clauses).joined(separator: " and ")
            let parts = PocketTtsSynthesizer.splitAtWordBoundaries(text, tokenizer: tokenizer, maxTokens: 50)

            XCTAssertEqual(parts.count, 3, "\(clauses) clauses")
            for part in parts {
                XCTAssertTrue(
                    PocketTtsSynthesizer.fitsTextChunk(
                        part, tokenizer: tokenizer, maxTokens: 50, voiceCachePosition: nil),
                    "part over the limit: \(part)")
            }
            let counts = parts.map { tokenizer.encode($0).count }
            XCTAssertLessThanOrEqual(counts.max()! - counts.min()!, 2, "unbalanced parts: \(counts)")
            XCTAssertEqual(parts.joined(separator: " "), text)
        }
    }

    func testTextWithinTheLimitIsNotSplit() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text = "the scheduler kept retrying the same request"
        XCTAssertEqual(
            PocketTtsSynthesizer.splitAtWordBoundaries(text, tokenizer: tokenizer, maxTokens: 50), [text])
    }

    func testZeroTokenLimitTerminates() throws {
        let tokenizer = try makeEnglishTokenizer()
        XCTAssertEqual(
            PocketTtsSynthesizer.chunkText("one two three, four five", tokenizer: tokenizer, maxTokens: 0),
            ["one", "two", "three,", "four", "five"])
    }

    // MARK: - Cuts within the KV cache (#934)

    func testClonedVoiceCutsAtTheDash() throws {
        let tokenizer = try makeEnglishTokenizer()
        XCTAssertEqual(
            try chunks(dashSentence, tokenizer: tokenizer, voiceCachePosition: 251),
            [
                "The humidity feels comfortable despite the approaching storm, and the air carries that "
                    + "distinctive electric charge that precedes rain \u{2014}",
                "a mix of ozone and wet soil that makes everything feel fresh and alive before the first "
                    + "drops hit the pavement.",
            ])
    }

    func testClonedVoiceCutsAfterTheBracketedAside() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text =
            "The control plane on the Pi (which also serves the dashboard) polls the shard every fifteen "
            + "minutes and merges whatever services it finds into a single response that the rest of the "
            + "platform treats as the source of truth."
        let result = try chunks(text, tokenizer: tokenizer, voiceCachePosition: 251)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result.first, "The control plane on the Pi (which also serves the dashboard)")
    }

    func testClonedVoiceBalancesASentenceWithoutPunctuation() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text =
            "Nobody should ever need to restart the whole machine just because one small demand loaded "
            + "model ran out of memory while another bigger one was still busy answering a long question "
            + "from the voice loop."
        let counts = try chunks(text, tokenizer: tokenizer, voiceCachePosition: 251)
            .map { tokenizer.encode($0).count }

        XCTAssertEqual(counts.count, 2)
        XCTAssertLessThanOrEqual(abs(counts[0] - counts[1]), 2, "unbalanced parts: \(counts)")
    }

    func testShippedVoiceKeepsTheSentenceWholeAndCutsALongerOneAtTheDash() throws {
        let tokenizer = try makeEnglishTokenizer()
        XCTAssertEqual(try chunks(dashSentence, tokenizer: tokenizer, voiceCachePosition: 126), [dashSentence])

        let longer =
            "The humidity feels comfortable despite the approaching storm and the air carries that "
            + "distinctive electric charge that precedes rain\u{2014}a mix of ozone and wet soil that makes "
            + "everything feel fresh and alive before the first drops hit the pavement and the children run "
            + "back inside laughing while their parents close the windows."
        let result = try chunks(longer, tokenizer: tokenizer, voiceCachePosition: 126)
        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result[0].hasSuffix("precedes rain\u{2014}"), "cut not at the dash: \(result)")
        XCTAssertTrue(result[1].hasPrefix("a mix of ozone"), "cut not at the dash: \(result)")
    }
}
