import Foundation
import XCTest

@testable import FluidAudio

final class ARPALanguageModelTests: XCTestCase {

    // MARK: - Helpers

    /// Create a temporary ARPA file with the given content and return its URL.
    private func writeTemporaryARPA(_ content: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("test_\(UUID().uuidString).arpa")
        try content.write(to: url, atomically: true, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private let sampleARPA = """
        \\data\\
        ngram 1=4
        ngram 2=2

        \\1-grams:
        -1.0\tthe\t-0.5
        -1.2\tcat\t-0.3
        -1.5\tsat\t0.0
        -2.0\t<unk>\t0.0

        \\2-grams:
        -0.5\tthe\tcat
        -0.8\tcat\tsat

        \\end\\
        """

    // MARK: - Loading

    func testLoadARPAFile() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        XCTAssertEqual(lm.unigrams.count, 4, "Should load 4 unigrams")
        XCTAssertEqual(lm.bigrams.count, 2, "Should load 2 bigram contexts")
    }

    func testLoadARPAUnigramValues() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        let theEntry = lm.unigrams["the"]
        XCTAssertNotNil(theEntry)

        // log10(-1.0) * 2.302585 ≈ -2.302585
        let expectedLogProb = -1.0 * ARPALanguageModel.log10ToNat
        XCTAssertEqual(theEntry!.logProb, expectedLogProb, accuracy: 0.001)

        // backoff: log10(-0.5) * 2.302585 ≈ -1.151293
        let expectedBackoff = -0.5 * ARPALanguageModel.log10ToNat
        XCTAssertEqual(theEntry!.backoff, expectedBackoff, accuracy: 0.001)
    }

    func testLoadARPABigrams() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        XCTAssertNotNil(lm.bigrams["the"]?["cat"])
        XCTAssertNotNil(lm.bigrams["cat"]?["sat"])
        XCTAssertNil(lm.bigrams["sat"]?["the"])
    }

    func testLoadNonexistentFileThrows() {
        let bogusURL = URL(fileURLWithPath: "/nonexistent/path/to/model.arpa")
        XCTAssertThrowsError(try ARPALanguageModel.load(from: bogusURL))
    }

    func testLoadEmptyARPA() throws {
        let content = """
            \\data\\
            ngram 1=0

            \\1-grams:

            \\end\\
            """
        let url = try writeTemporaryARPA(content)
        let lm = try ARPALanguageModel.load(from: url)

        XCTAssertTrue(lm.unigrams.isEmpty)
        XCTAssertTrue(lm.bigrams.isEmpty)
    }

    // MARK: - Scoring

    func testScoreBigramAvailable() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        let score = lm.score(word: "cat", prev: "the")
        let expected = -0.5 * ARPALanguageModel.log10ToNat
        XCTAssertEqual(score, expected, accuracy: 0.001)
    }

    func testScoreFallsBackToUnigram() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        // "sat" given "the" — no bigram exists, falls back to unigram("sat") + backoff("the")
        let score = lm.score(word: "sat", prev: "the")
        let unigramLogProb = -1.5 * ARPALanguageModel.log10ToNat
        let backoff = -0.5 * ARPALanguageModel.log10ToNat
        let expected = backoff + unigramLogProb
        XCTAssertEqual(score, expected, accuracy: 0.001)
    }

    func testScoreNoPrevContextSkipsBackoff() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        let score = lm.score(word: "cat", prev: nil)
        let expected = -1.2 * ARPALanguageModel.log10ToNat
        XCTAssertEqual(score, expected, accuracy: 0.001)
    }

    func testScoreOOVWordGetsUnkPenalty() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        let score = lm.score(word: "xyzzy", prev: nil)
        XCTAssertEqual(score, ARPALanguageModel.unkLogProb, accuracy: 0.001)
    }

    func testScoreOOVWithBackoff() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        // OOV word with prev context "the" — backoff("the") + unkLogProb
        let score = lm.score(word: "xyzzy", prev: "the")
        let backoff = -0.5 * ARPALanguageModel.log10ToNat
        let expected = backoff + ARPALanguageModel.unkLogProb
        XCTAssertEqual(score, expected, accuracy: 0.001)
    }

    // MARK: - Beam Search with LM

    func testBeamSearchWithLMInfluencesResult() throws {
        let url = try writeTemporaryARPA(sampleARPA)
        let lm = try ARPALanguageModel.load(from: url)

        // Vocabulary with word-start markers matching ARPA words
        let vocab: [Int: String] = [0: "▁the", 1: "▁cat", 2: "▁dog"]
        let blankId = 3

        // Make "dog" slightly better acoustically than "cat"
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0, -100.0],  // "the" clearly best
            [-100.0, -1.0, -0.9, -100.0],  // "cat" vs "dog" — dog slightly better acoustically
        ]

        // Without LM — should pick "dog" (better acoustic score)
        let noLM = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: nil,
            beamWidth: 10, lmWeight: 0.0, blankId: blankId
        )
        XCTAssertEqual(noLM, "the dog")

        // With strong LM — "the cat" has a bigram entry, "the dog" doesn't
        let withLM = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: lm,
            beamWidth: 10, lmWeight: 5.0, blankId: blankId
        )
        XCTAssertEqual(withLM, "the cat")
    }
}
