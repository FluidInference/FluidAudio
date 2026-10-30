import XCTest

@testable import FluidAudio

/// Model-free checks of the span-protection helpers (#1041 review): frame-exact overlap, alignment of recognized
/// phrases inside joined or split corrections, listed sub-terms beside a full name, and possessive handling.
final class VocabularySpanProtectionTests: XCTestCase {

    private let frameDuration = 0.08

    /// A word timed the way TDT builds it: start `frame * d`, end `start + frames * d`.
    private func word(_ text: String, frame: Int, frames: Int) -> VocabularyRescorer.WordTiming {
        let start = Double(frame) * frameDuration
        return VocabularyRescorer.WordTiming(
            word: text, startTime: start, endTime: start + Double(frames) * frameDuration, tokenRange: nil)
    }

    private func candidate(
        _ phrase: String, term: String, indices: [Int], words: [VocabularyRescorer.WordTiming],
        similarity: Float = 0.6
    ) -> VocabularyRescorer.CTCMatchCandidate {
        VocabularyRescorer.CTCMatchCandidate(
            origin: .termCentricMultiWord, originalPhrase: phrase, vocabTerm: term, matchedAlias: nil,
            vocabTokens: [1], similarity: similarity, spanLength: indices.count, spanIndices: indices,
            tokenRange: nil, spanStartTime: words[indices[0]].startTime,
            spanEndTime: words[indices[indices.count - 1]].endTime)
    }

    private func recognized(
        _ words: [VocabularyRescorer.WordTiming], _ terms: [String]
    ) -> VocabularyRescorer.RecognizedVocabularySpans {
        VocabularyRescorer.recognizedVocabularySpans(
            in: words, canonicalTerms: terms.map { CustomVocabularyTerm(text: $0) })
    }

    func testAnAdjacentRecognizedNameIsProtectedDespiteRoundingNoise() {
        let words = [word("mister", frame: 27, frames: 2), word("Quilter", frame: 29, frames: 4)]
        // 27·d + 2·d lands a few ulp after 29·d: the two words "overlap" in seconds.
        XCTAssertGreaterThan(words[0].endTime, words[1].startTime)
        let window = VocabularyRescorer.protectedSearchWindow(
            for: candidate("mister", term: "Quilter", indices: [0], words: words, similarity: 0.57),
            wordTimings: words, recognizedSpans: recognized(words, ["Quilter"]), frameDuration: frameDuration,
            frameCount: 100, marginSeconds: 0.5)
        XCTAssertTrue(window.createsAdjacentDuplicate)
        XCTAssertLessThanOrEqual(window.frames.upperBound, 29, "search must stop at the recognized name")
    }

    func testAListedSurnameBesideTheFullNameBoundsItsSearch() {
        let words = [
            word("Chandra", frame: 0, frames: 5), word("Shankar", frame: 5, frames: 5),
            word("Solanki", frame: 10, frames: 5),
        ]
        let window = VocabularyRescorer.protectedSearchWindow(
            for: candidate("Chandra Shankar", term: "Chandra Shekhar Solanki", indices: [0, 1], words: words),
            wordTimings: words, recognizedSpans: recognized(words, ["Chandra Shekhar Solanki", "Solanki"]),
            frameDuration: frameDuration, frameCount: 100, marginSeconds: 0.5)
        XCTAssertTrue(window.createsAdjacentDuplicate)
        XCTAssertLessThanOrEqual(window.frames.upperBound, 10)
    }

    func testRecognizedPhrasesSurviveJoinedAndSplitCorrectionsButAreNotSwallowed() {
        func preserves(_ texts: [String], term: String, listed: [String]) -> Bool {
            let words = texts.enumerated().map { word($1, frame: $0 * 5, frames: 5) }
            return VocabularyRescorer.preservesRecognizedSpans(
                candidate(texts.joined(separator: " "), term: term, indices: Array(texts.indices), words: words),
                recognizedSpans: recognized(words, listed + [term]), wordTimings: words)
        }
        XCTAssertTrue(preserves(["Open", "AI"], term: "OpenAI", listed: ["AI"]), "joined compound")
        XCTAssertTrue(
            preserves(["Chandra-Shankar", "Solanki"], term: "Chandra Shekhar Solanki", listed: ["Solanki"]),
            "hyphen-joined source word")
        XCTAssertTrue(preserves(["John", "Collar"], term: "John Collier", listed: ["John"]), "recognized prefix")
        XCTAssertFalse(preserves(["in", "Envious", "Labs."], term: "Envious Labs", listed: []), "#967 clip")
        XCTAssertFalse(preserves(["mister", "Quilter"], term: "Quilter", listed: []), "neighbour swallowed")
    }

    func testPossessivesFollowTheNameNotItsOwnFinalS() {
        let cases: [(original: String, term: String, expected: String)] = [
            ("Quiltor's", "Quilter", "Quilter's"),
            ("Quiltor’s", "Quilter", "Quilter’s"),
            ("Quilt er's", "Quilter", "Quilter's"),
            ("Collins's", "Collins", "Collins's"),
            ("Jonnes'", "Jones", "Jones'"),
            ("Envious Lab's", "Envious Labs", "Envious Labs"),
            ("Jone's", "Jones", "Jones"),
            ("Quiltor it's", "Quilter", "Quilter"),
        ]
        for (original, term, expected) in cases {
            XCTAssertEqual(
                VocabularyRescorer.preservingPossessive(term, from: original), expected, "\(original) → \(term)")
        }
    }

    func testCanonicalMatchAcceptsPossessivesOfTheTermOnly() {
        XCTAssertTrue(VocabularyRescorer.isCanonicalMatch("Quilter's", canonical: "Quilter"))
        XCTAssertTrue(VocabularyRescorer.isCanonicalMatch("Quilter’s,", canonical: "Quilter"))
        XCTAssertTrue(VocabularyRescorer.isCanonicalMatch("Jones'", canonical: "Jones"))
        XCTAssertFalse(VocabularyRescorer.isCanonicalMatch("Quilters", canonical: "Quilter"))
        XCTAssertFalse(VocabularyRescorer.isCanonicalMatch("Lab's", canonical: "Labs"))
    }
}
