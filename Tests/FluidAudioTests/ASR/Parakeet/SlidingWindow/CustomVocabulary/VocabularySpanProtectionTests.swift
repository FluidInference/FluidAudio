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

    func testRoundingNoiseIsIgnoredWhenCtcFramesAreOffTheTdtGrid() {
        // A 2.4 s clip: 31 CTC frames of 0.0774 s, while TDT words sit on the 0.08 s grid.
        let words = [word("mister", frame: 10, frames: 5), word("Quilter", frame: 15, frames: 4)]
        let window = VocabularyRescorer.protectedSearchWindow(
            for: candidate("mister", term: "Quilter", indices: [0], words: words, similarity: 0.57),
            wordTimings: words, recognizedSpans: recognized(words, ["Quilter"]),
            frameDuration: 38_400.0 / 31 / 16_000, frameCount: 31, marginSeconds: 0.5)
        XCTAssertTrue(window.createsAdjacentDuplicate)
        XCTAssertLessThanOrEqual(window.frames.upperBound, 15)
    }

    func testAListedSubTermOnlyCountsWhereTheNameWouldDuplicateIt() {
        // The span already starts with the listed first name: correcting it adds no second "Chandra".
        let leading = [
            word("Chandra", frame: 0, frames: 5), word("Chandra", frame: 5, frames: 5),
            word("Shankhar", frame: 10, frames: 5), word("Solansky", frame: 15, frames: 5),
        ]
        let leadingWindow = VocabularyRescorer.protectedSearchWindow(
            for: candidate(
                "Chandra Shankhar Solansky", term: "Chandra Shekhar Solanki", indices: [1, 2, 3], words: leading),
            wordTimings: leading, recognizedSpans: recognized(leading, ["Chandra", "Chandra Shekhar Solanki"]),
            frameDuration: frameDuration, frameCount: 100, marginSeconds: 0.5)
        XCTAssertFalse(leadingWindow.createsAdjacentDuplicate)

        // "Claude" after the span is not the end of "Claude Code".
        let trailing = [
            word("Claude", frame: 0, frames: 5), word("Cold", frame: 5, frames: 5),
            word("Claude", frame: 10, frames: 5), word("is", frame: 15, frames: 3),
        ]
        let trailingWindow = VocabularyRescorer.protectedSearchWindow(
            for: candidate("Claude Cold", term: "Claude Code", indices: [0, 1], words: trailing),
            wordTimings: trailing, recognizedSpans: recognized(trailing, ["Claude", "Claude Code"]),
            frameDuration: frameDuration, frameCount: 100, marginSeconds: 0.5)
        XCTAssertFalse(trailingWindow.createsAdjacentDuplicate)
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
            ("Envious Lab it's", "Envious Labs", "Envious Labs"),
            ("John Collar he's", "John Collier", "John Collier"),
            ("C. S. Solanki's", "Chandra Shekhar Solanki", "Chandra Shekhar Solanki's"),
            ("Archit's", "Arqit", "Arqit's"),
            ("Win's", "Nguyen", "Nguyen's"),
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
