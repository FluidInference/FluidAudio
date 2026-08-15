import XCTest

@testable import FluidAudio

final class VocabularyCandidateEvidenceTests: XCTestCase {
    func testCandidateEvidenceOutputExposesExactBaseWordsAndTwoStageOutcomes() {
        let accepted = makeEvidence(
            candidateID: 0,
            origin: .termCentricSingleWord,
            basePhrase: "testing",
            canonicalTerm: "ESLint",
            matchedAlias: "E S lint",
            rawVocabularyScore: -2.5,
            rawOriginalScore: -5.0,
            effectiveBoost: 1.25,
            wordRange: 2..<3,
            tokenRange: 3..<5,
            baseTextUTF8Range: 12..<19,
            startTime: 0.4,
            endTime: 0.8,
            reason: "accepted"
        )
        let rejected = makeEvidence(
            candidateID: 1,
            origin: .spotterRescue,
            basePhrase: "document",
            canonicalTerm: "DOCX",
            matchedAlias: nil,
            rawVocabularyScore: -8.0,
            rawOriginalScore: -3.0,
            effectiveBoost: 1.0,
            wordRange: 4..<5,
            tokenRange: nil,
            baseTextUTF8Range: nil,
            startTime: 1.2,
            endTime: 1.6,
            reason: "rejected"
        )

        let output = VocabularyRescorer.CandidateEvidenceOutput(
            baseText: "Please keep testing this document",
            baseWords: ["Please", "keep", "testing", "this", "document"],
            candidates: [accepted, rejected]
        )

        XCTAssertEqual(output.baseText, "Please keep testing this document")
        XCTAssertEqual(output.baseWords, ["Please", "keep", "testing", "this", "document"])
        XCTAssertEqual(output.candidates.map(\.candidateID), [0, 1])
        XCTAssertEqual(output.candidates.map(\.comparisonPassed), [true, false])
        XCTAssertEqual(output.candidates.map(\.legacyOutcome), [.applied, .rejectedByComparison])
        XCTAssertEqual(output.candidates.map(\.origin), [.termCentricSingleWord, .spotterRescue])
        XCTAssertEqual(output.candidates[0].matchedAlias, "E S lint")
        XCTAssertNil(output.candidates[1].matchedAlias, "nil means the canonical form matched")
    }

    func testCandidateIDsAreNonNegativeSequentialAndOutputLocal() {
        var firstCollector: VocabularyRescorer.CandidateEvidenceCollector? = collector(
            baseText: "testing document",
            baseWords: ["testing", "document"]
        )
        let firstID = VocabularyRescorer.recordCandidateEvidence(
            candidate: makeCandidate(
                origin: .termCentricSingleWord,
                basePhrase: "testing",
                canonicalTerm: "ESLint",
                spanIndices: [0],
                tokenRange: 0..<1
            ),
            result: passingResult(replacement: "ESLint"),
            candidateEvidence: &firstCollector
        )
        let secondID = VocabularyRescorer.recordCandidateEvidence(
            candidate: makeCandidate(
                origin: .spotterRescue,
                basePhrase: "document",
                canonicalTerm: "DOCX",
                spanIndices: [1],
                tokenRange: nil
            ),
            result: passingResult(replacement: "DOCX"),
            candidateEvidence: &firstCollector
        )

        var secondCollector: VocabularyRescorer.CandidateEvidenceCollector? = collector(
            baseText: "testing",
            baseWords: ["testing"]
        )
        let restartedID = VocabularyRescorer.recordCandidateEvidence(
            candidate: makeCandidate(
                origin: .wordCentric,
                basePhrase: "testing",
                canonicalTerm: "ESLint",
                spanIndices: [0],
                tokenRange: 0..<1
            ),
            result: passingResult(replacement: "ESLint"),
            candidateEvidence: &secondCollector
        )

        XCTAssertEqual([firstID, secondID], [0, 1])
        XCTAssertEqual(firstCollector?.candidates.map(\.candidateID), [0, 1])
        XCTAssertTrue(firstCollector?.candidates.allSatisfy { $0.candidateID >= 0 } == true)
        XCTAssertEqual(restartedID, 0, "IDs restart for each CandidateEvidenceOutput")
    }

    func testEmptySourceWordOutputIsExplicit() {
        let output = VocabularyRescorer.CandidateEvidenceOutput(
            baseText: "",
            baseWords: [],
            candidates: []
        )

        XCTAssertEqual(output.baseText, "")
        XCTAssertTrue(output.baseWords.isEmpty)
        XCTAssertTrue(output.candidates.isEmpty)
    }

    func testUnavailableScoresAndMissingAlignmentRemainNil() {
        var evidenceCollector: VocabularyRescorer.CandidateEvidenceCollector? = .init(
            baseText: "ordinary",
            baseWords: ["ordinary"],
            alignedWordRanges: [nil],
            candidates: []
        )
        let candidate = makeCandidate(
            origin: .spotterRescue,
            basePhrase: "ordinary",
            canonicalTerm: "Azure",
            spanIndices: [0],
            tokenRange: nil
        )
        let result = VocabularyRescorer.CTCMatchResult(
            shouldReplace: false,
            comparisonWasPerformed: false,
            originalScore: -.infinity,
            boostedVocabScore: -.infinity,
            rawVocabularyCTCScore: nil,
            rawOriginalCTCScore: nil,
            effectiveBoost: nil,
            replacement: "Azure",
            reason: "No tokenizer available"
        )

        _ = VocabularyRescorer.recordCandidateEvidence(
            candidate: candidate,
            result: result,
            candidateEvidence: &evidenceCollector
        )
        let evidence = try! XCTUnwrap(evidenceCollector?.candidates.first)

        XCTAssertNil(evidence.rawVocabularyCTCScore)
        XCTAssertNil(evidence.rawOriginalCTCScore)
        XCTAssertNil(evidence.effectiveBoost)
        XCTAssertNil(evidence.tokenRange)
        XCTAssertNil(evidence.baseTextUTF8Range)
        XCTAssertFalse(evidence.comparisonPassed)
        XCTAssertEqual(evidence.legacyOutcome, .unavailableEvidence)
    }

    func testNormalizedFormsDistinguishCanonicalFromExactAlias() {
        let forms = VocabularyRescorer.normalizedForms(
            canonicalTerm: "ESLint",
            aliases: ["E S lint", "es-lint"]
        )

        XCTAssertEqual(forms[0].normalized, "eslint")
        XCTAssertNil(forms[0].matchedAlias)
        XCTAssertEqual(forms[1].normalized, "e s lint")
        XCTAssertEqual(forms[1].matchedAlias, "E S lint")
        XCTAssertEqual(forms[2].normalized, "es-lint")
        XCTAssertEqual(forms[2].matchedAlias, "es-lint")
    }

    func testWordTimingsRetainContiguousHalfOpenTokenRanges() {
        let timings = VocabularyRescorer.buildWordTimings(from: [
            makeTokenTiming(token: "▁test", tokenID: 1, start: 0.0, end: 0.2),
            makeTokenTiming(token: "ing", tokenID: 2, start: 0.2, end: 0.4),
            makeTokenTiming(token: "▁again", tokenID: 3, start: 0.4, end: 0.8),
        ])

        XCTAssertEqual(timings.map(\.word), ["testing", "again"])
        XCTAssertEqual(timings.map(\.tokenRange), [0..<2, 2..<3].map(Optional.some))
    }

    func testWordTimingsUseNilForNoncontiguousTokenProvenance() {
        let timings = VocabularyRescorer.buildWordTimings(from: [
            makeTokenTiming(token: "▁test", tokenID: 1, start: 0.0, end: 0.2),
            makeTokenTiming(token: "<blank>", tokenID: 0, start: 0.2, end: 0.3),
            makeTokenTiming(token: "ing", tokenID: 2, start: 0.3, end: 0.5),
        ])

        XCTAssertEqual(timings.map(\.word), ["testing"])
        XCTAssertNil(timings[0].tokenRange)
    }

    func testExactAlignmentHandlesPunctuationContractionsHyphensAndUnicode() throws {
        let baseText = "“Hello,” don't re-enter Café ☕️."
        let baseWords = ["“Hello,”", "don't", "re-enter", "Café"]
        let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: baseText,
            baseWords: baseWords
        )

        XCTAssertEqual(try text(in: baseText, range: ranges[0]), "Hello")
        XCTAssertEqual(try text(in: baseText, range: ranges[1]), "don't")
        XCTAssertEqual(try text(in: baseText, range: ranges[2]), "re-enter")
        XCTAssertEqual(try text(in: baseText, range: ranges[3]), "Café")

        let phraseRange = VocabularyRescorer.candidateBaseTextUTF8Range(
            wordRange: 1..<3,
            alignedWordRanges: ranges,
            baseText: baseText,
            basePhrase: "don't re-enter"
        )
        XCTAssertEqual(try text(in: baseText, range: phraseRange), "don't re-enter")
        XCTAssertEqual(baseText, "“Hello,” don't re-enter Café ☕️.")
    }

    func testExactAlignmentPreservesTechnicalBoundarySymbols() throws {
        let baseText = "Use C++, C#, .NET, @MainActor, #if, and $PATH."
        let baseWords = ["Use", "C++,", "C#,", ".NET,", "@MainActor,", "#if,", "and", "$PATH."]
        let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: baseText,
            baseWords: baseWords
        )

        XCTAssertEqual(try text(in: baseText, range: ranges[1]), "C++")
        XCTAssertEqual(try text(in: baseText, range: ranges[2]), "C#")
        XCTAssertEqual(try text(in: baseText, range: ranges[3]), ".NET")
        XCTAssertEqual(try text(in: baseText, range: ranges[4]), "@MainActor")
        XCTAssertEqual(try text(in: baseText, range: ranges[5]), "#if")
        XCTAssertEqual(try text(in: baseText, range: ranges[7]), "$PATH")
        XCTAssertEqual(baseText, "Use C++, C#, .NET, @MainActor, #if, and $PATH.")
    }

    func testExactAlignmentFailsClosedForAmbiguousBoundarySyntax() {
        let cases = ["String?", "func()", "[Azure]"]

        for baseText in cases {
            let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
                baseText: baseText,
                baseWords: [baseText]
            )

            XCTAssertEqual(ranges.count, 1)
            XCTAssertNil(ranges[0], "Ambiguous boundary syntax must not produce a mutation range: \(baseText)")
        }
    }

    func testExactAlignmentFailsClosedForNonTechnicalSymbols() {
        let cases = ["Azure™", "Azure☕️"]

        for baseText in cases {
            let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
                baseText: baseText,
                baseWords: [baseText]
            )

            XCTAssertEqual(ranges.count, 1)
            XCTAssertNil(ranges[0], "Non-technical symbols must not become part of a mutation range: \(baseText)")
        }
    }

    func testBlankAndPadGapsDoNotFabricateTokenRangesOrBreakTextAlignment() throws {
        let timings = VocabularyRescorer.buildWordTimings(from: [
            makeTokenTiming(token: "▁hello", tokenID: 1, start: 0.0, end: 0.2),
            makeTokenTiming(token: "<blank>", tokenID: 0, start: 0.2, end: 0.3),
            makeTokenTiming(token: "▁world", tokenID: 2, start: 0.3, end: 0.5),
            makeTokenTiming(token: "<pad>", tokenID: 0, start: 0.5, end: 0.6),
        ])
        let baseWords = timings.map(\.word)
        let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: "hello world",
            baseWords: baseWords
        )

        XCTAssertEqual(baseWords, ["hello", "world"])
        XCTAssertEqual(timings[0].tokenRange, 0..<1)
        XCTAssertEqual(timings[1].tokenRange, 2..<3)
        XCTAssertNil(
            VocabularyRescorer.tokenRange(for: [0, 1], in: timings),
            "A blank-token gap must not become a fabricated contiguous multiword token range"
        )
        XCTAssertEqual(try text(in: "hello world", range: ranges[0]), "hello")
        XCTAssertEqual(try text(in: "hello world", range: ranges[1]), "world")
    }

    func testRepeatedWordsResolveOnlyThroughTheCompleteSequence() throws {
        let resolved = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: "go go",
            baseWords: ["go", "go"]
        )
        XCTAssertEqual(try text(in: "go go", range: resolved[0]), "go")
        XCTAssertEqual(try text(in: "go go", range: resolved[1]), "go")

        let ambiguous = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: "go go",
            baseWords: ["go"]
        )
        XCTAssertEqual(ambiguous.count, 1)
        XCTAssertNil(ambiguous[0])
    }

    func testMissingReorderedAndNormalizationOnlyMatchesFailClosed() {
        let cases: [(String, [String])] = [
            ("one two", ["two", "one"]),
            ("one", ["one", "two"]),
            ("Azure", ["azure"]),
            ("Café", ["Cafe"]),
            ("Café", ["Cafe\u{301}"]),
        ]

        for (baseText, baseWords) in cases {
            let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
                baseText: baseText,
                baseWords: baseWords
            )
            XCTAssertEqual(ranges.count, baseWords.count)
            XCTAssertTrue(ranges.allSatisfy { $0 == nil }, "Unexpected alignment for \(baseText) / \(baseWords)")
        }
    }

    func testCandidateRangeRejectsPhraseMismatchAndInvalidUTF8Boundaries() {
        let baseText = "Café noir"
        let ranges = VocabularyRescorer.alignBaseWordsToUTF8Ranges(
            baseText: baseText,
            baseWords: ["Café", "noir"]
        )

        XCTAssertNil(
            VocabularyRescorer.candidateBaseTextUTF8Range(
                wordRange: 0..<1,
                alignedWordRanges: ranges,
                baseText: baseText,
                basePhrase: "coffee"
            ))
        XCTAssertNil(VocabularyRescorer.substring(in: baseText, utf8Range: 4..<5))
    }

    private func collector(
        baseText: String,
        baseWords: [String]
    ) -> VocabularyRescorer.CandidateEvidenceCollector {
        VocabularyRescorer.CandidateEvidenceCollector(
            baseText: baseText,
            baseWords: baseWords,
            alignedWordRanges: VocabularyRescorer.alignBaseWordsToUTF8Ranges(
                baseText: baseText,
                baseWords: baseWords
            ),
            candidates: []
        )
    }

    private func makeEvidence(
        candidateID: Int,
        origin: VocabularyRescorer.CandidateOrigin,
        basePhrase: String,
        canonicalTerm: String,
        matchedAlias: String?,
        rawVocabularyScore: Float,
        rawOriginalScore: Float,
        effectiveBoost: Float,
        wordRange: Range<Int>,
        tokenRange: Range<Int>?,
        baseTextUTF8Range: Range<Int>?,
        startTime: TimeInterval,
        endTime: TimeInterval,
        reason: String
    ) -> VocabularyRescorer.CandidateEvidence {
        let candidate = VocabularyRescorer.CTCMatchCandidate(
            origin: origin,
            originalPhrase: basePhrase,
            vocabTerm: canonicalTerm,
            matchedAlias: matchedAlias,
            vocabTokens: [10, 11],
            similarity: 0.875,
            spanLength: wordRange.count,
            spanIndices: Array(wordRange),
            tokenRange: tokenRange,
            spanStartTime: startTime,
            spanEndTime: endTime
        )
        let result = VocabularyRescorer.CTCMatchResult(
            shouldReplace: rawVocabularyScore + effectiveBoost > rawOriginalScore,
            comparisonWasPerformed: true,
            originalScore: rawOriginalScore,
            boostedVocabScore: rawVocabularyScore + effectiveBoost,
            rawVocabularyCTCScore: rawVocabularyScore,
            rawOriginalCTCScore: rawOriginalScore,
            effectiveBoost: effectiveBoost,
            replacement: canonicalTerm,
            reason: reason
        )
        return VocabularyRescorer.makeCandidateEvidence(
            candidateID: candidateID,
            candidate: candidate,
            result: result,
            wordRange: wordRange,
            baseTextUTF8Range: baseTextUTF8Range
        )
    }

    private func makeCandidate(
        origin: VocabularyRescorer.CandidateOrigin,
        basePhrase: String,
        canonicalTerm: String,
        spanIndices: [Int],
        tokenRange: Range<Int>?
    ) -> VocabularyRescorer.CTCMatchCandidate {
        VocabularyRescorer.CTCMatchCandidate(
            origin: origin,
            originalPhrase: basePhrase,
            vocabTerm: canonicalTerm,
            matchedAlias: nil,
            vocabTokens: [10, 11],
            similarity: 0.875,
            spanLength: spanIndices.count,
            spanIndices: spanIndices,
            tokenRange: tokenRange,
            spanStartTime: 0,
            spanEndTime: 0.5
        )
    }

    private func passingResult(replacement: String) -> VocabularyRescorer.CTCMatchResult {
        VocabularyRescorer.CTCMatchResult(
            shouldReplace: true,
            comparisonWasPerformed: true,
            originalScore: -5,
            boostedVocabScore: -1,
            rawVocabularyCTCScore: -2,
            rawOriginalCTCScore: -5,
            effectiveBoost: 1,
            replacement: replacement,
            reason: "vocabulary > original"
        )
    }

    private func text(in baseText: String, range: Range<Int>?) throws -> String {
        let range = try XCTUnwrap(range)
        return String(try XCTUnwrap(VocabularyRescorer.substring(in: baseText, utf8Range: range)))
    }

    private func makeTokenTiming(
        token: String,
        tokenID: Int,
        start: TimeInterval,
        end: TimeInterval
    ) -> TokenTiming {
        TokenTiming(
            token: token,
            tokenId: tokenID,
            startTime: start,
            endTime: end,
            confidence: 1.0
        )
    }
}
