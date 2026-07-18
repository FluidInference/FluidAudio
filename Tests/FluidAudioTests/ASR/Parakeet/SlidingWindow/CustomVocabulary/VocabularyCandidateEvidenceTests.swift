import XCTest

@testable import FluidAudio

final class VocabularyCandidateEvidenceTests: XCTestCase {
    func testCandidateEvidenceOutputPreservesBaseAndAcceptedRejectedCandidates() {
        let accepted = makeEvidence(
            basePhrase: "testing",
            canonicalTerm: "ESLint",
            matchedAlias: "E S lint",
            rawVocabularyScore: -2.5,
            rawOriginalScore: -5.0,
            effectiveBoost: 1.25,
            legacyShouldReplace: true,
            wordRange: 1..<2,
            tokenRange: 3..<5,
            startTime: 0.4,
            endTime: 0.8,
            reason: "accepted"
        )
        let rejected = makeEvidence(
            basePhrase: "document",
            canonicalTerm: "DOCX",
            matchedAlias: nil,
            rawVocabularyScore: -8.0,
            rawOriginalScore: -3.0,
            effectiveBoost: 1.0,
            legacyShouldReplace: false,
            wordRange: 3..<4,
            tokenRange: 7..<8,
            startTime: 1.2,
            endTime: 1.6,
            reason: "rejected"
        )

        let output = VocabularyRescorer.CandidateEvidenceOutput(
            baseText: "Please keep testing this document",
            candidates: [accepted, rejected]
        )

        XCTAssertEqual(output.baseText, "Please keep testing this document")
        XCTAssertEqual(output.candidates.count, 2)
        XCTAssertEqual(output.candidates.map(\.legacyDecision), [.accepted, .rejected])
        XCTAssertEqual(output.candidates[0].matchedAlias, "E S lint")
        XCTAssertNil(output.candidates[1].matchedAlias, "nil must mean the canonical form matched")
    }

    func testCandidateEvidenceMapsScoresAndHalfOpenSpansWithoutMutatingBaseText() {
        let evidence = makeEvidence(
            basePhrase: "test ing",
            canonicalTerm: "Testing",
            matchedAlias: "test-ing",
            rawVocabularyScore: -3.25,
            rawOriginalScore: -4.75,
            effectiveBoost: 0.5,
            legacyShouldReplace: true,
            wordRange: 2..<4,
            tokenRange: 5..<9,
            startTime: 0.75,
            endTime: 1.5,
            reason: "legacy accepted"
        )

        XCTAssertEqual(evidence.basePhrase, "test ing")
        XCTAssertEqual(evidence.canonicalTerm, "Testing")
        XCTAssertEqual(evidence.matchedAlias, "test-ing")
        XCTAssertEqual(evidence.similarity, 0.875, accuracy: 0.0001)
        XCTAssertEqual(evidence.rawVocabularyCTCScore, -3.25)
        XCTAssertEqual(evidence.rawOriginalCTCScore, -4.75)
        XCTAssertEqual(evidence.effectiveBoost, 0.5)
        XCTAssertEqual(evidence.wordRange, 2..<4, "wordRange is half-open")
        XCTAssertEqual(evidence.tokenRange, 5..<9, "tokenRange is half-open")
        XCTAssertEqual(evidence.startTime, 0.75)
        XCTAssertEqual(evidence.endTime, 1.5)
        XCTAssertEqual(evidence.legacyDecision, .accepted)
        XCTAssertEqual(evidence.reason, "legacy accepted")
    }

    func testUnavailableScoresRemainNilInsteadOfUsingLegacySentinels() {
        let candidate = VocabularyRescorer.CTCMatchCandidate(
            originalPhrase: "ordinary",
            vocabTerm: "Azure",
            matchedAlias: nil,
            vocabTokens: [10, 11],
            similarity: 0.625,
            spanLength: 1,
            spanIndices: [0],
            tokenRange: nil,
            spanStartTime: 0.0,
            spanEndTime: 0.4
        )
        let result = VocabularyRescorer.CTCMatchResult(
            shouldReplace: false,
            originalScore: -.infinity,
            boostedVocabScore: -.infinity,
            rawVocabularyCTCScore: nil,
            rawOriginalCTCScore: nil,
            effectiveBoost: nil,
            replacement: "Azure",
            reason: "Tokenizer unavailable"
        )

        let evidence = VocabularyRescorer.makeCandidateEvidence(candidate: candidate, result: result)

        XCTAssertNil(evidence.rawVocabularyCTCScore)
        XCTAssertNil(evidence.rawOriginalCTCScore)
        XCTAssertNil(evidence.effectiveBoost)
        XCTAssertNil(evidence.tokenRange)
        XCTAssertEqual(evidence.legacyDecision, .rejected)
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

    private func makeEvidence(
        basePhrase: String,
        canonicalTerm: String,
        matchedAlias: String?,
        rawVocabularyScore: Float,
        rawOriginalScore: Float,
        effectiveBoost: Float,
        legacyShouldReplace: Bool,
        wordRange: Range<Int>,
        tokenRange: Range<Int>,
        startTime: TimeInterval,
        endTime: TimeInterval,
        reason: String
    ) -> VocabularyRescorer.CandidateEvidence {
        let candidate = VocabularyRescorer.CTCMatchCandidate(
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
            shouldReplace: legacyShouldReplace,
            originalScore: rawOriginalScore,
            boostedVocabScore: rawVocabularyScore + effectiveBoost,
            rawVocabularyCTCScore: rawVocabularyScore,
            rawOriginalCTCScore: rawOriginalScore,
            effectiveBoost: effectiveBoost,
            replacement: canonicalTerm,
            reason: reason
        )
        return VocabularyRescorer.makeCandidateEvidence(candidate: candidate, result: result)
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
