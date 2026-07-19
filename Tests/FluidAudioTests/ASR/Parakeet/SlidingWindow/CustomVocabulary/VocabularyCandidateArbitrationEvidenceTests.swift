import XCTest

@testable import FluidAudio

final class VocabularyCandidateArbitrationEvidenceTests: XCTestCase {
    func testEveryDiscoveryPathRetainsASeparateOutputLocalIdentity() {
        let baseText = "rom vimza"
        let baseWords = ["rom", "vimza"]
        var collector: VocabularyRescorer.CandidateEvidenceCollector? = .init(
            baseText: baseText,
            baseWords: baseWords,
            alignedWordRanges: VocabularyRescorer.alignBaseWordsToUTF8Ranges(
                baseText: baseText,
                baseWords: baseWords
            ),
            candidates: []
        )
        let origins: [VocabularyRescorer.CandidateOrigin] = [
            .wordCentric,
            .termCentricSingleWord,
            .termCentricMultiWord,
            .spotterRescue,
        ]

        for origin in origins {
            _ = VocabularyRescorer.recordCandidateEvidence(
                candidate: candidate(
                    origin: origin,
                    phrase: "rom vimza",
                    term: "Romvimza",
                    indices: [0, 1]
                ),
                result: passingResult(replacement: "Romvimza"),
                candidateEvidence: &collector
            )
        }

        let evidence = collector?.candidates ?? []
        XCTAssertEqual(evidence.map(\.candidateID), [0, 1, 2, 3])
        XCTAssertEqual(evidence.map(\.origin), origins)
        XCTAssertEqual(Set(evidence.map(\.candidateID)).count, evidence.count)
        XCTAssertTrue(evidence.allSatisfy(\.comparisonPassed))
        XCTAssertTrue(evidence.allSatisfy { $0.wordRange == 0..<2 })
        XCTAssertTrue(evidence.allSatisfy { $0.baseTextUTF8Range == 0..<baseText.utf8.count })
    }

    func testRejectedNumericReasonsUseLessThanOrEqualAndAcceptedReasonsUseGreaterThan() {
        let accepted = VocabularyRescorer.ctcComparisonReason(
            prefix: "CTC-vs-CTC",
            vocabularyTerm: "ESLint",
            boostedVocabularyScore: -2,
            originalPhrase: "testing",
            originalScore: -4,
            comparisonPassed: true
        )
        let rejected = VocabularyRescorer.ctcComparisonReason(
            prefix: "CTC-vs-CTC",
            vocabularyTerm: "Azure",
            boostedVocabularyScore: -8,
            originalPhrase: "as your",
            originalScore: -3,
            comparisonPassed: false
        )

        XCTAssertTrue(accepted.contains("'ESLint'=-2.00 > 'testing'=-4.00"))
        XCTAssertTrue(rejected.contains("'Azure'=-8.00 <= 'as your'=-3.00"))
        XCTAssertFalse(rejected.contains("-8.00 >"))
    }

    func testUnavailableComparisonDoesNotInventScoresOrANumericReason() {
        let evidence = makeEvidence(
            id: 0,
            origin: .spotterRescue,
            phrase: "ordinary",
            term: "Azure",
            indices: [0],
            result: VocabularyRescorer.CTCMatchResult(
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
        )

        XCTAssertFalse(evidence.comparisonPassed)
        XCTAssertEqual(evidence.legacyOutcome, .unavailableEvidence)
        XCTAssertNil(evidence.rawVocabularyCTCScore)
        XCTAssertNil(evidence.rawOriginalCTCScore)
        XCTAssertNil(evidence.effectiveBoost)
        XCTAssertEqual(evidence.reason, "No tokenizer available")
        XCTAssertFalse(evidence.reason.contains(" > "))
        XCTAssertFalse(evidence.reason.contains(" <= "))

        let nonFiniteNumericReason = makeEvidence(
            id: 1,
            origin: .wordCentric,
            phrase: "ordinary",
            term: "Azure",
            indices: [0],
            result: VocabularyRescorer.CTCMatchResult(
                shouldReplace: false,
                comparisonWasPerformed: true,
                originalScore: -.infinity,
                boostedVocabScore: -7,
                rawVocabularyCTCScore: -8,
                rawOriginalCTCScore: nil,
                effectiveBoost: 1,
                replacement: "Azure",
                reason: "CTC-vs-CTC: 'Azure'=-7.00 > 'ordinary'=-inf"
            )
        )
        XCTAssertFalse(nonFiniteNumericReason.comparisonPassed)
        XCTAssertEqual(nonFiniteNumericReason.legacyOutcome, .rejectedByComparison)
        XCTAssertEqual(
            nonFiniteNumericReason.reason,
            "CTC comparison rejected: one or more finite diagnostic scores were unavailable"
        )
    }

    func testPerformedComparisonRetainsLegacyPassWhenRawOriginalScoreIsNonFinite() {
        let evidence = makeEvidence(
            id: 0,
            origin: .wordCentric,
            phrase: "testing",
            term: "ESLint",
            indices: [0],
            result: VocabularyRescorer.CTCMatchResult(
                shouldReplace: true,
                comparisonWasPerformed: true,
                originalScore: -.infinity,
                boostedVocabScore: -7,
                rawVocabularyCTCScore: -8,
                rawOriginalCTCScore: nil,
                effectiveBoost: 1,
                replacement: "ESLint",
                reason: "CTC-vs-CTC: 'ESLint'=-7.00 > 'testing'=-inf"
            )
        )

        XCTAssertTrue(evidence.comparisonPassed)
        XCTAssertEqual(evidence.legacyOutcome, .applied)
        XCTAssertEqual(evidence.rawVocabularyCTCScore, -8)
        XCTAssertNil(evidence.rawOriginalCTCScore)
        XCTAssertEqual(evidence.effectiveBoost, 1)
        XCTAssertEqual(
            evidence.reason,
            "CTC comparison passed: one or more finite diagnostic scores were unavailable"
        )
        XCTAssertFalse(evidence.reason.lowercased().contains("inf"))
    }

    func testLegacyPathLeavesEvidenceCollectorUnallocated() {
        var collector: VocabularyRescorer.CandidateEvidenceCollector?
        let candidateID = VocabularyRescorer.recordCandidateEvidence(
            candidate: candidate(
                origin: .wordCentric,
                phrase: "testing",
                term: "ESLint",
                indices: [0]
            ),
            result: passingResult(replacement: "ESLint"),
            candidateEvidence: &collector
        )

        XCTAssertNil(candidateID)
        XCTAssertNil(collector)
    }

    func testRomvimzaCimziaOverlapHasOneAppliedAndOneSupersededOutcome() {
        let romvimza = pending(
            id: 0,
            origin: .termCentricSingleWord,
            phrase: "rom vimza",
            term: "Romvimza",
            indices: [0, 1],
            similarity: 0.89
        )
        let cimzia = pending(
            id: 1,
            origin: .termCentricSingleWord,
            phrase: "vimza",
            term: "Cimzia",
            indices: [1],
            similarity: 0.67
        )

        let arbitration = VocabularyRescorer.arbitratePendingReplacements([cimzia, romvimza])
        XCTAssertEqual(arbitration.applied.compactMap(\.candidateID), [0])
        XCTAssertEqual(arbitration.applied.map { $0.result.replacement }, ["Romvimza"])
        XCTAssertEqual(arbitration.supersededCandidateIDs, [1])

        var evidence = [
            makeEvidence(
                id: 0,
                origin: .termCentricSingleWord,
                phrase: "rom vimza",
                term: "Romvimza",
                indices: [0, 1],
                result: passingResult(replacement: "Romvimza")
            ),
            makeEvidence(
                id: 1,
                origin: .termCentricSingleWord,
                phrase: "vimza",
                term: "Cimzia",
                indices: [1],
                result: passingResult(replacement: "Cimzia")
            ),
        ]
        VocabularyRescorer.reconcileLegacyOutcomes(
            candidates: &evidence,
            appliedCandidateIDs: Set(arbitration.applied.compactMap(\.candidateID))
        )

        XCTAssertEqual(evidence.map(\.comparisonPassed), [true, true])
        XCTAssertEqual(evidence.map(\.legacyOutcome), [.applied, .supersededByOverlap])
    }

    func testNonOverlappingPassesApplyWhileFailuresKeepTheirTerminalOutcomes() {
        let first = pending(
            id: 0,
            origin: .wordCentric,
            phrase: "testing",
            term: "ESLint",
            indices: [0],
            similarity: 0.9
        )
        let second = pending(
            id: 1,
            origin: .spotterRescue,
            phrase: "jay son",
            term: "JSON",
            indices: [2, 3],
            similarity: 0.7
        )
        let arbitration = VocabularyRescorer.arbitratePendingReplacements([second, first])
        XCTAssertEqual(Set(arbitration.applied.compactMap(\.candidateID)), [0, 1])
        XCTAssertTrue(arbitration.supersededCandidateIDs.isEmpty)

        var evidence = [
            makeEvidence(
                id: 0,
                origin: .wordCentric,
                phrase: "testing",
                term: "ESLint",
                indices: [0],
                result: passingResult(replacement: "ESLint")
            ),
            makeEvidence(
                id: 1,
                origin: .spotterRescue,
                phrase: "jay son",
                term: "JSON",
                indices: [2, 3],
                result: passingResult(replacement: "JSON")
            ),
            makeEvidence(
                id: 2,
                origin: .termCentricSingleWord,
                phrase: "track",
                term: "Slack",
                indices: [4],
                result: rejectedResult(replacement: "Slack")
            ),
            makeEvidence(
                id: 3,
                origin: .termCentricMultiWord,
                phrase: "as your",
                term: "Azure",
                indices: [5, 6],
                result: unavailableResult(replacement: "Azure")
            ),
        ]
        VocabularyRescorer.reconcileLegacyOutcomes(
            candidates: &evidence,
            appliedCandidateIDs: Set(arbitration.applied.compactMap(\.candidateID))
        )

        XCTAssertEqual(
            evidence.map(\.legacyOutcome),
            [.applied, .applied, .rejectedByComparison, .unavailableEvidence]
        )
    }

    func testEvidenceOutputKeepsUntouchedBaseText() {
        let baseText = "Please keep testing, not ESLint."
        let output = VocabularyRescorer.CandidateEvidenceOutput(
            baseText: baseText,
            baseWords: ["Please", "keep", "testing,", "not", "ESLint."],
            candidates: []
        )

        XCTAssertEqual(Array(output.baseText.utf8), Array(baseText.utf8))
        XCTAssertFalse(output.baseText.isEmpty)
    }

    private func pending(
        id: Int,
        origin: VocabularyRescorer.CandidateOrigin,
        phrase: String,
        term: String,
        indices: [Int],
        similarity: Float
    ) -> VocabularyRescorer.PendingReplacement {
        VocabularyRescorer.PendingReplacement(
            candidateID: id,
            candidate: candidate(
                origin: origin,
                phrase: phrase,
                term: term,
                indices: indices,
                similarity: similarity
            ),
            result: passingResult(replacement: term),
            similarity: similarity
        )
    }

    private func makeEvidence(
        id: Int,
        origin: VocabularyRescorer.CandidateOrigin,
        phrase: String,
        term: String,
        indices: [Int],
        result: VocabularyRescorer.CTCMatchResult
    ) -> VocabularyRescorer.CandidateEvidence {
        VocabularyRescorer.makeCandidateEvidence(
            candidateID: id,
            candidate: candidate(
                origin: origin,
                phrase: phrase,
                term: term,
                indices: indices
            ),
            result: result,
            wordRange: (indices.first ?? 0)..<((indices.last ?? -1) + 1),
            baseTextUTF8Range: nil
        )
    }

    private func candidate(
        origin: VocabularyRescorer.CandidateOrigin,
        phrase: String,
        term: String,
        indices: [Int],
        similarity: Float = 0.8
    ) -> VocabularyRescorer.CTCMatchCandidate {
        VocabularyRescorer.CTCMatchCandidate(
            origin: origin,
            originalPhrase: phrase,
            vocabTerm: term,
            matchedAlias: nil,
            vocabTokens: [1, 2],
            similarity: similarity,
            spanLength: indices.count,
            spanIndices: indices,
            tokenRange: nil,
            spanStartTime: 0,
            spanEndTime: 1
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

    private func rejectedResult(replacement: String) -> VocabularyRescorer.CTCMatchResult {
        VocabularyRescorer.CTCMatchResult(
            shouldReplace: false,
            comparisonWasPerformed: true,
            originalScore: -3,
            boostedVocabScore: -7,
            rawVocabularyCTCScore: -8,
            rawOriginalCTCScore: -3,
            effectiveBoost: 1,
            replacement: replacement,
            reason: "vocabulary <= original"
        )
    }

    private func unavailableResult(replacement: String) -> VocabularyRescorer.CTCMatchResult {
        VocabularyRescorer.CTCMatchResult(
            shouldReplace: false,
            comparisonWasPerformed: false,
            originalScore: -.infinity,
            boostedVocabScore: -.infinity,
            rawVocabularyCTCScore: nil,
            rawOriginalCTCScore: nil,
            effectiveBoost: nil,
            replacement: replacement,
            reason: "No tokenizer available"
        )
    }
}
