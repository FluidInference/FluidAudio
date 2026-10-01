import XCTest

@testable import FluidAudio

/// `ctcAppliedTerms` says which vocabulary terms went in; `ctcReplacements`
/// says which decoded word each one displaced and on what scores. These tests
/// pin that the decisions survive to the public result on the batch
/// (`ASRResult`, `UnifiedAsrManager.transcribeDetailed`) and streaming
/// (`SlidingWindowTranscriptionUpdate`,
/// `StreamingUnifiedAsrManager.consumeVocabularyReplacements`) paths.
final class VocabularyRescoringSurfacingTests: XCTestCase {

    private typealias Decision = VocabularyRescorer.RescoringResult

    private let applied = Decision(
        originalWord: "codecs", originalScore: 0.21, replacementWord: "Codex", replacementScore: 0.88,
        shouldReplace: true, reason: "ctc-score")

    private let declined = Decision(
        originalWord: "favor", originalScore: 0.61, replacementWord: "flavor", replacementScore: 0.30,
        shouldReplace: false, reason: "below-floor")

    private func baseResult(text: String) -> ASRResult {
        ASRResult(text: text, confidence: 0.9, duration: 2.0, processingTime: 0.1)
    }

    // MARK: - Batch path

    func testWithRescoringCarriesReplacementDecisions() {
        let rescored = baseResult(text: "validate with codecs").withRescoring(
            text: "validate with Codex", detected: ["Codex"], applied: ["Codex"], replacements: [applied])

        XCTAssertEqual(rescored.text, "validate with Codex")
        XCTAssertEqual(rescored.ctcDetectedTerms, ["Codex"])
        XCTAssertEqual(rescored.ctcAppliedTerms, ["Codex"])
        XCTAssertEqual(rescored.ctcReplacements?.count, 1)
        XCTAssertEqual(rescored.ctcReplacements?.first?.originalWord, "codecs")
        XCTAssertEqual(rescored.ctcReplacements?.first?.replacementWord, "Codex")
        XCTAssertEqual(rescored.ctcReplacements?.first?.originalScore ?? 0, 0.21, accuracy: 1e-6)
        XCTAssertEqual(rescored.ctcReplacements?.first?.replacementScore ?? 0, 0.88, accuracy: 1e-6)
        XCTAssertTrue(rescored.ctcReplacements?.first?.shouldReplace ?? false)
        XCTAssertEqual(rescored.ctcReplacements?.first?.reason, "ctc-score")
    }

    /// The new parameter is defaulted, so the pre-existing three-argument call
    /// still compiles and leaves the decisions unset.
    func testWithRescoringWithoutReplacementsLeavesDecisionsNil() {
        let rescored = baseResult(text: "raw").withRescoring(text: "raw", detected: [], applied: nil)
        XCTAssertNil(rescored.ctcReplacements)
        XCTAssertEqual(rescored.ctcDetectedTerms, [])
    }

    /// `ASRResult` is `Codable`, so the decisions have to survive a round trip.
    func testReplacementDecisionsSurviveCodableRoundTrip() throws {
        let rescored = baseResult(text: "validate with Codex").withRescoring(
            text: "validate with Codex", detected: ["Codex"], applied: ["Codex"], replacements: [applied])

        let decoded = try JSONDecoder().decode(ASRResult.self, from: JSONEncoder().encode(rescored))

        XCTAssertEqual(decoded.ctcAppliedTerms, ["Codex"])
        XCTAssertEqual(decoded.ctcReplacements?.first?.originalWord, "codecs")
        XCTAssertEqual(decoded.ctcReplacements?.first?.replacementWord, "Codex")
        XCTAssertEqual(decoded.ctcReplacements?.first?.reason, "ctc-score")
    }

    // MARK: - Streaming path

    func testStreamingUpdateCarriesReplacementDecisions() {
        let rescored = baseResult(text: "validate with codecs").withRescoring(
            text: "validate with Codex", detected: ["Codex"], applied: ["Codex"], replacements: [applied])

        let update = SlidingWindowTranscriptionUpdate(
            text: rescored.text,
            isConfirmed: true,
            confidence: rescored.confidence,
            timestamp: Date(),
            ctcDetectedTerms: rescored.ctcDetectedTerms,
            ctcAppliedTerms: rescored.ctcAppliedTerms,
            ctcReplacements: rescored.ctcReplacements
        )

        XCTAssertEqual(update.ctcAppliedTerms, ["Codex"])
        XCTAssertEqual(update.ctcReplacements?.count, 1)
        XCTAssertEqual(update.ctcReplacements?.first?.originalWord, "codecs")
        XCTAssertEqual(update.ctcReplacements?.first?.replacementScore ?? 0, 0.88, accuracy: 1e-6)
    }

    func testStreamingUpdateWithoutBoostingLeavesDecisionsNil() {
        let update = SlidingWindowTranscriptionUpdate(
            text: "validate with codecs", isConfirmed: false, confidence: 0.5, timestamp: Date())
        XCTAssertNil(update.ctcReplacements)
        XCTAssertNil(update.ctcAppliedTerms)
    }

    // MARK: - UnifiedAsrManager.transcribeDetailed mapping

    /// `transcribeDetailed(_:)` folds the rescorer output in through
    /// `UnifiedAsrManager.applying(_:to:)`; only accepted decisions are
    /// reported, and they stay aligned with `ctcAppliedTerms`.
    func testUnifiedApplyingReportsAcceptedDecisionsOnly() {
        let output = VocabularyRescorer.RescoreOutput(
            text: "validate with Codex, in your favor",
            replacements: [applied, declined],
            wasModified: true,
            detectedTerms: ["Codex", "flavor"])

        let result = UnifiedAsrManager.applying(output, to: baseResult(text: "validate with codecs, in your favor"))

        XCTAssertEqual(result.text, "validate with Codex, in your favor")
        XCTAssertEqual(result.ctcDetectedTerms, ["Codex", "flavor"])
        XCTAssertEqual(result.ctcAppliedTerms, ["Codex"])
        XCTAssertEqual(result.ctcReplacements?.count, 1)
        XCTAssertEqual(result.ctcReplacements?.first?.originalWord, "codecs")
        XCTAssertEqual(result.ctcReplacements?.first?.reason, "ctc-score")
    }

    /// Rescoring ran but replaced nothing: the detections are still reported,
    /// and the decision list is `nil` rather than empty, as `ctcAppliedTerms` is.
    func testUnifiedApplyingWithNoAcceptedDecisions() {
        let output = VocabularyRescorer.RescoreOutput(
            text: "in your favor", replacements: [declined], wasModified: false, detectedTerms: ["flavor"])

        let result = UnifiedAsrManager.applying(output, to: baseResult(text: "in your favor"))

        XCTAssertEqual(result.ctcDetectedTerms, ["flavor"])
        XCTAssertNil(result.ctcAppliedTerms)
        XCTAssertNil(result.ctcReplacements)
    }

    /// The result `transcribeDetailed(_:)` builds carries a confidence, so the
    /// mean follows `AsrManager`'s rule rather than a fresh one.
    func testMeanConfidenceFollowsAsrManagerRule() {
        let emissions: [ChunkProcessor.TokenWindow] = [
            (token: 1, timestamp: 0, confidence: 0.8, duration: 1),
            (token: 2, timestamp: 2, confidence: 0.6, duration: 1),
        ]
        XCTAssertEqual(UnifiedAsrManager.meanConfidence(of: emissions, isEmpty: false), 0.7, accuracy: 1e-6)
        XCTAssertEqual(UnifiedAsrManager.meanConfidence(of: emissions, isEmpty: true), 0.1, accuracy: 1e-6)
        XCTAssertEqual(UnifiedAsrManager.meanConfidence(of: [], isEmpty: false), 0.5, accuracy: 1e-6)
        XCTAssertEqual(
            UnifiedAsrManager.meanConfidence(
                of: [(token: 1, timestamp: 0, confidence: 0.01, duration: 1)], isEmpty: false),
            0.1, accuracy: 1e-6)
    }

    // MARK: - StreamingUnifiedAsrManager drain

    /// Without boosting configured nothing is ever accumulated, so the drain is
    /// empty and stays empty.
    func testStreamingUnifiedDrainIsEmptyWithoutBoosting() async {
        let manager = StreamingUnifiedAsrManager()
        let first = await manager.consumeVocabularyReplacements()
        let second = await manager.consumeVocabularyReplacements()
        XCTAssertTrue(first.isEmpty)
        XCTAssertTrue(second.isEmpty)
    }
}
