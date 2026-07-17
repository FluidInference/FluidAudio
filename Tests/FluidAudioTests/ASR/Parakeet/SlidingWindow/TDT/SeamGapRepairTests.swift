import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the seam-gap repair pass (issue #758): the pure splice
/// logic that decides which probe-decoded tokens may be inserted into a
/// detected gap. The cases mirror artifacts observed on real conference
/// audio during validation.
final class SeamGapRepairTests: XCTestCase {

    // Small SentencePiece-style vocabulary. "▁" marks word-initial pieces.
    private let vocabulary: [Int: String] = [
        1: "▁for",
        2: "▁For",
        3: "▁else",
        4: ".",
        5: "▁and",
        6: "ing",
        7: "▁science",
        8: "▁I",
        9: "▁So",
        10: "▁examples",
    ]

    private var safeIds: Set<Int>? {
        ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary)
    }

    private func token(_ id: Int, _ timestamp: Int) -> ChunkProcessor.TokenWindow {
        (token: id, timestamp: timestamp, confidence: 0.9, duration: 1)
    }

    private func splice(
        tokens: [(Int, Int)],
        gap: (Int, Int),
        lead: ChunkProcessor.TokenWindow,
        tail: ChunkProcessor.TokenWindow,
        vocabulary: [Int: String]? = nil
    ) -> [ChunkProcessor.TokenWindow] {
        let vocab = vocabulary ?? self.vocabulary
        return ChunkProcessor.spliceCandidate(
            windowTokens: tokens.map { $0.0 },
            windowTimestamps: tokens.map { $0.1 },
            windowConfidences: tokens.map { _ in 0.9 },
            windowDurations: tokens.map { _ in 1 },
            gapStartFrame: gap.0,
            gapEndFrame: gap.1,
            leadNeighbor: lead,
            tailNeighbor: tail,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(vocabulary: vocab),
            vocabulary: vocab
        )
    }

    // MARK: - Config

    func testSeamGapRepairDefaults() {
        XCTAssertTrue(ASRConfig.default.seamGapRepair)
        XCTAssertEqual(ASRConfig.default.seamGapRepairMinGapSeconds, 1.5)
    }

    func testMinGapSecondsClampedToHalfSecond() {
        let config = ASRConfig(seamGapRepairMinGapSeconds: 0.1)
        XCTAssertEqual(config.seamGapRepairMinGapSeconds, 0.5)
    }

    // MARK: - In-gap filtering

    func testOnlyTokensStrictlyInsideGapAreKept() {
        // Gap frames 100–200: tokens at the exact bounds and the one-frame
        // margin at the tail must be excluded.
        let result = splice(
            tokens: [(5, 100), (5, 101), (5, 150), (5, 198), (5, 199), (5, 200)],
            gap: (100, 200),
            lead: token(10, 95),
            tail: token(10, 205)
        )
        XCTAssertEqual(result.map { $0.timestamp }, [101, 150, 198])
    }

    func testGenuineSilenceRecoversNothing() {
        // Probe re-decoded the window but every token lies outside the gap.
        let result = splice(
            tokens: [(5, 80), (7, 90), (5, 210)],
            gap: (100, 200),
            lead: token(10, 95),
            tail: token(10, 205)
        )
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Word-boundary hygiene (#683 rule)

    func testLeadingContinuationPiecesAreTrimmed() {
        // Candidate starts mid-word ("ing" continuation piece): trim to the
        // first word-initial piece.
        let result = splice(
            tokens: [(6, 110), (5, 120), (7, 130)],
            gap: (100, 200),
            lead: token(10, 95),
            tail: token(10, 205)
        )
        XCTAssertEqual(result.map { $0.token }, [5, 7])
    }

    // MARK: - Edge dedupe (observed artifacts)

    func testCaseInsensitiveLeadDedupe() {
        // Observed as "for For sharing": the probe re-hears the boundary
        // word with different capitalisation (different token id).
        let result = splice(
            tokens: [(2, 104), (10, 120)],
            gap: (100, 200),
            lead: token(1, 100),
            tail: token(10, 205)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testTailDedupeSameTokenId() {
        // Observed as "So So": probe re-hears the word that starts right
        // after the gap.
        let result = splice(
            tokens: [(10, 120), (9, 196)],
            gap: (100, 200),
            lead: token(10, 95),
            tail: token(9, 201)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testPunctuationWalkResolvesWordNeighbor() {
        // Observed as "else else.": the merged stream ends "…▁else ." — the
        // dedupe must compare against ▁else, not the "." piece.
        let stream = [token(5, 90), token(3, 98), token(4, 99)]
        let lead = ChunkProcessor.wordNeighbor(in: stream, from: 2, step: -1, vocabulary: vocabulary)
        XCTAssertEqual(lead.token, 3)

        let result = splice(
            tokens: [(3, 102), (10, 130)],
            gap: (100, 200),
            lead: lead,
            tail: token(10, 205)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testDedupeExposingOrphanPunctuationEmptiesCandidate() {
        // If the only recovered word is a re-hearing of the boundary word,
        // removing it must not leave its trailing punctuation behind.
        let result = splice(
            tokens: [(1, 104), (4, 106)],
            gap: (100, 200),
            lead: token(1, 100),
            tail: token(10, 205)
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testGenuineStutterBeyondToleranceIsKept() {
        // Observed as genuine "I I": separations beyond 6 frames are real
        // re-spoken words, not probe echoes, and must survive.
        let result = splice(
            tokens: [(8, 108), (10, 130)],
            gap: (100, 200),
            lead: token(8, 100),
            tail: token(10, 205)
        )
        XCTAssertEqual(result.map { $0.token }, [8, 10])
    }

    func testEmptyVocabularyStillAppliesIdEqualDedupe() {
        // Without a vocabulary there are no safe ids and no piece text, but
        // identical token ids at the edge still dedupe.
        let result = splice(
            tokens: [(1, 104), (10, 130)],
            gap: (100, 200),
            lead: token(1, 100),
            tail: token(10, 205),
            vocabulary: [:]
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    // MARK: - Speech-energy gate

    func testSpeechLikeSecondsOnSilenceIsZero() throws {
        let silence = [Float](repeating: 0.0, count: 16000 * 4)
        let processor = ChunkProcessor(audioSamples: silence)
        let seconds = try processor.speechLikeSecondsForTesting(from: 0, to: silence.count)
        XCTAssertEqual(seconds, 0.0)
    }

    func testSpeechLikeSecondsCountsLoudRegion() throws {
        // 4s of silence with 1s of speech-level tone in the middle.
        var samples = [Float](repeating: 0.0, count: 16000 * 4)
        for i in 16000..<32000 {
            samples[i] = 0.1 * sin(Float(i) * 0.1)
        }
        let processor = ChunkProcessor(audioSamples: samples)
        let seconds = try processor.speechLikeSecondsForTesting(from: 0, to: samples.count)
        XCTAssertEqual(seconds, 1.0, accuracy: 0.1)
    }

    func testSpeechLikeSecondsIgnoresRoomToneInNormalLevelRecording() throws {
        // 3s of normal-level speech followed by 2s of room tone: the adaptive
        // threshold anchors on the recording's speech level, so the room-tone
        // span must not count.
        var samples = [Float](repeating: 0.0, count: 16000 * 5)
        for i in 0..<(16000 * 3) {
            samples[i] = 0.1 * sin(Float(i) * 0.1)
        }
        for i in (16000 * 3)..<samples.count {
            samples[i] = 0.004 * sin(Float(i) * 0.1)
        }
        let processor = ChunkProcessor(audioSamples: samples)
        let seconds = try processor.speechLikeSecondsForTesting(from: 16000 * 3, to: samples.count)
        XCTAssertEqual(seconds, 0.0)
    }

    func testSpeechLikeSecondsCountsQuietSpeechInQuietRecording() throws {
        // Issue #747 reproducer shape: the whole recording peaks below 2% FS,
        // so speech-level frames sit far below any absolute gate tuned for
        // normal levels. The adaptive threshold scales down and still counts
        // the (relatively) loud region.
        var samples = [Float](repeating: 0.0, count: 16000 * 4)
        for i in 16000..<32000 {
            samples[i] = 0.005 * sin(Float(i) * 0.1)  // RMS ≈ 0.0035, below the 0.008 ceiling
        }
        let processor = ChunkProcessor(audioSamples: samples)
        let seconds = try processor.speechLikeSecondsForTesting(from: 0, to: samples.count)
        XCTAssertEqual(seconds, 1.0, accuracy: 0.1)
    }

    // MARK: - Adaptive speech threshold

    func testAdaptiveThresholdClampsToCeilingOnNormalLevelAudio() {
        XCTAssertEqual(
            ChunkProcessor.adaptiveSpeechRmsThreshold(referenceRms: 0.07, floor: 0.0005, ceiling: 0.008),
            0.008)
    }

    func testAdaptiveThresholdScalesWithQuietRecordingLevel() {
        XCTAssertEqual(
            ChunkProcessor.adaptiveSpeechRmsThreshold(referenceRms: 0.004, floor: 0.0005, ceiling: 0.008),
            0.0012, accuracy: 0.0001)
    }

    func testAdaptiveThresholdClampsToFloorOnSilence() {
        XCTAssertEqual(
            ChunkProcessor.adaptiveSpeechRmsThreshold(referenceRms: 0.0, floor: 0.0005, ceiling: 0.008),
            0.0005)
    }

    func testDigitalSilenceFramesAreExcludedFromTheReference() throws {
        // 80% exact zeros + 20% quiet tone: counting the zeros would put the
        // percentile on digital silence and collapse the gate to its floor.
        // Excluded, the reference lands on the tone's own level.
        var samples = [Float](repeating: 0.0, count: 16000 * 4)
        for i in 0..<(16000 * 4 / 5) {
            samples[i] = 0.004 * sin(Float(i) * 0.1)  // RMS ≈ 0.0028
        }
        let processor = ChunkProcessor(audioSamples: samples)
        let threshold = try processor.adaptiveSpeechRmsThresholdForTesting()
        XCTAssertGreaterThan(threshold, ChunkProcessor.speechRmsFloor)
        XCTAssertEqual(threshold, 0.0028 * 0.3, accuracy: 0.0002)
    }

    func testAllDigitalSilenceFileFallsBackToCeiling() throws {
        let processor = ChunkProcessor(audioSamples: [Float](repeating: 0.0, count: 16000 * 2))
        let threshold = try processor.adaptiveSpeechRmsThresholdForTesting()
        XCTAssertEqual(threshold, ChunkProcessor.speechRmsCeiling)
    }
}
