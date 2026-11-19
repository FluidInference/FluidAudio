import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the seam-gap repair pass (issue #758): the pure splice
/// logic that decides which probe-decoded tokens may be inserted into a
/// detected gap. The cases mirror artifacts observed on real conference
/// and oral-history audio during validation.
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
        11: "▁Dicev",
        12: "ano",
        13: "▁già",
        14: "▁l",
        15: "'",
        16: "in",
        17: "fanzia",
        18: "▁Meu",
        19: "▁è",
        20: "▁E",
        21: "▁nas",
        22: "ce",
        23: "amo",
    ]

    private var safeIds: Set<Int>? {
        ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary)
    }

    private func token(_ id: Int, _ timestamp: Int) -> ChunkProcessor.TokenWindow {
        (token: id, timestamp: timestamp, confidence: 0.9, duration: 1)
    }

    /// A bordering word as `neighborWord` would describe it.
    private func word(
        _ ids: [Int], _ timestamp: Int, vocabulary: [Int: String]? = nil
    ) -> (text: String, timestamp: Int) {
        (ChunkProcessor.comparableWord(ids, vocabulary: vocabulary ?? self.vocabulary), timestamp)
    }

    private func splice(
        tokens: [(Int, Int)],
        gap: (Int, Int),
        lead: (text: String, timestamp: Int),
        tail: (text: String, timestamp: Int),
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
            leadWord: lead,
            tailWord: tail,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(vocabulary: vocab),
            vocabulary: vocab
        )
    }

    // MARK: - Config

    func testSeamGapRepairDefaults() {
        XCTAssertTrue(ASRConfig.default.seamGapRepair)
        XCTAssertEqual(ASRConfig.default.seamGapRepairMinGapSeconds, 0.5)
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
            lead: word([10], 95),
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.timestamp }, [101, 150, 198])
    }

    func testGenuineSilenceRecoversNothing() {
        // Probe re-decoded the window but every token lies outside the gap.
        let result = splice(
            tokens: [(5, 80), (7, 90), (5, 210)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 205)
        )
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Word-boundary hygiene (#683 rule)

    func testLeadingGapKeepsSpeechAtFrameZero() {
        let result = ChunkProcessor.spliceCandidate(
            windowTokens: [8, 5, 7],
            windowTimestamps: [0, 10, 30],
            windowConfidences: [0.9, 0.9, 0.9],
            windowDurations: [1, 1, 1],
            gapStartFrame: 0,
            gapEndFrame: 50,
            leadWord: nil,
            tailWord: word([10], 50),
            spliceSafeTokenIds: safeIds,
            vocabulary: vocabulary
        )
        XCTAssertEqual(result.map { $0.timestamp }, [0, 10, 30])
        XCTAssertEqual(result.map { $0.token }, [8, 5, 7])
    }

    func testLeadingGapStillDeduplicatesTheSurvivingTailWord() {
        let result = ChunkProcessor.spliceCandidate(
            windowTokens: [8, 10],
            windowTimestamps: [0, 46],
            windowConfidences: [0.9, 0.9],
            windowDurations: [1, 1],
            gapStartFrame: 0,
            gapEndFrame: 50,
            leadWord: nil,
            tailWord: word([10], 50),
            spliceSafeTokenIds: safeIds,
            vocabulary: vocabulary
        )
        XCTAssertEqual(result.map { $0.token }, [8])
    }

    func testLeadingGapRejectsOutsideTokensAndPartialFirstWord() {
        let result = ChunkProcessor.spliceCandidate(
            windowTokens: [6, 5, 7, 8],
            windowTimestamps: [0, 10, 49, 50],
            windowConfidences: [0.9, 0.9, 0.9, 0.9],
            windowDurations: [1, 1, 1, 1],
            gapStartFrame: 0,
            gapEndFrame: 50,
            leadWord: nil,
            tailWord: word([10], 50),
            spliceSafeTokenIds: safeIds,
            vocabulary: vocabulary
        )
        XCTAssertEqual(result.map { $0.token }, [5])
    }

    func testLeadingContinuationPiecesAreTrimmed() {
        // Candidate starts mid-word ("ing" continuation piece): trim to the
        // first word-initial piece.
        let result = splice(
            tokens: [(6, 110), (5, 120), (7, 130)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.token }, [5, 7])
    }

    func testWordCutByGapEndIsCompleted() {
        // Observed as "Dicev già": the gap end fell between "▁Dicev" and
        // "ano". The continuation piece is pulled in and clamped inside the
        // gap so it still sorts before the token after it.
        let result = splice(
            tokens: [(11, 190), (12, 199), (13, 203)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([13], 203)
        )
        XCTAssertEqual(result.map { $0.token }, [11, 12])
        XCTAssertEqual(result.map { $0.timestamp }, [190, 199])
    }

    func testCompletionRunningPastGapEndDropsTheWord() {
        // Observed as "nasceamo parliamo": the probe heard the tail word's
        // audio as a different word. A completion that reaches past the gap
        // end is that disagreement, not a recovery.
        let result = splice(
            tokens: [(21, 190), (22, 195), (23, 205)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 200)
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testApostropheInsideWordIsPartOfTheWord() {
        // Observed as "l l'infanzia": "▁l ' in fanzia" is one word. The
        // apostrophe between a word start and a continuation piece belongs
        // to the word, so the completion carries on through it and the
        // whole word dedupes against the tail neighbour.
        let result = splice(
            tokens: [(14, 190), (15, 191), (16, 192), (17, 195)],
            gap: (100, 192),
            lead: word([10], 95),
            tail: word([14, 15, 16, 17], 191)
        )
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Edge dedupe (observed artifacts)

    func testCaseInsensitiveLeadDedupe() {
        // Observed as "for For sharing": the probe re-hears the boundary
        // word with different capitalisation (different token id).
        let result = splice(
            tokens: [(2, 104), (10, 120)],
            gap: (100, 200),
            lead: word([1], 100),
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testTailDedupeSameTokenId() {
        // Observed as "So So": probe re-hears the word that starts right
        // after the gap.
        let result = splice(
            tokens: [(10, 120), (9, 196)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([9], 201)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testDiacriticInsensitiveDedupe() {
        // Observed as "è E' morto": the merged stream wrote the verb as
        // "E'" and the probe as "è" — the same word.
        let result = splice(
            tokens: [(19, 104)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([20, 15], 108)
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testDifferentWordAtTheSameFramesIsTheNeighbor() {
        // Observed as "Meu Mio padre": the probe heard the tail word's audio
        // as another word. Anything starting within two frames of a
        // neighbour is that neighbour, whatever the text.
        let result = splice(
            tokens: [(18, 102)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 103)
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testDifferentWordClearOfTheNeighborIsKept() {
        let result = splice(
            tokens: [(18, 102)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 107)
        )
        XCTAssertEqual(result.map { $0.token }, [18])
    }

    func testNeighborWordSpansEveryPieceOfTheWord() {
        // The dedupe compares whole words: "▁l ' in fanzia" is "linfanzia",
        // timed at its first piece, however the stream's pieces fall.
        let stream = [token(10, 90), token(14, 95), token(15, 96), token(16, 97), token(17, 98)]
        let lead = ChunkProcessor.neighborWord(
            in: stream, from: 4, step: -1, spliceSafeTokenIds: safeIds, vocabulary: vocabulary)
        XCTAssertEqual(lead.text, "linfanzia")
        XCTAssertEqual(lead.timestamp, 95)
    }

    func testPunctuationWalkResolvesWordNeighbor() {
        // Observed as "else else.": the merged stream ends "…▁else ." — the
        // dedupe must compare against ▁else, not the "." piece.
        let stream = [token(5, 90), token(3, 98), token(4, 99)]
        let lead = ChunkProcessor.neighborWord(
            in: stream, from: 2, step: -1, spliceSafeTokenIds: safeIds, vocabulary: vocabulary)
        XCTAssertEqual(lead.text, "else")
        XCTAssertEqual(lead.timestamp, 98)

        let result = splice(
            tokens: [(3, 102), (10, 130)],
            gap: (100, 200),
            lead: lead,
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    func testDedupeExposingOrphanPunctuationEmptiesCandidate() {
        // If the only recovered word is a re-hearing of the boundary word,
        // removing it must not leave its trailing punctuation behind.
        let result = splice(
            tokens: [(1, 104), (4, 106)],
            gap: (100, 200),
            lead: word([1], 100),
            tail: word([10], 205)
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testRepeatedTrailingPunctuationCollapses() {
        // Observed as "grazie..": the probe emitted two full stops.
        let result = splice(
            tokens: [(7, 120), (4, 130), (4, 131)],
            gap: (100, 200),
            lead: word([10], 95),
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.token }, [7, 4])
    }

    func testGenuineStutterBeyondToleranceIsKept() {
        // Observed as genuine "I I": separations beyond 6 frames are real
        // re-spoken words, not probe echoes, and must survive.
        let result = splice(
            tokens: [(8, 108), (10, 130)],
            gap: (100, 200),
            lead: word([8], 100),
            tail: word([10], 205)
        )
        XCTAssertEqual(result.map { $0.token }, [8, 10])
    }

    func testEmptyVocabularyStillAppliesIdEqualDedupe() {
        // Without a vocabulary there are no safe ids and no piece text, but
        // identical token ids at the edge still dedupe.
        let result = splice(
            tokens: [(1, 104), (10, 130)],
            gap: (100, 200),
            lead: word([1], 100, vocabulary: [:]),
            tail: word([10], 205, vocabulary: [:]),
            vocabulary: [:]
        )
        XCTAssertEqual(result.map { $0.token }, [10])
    }

    // MARK: - Confidence gate

    func testMedianConfidenceGateTakesOneWeakFunctionWord() {
        // "Are you ready? Just a little bit." decoded with "a" at 0.38: the
        // run is confident as a whole and must be accepted.
        let candidate: [ChunkProcessor.TokenWindow] = [0.63, 1.0, 1.0, 0.97, 0.69, 0.38, 0.94, 1.0].enumerated().map {
            (token: 5, timestamp: $0.offset, confidence: $0.element, duration: 1)
        }
        XCTAssertTrue(ChunkProcessor.isConfidentSplice(candidate))
    }

    func testMedianConfidenceGateRejectsNoiseHallucination() {
        let candidate: [ChunkProcessor.TokenWindow] = [0.12, 0.31, 0.5, 0.9].enumerated().map {
            (token: 5, timestamp: $0.offset, confidence: $0.element, duration: 1)
        }
        XCTAssertFalse(ChunkProcessor.isConfidentSplice(candidate))
        XCTAssertFalse(ChunkProcessor.isConfidentSplice([]))
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

    // MARK: - Placement order

    /// A multi-second gap (the #758 seam-drop class) is probed cold-start at the gap first,
    /// then gap-centred, and only then with the short window. Tried first, the short window
    /// recovered only the back half of a 13 s Earnings-22 gap, and the residual front was
    /// never re-probed.
    func testLongGapTriesFullWindowsBeforeTheShortWindow() {
        let rate = 16_000
        let pad = rate * 4 / 5
        let placements = ChunkProcessor.seamGapProbePlacements(
            gapStartSample: 50 * rate, gapEndSample: 63 * rate, isLongGap: true,
            windowSamples: 15 * rate, padSamples: pad)
        XCTAssertEqual(
            placements.map { $0.start },
            [50 * rate, (50 * rate + 63 * rate) / 2 - 15 * rate / 2, 50 * rate - pad])
        XCTAssertEqual(placements.map { $0.length }, [15 * rate, 15 * rate, 13 * rate + 2 * pad])
    }

    func testShortGapTriesOnlyTheShortWindow() {
        let rate = 16_000
        let pad = rate * 4 / 5
        let placements = ChunkProcessor.seamGapProbePlacements(
            gapStartSample: 10 * rate, gapEndSample: 10 * rate + rate / 2, isLongGap: false,
            windowSamples: 15 * rate, padSamples: pad)
        XCTAssertEqual(placements.map { $0.start }, [10 * rate - pad])
        XCTAssertEqual(placements.map { $0.length }, [rate / 2 + 2 * pad])
    }
}
