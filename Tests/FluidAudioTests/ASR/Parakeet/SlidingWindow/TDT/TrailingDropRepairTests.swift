import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the trailing-drop repair pass (issue #747): the final
/// sliding window can decode to all-blank on quiet long-form audio, silently
/// dropping the last few words. These cover the pure pieces — the probe
/// decision (does the untranscribed tail look like a dropped span, and where
/// should the re-decode window sit) and the tail splice — without the CoreML
/// models, mirroring how `SeamGapRepairTests` covers issue #758.
final class TrailingDropRepairTests: XCTestCase {

    private let frameSamples = ASRConstants.samplesPerEncoderFrame  // 1280
    private let maxModelSamples = ASRConstants.maxModelSamples  // 240000

    /// windowSamples the probe advertises: one usable encoder window.
    private var windowSamples: Int {
        max(
            frameSamples,
            (maxModelSamples - ASRConstants.melHopSize) / frameSamples * frameSamples
        )
    }

    /// Probe with the processor's own adaptive threshold, as production does.
    private func probeTail(
        _ processor: ChunkProcessor,
        timestamp: Int,
        duration: Int = 1
    ) throws -> ChunkProcessor.TrailingTailProbe? {
        try processor.trailingTailProbe(
            lastTokenTimestamp: timestamp,
            lastTokenDuration: duration,
            minTailSeconds: 1.5,
            maxModelSamples: maxModelSamples,
            speechRmsThreshold: processor.adaptiveSpeechRmsThresholdForTesting()
        )
    }

    /// Build `totalSamples`-long audio that is silent except for a
    /// speech-level tone across `[loudStart, loudEnd)`.
    private func audio(total: Int, loud: Range<Int>? = nil) -> [Float] {
        var samples = [Float](repeating: 0, count: total)
        if let loud {
            for i in loud where i >= 0 && i < total {
                samples[i] = 0.1 * sin(Float(i) * 0.1)
            }
        }
        return samples
    }

    // MARK: - Probe decision

    func testSpeechTailTriggersProbeWithColdStartAndEndAlignedPlacements() throws {
        // ~45s of audio; last token ends at frame 484 with a loud ~6s tail —
        // the issue's signature.
        let total = 720_000
        let tailStartFrame = 485  // 484 timestamp + 1-frame duration
        let tailStartSample = tailStartFrame * frameSamples
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: tailStartSample..<total))

        let probe = try probeTail(processor, timestamp: 484)

        let unwrapped = try XCTUnwrap(probe)
        XCTAssertEqual(unwrapped.tailStartFrame, tailStartFrame)
        XCTAssertEqual(unwrapped.windowSamples, windowSamples)
        // Cold-start at the drop first, then the end-aligned full window.
        let endAligned = max(0, total - windowSamples) / frameSamples * frameSamples
        XCTAssertEqual(unwrapped.placements, [tailStartSample, endAligned])
        // Placement 1 cold-starts strictly later than the end-aligned window,
        // so the re-decode sees the tail without the pre-gap history.
        XCTAssertGreaterThan(unwrapped.placements[0], unwrapped.placements[1])
    }

    func testQuietSpeechTailTriggersProbe() throws {
        // Regression for the issue #747 reproducer: the recording peaks below
        // 2% FS, so tail-speech RMS (≈ 0.001–0.003) never crosses an absolute
        // 0.008 gate — the very audio class whose final window blanks out. The
        // adaptive threshold must scale down and still fire the probe.
        let total = 720_000
        let tailStartFrame = 485
        let tailStartSample = tailStartFrame * frameSamples
        var samples = [Float](repeating: 0, count: total)
        for i in 0..<tailStartSample {
            samples[i] = 0.005 * sin(Float(i) * 0.1)  // quiet transcribed speech
        }
        for i in tailStartSample..<total {
            samples[i] = 0.003 * sin(Float(i) * 0.1)  // even quieter dropped tail
        }
        let processor = ChunkProcessor(audioSamples: samples)

        XCTAssertNotNil(try probeTail(processor, timestamp: 484))
    }

    func testSilentTailDoesNotTriggerProbe() throws {
        let total = 720_000
        // Entirely silent audio: the tail carries no speech energy.
        let processor = ChunkProcessor(audioSamples: audio(total: total))

        XCTAssertNil(try probeTail(processor, timestamp: 484))
    }

    func testShortTailBelowMinimumDoesNotTriggerProbe() throws {
        let total = 720_000
        // Last token ends only ~1.9k samples before the end — a shorter tail
        // than `minTailSeconds`, even though it is loud.
        let lastTimestamp = 560
        let tailStartSample = (lastTimestamp + 1) * frameSamples
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: tailStartSample..<total))

        XCTAssertNil(try probeTail(processor, timestamp: lastTimestamp))
    }

    func testTailWithSubThresholdSpeechDoesNotTriggerProbe() throws {
        // Long-enough tail, but only ~0.3s of it is loud — below the 0.5s
        // cumulative-speech gate, so a pause with a stray sound is left alone.
        let total = 720_000
        let tailStartFrame = 485
        let tailStartSample = tailStartFrame * frameSamples
        let loudFrames = 3  // 0.24s < 0.5s
        let loud = tailStartSample..<(tailStartSample + loudFrames * frameSamples)
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: loud))

        XCTAssertNil(try probeTail(processor, timestamp: 484))
    }

    func testLongerLastTokenDurationMovesTheTailBoundary() throws {
        // The drop boundary is the token's decoded END (timestamp + duration),
        // not its start.
        let total = 720_000
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: 0..<total))

        let probe = try XCTUnwrap(try probeTail(processor, timestamp: 480, duration: 5))
        XCTAssertEqual(probe.tailStartFrame, 485)
        XCTAssertEqual(probe.placements[0], 485 * frameSamples)
    }

    // MARK: - Tail splice (reuses spliceCandidate as repairTrailingDrop does)

    private let vocabulary: [Int: String] = [
        5: "▁and",
        7: "▁science",
        10: "▁examples",
    ]

    func testTailSpliceKeepsWordsPastLastTokenAndDedupesRehearing() {
        // Merged stream ends "…▁science" at frame 484 (duration 1, so the
        // tail starts at 485). The tail probe re-hears "▁science" at 486 (an
        // echo of the last word) then the dropped words.
        let total = 720_000
        let lastAudioFrame = (total - 1) / frameSamples
        let last: ChunkProcessor.TokenWindow = (token: 7, timestamp: 484, confidence: 0.9, duration: 1)

        let candidate = ChunkProcessor.spliceCandidate(
            windowTokens: [7, 5, 7, 10],
            windowTimestamps: [486, 500, 520, 540],
            windowConfidences: [0.9, 0.9, 0.9, 0.9],
            windowDurations: [1, 1, 1, 1],
            gapStartFrame: last.timestamp + 1,
            gapEndFrame: lastAudioFrame + 2,
            leadNeighbor: last,
            tailNeighbor: last,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary),
            vocabulary: vocabulary
        )

        // The re-heard last word is dropped; the dropped span survives.
        XCTAssertEqual(candidate.map { $0.token }, [5, 7, 10])
        XCTAssertEqual(candidate.map { $0.timestamp }, [500, 520, 540])
    }

    // MARK: - Stale sentence-end trim

    /// Decoded-vocabulary form: word-initial pieces carry a " " marker (the
    /// raw SentencePiece "▁" form is covered separately below).
    private let trimVocabulary: [Int: String] = [
        2: " me",
        3: ".",
        5: " as",
        6: " Also",
    ]

    private func window(_ token: Int, _ timestamp: Int) -> ChunkProcessor.TokenWindow {
        (token: token, timestamp: timestamp, confidence: 0.9, duration: 1)
    }

    func testLowercaseContinuationTrimsHallucinatedPeriod() {
        // "…without me." + recovered "as long as…": the period was a
        // window-end artifact and must go.
        let trimmed = ChunkProcessor.trimmingStaleSentenceEnd(
            from: [window(2, 480), window(3, 482)],
            beforeSplicing: [window(5, 500)],
            vocabulary: trimVocabulary
        )
        XCTAssertEqual(trimmed.map { $0.token }, [2])
    }

    func testCapitalizedContinuationKeepsSentenceEnd() {
        // "…without me." + recovered "Also, …": a genuine sentence boundary.
        let trimmed = ChunkProcessor.trimmingStaleSentenceEnd(
            from: [window(2, 480), window(3, 482)],
            beforeSplicing: [window(6, 500)],
            vocabulary: trimVocabulary
        )
        XCTAssertEqual(trimmed.map { $0.token }, [2, 3])
    }

    func testNoTrailingPunctuationLeavesStreamUntouched() {
        let trimmed = ChunkProcessor.trimmingStaleSentenceEnd(
            from: [window(2, 480)],
            beforeSplicing: [window(5, 500)],
            vocabulary: trimVocabulary
        )
        XCTAssertEqual(trimmed.map { $0.token }, [2])
    }

    func testLowercaseContinuationTrimsWithSentencePieceMarker() {
        // Same trim with a raw "▁"-marked vocabulary.
        let trimmed = ChunkProcessor.trimmingStaleSentenceEnd(
            from: [window(2, 480), window(3, 482)],
            beforeSplicing: [window(5, 500)],
            vocabulary: [2: "▁me", 3: ".", 5: "▁as"]
        )
        XCTAssertEqual(trimmed.map { $0.token }, [2])
    }

    func testTailSpliceEmptyWhenProbeOnlyRehearsLastWord() {
        // Probe recovers nothing new (genuine end of speech): only an echo of
        // the last word, which dedupes away.
        let total = 720_000
        let lastAudioFrame = (total - 1) / frameSamples
        let last: ChunkProcessor.TokenWindow = (token: 7, timestamp: 484, confidence: 0.9, duration: 1)

        let candidate = ChunkProcessor.spliceCandidate(
            windowTokens: [7],
            windowTimestamps: [486],
            windowConfidences: [0.9],
            windowDurations: [1],
            gapStartFrame: last.timestamp + 1,
            gapEndFrame: lastAudioFrame + 2,
            leadNeighbor: last,
            tailNeighbor: last,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary),
            vocabulary: vocabulary
        )
        XCTAssertTrue(candidate.isEmpty)
    }
}
