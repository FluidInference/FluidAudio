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

        let probe = try processor.trailingTailProbe(
            lastTokenTimestamp: 484,
            lastTokenDuration: 1,
            minTailSeconds: 1.5,
            maxModelSamples: maxModelSamples
        )

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

    func testSilentTailDoesNotTriggerProbe() throws {
        let total = 720_000
        // Entirely silent audio: the tail carries no speech energy.
        let processor = ChunkProcessor(audioSamples: audio(total: total))

        let probe = try processor.trailingTailProbe(
            lastTokenTimestamp: 484,
            lastTokenDuration: 1,
            minTailSeconds: 1.5,
            maxModelSamples: maxModelSamples
        )
        XCTAssertNil(probe)
    }

    func testShortTailBelowMinimumDoesNotTriggerProbe() throws {
        let total = 720_000
        // Last token ends only ~1.9k samples before the end — a shorter tail
        // than `minTailSeconds`, even though it is loud.
        let lastTimestamp = 560
        let tailStartSample = (lastTimestamp + 1) * frameSamples
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: tailStartSample..<total))

        let probe = try processor.trailingTailProbe(
            lastTokenTimestamp: lastTimestamp,
            lastTokenDuration: 1,
            minTailSeconds: 1.5,
            maxModelSamples: maxModelSamples
        )
        XCTAssertNil(probe)
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

        let probe = try processor.trailingTailProbe(
            lastTokenTimestamp: 484,
            lastTokenDuration: 1,
            minTailSeconds: 1.5,
            maxModelSamples: maxModelSamples
        )
        XCTAssertNil(probe)
    }

    func testLongerLastTokenDurationMovesTheTailBoundary() throws {
        // The drop boundary is the token's decoded END (timestamp + duration),
        // not its start.
        let total = 720_000
        let processor = ChunkProcessor(audioSamples: audio(total: total, loud: 0..<total))

        let probe = try XCTUnwrap(
            try processor.trailingTailProbe(
                lastTokenTimestamp: 480,
                lastTokenDuration: 5,
                minTailSeconds: 1.5,
                maxModelSamples: maxModelSamples
            )
        )
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
        // Merged stream ends "…▁science" at frame 484. The tail probe re-hears
        // "▁science" at 485 (an echo of the last word) then the dropped words.
        let total = 720_000
        let lastAudioFrame = (total - 1) / frameSamples
        let last: ChunkProcessor.TokenWindow = (token: 7, timestamp: 484, confidence: 0.9, duration: 1)

        let candidate = ChunkProcessor.spliceCandidate(
            windowTokens: [7, 5, 7, 10],
            windowTimestamps: [485, 500, 520, 540],
            windowConfidences: [0.9, 0.9, 0.9, 0.9],
            windowDurations: [1, 1, 1, 1],
            gapStartFrame: last.timestamp,
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
            gapStartFrame: last.timestamp,
            gapEndFrame: lastAudioFrame + 2,
            leadNeighbor: last,
            tailNeighbor: last,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary),
            vocabulary: vocabulary
        )
        XCTAssertTrue(candidate.isEmpty)
    }
}
