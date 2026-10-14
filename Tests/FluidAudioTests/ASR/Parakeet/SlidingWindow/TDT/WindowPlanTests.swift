import Foundation
import XCTest

@testable import FluidAudio

/// `ChunkProcessor.planWindows` is the single source of truth for which
/// windows `process()` dispatches and for the `windowCount` hosts see in
/// `WindowProgress`. These pin the plan against the cases where a count
/// derived from the planned starts alone drifts from the loop.
final class WindowPlanTests: XCTestCase {

    private let frameSamples = ASRConstants.samplesPerEncoderFrame
    private let sampleRate = ASRConstants.sampleRate

    private func seconds(_ value: Double) -> Int { Int(value * Double(sampleRate)) }

    /// Deterministic above-floor signal (RMS ≫ `speechRmsFloor`) so the
    /// speech-end rule treats it as speech. Layout-only; never decoded.
    private func aboveFloorSamples(count: Int) -> [Float] {
        var state: UInt32 = 0x9E37_79B9
        return (0..<count).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return (Float(state >> 8) / Float(1 << 24) - 0.5) * 0.2
        }
    }

    private func plan(
        chunkStarts: [Int],
        warmupStarts: Set<Int> = [],
        pauseCutStarts: Set<Int> = [],
        chunkSamples: Int,
        strideSamples: Int,
        warmupPrefixSamples: Int = 0,
        totalSamples: Int,
        speechEnd: Int? = nil,
        endAligned: Bool = true,
        pauseCutWindows: Bool = false
    ) -> [ChunkProcessor.PlannedWindow] {
        ChunkProcessor.planWindows(
            chunkStarts: chunkStarts.map {
                ChunkProcessor.ChunkStartDecision(
                    start: $0, useWarmupPrefix: warmupStarts.contains($0), pauseCut: pauseCutStarts.contains($0))
            },
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            warmupPrefixSamples: warmupPrefixSamples,
            melContextSamples: 0,
            totalSamples: totalSamples,
            speechEnd: speechEnd ?? totalSamples,
            endAligned: endAligned,
            pauseCutWindows: pauseCutWindows
        )
    }

    // MARK: - Plan shape

    func testRegularStartsEndAtTheFirstWindowReachingEOF() throws {
        let totalSamples = seconds(60)
        let processor = ChunkProcessor(audioSamples: aboveFloorSamples(count: totalSamples))
        let layout = processor.chunkLayoutForTesting(melChunkContext: true, modelVersion: .v3)
        let starts = try processor.chunkStartsForTesting(melChunkContext: true, modelVersion: .v3)
        let plan = try processor.plannedWindowsForTesting(melChunkContext: true, modelVersion: .v3)

        let firstFullWindow = try XCTUnwrap(
            starts.firstIndex { $0 + layout.chunkSamples >= totalSamples })
        XCTAssertEqual(plan.count, firstFullWindow + 1)
        XCTAssertLessThanOrEqual(plan.count, starts.count)
        XCTAssertEqual(plan.map(\.index), Array(0..<plan.count))
        XCTAssertEqual(plan.map(\.start), Array(starts.prefix(plan.count)))
        XCTAssertEqual(plan.filter(\.isLastChunk).count, 1)
        XCTAssertTrue(try XCTUnwrap(plan.last).isLastChunk)
        for window in plan.dropLast() {
            XCTAssertEqual(window.chunkEnd, window.start + layout.chunkSamples)
            XCTAssertEqual(window.audioEnd, window.chunkEnd)
        }
    }

    func testWindowFramesMatchTheDecodedSpan() {
        let chunkSamples = 187 * frameSamples
        let plan = plan(
            chunkStarts: [0, 161 * frameSamples],
            warmupStarts: [161 * frameSamples],
            chunkSamples: chunkSamples,
            strideSamples: 161 * frameSamples,
            warmupPrefixSamples: 7 * frameSamples,
            totalSamples: 400 * frameSamples,
            endAligned: false
        )
        XCTAssertEqual(plan[1].warmupSamples, 7 * frameSamples)
        XCTAssertEqual(plan[1].contextStart, 154 * frameSamples)
        XCTAssertEqual(plan[1].chunkStartOffset, 154 * frameSamples)
        XCTAssertEqual(plan[1].emitTokensAfterFrame, 161)
        XCTAssertEqual(plan[1].frames, ChunkProcessor.WindowFrames(decodeStart: 154, end: 154 + 187))
        XCTAssertNil(plan[0].emitTokensAfterFrame)
        XCTAssertEqual(plan[0].frames, ChunkProcessor.WindowFrames(decodeStart: 0, end: 187))
    }

    // MARK: - Cases where "count the starts before speechEnd" is wrong

    func testPostSpeechWindowsBeforeTheLastOneStillDecode() throws {
        // Speech for 20 s, then 40 s below the RMS floor. Only the window
        // that reaches EOF is cut back to the speech end (and dropped when
        // it starts past it); the interior post-speech windows still decode,
        // so the host must be told about them.
        let totalSamples = seconds(60)
        var samples = aboveFloorSamples(count: seconds(20))
        samples += [Float](repeating: 0, count: totalSamples - samples.count)
        let processor = ChunkProcessor(audioSamples: samples)
        let layout = processor.chunkLayoutForTesting(melChunkContext: true, modelVersion: .v3)
        let starts = try processor.chunkStartsForTesting(melChunkContext: true, modelVersion: .v3)
        let speechEnd = try processor.speechEndSamples()
        XCTAssertLessThan(speechEnd, seconds(21))

        let plan = try processor.plannedWindowsForTesting(melChunkContext: true, modelVersion: .v3)

        let interior = starts.filter { $0 + layout.chunkSamples < totalSamples }
        XCTAssertEqual(plan.map(\.start), interior)
        XCTAssertFalse(try XCTUnwrap(plan.last).isLastChunk, "the EOF window starts past speech and is dropped")
        let startsBeforeSpeechEnd = starts.filter { $0 < speechEnd }.count
        XCTAssertGreaterThan(plan.count, startsBeforeSpeechEnd)
    }

    func testStrideExtensionPastThePlannedStarts() {
        // Silence-aligned starts can land well before their target, so the
        // last planned start may not reach EOF. The loop keeps striding and
        // decodes one more end-aligned window; the plan must include it.
        let chunkSamples = 187 * frameSamples
        let strideSamples = 161 * frameSamples
        let totalSamples = 750 * frameSamples
        let starts = [0, 161, 322, 483].map { $0 * frameSamples }
        XCTAssertLessThan(starts.last! + chunkSamples, totalSamples)

        let plan = plan(
            chunkStarts: starts, chunkSamples: chunkSamples, strideSamples: strideSamples,
            totalSamples: totalSamples)

        XCTAssertEqual(plan.count, starts.count + 1)
        XCTAssertEqual(plan[4].start, 483 * frameSamples + strideSamples)
        XCTAssertTrue(plan[4].isLastChunk)
        XCTAssertEqual(plan[4].audioEnd, totalSamples)
        XCTAssertFalse(plan[3].isLastChunk)
    }

    func testWarmupPrefixCountsTheVisibleLengthNotTheFullWindow() {
        // A window whose full length reaches EOF but whose visible span
        // (after the warmup prefix) does not is not the last window.
        let chunkSamples = 187 * frameSamples
        let warmup = 7 * frameSamples
        let strideSamples = 161 * frameSamples
        let secondStart = 161 * frameSamples
        let totalSamples = secondStart + chunkSamples - warmup + frameSamples
        XCTAssertGreaterThanOrEqual(secondStart + chunkSamples, totalSamples)

        let plan = plan(
            chunkStarts: [0, secondStart],
            warmupStarts: [secondStart],
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            warmupPrefixSamples: warmup,
            totalSamples: totalSamples,
            endAligned: false
        )

        XCTAssertEqual(plan.count, 3)
        XCTAssertEqual(plan[1].warmupSamples, warmup)
        XCTAssertFalse(plan[1].isLastChunk)
        XCTAssertEqual(plan[2].start, secondStart + strideSamples)
        XCTAssertTrue(plan[2].isLastChunk)
    }

    func testFinalWindowStartingPastSpeechEndIsDropped() {
        let chunkSamples = 187 * frameSamples
        let plan = plan(
            chunkStarts: [0, 161 * frameSamples],
            chunkSamples: chunkSamples,
            strideSamples: 161 * frameSamples,
            totalSamples: 300 * frameSamples,
            speechEnd: 100 * frameSamples
        )
        XCTAssertEqual(plan.count, 1)
        XCTAssertEqual(plan[0].audioEnd, 187 * frameSamples)
    }

    func testPauseCutEndsTheWindowAtTheNextStart() {
        let chunkSamples = 187 * frameSamples
        let cut = 120 * frameSamples
        let plan = plan(
            chunkStarts: [0, cut],
            pauseCutStarts: [cut],
            chunkSamples: chunkSamples,
            strideSamples: 161 * frameSamples,
            totalSamples: 300 * frameSamples,
            pauseCutWindows: true
        )
        XCTAssertEqual(plan[0].audioEnd, cut)
        XCTAssertEqual(plan[0].chunkEnd, chunkSamples, "host progress still tracks the visible span")
        XCTAssertEqual(plan[0].frames.end, 120)
    }

    func testEmptyAudioHasNoWindows() {
        XCTAssertTrue(
            plan(chunkStarts: [], chunkSamples: 187 * frameSamples, strideSamples: 161 * frameSamples, totalSamples: 0)
                .isEmpty)
    }
}
