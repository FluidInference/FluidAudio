import Foundation
import XCTest

@testable import FluidAudio

/// Verifies the streaming segment-merge logic in `DiarizerTimeline.updateSegments`
/// matches an offline pipeline that:
///   1. removes raw segments shorter than `minFramesOn`, then
///   2. closes gaps `<= minFramesOff` between surviving padded segments.
final class DiarizerTimelineMergeTests: XCTestCase {

    /// numSpeakers=1, padOnset=padOffset=2, minFramesOn=4, minFramesOff=3.
    /// minSegmentLength = padOnset + padOffset + minFramesOn = 8.
    private static let mergeConfig = DiarizerTimelineConfig(
        numSpeakers: 1,
        frameDurationSeconds: 0.1,
        onsetThreshold: 0.5,
        offsetThreshold: 0.5,
        onsetPadFrames: 2,
        offsetPadFrames: 2,
        minFramesOn: 4,
        minFramesOff: 3
    )

    private func runFinalized(predictions: [Float]) throws -> [DiarizerSegment] {
        let timeline = DiarizerTimeline(config: Self.mergeConfig)
        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: predictions,
            finalizedFrameCount: predictions.count
        )
        _ = try timeline.addChunk(chunk)
        timeline.finalize()
        return timeline.speakers[0]?.finalizedSegments ?? []
    }

    /// Returns the tentative segments emitted from a single chunk.
    private func runTentative(predictions: [Float]) throws -> [DiarizerSegment] {
        let timeline = DiarizerTimeline(config: Self.mergeConfig)
        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: [],
            finalizedFrameCount: 0,
            tentativePredictions: predictions,
            tentativeFrameCount: predictions.count
        )
        let update = try timeline.addChunk(chunk)
        return update.tentativeSegments
    }

    /// Bug regression: A long, small gap, B too short.
    /// Offline behavior keeps A and drops B; the streaming code must do the same.
    func testShortSegmentAfterSmallGapDoesNotDropPriorSegment() throws {
        var predictions = [Float](repeating: 0.0, count: 30)
        // A: frames 5..14 active (raw 10 frames) → padded [3, 17), padded len 14 >= 8.
        for i in 5...14 { predictions[i] = 0.9 }
        // Gap: frames 15..18 silent.
        // B: frames 19..20 active (raw 2 frames) → padded len = 23 - 17 = 6 < 8.
        for i in 19...20 { predictions[i] = 0.9 }
        // Gap check between padded segments: B.start(17) - A.end(17) = 0 <= minFramesOff(3) → small gap.

        let segments = try runFinalized(predictions: predictions)

        XCTAssertEqual(segments.count, 1, "B is too short and should not drag A out of the timeline")
        XCTAssertEqual(segments.first?.startFrame, 3)
        XCTAssertEqual(segments.first?.endFrame, 17)
    }

    /// Two long segments separated by a small gap should merge into one segment.
    func testSmallGapMergesTwoLongSegments() throws {
        var predictions = [Float](repeating: 0.0, count: 40)
        // A: frames 5..14 active → padded [3, 17).
        for i in 5...14 { predictions[i] = 0.9 }
        // Gap: frames 15..18 silent.
        // B: frames 19..28 active → padded [17, 31). Padded len 14 >= 8.
        for i in 19...28 { predictions[i] = 0.9 }
        // Gap: 17 - 17 = 0 <= 3 → small gap → merge.

        let segments = try runFinalized(predictions: predictions)

        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments.first?.startFrame, 3)
        XCTAssertEqual(segments.first?.endFrame, 31)
    }

    /// Tentative emission: A long, small-gap merge intent, tail still too short at buffer end.
    /// Offline equivalent treats B as filtered → tentative should be A standalone, not the
    /// merged span (which would overstate the segment and dilute its activity).
    func testTrailingTentativeShortTailEmitsHeldSegmentAlone() throws {
        // Buffer ends mid-B, B too short so far.
        // A frames 5..14 (raw 10) → padded [3, 17).
        // Gap 15..18.
        // B frames 19..21 in progress (raw 3). Tail padded so far = (22+2) - (19-2) = 7 < 8.
        var predictions = [Float](repeating: 0.0, count: 22)
        for i in 5...14 { predictions[i] = 0.9 }
        for i in 19...21 { predictions[i] = 0.9 }

        let segments = try runTentative(predictions: predictions)

        XCTAssertEqual(segments.count, 1, "Tail not yet valid → held A is the only tentative segment")
        XCTAssertEqual(segments.first?.startFrame, 3)
        XCTAssertEqual(segments.first?.endFrame, 17)
        XCTAssertEqual(segments.first?.activity ?? 0, 0.9, accuracy: 1e-5)
    }

    /// Regression: a segment closing inside the chunk-end buffer zone must
    /// survive across a subsequent addChunk that wipes tentative storage.
    /// Before the fix, such segments were emitted as tentative + cleared from
    /// scratches by the finalized call, then wiped by the next chunk's
    /// clearTentative — silently dropped.
    func testSegmentInBufferZoneSurvivesNextChunk() throws {
        let timeline = DiarizerTimeline(config: Self.mergeConfig)

        // Chunk 1: A active frames 5..14, padded [3, 17]. Chunk ends at frame
        // 22; finalizedEndFrame = 22 - minFramesOff(3) - pad(4) = 15.
        // A.endFrame=17 > 15 → A is still mutable (a future-chunk small-gap
        // onset could extend it), so it must be held in scratches.
        var chunk1Predictions = [Float](repeating: 0.0, count: 22)
        for i in 5...14 { chunk1Predictions[i] = 0.9 }
        _ = try timeline.addChunk(
            DiarizerChunkResult(
                startFrame: 0,
                finalizedPredictions: chunk1Predictions,
                finalizedFrameCount: 22
            )
        )

        // Chunk 2: all silent. clearTentative would have dropped A under the
        // old behavior. With the fix, A is still held, ages past chunk 2's
        // finalizedEndFrame, and emits as finalized.
        _ = try timeline.addChunk(
            DiarizerChunkResult(
                startFrame: 22,
                finalizedPredictions: [Float](repeating: 0.0, count: 22),
                finalizedFrameCount: 22
            )
        )
        timeline.finalize()

        let segments = timeline.speakers[0]?.finalizedSegments ?? []
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments.first?.startFrame, 3)
        XCTAssertEqual(segments.first?.endFrame, 17)
    }

    /// Tentative emission: A long, small-gap merge intent, tail already long enough.
    /// Tentative should emit the merged span with combined activity.
    func testTrailingTentativeLongTailEmitsMergedSpan() throws {
        // A frames 5..14 → padded [3, 17). Gap 15..18.
        // B frames 19..28 in progress (raw 10). Tail padded so far = (29+2) - (19-2) = 14 >= 8.
        var predictions = [Float](repeating: 0.0, count: 29)
        for i in 5...14 { predictions[i] = 0.9 }
        for i in 19...28 { predictions[i] = 0.9 }

        let segments = try runTentative(predictions: predictions)

        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments.first?.startFrame, 3)
        XCTAssertEqual(segments.first?.endFrame, 31)
        XCTAssertEqual(segments.first?.activity ?? 0, 0.9, accuracy: 1e-5)
    }
}
