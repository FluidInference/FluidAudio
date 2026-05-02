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
}
