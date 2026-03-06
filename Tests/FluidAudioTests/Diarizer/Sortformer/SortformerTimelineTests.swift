import Foundation
import XCTest

@testable import FluidAudio

final class SortformerTimelineTests: XCTestCase {

    // MARK: - Empty Timeline

    func testEmptyTimelineHasZeroDuration() {
        let timeline = SortformerTimeline()
        XCTAssertEqual(timeline.numFrames, 0)
        XCTAssertEqual(timeline.duration, 0)
        XCTAssertTrue(timeline.framePredictions.isEmpty)
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
    }

    func testEmptyTimelineHasEmptySegments() {
        let timeline = SortformerTimeline()
        XCTAssertEqual(timeline.segments.count, 4, "Should have segment arrays for 4 speakers")
        for speakerSegments in timeline.segments {
            XCTAssertTrue(speakerSegments.isEmpty)
        }
    }

    // MARK: - Adding Chunks

    func testAddChunkUpdatesDuration() {
        let timeline = SortformerTimeline()
        let numSpeakers = 4
        let frameCount = 6

        let chunk = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
            frameCount: frameCount
        )

        timeline.addChunk(chunk)

        XCTAssertEqual(timeline.numFrames, 6)
        XCTAssertEqual(timeline.duration, Float(6) * 0.08, accuracy: 1e-5)
    }

    func testAddMultipleChunksAccumulatesFrames() {
        let timeline = SortformerTimeline()
        let numSpeakers = 4
        let frameCount = 6

        for i in 0..<3 {
            let chunk = SortformerChunkResult(
                startFrame: i * frameCount,
                speakerPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
                frameCount: frameCount
            )
            timeline.addChunk(chunk)
        }

        XCTAssertEqual(timeline.numFrames, 18, "3 chunks of 6 frames = 18")
    }

    // MARK: - Segment Generation

    func testHighProbabilityUpdatesFramePredictions() {
        let timeline = SortformerTimeline()
        let numSpeakers = 4
        let frameCount = 12

        // Speaker 0 has high probability (0.9) for all frames, others are silent
        var predictions = [Float](repeating: 0.0, count: frameCount * numSpeakers)
        for frame in 0..<frameCount {
            predictions[frame * numSpeakers + 0] = 0.9
        }

        let chunk = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: predictions,
            frameCount: frameCount
        )
        timeline.addChunk(chunk)
        timeline.finalize()

        // Verify frame predictions are stored correctly
        XCTAssertEqual(timeline.numFrames, frameCount)
        XCTAssertEqual(timeline.probability(speaker: 0, frame: 0), 0.9, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 1, frame: 0), 0.0, accuracy: 1e-5)
    }

    // MARK: - Reset

    func testResetClearsState() {
        let timeline = SortformerTimeline()
        let numSpeakers = 4
        let frameCount = 6

        let chunk = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: [Float](repeating: 0.9, count: frameCount * numSpeakers),
            frameCount: frameCount
        )
        timeline.addChunk(chunk)
        timeline.reset()

        XCTAssertEqual(timeline.numFrames, 0)
        XCTAssertTrue(timeline.framePredictions.isEmpty)
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
        for speakerSegments in timeline.segments {
            XCTAssertTrue(speakerSegments.isEmpty)
        }
    }

    // MARK: - Finalize

    func testFinalizeMovesDataToFinalized() {
        let timeline = SortformerTimeline()
        let numSpeakers = 4
        let frameCount = 6

        // Create chunk with tentative predictions
        let tentativeCount = 4
        let chunk = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
            frameCount: frameCount,
            tentativePredictions: [Float](repeating: 0, count: tentativeCount * numSpeakers),
            tentativeFrameCount: tentativeCount
        )
        timeline.addChunk(chunk)

        let framesBefore = timeline.numFrames
        let tentativeBefore = timeline.numTentative

        timeline.finalize()

        XCTAssertEqual(timeline.numFrames, framesBefore + tentativeBefore)
        XCTAssertEqual(timeline.numTentative, 0, "After finalize, no tentative predictions should remain")
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
    }

    // MARK: - Probability Access

    func testProbabilityAccess() {
        let numSpeakers = 4
        // [f0s0=0.1, f0s1=0.2, f0s2=0.3, f0s3=0.4, f1s0=0.5, ...]
        let predictions: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        let timeline = SortformerTimeline(
            allPredictions: predictions,
            config: .default,
            isComplete: true
        )

        XCTAssertEqual(timeline.probability(speaker: 0, frame: 0), 0.1, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 3, frame: 0), 0.4, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 0, frame: 1), 0.5, accuracy: 1e-5)
        XCTAssertEqual(
            timeline.probability(speaker: 0, frame: 999), 0.0, "Out of range should return 0"
        )
    }

    // MARK: - SortformerSegment

    func testSegmentTimeConversion() {
        let segment = SortformerSegment(speakerIndex: 0, startFrame: 10, endFrame: 20)
        XCTAssertEqual(segment.startTime, 0.8, accuracy: 1e-5, "10 * 0.08 = 0.8")
        XCTAssertEqual(segment.endTime, 1.6, accuracy: 1e-5, "20 * 0.08 = 1.6")
        XCTAssertEqual(segment.duration, 0.8, accuracy: 1e-5, "(20-10) * 0.08 = 0.8")
        XCTAssertEqual(segment.length, 10)
    }

    func testSegmentOverlap() {
        let a = SortformerSegment(speakerIndex: 0, startFrame: 0, endFrame: 10)
        let b = SortformerSegment(speakerIndex: 0, startFrame: 5, endFrame: 15)
        let c = SortformerSegment(speakerIndex: 0, startFrame: 11, endFrame: 20)

        XCTAssertTrue(a.overlaps(with: b), "Overlapping segments")
        XCTAssertFalse(a.overlaps(with: c), "Non-overlapping segments")
    }

    func testSegmentAbsorb() {
        var a = SortformerSegment(speakerIndex: 0, startFrame: 5, endFrame: 10)
        let b = SortformerSegment(speakerIndex: 0, startFrame: 3, endFrame: 15)
        a.absorb(b)

        XCTAssertEqual(a.startFrame, 3)
        XCTAssertEqual(a.endFrame, 15)
    }

    func testSegmentInitFromTime() {
        let segment = SortformerSegment(
            speakerIndex: 1,
            startTime: 0.8,
            endTime: 1.6
        )
        XCTAssertEqual(segment.startFrame, 10, "0.8 / 0.08 = 10")
        XCTAssertEqual(segment.endFrame, 20, "1.6 / 0.08 = 20")
        XCTAssertEqual(segment.speakerIndex, 1)
    }
}
