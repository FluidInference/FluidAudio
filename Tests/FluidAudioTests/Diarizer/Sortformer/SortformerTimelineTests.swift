import Foundation
import XCTest

@testable import FluidAudio

final class SortformerTimelineTests: XCTestCase {

    // MARK: - Empty Timeline

    func testEmptyTimelineHasZeroDuration() {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        XCTAssertEqual(timeline.numFinalizedFrames, 0)
        XCTAssertEqual(timeline.finalizedDuration, 0)
        XCTAssertTrue(timeline.finalizedPredictions.isEmpty)
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
    }

    func testEmptyTimelineHasEmptySegments() {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        for speakerSegments in timeline.speakers {
            XCTAssertTrue(speakerSegments.value.finalizedSegments.isEmpty)
        }
    }

    // MARK: - Adding Chunks

    func testAddChunkUpdatesDuration() throws {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        let numSpeakers = 4
        let frameCount = 6

        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
            finalizedFrameCount: frameCount
        )

        try timeline.addChunk(chunk)

        XCTAssertEqual(timeline.numFinalizedFrames, 6)
        XCTAssertEqual(timeline.finalizedDuration, Float(6) * 0.08, accuracy: 1e-5)
    }

    func testAddMultipleChunksAccumulatesFrames() throws {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        let numSpeakers = 4
        let frameCount = 6

        for i in 0..<3 {
            let chunk = DiarizerChunkResult(
                startFrame: i * frameCount,
                finalizedPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
                finalizedFrameCount: frameCount
            )
            try timeline.addChunk(chunk)
        }

        XCTAssertEqual(timeline.numFinalizedFrames, 18, "3 chunks of 6 frames = 18")
    }

    // MARK: - Prediction Storage

    func testHighProbabilityStoresPredictions() throws {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        let numSpeakers = 4
        let frameCount = 12

        // Speaker 0 has high probability (0.9) for all frames, others are silent
        var predictions = [Float](repeating: 0.0, count: frameCount * numSpeakers)
        for frame in 0..<frameCount {
            predictions[frame * numSpeakers + 0] = 0.9
        }

        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: predictions,
            finalizedFrameCount: frameCount
        )
        try timeline.addChunk(chunk)
        timeline.finalize()

        // Verify frame predictions are stored correctly
        XCTAssertEqual(timeline.numFinalizedFrames, frameCount)
        XCTAssertEqual(timeline.probability(speaker: 0, frame: 0), 0.9, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 1, frame: 0), 0.0, accuracy: 1e-5)
    }

    // MARK: - Reset

    func testResetClearsState() throws {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        let numSpeakers = 4
        let frameCount = 6

        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: [Float](repeating: 0.9, count: frameCount * numSpeakers),
            finalizedFrameCount: frameCount
        )
        try timeline.addChunk(chunk)
        timeline.reset()

        XCTAssertEqual(timeline.numFinalizedFrames, 0)
        XCTAssertTrue(timeline.finalizedPredictions.isEmpty)
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
        for speakerSegments in timeline.speakers.values {
            XCTAssertTrue(speakerSegments.finalizedSegments.isEmpty)
        }
    }

    // MARK: - Finalize

    func testFinalizeMovesDataToFinalized() throws {
        let timeline = DiarizerTimeline(config: .sortformerDefault)
        let numSpeakers = 4
        let frameCount = 6

        // Create chunk with tentative predictions
        let tentativeCount = 4
        let chunk = DiarizerChunkResult(
            startFrame: 0,
            finalizedPredictions: [Float](repeating: 0, count: frameCount * numSpeakers),
            finalizedFrameCount: frameCount,
            tentativePredictions: [Float](repeating: 0, count: tentativeCount * numSpeakers),
            tentativeFrameCount: tentativeCount
        )
        try timeline.addChunk(chunk)

        let framesBefore = timeline.numFinalizedFrames
        let tentativeBefore = timeline.numTentativeFrames

        timeline.finalize()

        XCTAssertEqual(timeline.numFinalizedFrames, framesBefore + tentativeBefore)
        XCTAssertEqual(timeline.numTentativeFrames, 0, "After finalize, no tentative predictions should remain")
        XCTAssertTrue(timeline.tentativePredictions.isEmpty)
    }

    func testSegmentConfidenceExcludesPaddingFrames() throws {
        let config = DiarizerTimelineConfig(
            numSpeakers: 1,
            frameDurationSeconds: 0.08,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 1,
            offsetPadFrames: 2,
            minFramesOn: 0,
            minFramesOff: 0
        )
        let timeline = DiarizerTimeline(config: config)
        let predictions: [Float] = [0.0, 0.8, 0.6, 0.0]

        try timeline.addChunk(
            DiarizerChunkResult(
                startFrame: 0,
                finalizedPredictions: predictions,
                finalizedFrameCount: predictions.count
            )
        )

        timeline.finalize()

        let segment = try XCTUnwrap(timeline.speakers[0]?.finalizedSegments.first)
        XCTAssertEqual(segment.startFrame, 0)
        XCTAssertEqual(segment.endFrame, 5)
        XCTAssertEqual(segment.confidence, 0.7, accuracy: 1e-6)
    }

    func testSegmentConfidenceExcludesBridgedGapFrames() throws {
        let config = DiarizerTimelineConfig(
            numSpeakers: 1,
            frameDurationSeconds: 0.08,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 0,
            minFramesOff: 1
        )
        let timeline = DiarizerTimeline(config: config)
        let predictions: [Float] = [0.9, 0.0, 0.7, 0.7, 0.0]

        try timeline.addChunk(
            DiarizerChunkResult(
                startFrame: 0,
                finalizedPredictions: predictions,
                finalizedFrameCount: predictions.count
            )
        )

        timeline.finalize()

        let segment = try XCTUnwrap(timeline.speakers[0]?.finalizedSegments.first)
        XCTAssertEqual(segment.startFrame, 0)
        XCTAssertEqual(segment.endFrame, 4)
        XCTAssertEqual(segment.confidence, (0.9 + 0.7 + 0.7) / 3.0, accuracy: 1e-6)
    }

    // MARK: - Probability Access

    func testProbabilityAccess() throws {
        // [f0s0=0.1, f0s1=0.2, f0s2=0.3, f0s3=0.4, f1s0=0.5, ...]
        let predictions: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        let timeline = try DiarizerTimeline(
            allPredictions: predictions,
            config: .sortformerDefault,
            isComplete: true
        )

        XCTAssertEqual(timeline.probability(speaker: 0, frame: 0), 0.1, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 3, frame: 0), 0.4, accuracy: 1e-5)
        XCTAssertEqual(timeline.probability(speaker: 0, frame: 1), 0.5, accuracy: 1e-5)
        XCTAssertTrue(
            timeline.probability(speaker: 0, frame: 999).isNaN,
            "Out of range should return NaN"
        )
    }

    // MARK: - SortformerSegment

    func testSegmentTimeConversion() {
        let segment = DiarizerSegment(speakerIndex: 0, startFrame: 10, endFrame: 20, frameDurationSeconds: 0.08)
        XCTAssertEqual(segment.startTime, 0.8, accuracy: 1e-5, "10 * 0.08 = 0.8")
        XCTAssertEqual(segment.endTime, 1.6, accuracy: 1e-5, "20 * 0.08 = 1.6")
        XCTAssertEqual(segment.duration, 0.8, accuracy: 1e-5, "(20-10) * 0.08 = 0.8")
        XCTAssertEqual(segment.length, 10)
    }

    func testSegmentOverlap() {
        let a = DiarizerSegment(speakerIndex: 0, startFrame: 0, endFrame: 10, frameDurationSeconds: 0.08)
        let b = DiarizerSegment(speakerIndex: 0, startFrame: 5, endFrame: 15, frameDurationSeconds: 0.08)
        let c = DiarizerSegment(speakerIndex: 0, startFrame: 11, endFrame: 20, frameDurationSeconds: 0.08)
        let d = DiarizerSegment(speakerIndex: 0, startFrame: 10, endFrame: 20, frameDurationSeconds: 0.08)

        XCTAssertTrue(a.overlaps(with: b), "Overlapping segments")
        XCTAssertFalse(a.overlaps(with: c), "Non-overlapping segments (gap of 1)")
        XCTAssertTrue(a.overlaps(with: d), "Touching segments (endFrame == startFrame) count as overlapping")
    }

    func testSegmentAbsorb() {
        var a = DiarizerSegment(speakerIndex: 0, startFrame: 5, endFrame: 10, frameDurationSeconds: 0.08)
        let b = DiarizerSegment(speakerIndex: 0, startFrame: 3, endFrame: 15, frameDurationSeconds: 0.08)
        a.absorb(b)

        XCTAssertEqual(a.startFrame, 3)
        XCTAssertEqual(a.endFrame, 15)
    }

    func testSegmentInitFromTime() {
        let segment = DiarizerSegment(
            speakerIndex: 1,
            startTime: 0.8,
            endTime: 1.6,
            frameDurationSeconds: 0.08
        )
        XCTAssertEqual(segment.startFrame, 10, "0.8 / 0.08 = 10")
        XCTAssertEqual(segment.endFrame, 20, "1.6 / 0.08 = 20")
        XCTAssertEqual(segment.speakerIndex, 1)
    }
}
