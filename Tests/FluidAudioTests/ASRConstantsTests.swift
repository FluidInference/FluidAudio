import Foundation
import XCTest

@testable import FluidAudio

final class ASRConstantsTests: XCTestCase {

    // MARK: - Frame Calculation Tests

    func testCalculateEncoderFramesBasicCases() {
        // Test basic frame calculations using ceiling division
        XCTAssertEqual(ASRConstants.calculateEncoderFrames(from: 0), 0, "0 samples should be 0 frames")
        XCTAssertEqual(ASRConstants.calculateEncoderFrames(from: 1280), 1, "1280 samples should be 1 frame")
        XCTAssertEqual(ASRConstants.calculateEncoderFrames(from: 2560), 2, "2560 samples should be 2 frames")
        XCTAssertEqual(
            ASRConstants.calculateEncoderFrames(from: 16000), 13, "1 second should be 13 frames (ceiling of 12.5)")
    }

    func testCalculateEncoderFramesChunkBoundaries() {
        // Test frame calculations for new stateless chunk processing boundaries
        let chunkSeconds = 14.96  // ~14.96s chunks to stay under 240,000 sample limit
        let overlapSeconds = 2.0  // 2.0s overlap between chunks
        let sampleRate = 16000

        let chunkSamples = Int(chunkSeconds * Double(sampleRate))  // ~239,360 samples
        let overlapSamples = Int(overlapSeconds * Double(sampleRate))  // 32,000 samples

        let chunkFrames = ASRConstants.calculateEncoderFrames(from: chunkSamples)
        let overlapFrames = ASRConstants.calculateEncoderFrames(from: overlapSamples)

        XCTAssertEqual(chunkFrames, 187, "~14.96s chunks should be 187 frames")
        XCTAssertEqual(overlapFrames, 25, "2.0s overlap should be exactly 25 frames")

        // Total model capacity is 15s = 188 frames
        XCTAssertLessThanOrEqual(chunkFrames, 188, "Chunks should be within 15s model limit")
    }

    func testCalculateEncoderFramesModelLimits() {
        // Test frame calculations for model capacity limits
        let maxModelSamples = 240_000  // 15 seconds
        let maxModelFrames = ASRConstants.calculateEncoderFrames(from: maxModelSamples)

        XCTAssertEqual(maxModelFrames, 188, "15s should be 188 frames (ceiling of 187.5)")

        // Test just over the limit
        let overLimitSamples = 240_001
        let overLimitFrames = ASRConstants.calculateEncoderFrames(from: overLimitSamples)
        XCTAssertEqual(overLimitFrames, 188, "Slightly over 15s should still be 188 frames")

        // Test exactly at frame boundary
        let exactFrameBoundary = 188 * ASRConstants.samplesPerEncoderFrame  // 188 * 1280
        let exactFrames = ASRConstants.calculateEncoderFrames(from: exactFrameBoundary)
        XCTAssertEqual(exactFrames, 188, "Exact frame boundary should match")
    }

    func testCalculateEncoderFramesRoundingBehavior() {
        // Test ceiling division behavior for fractional frames
        let testCases: [(samples: Int, expectedFrames: Int, description: String)] = [
            (640, 1, "0.5 frames should round up to 1"),
            (1280, 1, "1.0 frames should be exactly 1"),
            (1920, 2, "1.5 frames should round up to 2"),
            (2559, 2, "1.999 frames should round up to 2"),
            (2560, 2, "2.0 frames should be exactly 2"),
            (3200, 3, "2.5 frames should round up to 3"),
        ]

        for (samples, expectedFrames, description) in testCases {
            let actualFrames = ASRConstants.calculateEncoderFrames(from: samples)
            XCTAssertEqual(actualFrames, expectedFrames, description)
        }
    }

    func testCalculateEncoderFramesLargeNumbers() {
        // Test with large sample counts (long audio)
        let oneMinuteSamples = 60 * 16000  // 960,000 samples
        let oneMinuteFrames = ASRConstants.calculateEncoderFrames(from: oneMinuteSamples)
        XCTAssertEqual(oneMinuteFrames, 750, "1 minute should be 750 frames")

        let oneHourSamples = 3600 * 16000  // 57,600,000 samples
        let oneHourFrames = ASRConstants.calculateEncoderFrames(from: oneHourSamples)
        XCTAssertEqual(oneHourFrames, 45_000, "1 hour should be 45,000 frames")
    }

    // MARK: - Constants Validation Tests

    func testSamplesPerEncoderFrameConstant() {
        // Verify the fundamental constant used throughout the system
        XCTAssertEqual(ASRConstants.samplesPerEncoderFrame, 1280, "Samples per encoder frame should be 1280")

        // Verify it matches the expected frame duration at 16kHz
        let expectedFrameDurationMs = 80.0  // Each frame = 80ms
        let actualFrameDurationMs = Double(ASRConstants.samplesPerEncoderFrame) / 16000.0 * 1000.0
        XCTAssertEqual(actualFrameDurationMs, expectedFrameDurationMs, accuracy: 0.1, "Frame duration should be 80ms")
    }

    func testFrameCalculationConsistencyWithChunkProcessor() {
        // Test that frame calculations are consistent with stateless ChunkProcessor expectations
        let chunkSeconds = 14.96  // New stateless chunks: ~14.96s
        let sampleRate = 16000
        let chunkSamples = Int(chunkSeconds * Double(sampleRate))
        let chunkFrames = ASRConstants.calculateEncoderFrames(from: chunkSamples)

        // This should match the chunk size in new stateless ChunkProcessor
        // ~14.96s * 16000 / 1280 = ~187 frames
        let expectedFrames = Int(ceil(14.96 * 16000.0 / 1280.0))
        XCTAssertEqual(chunkFrames, expectedFrames, "Chunk frame calculation should match ChunkProcessor")
        XCTAssertEqual(chunkFrames, 187, "~14.96s chunks should be 187 frames")
    }

    func testFrameCalculationConsistencyWithTdtDecoder() {
        // Test frame calculations match new overlap-based merging in ChunkProcessor
        let overlapSeconds = 2.0  // New stateless approach: 2.0s overlap
        let sampleRate = 16000
        let overlapSamples = Int(overlapSeconds * Double(sampleRate))
        let overlapFrames = ASRConstants.calculateEncoderFrames(from: overlapSamples)

        // This should match the overlap configuration in stateless ChunkProcessor
        // 2.0s * 16000 / 1280 = exactly 25 frames
        XCTAssertEqual(overlapFrames, 25, "2.0s overlap should be exactly 25 frames")

        // Verify: halfOverlapWindow = overlapSeconds / 2 = 1.0s
        let halfOverlapWindow = overlapSeconds / 2.0
        XCTAssertEqual(halfOverlapWindow, 1.0, "Half overlap window should be 1.0s for token matching tolerance")
    }

    // MARK: - Edge Case Tests

    func testCalculateEncoderFramesWithZeroInput() {
        let frames = ASRConstants.calculateEncoderFrames(from: 0)
        XCTAssertEqual(frames, 0, "Zero samples should produce zero frames")
    }

    func testCalculateEncoderFramesWithNegativeInput() {
        // Although this shouldn't happen in practice, test the behavior
        let frames = ASRConstants.calculateEncoderFrames(from: -100)
        XCTAssertEqual(frames, 0, "Negative samples should produce zero frames")
    }

    func testCalculateEncoderFramesWithVerySmallInput() {
        // Test sub-frame sample counts - with ceiling division, even 1 sample produces 1 frame
        for samples in 1..<1280 {
            let frames = ASRConstants.calculateEncoderFrames(from: samples)
            XCTAssertEqual(frames, 1, "Any positive samples less than 1280 should produce 1 frame (sample \(samples))")
        }
    }

    func testCalculateEncoderFramesWithMaxInt() {
        // Test with maximum possible input (edge case for very long audio)
        let maxSamples = Int.max
        let frames = ASRConstants.calculateEncoderFrames(from: maxSamples)

        // Should not crash and should produce a reasonable result
        XCTAssertGreaterThan(frames, 0, "Max int samples should produce positive frames")
        XCTAssertLessThan(frames, Int.max, "Frame count should be less than max int")

        // Verify calculation doesn't overflow - use ceiling division
        let expectedMaxFrames = Int(ceil(Double(maxSamples) / Double(ASRConstants.samplesPerEncoderFrame)))
        XCTAssertEqual(frames, expectedMaxFrames, "Max calculation should match expected ceiling division")
    }

    // MARK: - Performance Tests

    func testCalculateEncoderFramesPerformance() {
        // Test performance of frame calculation (should be very fast)
        let sampleCounts = [16000, 160000, 1_600_000, 16_000_000]  // Various audio lengths

        measure {
            for _ in 0..<10000 {
                for samples in sampleCounts {
                    _ = ASRConstants.calculateEncoderFrames(from: samples)
                }
            }
        }
    }

    // MARK: - Consistency Tests

    func testFrameToSampleConversion() {
        // Test that we can convert frames back to samples consistently
        let testFrameCounts = [0, 1, 10, 100, 1000]

        for frameCount in testFrameCounts {
            let samples = frameCount * ASRConstants.samplesPerEncoderFrame
            let calculatedFrames = ASRConstants.calculateEncoderFrames(from: samples)
            XCTAssertEqual(calculatedFrames, frameCount, "Frame -> Sample -> Frame conversion should be consistent")
        }
    }

    func testTimestampConversion() {
        // Test conversion between frames and time (as mentioned in TdtDecoder comments)
        let frameIndices = [0, 10, 100, 1000]
        let expectedTimesSeconds = [0.0, 0.8, 8.0, 80.0]  // frame_index * 0.08 = time_in_seconds

        for (frameIndex, expectedTime) in zip(frameIndices, expectedTimesSeconds) {
            let timeSeconds = Double(frameIndex) * 0.08
            XCTAssertEqual(
                timeSeconds, expectedTime, accuracy: 0.001, "Frame \(frameIndex) should convert to \(expectedTime)s")

            // Reverse conversion: time -> samples -> frames
            let samples = Int(timeSeconds * 16000.0)
            let calculatedFrames = ASRConstants.calculateEncoderFrames(from: samples)
            XCTAssertEqual(calculatedFrames, frameIndex, "Time -> Samples -> Frames should round-trip correctly")
        }
    }
}
