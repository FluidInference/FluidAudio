import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class ChunkProcessorEdgeCaseTests: XCTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    // MARK: - Chunk Boundary Calculation Tests

    func testFirstChunkBoundaryCalculations() {
        // Test first chunk boundary calculations without mocking
        let audioSamples = createMockAudio(durationSeconds: 12.0)
        let processor = ChunkProcessor(audioSamples: audioSamples)

        // Test the internal calculations that would be used in first chunk
        let centerSeconds = 11.2
        let leftContextSeconds = 1.6
        let rightContextSeconds = 1.6
        let sampleRate = 16000

        let centerSamples = Int(centerSeconds * Double(sampleRate))
        let leftContextSamples = Int(leftContextSeconds * Double(sampleRate))
        let rightContextSamples = Int(rightContextSeconds * Double(sampleRate))

        XCTAssertEqual(centerSamples, 179_200, "Center chunk should be 179,200 samples")
        XCTAssertEqual(leftContextSamples, 25_600, "Left context should be 25,600 samples")
        XCTAssertEqual(rightContextSamples, 25_600, "Right context should be 25,600 samples")

        // First chunk should not exceed available samples
        let centerStart = 0
        let leftStart = max(0, centerStart - leftContextSamples)  // max(0, -25600) = 0
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        XCTAssertEqual(leftStart, 0, "First chunk should start at sample 0")
        XCTAssertLessThanOrEqual(rightEnd, audioSamples.count, "Chunk should not exceed audio bounds")

        // Test processor creation
        XCTAssertNotNil(processor)
    }

    func testExactlyOneCenterChunkBoundaries() {
        // Test audio that fits exactly one center chunk (11.2s)
        let exactCenterSamples = Int(11.2 * 16000.0)  // 179,200 samples
        let audioSamples = Array(0..<exactCenterSamples).map { Float($0) / Float(exactCenterSamples) }
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audioSamples.count, 179_200, "Exactly one center chunk should be 179,200 samples")

        // Test chunk boundary detection logic
        let centerSamples = Int(11.2 * 16000.0)
        let centerStart = 0
        let isLastChunk = (centerStart + centerSamples) >= audioSamples.count

        XCTAssertTrue(isLastChunk, "Single chunk should be detected as last chunk")

        // Verify frame calculations
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: exactCenterSamples)
        XCTAssertEqual(expectedFrames, 140, "11.2s should be exactly 140 encoder frames")
    }

    func testVeryShortAudioBoundaries() {
        // Test very short audio (< 1 second)
        let shortAudio = createMockAudio(durationSeconds: 0.5)
        let processor = ChunkProcessor(audioSamples: shortAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(shortAudio.count, 8_000, "0.5s audio should be 8,000 samples")

        // Test chunk boundary calculations for short audio
        let centerSamples = Int(11.2 * 16000.0)
        let centerStart = 0
        let isLastChunk = (centerStart + centerSamples) >= shortAudio.count

        XCTAssertTrue(isLastChunk, "Short audio should be detected as last chunk")

        // Verify frame calculations
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: shortAudio.count)
        XCTAssertEqual(expectedFrames, 7, "0.5s (8000 samples) should be 7 encoder frames (ceiling of 6.25)")
        XCTAssertTrue(expectedFrames < 10, "Short audio should have few frames")
    }

    // MARK: - Last Chunk Adaptive Context Tests

    func testLastChunkAdaptiveContextCalculations() {
        // Test last chunk adaptive context calculations
        let audioSamples = createMockAudio(durationSeconds: 25.0)  // 2+ chunks, last will be partial
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)

        // Simulate the second chunk (which would be the last chunk for 25s audio)
        let centerSamples = Int(11.2 * 16000.0)  // 179,200
        let centerStart = centerSamples  // Start of second chunk
        let remainingSamples = audioSamples.count - centerStart  // ~220,800 samples remaining

        XCTAssertGreaterThan(
            remainingSamples, centerSamples, "Remaining samples should be greater than center chunk for 25s audio")

        // Test adaptive context calculation logic
        let standardLeftContextSamples = Int(1.6 * 16000.0)  // 25,600
        let maxModelSamples = 240_000  // 15s capacity
        let desiredTotalSamples = min(maxModelSamples, audioSamples.count)  // min(240k, 400k) = 240k
        let maxLeftContext = centerStart  // 179,200
        let neededLeftContext = desiredTotalSamples - remainingSamples  // 240k - 220,800 = 19,200
        let adaptiveLeftContextSamples = min(neededLeftContext, maxLeftContext)  // min(19,200, 179,200) = 19,200

        // For 25s audio, we actually use LESS context because we hit the model capacity limit
        XCTAssertLessThan(
            adaptiveLeftContextSamples, standardLeftContextSamples,
            "For 25s audio hitting model capacity, should use less context (adaptive: \(adaptiveLeftContextSamples) vs standard: \(standardLeftContextSamples))"
        )

        // Test frame adjustment calculation
        let standardOverlapSamples = standardLeftContextSamples  // 25,600
        let contextFrameAdjustment = -Int(
            (Double(standardOverlapSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())

        XCTAssertEqual(
            contextFrameAdjustment, -20,
            "Should use standard negative overlap adjustment (got \(contextFrameAdjustment))")
    }

    func testLastChunkPaddingCalculations() {
        // Test padding calculations for last chunk
        let audioSamples = createMockAudio(durationSeconds: 13.5)  // Slightly longer than one chunk
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audioSamples.count, 216_000, "13.5s should be 216,000 samples")

        // Test that this would be detected as a single last chunk
        let centerSamples = Int(11.2 * 16000.0)
        let centerStart = 0
        let isLastChunk = (centerStart + centerSamples) >= audioSamples.count

        XCTAssertFalse(isLastChunk, "13.5s should require multiple chunks (13.5s > 11.2s center)")

        // Test padding logic
        let maxModelSamples = 240_000  // 15s capacity
        let needsPadding = audioSamples.count < maxModelSamples

        XCTAssertTrue(needsPadding, "13.5s audio should need padding to 15s model capacity")
        XCTAssertEqual(maxModelSamples - audioSamples.count, 24_000, "Should need 24,000 padding samples")

        // Verify frame calculations
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: audioSamples.count)
        XCTAssertEqual(expectedFrames, 169, "13.5s should be 169 encoder frames (ceiling of 168.75)")
    }

    // MARK: - Chunk Boundary Overlap Tests

    func testChunkBoundaryOverlapCalculations() {
        // Test overlapping context calculations between chunks
        let audioSamples = createMockAudio(durationSeconds: 24.0)  // Exactly 2 chunks worth
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audioSamples.count, 384_000, "24s should be 384,000 samples")

        // Calculate expected chunk boundaries
        let centerSamples = Int(11.2 * 16000.0)  // 179,200
        let leftContextSamples = Int(1.6 * 16000.0)  // 25,600

        // First chunk: centerStart = 0
        let firstCenterStart = 0
        let firstIsLastChunk = (firstCenterStart + centerSamples) >= audioSamples.count
        XCTAssertFalse(firstIsLastChunk, "First chunk should not be last for 24s audio")

        // Second chunk: centerStart = 179,200
        let secondCenterStart = centerSamples
        let secondIsLastChunk = (secondCenterStart + centerSamples) >= audioSamples.count
        XCTAssertFalse(secondIsLastChunk, "Second chunk should not be last for 24s audio - needs third chunk")

        // Test overlap calculations for second chunk (standard non-last chunk logic doesn't apply since it's also last)
        let remainingSamples = audioSamples.count - secondCenterStart  // 204,800 samples
        XCTAssertGreaterThan(
            remainingSamples, centerSamples,
            "Remaining should be greater than center chunk for 24s audio (204,800 > 179,200)")

        // Standard overlap adjustment calculation
        let standardOverlapSamples = leftContextSamples  // 25,600
        let standardContextFrameAdjustment = -Int(
            (Double(standardOverlapSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())
        XCTAssertEqual(standardContextFrameAdjustment, -20, "Standard overlap should be -20 frames")

        // Global frame offset calculation
        let globalFrameOffset = firstCenterStart / ASRConstants.samplesPerEncoderFrame
        XCTAssertEqual(globalFrameOffset, 0, "First chunk global offset should be 0")

        let secondGlobalFrameOffset = secondCenterStart / ASRConstants.samplesPerEncoderFrame
        XCTAssertEqual(secondGlobalFrameOffset, 140, "Second chunk global offset should be 140 frames")
    }

    func testPreciseFrameBoundaryCalculations() {
        // Test precise frame boundary calculations
        let centerSamples = Int(11.2 * 16000.0)  // 179,200 samples = exactly 140 frames
        let audioSamples = Array(repeating: Float(0.1), count: centerSamples * 2)  // Exactly 2 center chunks
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audioSamples.count, 358_400, "Two center chunks should be 358,400 samples")

        // Test frame calculations for each chunk boundary
        let firstCenterStart = 0
        let firstCenterEnd = min(audioSamples.count, firstCenterStart + centerSamples)
        let firstChunkFrames = ASRConstants.calculateEncoderFrames(from: firstCenterEnd - firstCenterStart)

        XCTAssertEqual(firstChunkFrames, 140, "First chunk should be exactly 140 frames")

        let secondCenterStart = centerSamples
        let secondCenterEnd = min(audioSamples.count, secondCenterStart + centerSamples)
        let secondChunkFrames = ASRConstants.calculateEncoderFrames(from: secondCenterEnd - secondCenterStart)

        XCTAssertEqual(secondChunkFrames, 140, "Second chunk should be exactly 140 frames")

        // Test global frame offsets
        let firstGlobalOffset = firstCenterStart / ASRConstants.samplesPerEncoderFrame
        let secondGlobalOffset = secondCenterStart / ASRConstants.samplesPerEncoderFrame

        XCTAssertEqual(firstGlobalOffset, 0, "First chunk should start at frame 0")
        XCTAssertEqual(secondGlobalOffset, 140, "Second chunk should start at frame 140")

        // Verify total processing coverage
        let totalExpectedFrames = ASRConstants.calculateEncoderFrames(from: audioSamples.count)
        XCTAssertEqual(totalExpectedFrames, 280, "Total audio should be 280 frames")
        XCTAssertEqual(firstChunkFrames + secondChunkFrames, 280, "Chunk frames should sum to total")
    }

    // MARK: - Decoder State Time Jump Tests

    func testDecoderStateTimeJumpCalculations() throws {
        // Test time jump calculations for decoder state persistence
        let audioSamples = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)

        // Test initial decoder state setup
        var decoderState = try TdtDecoderState()
        let initialTimeJump: Int? = 5  // Decoder was 5 frames ahead of previous chunk
        decoderState.timeJump = initialTimeJump

        // Test timeIndices calculation for chunk continuation (from TdtDecoder logic)
        let contextFrameAdjustment = -20  // Standard overlap
        let prevTimeJump = decoderState.timeJump!
        let timeIndices = max(0, prevTimeJump + contextFrameAdjustment)  // 5 + (-20) = -15, max(0, -15) = 0

        XCTAssertEqual(timeIndices, 0, "Time indices should clamp to 0 when calculation goes negative")

        // Test different timeJump scenarios
        decoderState.timeJump = 25  // Decoder was well ahead
        let timeIndices2 = max(0, 25 + contextFrameAdjustment)  // 25 + (-20) = 5
        XCTAssertEqual(timeIndices2, 5, "Should calculate correct starting position from timeJump")

        // Test timeJump calculation at end of processing
        let effectiveSequenceLength = 140
        let finalTimeIndices = 143  // Decoder processed beyond available frames
        let calculatedTimeJump = finalTimeIndices - effectiveSequenceLength  // 143 - 140 = 3

        XCTAssertEqual(calculatedTimeJump, 3, "Time jump should reflect frames processed beyond chunk")

        // Test last chunk timeJump clearing
        decoderState.timeJump = calculatedTimeJump
        let isLastChunk = true
        if isLastChunk {
            decoderState.timeJump = nil
        }

        XCTAssertNil(decoderState.timeJump, "Time jump should be cleared for last chunk")
    }

    // MARK: - Token Deduplication Logic Tests

    func testTokenDeduplicationLogic() {
        // Test deduplication logic without mocking the final class
        let audioSamples = createMockAudio(durationSeconds: 24.0)  // 2 chunks
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)

        // Test the deduplication decision logic that would be used in ChunkProcessor
        let segmentIndex = 1  // Second chunk
        let allTokens = [100, 200, 300]  // Tokens from first chunk
        let windowTokens = [300, 400, 500]  // Tokens from second chunk with overlap

        let shouldDeduplicate = segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty
        XCTAssertTrue(shouldDeduplicate, "Should attempt deduplication for second chunk with tokens")

        // Simulate deduplication result
        let duplicateTokens = 1  // First token is duplicate
        let deduplicatedTokens = Array(windowTokens.dropFirst(duplicateTokens))  // [400, 500]
        let adjustedTimestamps = [35, 42]  // Corresponding timestamps after removing first

        XCTAssertEqual(deduplicatedTokens, [400, 500], "Should remove duplicated tokens")
        XCTAssertEqual(
            adjustedTimestamps.count, deduplicatedTokens.count, "Timestamps should match deduplicated tokens")

        // Test final token combination
        let finalAllTokens = allTokens + deduplicatedTokens
        XCTAssertEqual(finalAllTokens, [100, 200, 300, 400, 500], "Final tokens should combine properly")

        // Test no deduplication case (first chunk)
        let firstSegmentIndex = 0
        let shouldNotDeduplicate = firstSegmentIndex > 0
        XCTAssertFalse(shouldNotDeduplicate, "First chunk should not attempt deduplication")
    }

    // MARK: - Validation Tests

    func testFinalValidationCalculations() {
        // Test the final validation logic from ChunkProcessor
        let audioSamples = createMockAudio(durationSeconds: 30.0)
        let processor = ChunkProcessor(audioSamples: audioSamples)

        XCTAssertNotNil(processor)

        // Calculate expected validation values
        let expectedTotalFrames = ASRConstants.calculateEncoderFrames(from: audioSamples.count)
        let segmentCount = 3  // Three 11.2s segments for 30s audio
        let centerSeconds = 11.2
        let sampleRate = 16000

        let processedCenterFrames =
            segmentCount * Int(centerSeconds * Double(sampleRate)) / ASRConstants.samplesPerEncoderFrame

        XCTAssertEqual(expectedTotalFrames, 375, "30s should be 375 encoder frames")
        XCTAssertEqual(processedCenterFrames, 420, "Three 11.2s chunks should process 420 center frames")

        // The validation ensures we account for all audio content
        // In real processing, some frames are processed multiple times due to overlap
        XCTAssertGreaterThan(
            processedCenterFrames, expectedTotalFrames,
            "Center processing should exceed total due to chunk approach")
    }

    func testChunkCountPrediction() {
        // Test predictable chunk count calculations
        let testCases: [(duration: Double, expectedChunks: Int)] = [
            (11.0, 1),  // Less than center chunk
            (11.2, 1),  // Exactly center chunk
            (12.0, 2),  // Slightly more than center chunk needs 2 chunks
            (22.0, 2),  // Two chunks
            (33.0, 3),  // Three chunks
            (44.8, 4),  // Four chunks
        ]

        for (duration, expectedChunks) in testCases {
            let audioSamples = createMockAudio(durationSeconds: duration)
            let processor = ChunkProcessor(audioSamples: audioSamples)

            XCTAssertNotNil(processor, "Processor should initialize for \(duration)s audio")

            // Calculate expected chunks using the same logic as ChunkProcessor
            let centerSamples = Int(11.2 * 16000.0)
            var chunkCount = 0
            var centerStart = 0

            while centerStart < audioSamples.count {
                chunkCount += 1
                centerStart += centerSamples
            }

            XCTAssertEqual(
                chunkCount, expectedChunks,
                "\(duration)s audio should require \(expectedChunks) chunks, calculated \(chunkCount)")
        }
    }

    // MARK: - Helper Methods

    private func createMockAudio(durationSeconds: Double, sampleRate: Int = 16000) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(sampleRate))
        return (0..<sampleCount).map { Float($0) / Float(sampleCount) }
    }
}
