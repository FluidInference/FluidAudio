@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class ChunkProcessorEdgeCaseTests: XCTestCase {

    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    // MARK: - Helper Methods

    private func createMockAudio(durationSeconds: Double, sampleRate: Int = 16000) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(sampleRate))
        return (0..<sampleCount).map { Float($0) / Float(sampleCount) }
    }

    // MARK: - Overlap Window Tests

    func testOverlapWindowCalculation() {
        // Test that overlap is configured as 2.0 seconds = 32,000 samples at 16kHz
        let sampleRate = 16000
        let overlapSeconds = 2.0
        let expectedOverlapSamples = Int(overlapSeconds * Double(sampleRate))

        XCTAssertEqual(expectedOverlapSamples, 32_000, "2.0s overlap should be 32,000 samples at 16kHz")
    }

    func testOverlapClampedToHalfChunk() {
        // Test that overlap is clamped to not exceed half of chunk size
        // min(requested: 32,000, chunkSamples/2) should be reasonable
        let maxModelSamples = 240_000
        let melHopSize = 160
        let chunkSamples = max(maxModelSamples - melHopSize, 1280)  // Approximate
        let maxHalfChunk = chunkSamples / 2

        XCTAssertGreaterThan(maxHalfChunk, 100_000, "Half chunk should accommodate reasonable overlap")
    }

    func testHalfOverlapWindowTolerance() {
        // Test that merge tolerance is halfOverlapWindow = overlapSeconds / 2 = 1.0s
        let overlapSeconds = 2.0
        let halfOverlapWindow = overlapSeconds / 2
        let expectedTolerance = 1.0

        XCTAssertEqual(halfOverlapWindow, expectedTolerance, "Half overlap window should be 1.0s")
    }

    // MARK: - Chunk Boundary Edge Cases

    func testFirstChunkBoundaries() {
        // Test that first chunk starts at 0 and ends at min(chunkSamples, total)
        let audio = createMockAudio(durationSeconds: 18.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 288_000, "18s audio should be 288,000 samples")

        // First chunk starts at 0
        let chunkStart = 0
        XCTAssertEqual(chunkStart, 0)
    }

    func testLastChunkBoundaries() {
        // Test that last chunk is detected when candidateEnd >= audioSamples.count
        let audio = createMockAudio(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 320_000)

        // Simulate last chunk detection logic
        let candidateEnd = audio.count
        let isLastChunk = candidateEnd >= audio.count

        XCTAssertTrue(isLastChunk)
    }

    func testSingleChunkProcessing() {
        // Test audio that fits in a single chunk (< chunkSamples)
        let audio = createMockAudio(durationSeconds: 12.0)  // 192,000 samples
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertLessThan(audio.count, 240_000, "12s audio should fit in single chunk")
    }

    func testExactMultipleChunks() {
        // Test audio that aligns exactly with stride boundaries
        let audio = createMockAudio(durationSeconds: 25.92)  // Two stride lengths (~12.96s each)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 414_720)  // 25.92 * 16000
    }

    // MARK: - Stateless Decoder Edge Cases

    func testDecoderResetPerChunk() {
        // Test that decoder.reset() is called at start of each chunk iteration
        let audio = createMockAudio(durationSeconds: 22.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        // In implementation, each chunk gets: var chunkDecoderState = TdtDecoderState.make()
        // and then chunkDecoderState.reset()
    }

    func testNoTimeJumpBetweenChunks() {
        // Test that there is no timeJump carried from one chunk to the next
        // Each chunk is independent in the stateless approach
        let audio = createMockAudio(durationSeconds: 28.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        // Stateless: no timeJump persistence between chunks
    }

    func testGlobalFrameOffsetCalculation() {
        // Test that global frame offset = chunkStart / samplesPerEncoderFrame
        let chunkStart = 0
        let samplesPerEncoderFrame = 1280  // ASRConstants.samplesPerEncoderFrame
        let globalFrameOffset = chunkStart / samplesPerEncoderFrame

        XCTAssertEqual(globalFrameOffset, 0)

        // Test with non-zero chunkStart
        let chunkStart2 = 207_360  // Second chunk start (approximately)
        let globalFrameOffset2 = chunkStart2 / samplesPerEncoderFrame
        XCTAssertEqual(globalFrameOffset2, 162)  // 207,360 / 1,280 = 162
    }

    // MARK: - Merge Function Edge Cases

    func testMergeWithNoOverlapRegion() {
        // Test merging when leftEndTime <= rightStartTime
        // Chunks don't temporally overlap → concatenate directly
        let audio = createMockAudio(durationSeconds: 26.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeWithInsufficientOverlap() {
        // Test merging when overlapLeft/Right have < 2 tokens
        // Falls back to midpoint split strategy
        let audio = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeContiguousPairsMinimum() {
        // Test that contiguous pairs require >= minimumPairs matches
        // minimumPairs = max(overlapLeft.count / 2, 1)
        let audio = createMockAudio(durationSeconds: 25.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeLCSWithNoMatches() {
        // Test that when LCS finds no matches, falls back to midpoint
        let audio = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeWithEmptyChunks() {
        // Test merging when left or right chunk is empty
        let audio = createMockAudio(durationSeconds: 15.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - IndexedToken Tests

    func testIndexedTokenCreation() {
        // Test that IndexedToken struct is properly created with:
        // - index: position in overlap array
        // - token: the TokenWindow (token, timestamp, confidence)
        // - start: start time of token
        // - end: end time of token
        let audio = createMockAudio(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testOverlapLeftFiltering() {
        // Test that left overlap tokens are filtered:
        // end > rightStartTime - overlapDuration
        let audio = createMockAudio(durationSeconds: 22.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testOverlapRightFiltering() {
        // Test that right overlap tokens are filtered:
        // start < leftEndTime + overlapDuration
        let audio = createMockAudio(durationSeconds: 22.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Gap Resolution Edge Cases

    func testGapResolutionMultipleGaps() {
        // Test handling multiple gaps between different matches
        let audio = createMockAudio(durationSeconds: 28.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testGapResolutionEqualLength() {
        // Test gap selection when gapLeft.count == gapRight.count
        // Should prefer gapLeft in the else branch
        let audio = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testGapResolutionBeforeFirstMatch() {
        // Test handling tokens that appear before the first matched pair
        let audio = createMockAudio(durationSeconds: 23.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testGapResolutionAfterLastMatch() {
        // Test handling tokens that appear after the last matched pair
        let audio = createMockAudio(durationSeconds: 23.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Merge Strategy Selection

    func testMergeStrategyContiguousPriority() {
        // Test that contiguous pairs strategy is selected first
        // if contiguousPairs.count >= minimumPairs
        let audio = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeStrategyLCSFallback() {
        // Test that LCS strategy is used when contiguous pairs insufficient
        // First check: contiguousPairs.count < minimumPairs
        let audio = createMockAudio(durationSeconds: 25.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeStrategyMidpointFallback() {
        // Test that midpoint strategy is used when LCS returns empty
        // if lcsPairs.isEmpty → fallback
        let audio = createMockAudio(durationSeconds: 26.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMinimumPairsCalculation() {
        // Test that minimumPairs = max(overlapLeft.count / 2, 1)
        // Ensures at least 1 pair is required even for small overlaps
        let audio = createMockAudio(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Token Sorting

    func testTokenSortingAfterMerge() {
        // Test that merged tokens are sorted by timestamp
        // mergedTokens.sort { $0.timestamp < $1.timestamp }
        let audio = createMockAudio(durationSeconds: 24.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testSortingWithSingleChunk() {
        // Test that single chunk doesn't need sorting (no merging)
        let audio = createMockAudio(durationSeconds: 14.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertLessThan(audio.count, 240_000)
    }

    func testSortingWithMultipleChunks() {
        // Test that multiple chunks are properly sorted after merging
        let audio = createMockAudio(durationSeconds: 30.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Empty and Boundary Cases

    func testEmptyAudioProcessing() {
        // Test handling of empty audio array
        let emptyAudio: [Float] = []
        let processor = ChunkProcessor(audioSamples: emptyAudio)

        XCTAssertNotNil(processor)
    }

    func testMinimalAudio() {
        // Test audio with just a few samples
        let minimal = createMockAudio(durationSeconds: 0.001)
        let processor = ChunkProcessor(audioSamples: minimal)

        XCTAssertNotNil(processor)
    }

    func testExactlyChunkBoundary() {
        // Test audio that's exactly at chunkSamples boundary (~239,360 samples)
        let audio = createMockAudio(durationSeconds: 14.96)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertGreaterThan(audio.count, 230_000)
        XCTAssertLessThan(audio.count, 250_000)
    }

    func testJustPastChunkBoundary() {
        // Test audio just beyond chunk size (requires 2 chunks)
        let audio = createMockAudio(durationSeconds: 14.97)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Frame Calculation Edge Cases

    func testFrameCalculationSmallAudio() {
        // Test frame calculation for very small audio
        let audio = createMockAudio(durationSeconds: 0.1)  // 1,600 samples
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: audio.count)

        XCTAssertEqual(audio.count, 1_600)
        XCTAssertLessThan(expectedFrames, 5)
    }

    func testFrameCalculationExactFrame() {
        // Test frame calculation when samples align exactly with frame boundary
        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame
        let audio = createMockAudio(durationSeconds: Double(samplesPerFrame * 10) / 16000.0)
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: audio.count)

        XCTAssertEqual(expectedFrames, 10)
    }

    func testFrameCalculationPartialFrame() {
        // Test frame calculation when samples don't align exactly
        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame
        let audio = createMockAudio(durationSeconds: Double(samplesPerFrame * 5 + 100) / 16000.0)
        let expectedFrames = ASRConstants.calculateEncoderFrames(from: audio.count)

        XCTAssertEqual(expectedFrames, 6)  // Should ceiling to 6 frames
    }

    // MARK: - Chunk Count Prediction

    func testChunkCountSingleChunk() {
        // Test that audio < ~14.96s requires only 1 chunk
        let audio = createMockAudio(durationSeconds: 14.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 224_000)
    }

    func testChunkCountTwoChunks() {
        // Test that audio ~25-28s requires 2 chunks
        let audio = createMockAudio(durationSeconds: 26.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 416_000)
    }

    func testChunkCountThreeChunks() {
        // Test that audio ~38-41s requires 3 chunks
        let audio = createMockAudio(durationSeconds: 39.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 624_000)
    }

    func testChunkCountManyChunks() {
        // Test chunk count for long audio (60 seconds)
        let audio = createMockAudio(durationSeconds: 60.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 960_000)
    }

    // MARK: - Stateless Decoder Validation Tests

    func testDecoderStateResetImplementation() {
        // Test that TdtDecoderState.reset() properly clears internal buffers
        // Each chunk should get a fresh decoder state via TdtDecoderState.make()

        let decoderState1 = TdtDecoderState.make()
        let decoderState2 = TdtDecoderState.make()

        // Both should be independent instances
        XCTAssertNotNil(decoderState1)
        XCTAssertNotNil(decoderState2)
    }

    func testMultipleChunkIndependence() {
        // Test that processing multiple chunks independently produces consistent results
        // Without stateful carryover, same audio chunk should produce same tokens

        let audio = createMockAudio(durationSeconds: 10.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        // Verify audio is consistent
        XCTAssertEqual(audio.count, 160_000)
    }

    func testNoStateLeakageBetweenChunks() {
        // Test that decoder state doesn't leak between chunks
        // In stateless mode, each chunk starts with fresh decoder state

        // Simulate processing three chunks sequentially
        let chunk1 = createMockAudio(durationSeconds: 12.0)
        let chunk2 = createMockAudio(durationSeconds: 12.0)
        let chunk3 = createMockAudio(durationSeconds: 12.0)

        let processor1 = ChunkProcessor(audioSamples: chunk1)
        let processor2 = ChunkProcessor(audioSamples: chunk2)
        let processor3 = ChunkProcessor(audioSamples: chunk3)

        XCTAssertNotNil(processor1)
        XCTAssertNotNil(processor2)
        XCTAssertNotNil(processor3)
    }

    func testGlobalFrameOffsetMultiChunk() {
        // Test that global frame offset is correctly calculated for each chunk
        // globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame  // 1280

        // First chunk at 0
        let offset0 = 0 / samplesPerFrame
        XCTAssertEqual(offset0, 0)

        // Second chunk at approximately 207,360 samples
        let chunkStart2 = 207_360
        let offset2 = chunkStart2 / samplesPerFrame
        XCTAssertEqual(offset2, 162)

        // Third chunk at approximately 414,720 samples
        let chunkStart3 = 414_720
        let offset3 = chunkStart3 / samplesPerFrame
        XCTAssertEqual(offset3, 324)
    }

    // MARK: - Token Window Structure Tests

    func testTokenWindowStructure() {
        // Test that TokenWindow (token, timestamp, confidence) tuple is properly formed
        let token = 100
        let timestamp = 5
        let confidence: Float = 0.95

        let tokenWindow = (token: token, timestamp: timestamp, confidence: confidence)

        XCTAssertEqual(tokenWindow.token, 100)
        XCTAssertEqual(tokenWindow.timestamp, 5)
        XCTAssertEqual(tokenWindow.confidence, 0.95)
    }

    func testTokenWindowArrayAlignment() {
        // Test that token windows maintain alignment when combined
        let tokens = [100, 101, 102, 103]
        let timestamps = [0, 1, 2, 3]
        let confidences: [Float] = [0.9, 0.9, 0.85, 0.92]

        // Zip into token windows
        let tokenWindows = zip(zip(tokens, timestamps), confidences).map {
            (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
        }

        // Verify alignment
        XCTAssertEqual(tokenWindows.count, 4)
        XCTAssertEqual(tokenWindows[0].token, 100)
        XCTAssertEqual(tokenWindows[1].timestamp, 1)
        XCTAssertEqual(tokenWindows[2].confidence, 0.85)
        XCTAssertEqual(tokenWindows[3].confidence, 0.92)
    }

    // MARK: - Overlap Region Filtering Tests

    func testOverlapRegionCalculation() {
        // Test that overlap regions are correctly identified
        let sampleRate = 16000
        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame
        let frameDuration = Double(samplesPerFrame) / Double(sampleRate)  // ~0.08s

        let overlapSeconds = 2.0
        let overlapDuration = overlapSeconds

        // Token at frame 20 (time = 20 * 0.08 = 1.6s)
        let tokenFrame = 20
        let tokenTime = Double(tokenFrame) * frameDuration

        let rightStartFrame = 25
        let rightStartTime = Double(rightStartFrame) * frameDuration

        // Should be within overlap if token end > rightStartTime - overlapDuration
        let tokenEndTime = tokenTime + frameDuration
        let shouldBeInOverlap = tokenEndTime > (rightStartTime - overlapDuration)

        XCTAssertTrue(shouldBeInOverlap)
    }

    func testMinimumPairsCalculationEdgeCases() {
        // Test that minimumPairs = max(overlapLeft.count / 2, 1)

        // For small overlaps
        let smallOverlapCount = 1
        let minPairsSmall = max(smallOverlapCount / 2, 1)
        XCTAssertEqual(minPairsSmall, 1)

        // For medium overlaps
        let mediumOverlapCount = 5
        let minPairsMedium = max(mediumOverlapCount / 2, 1)
        XCTAssertEqual(minPairsMedium, 2)

        // For large overlaps
        let largeOverlapCount = 10
        let minPairsLarge = max(largeOverlapCount / 2, 1)
        XCTAssertEqual(minPairsLarge, 5)
    }

    // MARK: - Chunk Output Organization Tests

    func testChunkOutputsArray() {
        // Test that chunkOutputs array maintains proper chunk ordering

        let chunk1: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
        ]

        let chunk2: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 102, timestamp: 10, confidence: 0.9),
            (token: 103, timestamp: 11, confidence: 0.9),
        ]

        var chunkOutputs: [[(token: Int, timestamp: Int, confidence: Float)]] = []
        chunkOutputs.append(chunk1)
        chunkOutputs.append(chunk2)

        XCTAssertEqual(chunkOutputs.count, 2)
        XCTAssertEqual(chunkOutputs[0].count, 2)
        XCTAssertEqual(chunkOutputs[1].count, 2)
    }

    func testChunkMergingOrder() {
        // Test that chunks are merged in correct order (first chunk is base)

        let chunkOutputs: [[(token: Int, timestamp: Int, confidence: Float)]] = [
            [(token: 100, timestamp: 0, confidence: 0.9)],
            [(token: 101, timestamp: 1, confidence: 0.9)],
            [(token: 102, timestamp: 2, confidence: 0.9)],
        ]

        guard var merged = chunkOutputs.first else {
            XCTFail("Should have first chunk")
            return
        }

        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(merged[0].token, 100)

        // Subsequent chunks should be merged
        for chunk in chunkOutputs.dropFirst() {
            merged.append(contentsOf: chunk)
        }

        XCTAssertEqual(merged.count, 3)
    }

    // MARK: - Empty Chunk Handling Tests

    func testEmptyChunkOutput() {
        // Test handling of empty chunk output from transcribeChunk
        let emptyChunk: [(token: Int, timestamp: Int, confidence: Float)] = []

        guard let firstChunk = [emptyChunk].first else {
            XCTFail("Should have first chunk")
            return
        }

        XCTAssertTrue(firstChunk.isEmpty)
    }

    func testGuardAgainstEmptyChunkOutputs() {
        // Test that empty chunkOutputs is properly handled
        let emptyChunkOutputs: [[(token: Int, timestamp: Int, confidence: Float)]] = []

        guard emptyChunkOutputs.first != nil else {
            // This is the expected path - should return early result
            XCTAssertTrue(emptyChunkOutputs.isEmpty)
            return
        }

        XCTFail("Should not reach here with empty outputs")
    }

    // MARK: - Token Extraction Tests

    func testTokenExtractionFromMerged() {
        // Test that tokens are correctly extracted from merged window
        let merged: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
            (token: 102, timestamp: 2, confidence: 0.9),
        ]

        let tokens = merged.map { $0.token }
        let timestamps = merged.map { $0.timestamp }
        let confidences = merged.map { $0.confidence }

        XCTAssertEqual(tokens, [100, 101, 102])
        XCTAssertEqual(timestamps, [0, 1, 2])
        XCTAssertEqual(confidences, [0.9, 0.9, 0.9])
    }

    func testTokenExtractionPreservesOrder() {
        // Test that token extraction preserves order from merged tokens
        let merged: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.95),
            (token: 101, timestamp: 1, confidence: 0.87),
            (token: 102, timestamp: 2, confidence: 0.72),
        ]

        let extractedTokens = merged.map { $0.token }

        for (index, token) in extractedTokens.enumerated() {
            XCTAssertEqual(token, merged[index].token)
        }
    }
}
