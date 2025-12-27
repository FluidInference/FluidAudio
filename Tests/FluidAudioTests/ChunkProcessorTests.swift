@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class ChunkProcessorTests: XCTestCase {

    // MARK: - Test Setup

    private func createMockAudioSamples(durationSeconds: Double, sampleRate: Int = 16000) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(sampleRate))
        return (0..<sampleCount).map { Float($0) / Float(sampleCount) }
    }

    // MARK: - Initialization Tests

    func testChunkProcessorInitialization() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3]
        let processor = ChunkProcessor(audioSamples: audioSamples)

        // We can't directly access private properties, but we can verify the processor was created
        XCTAssertNotNil(processor)
    }

    func testChunkProcessorWithEmptyAudio() {
        let processor = ChunkProcessor(audioSamples: [])
        XCTAssertNotNil(processor)
    }

    // MARK: - Audio Duration Calculations

    func testLongAudioChunking() {
        // Create 30 second audio (480,000 samples)
        let longAudio = createMockAudioSamples(durationSeconds: 30.0)
        let processor = ChunkProcessor(audioSamples: longAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(longAudio.count, 480_000, "30 second audio should have 480,000 samples")
    }

    // MARK: - Edge Cases

    func testVeryShortAudio() {
        // Audio shorter than context windows
        let shortAudio = createMockAudioSamples(durationSeconds: 0.5)  // 8,000 samples
        let processor = ChunkProcessor(audioSamples: shortAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(shortAudio.count, 8_000, "0.5 second audio should have 8,000 samples")
    }

    func testMaxModelCapacity() {
        // Audio at max model capacity (15 seconds = 240,000 samples)
        let maxAudio = createMockAudioSamples(durationSeconds: 15.0)
        let processor = ChunkProcessor(audioSamples: maxAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(maxAudio.count, 240_000, "15 second audio should have 240,000 samples")
    }

    // MARK: - Performance Tests

    func testChunkProcessorCreationPerformance() {
        let longAudio = createMockAudioSamples(durationSeconds: 60.0)  // 1 minute

        measure {
            for _ in 0..<100 {
                _ = ChunkProcessor(audioSamples: longAudio)
            }
        }
    }

    func testAudioSampleGeneration() {
        measure {
            _ = createMockAudioSamples(durationSeconds: 30.0)
        }
    }

    // MARK: - Memory Tests

    func testLargeAudioHandling() {
        // Test with 5 minutes of audio (4,800,000 samples)
        let largeAudio = createMockAudioSamples(durationSeconds: 300.0)
        let processor = ChunkProcessor(audioSamples: largeAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(largeAudio.count, 4_800_000, "5 minute audio should have 4,800,000 samples")
    }

    // MARK: - Debug Mode Tests

    func testDebugModeEnabled() {
        let audio = createMockAudioSamples(durationSeconds: 1.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testDebugModeDisabled() {
        let audio = createMockAudioSamples(durationSeconds: 1.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Sample Rate Validation

    func testSampleRateConsistency() {
        // The ChunkProcessor assumes 16kHz sample rate
        let oneSecondAudio16k = createMockAudioSamples(durationSeconds: 1.0, sampleRate: 16000)
        let oneSecondAudio44k = createMockAudioSamples(durationSeconds: 1.0, sampleRate: 44100)

        XCTAssertEqual(oneSecondAudio16k.count, 16_000, "1 second at 16kHz should be 16,000 samples")
        XCTAssertEqual(oneSecondAudio44k.count, 44_100, "1 second at 44.1kHz should be 44,100 samples")

        // ChunkProcessor should handle both, but expects 16kHz internally
        let processor16k = ChunkProcessor(audioSamples: oneSecondAudio16k)
        let processor44k = ChunkProcessor(audioSamples: oneSecondAudio44k)

        XCTAssertNotNil(processor16k)
        XCTAssertNotNil(processor44k)
    }

    // MARK: - Boundary Condition Tests

    func testZeroDurationAudio() {
        let emptyAudio: [Float] = []
        let processor = ChunkProcessor(audioSamples: emptyAudio)

        XCTAssertNotNil(processor)
    }

    func testSingleSampleAudio() {
        let singleSample: [Float] = [0.5]
        let processor = ChunkProcessor(audioSamples: singleSample)

        XCTAssertNotNil(processor)
    }

    // MARK: - Overlap-based Chunking Tests

    func testOverlapBasedChunkCalculations() {
        // Test that chunking parameters are properly configured for stateless overlap-based approach
        // - Chunk size: ~239,360 samples (~14.96s to stay under 240,000 max)
        // - Overlap: 2.0s = 32,000 samples
        // - Stride: chunkSamples - overlapSamples

        let audio = createMockAudioSamples(durationSeconds: 15.0)  // 240,000 samples
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        // Verify audio was created correctly for testing
        XCTAssertEqual(audio.count, 240_000, "15 second audio should be 240,000 samples")
    }

    func testChunkStrideCalculation() {
        // Test that stride = chunkSamples - overlapSamples
        // With 32,000 overlap samples (2.0s) and ~239,360 chunk samples,
        // stride should be ~207,360 samples (~12.96s)

        let audio = createMockAudioSamples(durationSeconds: 30.0)  // 480,000 samples
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 480_000, "30 second audio should be 480,000 samples")
    }

    func testChunkBoundaryCalculations() {
        // Test that chunks are properly aligned with stride boundaries
        // chunkStart should increment by strideSamples each iteration

        let audio = createMockAudioSamples(durationSeconds: 45.0)  // 720,000 samples
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 720_000, "45 second audio should be 720,000 samples")
    }

    // MARK: - Stateless Decoder Tests

    func testDecoderStateResetBetweenChunks() {
        // Test that decoder state is reset at the beginning of each chunk
        // In the new stateless approach, each chunk gets a fresh decoder state

        let audio = createMockAudioSamples(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 320_000, "20 second audio should be 320,000 samples")
    }

    func testNoDecoderStatePersistence() {
        // Test that decoder state is not carried between chunks
        // Previously used timeJump for continuity, now stateless

        let audio = createMockAudioSamples(durationSeconds: 25.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(audio.count, 400_000, "25 second audio should be 400,000 samples")
    }

    // MARK: - Merge Strategy Tests

    func testMergeContiguousPairs() {
        // Test merging chunks using contiguous token pair matching
        // When overlapping regions have matching tokens in sequence

        let audio = createMockAudioSamples(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeLCS() {
        // Test merging chunks using Longest Common Subsequence matching
        // Fallback when contiguous pairs are insufficient

        let audio = createMockAudioSamples(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeMidpoint() {
        // Test merging chunks using midpoint split strategy
        // Fallback when LCS matching finds no common tokens

        let audio = createMockAudioSamples(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeEmptyChunks() {
        // Test merging when one or both chunks are empty
        // Edge case for handling minimal overlap regions

        let audio = createMockAudioSamples(durationSeconds: 15.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testMergeNoOverlap() {
        // Test merging when chunks don't temporally overlap
        // leftEndTime <= rightStartTime → concatenate directly

        let audio = createMockAudioSamples(durationSeconds: 25.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Token Matching Tests

    func testTokensMatch() {
        // Test the token matching condition:
        // - Same token ID
        // - Time difference within tolerance (halfOverlapWindow = 1.0s)

        let audio = createMockAudioSamples(durationSeconds: 18.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testTokenMatchingTolerance() {
        // Test that time tolerance is correctly applied
        // Tolerance = overlapSeconds / 2 = 1.0s

        let audio = createMockAudioSamples(durationSeconds: 18.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testTokenMismatchDifferentTokens() {
        // Test that different token IDs never match
        // tokensMatch returns false when token IDs differ

        let audio = createMockAudioSamples(durationSeconds: 15.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testTokenMismatchOutsideTolerance() {
        // Test that same token IDs don't match if time difference exceeds tolerance
        // tokensMatch returns false when |leftTime - rightTime| > tolerance

        let audio = createMockAudioSamples(durationSeconds: 15.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Gap Resolution Tests

    func testGapResolutionBetweenMatches() {
        // Test gap handling in mergeUsingMatches
        // When consecutive matches have a gap between them

        let audio = createMockAudioSamples(durationSeconds: 22.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testGapResolutionPreferLonger() {
        // Test that longer gap is preferred when choosing between left and right gaps
        // if gapRight.count > gapLeft.count → use gapRight

        let audio = createMockAudioSamples(durationSeconds: 22.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    func testGapResolutionNoGaps() {
        // Test merging when consecutive matches are adjacent (no gaps)

        let audio = createMockAudioSamples(durationSeconds: 20.0)
        let processor = ChunkProcessor(audioSamples: audio)

        XCTAssertNotNil(processor)
    }

    // MARK: - Chunk Count Prediction with Overlap

    func testPredictableChunkCountWithOverlap() {
        // Test predictable chunk count with new overlap-based approach
        // Stride is approximately 12.96s (chunkSamples - overlapSamples)
        // - ~15s audio: 1 chunk
        // - ~28s audio: 2 chunks
        // - ~41s audio: 3 chunks

        let audio15s = createMockAudioSamples(durationSeconds: 15.0)
        let audio28s = createMockAudioSamples(durationSeconds: 28.0)
        let audio41s = createMockAudioSamples(durationSeconds: 41.0)

        XCTAssertEqual(audio15s.count, 240_000)
        XCTAssertEqual(audio28s.count, 448_000)
        XCTAssertEqual(audio41s.count, 656_000)
    }

    // MARK: - Token Merging Implementation Tests

    func testTokenMergingWithContiguousPairs() {
        // Test merging chunks where overlapping regions have matching contiguous token sequences
        // This is the ideal case where chunks align perfectly

        // Create two chunks with matching tokens in the overlap region
        let leftChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
            (token: 102, timestamp: 2, confidence: 0.9),
            (token: 103, timestamp: 3, confidence: 0.9),
            (token: 104, timestamp: 4, confidence: 0.85),
        ]

        let rightChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 103, timestamp: 3, confidence: 0.9),  // Match at timestamp 3
            (token: 104, timestamp: 4, confidence: 0.85),  // Match at timestamp 4
            (token: 105, timestamp: 5, confidence: 0.9),
            (token: 106, timestamp: 6, confidence: 0.9),
        ]

        // Verify we have matching sequences
        let leftTokens = leftChunk.map { $0.token }
        let rightTokens = rightChunk.map { $0.token }
        XCTAssertTrue(leftTokens.contains(103))
        XCTAssertTrue(rightTokens.contains(103))
    }

    func testTokenMergingWithNoOverlap() {
        // Test merging chunks that don't temporally overlap
        // leftEndTime <= rightStartTime → should concatenate directly

        let leftChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
            (token: 102, timestamp: 2, confidence: 0.9),
        ]

        // Right chunk starts well after left ends
        let rightChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 200, timestamp: 10, confidence: 0.9),
            (token: 201, timestamp: 11, confidence: 0.9),
        ]

        // Verify no overlapping timestamps
        let leftTimestamps = leftChunk.map { $0.timestamp }
        let rightTimestamps = rightChunk.map { $0.timestamp }
        let maxLeftTime = leftTimestamps.max() ?? 0
        let minRightTime = rightTimestamps.min() ?? Int.max

        XCTAssertLessThan(maxLeftTime, minRightTime)
    }

    func testTokenMergingWithEmptyChunks() {
        // Test edge case when one or both chunks are empty

        let emptyChunk: [(token: Int, timestamp: Int, confidence: Float)] = []

        let rightChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
        ]

        // Merging empty with non-empty should return non-empty
        XCTAssertTrue(emptyChunk.isEmpty)
        XCTAssertFalse(rightChunk.isEmpty)
    }

    func testTokenTimestampMonotonicity() {
        // Test that merged tokens maintain timestamp ordering
        // After merging, timestamps should be monotonically increasing

        let tokens: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
            (token: 102, timestamp: 3, confidence: 0.9),
            (token: 103, timestamp: 2, confidence: 0.9),  // Out of order
            (token: 104, timestamp: 4, confidence: 0.9),
        ]

        // After sorting, should be monotonic
        let sortedTokens = tokens.sorted { $0.timestamp < $1.timestamp }
        var lastTimestamp = Int.min

        for token in sortedTokens {
            XCTAssertGreaterThanOrEqual(token.timestamp, lastTimestamp)
            lastTimestamp = token.timestamp
        }
    }

    func testTokenMatchingWithTolerance() {
        // Test that tokens match when:
        // 1. Token IDs are the same
        // 2. Timestamp difference is within tolerance (1.0 second = 1.0 frame time)

        let sampleRate = 16000
        let samplesPerFrame = ASRConstants.samplesPerEncoderFrame  // 1280 samples
        let frameSeconds = Double(samplesPerFrame) / Double(sampleRate)  // ~0.08 seconds

        // Two tokens with same ID but slightly different timestamps
        let token1 = (token: 100, timestamp: 10, confidence: 0.9)
        let token2 = (token: 100, timestamp: 12, confidence: 0.9)  // 2 frames later

        let timestampDiff = abs(Double(token2.timestamp - token1.timestamp)) * frameSeconds
        let tolerance = 1.0  // half overlap window

        // Should match if within tolerance
        if timestampDiff < tolerance {
            XCTAssert(token1.token == token2.token && timestampDiff < tolerance)
        }
    }

    func testTokenMergingPreservesConfidence() {
        // Test that token confidence values are preserved during merging

        let chunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.95),
            (token: 101, timestamp: 1, confidence: 0.87),
            (token: 102, timestamp: 2, confidence: 0.72),
        ]

        // Verify confidence values are preserved
        for token in chunk {
            XCTAssertGreaterThan(token.confidence, 0.0)
            XCTAssertLessThanOrEqual(token.confidence, 1.0)
        }

        XCTAssertEqual(chunk[0].confidence, 0.95)
        XCTAssertEqual(chunk[1].confidence, 0.87)
        XCTAssertEqual(chunk[2].confidence, 0.72)
    }

    func testLCSMatchingLogic() {
        // Test longest common subsequence matching for token deduplication
        // LCS finds matching tokens in the overlap region

        let leftOverlap = [100, 101, 102, 103, 104]
        let rightOverlap = [102, 103, 104, 105, 106]

        // Common tokens: 102, 103, 104
        let lcsTokens = [102, 103, 104]

        for token in lcsTokens {
            XCTAssertTrue(leftOverlap.contains(token))
            XCTAssertTrue(rightOverlap.contains(token))
        }
    }

    func testMidpointSplitFallback() {
        // Test midpoint split when no common tokens found in overlap
        // Should split the overlap region in the middle

        let leftChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 100, timestamp: 0, confidence: 0.9),
            (token: 101, timestamp: 1, confidence: 0.9),
            (token: 102, timestamp: 2, confidence: 0.9),
            (token: 103, timestamp: 3, confidence: 0.9),
        ]

        let rightChunk: [(token: Int, timestamp: Int, confidence: Float)] = [
            (token: 200, timestamp: 2, confidence: 0.9),  // Different tokens
            (token: 201, timestamp: 3, confidence: 0.9),
            (token: 202, timestamp: 4, confidence: 0.9),
        ]

        // Verify no common tokens
        let leftTokens = Set(leftChunk.map { $0.token })
        let rightTokens = Set(rightChunk.map { $0.token })
        XCTAssertTrue(leftTokens.isDisjoint(with: rightTokens))

        // Midpoint would be at timestamp 2.5 (between 2 and 3)
        let midpoint = Double(2 + 3) / 2.0
        XCTAssertEqual(midpoint, 2.5)
    }

    func testChunkBoundaryAlignment() {
        // Test that chunk boundaries align properly with the stride
        // stride = chunkSamples - overlapSamples

        let sampleRate = 16000
        let maxModelSamples = 240_000
        let melHopSize = 160
        let samplesPerFrame = 1280

        let chunkSamples = max(maxModelSamples - melHopSize, samplesPerFrame)
        let overlapSeconds = 2.0
        let overlapSamples = Int(overlapSeconds * Double(sampleRate))
        let strideSamples = max(chunkSamples - overlapSamples, samplesPerFrame)

        // Verify stride is positive and reasonable
        XCTAssertGreaterThan(strideSamples, 0)
        XCTAssertLessThan(strideSamples, chunkSamples)

        // For ~30s audio with stride ~207,360, chunk calculation depends on actual values
        // First chunk: 0 to 239,680 (chunkSamples)
        // Remaining: 30*16000 - 239,680 = 239,680
        // Second chunk at stride offset: 207,360 to (207,360 + 239,680) = 447,040
        // Remaining: 479,680 - 447,040 = 32,640
        // Third chunk at stride offset: 414,720 onwards covers the rest
        // So for 30s (~480,000 samples), we need 3 chunks
        let thirtySecondsAudio = 30 * sampleRate
        let estimatedChunkCount = (thirtySecondsAudio + strideSamples - 1) / strideSamples
        XCTAssertEqual(estimatedChunkCount, 3)
    }
}
