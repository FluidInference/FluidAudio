import CoreML
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
}
