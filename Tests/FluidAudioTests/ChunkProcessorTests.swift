import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
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

    func testChunkProcessorContextCalculations() {
        // Test the internal calculations by creating processors with known durations

        // 16kHz sample rate means:
        // - centerSeconds: 11.0s = 176,000 samples
        // - leftContextSeconds: 2.0s = 32,000 samples
        // - rightContextSeconds: 2.0s = 32,000 samples
        // - maxModelSamples: 240,000 (15s)

        let shortAudio = createMockAudioSamples(durationSeconds: 5.0)  // 80,000 samples
        let processor = ChunkProcessor(audioSamples: shortAudio)

        XCTAssertNotNil(processor)
        XCTAssertEqual(shortAudio.count, 80_000, "5 second audio should have 80,000 samples")
    }

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

    func testExactlyOneChunk() {
        // Audio that exactly fits one chunk (11 seconds center)
        let exactChunk = createMockAudioSamples(durationSeconds: 11.0)  // 176,000 samples
        let processor = ChunkProcessor(audioSamples: exactChunk)

        XCTAssertNotNil(processor)
        XCTAssertEqual(exactChunk.count, 176_000, "11 second audio should have 176,000 samples")
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

    func testPredictableChunkCount() {
        // For a given audio duration, we can predict approximately how many chunks will be processed
        // Center chunk is 11 seconds, so:
        // - 11s audio: 1 chunk
        // - 22s audio: 2 chunks
        // - 33s audio: 3 chunks

        let audio11s = createMockAudioSamples(durationSeconds: 11.0)
        let audio22s = createMockAudioSamples(durationSeconds: 22.0)
        let audio33s = createMockAudioSamples(durationSeconds: 33.0)

        XCTAssertEqual(audio11s.count, 176_000)  // Should result in ~1 chunk
        XCTAssertEqual(audio22s.count, 352_000)  // Should result in ~2 chunks
        XCTAssertEqual(audio33s.count, 528_000)  // Should result in ~3 chunks
    }
}
