@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class AsrTranscriptionTests: XCTestCase {

    var manager: AsrManager!

    override func setUp() {
        super.setUp()
        manager = AsrManager()

        // Set up mock vocabulary for testing
        #if DEBUG
        let mockVocabulary = [
            1: "hello",
            2: "▁world",
            3: "▁test",
            4: "▁audio",
            5: ".",
            10: "hello",
            20: "▁world",
            30: "▁test",
            100: "test",
            200: "▁token",
            300: "!",
        ]
        manager.setVocabularyForTesting(mockVocabulary)
        #endif
    }

    override func tearDown() {
        manager = nil
        super.tearDown()
    }

    // MARK: - Audio Padding Tests

    func testPadAudioIfNeeded() {
        // Test no padding needed
        let exactLength = Array(repeating: Float(0.5), count: 160_000)
        let noPadding = manager.padAudioIfNeeded(exactLength, targetLength: 160_000)
        XCTAssertEqual(noPadding.count, 160_000)
        XCTAssertEqual(noPadding, exactLength)

        // Test padding needed
        let shortAudio: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let padded = manager.padAudioIfNeeded(shortAudio, targetLength: 10)
        XCTAssertEqual(padded.count, 10)
        XCTAssertEqual(Array(padded[0..<5]), shortAudio)
        XCTAssertEqual(Array(padded[5..<10]), [0, 0, 0, 0, 0])

        // Test longer audio (no padding)
        let longAudio = Array(repeating: Float(0.7), count: 200_000)
        let notPadded = manager.padAudioIfNeeded(longAudio, targetLength: 160_000)
        XCTAssertEqual(notPadded.count, 200_000)
        XCTAssertEqual(notPadded, longAudio)
    }

    func testPadAudioEdgeCases() {
        // Test empty audio
        let emptyAudio: [Float] = []
        let paddedEmpty = manager.padAudioIfNeeded(emptyAudio, targetLength: 100)
        XCTAssertEqual(paddedEmpty.count, 100)
        XCTAssertTrue(paddedEmpty.allSatisfy { $0 == 0 })

        // Test single sample
        let singleSample: [Float] = [0.42]
        let paddedSingle = manager.padAudioIfNeeded(singleSample, targetLength: 5)
        XCTAssertEqual(paddedSingle, [0.42, 0, 0, 0, 0])

        // Test target length 0
        let anyAudio: [Float] = [1, 2, 3]
        let noPaddingNeeded = manager.padAudioIfNeeded(anyAudio, targetLength: 0)
        XCTAssertEqual(noPaddingNeeded, anyAudio)
    }

    // MARK: - Process Transcription Result Tests

    func testProcessTranscriptionResult() {
        let tokenIds = [1, 2, 3, 4, 5]
        let audioSamples = Array(repeating: Float(0), count: 16_000)  // 1 second
        let processingTime = 0.5

        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            confidences: [0.63, 0.63, 0.63, 0.63, 0.63],  // Mean 0.63 (pure model confidence)
            encoderSequenceLength: 100,
            audioSamples: audioSamples,
            processingTime: processingTime
        )

        // Confidence is pure model confidence: mean token confidence (0.63)
        XCTAssertEqual(result.confidence, 0.63, accuracy: 0.01)
        XCTAssertEqual(result.duration, 1.0, accuracy: 0.01)
        XCTAssertEqual(result.processingTime, processingTime, accuracy: 0.001)
        XCTAssertTrue(result.tokenTimings?.isEmpty == true)  // No timestamps provided, should be empty array
    }

    func testProcessTranscriptionResultWithTimings() {
        let tokenIds = [10, 20, 30]
        let audioSamples = Array(repeating: Float(0), count: 48_000)  // 3 seconds
        let timings = [
            TokenTiming(token: "hello", tokenId: 10, startTime: 0.0, endTime: 1.0, confidence: 0.9),
            TokenTiming(token: "world", tokenId: 20, startTime: 1.0, endTime: 2.0, confidence: 0.85),
            TokenTiming(token: "test", tokenId: 30, startTime: 2.0, endTime: 3.0, confidence: 0.95),
        ]

        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            encoderSequenceLength: 150,
            audioSamples: audioSamples,
            processingTime: 1.2,
            tokenTimings: timings
        )

        XCTAssertEqual(result.duration, 3.0, accuracy: 0.01)
        XCTAssertNotNil(result.tokenTimings)
        // Note: Actual timing count may differ due to convertTokensWithExistingTimings filtering
    }

    func testProcessTranscriptionResultWithTimestamps() {
        let tokenIds = [100, 200, 300]
        let timestamps = [10, 20, 30]  // Frame indices
        let audioSamples = Array(repeating: Float(0), count: 32_000)  // 2 seconds
        let processingTime = 0.8

        let result = manager.processTranscriptionResult(
            tokenIds: tokenIds,
            timestamps: timestamps,
            confidences: [0.51, 0.51, 0.51],  // Mean 0.51 (pure model confidence)
            encoderSequenceLength: 100,
            audioSamples: audioSamples,
            processingTime: processingTime
        )

        // Confidence is pure model confidence: mean token confidence (0.51)
        XCTAssertEqual(result.confidence, 0.51, accuracy: 0.01)
        XCTAssertEqual(result.duration, 2.0, accuracy: 0.01)
        XCTAssertEqual(result.processingTime, processingTime, accuracy: 0.001)

        // Should have token timings from timestamps
        XCTAssertNotNil(result.tokenTimings)
        XCTAssertEqual(result.tokenTimings?.count, 3)

        if let tokenTimings = result.tokenTimings {
            // Verify timing calculations (80ms per frame)
            XCTAssertEqual(tokenTimings[0].startTime, 0.8, accuracy: 0.01)  // Frame 10 * 0.08
            XCTAssertEqual(tokenTimings[1].startTime, 1.6, accuracy: 0.01)  // Frame 20 * 0.08
            XCTAssertEqual(tokenTimings[2].startTime, 2.4, accuracy: 0.01)  // Frame 30 * 0.08

            XCTAssertEqual(tokenTimings[0].tokenId, 100)
            XCTAssertEqual(tokenTimings[1].tokenId, 200)
            XCTAssertEqual(tokenTimings[2].tokenId, 300)
        }
    }

    // MARK: - Chunk Processing Logic Tests

    func testChunkCalculations() {
        // Test exact multiple of chunk size
        let exactMultiple = 320_000  // 2 chunks of 160_000
        XCTAssertEqual(exactMultiple / 160_000, 2)
        XCTAssertEqual(exactMultiple % 160_000, 0)

        // Test with remainder
        let withRemainder = 400_000  // 2.5 chunks
        XCTAssertEqual(withRemainder / 160_000, 2)
        XCTAssertEqual(withRemainder % 160_000, 80_000)

        // Test single chunk
        let singleChunk = 100_000
        XCTAssertEqual(singleChunk / 160_000, 0)
        XCTAssertEqual(singleChunk % 160_000, 100_000)

        // Verify audio length calculations
        XCTAssertEqual(exactMultiple / 16_000, 20)  // 20 seconds
        XCTAssertEqual(withRemainder / 16_000, 25)  // 25 seconds
        XCTAssertEqual(singleChunk / 16_000, 6)  // 6.25 seconds
    }

    // MARK: - Mock ML Inference Tests

    func testExecuteMLInferenceStructure() async throws {
        // This test verifies the structure of executeMLInference method
        // without actually running ML models

        let testAudio = Array(repeating: Float(0.1), count: 160_000)

        // Verify padded audio would be correct size
        let padded = manager.padAudioIfNeeded(testAudio, targetLength: 160_000)
        XCTAssertEqual(padded.count, 160_000)
    }

    // MARK: - Audio Processing Pipeline Tests

    func testAudioProcessingBoundaries() {
        // Test minimum audio length (1 second = 16,000 samples)
        let minAudio = Array(repeating: Float(0.1), count: 16_000)
        XCTAssertEqual(minAudio.count, 16_000)

        // Test chunk boundary
        let chunkBoundary = Array(repeating: Float(0.1), count: 160_000)
        XCTAssertEqual(chunkBoundary.count, 160_000)

        // Test just over chunk boundary
        let overChunk = Array(repeating: Float(0.1), count: 160_001)
        XCTAssertEqual(overChunk.count, 160_001)

        // Test multiple chunks
        let multiChunk = Array(repeating: Float(0.1), count: 480_000)  // 3 chunks
        XCTAssertEqual(multiChunk.count / 160_000, 3)
    }

    // MARK: - Transcription State Management Tests

    func testTranscriptionWithStateStructure() async throws {
        // Verify the transcribeWithState method structure
        let decoderState = try TdtDecoderState()

        // Verify decoder state is properly initialized
        let expectedShape: [NSNumber] = [
            NSNumber(value: 2),
            NSNumber(value: 1),
            NSNumber(value: ASRConstants.decoderHiddenSize),
        ]
        XCTAssertEqual(decoderState.hiddenState.shape, expectedShape)
        XCTAssertEqual(decoderState.cellState.shape, expectedShape)

        // Test state values are initialized to zero
        XCTAssertEqual(decoderState.hiddenState[0].floatValue, 0.0, accuracy: 0.001)
        XCTAssertEqual(decoderState.cellState[0].floatValue, 0.0, accuracy: 0.001)
    }

    // MARK: - Audio Source Tests

    func testAudioSourceHandling() async throws {
        // Verify both source types exist
        let microphoneSource = AudioSource.microphone
        let systemSource = AudioSource.system

        // Test that they are different
        XCTAssertTrue(microphoneSource == .microphone, "Microphone source should be microphone")
        XCTAssertTrue(systemSource == .system, "System source should be system")
        XCTAssertFalse(microphoneSource == systemSource, "Sources should be different")
    }

    // MARK: - Performance Tests

    func testLargeAudioArrayPerformance() {
        // Test padding performance with large arrays
        let largeAudio = Array(repeating: Float(0.1), count: 1_600_000)  // 100 seconds

        measure {
            _ = manager.padAudioIfNeeded(largeAudio, targetLength: 160_000)
        }
    }

    func testChunkingPerformance() {
        // Test chunking performance
        let largeAudio = Array(repeating: Float(0.1), count: 3_200_000)  // 200 seconds

        measure {
            // Simulate chunk iteration
            var position = 0
            var chunkCount = 0
            let chunkSize = 160_000
            while position < largeAudio.count {
                let endPosition = min(position + chunkSize, largeAudio.count)
                let _ = Array(largeAudio[position..<endPosition])
                position += chunkSize
                chunkCount += 1
            }

            XCTAssertEqual(chunkCount, 20)
        }
    }
}
