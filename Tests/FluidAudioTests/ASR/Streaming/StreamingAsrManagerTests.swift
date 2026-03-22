import AVFoundation
import XCTest

@testable import FluidAudio

final class StreamingAsrManagerTests: XCTestCase {
    override func setUp() {
        super.setUp()
    }

    override func tearDown() {
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInitializationWithDefaultConfig() async throws {
        let manager = StreamingAsrManager()
        let volatileTranscript = await manager.volatileTranscript
        let confirmedTranscript = await manager.confirmedTranscript
        let source = await manager.source

        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(confirmedTranscript, "")
        XCTAssertEqual(source, .microphone)
    }

    func testInitializationWithCustomConfig() async throws {
        let config = StreamingAsrConfig(
            confirmationThreshold: 0.9,
            chunkDuration: 10.0,
        )
        let manager = StreamingAsrManager(config: config)
        let volatileTranscript = await manager.volatileTranscript
        let confirmedTranscript = await manager.confirmedTranscript

        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(confirmedTranscript, "")
    }

    // MARK: - Configuration Tests

    func testConfigPresets() {
        // Test default config
        let defaultConfig = StreamingAsrConfig.default
        XCTAssertEqual(defaultConfig.confirmationThreshold, 0.85)
        XCTAssertEqual(defaultConfig.chunkDuration, 15.0)
    }

    func testConfigCalculatedProperties() {
        let config = StreamingAsrConfig(chunkDuration: 5.0)
        XCTAssertEqual(config.bufferCapacity, 240000)  // 15 seconds at 16kHz
        XCTAssertEqual(config.chunkSizeInSamples, 80000)  // 5 seconds at 16kHz

        // Test ASR config generation
        let asrConfig = config.asrConfig
        XCTAssertEqual(asrConfig.sampleRate, 16000)
        XCTAssertNotNil(asrConfig.tdtConfig)
    }

    // MARK: - Stream Management Tests

    func testAudioBufferBasicOperations() async throws {
        let buffer = AudioBuffer(capacity: 1000)

        // Test initial state
        let initialChunk = await buffer.getChunk(size: 100)
        XCTAssertNil(initialChunk, "Buffer should be empty initially")

        // Test appending samples
        let samples: [Float] = Array(repeating: 1.0, count: 500)
        try await buffer.append(samples)

        // Test getting chunk
        let chunk = await buffer.getChunk(size: 100)
        XCTAssertNotNil(chunk, "Should be able to get chunk after appending")
        XCTAssertEqual(chunk?.count, 100, "Chunk should have correct size")
        XCTAssertEqual(chunk?.first, 1.0, "Chunk should contain correct values")
    }

    func testAudioBufferOverflow() async throws {
        let buffer = AudioBuffer(capacity: 100)

        // Fill buffer to capacity
        let samples1: [Float] = Array(repeating: 1.0, count: 50)
        try await buffer.append(samples1)

        // Add more samples that would overflow
        let samples2: [Float] = Array(repeating: 2.0, count: 80)
        try await buffer.append(samples2)  // Should handle overflow gracefully

        // Verify buffer still works
        let chunk = await buffer.getChunk(size: 50)
        XCTAssertNotNil(chunk, "Buffer should still work after overflow")
        XCTAssertEqual(chunk?.count, 50, "Chunk should have correct size")

        // After overflow, the buffer now prioritizes new samples and adjusts read position
        // to start from the newly added samples, so first sample should be 2.0
        XCTAssertEqual(chunk?.first, 2.0, "Should contain newer samples after overflow")

        // All samples in the chunk should be from the new samples (2.0)
        XCTAssertTrue(chunk!.allSatisfy { $0 == 2.0 }, "All samples should be new samples (2.0) after overflow")
    }

    func testStreamAudioBuffering() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    func testTranscriptionUpdatesStream() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    func testResetFunctionality() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    func testCancelFunctionality() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    // MARK: - Update Structure Tests

    func testStreamingTranscriptionUpdateCreation() {
        let update = StreamingTranscriptionUpdate(
            text: "Hello world",
            isConfirmed: true,
            confidence: 0.95,
            timestamp: Date()
        )

        XCTAssertEqual(update.text, "Hello world")
        XCTAssertTrue(update.isConfirmed)
        XCTAssertEqual(update.confidence, 0.95)
        XCTAssertNotNil(update.timestamp)
        XCTAssertTrue(update.tokenIds.isEmpty)
        XCTAssertTrue(update.tokenTimings.isEmpty)
        XCTAssertTrue(update.tokens.isEmpty)
    }

    func testApplyGlobalFrameOffset() {
        let baseTimestamps = [0, 5, 10]
        let offsetSamples = 3 * ASRConstants.samplesPerEncoderFrame  // 3 frames of left context

        let adjusted = StreamingAsrManager.applyGlobalFrameOffset(
            to: baseTimestamps,
            windowStartSample: offsetSamples
        )

        XCTAssertEqual(adjusted, [3, 8, 13], "Timestamps should be shifted by frame offset")

        let zeroOffset = StreamingAsrManager.applyGlobalFrameOffset(
            to: baseTimestamps,
            windowStartSample: 0
        )
        XCTAssertEqual(zeroOffset, baseTimestamps, "Zero offset should preserve timestamps")

        let emptyAdjusted = StreamingAsrManager.applyGlobalFrameOffset(to: [], windowStartSample: offsetSamples)
        XCTAssertTrue(emptyAdjusted.isEmpty, "Empty input should remain empty")
    }

    func testStreamingTranscriptionUpdateTokenMetadata() {
        let tokenTimings = [
            TokenTiming(token: "hello", tokenId: 1, startTime: 0.0, endTime: 0.32, confidence: 0.98),
            TokenTiming(token: "world", tokenId: 2, startTime: 0.32, endTime: 0.64, confidence: 0.97),
        ]

        let update = StreamingTranscriptionUpdate(
            text: "Hello world",
            isConfirmed: true,
            confidence: 0.95,
            timestamp: Date(),
            tokenIds: [1, 2],
            tokenTimings: tokenTimings
        )

        XCTAssertEqual(update.tokenIds, [1, 2])
        XCTAssertEqual(update.tokenTimings.count, 2)
        XCTAssertEqual(update.tokens, ["hello", "world"])
    }

    func testStreamingTranscriptionUpdateConfidence() {
        // Test low confidence update
        let lowConfUpdate = StreamingTranscriptionUpdate(
            text: "uncertain text",
            isConfirmed: false,
            confidence: 0.5,
            timestamp: Date()
        )
        XCTAssertFalse(lowConfUpdate.isConfirmed)
        XCTAssertLessThan(lowConfUpdate.confidence, 0.75)

        // Test high confidence update
        let highConfUpdate = StreamingTranscriptionUpdate(
            text: "certain text",
            isConfirmed: true,
            confidence: 0.95,
            timestamp: Date()
        )
        XCTAssertTrue(highConfUpdate.isConfirmed)
        XCTAssertGreaterThan(highConfUpdate.confidence, 0.85)
    }

    // MARK: - Audio Source Tests

    func testAudioSourceConfiguration() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    // MARK: - Custom Configuration Tests

    func testCustomConfigurationFactory() {
        let customConfig = StreamingAsrConfig.custom(
            chunkDuration: 7.5,
            confirmationThreshold: 0.8,
        )

        XCTAssertEqual(customConfig.chunkDuration, 7.5)
        XCTAssertEqual(customConfig.confirmationThreshold, 0.8)
    }

    // MARK: - Performance Tests

    func testChunkSizeCalculationPerformance() {
        measure {
            for duration in stride(from: 1.0, to: 20.0, by: 0.5) {
                let config = StreamingAsrConfig(chunkDuration: duration)
                _ = config.chunkSizeInSamples
                _ = config.bufferCapacity
            }
        }
    }
}
