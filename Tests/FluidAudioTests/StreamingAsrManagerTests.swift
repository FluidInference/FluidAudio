import AVFoundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
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
            enableDebug: true
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
        XCTAssertEqual(defaultConfig.chunkDuration, 10.0)
        XCTAssertFalse(defaultConfig.enableDebug)

        // Test low latency config
        let lowLatencyConfig = StreamingAsrConfig.lowLatency
        XCTAssertEqual(lowLatencyConfig.confirmationThreshold, 0.75)
        XCTAssertEqual(lowLatencyConfig.chunkDuration, 10.0)
        XCTAssertFalse(lowLatencyConfig.enableDebug)

        // Test high accuracy config
        let highAccuracyConfig = StreamingAsrConfig.highAccuracy
        XCTAssertEqual(highAccuracyConfig.confirmationThreshold, 0.9)
        XCTAssertEqual(highAccuracyConfig.chunkDuration, 10.0)
        XCTAssertFalse(highAccuracyConfig.enableDebug)

    }

    func testConfigCalculatedProperties() {
        let config = StreamingAsrConfig(chunkDuration: 5.0)
        XCTAssertEqual(config.bufferCapacity, 160000)  // 10 seconds at 16kHz
        XCTAssertEqual(config.chunkSizeInSamples, 80000)  // 5 seconds at 16kHz

        // Test ASR config generation
        let asrConfig = config.asrConfig
        XCTAssertEqual(asrConfig.sampleRate, 16000)
        XCTAssertEqual(asrConfig.chunkSizeMs, 5000)
        XCTAssertTrue(asrConfig.realtimeMode)
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
        let manager = StreamingAsrManager()

        // Create a test audio buffer
        let sampleRate = 16000.0
        let duration = 1.0
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!

        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(sampleRate * duration)
        )!
        buffer.frameLength = buffer.frameCapacity

        // Stream audio (should not throw)
        await manager.streamAudio(buffer)
    }

    func testTranscriptionUpdatesStream() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
        let manager = StreamingAsrManager()

        // Create expectation for updates
        let expectation = self.expectation(description: "Receive transcription updates")
        expectation.isInverted = true  // We don't expect updates without starting

        Task {
            for await _ in await manager.transcriptionUpdates {
                expectation.fulfill()
                break
            }
        }

        await fulfillment(of: [expectation], timeout: 0.5)
    }

    func testResetFunctionality() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
        let manager = StreamingAsrManager()

        // Simulate some state (these would normally be set during transcription)
        // Note: We can't directly set these as they're private(set), but we can test reset behavior

        // Reset
        try await manager.reset()

        // Verify state is cleared
        let volatileTranscript = await manager.volatileTranscript
        let confirmedTranscript = await manager.confirmedTranscript
        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(confirmedTranscript, "")
    }

    func testCancelFunctionality() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
        let manager = StreamingAsrManager()

        // Create a task that will be cancelled
        let expectation = self.expectation(description: "Stream cancelled")

        let task = Task {
            do {
                for await _ in await manager.transcriptionUpdates {
                    // Should not receive any updates
                }
                expectation.fulfill()
            } catch {
                // Task cancellation is expected
                expectation.fulfill()
            }
        }

        // Give the task a moment to start
        try await Task.sleep(nanoseconds: 100_000_000)  // 0.1 seconds

        // Cancel the manager
        await manager.cancel()

        // Cancel the task as well
        task.cancel()

        await fulfillment(of: [expectation], timeout: 1.0)
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
        let manager = StreamingAsrManager()

        // Default should be microphone
        let source = await manager.source
        XCTAssertEqual(source, .microphone)

        // Test with mock models (would need to be expanded for real testing)
        // This is a placeholder for when models can be mocked
    }

    // MARK: - Custom Configuration Tests

    func testCustomConfigurationFactory() {
        let customConfig = StreamingAsrConfig.custom(
            chunkDuration: 7.5,
            confirmationThreshold: 0.8,
            enableDebug: true
        )

        XCTAssertEqual(customConfig.chunkDuration, 7.5)
        XCTAssertEqual(customConfig.confirmationThreshold, 0.8)
        XCTAssertTrue(customConfig.enableDebug)
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
