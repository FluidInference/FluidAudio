import AVFoundation
@testable import FluidAudio
import XCTest

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
        XCTAssertEqual(manager.volatileTranscript, "")
        XCTAssertEqual(manager.confirmedTranscript, "")
        XCTAssertEqual(manager.source, .microphone)
    }

    func testInitializationWithCustomConfig() async throws {
        let config = StreamingAsrConfig(
            confirmationThreshold: 0.9,
            chunkDuration: 5.0,
            enableDebug: true
        )
        let manager = StreamingAsrManager(config: config)
        XCTAssertEqual(manager.volatileTranscript, "")
        XCTAssertEqual(manager.confirmedTranscript, "")
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
        XCTAssertEqual(lowLatencyConfig.chunkDuration, 5.0)
        XCTAssertFalse(lowLatencyConfig.enableDebug)

        // Test high accuracy config
        let highAccuracyConfig = StreamingAsrConfig.highAccuracy
        XCTAssertEqual(highAccuracyConfig.confirmationThreshold, 0.9)
        XCTAssertEqual(highAccuracyConfig.chunkDuration, 10.0)
        XCTAssertFalse(highAccuracyConfig.enableDebug)

        // Test legacy config
        let legacyConfig = StreamingAsrConfig.legacy
        XCTAssertEqual(legacyConfig.confirmationThreshold, 0.85)
        XCTAssertEqual(legacyConfig.chunkDuration, 2.5)
        XCTAssertFalse(legacyConfig.enableDebug)
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

    func testStreamAudioBuffering() async throws {
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
        manager.streamAudio(buffer)
    }

    func testTranscriptionUpdatesStream() async throws {
        let manager = StreamingAsrManager()

        // Create expectation for updates
        let expectation = self.expectation(description: "Receive transcription updates")
        expectation.isInverted = true  // We don't expect updates without starting

        Task {
            for await _ in manager.transcriptionUpdates {
                expectation.fulfill()
                break
            }
        }

        await fulfillment(of: [expectation], timeout: 0.5)
    }

    func testResetFunctionality() async throws {
        let manager = StreamingAsrManager()

        // Simulate some state (these would normally be set during transcription)
        // Note: We can't directly set these as they're private(set), but we can test reset behavior

        // Reset
        try await manager.reset()

        // Verify state is cleared
        XCTAssertEqual(manager.volatileTranscript, "")
        XCTAssertEqual(manager.confirmedTranscript, "")
    }

    func testCancelFunctionality() async throws {
        let manager = StreamingAsrManager()

        // Start a transcription updates listener
        let task = Task {
            for await _ in manager.transcriptionUpdates {
                // Should terminate when cancelled
            }
        }

        // Cancel the manager
        await manager.cancel()

        // Task should complete
        await task.value
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
        let manager = StreamingAsrManager()

        // Default should be microphone
        XCTAssertEqual(manager.source, .microphone)

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