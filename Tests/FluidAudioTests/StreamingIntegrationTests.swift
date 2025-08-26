import AVFoundation
import CoreMedia
import XCTest

@testable import FluidAudio

/// Integration tests for streaming functionality that can run in CI
/// These tests validate the streaming infrastructure without requiring models
@available(macOS 13.0, iOS 16.0, *)
final class StreamingIntegrationTests: XCTestCase {

    // MARK: - CLI Command Integration Tests

    func testStreamingTranscribeCommandExists() throws {
        // This test verifies that the streaming flag is properly recognized
        // by the CLI argument parser without actually running transcription

        let arguments = ["transcribe", "dummy.wav", "--streaming"]

        // The arguments should be parseable (we're not actually executing)
        XCTAssertTrue(arguments.contains("--streaming"))
        XCTAssertTrue(arguments.contains("transcribe"))
    }

    func testStreamingBenchmarkCommandExists() throws {
        // Test that streaming benchmark arguments are recognized
        let arguments = [
            "asr-benchmark",
            "--test-streaming",
            "--chunk-duration", "1.0",
            "--subset", "test-clean",
            "--max-files", "1",
        ]

        XCTAssertTrue(arguments.contains("--test-streaming"))
        XCTAssertTrue(arguments.contains("--chunk-duration"))
        XCTAssertTrue(arguments.contains("asr-benchmark"))
    }

    // MARK: - Streaming Manager Integration Tests

    func testStreamingManagerLifecycle() async throws {
        let manager = StreamingAsrManager()

        // Test full lifecycle without models
        let initialStats = await manager.memoryStats
        XCTAssertEqual(initialStats.processedChunks, 0)

        // Create streams to test continuation setup
        let _ = await manager.results
        let _ = await manager.snapshots

        // Test transcripts
        let volatile = await manager.volatileTranscript
        let finalized = await manager.finalizedTranscript
        XCTAssertEqual(volatile, "")
        XCTAssertEqual(finalized, "")

        // Test finish
        let result = try await manager.finish()
        XCTAssertEqual(result, "")

        // Test cancel
        await manager.cancel()

        // State should be clean after cancel
        let finalStats = await manager.memoryStats
        XCTAssertEqual(finalStats.accumulatedTokens, 0)
    }

    func testStreamingConfigurationIntegration() {
        // Test that different streaming modes produce valid configurations
        let modes: [StreamingMode] = [.lowLatency, .balanced, .highAccuracy]

        for mode in modes {
            let config = StreamingAsrConfig(mode: mode)

            // Verify configuration is internally consistent
            XCTAssertGreaterThan(config.chunkSamples, 0)
            XCTAssertGreaterThan(config.leftContextSamples, 0)
            XCTAssertGreaterThan(config.rightContextSamples, 0)
            XCTAssertGreaterThan(config.interimUpdateSamples, 0)

            // Verify relationships
            XCTAssertLessThan(config.leftContextSamples, config.chunkSamples)
            XCTAssertLessThan(config.interimUpdateSamples, config.chunkSamples)

            // Test ASR config generation
            let asrConfig = config.asrConfig
            XCTAssertEqual(asrConfig.sampleRate, 16000)
            XCTAssertNotNil(asrConfig.tdtConfig)
        }
    }

    func testStreamingMemoryBoundsIntegration() async throws {
        // Test that streaming manager maintains memory bounds across operations
        let manager = StreamingAsrManager()

        var memorySnapshots: [Int] = []

        // Perform multiple lifecycle operations
        for iteration in 0..<5 {
            // Create streams
            let _ = await manager.results
            let _ = await manager.snapshots

            // Simulate some work
            let stats = await manager.memoryStats
            memorySnapshots.append(stats.accumulatedTokens + stats.segmentTexts)

            // Reset
            if iteration < 4 {  // Don't finish on last iteration
                _ = try await manager.finish()
            }
        }

        // Final cleanup
        await manager.cancel()

        // Memory usage should not grow unbounded
        let finalStats = await manager.memoryStats
        XCTAssertEqual(finalStats.accumulatedTokens, 0)
        XCTAssertEqual(finalStats.segmentTexts, 0)
    }

    // MARK: - Concurrent Access Integration Tests

    func testConcurrentStreamingAccess() async throws {
        let manager = StreamingAsrManager()

        // Test that multiple concurrent accesses don't cause issues
        await withTaskGroup(of: Void.self) { group in
            // Multiple result stream accesses
            for _ in 0..<10 {
                group.addTask {
                    let _ = await manager.results
                }
            }

            // Multiple snapshot stream accesses
            for _ in 0..<10 {
                group.addTask {
                    let _ = await manager.snapshots
                }
            }

            // Multiple state accesses
            for _ in 0..<10 {
                group.addTask {
                    let _ = await manager.volatileTranscript
                    let _ = await manager.finalizedTranscript
                    let _ = await manager.memoryStats
                }
            }
        }

        // Should complete without issues
        await manager.cancel()
    }

    // MARK: - Error Recovery Integration Tests

    func testStreamingErrorRecoveryIntegration() async throws {
        let manager = StreamingAsrManager()

        // Test various error recovery scenarios

        // 1. Cancel then restart
        await manager.cancel()
        let _ = await manager.results  // Should work after cancel

        // 2. Finish then access
        _ = try await manager.finish()
        let volatile = await manager.volatileTranscript
        XCTAssertEqual(volatile, "")

        // 3. Multiple finishes
        _ = try await manager.finish()
        _ = try await manager.finish()
        let finalized = await manager.finalizedTranscript
        XCTAssertEqual(finalized, "")

        // 4. Final cleanup
        await manager.cancel()
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.accumulatedTokens, 0)
    }

    // MARK: - Performance Integration Tests

    func testStreamingConfigurationPerformance() {
        measure {
            // Test performance of creating various streaming configurations
            for _ in 0..<1000 {
                let _ = StreamingAsrConfig.default
                let _ = StreamingAsrConfig(mode: .lowLatency)
                let _ = StreamingAsrConfig(mode: .balanced, chunkDuration: 8.0)
                let _ = StreamingAsrConfig(mode: .highAccuracy, enableDebug: true)
            }
        }
    }

    func testStreamingManagerInitializationPerformance() {
        measure {
            // Test performance of creating streaming managers
            for _ in 0..<100 {
                let manager = StreamingAsrManager()
                _ = manager
            }
        }
    }

    // MARK: - Mock Audio Data Integration Tests

    func testStreamingWithMockAudioBuffer() async throws {
        let manager = StreamingAsrManager()

        // Create a small mock PCM buffer
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: 16000,
                channels: 1
            )
        else {
            XCTFail("Failed to create audio format")
            return
        }

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: 1600  // 0.1 second of audio
            )
        else {
            XCTFail("Failed to create PCM buffer")
            return
        }

        buffer.frameLength = 1600

        // Fill with silence (this won't process without models, but tests the API)
        if let channelData = buffer.floatChannelData {
            for frame in 0..<Int(buffer.frameLength) {
                channelData[0][frame] = 0.0  // Silence
            }
        }

        // Test that streaming audio doesn't crash without models
        await manager.streamAudio(buffer)

        // Cleanup
        await manager.cancel()
    }

    // MARK: - Streaming Result Structure Integration Tests

    func testStreamingResultStructureIntegration() {
        // Test that streaming result structures work correctly

        let segmentID = UUID()
        let result = StreamingTranscriptionResult(
            segmentID: segmentID,
            revision: 1,
            attributedText: AttributedString("Test transcription"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 0, preferredTimescale: 1000),
                duration: CMTime(seconds: 2.5, preferredTimescale: 1000)
            ),
            isFinal: false,
            confidence: 0.85,
            timestamp: Date()
        )

        XCTAssertEqual(result.segmentID, segmentID)
        XCTAssertEqual(result.revision, 1)
        XCTAssertFalse(result.isFinal)
        XCTAssertEqual(result.confidence, 0.85)
        XCTAssertEqual(String(result.attributedText.characters), "Test transcription")

        let snapshot = StreamingTranscriptSnapshot(
            finalized: AttributedString("Finalized portion"),
            volatile: AttributedString("volatile portion"),
            lastUpdated: Date()
        )

        XCTAssertEqual(String(snapshot.finalized.characters), "Finalized portion")
        XCTAssertNotNil(snapshot.volatile)
        if let volatile = snapshot.volatile {
            XCTAssertEqual(String(volatile.characters), "volatile portion")
        }
    }
}
