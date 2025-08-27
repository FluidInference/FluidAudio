import AVFoundation
import Darwin
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
        let finalizedTranscript = await manager.finalizedTranscript
        let source = await manager.source

        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(finalizedTranscript, "")
        XCTAssertEqual(source, .microphone)
    }

    func testInitializationWithCustomConfig() async throws {
        let config = StreamingAsrConfig(
            mode: .highAccuracy,
            enableDebug: true,
            chunkDuration: 10.0
        )
        let manager = StreamingAsrManager(config: config)
        let volatileTranscript = await manager.volatileTranscript
        let finalizedTranscript = await manager.finalizedTranscript

        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(finalizedTranscript, "")
    }

    // MARK: - Configuration Tests

    func testConfigPresets() {
        // Test default config
        let defaultConfig = StreamingAsrConfig.default
        XCTAssertEqual(defaultConfig.mode, .balanced)
        XCTAssertEqual(defaultConfig.chunkSeconds, 10.0)
        XCTAssertFalse(defaultConfig.enableDebug)
    }

    func testConfigCalculatedProperties() {
        let config = StreamingAsrConfig(
            mode: .lowLatency,
            enableDebug: false,
            chunkDuration: 5.0
        )
        XCTAssertEqual(config.bufferCapacity, 240000)  // 15 seconds at 16kHz
        XCTAssertEqual(config.chunkSamples, 80000)  // 5 seconds at 16kHz
        XCTAssertEqual(config.leftContextSamples, 16000)  // 1 second at 16kHz (low latency mode)
        XCTAssertEqual(config.rightContextSamples, 16000)  // 1 second at 16kHz (low latency mode)

        // Test ASR config generation
        let asrConfig = config.asrConfig
        XCTAssertEqual(asrConfig.sampleRate, 16000)
        XCTAssertNotNil(asrConfig.tdtConfig)
    }

    func testStreamingModes() {
        // Test low latency mode
        let lowLatencyConfig = StreamingAsrConfig(mode: .lowLatency)
        XCTAssertEqual(lowLatencyConfig.chunkSeconds, 5.0)
        XCTAssertEqual(lowLatencyConfig.leftContextSeconds, 1.0)
        XCTAssertEqual(lowLatencyConfig.interimUpdateFrequency, 0.5)

        // Test balanced mode
        let balancedConfig = StreamingAsrConfig(mode: .balanced)
        XCTAssertEqual(balancedConfig.chunkSeconds, 10.0)
        XCTAssertEqual(balancedConfig.leftContextSeconds, 2.0)
        XCTAssertEqual(balancedConfig.interimUpdateFrequency, 1.0)

        // Test high accuracy mode
        let highAccuracyConfig = StreamingAsrConfig(mode: .highAccuracy)
        XCTAssertEqual(highAccuracyConfig.chunkSeconds, 15.0)
        XCTAssertEqual(highAccuracyConfig.leftContextSeconds, 3.0)
        XCTAssertEqual(highAccuracyConfig.interimUpdateFrequency, 2.0)
    }

    // MARK: - Stream Management Tests

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

    // MARK: - Result Structure Tests

    func testStreamingTranscriptSnapshotCreation() {
        let snapshot = StreamingTranscriptSnapshot(
            finalized: AttributedString("Finalized text"),
            volatile: AttributedString("volatile text"),
            lastUpdated: Date()
        )

        XCTAssertEqual(String(snapshot.finalized.characters), "Finalized text")
        if let volatile = snapshot.volatile {
            XCTAssertEqual(String(volatile.characters), "volatile text")
        } else {
            XCTFail("Expected volatile text to be present")
        }
        XCTAssertNotNil(snapshot.lastUpdated)
    }

    // MARK: - Audio Source Tests

    func testAudioSourceConfiguration() async throws {
        throw XCTSkip("Skipping test that requires model initialization")
    }

    // MARK: - Custom Configuration Tests

    func testCustomConfiguration() {
        let customConfig = StreamingAsrConfig(
            mode: .balanced,
            enableDebug: true,
            chunkDuration: 7.5,
            updateFrequency: 0.4
        )

        XCTAssertEqual(customConfig.chunkSeconds, 7.5)
        XCTAssertEqual(customConfig.interimUpdateFrequency, 0.4)
        XCTAssertTrue(customConfig.enableDebug)
        XCTAssertEqual(customConfig.leftContextSeconds, 2.0)  // From balanced mode
    }

    // MARK: - Performance Tests

    func testChunkSizeCalculationPerformance() {
        measure {
            for duration in stride(from: 1.0, to: 20.0, by: 0.5) {
                let config = StreamingAsrConfig(mode: .balanced, chunkDuration: duration)
                _ = config.chunkSamples
                _ = config.bufferCapacity
            }
        }
    }

    // MARK: - New Streaming API Tests

    func testSnapshotsStreamCreation() async throws {
        let manager = StreamingAsrManager()

        // Test that snapshots stream can be created
        let snapshotsStream = await manager.snapshots
        XCTAssertNotNil(snapshotsStream)
    }

    func testVolatileAndFinalizedTranscriptInitialState() async throws {
        let manager = StreamingAsrManager()

        let volatileTranscript = await manager.volatileTranscript
        let finalizedTranscript = await manager.finalizedTranscript

        XCTAssertEqual(volatileTranscript, "")
        XCTAssertEqual(finalizedTranscript, "")
    }

    func testConfigurationPropertyMapping() {
        let config = StreamingAsrConfig.default

        // Test that all new properties are accessible
        XCTAssertGreaterThan(config.chunkSeconds, 0)
        XCTAssertGreaterThan(config.leftContextSeconds, 0)
        XCTAssertGreaterThan(config.rightContextSeconds, 0)
        XCTAssertGreaterThan(config.interimUpdateFrequency, 0)
        XCTAssertEqual(config.mode, .balanced)
        XCTAssertFalse(config.enableDebug)

        // Test sample conversions
        XCTAssertEqual(config.chunkSamples, Int(config.chunkSeconds * 16000))
        XCTAssertEqual(config.leftContextSamples, Int(config.leftContextSeconds * 16000))
        XCTAssertEqual(config.rightContextSamples, Int(config.rightContextSeconds * 16000))
        XCTAssertEqual(config.interimUpdateSamples, Int(config.interimUpdateFrequency * 16000))
    }

    func testResetFunctionalityWithNewAPI() async throws {
        let manager = StreamingAsrManager()

        // Verify initial state
        let initialVolatile = await manager.volatileTranscript
        let initialFinalized = await manager.finalizedTranscript
        XCTAssertEqual(initialVolatile, "")
        XCTAssertEqual(initialFinalized, "")

        _ = try await manager.finish()

        // State should remain empty after reset
        let afterResetVolatile = await manager.volatileTranscript
        let afterResetFinalized = await manager.finalizedTranscript
        XCTAssertEqual(afterResetVolatile, "")
        XCTAssertEqual(afterResetFinalized, "")
    }

    // MARK: - Memory Management Tests

    func testMemoryLeakPrevention() async throws {
        // Test that collections don't grow unbounded by simulating segment finalization
        let manager = StreamingAsrManager()

        // Simulate many segments being finalized through public API
        // This indirectly tests the cleanup mechanism without requiring internal access

        let segmentTexts = [
            "This is segment one",
            "This is segment two",
            "This is segment three",
            "This is segment four",
            "This is segment five",
        ]

        // Test that multiple resets don't accumulate memory
        for _ in 0..<10 {
            // Reset should clear all internal collections
            _ = try await manager.finish()

            // Verify transcripts are empty after reset
            let volatileAfterReset = await manager.volatileTranscript
            let finalizedAfterReset = await manager.finalizedTranscript
            XCTAssertEqual(volatileAfterReset, "")
            XCTAssertEqual(finalizedAfterReset, "")

            // Simulate some activity (this would normally trigger segment processing)
            for _ in segmentTexts {
                // Access properties to trigger internal state changes
                _ = await manager.volatileTranscript
                _ = await manager.finalizedTranscript
            }
        }

        // Final verification - memory should be clean
        let finalVolatile = await manager.volatileTranscript
        let finalFinalized = await manager.finalizedTranscript
        XCTAssertEqual(finalVolatile, "")
        XCTAssertEqual(finalFinalized, "")
    }

    func testSegmentMemoryAccumulationDetection() async throws {
        // This test can run without models - it tests the manager's memory behavior

        // Measure memory usage over multiple reset cycles
        var memoryUsages: [Int] = []

        for _ in 0..<5 {
            autoreleasepool {
                // Create multiple manager instances to simulate segment accumulation
                let tempManagers = (0..<100).map { _ in StreamingAsrManager() }

                // Force some internal state changes
                Task {
                    for tempManager in tempManagers {
                        _ = await tempManager.volatileTranscript
                        _ = await tempManager.finalizedTranscript
                        _ = try await tempManager.finish()
                    }
                }

                // Measure memory after operations
                var memoryInfo = mach_task_basic_info()
                var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

                let kerr = withUnsafeMutablePointer(to: &memoryInfo) {
                    $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                    }
                }

                if kerr == KERN_SUCCESS {
                    memoryUsages.append(Int(memoryInfo.resident_size))
                }
            }

            // Force garbage collection between cycles
            autoreleasepool {}
        }

        // Verify memory doesn't grow unbounded across cycles
        if memoryUsages.count >= 3 {
            let firstUsage = memoryUsages[0]
            let lastUsage = memoryUsages.last!
            let growth = lastUsage - firstUsage

            // Allow up to 10MB growth across all cycles (reasonable buffer)
            let maxAllowedGrowth = 10 * 1024 * 1024  // 10MB
            XCTAssertLessThan(
                growth, maxAllowedGrowth,
                "Memory usage grew by \(growth) bytes across cycles, which may indicate a leak")
        }
    }

    func testResetClearsMemoryState() async throws {
        let manager = StreamingAsrManager()

        // Reset should clear all internal state
        _ = try await manager.finish()

        // Verify transcripts are empty after reset
        let volatileAfterReset = await manager.volatileTranscript
        let finalizedAfterReset = await manager.finalizedTranscript

        XCTAssertEqual(volatileAfterReset, "")
        XCTAssertEqual(finalizedAfterReset, "")
    }

    func testConcurrentStreamAccess() async throws {
        let manager = StreamingAsrManager()

        // Test concurrent access to snapshots stream
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                let _ = await manager.snapshots
            }
            group.addTask {
                let _ = await manager.volatileTranscript
            }
            group.addTask {
                let _ = await manager.finalizedTranscript
            }
        }

        // All tasks should complete without race conditions
    }

    // MARK: - Performance Boundary Tests

    func testHighFrequencyAudioCallbacks() async throws {
        throw XCTSkip("Requires model initialization and mock audio data")

        // This test should verify system behavior under high audio callback frequency
        // to ensure the task creation in audio callbacks doesn't overwhelm the system
    }

    func testLongRunningSessionStability() async throws {
        throw XCTSkip("Requires extended runtime testing")

        // This test should verify that multi-hour sessions don't degrade
        // performance due to memory accumulation or resource leaks
    }

    // MARK: - Streaming Performance Tests

    func testStreamingMemoryManagement() async throws {
        let manager = StreamingAsrManager()

        // Test memory stats API
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.sampleBufferSize, 0)
        XCTAssertEqual(stats.accumulatedTokens, 0)
        XCTAssertEqual(stats.segmentTexts, 0)
        XCTAssertEqual(stats.processedChunks, 0)
    }

    func testStreamingTokenBoundedGrowth() async throws {
        let manager = StreamingAsrManager()

        // Test that the manager can be initialized and maintains proper state
        // without models (this tests the memory management infrastructure)

        let initialStats = await manager.memoryStats
        XCTAssertEqual(initialStats.accumulatedTokens, 0)

        // Multiple finish() calls should not cause unbounded growth
        for _ in 0..<5 {
            _ = try await manager.finish()
            let stats = await manager.memoryStats
            XCTAssertEqual(stats.accumulatedTokens, 0)
        }
    }

    func testStreamingConfigurationValidation() {
        // Test various streaming configurations for validity
        let configurations = [
            StreamingAsrConfig(mode: .lowLatency, chunkDuration: 2.0),
            StreamingAsrConfig(mode: .balanced, chunkDuration: 5.0),
            StreamingAsrConfig(mode: .highAccuracy, chunkDuration: 10.0),
        ]

        for config in configurations {
            XCTAssertGreaterThan(config.chunkSamples, 0)
            XCTAssertGreaterThan(config.leftContextSamples, 0)
            XCTAssertGreaterThan(config.rightContextSamples, 0)
            XCTAssertGreaterThan(config.interimUpdateSamples, 0)
            XCTAssertLessThan(config.leftContextSamples, config.chunkSamples)
        }
    }

    func testStreamingErrorRecovery() async throws {
        let manager = StreamingAsrManager()

        // Test that cancel works without issues
        await manager.cancel()

        // Test that finish works after cancel
        _ = try await manager.finish()

        // State should be clean
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.accumulatedTokens, 0)
        XCTAssertEqual(stats.segmentTexts, 0)
    }

    func testStreamingContinuationSafety() async throws {
        let manager = StreamingAsrManager()

        // Create multiple streams simultaneously
        let snapshots1 = await manager.snapshots
        let snapshots2 = await manager.snapshots

        // All should be valid stream references
        XCTAssertNotNil(snapshots1)
        XCTAssertNotNil(snapshots2)

        // Cancel should clean up safely
        await manager.cancel()
    }

    func testMemoryGrowthBounds() {
        measure(metrics: [XCTMemoryMetric()]) {
            // Test memory usage during typical operations
            let configs = [
                StreamingAsrConfig.default,
                StreamingAsrConfig(
                    mode: .lowLatency,
                    enableDebug: false,
                    chunkDuration: 7.0,
                    updateFrequency: 10.0
                ),
            ]

            for config in configs {
                let manager = StreamingAsrManager(config: config)
                // Basic operations that shouldn't cause significant memory growth
                _ = manager
            }
        }
    }
}
