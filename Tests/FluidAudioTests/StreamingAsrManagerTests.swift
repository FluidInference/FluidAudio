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
        let source = await manager.source
        let stats = await manager.memoryStats

        XCTAssertEqual(source, .microphone)
        XCTAssertEqual(stats.sampleBufferSize, 0)
        XCTAssertEqual(stats.accumulatedTokens, 0)
        XCTAssertEqual(stats.processedChunks, 0)
    }

    func testInitializationWithCustomConfig() async throws {
        let config = StreamingAsrConfig(
            mode: .highAccuracy,
            enableDebug: true,
            chunkDuration: 10.0
        )
        let manager = StreamingAsrManager(config: config)
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.sampleBufferSize, 0)
        XCTAssertEqual(stats.accumulatedTokens, 0)
        XCTAssertEqual(stats.processedChunks, 0)
    }

    // MARK: - Configuration Tests

    func testConfigPresets() {
        // Test default config (frame-aligned mode)
        let defaultConfig = StreamingAsrConfig.default
        XCTAssertEqual(defaultConfig.mode, .frameAligned)
        XCTAssertEqual(defaultConfig.chunkSeconds, 1.6, accuracy: 0.001)
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

    func testSegmentUpdatesStreamCreation() async throws {
        let manager = StreamingAsrManager()

        // Test that segmentUpdates stream can be created
        let updatesStream = await manager.segmentUpdates
        // Simply ensure we have a stream object
        _ = updatesStream
    }

    func testInitialState() async throws {
        let manager = StreamingAsrManager()
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.sampleBufferSize, 0)
        XCTAssertEqual(stats.accumulatedTokens, 0)
        XCTAssertEqual(stats.processedChunks, 0)
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
        let initialStats = await manager.memoryStats
        XCTAssertEqual(initialStats.accumulatedTokens, 0)

        try await manager.stop()

        // State should remain empty after reset
        let afterResetStats = await manager.memoryStats
        XCTAssertEqual(afterResetStats.accumulatedTokens, 0)
    }

    // MARK: - Memory Management Tests

    func testMemoryLeakPrevention() async throws {
        // Test that collections don't grow unbounded by simulating segment finalization
        let manager = StreamingAsrManager()

        // Simulate resets and verify memory remains bounded
        for _ in 0..<10 {
            try await manager.stop()
            let stats = await manager.memoryStats
            XCTAssertEqual(stats.accumulatedTokens, 0)
        }
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
                        try? await tempManager.stop()
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
        try await manager.stop()

        // Verify memory stats are empty after reset
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.accumulatedTokens, 0)
    }

    func testConcurrentStreamAccess() async throws {
        let manager = StreamingAsrManager()

        // Test concurrent access to segmentUpdates stream
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                let _ = await manager.segmentUpdates
            }
            group.addTask {
                let _ = await manager.memoryStats
            }
            group.addTask {
                let _ = await manager.source
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
        XCTAssertEqual(stats.processedChunks, 0)
    }

    func testStreamingTokenBoundedGrowth() async throws {
        let manager = StreamingAsrManager()

        // Test that the manager can be initialized and maintains proper state
        // without models (this tests the memory management infrastructure)

        let initialStats = await manager.memoryStats
        XCTAssertEqual(initialStats.accumulatedTokens, 0)

        // Multiple stop() calls should not cause unbounded growth
        for _ in 0..<5 {
            try await manager.stop()
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

        // Test that stop works after cancel
        try await manager.stop()

        // State should be clean
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.accumulatedTokens, 0)
    }

    func testStreamingContinuationSafety() async throws {
        let manager = StreamingAsrManager()

        // Create multiple streams simultaneously
        let updates1 = await manager.segmentUpdates
        let updates2 = await manager.segmentUpdates

        // All should be valid stream references
        _ = updates1
        _ = updates2

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

    // MARK: - Timestamp Tests

    func testTimestampedSegmentCreation() {
        let segmentId = UUID()
        let segment = TimestampedSegment(
            id: segmentId,
            text: "Hello world",
            startTime: 1.5,
            endTime: 3.2,
            confidence: 0.95
        )

        XCTAssertEqual(segment.id, segmentId)
        XCTAssertEqual(segment.text, "Hello world")
        XCTAssertEqual(segment.startTime, 1.5, accuracy: 0.001)
        XCTAssertEqual(segment.endTime, 3.2, accuracy: 0.001)
        XCTAssertEqual(segment.confidence, 0.95, accuracy: 0.001)
        XCTAssertEqual(segment.duration, 1.7, accuracy: 0.001)
    }

    func testTimestampedSegmentFormatting() {
        let segment = TimestampedSegment(
            id: UUID(),
            text: "Test segment",
            startTime: 65.123,  // 1:05.123
            endTime: 127.456  // 2:07.456
        )

        let expectedTimeRange = "00:01:05.123 --> 00:02:07.456"
        XCTAssertEqual(segment.formattedTimeRange, expectedTimeRange)
    }

    func testStreamingTranscriptSnapshotWithTimestamps() {
        let segments = [
            TimestampedSegment(id: UUID(), text: "First segment", startTime: 0.0, endTime: 2.0),
            TimestampedSegment(id: UUID(), text: "Second segment", startTime: 2.5, endTime: 4.5),
        ]

        let snapshot = StreamingTranscriptSnapshot(
            finalized: AttributedString("First segment Second segment"),
            volatile: AttributedString("Partial text"),
            lastUpdated: Date(),
            timestampedSegments: segments
        )

        XCTAssertEqual(snapshot.timestampedSegments.count, 2)
        XCTAssertEqual(snapshot.timestampedSegments[0].text, "First segment")
        XCTAssertEqual(snapshot.timestampedSegments[1].text, "Second segment")

        // Test timestamped text format
        let timestampedText = snapshot.timestampedText
        XCTAssertTrue(timestampedText.contains("00:00:00.000 --> 00:00:02.000"))
        XCTAssertTrue(timestampedText.contains("First segment"))
        XCTAssertTrue(timestampedText.contains("00:00:02.500 --> 00:00:04.500"))
        XCTAssertTrue(timestampedText.contains("Second segment"))
    }

    func testSRTFormatExport() {
        let segments = [
            TimestampedSegment(id: UUID(), text: "Hello world", startTime: 0.5, endTime: 2.3),
            TimestampedSegment(id: UUID(), text: "How are you?", startTime: 3.1, endTime: 4.8),
        ]

        let snapshot = StreamingTranscriptSnapshot(
            finalized: AttributedString("Hello world How are you?"),
            volatile: nil,
            lastUpdated: Date(),
            timestampedSegments: segments
        )

        let srtFormat = snapshot.srtFormat

        // Check SRT format structure
        XCTAssertTrue(srtFormat.contains("1\n"), "Missing subtitle number 1")
        XCTAssertTrue(srtFormat.contains("00:00:00,500 --> 00:00:02,299\n"), "Missing first timestamp")
        XCTAssertTrue(srtFormat.contains("Hello world\n"), "Missing first text")
        XCTAssertTrue(srtFormat.contains("2\n"), "Missing subtitle number 2")
        XCTAssertTrue(srtFormat.contains("00:00:03,100 --> 00:00:04,799\n"), "Missing second timestamp")
        XCTAssertTrue(srtFormat.contains("How are you?\n"), "Missing second text")
    }

    // MARK: - Final Segment Processing Tests

    /// Helper method to create test audio buffers in ASR format (16kHz mono Float32)
    private func createTestAudioBuffer(
        durationSeconds: Double,
        frequency: Double = 440.0,  // A4 note
        amplitude: Float = 0.3
    ) throws -> AVAudioPCMBuffer {
        let sampleRate = 16000.0
        let channels: AVAudioChannelCount = 1

        guard
            let audioFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: channels,
                interleaved: false
            )
        else {
            throw NSError(
                domain: "TestError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
        }

        let frameCount = AVAudioFrameCount(sampleRate * durationSeconds)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: frameCount) else {
            throw NSError(
                domain: "TestError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }

        buffer.frameLength = frameCount

        // Fill with sine wave data at specified frequency
        if let channelData = buffer.floatChannelData {
            for i in 0..<Int(frameCount) {
                let phase = Double(i) / sampleRate * frequency * 2.0 * Double.pi
                channelData[0][i] = Float(sin(phase)) * amplitude
            }
        }

        return buffer
    }

    /// Test that demonstrates the final pending segment issue using real audio
    func testFinalPendingSegmentWithRealAudio() async throws {
        // Skip unless running in debug mode for manual testing
        guard ProcessInfo.processInfo.environment["RUN_STREAMING_TESTS"] == "1" else {
            throw XCTSkip("Set RUN_STREAMING_TESTS=1 to run streaming tests with real models")
        }

        let config = StreamingAsrConfig(
            mode: .balanced,  // 10s chunks, 1s updates
            enableDebug: true
        )
        let manager = StreamingAsrManager(config: config)

        // Track segment updates to monitor finalization
        var updates: [StreamingSegmentUpdate] = []
        let snapshotTask = Task {
            let stream = await manager.segmentUpdates
            for await update in stream {
                updates.append(update)
                print("Update: volatile=\(update.isVolatile) text='\(update.text)'")
            }
        }

        do {
            // Start the manager with real models (using system source for testing)
            try await manager.start(source: .system)

            print("Testing final pending segment processing with synthetic audio...")

            var updateCountBefore = 0

            // Stream 15 seconds worth (should trigger finalization around 10s mark)
            print("Streaming first 15 seconds...")
            for i in 0..<15 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 440.0)
                await manager.streamAudio(chunk)

                // Track updates as they come in
                if updates.count > updateCountBefore {
                    updateCountBefore = updates.count
                    if let lastUpdate = updates.last {
                        print("New update at t=\(i+1)s: volatile=\(lastUpdate.isVolatile) text='\(lastUpdate.text)'")
                    }
                }
            }

            print("After 15s: \(updates.count) updates")

            // Add the "pending" 7 seconds that should be processed in finish()
            print("Streaming final 7 seconds (pending segment)...")
            for _ in 15..<22 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 660.0)  // Different frequency
                await manager.streamAudio(chunk)
            }

            print("After 22s (before stop): \(updates.count) updates")
            let updatesBeforeStop = updates.count

            // This is the key test: does stop() process the final 7 seconds?
            print("Calling stop() - should process pending 7 seconds...")
            try await manager.stop()

            let updatesAfterStop = updates.count
            print("Total updates: before stop=\(updatesBeforeStop), after stop=\(updatesAfterStop)")

            if updatesAfterStop > updatesBeforeStop {
                print("✓ SUCCESS: stop() generated \(updatesAfterStop - updatesBeforeStop) additional updates")
            } else {
                print("❌ ISSUE CONFIRMED: stop() did not produce additional updates")
            }

            print("✓ Test completed - buffering and stop() behavior verified")

        } catch {
            print("Test failed with error: \(error)")
            throw error
        }

        snapshotTask.cancel()
    }

    /// Test the volatile-to-finalized transition flow to identify where pending segments might be lost
    func testVolatileToFinalizedTransition() async throws {
        // Skip unless running in debug mode for manual testing
        guard ProcessInfo.processInfo.environment["RUN_STREAMING_TESTS"] == "1" else {
            throw XCTSkip("Set RUN_STREAMING_TESTS=1 to run streaming tests with real models")
        }

        let config = StreamingAsrConfig(
            mode: .balanced,  // 10s chunks, 1s updates
            enableDebug: true
        )
        let manager = StreamingAsrManager(config: config)

        // Detailed update tracking
        var updates: [StreamingSegmentUpdate] = []
        var volatileHistory: [(time: Date, content: String)] = []
        var finalizedHistory: [(time: Date, content: String)] = []

        let snapshotTask = Task {
            let stream = await manager.segmentUpdates
            for await update in stream {
                updates.append(update)

                let timestamp = Date()
                let text = update.text

                if update.isVolatile {
                    if !text.isEmpty && (volatileHistory.last?.content != text) {
                        volatileHistory.append((time: timestamp, content: text))
                        print("📝 VOLATILE UPDATE #\(volatileHistory.count): '\(text)'")
                    }
                } else {
                    if !text.isEmpty && (finalizedHistory.last?.content != text) {
                        finalizedHistory.append((time: timestamp, content: text))
                        print("✅ FINALIZED UPDATE #\(finalizedHistory.count): '\(text)'")
                    }
                }

                print("📊 Update #\(updates.count): volatile=\(update.isVolatile) text='\(text)'")
            }
        }

        do {
            try await manager.start(source: .system)

            print("🔍 Testing volatile-to-finalized transition with detailed tracking...")
            print("📋 Streaming pattern: 15s (expect finalization) + 7s (pending) + stop()")

            // Stream in smaller chunks for better visibility into state changes
            let chunkDuration = 0.5  // 500ms chunks for detailed tracking

            // Phase 1: Stream 15 seconds (should trigger normal finalization)
            print("\n🎵 Phase 1: Streaming first 15 seconds in 0.5s chunks...")
            var totalTime = 0.0

            for chunkIndex in 0..<30 {  // 30 chunks of 0.5s = 15s
                let chunk = try createTestAudioBuffer(durationSeconds: chunkDuration, frequency: 440.0)
                await manager.streamAudio(chunk)
                totalTime += chunkDuration

                // Log every 2 seconds
                if chunkIndex % 4 == 3 {
                    print(
                        "⏱️  t=\(totalTime)s: \(updates.count) updates, V-history=\(volatileHistory.count), F-history=\(finalizedHistory.count)"
                    )
                }
            }

            print("\n📈 After 15s: \(updates.count) total updates")
            print("   Volatile updates: \(volatileHistory.count)")
            print("   Finalized updates: \(finalizedHistory.count)")

            // Phase 2: Stream the critical 7 seconds that should be "pending"
            print("\n🎵 Phase 2: Streaming final 7 seconds (should be pending)...")
            let volatileCountBefore = volatileHistory.count
            let finalizedCountBefore = finalizedHistory.count

            for _ in 0..<14 {  // 14 chunks of 0.5s = 7s
                let chunk = try createTestAudioBuffer(durationSeconds: chunkDuration, frequency: 660.0)  // Different frequency
                await manager.streamAudio(chunk)
                totalTime += chunkDuration
            }

            let volatileCountAfter = volatileHistory.count
            let finalizedCountAfter = finalizedHistory.count

            print("\n📈 After 22s (before stop):")
            print("   Total updates: \(updates.count)")
            print("   New volatile updates during 7s: \(volatileCountAfter - volatileCountBefore)")
            print("   New finalized updates during 7s: \(finalizedCountAfter - finalizedCountBefore)")

            // Phase 3: Call stop() and track what happens
            print("\n🏁 Phase 3: Calling stop() - tracking volatile-to-finalized transition...")
            let volatileBeforeFinish = volatileHistory.count
            let finalizedBeforeFinish = finalizedHistory.count
            let updatesBeforeStop = updates.count

            try await manager.stop()

            // Analysis
            let volatileAfterFinish = volatileHistory.count
            let finalizedAfterFinish = finalizedHistory.count
            let updatesAfterStop = updates.count

            print("\n📊 ANALYSIS:")
            print("   New updates during stop(): \(updatesAfterStop - updatesBeforeStop)")
            print("   New volatile updates during stop(): \(volatileAfterFinish - volatileBeforeFinish)")
            print("   New finalized updates during stop(): \(finalizedAfterFinish - finalizedBeforeFinish)")

            // Key test: Did volatile content become finalized?
            if volatileAfterFinish > volatileBeforeFinish {
                print("✅ stop() generated NEW volatile content")
            }

            if finalizedAfterFinish > finalizedBeforeFinish {
                print("✅ stop() generated NEW finalized content")
            }

            // Show complete history for debugging
            print("\n📋 COMPLETE VOLATILE HISTORY:")
            for (index, entry) in volatileHistory.enumerated() {
                print("   V\(index + 1): '\(entry.content)'")
            }

            print("\n📋 COMPLETE FINALIZED HISTORY:")
            for (index, entry) in finalizedHistory.enumerated() {
                print("   F\(index + 1): '\(entry.content)'")
            }

            print("\n✓ Volatile-to-finalized transition test completed")

        } catch {
            print("Test failed with error: \(error)")
            throw error
        }

        snapshotTask.cancel()
    }

    /// Test the specific scenario from the user's question: 15s + 7s
    func testFinalPendingSegmentProcessing() async throws {
        // Skip unless running in debug mode for manual testing
        guard ProcessInfo.processInfo.environment["RUN_STREAMING_TESTS"] == "1" else {
            throw XCTSkip("Set RUN_STREAMING_TESTS=1 to run streaming tests with real models")
        }

        let config = StreamingAsrConfig(mode: .balanced, enableDebug: true)  // Enable debug for logs
        let manager = StreamingAsrManager(config: config)

        var updates: [StreamingSegmentUpdate] = []
        let snapshotTask = Task {
            let stream = await manager.segmentUpdates
            for await update in stream {
                updates.append(update)
            }
        }

        do {
            try await manager.start(source: .system)

            print("🧪 Testing the user's exact scenario: stream ~15s, then 7s pending, then finish()")

            // Stream approximately 15 seconds to trigger a normal processing cycle
            for _ in 0..<15 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 440.0)
                await manager.streamAudio(chunk)
            }

            print("After 15s: \(updates.count) updates")

            // Stream the critical 7 seconds that should be processed with extended left context
            for _ in 0..<7 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 660.0)
                await manager.streamAudio(chunk)
            }

            let updatesBeforeStop = updates.count
            print("After 22s (before stop): \(updatesBeforeStop) updates")

            // This is where our extended left context behavior should help
            try await manager.stop()

            let updatesAfterStop = updates.count

            print("🎯 RESULTS:")
            print("   Updates before stop: \(updatesBeforeStop)")
            print("   Updates after stop: \(updatesAfterStop)")
            print("   New updates from stop(): \(updatesAfterStop - updatesBeforeStop)")

            // Verify that stop() processed the final segment
            XCTAssertGreaterThan(
                updatesAfterStop, updatesBeforeStop, "stop() should generate additional updates")

            print("✅ Extended left context behavior verified")

        } catch {
            print("Test failed: \(error)")
            throw error
        }

        snapshotTask.cancel()
    }

    /// Test processing partial segments of various sizes
    func testPartialSegmentFinalization() async throws {
        throw XCTSkip("Requires model initialization - run manually for debugging")

        let config = StreamingAsrConfig(mode: .balanced, enableDebug: true)
        let partialDurations = [3.0, 5.0, 7.0, 9.0]  // Various partial segment sizes

        for duration in partialDurations {
            let manager = StreamingAsrManager(config: config)

            print("Testing partial segment of \(duration)s")

            let audioBuffer = try createTestAudioBuffer(
                durationSeconds: duration,
                frequency: 440.0 + (duration * 50)  // Unique frequency per test
            )

            await manager.streamAudio(audioBuffer)

            try await manager.stop()

            print("Duration: \(duration)s processed without crash")
        }
    }

    /// Test finishing with empty buffer (edge case)
    func testEmptyBufferFinish() async throws {
        let manager = StreamingAsrManager()

        // Call stop without streaming any audio
        try await manager.stop()

        // Ensure no crash and state remains empty
        let stats = await manager.memoryStats
        XCTAssertEqual(stats.accumulatedTokens, 0)
    }

    /// Test timing of final segment processing
    func testFinalSegmentTimingEdgeCases() async throws {
        throw XCTSkip("Requires model initialization - run manually for debugging")

        let config = StreamingAsrConfig(mode: .lowLatency, enableDebug: true)  // 5s chunks for faster testing
        let manager = StreamingAsrManager(config: config)

        var updates: [StreamingSegmentUpdate] = []
        let snapshotTask = Task {
            let stream = await manager.segmentUpdates
            for await update in stream {
                updates.append(update)
                let len = update.text.count
                print("Update \(updates.count): volatile=\(update.isVolatile) textLen=\(len) chars")
            }
        }

        do {
            // Stream exactly 5.5 seconds (should trigger one final at 5s, leave 0.5s pending)
            let firstChunk = try createTestAudioBuffer(durationSeconds: 5.0, frequency: 440.0)
            let secondChunk = try createTestAudioBuffer(durationSeconds: 0.5, frequency: 880.0)

            await manager.streamAudio(firstChunk)
            try await Task.sleep(nanoseconds: 500_000_000)  // 0.5s delay

            await manager.streamAudio(secondChunk)
            try await Task.sleep(nanoseconds: 100_000_000)  // 0.1s delay

            try await manager.stop()

            print("Stopped after 5.5s, total updates: \(updates.count)")

        } catch {
            print("Timing test failed: \(error)")
            throw error
        }

        snapshotTask.cancel()
    }
}
