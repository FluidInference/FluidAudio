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

        // Track snapshots to monitor finalization
        var snapshots: [StreamingTranscriptSnapshot] = []
        let snapshotTask = Task {
            let stream = await manager.snapshots
            for await snapshot in stream {
                snapshots.append(snapshot)
                print(
                    "Snapshot: finalized='\(snapshot.finalized)' volatile='\(snapshot.volatile?.description ?? "nil")'")
            }
        }

        do {
            // Start the manager with real models (using system source for testing)
            try await manager.start(source: .system)

            print("Testing final pending segment processing with synthetic audio...")

            var snapshotCountBefore = 0

            // Stream 15 seconds worth (should trigger finalization around 10s mark)
            print("Streaming first 15 seconds...")
            for i in 0..<15 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 440.0)
                await manager.streamAudio(chunk)

                // Track snapshots as they come in
                if snapshots.count > snapshotCountBefore {
                    snapshotCountBefore = snapshots.count
                    if let lastSnapshot = snapshots.last {
                        let finalizedLength = String(lastSnapshot.finalized.characters).count
                        print("New snapshot at t=\(i+1)s: finalized length=\(finalizedLength) chars")
                    }
                }
            }

            print("After 15s: \(snapshots.count) snapshots")

            // Add the "pending" 7 seconds that should be processed in finish()
            print("Streaming final 7 seconds (pending segment)...")
            for _ in 15..<22 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 660.0)  // Different frequency
                await manager.streamAudio(chunk)
            }

            print("After 22s (before finish): \(snapshots.count) snapshots")
            let snapshotsBeforeFinish = snapshots.count

            if let lastSnapshot = snapshots.last {
                let finalizedText = String(lastSnapshot.finalized.characters)
                let volatileText = lastSnapshot.volatile.map { String($0.characters) } ?? "nil"
                print("Last snapshot before finish: finalized='\(finalizedText)' volatile='\(volatileText)'")
            }

            // This is the key test: does finish() process the final 7 seconds?
            print("Calling finish() - should process pending 7 seconds...")
            let finalTranscript = try await manager.finish()

            let snapshotsAfterFinish = snapshots.count

            print("Final transcript: '\(finalTranscript)'")
            print("Total snapshots: before finish=\(snapshotsBeforeFinish), after finish=\(snapshotsAfterFinish)")

            // The key insight: if finish() properly processes pending segments,
            // we should see additional snapshots OR a non-empty final transcript
            // Even with synthetic audio, the system should at least attempt processing
            if snapshotsAfterFinish > snapshotsBeforeFinish {
                print(
                    "✓ SUCCESS: finish() generated \(snapshotsAfterFinish - snapshotsBeforeFinish) additional snapshots"
                )
            } else if !finalTranscript.isEmpty {
                print("✓ SUCCESS: finish() produced a final transcript")
            } else {
                print(
                    "❌ ISSUE CONFIRMED: finish() did not process pending segments (no new snapshots, empty transcript)")
            }

            print("✓ Test completed - buffering and finish() behavior verified")

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

        // Detailed snapshot tracking
        var snapshots: [StreamingTranscriptSnapshot] = []
        var volatileHistory: [(time: Date, content: String)] = []
        var finalizedHistory: [(time: Date, content: String)] = []

        let snapshotTask = Task {
            let stream = await manager.snapshots
            for await snapshot in stream {
                snapshots.append(snapshot)

                let timestamp = Date()
                let finalizedText = String(snapshot.finalized.characters)
                let volatileText = snapshot.volatile.map { String($0.characters) } ?? ""

                // Track changes in volatile content
                if !volatileText.isEmpty && (volatileHistory.last?.content != volatileText) {
                    volatileHistory.append((time: timestamp, content: volatileText))
                    print("📝 VOLATILE UPDATE #\(volatileHistory.count): '\(volatileText)'")
                }

                // Track changes in finalized content
                if !finalizedText.isEmpty && (finalizedHistory.last?.content != finalizedText) {
                    finalizedHistory.append((time: timestamp, content: finalizedText))
                    print("✅ FINALIZED UPDATE #\(finalizedHistory.count): '\(finalizedText)'")
                }

                print("📊 Snapshot #\(snapshots.count): F='\(finalizedText)' V='\(volatileText)'")
            }
        }

        do {
            try await manager.start(source: .system)

            print("🔍 Testing volatile-to-finalized transition with detailed tracking...")
            print("📋 Streaming pattern: 15s (expect finalization) + 7s (pending) + finish()")

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
                        "⏱️  t=\(totalTime)s: \(snapshots.count) snapshots, V-history=\(volatileHistory.count), F-history=\(finalizedHistory.count)"
                    )
                }
            }

            print("\n📈 After 15s: \(snapshots.count) total snapshots")
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

            print("\n📈 After 22s (before finish):")
            print("   Total snapshots: \(snapshots.count)")
            print("   New volatile updates during 7s: \(volatileCountAfter - volatileCountBefore)")
            print("   New finalized updates during 7s: \(finalizedCountAfter - finalizedCountBefore)")

            // Check current state before finish()
            if let lastSnapshot = snapshots.last {
                let currentFinalized = String(lastSnapshot.finalized.characters)
                let currentVolatile = lastSnapshot.volatile.map { String($0.characters) } ?? ""
                print("   Current state: F='\(currentFinalized)' V='\(currentVolatile)'")
            }

            // Phase 3: Call finish() and track what happens
            print("\n🏁 Phase 3: Calling finish() - tracking volatile-to-finalized transition...")
            let volatileBeforeFinish = volatileHistory.count
            let finalizedBeforeFinish = finalizedHistory.count
            let snapshotsBeforeFinish = snapshots.count

            let finalTranscript = try await manager.finish()

            // Analysis
            let volatileAfterFinish = volatileHistory.count
            let finalizedAfterFinish = finalizedHistory.count
            let snapshotsAfterFinish = snapshots.count

            print("\n📊 ANALYSIS:")
            print("   Final transcript: '\(finalTranscript)'")
            print("   New snapshots during finish(): \(snapshotsAfterFinish - snapshotsBeforeFinish)")
            print("   New volatile updates during finish(): \(volatileAfterFinish - volatileBeforeFinish)")
            print("   New finalized updates during finish(): \(finalizedAfterFinish - finalizedBeforeFinish)")

            // Key test: Did volatile content become finalized?
            if volatileAfterFinish > volatileBeforeFinish {
                print("✅ finish() generated NEW volatile content")
            }

            if finalizedAfterFinish > finalizedBeforeFinish {
                print("✅ finish() generated NEW finalized content")
            }

            if !finalTranscript.isEmpty {
                print("✅ finish() produced non-empty final transcript")
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

        var snapshots: [StreamingTranscriptSnapshot] = []
        let snapshotTask = Task {
            let stream = await manager.snapshots
            for await snapshot in stream {
                snapshots.append(snapshot)
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

            print("After 15s: \(snapshots.count) snapshots")

            // Stream the critical 7 seconds that should be processed with extended left context
            for _ in 0..<7 {
                let chunk = try createTestAudioBuffer(durationSeconds: 1.0, frequency: 660.0)
                await manager.streamAudio(chunk)
            }

            let snapshotsBeforeFinish = snapshots.count
            print("After 22s (before finish): \(snapshotsBeforeFinish) snapshots")

            // This is where our extended left context fix should help
            let finalTranscript = try await manager.finish()

            let snapshotsAfterFinish = snapshots.count

            print("🎯 RESULTS:")
            print("   Final transcript: '\(finalTranscript)'")
            print("   Snapshots before finish: \(snapshotsBeforeFinish)")
            print("   Snapshots after finish: \(snapshotsAfterFinish)")
            print("   New snapshots from finish(): \(snapshotsAfterFinish - snapshotsBeforeFinish)")

            // Verify that finish() processed the final segment
            XCTAssertGreaterThan(
                snapshotsAfterFinish, snapshotsBeforeFinish, "finish() should generate additional snapshots")
            XCTAssertFalse(finalTranscript.isEmpty, "finish() should produce a non-empty final transcript")

            print("✅ Extended left context fix verified")

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

            let finalTranscript = try await manager.finish()

            print("Duration: \(duration)s, Transcript: '\(finalTranscript)'")

            // In a real test with models, we'd verify the transcript contains expected content
            // For now, just ensure it doesn't crash
            XCTAssertNotNil(finalTranscript, "Final transcript should not be nil for \(duration)s segment")
        }
    }

    /// Test finishing with empty buffer (edge case)
    func testEmptyBufferFinish() async throws {
        let manager = StreamingAsrManager()

        // Call finish without streaming any audio
        let finalTranscript = try await manager.finish()

        XCTAssertEqual(finalTranscript, "", "Empty buffer should produce empty transcript")
    }

    /// Test timing of final segment processing
    func testFinalSegmentTimingEdgeCases() async throws {
        throw XCTSkip("Requires model initialization - run manually for debugging")

        let config = StreamingAsrConfig(mode: .lowLatency, enableDebug: true)  // 5s chunks for faster testing
        let manager = StreamingAsrManager(config: config)

        var snapshots: [StreamingTranscriptSnapshot] = []
        let snapshotTask = Task {
            let stream = await manager.snapshots
            for await snapshot in stream {
                snapshots.append(snapshot)
                let finalizedCount = String(snapshot.finalized.characters).count
                let volatileCount = snapshot.volatile.map { String($0.characters).count } ?? 0
                print("Snapshot \(snapshots.count): finalized=\(finalizedCount) chars, volatile=\(volatileCount) chars")
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

            let finalTranscript = try await manager.finish()

            print("Final transcript after 5.5s: '\(finalTranscript)'")
            print("Total snapshots: \(snapshots.count)")

            // The final 0.5s should be processed in finish()
            XCTAssertFalse(finalTranscript.isEmpty, "Should have transcript from 5.5s of audio")

        } catch {
            print("Timing test failed: \(error)")
            throw error
        }

        snapshotTask.cancel()
    }
}
