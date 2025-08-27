import AVFoundation
import XCTest

@testable import FluidAudio

/// Integration tests for the unified streaming ASR + diarization pipeline
@available(macOS 13.0, iOS 16.0, *)
final class StreamingDiarizedAsrManagerTests: XCTestCase {

    override func setUpWithError() throws {
        // Skip tests if running on CI
        try XCTSkipUnless(!isRunningOnCI(), "Skipping streaming diarized ASR tests on CI")
    }

    // MARK: - Basic Initialization Tests

    func testInitialization() throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Manager should initialize without throwing
        XCTAssertNotNil(manager)
    }

    func testConfigurationPresets() throws {
        let defaultConfig = StreamingDiarizedAsrConfig.default
        let accurateConfig = StreamingDiarizedAsrConfig.accurate

        // Default config should prioritize speed
        XCTAssertEqual(defaultConfig.asrConfig.chunkSeconds, 11.0)
        XCTAssertEqual(defaultConfig.asrConfig.leftContextSeconds, 2.0)
        XCTAssertEqual(defaultConfig.alignmentTolerance, 0.5)

        // Accurate config should prioritize quality
        XCTAssertEqual(accurateConfig.asrConfig.chunkSeconds, 11.0)
        XCTAssertEqual(accurateConfig.alignmentTolerance, 0.3)
        XCTAssertEqual(accurateConfig.diarizerConfig.chunkOverlap, 1.0)
    }

    // MARK: - Mock Audio Processing Tests

    func testBasicAudioProcessing() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Create mock audio buffer (1 second of silence at 16kHz)
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let frameCount = AVAudioFrameCount(16000)  // 1 second
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFormat,
                frameCapacity: frameCount
            )
        else {
            XCTFail("Failed to create PCM buffer")
            return
        }

        buffer.frameLength = frameCount

        // Fill with silence (zeros)
        let channelData = buffer.floatChannelData![0]
        for i in 0..<Int(frameCount) {
            channelData[i] = 0.0
        }

        // Test that we can feed audio without crashing
        // Note: We can't easily test full pipeline without models
        // so this is mainly a smoke test
        await manager.streamAudio(buffer)
        await manager.cancel()
    }

    func testStartWithoutModels() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Starting without models will try to download them automatically
        // This is the expected behavior, so we'll test that it either:
        // 1. Succeeds (if models can be downloaded)
        // 2. Fails with a model-related error (if network/download fails)
        do {
            try await manager.start()
            // If it succeeds, clean up
            await manager.cancel()
        } catch {
            // Expected to throw due to model download/loading issues
            let errorMessage = error.localizedDescription.lowercased()
            XCTAssertTrue(
                error is DiarizedAsrError || errorMessage.contains("model") || errorMessage.contains("download")
                    || errorMessage.contains("network") || errorMessage.contains("load"),
                "Unexpected error: \(error)"
            )
        }
    }

    // MARK: - Result Stream Tests

    func testResultStreams() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Test that result streams can be accessed
        let resultsStream = await manager.results
        let snapshotsStream = await manager.snapshots

        XCTAssertNotNil(resultsStream)
        XCTAssertNotNil(snapshotsStream)

        await manager.cancel()
    }

    func testSessionStatistics() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Initially should have no session statistics
        let initialStats = await manager.sessionStatistics
        XCTAssertTrue(initialStats.isEmpty)

        await manager.cancel()
    }

    // MARK: - Configuration Validation Tests

    func testInvalidConfiguration() async throws {
        // Test configuration with very small alignment tolerance
        let config = StreamingDiarizedAsrConfig(
            asrConfig: .default,
            diarizerConfig: .default,
            enableDebug: false,
            alignmentTolerance: -1.0  // Invalid negative value
        )

        let manager = StreamingDiarizedAsrManager(config: config)

        // Manager should still initialize but might produce warnings
        XCTAssertNotNil(manager)

        await manager.cancel()
    }

    // MARK: - Concurrency and Memory Tests

    func testConcurrentAccess() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Test multiple concurrent operations
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                let _ = await manager.sessionStatistics
            }

            group.addTask {
                let _ = await manager.results
            }

            group.addTask {
                let _ = await manager.snapshots
            }
        }

        await manager.cancel()
    }

    func testResetFunctionality() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Reset should complete without error even if not started
        do {
            try await manager.reset()
        } catch {
            // Reset might throw if components aren't initialized, which is fine
            XCTAssertTrue(
                error.localizedDescription.contains("not initialized") || error.localizedDescription.contains("reset"))
        }

        await manager.cancel()
    }

    func testMultipleCancel() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Multiple cancels should be safe
        await manager.cancel()
        await manager.cancel()
        await manager.cancel()

        // Should complete without issues
        XCTAssertTrue(true)
    }

    // MARK: - Performance Tests

    func testMemoryUsage() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Create multiple audio buffers
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let frameCount = AVAudioFrameCount(8000)  // 0.5 seconds

        for _ in 0..<10 {
            guard
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: audioFormat,
                    frameCapacity: frameCount
                )
            else {
                continue
            }

            buffer.frameLength = frameCount

            // Stream the buffer
            await manager.streamAudio(buffer)
        }

        // Cleanup
        await manager.cancel()

        // Test should complete without memory issues
        XCTAssertTrue(true)
    }

    // MARK: - Edge Case Tests

    func testEmptyAudioBuffer() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        // Create buffer with zero frames
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFormat,
                frameCapacity: 1000
            )
        else {
            XCTFail("Failed to create PCM buffer")
            return
        }

        buffer.frameLength = 0  // Empty buffer

        // Should handle empty buffer gracefully
        await manager.streamAudio(buffer)
        await manager.cancel()
    }

    func testDifferentAudioFormats() async throws {
        let config = StreamingDiarizedAsrConfig.default
        let manager = StreamingDiarizedAsrManager(config: config)

        // Test different sample rates (should be converted internally)
        let formats = [
            AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44100, channels: 1, interleaved: false)!,
            AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false)!,
            AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 2, interleaved: false)!,  // Stereo
        ]

        for format in formats {
            let frameCount = AVAudioFrameCount(format.sampleRate * 0.1)  // 0.1 seconds
            guard
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: frameCount
                )
            else {
                continue
            }

            buffer.frameLength = frameCount

            // Fill with low-amplitude sine wave
            if let channelData = buffer.floatChannelData {
                for channel in 0..<Int(format.channelCount) {
                    for i in 0..<Int(frameCount) {
                        channelData[channel][i] = sin(Float(i) * 0.01) * 0.1
                    }
                }
            }

            // Should handle different formats
            await manager.streamAudio(buffer)
        }

        await manager.cancel()
    }

    // MARK: - Helper Methods

    private func isRunningOnCI() -> Bool {
        return ProcessInfo.processInfo.environment["CI"] != nil
            || ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != nil
    }
}

// MARK: - Result Alignment Tests

@available(macOS 13.0, iOS 16.0, *)
final class ResultAlignmentProcessorTests: XCTestCase {

    func testAlignmentProcessorInitialization() async {
        let processor = ResultAlignmentProcessor(alignmentTolerance: 0.5)

        let stats = await processor.alignmentStats
        XCTAssertEqual(stats.pendingAsrCount, 0)
        XCTAssertEqual(stats.pendingSpeakerCount, 0)
        XCTAssertEqual(stats.alignedCount, 0)
    }

    func testAddingResults() async {
        let processor = ResultAlignmentProcessor(alignmentTolerance: 0.5)

        // Create mock ASR result
        let asrResult = StreamingTranscriptionResult(
            segmentID: UUID(),
            revision: 1,
            attributedText: AttributedString("Hello world"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 0, preferredTimescale: 16000),
                duration: CMTime(seconds: 2, preferredTimescale: 16000)
            ),
            isFinal: true,
            confidence: 0.9,
            timestamp: Date()
        )

        // Create mock speaker segment
        let speakerSegment = TimedSpeakerSegment(
            speakerId: "1",
            embedding: Array(repeating: 0.1, count: 256),
            startTimeSeconds: 0.5,
            endTimeSeconds: 2.5,
            qualityScore: 0.8
        )

        await processor.addAsrResults([asrResult])
        await processor.addSpeakerSegments([speakerSegment])

        let stats = await processor.alignmentStats
        XCTAssertEqual(stats.pendingAsrCount, 1)
        XCTAssertEqual(stats.pendingSpeakerCount, 1)
    }

    func testBasicAlignment() async {
        let processor = ResultAlignmentProcessor(alignmentTolerance: 1.0)

        // Create overlapping ASR result and speaker segment
        let asrResult = StreamingTranscriptionResult(
            segmentID: UUID(),
            revision: 1,
            attributedText: AttributedString("Test message"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 1, preferredTimescale: 16000),
                duration: CMTime(seconds: 2, preferredTimescale: 16000)
            ),
            isFinal: true,
            confidence: 0.9,
            timestamp: Date()
        )

        let speakerSegment = TimedSpeakerSegment(
            speakerId: "1",
            embedding: Array(repeating: 0.1, count: 256),
            startTimeSeconds: 1.2,
            endTimeSeconds: 2.8,
            qualityScore: 0.8
        )

        await processor.addAsrResults([asrResult])
        await processor.addSpeakerSegments([speakerSegment])

        // Process alignment
        let alignedResults = await processor.processAlignment()

        // Should produce one aligned result
        XCTAssertEqual(alignedResults.count, 1)

        let aligned = alignedResults[0]
        XCTAssertEqual(aligned.speakerId, "1")
        XCTAssertTrue(aligned.speakerConfidence > 0.0)
        XCTAssertTrue(aligned.transcriptionConfidence > 0.0)
    }

    func testNoAlignment() async {
        let processor = ResultAlignmentProcessor(alignmentTolerance: 0.1)  // Very strict

        // Create non-overlapping ASR result and speaker segment with current timestamps
        // to avoid cleanup behavior
        let now = Date()
        let asrResult = StreamingTranscriptionResult(
            segmentID: UUID(),
            revision: 1,
            attributedText: AttributedString("Test message"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 1, preferredTimescale: 16000),
                duration: CMTime(seconds: 1, preferredTimescale: 16000)
            ),
            isFinal: true,
            confidence: 0.9,
            timestamp: now  // Current timestamp
        )

        let speakerSegment = TimedSpeakerSegment(
            speakerId: "1",
            embedding: Array(repeating: 0.1, count: 256),
            startTimeSeconds: Float(now.timeIntervalSince1970 + 10.0),  // Far from ASR result
            endTimeSeconds: Float(now.timeIntervalSince1970 + 11.0),
            qualityScore: 0.8
        )

        await processor.addAsrResults([asrResult])
        await processor.addSpeakerSegments([speakerSegment])

        // Process alignment
        let alignedResults = await processor.processAlignment()

        // Should produce no aligned results due to poor temporal overlap and distance
        XCTAssertEqual(alignedResults.count, 0, "Expected no alignment due to large temporal distance")

        // Note: Results might be cleaned up due to age-based cleanup in the processor
        // So we just verify no alignment occurred rather than checking pending counts
    }

    func testReset() async {
        let processor = ResultAlignmentProcessor(alignmentTolerance: 0.5)

        // Add some results
        let asrResult = StreamingTranscriptionResult(
            segmentID: UUID(),
            revision: 1,
            attributedText: AttributedString("Test"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 0, preferredTimescale: 16000),
                duration: CMTime(seconds: 1, preferredTimescale: 16000)
            ),
            isFinal: true,
            confidence: 0.9,
            timestamp: Date()
        )

        await processor.addAsrResults([asrResult])

        // Verify results are pending
        let beforeStats = await processor.alignmentStats
        XCTAssertEqual(beforeStats.pendingAsrCount, 1)

        // Reset
        await processor.reset()

        // Verify everything is cleared
        let afterStats = await processor.alignmentStats
        XCTAssertEqual(afterStats.pendingAsrCount, 0)
        XCTAssertEqual(afterStats.pendingSpeakerCount, 0)
        XCTAssertEqual(afterStats.alignedCount, 0)
    }
}

// MARK: - DiarizedAsrTypes Tests

@available(macOS 13.0, iOS 16.0, *)
final class DiarizedAsrTypesTests: XCTestCase {

    func testDiarizedTranscriptionResult() {
        let result = DiarizedTranscriptionResult(
            segmentID: UUID(),
            revision: 1,
            speakerId: "speaker_1",
            attributedText: AttributedString("Hello world"),
            audioTimeRange: CMTimeRange(
                start: CMTime(seconds: 1, preferredTimescale: 16000),
                duration: CMTime(seconds: 2, preferredTimescale: 16000)
            ),
            isFinal: true,
            transcriptionConfidence: 0.9,
            speakerConfidence: 0.8,
            timestamp: Date()
        )

        XCTAssertEqual(result.speakerId, "speaker_1")
        XCTAssertTrue(result.isFinal)
        XCTAssertEqual(result.transcriptionConfidence, 0.9)
        XCTAssertEqual(result.speakerConfidence, 0.8)
        XCTAssertEqual(result.combinedConfidence, 0.85)  // Average of 0.9 and 0.8
    }

    func testDiarizedTranscriptSnapshot() {
        let finalized = ["1": AttributedString("Hello"), "2": AttributedString("World")]
        let volatile = ["1": AttributedString("How")]
        let combined = AttributedString("Speaker 1: Hello How\nSpeaker 2: World")
        let active = Set<String>(["1"])
        let timestamp = Date()

        let snapshot = DiarizedTranscriptSnapshot(
            finalizedBySpeaker: finalized,
            volatileBySpeaker: volatile,
            combinedTranscript: combined,
            activeSpeakers: active,
            lastUpdated: timestamp
        )

        XCTAssertEqual(snapshot.finalizedBySpeaker.count, 2)
        XCTAssertEqual(snapshot.volatileBySpeaker.count, 1)
        XCTAssertEqual(snapshot.activeSpeakers, active)
        XCTAssertEqual(snapshot.lastUpdated, timestamp)
    }

    func testSpeakerSessionStats() {
        let stats = SpeakerSessionStats(
            speakerId: "speaker_1",
            totalSpeakingTime: 120.5,
            segmentCount: 10,
            averageSegmentDuration: 12.05,
            averageConfidence: 0.85,
            lastActivity: Date()
        )

        XCTAssertEqual(stats.speakerId, "speaker_1")
        XCTAssertEqual(stats.totalSpeakingTime, 120.5)
        XCTAssertEqual(stats.segmentCount, 10)
        XCTAssertEqual(stats.averageSegmentDuration, 12.05, accuracy: 0.01)
        XCTAssertEqual(stats.averageConfidence, 0.85)
    }

    func testStreamingDiarizedAsrConfigPresets() {
        let defaultConfig = StreamingDiarizedAsrConfig.default
        let accurateConfig = StreamingDiarizedAsrConfig.accurate

        // Default should prioritize real-time performance
        XCTAssertFalse(defaultConfig.enableDebug)
        XCTAssertEqual(defaultConfig.alignmentTolerance, 0.5)

        // Accurate should prioritize quality
        XCTAssertEqual(accurateConfig.alignmentTolerance, 0.3)
        XCTAssertEqual(accurateConfig.diarizerConfig.chunkOverlap, 1.0)

        // Both should have valid configurations
        XCTAssertGreaterThan(defaultConfig.asrConfig.chunkSeconds, 0)
        XCTAssertGreaterThan(accurateConfig.asrConfig.chunkSeconds, 0)
    }

    func testDiarizedAsrErrors() {
        let errors: [DiarizedAsrError] = [
            .asrNotInitialized,
            .diarizerNotInitialized,
            .alignmentFailed("test"),
            .processingFailed("test"),
            .configurationInvalid("test"),
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}
