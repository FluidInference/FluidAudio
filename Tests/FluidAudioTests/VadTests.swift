import XCTest

@testable import FluidAudio

final class VadTests: XCTestCase {

    private let chunkSize = VadManager.chunkSize
    private let sampleRate = Float(VadManager.sampleRate)

    override func setUp() async throws {
        // Skip VAD tests in CI environment where model may not be available
        if ProcessInfo.processInfo.environment["CI"] != nil {
            throw XCTSkip("Skipping VAD tests in CI environment")
        }
    }

    func testVadModelLoading() async throws {
        // Test loading the VAD model
        let config = VadConfig(
            defaultThreshold: 0.5,
            debugMode: true
        )

        let vad: VadManager
        do {
            vad = try await VadManager(config: config)
        } catch {
            // If model loading fails in CI, skip the test
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("VAD model not available in CI")
            }
            throw error
        }
        let isAvailable = await vad.isAvailable
        XCTAssertTrue(isAvailable, "VAD should be available after loading")
    }

    func testVadProcessing() async throws {
        // Test processing audio through the model
        let config = VadConfig(
            defaultThreshold: 0.5,
            debugMode: true
        )

        let vad: VadManager
        do {
            vad = try await VadManager(config: config)
        } catch {
            // If model loading fails in CI, skip the test
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("VAD model not available in CI")
            }
            throw error
        }

        // Test with silence (should return low probability)
        let silenceChunk = Array(repeating: Float(0.0), count: chunkSize)
        let silenceResult = try await vad.processChunk(silenceChunk)

        print("Silence probability: \(silenceResult.probability)")
        XCTAssertLessThan(silenceResult.probability, 0.5, "Silence should have low probability")
        XCTAssertFalse(silenceResult.isVoiceActive, "Silence should not be detected as voice")

        // Test with noise (should return moderate probability)
        let noiseChunk = (0..<chunkSize).map { _ in Float.random(in: -0.1...0.1) }
        let noiseResult = try await vad.processChunk(noiseChunk)

        print("Noise probability: \(noiseResult.probability)")

        // Test with sine wave (simulated tone)
        let sineChunk = (0..<chunkSize).map { i in
            sin(2 * .pi * 440 * Float(i) / sampleRate)
        }
        let sineResult = try await vad.processChunk(sineChunk)

        print("Sine wave probability: \(sineResult.probability)")

        // Processing time should be reasonable
        XCTAssertLessThan(silenceResult.processingTime, 1.0, "Processing should be fast")
    }

    func testVadBatchProcessing() async throws {
        let config = VadConfig(
            defaultThreshold: 0.5,
            debugMode: false
        )

        let vad: VadManager
        do {
            vad = try await VadManager(config: config)
        } catch {
            // If model loading fails in CI, skip the test
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("VAD model not available in CI")
            }
            throw error
        }

        // Create batch of different audio types
        let chunks: [[Float]] = [
            Array(repeating: Float(0.0), count: chunkSize),  // Silence
            (0..<chunkSize).map { _ in Float.random(in: -0.1...0.1) },  // Noise
            (0..<chunkSize).map { i in sin(2 * .pi * 440 * Float(i) / sampleRate) },  // Tone
        ]

        var results: [VadResult] = []
        var state: VadState? = nil
        for chunk in chunks {
            let result = try await vad.processChunk(chunk, inputState: state)
            results.append(result)
            state = result.outputState
        }

        XCTAssertEqual(results.count, 3, "Should process all chunks")

        // First should be silence
        XCTAssertFalse(results[0].isVoiceActive, "First chunk (silence) should not be active")

        print("Batch results:")
        for (i, result) in results.enumerated() {
            print("  Chunk \(i): probability=\(result.probability), active=\(result.isVoiceActive)")
        }
    }

    func testVadStateReset() async throws {
        let config = VadConfig(defaultThreshold: 0.1)
        let vad: VadManager
        do {
            vad = try await VadManager(config: config)
        } catch {
            // If model loading fails in CI, skip the test
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("VAD model not available in CI")
            }
            throw error
        }

        // Process some chunks
        let chunk = Array(repeating: Float(0.0), count: chunkSize)
        _ = try await vad.processChunk(chunk)

        // No state to reset anymore - VAD is stateless
        // Just verify it still works with subsequent calls
        let result = try await vad.processChunk(chunk)
        XCTAssertNotNil(result, "Should process subsequent chunks")
    }

    func testVadPaddingAndTruncation() async throws {
        let config = VadConfig(
            defaultThreshold: 0.5,
            debugMode: true
        )

        let vad: VadManager
        do {
            vad = try await VadManager(config: config)
        } catch {
            // If model loading fails in CI, skip the test
            if ProcessInfo.processInfo.environment["CI"] != nil {
                throw XCTSkip("VAD model not available in CI")
            }
            throw error
        }

        // Test with short chunk (should pad)
        let shortChunk = Array(repeating: Float(0.0), count: chunkSize / 2)
        let shortResult = try await vad.processChunk(shortChunk)
        XCTAssertNotNil(shortResult, "Should handle short chunks")

        // Test with long chunk (should truncate)
        let longChunk = Array(repeating: Float(0.0), count: chunkSize * 2)
        let longResult = try await vad.processChunk(longChunk)
        XCTAssertNotNil(longResult, "Should handle long chunks")
    }

    // MARK: - Edge Case Tests

    func testVadWithEmptyAudio() async throws {
        let vad = try await VadManager(config: .default)

        // Test with empty array
        let emptyChunk: [Float] = []
        let result = try await vad.processChunk(emptyChunk)
        XCTAssertNotNil(result, "Should handle empty audio")
        XCTAssertFalse(result.isVoiceActive, "Empty audio should not be active")
    }

    func testVadWithExtremeValues() async throws {
        let vad = try await VadManager(config: .default)

        // Test with maximum values
        let maxChunk = Array(repeating: Float(1.0), count: chunkSize)
        let maxResult = try await vad.processChunk(maxChunk)
        XCTAssertNotNil(maxResult, "Should handle maximum values")

        // Test with minimum values
        let minChunk = Array(repeating: Float(-1.0), count: chunkSize)
        let minResult = try await vad.processChunk(minChunk)
        XCTAssertNotNil(minResult, "Should handle minimum values")

        // Test with alternating extremes
        let alternatingChunk = (0..<chunkSize).map { i in
            i % 2 == 0 ? Float(1.0) : Float(-1.0)
        }
        let alternatingResult = try await vad.processChunk(alternatingChunk)
        XCTAssertNotNil(alternatingResult, "Should handle alternating extreme values")
    }

    func testVadWithNaNAndInfinity() async throws {
        let vad = try await VadManager(config: .default)

        // Test with NaN values (should be handled gracefully)
        var nanChunk = Array(repeating: Float(0.0), count: chunkSize)
        nanChunk[chunkSize / 2] = Float.nan
        let nanResult = try await vad.processChunk(nanChunk)
        XCTAssertNotNil(nanResult, "Should handle NaN values")
        XCTAssertFalse(nanResult.probability.isNaN, "Result should not be NaN")

        // Test with infinity values
        var infChunk = Array(repeating: Float(0.0), count: chunkSize)
        infChunk[chunkSize / 2] = Float.infinity
        let infResult = try await vad.processChunk(infChunk)
        XCTAssertNotNil(infResult, "Should handle infinity values")
        XCTAssertFalse(infResult.probability.isInfinite, "Result should not be infinite")
    }

    // MARK: - Performance Tests

    func testVadPerformance() async throws {
        let vad = try await VadManager(config: .default)
        let chunk = Array(repeating: Float(0.0), count: chunkSize)
        let chunkDuration = Double(chunkSize) / Double(sampleRate)

        // Measure single chunk processing time
        let startTime = Date()
        _ = try await vad.processChunk(chunk)
        let singleChunkTime = Date().timeIntervalSince(startTime)

        // Should maintain at least 5x real-time processing
        let singleChunkRTF = singleChunkTime / chunkDuration
        XCTAssertLessThan(singleChunkRTF, 0.2, "Single chunk should process at ≥5x real-time")

        // Measure batch processing time
        let batchSize = 100
        let chunks = Array(repeating: chunk, count: batchSize)

        let batchStartTime = Date()
        var state: VadState? = nil
        for chunk in chunks {
            let result = try await vad.processChunk(chunk, inputState: state)
            state = result.outputState
        }
        let batchTime = Date().timeIntervalSince(batchStartTime)

        // Batch should be reasonably efficient
        let avgTimePerChunk = batchTime / Double(batchSize)
        let avgRTF = avgTimePerChunk / chunkDuration
        XCTAssertLessThan(avgRTF, 0.2, "Average time per chunk in batch should maintain ≥5x real-time")
    }

    func testVadRealTimeFactorPerformance() async throws {
        let vad = try await VadManager(config: .default)

        // 4096 samples at 16kHz = 256ms of audio
        let audioDurationSeconds = Double(chunkSize) / Double(sampleRate)
        let chunk = Array(repeating: Float(0.0), count: chunkSize)

        let startTime = Date()
        _ = try await vad.processChunk(chunk)
        let processingTime = Date().timeIntervalSince(startTime)

        let rtf = processingTime / audioDurationSeconds

        // Should achieve at least 5x real-time factor (more realistic on various hardware)
        XCTAssertLessThan(rtf, 0.2, "Should achieve at least 5x real-time factor, got \(1/rtf)x")
    }

    // MARK: - Different Audio Condition Tests

    func testVadWithDifferentFrequencies() async throws {
        let vad = try await VadManager(config: .default)
        let sampleRate = self.sampleRate

        // Test low frequency (100 Hz - below typical speech)
        let lowFreqChunk = (0..<chunkSize).map { i in
            sin(2 * .pi * 100 * Float(i) / sampleRate)
        }
        let lowFreqResult = try await vad.processChunk(lowFreqChunk)

        // Test mid frequency (1000 Hz - speech range)
        let midFreqChunk = (0..<chunkSize).map { i in
            sin(2 * .pi * 1000 * Float(i) / sampleRate)
        }
        let midFreqResult = try await vad.processChunk(midFreqChunk)

        // Test high frequency (8000 Hz - upper speech range)
        let highFreqChunk = (0..<chunkSize).map { i in
            sin(2 * .pi * 8000 * Float(i) / sampleRate)
        }
        let highFreqResult = try await vad.processChunk(highFreqChunk)

        // All should process without error
        XCTAssertNotNil(lowFreqResult)
        XCTAssertNotNil(midFreqResult)
        XCTAssertNotNil(highFreqResult)
    }

    func testVadWithVaryingAmplitudes() async throws {
        let vad = try await VadManager(config: .default)

        // Very quiet signal
        let quietChunk = (0..<chunkSize).map { _ in Float.random(in: -0.001...0.001) }
        let quietResult = try await vad.processChunk(quietChunk)
        XCTAssertFalse(quietResult.isVoiceActive, "Very quiet signal should not be active")

        // Moderate signal
        let moderateChunk = (0..<chunkSize).map { _ in Float.random(in: -0.1...0.1) }
        let moderateResult = try await vad.processChunk(moderateChunk)
        XCTAssertNotNil(moderateResult)

        // Loud signal
        let loudChunk = (0..<chunkSize).map { _ in Float.random(in: -0.9...0.9) }
        let loudResult = try await vad.processChunk(loudChunk)
        XCTAssertNotNil(loudResult)
    }

    func testVadWithTransientSpikes() async throws {
        let vad = try await VadManager(config: .default)

        // Mostly silence with sudden spike
        var spikeChunk = Array(repeating: Float(0.0), count: chunkSize)
        for i in 200..<210 {
            spikeChunk[i] = Float.random(in: 0.8...1.0)
        }

        let spikeResult = try await vad.processChunk(spikeChunk)
        XCTAssertNotNil(spikeResult, "Should handle transient spikes")
    }

    // MARK: - Concurrent Processing Tests

    func testVadConcurrentProcessing() async throws {
        let vad = try await VadManager(config: .default)
        let chunk = Array(repeating: Float(0.0), count: chunkSize)

        // Process multiple chunks concurrently
        let results = await withTaskGroup(of: VadResult?.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    try? await vad.processChunk(chunk)
                }
            }

            var results: [VadResult] = []
            for await result in group {
                if let result = result {
                    results.append(result)
                }
            }
            return results
        }

        XCTAssertEqual(results.count, 10, "All concurrent tasks should complete")

        // All results should be consistent for the same input
        let firstProbability = results[0].probability
        for result in results {
            XCTAssertEqual(
                result.probability, firstProbability, accuracy: 0.001,
                "Concurrent processing should yield consistent results")
        }
    }

    func testVadThreadSafety() async throws {
        let vad = try await VadManager(config: .default)

        // Create different chunks for variety
        let chunks: [[Float]] = (0..<100).map { i in
            let amplitude = Float(i) / 100.0
            return (0..<chunkSize).map { _ in Float.random(in: -amplitude...amplitude) }
        }

        // Process concurrently and verify no crashes or data races
        let results = await withTaskGroup(of: (Int, VadResult?).self) { group in
            for (index, chunk) in chunks.enumerated() {
                group.addTask {
                    let result = try? await vad.processChunk(chunk)
                    return (index, result)
                }
            }

            var results: [(Int, VadResult)] = []
            for await (index, result) in group {
                if let result = result {
                    results.append((index, result))
                }
            }
            return results
        }

        XCTAssertEqual(results.count, chunks.count, "All chunks should be processed")
    }

    // MARK: - Configuration Tests

    func testVadWithDifferentThresholds() async throws {
        let chunk = (0..<chunkSize).map { _ in Float.random(in: -0.1...0.1) }

        // Test with low threshold
        let lowThresholdVad = try await VadManager(config: VadConfig(defaultThreshold: 0.1))
        let lowResult = try await lowThresholdVad.processChunk(chunk)

        // Test with high threshold
        let highThresholdVad = try await VadManager(config: VadConfig(defaultThreshold: 0.9))
        let highResult = try await highThresholdVad.processChunk(chunk)

        // With same input, low threshold should be more likely to detect voice
        if lowResult.probability == highResult.probability {
            // Probabilities are the same, so threshold affects isVoiceActive
            XCTAssertTrue(
                lowResult.isVoiceActive || !highResult.isVoiceActive,
                "Low threshold should be more permissive"
            )
        }
    }

    func testVadConfigurationAccessibility() async throws {
        let config = VadConfig(defaultThreshold: 0.3, debugMode: true)
        let vad = try await VadManager(config: config)

        let currentConfig = await vad.currentConfig
        XCTAssertEqual(currentConfig.defaultThreshold, 0.3, "Should maintain configured threshold")
        XCTAssertEqual(currentConfig.debugMode, true, "Should maintain debug mode")
    }

    // Segmentation tests moved entirely to VadSegmentationTests.

    func testExtractSpeechSegmentsWithMultipleSegments() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        let pattern: [(Bool, Double)] = [(false, 1.0), (true, 2.0), (false, 1.0), (true, 2.0), (false, 1.0)]
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertEqual(segments.count, 2, "Should produce two separate speech segments")

        for segment in segments {
            let segmentDuration = segment.endTime - segment.startTime
            XCTAssertLessThan(segmentDuration, 15.0, "Each segment should be under 15s")
        }
    }

    // MARK: - Segment Merging Tests

    func testSegmentMergingWithinMinSilenceDuration() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        // 1s speech + 0.5s silence + 1s speech (should merge)
        let pattern: [(Bool, Double)] = [(true, 1.0), (false, 0.5), (true, 1.0)]
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertEqual(segments.count, 1, "Segments with <750ms gap should merge into one segment")
        if let seg = segments.first {
            let dur = seg.endTime - seg.startTime
            // Merged segment should include both speech segments plus the silence gap: ~2.5s
            XCTAssertGreaterThan(dur, 2.4)
            XCTAssertLessThan(dur, 2.6)
        }
    }

    func testSegmentNotMergingBeyondMinSilenceDuration() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, minSilenceDuration: 0.75)
        // 1s speech + 1s silence + 1s speech (should NOT merge)
        let pattern: [(Bool, Double)] = [(true, 1.0), (false, 1.0), (true, 1.0)]
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertEqual(segments.count, 2, "Segments with >750ms gap should remain separate")
        for seg in segments {
            let dur = seg.endTime - seg.startTime
            XCTAssertGreaterThan(dur, 0.9)
            XCTAssertLessThan(dur, 1.3)
        }
    }

    func testMinSpeechDurationFiltering() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.5, minSilenceDuration: 0.75)
        let pattern: [(Bool, Double)] = [(true, 0.2), (false, 1.0), (true, 0.8), (false, 1.0), (true, 0.1)]
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertEqual(segments.count, 1, "Only segments ≥500ms should be kept")
        if let segment = segments.first {
            let duration = segment.endTime - segment.startTime
            XCTAssertGreaterThan(duration, 0.7)
            XCTAssertLessThan(duration, 1.1)
        }
    }

    // MARK: - Long Segment Splitting Tests

    func testSplitLongContinuousSpeech() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 15.0)
        let (vadResults, totalSamples) = makeVadResults([(true, 30.0)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(segments.count, 2, "30s speech should split into multiple segments")
        for (index, seg) in segments.enumerated() {
            let dur = seg.endTime - seg.startTime
            XCTAssertLessThan(dur, 15.1, "Segment \(index) should be under 15s")
        }
    }

    func testMaxSpeechDurationEnforcement() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 10.0)
        let (vadResults, totalSamples) = makeVadResults([(true, 25.0)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(segments.count, 3, "25s speech with 10s max should create at least 3 segments")
        for (index, segment) in segments.enumerated() {
            let duration = segment.endTime - segment.startTime
            XCTAssertLessThan(duration, 10.1, "Segment \(index) must be under 10s")
        }
    }

    func testSplitAtOrBeforeMaxDuration() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 15.0)
        let (vadResults, totalSamples) = makeVadResults([(true, 16.0)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(segments.count, 2, "16s speech should split into at least 2 segments")
        for seg in segments {
            let dur = seg.endTime - seg.startTime
            XCTAssertLessThanOrEqual(dur, 15.1)
        }
    }

    func testRealWorldScenario120Seconds() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(
            minSpeechDuration: 0.15, minSilenceDuration: 0.75, maxSpeechDuration: 15.0)
        let pattern: [(Bool, Double)] = [(true, 5.0), (false, 85.0), (true, 30.0)]
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(segments.count, 3)
        XCTAssertLessThanOrEqual(segments.count, 4)
        for seg in segments { XCTAssertLessThan(seg.endTime - seg.startTime, 15.1) }
        XCTAssertGreaterThan(segments.first?.endTime ?? 0 - (segments.first?.startTime ?? 0), 4.9)
    }

    // MARK: - Edge Cases and Configuration Tests

    func testEmptyAudioSegmentation() async throws {
        let vad = try await VadManager(config: .default)
        let segments = await vad.segmentSpeech(from: [], totalSamples: 0)
        XCTAssertTrue(segments.isEmpty, "Empty audio should produce no segments")
    }

    func testVeryShortAudio() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15)
        let (vadResults, totalSamples) = makeVadResults([(true, 0.05)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertTrue(segments.isEmpty, "Audio below minimum duration should produce no segments")
    }

    func testExactlyMaxDurationSegment() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.15, maxSpeechDuration: 5.0)
        let (vadResults, totalSamples) = makeVadResults([(true, 5.0)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertFalse(segments.isEmpty, "Should produce at least one segment")
        for (index, segment) in segments.enumerated() {
            let duration = segment.endTime - segment.startTime
            XCTAssertLessThanOrEqual(duration, 5.1, "Segment \(index) should not exceed max duration")
        }
    }

    func testAlternatingSpeechSilence() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.1, minSilenceDuration: 0.2)
        let pairs = 5
        var pattern: [(Bool, Double)] = []
        for _ in 0..<pairs {
            pattern.append((true, 0.3))
            pattern.append((false, 0.3))
        }
        let (vadResults, totalSamples) = makeVadResults(pattern)
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(
            segments.count, 1, "Chunk resolution may merge short silences, but at least one segment should remain")
    }

    func testCustomSegmentationConfig() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(
            minSpeechDuration: 1.0,
            minSilenceDuration: 2.0,
            maxSpeechDuration: 8.0,
            speechPadding: 0.2,
            silenceThresholdForSplit: 0.5
        )
        let (vadResults, totalSamples) = makeVadResults([(true, 20.0)])
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        XCTAssertGreaterThanOrEqual(segments.count, 3, "Should split into at least 3 segments with 8s max")
        for (index, segment) in segments.enumerated() {
            let duration = segment.endTime - segment.startTime
            XCTAssertLessThan(duration, 8.1, "Segment \(index) should be under 8s")
        }
    }

    func testSpeechPaddingApplication() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig(minSpeechDuration: 0.25, speechPadding: 0.2)

        let speechDuration = 2.0
        let paddingDuration = 0.2
        let silenceDuration = 1.0

        // Create pattern: 1s silence + 2s speech + 1s silence
        let pattern: [(Bool, Double)] = [(false, silenceDuration), (true, speechDuration), (false, silenceDuration)]
        let (vadResults, totalSamples) = makeVadResults(pattern)

        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)

        XCTAssertEqual(segments.count, 1, "Should produce one segment")

        if let segment = segments.first {
            let segmentDuration = segment.endTime - segment.startTime
            // Should be speech duration + padding on both sides (but limited by audio bounds)
            let expectedMinDuration = speechDuration
            let expectedMaxDuration = speechDuration + paddingDuration * 2 + 0.1  // Some tolerance

            XCTAssertGreaterThan(segmentDuration, expectedMinDuration, "Segment should be at least speech duration")
            XCTAssertLessThan(segmentDuration, expectedMaxDuration, "Segment shouldn't be too much longer")
        }
    }

    // MARK: - Performance Tests for Segmentation

    func testSegmentationPerformance() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig()

        // Create pattern: 12 alternating 5-second speech/silence segments for 60 seconds total
        let pattern: [(Bool, Double)] = [
            (true, 5.0), (false, 5.0), (true, 5.0), (false, 5.0),
            (true, 5.0), (false, 5.0), (true, 5.0), (false, 5.0),
            (true, 5.0), (false, 5.0), (true, 5.0), (false, 5.0),
        ]
        let (vadResults, totalSamples) = makeVadResults(pattern)

        let audioDuration = 60.0  // 1 minute of audio

        let startTime = Date()
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        let processingTime = Date().timeIntervalSince(startTime)

        XCTAssertFalse(segments.isEmpty, "Should produce segments")

        // Should process reasonably fast (allow generous time for CI)
        let processingRTF = processingTime / audioDuration
        XCTAssertLessThan(processingRTF, 0.5, "Processing should be faster than 0.5x real-time, got \(processingRTF)x")

        print(
            "Segmentation performance: \(String(format: "%.3f", processingRTF))x real-time for \(segments.count) segments"
        )
    }

    func testSegmentationWithLargeAudio() async throws {
        let vad = try await VadManager(config: .default)
        let segConfig = VadSegmentationConfig()

        // Create pattern: alternating 10-second speech and 20-second silence for 5 minutes
        // Every 3rd chunk (30s) has 10s speech, creating sparse speech pattern
        let pattern: [(Bool, Double)] = [
            (true, 10.0), (false, 20.0), (true, 10.0), (false, 20.0),
            (true, 10.0), (false, 20.0), (true, 10.0), (false, 20.0),
            (true, 10.0), (false, 20.0), (true, 10.0), (false, 20.0),
            (true, 10.0), (false, 20.0), (true, 10.0), (false, 20.0),
            (true, 10.0), (false, 20.0),
        ]
        let (vadResults, totalSamples) = makeVadResults(pattern)

        let audioDuration = 300.0  // 5 minutes of audio

        let startTime = Date()
        let segments = await vad.segmentSpeech(from: vadResults, totalSamples: totalSamples, config: segConfig)
        let processingTime = Date().timeIntervalSince(startTime)

        XCTAssertFalse(segments.isEmpty, "Should produce segments from large audio")

        // All segments should respect max duration
        for (index, segment) in segments.enumerated() {
            let segmentDuration = segment.endTime - segment.startTime
            XCTAssertLessThan(segmentDuration, 15.1, "Large audio segment \(index) should be under 15s")
        }

        print("Large audio segmentation: \(processingTime)s for \(audioDuration)s audio (\(segments.count) segments)")
    }

}
