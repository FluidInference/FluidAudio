import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class VadTests: XCTestCase {

    override func setUp() async throws {
        // Skip VAD tests in CI environment where model may not be available
        if ProcessInfo.processInfo.environment["CI"] != nil {
            throw XCTSkip("Skipping VAD tests in CI environment")
        }
    }

    func testVadModelLoading() async throws {
        // Test loading the VAD model
        let config = VadConfig(
            threshold: 0.5,
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
            threshold: 0.5,
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
        let silenceChunk = Array(repeating: Float(0.0), count: 512)
        let silenceResult = try await vad.processChunk(silenceChunk)

        print("Silence probability: \(silenceResult.probability)")
        XCTAssertLessThan(silenceResult.probability, 0.5, "Silence should have low probability")
        XCTAssertFalse(silenceResult.isVoiceActive, "Silence should not be detected as voice")

        // Test with noise (should return moderate probability)
        let noiseChunk = (0..<512).map { _ in Float.random(in: -0.1...0.1) }
        let noiseResult = try await vad.processChunk(noiseChunk)

        print("Noise probability: \(noiseResult.probability)")

        // Test with sine wave (simulated tone)
        let sineChunk = (0..<512).map { i in
            sin(2 * .pi * 440 * Float(i) / 16000)
        }
        let sineResult = try await vad.processChunk(sineChunk)

        print("Sine wave probability: \(sineResult.probability)")

        // Processing time should be reasonable
        XCTAssertLessThan(silenceResult.processingTime, 1.0, "Processing should be fast")
    }

    func testVadBatchProcessing() async throws {
        let config = VadConfig(
            threshold: 0.5,
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
            Array(repeating: Float(0.0), count: 512),  // Silence
            (0..<512).map { _ in Float.random(in: -0.1...0.1) },  // Noise
            (0..<512).map { i in sin(2 * .pi * 440 * Float(i) / 16000) },  // Tone
        ]

        let results = try await vad.processBatch(chunks)

        XCTAssertEqual(results.count, 3, "Should process all chunks")

        // First should be silence
        XCTAssertFalse(results[0].isVoiceActive, "First chunk (silence) should not be active")

        print("Batch results:")
        for (i, result) in results.enumerated() {
            print("  Chunk \(i): probability=\(result.probability), active=\(result.isVoiceActive)")
        }
    }

    func testVadStateReset() async throws {
        let config = VadConfig(threshold: 0.5)
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
        let chunk = Array(repeating: Float(0.0), count: 512)
        _ = try await vad.processChunk(chunk)

        // No state to reset anymore - VAD is stateless
        // Just verify it still works with subsequent calls
        let result = try await vad.processChunk(chunk)
        XCTAssertNotNil(result, "Should process subsequent chunks")
    }

    func testVadPaddingAndTruncation() async throws {
        let config = VadConfig(
            threshold: 0.5,
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
        let shortChunk = Array(repeating: Float(0.0), count: 256)
        let shortResult = try await vad.processChunk(shortChunk)
        XCTAssertNotNil(shortResult, "Should handle short chunks")

        // Test with long chunk (should truncate)
        let longChunk = Array(repeating: Float(0.0), count: 1024)
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
        let maxChunk = Array(repeating: Float(1.0), count: 512)
        let maxResult = try await vad.processChunk(maxChunk)
        XCTAssertNotNil(maxResult, "Should handle maximum values")

        // Test with minimum values
        let minChunk = Array(repeating: Float(-1.0), count: 512)
        let minResult = try await vad.processChunk(minChunk)
        XCTAssertNotNil(minResult, "Should handle minimum values")

        // Test with alternating extremes
        let alternatingChunk = (0..<512).map { i in
            i % 2 == 0 ? Float(1.0) : Float(-1.0)
        }
        let alternatingResult = try await vad.processChunk(alternatingChunk)
        XCTAssertNotNil(alternatingResult, "Should handle alternating extreme values")
    }

    func testVadWithNaNAndInfinity() async throws {
        let vad = try await VadManager(config: .default)

        // Test with NaN values (should be handled gracefully)
        var nanChunk = Array(repeating: Float(0.0), count: 512)
        nanChunk[256] = Float.nan
        let nanResult = try await vad.processChunk(nanChunk)
        XCTAssertNotNil(nanResult, "Should handle NaN values")
        XCTAssertFalse(nanResult.probability.isNaN, "Result should not be NaN")

        // Test with infinity values
        var infChunk = Array(repeating: Float(0.0), count: 512)
        infChunk[256] = Float.infinity
        let infResult = try await vad.processChunk(infChunk)
        XCTAssertNotNil(infResult, "Should handle infinity values")
        XCTAssertFalse(infResult.probability.isInfinite, "Result should not be infinite")
    }

    // MARK: - Performance Tests

    func testVadPerformance() async throws {
        let vad = try await VadManager(config: .default)
        let chunk = Array(repeating: Float(0.0), count: 512)

        // Measure single chunk processing time
        let startTime = Date()
        _ = try await vad.processChunk(chunk)
        let singleChunkTime = Date().timeIntervalSince(startTime)

        // Should process a single chunk in under 10ms
        XCTAssertLessThan(singleChunkTime, 0.01, "Single chunk should process in under 10ms")

        // Measure batch processing time
        let batchSize = 100
        let chunks = Array(repeating: chunk, count: batchSize)

        let batchStartTime = Date()
        _ = try await vad.processBatch(chunks)
        let batchTime = Date().timeIntervalSince(batchStartTime)

        // Batch should be reasonably efficient
        let avgTimePerChunk = batchTime / Double(batchSize)
        XCTAssertLessThan(avgTimePerChunk, 0.01, "Average time per chunk in batch should be under 10ms")
    }

    func testVadRealTimeFactorPerformance() async throws {
        let vad = try await VadManager(config: .default)

        // 512 samples at 16kHz = 32ms of audio
        let audioDurationSeconds = 512.0 / 16000.0
        let chunk = Array(repeating: Float(0.0), count: 512)

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
        let sampleRate: Float = 16000

        // Test low frequency (100 Hz - below typical speech)
        let lowFreqChunk = (0..<512).map { i in
            sin(2 * .pi * 100 * Float(i) / sampleRate)
        }
        let lowFreqResult = try await vad.processChunk(lowFreqChunk)

        // Test mid frequency (1000 Hz - speech range)
        let midFreqChunk = (0..<512).map { i in
            sin(2 * .pi * 1000 * Float(i) / sampleRate)
        }
        let midFreqResult = try await vad.processChunk(midFreqChunk)

        // Test high frequency (8000 Hz - upper speech range)
        let highFreqChunk = (0..<512).map { i in
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
        let quietChunk = (0..<512).map { _ in Float.random(in: -0.001...0.001) }
        let quietResult = try await vad.processChunk(quietChunk)
        XCTAssertFalse(quietResult.isVoiceActive, "Very quiet signal should not be active")

        // Moderate signal
        let moderateChunk = (0..<512).map { _ in Float.random(in: -0.1...0.1) }
        let moderateResult = try await vad.processChunk(moderateChunk)
        XCTAssertNotNil(moderateResult)

        // Loud signal
        let loudChunk = (0..<512).map { _ in Float.random(in: -0.9...0.9) }
        let loudResult = try await vad.processChunk(loudChunk)
        XCTAssertNotNil(loudResult)
    }

    func testVadWithTransientSpikes() async throws {
        let vad = try await VadManager(config: .default)

        // Mostly silence with sudden spike
        var spikeChunk = Array(repeating: Float(0.0), count: 512)
        for i in 200..<210 {
            spikeChunk[i] = Float.random(in: 0.8...1.0)
        }

        let spikeResult = try await vad.processChunk(spikeChunk)
        XCTAssertNotNil(spikeResult, "Should handle transient spikes")
    }

    // MARK: - Concurrent Processing Tests

    func testVadConcurrentProcessing() async throws {
        let vad = try await VadManager(config: .default)
        let chunk = Array(repeating: Float(0.0), count: 512)

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
            return (0..<512).map { _ in Float.random(in: -amplitude...amplitude) }
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
        let chunk = (0..<512).map { _ in Float.random(in: -0.1...0.1) }

        // Test with low threshold
        let lowThresholdVad = try await VadManager(config: VadConfig(threshold: 0.1))
        let lowResult = try await lowThresholdVad.processChunk(chunk)

        // Test with high threshold
        let highThresholdVad = try await VadManager(config: VadConfig(threshold: 0.9))
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
        let config = VadConfig(threshold: 0.3, debugMode: true)
        let vad = try await VadManager(config: config)

        let currentConfig = await vad.currentConfig
        XCTAssertEqual(currentConfig.threshold, 0.3, "Should maintain configured threshold")
        XCTAssertEqual(currentConfig.debugMode, true, "Should maintain debug mode")
    }
}
