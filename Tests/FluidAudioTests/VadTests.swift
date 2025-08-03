import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class VadTests: XCTestCase {

    var vadManager: VadManager!

    override func setUp() async throws {
        try await super.setUp()

        // Use default config for tests (which now has optimal settings)
        vadManager = VadManager()

        do {
            try await vadManager.initialize()
        } catch {
            throw XCTSkip("VAD models not available, skipping tests: \(error)")
        }
    }

    override func tearDown() async throws {
        await vadManager?.cleanup()
        vadManager = nil
        try await super.tearDown()
    }

    // MARK: - Absolute Probability Range Tests

    func testVadProbabilityRangeForSilence() async throws {
        // Create silent audio (all zeros)
        let silentAudio = [Float](repeating: 0.0, count: 512)

        let result = try await vadManager.processChunk(silentAudio)

        XCTAssertGreaterThanOrEqual(result.probability, 0.0, "Probability should be >= 0")
        XCTAssertLessThanOrEqual(result.probability, 1.0, "Probability should be <= 1")
        XCTAssertLessThan(result.probability, 0.2, "Silent audio should have low probability")
        XCTAssertFalse(result.isVoiceActive, "Silent audio should not be detected as voice")
    }

    func testVadProbabilityRangeForNoise() async throws {
        // Create white noise
        let noise = (0..<512).map { _ in Float.random(in: -0.1...0.1) }

        let result = try await vadManager.processChunk(noise)

        XCTAssertGreaterThanOrEqual(result.probability, 0.0, "Probability should be >= 0")
        XCTAssertLessThanOrEqual(result.probability, 1.0, "Probability should be <= 1")
        XCTAssertLessThan(result.probability, 0.445, "Noise should have probability below threshold")
        XCTAssertFalse(result.isVoiceActive, "Noise should not be detected as voice")
    }

    func testVadProbabilityRangeForSpeechLikeSignal() async throws {
        // Create a simple speech-like signal (sine wave with envelope)
        let frequency: Float = 200.0  // Hz (in speech range)
        let sampleRate: Float = 16000.0
        var speechSignal = [Float](repeating: 0.0, count: 512)

        for i in 0..<512 {
            let time = Float(i) / sampleRate
            let envelope = sin(Float.pi * time * 10.0)  // 10 Hz envelope
            speechSignal[i] = envelope * sin(2.0 * Float.pi * frequency * time) * 0.5
        }

        let result = try await vadManager.processChunk(speechSignal)

        XCTAssertGreaterThanOrEqual(result.probability, 0.0, "Probability should be >= 0")
        XCTAssertLessThanOrEqual(result.probability, 1.0, "Probability should be <= 1")
        // Note: Actual threshold depends on the signal characteristics
        print("Speech-like signal probability: \(result.probability)")
    }

    func testVadThresholdBehavior() async throws {
        // Test that threshold actually affects isVoiceActive
        let testSignal = (0..<512).map { _ in Float.random(in: -0.3...0.3) }

        // Test with different thresholds
        let thresholds: [Float] = [0.1, 0.3, 0.445, 0.7, 0.9]
        var results: [(threshold: Float, probability: Float, isActive: Bool)] = []

        for threshold in thresholds {
            let config = VadConfig(
                threshold: threshold,
                debugMode: false,
                adaptiveThreshold: false  // Disable adaptive for consistent testing
            )
            let tempVadManager = VadManager(config: config)
            try await tempVadManager.initialize()

            let result = try await tempVadManager.processChunk(testSignal)
            results.append((threshold, result.probability, result.isVoiceActive))

            await tempVadManager.cleanup()
        }

        // Verify that isVoiceActive respects threshold
        for result in results {
            if result.probability >= result.threshold {
                XCTAssertTrue(
                    result.isActive,
                    "isVoiceActive should be true when probability (\(result.probability)) >= threshold (\(result.threshold))"
                )
            } else {
                XCTAssertFalse(
                    result.isActive,
                    "isVoiceActive should be false when probability (\(result.probability)) < threshold (\(result.threshold))"
                )
            }
        }
    }

    func testVadConsistencyAcrossMultipleChunks() async throws {
        // Process the same chunk multiple times to ensure consistency
        let testChunk = (0..<512).map { _ in Float.random(in: -0.2...0.2) }

        var probabilities: [Float] = []

        for _ in 0..<5 {
            await vadManager.resetState()  // Reset state between tests
            let result = try await vadManager.processChunk(testChunk)
            probabilities.append(result.probability)
        }

        // Check that probabilities are consistent (within small tolerance)
        let firstProb = probabilities[0]
        for prob in probabilities {
            XCTAssertEqual(
                prob, firstProb, accuracy: 0.01,
                "VAD should produce consistent results for the same input")
        }
    }

    func testVadProcessAudioFile() async throws {
        // Test processing a complete "audio file" (simulated)
        let audioLength = 16000  // 1 second at 16kHz
        let audioData = (0..<audioLength).map { i in
            // Create alternating speech-like and silence segments
            let segmentSize = 1600  // 100ms segments
            let segmentIndex = i / segmentSize
            if segmentIndex % 2 == 0 {
                // "Speech" segment
                return sin(Float(i) * 0.1) * 0.3
            } else {
                // Silence
                return Float.random(in: -0.01...0.01)
            }
        }

        let results = try await vadManager.processAudioFile(audioData)

        // Should have ~31 chunks (16000 samples / 512 samples per chunk)
        XCTAssertGreaterThan(results.count, 20, "Should process multiple chunks")

        // Verify all probabilities are in valid range
        for result in results {
            XCTAssertGreaterThanOrEqual(result.probability, 0.0)
            XCTAssertLessThanOrEqual(result.probability, 1.0)
        }

        // Count voice active segments
        let voiceActiveCount = results.filter { $0.isVoiceActive }.count
        print("Voice active in \(voiceActiveCount) of \(results.count) chunks")
    }

    // MARK: - Configuration Tests

    func testDefaultConfigUsesOptimalThreshold() async throws {
        let defaultConfig = VadConfig()
        XCTAssertEqual(defaultConfig.threshold, 0.445, "Default config should use optimal threshold")
        XCTAssertFalse(defaultConfig.debugMode, "Default config should have debug mode off")
        XCTAssertTrue(defaultConfig.adaptiveThreshold, "Default config should have adaptive threshold on")
    }

    func testDefaultConfigHasOptimalSettings() async throws {
        // Verify that default config now uses optimal settings
        let defaultConfig = VadConfig.default
        XCTAssertEqual(defaultConfig.threshold, 0.445, "Default should use optimal threshold")
        XCTAssertFalse(defaultConfig.debugMode, "Debug mode should be off by default")
        XCTAssertTrue(defaultConfig.adaptiveThreshold, "Adaptive threshold should be on by default")
        XCTAssertTrue(defaultConfig.enableSNRFiltering, "SNR filtering should be on by default")
        XCTAssertEqual(defaultConfig.minSNRThreshold, 6.0, "Should use optimal SNR threshold")
    }
}
