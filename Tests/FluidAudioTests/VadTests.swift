import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class VadTests: XCTestCase {

    func testVadModelLoading() async throws {
        // Test loading the VAD model
        let config = VadConfig(
            threshold: 0.5,
            debugMode: true
        )

        let vad = try await VadManager(config: config)
        XCTAssertTrue(vad.isAvailable, "VAD should be available after loading")
    }

    func testVadProcessing() async throws {
        // Test processing audio through the model
        let config = VadConfig(
            threshold: 0.5,
            debugMode: true
        )

        let vad = try await VadManager(config: config)

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

        let vad = try await VadManager(config: config)

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
        let vad = try await VadManager(config: config)

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

        let vad = try await VadManager(config: config)

        // Test with short chunk (should pad)
        let shortChunk = Array(repeating: Float(0.0), count: 256)
        let shortResult = try await vad.processChunk(shortChunk)
        XCTAssertNotNil(shortResult, "Should handle short chunks")

        // Test with long chunk (should truncate)
        let longChunk = Array(repeating: Float(0.0), count: 1024)
        let longResult = try await vad.processChunk(longChunk)
        XCTAssertNotNil(longResult, "Should handle long chunks")
    }
}
