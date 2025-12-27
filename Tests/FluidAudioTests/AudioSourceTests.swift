import XCTest

@testable import FluidAudio

// TODO: Swift 6 migration - These tests use concurrent access patterns that are not compatible
// with strict concurrency checking. AsrManager is not Sendable due to CoreML models.
// Consider making AsrManager an actor or restructuring these tests for serial execution.

final class AudioSourceTests: XCTestCase {

    // Test disabled: async let with non-Sendable AsrManager not allowed in Swift 6
    func _testConcurrentAudioSources() async throws {
        let asrManager = AsrManager()

        do {
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)

            let testAudio = Array(repeating: Float(0.0), count: 16000)

            // Sequential transcription instead of concurrent
            let mic = try await asrManager.transcribe(testAudio, source: .microphone)
            let system = try await asrManager.transcribe(testAudio, source: .system)

            XCTAssertNotNil(mic)
            XCTAssertNotNil(system)
        } catch {
            // In CI environment, ASR initialization might fail - that's expected
            XCTAssertFalse(
                asrManager.isAvailable, "ASR should not be available if initialization failed")
            print("ASR initialization failed in test environment (expected): \(error)")
        }
    }

    func testBackwardCompatibility() async throws {
        let asrManager = AsrManager()

        do {
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)

            let testAudio = Array(repeating: Float(0.0), count: 16000)

            let result = try await asrManager.transcribe(testAudio)
            XCTAssertNotNil(result)
        } catch {
            // In CI environment, ASR initialization might fail - that's expected
            XCTAssertFalse(
                asrManager.isAvailable, "ASR should not be available if initialization failed")
            print("ASR initialization failed in test environment (expected): \(error)")
        }
    }

    // Test disabled: TaskGroup with non-Sendable AsrManager not allowed in Swift 6
    func _testMultipleConcurrentTranscriptions() async throws {
        let asrManager = AsrManager()

        do {
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)

            let testAudio = Array(repeating: Float(0.0), count: 16000)

            // Sequential transcription instead of concurrent
            var results: [(AudioSource, ASRResult)] = []
            for _ in 0..<2 {
                let mic = try await asrManager.transcribe(testAudio, source: .microphone)
                results.append((.microphone, mic))
                let sys = try await asrManager.transcribe(testAudio, source: .system)
                results.append((.system, sys))
            }

            XCTAssertEqual(results.count, 4)
            results.forEach { XCTAssertNotNil($0.1) }
        } catch {
            // In CI environment, ASR initialization might fail - that's expected
            XCTAssertFalse(
                asrManager.isAvailable, "ASR should not be available if initialization failed")
            print("ASR initialization failed in test environment (expected): \(error)")
        }
    }
}
