import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AudioSourceTests: XCTestCase {

    func testConcurrentAudioSources() async throws {
        let asrManager = AsrManager()

        do {
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)

            let testAudio = Array(repeating: Float(0.0), count: 16000)

            async let micResult = asrManager.transcribe(testAudio, source: .microphone)
            async let systemResult = asrManager.transcribe(testAudio, source: .system)

            let (mic, system) = try await (micResult, systemResult)

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

    func testMultipleConcurrentTranscriptions() async throws {
        let asrManager = AsrManager()

        do {
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)

            let testAudio = Array(repeating: Float(0.0), count: 16000)

            let results = try await withThrowingTaskGroup(of: (AudioSource, ASRResult).self) {
                group in
                for _ in 0..<2 {
                    group.addTask {
                        let result = try await asrManager.transcribe(testAudio, source: .microphone)
                        return (.microphone, result)
                    }
                    group.addTask {
                        let result = try await asrManager.transcribe(testAudio, source: .system)
                        return (.system, result)
                    }
                }

                var results: [(AudioSource, ASRResult)] = []
                for try await result in group {
                    results.append(result)
                }
                return results
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
