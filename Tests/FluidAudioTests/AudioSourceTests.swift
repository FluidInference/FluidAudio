import XCTest

@testable import FluidAudio

final class AudioSourceTests: XCTestCase {

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

    // Note: Concurrent transcription tests removed in Swift 6 migration.
    // AsrManager is not Sendable (contains CoreML models), so async let / TaskGroup
    // patterns that capture it across concurrency boundaries are not allowed.
    // To test concurrent transcription, create separate AsrManager instances per task.
}
