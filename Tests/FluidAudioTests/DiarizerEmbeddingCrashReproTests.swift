import AVFoundation
@preconcurrency @testable import FluidAudio
import XCTest

final class DiarizerEmbeddingCrashReproTests: XCTestCase {

    func testConcurrentDiarizerUsageStress() async throws {
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Skipping crash reproduction because diarizer models are unavailable.")
        }

        let audioSamples = try loadAudioSamples(named: "journey.wav")
        let manager = DiarizerManager()
        manager.initialize(models: models)

        let concurrentCalls = 12

        let errorRecorder = ErrorRecorder()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<concurrentCalls {
                group.addTask {
                    do {
                        _ = try manager.performCompleteDiarization(audioSamples)
                    } catch {
                        await errorRecorder.record(error)
                    }
                }
            }
        }

        let capturedError = await errorRecorder.value()
        if let error = capturedError {
            XCTFail("Diarization failed under concurrency stress: \(error)")
        }
    }

    private func loadAudioSamples(named fileName: String) throws -> [Float] {
        let fileURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(fileName, isDirectory: false)

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw CrashReproError.missingAudioFixture(fileURL.path)
        }

        return try AudioConverter().resampleAudioFile(fileURL)
    }
}

private actor ErrorRecorder {
    private var storedError: Error?

    func record(_ error: Error) {
        if storedError == nil {
            storedError = error
        }
    }

    func value() -> Error? {
        storedError
    }
}

private enum CrashReproError: Error {
    case missingAudioFixture(String)
}
