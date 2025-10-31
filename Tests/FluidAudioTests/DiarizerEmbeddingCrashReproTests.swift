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
        let resourceURL = try resolveFixtureURL(named: fileName)
        return try AudioConverter().resampleAudioFile(resourceURL)
    }

    private func resolveFixtureURL(named fileName: String) throws -> URL {
        let nameURL = URL(fileURLWithPath: fileName)
        let resourceName = nameURL.deletingPathExtension().lastPathComponent
        let resourceExtension = nameURL.pathExtension.isEmpty ? nil : nameURL.pathExtension

        if let bundleURL = Bundle(for: Self.self).url(
            forResource: resourceName,
            withExtension: resourceExtension
        ) {
            return bundleURL
        }

        let testDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidateDirectories: [URL] = [
            testDirectory.deletingLastPathComponent(),
            testDirectory,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
        ]

        for directory in candidateDirectories {
            let candidateURL = directory.appendingPathComponent(fileName, isDirectory: false)
            if FileManager.default.fileExists(atPath: candidateURL.path) {
                return candidateURL
            }
        }

        throw XCTSkip("Skipping concurrency stress test: missing audio fixture \(fileName)")
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
