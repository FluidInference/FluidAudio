import AVFoundation
import XCTest

@preconcurrency @testable import FluidAudio

final class DiarizerEmbeddingCrashReproTests: XCTestCase {

    func testConcurrentDiarizerUsageStress() async throws {
        guard let models = try? await DiarizerModels.download() else {
            throw XCTSkip("Skipping crash reproduction because diarizer models are unavailable.")
        }

        let audioSamples = try loadAudioSamples(named: "journey.wav")
        let manager = DiarizerManager()
        manager.initialize(models: models)

        let queue = DispatchQueue(label: "concurrent-diarizer", qos: .userInitiated, attributes: .concurrent)
        let group = DispatchGroup()
        let concurrentCalls = 12

        for _ in 0..<concurrentCalls {
            group.enter()
            queue.async {
                autoreleasepool {
                    do {
                        _ = try manager.performCompleteDiarization(audioSamples)
                    } catch {
                        // We intentionally ignore errors; the crash happens before completion.
                    }
                }
                group.leave()
            }
        }

        await withCheckedContinuation { continuation in
            group.notify(queue: queue) {
                continuation.resume()
            }
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

private enum CrashReproError: Error {
    case missingAudioFixture(String)
}
