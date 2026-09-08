import os
import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class OfflineDiarizerCancellationTests: XCTestCase {
    func testCancellingFileProcessingStopsAllInferenceWorkers() async throws {
        try requireOfflineDiarizerModels()

        guard
            let audioPath = ProcessInfo.processInfo.environment[
                "FLUIDAUDIO_CANCELLATION_TEST_AUDIO"
            ],
            FileManager.default.fileExists(atPath: audioPath)
        else {
            throw XCTSkip(
                "Set FLUIDAUDIO_CANCELLATION_TEST_AUDIO to a real meeting recording"
            )
        }
        let audioURL = URL(fileURLWithPath: audioPath)

        let manager = OfflineDiarizerManager()
        let progress = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let progressCount = OSAllocatedUnfairLock<Int>(initialState: 0)
        var progressIterator = progress.stream.makeAsyncIterator()

        let processingTask = Task {
            try await manager.process(audioURL) { _, _ in
                progressCount.withLock { $0 += 1 }
                progress.continuation.yield(())
            }
        }

        guard await progressIterator.next() != nil else {
            processingTask.cancel()
            XCTFail("Diarization ended before inference began")
            return
        }

        let cancellationStart = ContinuousClock.now
        processingTask.cancel()

        do {
            _ = try await processingTask.value
            XCTFail("Cancelled diarization unexpectedly completed")
        } catch is CancellationError {
            // Expected. Returning from the structured task group also proves both
            // inference workers have stopped rather than continuing in the background.
        } catch {
            XCTFail("Expected CancellationError, received \(error)")
        }

        let cancellationDuration = cancellationStart.duration(to: .now)
        XCTAssertLessThan(
            cancellationDuration,
            .seconds(2),
            "Cancellation should stop segmentation and embedding between model calls"
        )

        let countWhenCancelledCallReturned = progressCount.withLock { $0 }
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(
            progressCount.withLock { $0 },
            countWhenCancelledCallReturned,
            "No inference progress may continue after process(URL) returns"
        )
    }

    private func requireOfflineDiarizerModels() throws {
        let repoDirectory = OfflineDiarizerModels.defaultModelsDirectory()
            .appendingPathComponent(Repo.diarizer.folderName, isDirectory: true)
        let allPresent = ModelNames.OfflineDiarizer.requiredModels.allSatisfy {
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent($0).path)
        }
        guard allPresent else {
            throw XCTSkip("Offline diarizer models not available")
        }
    }
}
