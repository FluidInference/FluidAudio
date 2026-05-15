import os
import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class OfflineDiarizerManagerProgressTests: XCTestCase {

    // MARK: - totalChunks calculation

    func testTotalChunksForDefaultConfig() {
        let config = OfflineDiarizerConfig.default
        // default: samplesPerStep = 160_000 * 0.2 = 32_000
        XCTAssertEqual(config.samplesPerStep, 32_000)

        XCTAssertEqual(totalChunks(sampleCount: 1, config: config), 1)
        XCTAssertEqual(totalChunks(sampleCount: 32_000, config: config), 1)
        XCTAssertEqual(totalChunks(sampleCount: 32_001, config: config), 2)
        XCTAssertEqual(totalChunks(sampleCount: 160_000, config: config), 5)
        XCTAssertEqual(totalChunks(sampleCount: 160_001, config: config), 6)
    }

    func testTotalChunksIsAtLeastOneForZeroSamples() {
        let config = OfflineDiarizerConfig.default
        XCTAssertEqual(totalChunks(sampleCount: 0, config: config), 1)
    }

    // MARK: - Progress callback integration

    func testProgressCallbackFiresAndIsMonotonic() async throws {
        let modelsDir = OfflineDiarizerModels.defaultModelsDirectory()
        guard FileManager.default.fileExists(atPath: modelsDir.path) else {
            throw XCTSkip("Offline diarizer models not available")
        }

        let manager = OfflineDiarizerManager()
        let audio = try DiarizationTestFixtures.fixtureAudio(sampleRate: 16_000)

        let updatesLock = OSAllocatedUnfairLock<[(Int, Int)]>(initialState: [])
        _ = try await manager.process(audio: audio) { chunksProcessed, totalChunks in
            updatesLock.withLock { $0.append((chunksProcessed, totalChunks)) }
        }
        let updates = updatesLock.withLock { $0 }

        XCTAssertFalse(updates.isEmpty, "Progress callback should fire at least once")

        let total = updates[0].1
        XCTAssertGreaterThan(total, 0)

        for update in updates {
            XCTAssertEqual(update.1, total, "totalChunks must be consistent across updates")
            XCTAssertGreaterThan(update.0, 0)
            XCTAssertLessThanOrEqual(update.0, total)
        }

        for i in 1..<updates.count {
            XCTAssertGreaterThanOrEqual(
                updates[i].0, updates[i - 1].0,
                "chunksProcessed must be non-decreasing")
        }

        XCTAssertEqual(updates.last?.0, total, "Final update should reach 100%")
    }

    func testProgressCallbackIsOptional() async throws {
        let modelsDir = OfflineDiarizerModels.defaultModelsDirectory()
        guard FileManager.default.fileExists(atPath: modelsDir.path) else {
            throw XCTSkip("Offline diarizer models not available")
        }

        let manager = OfflineDiarizerManager()
        let audio = try DiarizationTestFixtures.fixtureAudio(sampleRate: 16_000)
        _ = try await manager.process(audio: audio)
    }

    // MARK: - Helpers

    private func totalChunks(sampleCount: Int, config: OfflineDiarizerConfig) -> Int {
        max(1, (sampleCount + config.samplesPerStep - 1) / config.samplesPerStep)
    }
}
