@preconcurrency @testable import FluidAudio
import XCTest

@MainActor
final class SortformerStreamingIntegrationTests: XCTestCase {
    private static var cachedModels: SortformerModels?

    private func loadModelsForTest(config: SortformerConfig) async throws -> SortformerModels {
        if let cachedModels = Self.cachedModels {
            return cachedModels
        }

        let models = try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
        Self.cachedModels = models
        return models
    }

    func testFinalizeSessionMatchesProcessCompleteFrameCount() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let streamingDiarizer = SortformerDiarizer(config: config)
        streamingDiarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try streamingDiarizer.process(samples: chunk)
        }
        let finalChunk = try streamingDiarizer.finalizeSession()

        let completeDiarizer = SortformerDiarizer(config: config)
        completeDiarizer.initialize(models: models)
        let expectedTimeline = try completeDiarizer.processComplete(samples)

        XCTAssertLessThanOrEqual(
            abs(streamingDiarizer.timeline.numFinalizedFrames - expectedTimeline.numFinalizedFrames),
            1
        )
        XCTAssertEqual(finalChunk?.tentativeFrameCount, 0)
        XCTAssertEqual(finalChunk?.tentativePredictions.count, 0)
        XCTAssertEqual(streamingDiarizer.timeline.numTentativeFrames, 0)
    }

    func testFinalizeSessionFlushesTentativeTailAfterAddAudioOnly() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let bufferedDiarizer = SortformerDiarizer(config: config)
        bufferedDiarizer.initialize(models: models)
        bufferedDiarizer.addAudio(samples)
        let bufferedFinalChunk = try bufferedDiarizer.finalizeSession()

        let streamingDiarizer = SortformerDiarizer(config: config)
        streamingDiarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try streamingDiarizer.process(samples: chunk)
        }
        let _ = try streamingDiarizer.finalizeSession()

        XCTAssertNotNil(bufferedFinalChunk)
        XCTAssertEqual(bufferedFinalChunk?.tentativeFrameCount, 0)
        XCTAssertEqual(bufferedDiarizer.timeline.numTentativeFrames, 0)
        XCTAssertEqual(
            bufferedDiarizer.timeline.numFinalizedFrames,
            streamingDiarizer.timeline.numFinalizedFrames
        )
    }
}
