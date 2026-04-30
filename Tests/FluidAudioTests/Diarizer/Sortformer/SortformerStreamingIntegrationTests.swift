import XCTest

@preconcurrency @testable import FluidAudio

// Note: Import order is not alphabetical due to Swift 6.1 (CI) vs 6.3 (local) formatter incompatibility.
// OrderedImports rule is disabled in .swift-format until GitHub Actions supports Swift 6.3.

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
        XCTAssertEqual(finalChunk?.chunkResult.tentativeFrameCount, 0)
        XCTAssertEqual(finalChunk?.chunkResult.tentativePredictions.count, 0)
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
        XCTAssertEqual(bufferedFinalChunk?.chunkResult.tentativeFrameCount, 0)
        XCTAssertEqual(bufferedDiarizer.timeline.numTentativeFrames, 0)
        XCTAssertEqual(
            bufferedDiarizer.timeline.numFinalizedFrames,
            streamingDiarizer.timeline.numFinalizedFrames
        )
    }

    func testFinalizeSessionIsIdempotent() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try diarizer.process(samples: chunk)
        }

        let firstUpdate = try diarizer.finalizeSession()
        let frameCountAfterFirst = diarizer.timeline.numFinalizedFrames
        let secondUpdate = try diarizer.finalizeSession()

        XCTAssertNotNil(firstUpdate)
        XCTAssertNil(secondUpdate)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, frameCountAfterFirst)
    }

    func testFinalizeSessionDrainsUnprocessedAudioAndMelFeatures() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        // Feed audio without ever calling process(), so the entire buffer is
        // unprocessed audio at finalize time. finalizeSession must drain it.
        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)
        try diarizer.addAudio(samples, sourceSampleRate: nil)
        let update = try diarizer.finalizeSession()

        let reference = SortformerDiarizer(config: config)
        reference.initialize(models: models)
        let referenceTimeline = try reference.processComplete(samples)

        XCTAssertNotNil(update)
        XCTAssertGreaterThan(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(diarizer.timeline.numTentativeFrames, 0)
        XCTAssertLessThanOrEqual(
            abs(diarizer.timeline.numFinalizedFrames - referenceTimeline.numFinalizedFrames),
            1
        )
        // Returned update must cover the freshly-finalized frames produced by
        // the call (drained chunks + absorbed trailing tentative).
        XCTAssertEqual(
            update?.chunkResult.finalizedFrameCount,
            diarizer.timeline.numFinalizedFrames
        )
    }
}
