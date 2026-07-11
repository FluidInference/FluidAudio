import XCTest

@preconcurrency @testable import FluidAudio

// Note: Import order is not alphabetical due to Swift 6.1 (CI) vs 6.3 (local) formatter incompatibility.
// OrderedImports rule is disabled in .swift-format until GitHub Actions supports Swift 6.3.

/// Integration tests for the actor-isolated Sortformer entry points.
///
/// The actor path is documented to produce byte-identical output to the same sequence
/// of synchronous calls; the equivalence test pins that down chunk by chunk. Model loading
/// follows the same convention as `SortformerStreamingIntegrationTests` and skips cleanly
/// when models are unavailable.
@MainActor
final class SortformerAsyncIntegrationTests: XCTestCase {
    private static var cachedModels: SortformerModels?

    private func loadModelsForTest(config: SortformerConfig) async throws -> SortformerModels {
        if let cachedModels = Self.cachedModels {
            return cachedModels
        }

        let models = try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
        Self.cachedModels = models
        return models
    }

    /// Feeding the same chunk sequence through the async path must produce results identical
    /// to the sync path, chunk for chunk (same values, same ordering) — including the
    /// buffered-drain call (`process()`) and finalization (`finalizeSession()`).
    func testAsyncPathMatchesSyncPathChunkForChunk() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)
        let chunks = DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600])

        let syncDiarizer = SortformerDiarizer(config: config)
        syncDiarizer.initialize(models: models)
        var syncUpdates: [DiarizerTimelineUpdate?] = []
        for chunk in chunks {
            syncUpdates.append(try syncDiarizer.process(samples: chunk))
        }
        let syncDrain = try syncDiarizer.process()
        let syncFinal = try syncDiarizer.finalizeSession()

        let asyncDiarizer = SortformerDiarizerActor(config: config)
        try await asyncDiarizer.initializeFromHuggingFace(computeUnits: .cpuOnly)
        var asyncUpdates: [DiarizerTimelineUpdate?] = []
        for chunk in chunks {
            asyncUpdates.append(try await asyncDiarizer.process(samples: chunk))
        }
        let asyncDrain = try await asyncDiarizer.process()
        let asyncFinal = try await asyncDiarizer.finalizeSession()

        XCTAssertEqual(syncUpdates.count, asyncUpdates.count)
        for (index, (syncUpdate, asyncUpdate)) in zip(syncUpdates, asyncUpdates).enumerated() {
            assertUpdatesEqual(syncUpdate, asyncUpdate, context: "chunk \(index)")
        }
        assertUpdatesEqual(syncDrain, asyncDrain, context: "buffered drain")
        assertUpdatesEqual(syncFinal, asyncFinal, context: "finalize")

        let asyncTimeline = await asyncDiarizer.timelineSnapshot()
        XCTAssertEqual(syncDiarizer.timeline.numFinalizedFrames, asyncTimeline.numFinalizedFrames)
        XCTAssertEqual(
            syncDiarizer.timeline.finalizedPredictions,
            asyncTimeline.finalizedPredictions,
            "final prediction buffers must be byte-identical")
        XCTAssertEqual(
            syncDiarizer.timeline.tentativePredictions,
            asyncTimeline.tentativePredictions,
            "final tentative buffers must be byte-identical")

        let syncSpeakers = syncDiarizer.timeline.speakers
        let asyncSpeakers = asyncTimeline.speakers
        XCTAssertEqual(Set(syncSpeakers.keys), Set(asyncSpeakers.keys), "speaker slots must match")
        for (slot, syncSpeaker) in syncSpeakers {
            XCTAssertEqual(
                syncSpeaker.finalizedSegments,
                asyncSpeakers[slot]?.finalizedSegments,
                "speaker \(slot) finalized segments must match")
            XCTAssertEqual(
                syncSpeaker.tentativeSegments,
                asyncSpeakers[slot]?.tentativeSegments,
                "speaker \(slot) tentative segments must match")
        }
    }

    /// The actor entry points check `Task.checkCancellation()` before running any work, so a
    /// cancelled task must observe `CancellationError` — not `SortformerError.notInitialized`,
    /// which is what the uninitialized diarizer would throw if inference were attempted.
    /// (No model gating needed: the call must fail before touching diarizer state.)
    ///
    /// Mid-inference cancellation is intentionally not asserted here: once a forward pass has
    /// started on the queue it runs to completion (CoreML predictions are not interruptible),
    /// so there is no cheap, deterministic observation point for it.
    func testActorProcessThrowsCancellationErrorBeforeInference() async throws {
        let diarizer = SortformerDiarizerActor()

        let task = Task {
            // Ensure the body observes cancellation deterministically even if it starts
            // before `cancel()` lands.
            while !Task.isCancelled {
                await Task.yield()
            }
            return try await diarizer.process(samples: [Float](repeating: 0, count: 1_600))
        }
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("Expected CancellationError from a cancelled task")
        } catch is CancellationError {
            // Expected: cancellation is detected before inference begins.
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
    }

    // MARK: - Helpers

    /// Asserts two timeline updates are identical: same segments (order-sensitive) and the
    /// same chunk-result payload, element for element.
    private func assertUpdatesEqual(
        _ lhs: DiarizerTimelineUpdate?,
        _ rhs: DiarizerTimelineUpdate?,
        context: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs == nil, rhs == nil, "[\(context)] update presence", file: file, line: line)
        guard let lhs, let rhs else { return }

        XCTAssertEqual(
            lhs.finalizedSegments, rhs.finalizedSegments,
            "[\(context)] finalizedSegments", file: file, line: line)
        XCTAssertEqual(
            lhs.tentativeSegments, rhs.tentativeSegments,
            "[\(context)] tentativeSegments", file: file, line: line)
        XCTAssertEqual(
            lhs.chunkResult.startFrame, rhs.chunkResult.startFrame,
            "[\(context)] chunkResult.startFrame", file: file, line: line)
        XCTAssertEqual(
            lhs.chunkResult.finalizedFrameCount, rhs.chunkResult.finalizedFrameCount,
            "[\(context)] chunkResult.finalizedFrameCount", file: file, line: line)
        XCTAssertEqual(
            lhs.chunkResult.finalizedPredictions, rhs.chunkResult.finalizedPredictions,
            "[\(context)] chunkResult.finalizedPredictions", file: file, line: line)
        XCTAssertEqual(
            lhs.chunkResult.tentativeFrameCount, rhs.chunkResult.tentativeFrameCount,
            "[\(context)] chunkResult.tentativeFrameCount", file: file, line: line)
        XCTAssertEqual(
            lhs.chunkResult.tentativePredictions, rhs.chunkResult.tentativePredictions,
            "[\(context)] chunkResult.tentativePredictions", file: file, line: line)
    }
}
