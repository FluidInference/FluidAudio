import XCTest

@preconcurrency @testable import FluidAudio

// Note: Import order is not alphabetical due to Swift 6.1 (CI) vs 6.3 (local) formatter incompatibility.
// OrderedImports rule is disabled in .swift-format until GitHub Actions supports Swift 6.3.

/// A complete buffer gets a prediction for every frame (#1085). The final chunk takes whatever
/// right context is left, like NeMo's `streaming_feat_loader`. Waiting for a full right context
/// dropped audio shorter than one chunk entirely (30.4 s for `highContextV2_1`) and up to a chunk
/// at the end of longer audio.
///
/// Uses a real 19.8 s recording from the ASR test fixtures (16 kHz mono, cleared for public release;
/// see `SlidingWindowFinalWindowRegressionTests`).
@MainActor
final class SortformerTailCoverageTests: XCTestCase {

    private func fixtureSamples() throws -> [Float] {
        let name = "02-release-readiness-19.8s.wav"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        return try AudioConverter(sampleRate: 16_000).resampleAudioFile(url)
    }

    private func loadModels(config: SortformerConfig) async throws -> SortformerModels {
        do {
            return try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
    }

    /// Mel frames of the batch (center-padded) stream.
    private func melFrameCount(_ samples: [Float]) -> Int {
        AudioMelSpectrogram().computeFlatTransposed(audio: samples).melLength
    }

    /// Output frames that cover every mel frame: one per `subsamplingFactor` mel frames, rounded up.
    private func expectedFrames(_ samples: [Float], config: SortformerConfig) -> Int {
        (melFrameCount(samples) + config.subsamplingFactor - 1) / config.subsamplingFactor
    }

    // MARK: - Chunking (no models)

    func testFeatureLoaderGivesAudioShorterThanAChunkOneChunk() throws {
        let config = SortformerConfig.highContextV2_1
        let samples = try fixtureSamples()
        let melFrames = melFrameCount(samples)
        XCTAssertLessThan(melFrames, config.coreFrames + config.chunkRightContext * config.subsamplingFactor)

        var loader = SortformerFeatureLoader(config: config, audio: samples)
        XCTAssertEqual(loader.numChunks, 1)
        guard let chunk = loader.next() else {
            return XCTFail("Audio shorter than one chunk produced no chunk")
        }
        XCTAssertEqual(chunk.chunkLength, melFrames)
        XCTAssertEqual(chunk.leftOffset, 0)
        XCTAssertEqual(chunk.rightOffset, 0)
        XCTAssertNil(loader.next())
    }

    func testFeatureLoaderChunkCoresCoverEveryFrame() throws {
        let config = SortformerConfig.default
        let samples = try fixtureSamples()
        let melFrames = melFrameCount(samples)
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor

        var loader = SortformerFeatureLoader(config: config, audio: samples)
        var coreFrames = 0
        var chunks = 0
        var lastRightOffset = -1
        while let chunk = loader.next() {
            coreFrames += chunk.chunkFeatures.count / config.melFeatures - chunk.leftOffset - chunk.rightOffset
            XCTAssertLessThanOrEqual(chunk.rightOffset, rightContextFrames)
            lastRightOffset = chunk.rightOffset
            chunks += 1
        }
        XCTAssertEqual(coreFrames, melFrames)
        XCTAssertEqual(chunks, loader.numChunks)
        XCTAssertEqual(lastRightOffset, 0)
    }

    func testFinalizingYieldsTheFinalPartialChunk() throws {
        let config = SortformerConfig.highContextV2_1
        let samples = try fixtureSamples()
        let diarizer = SortformerDiarizer(config: config)
        diarizer.addAudio(samples)
        // While audio can still arrive, a chunk waits for its full right context.
        XCTAssertNil(diarizer.getNextChunkFeatures())

        diarizer.padAndEmitRemainingMel()
        XCTAssertNil(diarizer.getNextChunkFeatures(), "Only finalizing takes a partial right context")
        guard let chunk = diarizer.getNextChunkFeatures(isFinal: true) else {
            return XCTFail("Finalizing produced no chunk")
        }
        XCTAssertEqual(chunk.melLength, diarizer.melFramesEmitted)
        XCTAssertEqual(chunk.melLength, melFrameCount(samples))
        XCTAssertNil(diarizer.getNextChunkFeatures(isFinal: true))
    }

    // MARK: - End to end (real models)

    func testShortAudioGetsAPredictionForEveryFrame() async throws {
        let config = SortformerConfig.highContextV2_1
        let models = try await loadModels(config: config)
        let samples = try fixtureSamples()
        let expected = expectedFrames(samples, config: config)

        let complete = SortformerDiarizer(config: config)
        complete.initialize(models: models)
        let timeline = try complete.processComplete(samples)
        XCTAssertEqual(timeline.numFinalizedFrames, expected)
        XCTAssertEqual(timeline.numTentativeFrames, 0)
        XCTAssertFalse(timeline.speakers.values.flatMap(\.finalizedSegments).isEmpty, "No speech found")

        let streaming = SortformerDiarizer(config: config)
        streaming.initialize(models: models)
        for start in stride(from: 0, to: samples.count, by: 16_000) {
            _ = try streaming.process(samples: samples[start..<min(start + 16_000, samples.count)])
        }
        XCTAssertEqual(streaming.timeline.numFinalizedFrames, 0)
        try streaming.finalizeSession()
        XCTAssertEqual(streaming.timeline.numFinalizedFrames, expected)
        XCTAssertEqual(streaming.timeline.numTentativeFrames, 0)
        // Same chunks; the incremental mel matches batch to ~1e-5, which the fp16 model turns into
        // differences of a few 1e-3.
        let streamed = streaming.timeline.finalizedPredictions
        let batch = timeline.finalizedPredictions
        XCTAssertEqual(streamed.count, batch.count)
        let maxDiff = zip(streamed, batch).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 0.02, "Streamed and complete predictions differ")
    }

    func testProcessCompleteCoversTheTail() async throws {
        let config = SortformerConfig.default
        let models = try await loadModels(config: config)
        let samples = try fixtureSamples()

        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)
        let timeline = try diarizer.processComplete(samples)
        XCTAssertEqual(timeline.numFinalizedFrames, expectedFrames(samples, config: config))
        XCTAssertEqual(timeline.numTentativeFrames, 0)
    }
}
