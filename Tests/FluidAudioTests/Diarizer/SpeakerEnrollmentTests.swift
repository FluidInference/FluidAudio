import Foundation
import XCTest

@testable import FluidAudio

/// Tests for speaker pre-enrollment APIs:
/// - `DiarizerManager.extractSpeakerEmbedding(from:)`
/// - `SortformerDiarizer.enrollSpeaker(withAudio:named:)`
/// - `LSEENDDiarizer.enrollSpeaker(withSamples:named:)`
final class SpeakerEnrollmentTests: XCTestCase {
    nonisolated(unsafe) private static var cachedLseendEngine: LSEENDInferenceHelper?

    private func loadSortformerModelsForTest(config: SortformerConfig) async throws -> SortformerModels {
        // These tests validate Sortformer behavior after initialization, not accelerator selection.
        try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
    }

    private func loadLseendEngineForTest(variant: LSEENDVariant = .dihard3) async throws -> LSEENDInferenceHelper {
        if let cached = Self.cachedLseendEngine {
            return cached
        }

        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        let engine = try LSEENDInferenceHelper(descriptor: descriptor, computeUnits: .cpuOnly)
        Self.cachedLseendEngine = engine
        return engine
    }

    // MARK: - extractSpeakerEmbedding: Error Cases

    func testExtractEmbeddingThrowsWhenNotInitialized() {
        let manager = DiarizerManager()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try manager.extractSpeakerEmbedding(from: audio)) { error in
            XCTAssertTrue(
                error is DiarizerError,
                "Expected DiarizerError but got \(type(of: error))"
            )
            guard case DiarizerError.notInitialized = error else {
                XCTFail("Expected .notInitialized but got \(error)")
                return
            }
        }
    }

    func testExtractEmbeddingThrowsWhenCleanedUp() {
        let manager = DiarizerManager()
        manager.cleanup()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try manager.extractSpeakerEmbedding(from: audio)) { error in
            guard case DiarizerError.notInitialized = error else {
                XCTFail("Expected .notInitialized but got \(error)")
                return
            }
        }
    }

    // MARK: - extractSpeakerEmbedding: Integration (requires model download)

    func testExtractEmbeddingProducesValidResult() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let manager = DiarizerManager()
        let models = try await DiarizerModels.downloadIfNeeded()
        manager.initialize(models: models)

        // 3 seconds of sine wave audio (simulates single speaker)
        let audio = (0..<48000).map { i in sin(Float(i) * 0.1) * 0.3 }

        let embedding = try manager.extractSpeakerEmbedding(from: audio)

        // Should be a 256-dimensional embedding
        XCTAssertEqual(embedding.count, 256, "Embedding should be 256-dimensional")

        // Should not be all zeros (valid speaker audio)
        let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        XCTAssertGreaterThan(magnitude, 0.01, "Embedding should have non-trivial magnitude")

        // Should not contain NaN or Inf
        XCTAssertFalse(embedding.contains(where: { $0.isNaN }), "Embedding should not contain NaN")
        XCTAssertFalse(embedding.contains(where: { $0.isInfinite }), "Embedding should not contain Inf")
    }

    func testExtractEmbeddingSameAudioProducesSimilarEmbeddings() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let manager = DiarizerManager()
        let models = try await DiarizerModels.downloadIfNeeded()
        manager.initialize(models: models)

        // Same audio extracted twice should produce identical embeddings
        let audio = (0..<48000).map { i in sin(Float(i) * 0.1) * 0.3 }

        let embedding1 = try manager.extractSpeakerEmbedding(from: audio)
        let embedding2 = try manager.extractSpeakerEmbedding(from: audio)

        XCTAssertEqual(embedding1.count, embedding2.count)
        for i in 0..<embedding1.count {
            XCTAssertEqual(
                embedding1[i], embedding2[i], accuracy: 1e-5, "Embeddings should be identical for same input")
        }
    }

    func testExtractEmbeddingUsableWithKnownSpeakers() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let manager = DiarizerManager()
        let models = try await DiarizerModels.downloadIfNeeded()
        manager.initialize(models: models)

        let audio = (0..<48000).map { i in sin(Float(i) * 0.1) * 0.3 }
        let embedding = try manager.extractSpeakerEmbedding(from: audio)

        // Verify the embedding can be used with initializeKnownSpeakers
        let speaker = Speaker(id: "test", name: "Test", currentEmbedding: embedding, isPermanent: true)
        manager.initializeKnownSpeakers([speaker])

        XCTAssertEqual(manager.speakerManager.speakerCount, 1, "Known speaker should be registered")
    }

    // MARK: - Sortformer enrollSpeaker: Error Cases

    func testSortformerEnrollSpeakerThrowsWhenNotInitialized() {
        let diarizer = SortformerDiarizer()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.enrollSpeaker(withAudio: audio)) { error in
            XCTAssertTrue(
                error is SortformerError,
                "Expected SortformerError but got \(type(of: error))"
            )
            guard case SortformerError.notInitialized = error else {
                XCTFail("Expected .notInitialized but got \(error)")
                return
            }
        }
    }

    func testSortformerEnrollSpeakerThrowsAfterCleanup() {
        let diarizer = SortformerDiarizer()
        diarizer.cleanup()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.enrollSpeaker(withAudio: audio)) { error in
            guard case SortformerError.notInitialized = error else {
                XCTFail("Expected .notInitialized but got \(error)")
                return
            }
        }
    }

    // MARK: - Sortformer enrollSpeaker: Integration (requires model download)

    func testSortformerEnrollSpeakerReturnsNamedSpeakerAndResetsTimeline() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)
        let models = try await loadSortformerModelsForTest(config: config)
        diarizer.initialize(models: models)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)

        let speaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")

        try XCTSkipIf(
            speaker == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")
        XCTAssertEqual(speaker?.name, "Alice")
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(namedSpeakerIndices(in: diarizer.timeline), [speaker?.index].compactMap { $0 })

        let state = diarizer.state
        XCTAssertTrue(state.spkcacheLength > 0 || state.fifoLength > 0)
    }

    func testSortformerEnrollSpeakerFollowedByStreamingProcessing() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)
        let models = try await loadSortformerModelsForTest(config: config)
        diarizer.initialize(models: models)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)
        let liveAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 5.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
        try XCTSkipIf(
            speaker == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")

        var update: DiarizerTimelineUpdate?
        for chunk in DiarizationTestFixtures.chunk(liveAudio, sizes: [7_680, 9_600, 11_520]) {
            diarizer.addAudio(chunk)
            if let next = try diarizer.process() {
                update = next
                break
            }
        }

        XCTAssertNotNil(update)
        if let update {
            XCTAssertEqual(update.chunkResult.startFrame, 0)
            XCTAssertTrue(
                update.chunkResult.finalizedFrameCount > 0
                    || update.chunkResult.tentativeFrameCount > 0
            )
        }
        XCTAssertEqual(namedSpeakerIndices(in: diarizer.timeline), [speaker?.index].compactMap { $0 })
    }

    func testSortformerEnrollmentClearsDiscardedSampleCountBeforeFinalize() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let models = try await loadSortformerModelsForTest(config: config)

        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)
        let discardedAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 5.0, durationSeconds: 1.5)
        let liveAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 6.5, durationSeconds: 3.0)

        let dirtyDiarizer = SortformerDiarizer(config: config)
        dirtyDiarizer.initialize(models: models)
        dirtyDiarizer.addAudio(discardedAudio)
        let enrolledSpeaker = try dirtyDiarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
        try XCTSkipIf(
            enrolledSpeaker == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")

        for chunk in DiarizationTestFixtures.chunk(liveAudio, sizes: [4_800, 7_680, 9_600]) {
            let _ = try dirtyDiarizer.process(samples: chunk)
        }
        let _ = try dirtyDiarizer.finalizeSession()

        let cleanDiarizer = SortformerDiarizer(config: config)
        cleanDiarizer.initialize(models: models)
        let cleanSpeaker = try cleanDiarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
        try XCTSkipIf(
            cleanSpeaker == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")

        for chunk in DiarizationTestFixtures.chunk(liveAudio, sizes: [4_800, 7_680, 9_600]) {
            let _ = try cleanDiarizer.process(samples: chunk)
        }
        let _ = try cleanDiarizer.finalizeSession()

        XCTAssertLessThanOrEqual(
            abs(dirtyDiarizer.timeline.numFinalizedFrames - cleanDiarizer.timeline.numFinalizedFrames),
            1
        )
        XCTAssertEqual(dirtyDiarizer.timeline.numTentativeFrames, 0)
        XCTAssertEqual(cleanDiarizer.timeline.numTentativeFrames, 0)
    }

    func testSortformerMultipleEnrollmentsRetainNamedSpeakersAndState() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)
        let models = try await loadSortformerModelsForTest(config: config)
        diarizer.initialize(models: models)
        let speakerAAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let speakerBAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 3.4, durationSeconds: 3.0)

        let speakerA = try diarizer.enrollSpeaker(withAudio: speakerAAudio, named: "Alice")
        try XCTSkipIf(
            speakerA == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")

        let stateAfterA = diarizer.state
        let cachedLengthAfterA = stateAfterA.spkcacheLength + stateAfterA.fifoLength

        let speakerB = try diarizer.enrollSpeaker(withAudio: speakerBAudio, named: "Bob")
        try XCTSkipIf(
            speakerB == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")

        let stateAfterB = diarizer.state
        XCTAssertGreaterThanOrEqual(
            stateAfterB.spkcacheLength + stateAfterB.fifoLength,
            cachedLengthAfterA
        )
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        if speakerA?.index == speakerB?.index {
            XCTAssertEqual(namedSpeakerNames(in: diarizer.timeline), ["Bob"])
        } else {
            XCTAssertEqual(Set(namedSpeakerNames(in: diarizer.timeline)), Set(["Alice", "Bob"]))
        }
    }

    func testSortformerEnrollmentCanRefuseToOverwriteNamedSpeaker() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)
        let models = try await loadSortformerModelsForTest(config: config)
        diarizer.initialize(models: models)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)

        let firstSpeaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
        try XCTSkipIf(
            firstSpeaker == nil, "Fixture did not produce a confident Sortformer speaker segment on this host.")
        let secondSpeaker = try diarizer.enrollSpeaker(
            withAudio: enrollmentAudio,
            named: "Bob",
            overwritingAssignedSpeakerName: false
        )

        XCTAssertNotNil(firstSpeaker)
        XCTAssertNil(secondSpeaker)
        XCTAssertEqual(namedSpeakerNames(in: diarizer.timeline), ["Alice"])
    }

    // MARK: - LS-EEND enrollSpeaker: Error Cases

    func testLseendEnrollSpeakerThrowsWhenNotInitialized() {
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.enrollSpeaker(withSamples: audio)) { error in
            guard case LSEENDError.modelPredictionFailed(let message) = error else {
                XCTFail("Expected modelPredictionFailed but got \(error)")
                return
            }
            XCTAssertTrue(message.contains("not initialized"))
        }
    }

    // MARK: - LS-EEND enrollSpeaker: Integration (requires model download)

    func testLseendEnrollSpeakerResetsTimelineAndWarmsSession() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")

        if let speaker {
            XCTAssertEqual(speaker.name, "Alice")
        }
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(namedSpeakerIndices(in: diarizer.timeline), [speaker?.index].compactMap { $0 })
        XCTAssertTrue(diarizer.hasActiveSession)
    }

    func testLseendEnrollSpeakerFollowedByStreamingProcessingStartsAtFrameZero() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let liveAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 3.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")

        var firstUpdate: DiarizerTimelineUpdate?
        for chunk in DiarizationTestFixtures.chunk(liveAudio, sizes: [977, 1231, 1607]) {
            if let update = try diarizer.process(samples: chunk) {
                firstUpdate = update
                break
            }
        }
        let finalChunk = try diarizer.finalizeSession()

        XCTAssertTrue(firstUpdate != nil || finalChunk != nil)
        if let firstUpdate {
            XCTAssertEqual(firstUpdate.chunkResult.startFrame, 0)
        }
        XCTAssertGreaterThan(diarizer.timeline.numFinalizedFrames, 0)
        if let speaker {
            XCTAssertEqual(namedSpeakerIndices(in: diarizer.timeline), [speaker.index])
        }
    }

    func testLseendMultipleEnrollmentsRetainNamedSpeakersAndSession() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let speakerAAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let speakerBAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 3.0, durationSeconds: 3.0)

        let speakerA = try diarizer.enrollSpeaker(withSamples: speakerAAudio, named: "Alice")
        let speakerB = try diarizer.enrollSpeaker(withSamples: speakerBAudio, named: "Bob")

        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertTrue(diarizer.hasActiveSession)
        let expectedNames = Set([speakerA?.name, speakerB?.name].compactMap { $0 })
        XCTAssertEqual(Set(namedSpeakerNames(in: diarizer.timeline)), expectedNames)
    }

    func testLseendEnrollmentCanRefuseToOverwriteNamedSpeaker() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let enrollmentAudio = try DiarizationTestFixtures.fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)

        let firstSpeaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")
        try XCTSkipIf(
            firstSpeaker == nil, "Fixture did not produce a confident LS-EEND speaker segment on this host.")
        let secondSpeaker = try diarizer.enrollSpeaker(
            withSamples: enrollmentAudio,
            named: "Bob",
            overwritingAssignedSpeakerName: false
        )

        XCTAssertNotNil(firstSpeaker)
        XCTAssertNil(secondSpeaker)
        XCTAssertEqual(namedSpeakerNames(in: diarizer.timeline), ["Alice"])
    }

    private func namedSpeakerIndices(in timeline: DiarizerTimeline) -> [Int] {
        timeline.speakers.values
            .filter { $0.name != nil }
            .map(\.index)
            .sorted()
    }

    private func namedSpeakerNames(in timeline: DiarizerTimeline) -> [String] {
        timeline.speakers.values
            .compactMap(\.name)
            .sorted()
    }
}
