import XCTest

@testable import FluidAudio

/// Tests for speaker pre-enrollment APIs:
/// - `DiarizerManager.extractSpeakerEmbedding(from:)`
/// - `SortformerDiarizer.primeWithAudio(_:)`
final class SpeakerEnrollmentTests: XCTestCase {

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

    // MARK: - primeWithAudio: Error Cases

    func testPrimeWithAudioThrowsWhenNotInitialized() {
        let diarizer = SortformerDiarizer()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.primeWithAudio(audio)) { error in
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

    func testPrimeWithAudioThrowsAfterCleanup() {
        let diarizer = SortformerDiarizer()
        diarizer.cleanup()
        let audio = [Float](repeating: 0.1, count: 16000)

        XCTAssertThrowsError(try diarizer.primeWithAudio(audio)) { error in
            guard case SortformerError.notInitialized = error else {
                XCTFail("Expected .notInitialized but got \(error)")
                return
            }
        }
    }

    // MARK: - primeWithAudio: State Verification (requires model download)

    func testPrimeResetsTimelineButKeepsState() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)

        let models = try await SortformerModels.loadFromHuggingFace(config: config)
        diarizer.initialize(models: models)

        // Prime with 5 seconds of audio
        let enrollmentAudio = (0..<80000).map { i in sin(Float(i) * 0.05) * 0.3 }
        try diarizer.primeWithAudio(enrollmentAudio)

        // Timeline should be reset (frame count = 0)
        XCTAssertEqual(diarizer.numFramesProcessed, 0, "Frame counter should be 0 after priming")
        XCTAssertEqual(diarizer.timeline.numFrames, 0, "Timeline should have 0 frames after priming")

        // Streaming state should be preserved (spkcache or fifo may be populated)
        let state = diarizer.state
        let hasState = state.spkcacheLength > 0 || state.fifoLength > 0
        XCTAssertTrue(hasState, "Streaming state (spkcache/fifo) should be populated after priming")
    }

    func testPrimeFollowedByStreamingProcessing() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)

        let models = try await SortformerModels.loadFromHuggingFace(config: config)
        diarizer.initialize(models: models)

        // Prime with enrollment audio
        let enrollmentAudio = (0..<80000).map { i in sin(Float(i) * 0.05) * 0.3 }
        try diarizer.primeWithAudio(enrollmentAudio)

        // Stream new audio after priming — should not crash
        let liveAudio = (0..<48000).map { i in sin(Float(i) * 0.08) * 0.2 }
        diarizer.addAudio(liveAudio)

        // Process should work without errors
        let result = try diarizer.process()
        // Result may or may not contain data depending on buffer thresholds — that's fine
        _ = result
    }

    func testMultiplePrimeCalls() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)

        let models = try await SortformerModels.loadFromHuggingFace(config: config)
        diarizer.initialize(models: models)

        // Prime with speaker A
        let speakerA = (0..<80000).map { i in sin(Float(i) * 0.05) * 0.3 }
        try diarizer.primeWithAudio(speakerA)

        let stateAfterA = diarizer.state
        let spkcacheAfterA = stateAfterA.spkcacheLength

        // Prime with speaker B
        let speakerB = (0..<80000).map { i in cos(Float(i) * 0.07) * 0.4 }
        try diarizer.primeWithAudio(speakerB)

        // State should accumulate (more data in cache)
        let stateAfterB = diarizer.state
        XCTAssertGreaterThanOrEqual(
            stateAfterB.spkcacheLength + stateAfterB.fifoLength,
            spkcacheAfterA,
            "State should accumulate across prime calls"
        )

        // Timeline should still be reset
        XCTAssertEqual(diarizer.numFramesProcessed, 0, "Frame counter should be 0 after priming")
    }
}
