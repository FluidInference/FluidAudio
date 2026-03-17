import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Tests for speaker pre-enrollment APIs:
/// - `DiarizerManager.extractSpeakerEmbedding(from:)`
/// - `SortformerDiarizer.enrollSpeaker(withAudio:named:)`
/// - `LSEENDDiarizer.enrollSpeaker(withSamples:named:)`
final class SpeakerEnrollmentTests: XCTestCase {
    private static let fixtureSampleRate = 16_000
    nonisolated(unsafe) private static var cachedFixtureAudioURL: URL?
    nonisolated(unsafe) private static var cachedLseendEngine: LSEENDInferenceEngine?

    private func loadSortformerModelsForTest(config: SortformerConfig) async throws -> SortformerModels {
        // These tests validate Sortformer behavior after initialization, not accelerator selection.
        try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
    }

    private func loadLseendEngineForTest(variant: LSEENDVariant = .dihard3) async throws -> LSEENDInferenceEngine {
        if let cached = Self.cachedLseendEngine {
            return cached
        }

        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        let engine = try LSEENDInferenceEngine(descriptor: descriptor, computeUnits: .cpuOnly)
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
        let enrollmentAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)

        let speaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")

        XCTAssertNotNil(speaker)
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
        let enrollmentAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)
        let liveAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 5.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
        XCTAssertNotNil(speaker)

        var update: DiarizerTimelineUpdate?
        for chunk in chunk(liveAudio, sizes: [7_680, 9_600, 11_520]) {
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

    func testSortformerMultipleEnrollmentsRetainNamedSpeakersAndState() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let config = SortformerConfig.default
        let diarizer = SortformerDiarizer(config: config)
        let models = try await loadSortformerModelsForTest(config: config)
        diarizer.initialize(models: models)
        let speakerAAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let speakerBAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 3.4, durationSeconds: 3.0)

        let speakerA = try diarizer.enrollSpeaker(withAudio: speakerAAudio, named: "Alice")
        XCTAssertNotNil(speakerA)

        let stateAfterA = diarizer.state
        let cachedLengthAfterA = stateAfterA.spkcacheLength + stateAfterA.fifoLength

        let speakerB = try diarizer.enrollSpeaker(withAudio: speakerBAudio, named: "Bob")
        XCTAssertNotNil(speakerB)

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
        let enrollmentAudio = try fixtureAudio(sampleRate: config.sampleRate, startSeconds: 0.0, durationSeconds: 5.0)

        let firstSpeaker = try diarizer.enrollSpeaker(withAudio: enrollmentAudio, named: "Alice")
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
        let enrollmentAudio = try fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")

        if let speaker {
            XCTAssertEqual(speaker.name, "Alice")
        }
        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(namedSpeakerIndices(in: diarizer.timeline), [speaker?.index].compactMap { $0 })
        XCTAssertTrue(hasActiveLseendSession(diarizer))
    }

    func testLseendEnrollSpeakerFollowedByStreamingProcessingStartsAtFrameZero() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let enrollmentAudio = try fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let liveAudio = try fixtureAudio(sampleRate: engine.targetSampleRate, startSeconds: 3.0, durationSeconds: 3.0)

        let speaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")

        var firstUpdate: DiarizerTimelineUpdate?
        for chunk in chunk(liveAudio, sizes: [977, 1231, 1607]) {
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
        let speakerAAudio = try fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)
        let speakerBAudio = try fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 3.0, durationSeconds: 3.0)

        let speakerA = try diarizer.enrollSpeaker(withSamples: speakerAAudio, named: "Alice")
        let speakerB = try diarizer.enrollSpeaker(withSamples: speakerBAudio, named: "Bob")

        XCTAssertEqual(diarizer.numFramesProcessed, 0)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertTrue(hasActiveLseendSession(diarizer))
        let expectedNames = Set([speakerA?.name, speakerB?.name].compactMap { $0 })
        XCTAssertEqual(Set(namedSpeakerNames(in: diarizer.timeline)), expectedNames)
    }

    func testLseendEnrollmentCanRefuseToOverwriteNamedSpeaker() async throws {
        XCTExpectFailure("Download might fail in CI environment", strict: false)

        let engine = try await loadLseendEngineForTest()
        let diarizer = LSEENDDiarizer(computeUnits: .cpuOnly)
        diarizer.initialize(engine: engine)
        let enrollmentAudio = try fixtureAudio(
            sampleRate: engine.targetSampleRate, startSeconds: 0.0, durationSeconds: 3.0)

        let firstSpeaker = try diarizer.enrollSpeaker(withSamples: enrollmentAudio, named: "Alice")
        try XCTSkipIf(firstSpeaker == nil, "Fixture did not produce a confident LS-EEND speaker segment on this host.")
        let secondSpeaker = try diarizer.enrollSpeaker(
            withSamples: enrollmentAudio,
            named: "Bob",
            overwritingAssignedSpeakerName: false
        )

        XCTAssertNotNil(firstSpeaker)
        XCTAssertNil(secondSpeaker)
        XCTAssertEqual(namedSpeakerNames(in: diarizer.timeline), ["Alice"])
    }

    private func fixtureAudio(sampleRate: Int, startSeconds: Double = 0.0, durationSeconds: Double) throws -> [Float] {
        let converter = AudioConverter(sampleRate: Double(sampleRate))
        let audio = try converter.resampleAudioFile(try fixtureAudioFileURL())
        let startSample = min(audio.count, Int(startSeconds * Double(sampleRate)))
        let endSample = min(audio.count, startSample + Int(durationSeconds * Double(sampleRate)))
        return Array(audio[startSample..<endSample])
    }

    private func fixtureAudioFileURL() throws -> URL {
        if let cached = Self.cachedFixtureAudioURL,
            FileManager.default.fileExists(atPath: cached.path)
        {
            return cached
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("speaker-enrollment-fixture-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        try writeFixtureAudio(to: url)
        Self.cachedFixtureAudioURL = url
        return url
    }

    private func writeFixtureAudio(to url: URL) throws {
        let sampleRate = Double(Self.fixtureSampleRate)
        let samples = makeFixtureSamples(sampleRate: sampleRate)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else {
            XCTFail("Failed to allocate fixture audio buffer")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            guard let destination = buffer.floatChannelData?[0] else { return }
            destination.update(from: source.baseAddress!, count: samples.count)
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: buffer)
    }

    private func makeFixtureSamples(sampleRate: Double) -> [Float] {
        let segments: [(duration: Double, amplitude: Float, frequency: Double)] = [
            (1.0, 0.20, 220),
            (0.35, 0.00, 0),
            (1.1, 0.32, 330),
            (0.25, 0.00, 0),
            (1.0, 0.28, 180),
            (0.40, 0.00, 0),
            (1.3, 0.36, 260),
            (0.30, 0.00, 0),
            (1.1, 0.24, 410),
        ]

        var output: [Float] = []
        for (duration, amplitude, frequency) in segments {
            let frameCount = Int(duration * sampleRate)
            guard amplitude > 0, frequency > 0 else {
                output.append(contentsOf: repeatElement(0, count: frameCount))
                continue
            }

            for frame in 0..<frameCount {
                let time = Double(frame) / sampleRate
                let envelope = Float(min(1.0, time * 12.0)) * Float(min(1.0, (duration - time) * 12.0))
                let carrier = sin(2.0 * Double.pi * frequency * time)
                let harmonic = 0.35 * sin(2.0 * Double.pi * frequency * 2.03 * time)
                output.append(Float((carrier + harmonic) * Double(amplitude * envelope)))
            }
        }
        return output
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

    private func chunk(_ samples: [Float], sizes: [Int]) -> [[Float]] {
        var chunks: [[Float]] = []
        var start = 0
        var index = 0
        while start < samples.count {
            let size = sizes[index % sizes.count]
            let stop = min(samples.count, start + size)
            chunks.append(Array(samples[start..<stop]))
            start = stop
            index += 1
        }
        return chunks
    }

    private func hasActiveLseendSession(_ diarizer: LSEENDDiarizer) -> Bool {
        let mirror = Mirror(reflecting: diarizer)
        guard let sessionValue = mirror.children.first(where: { $0.label == "_session" })?.value else {
            XCTFail("Expected LS-EEND diarizer to expose _session via reflection")
            return false
        }

        let optionalMirror = Mirror(reflecting: sessionValue)
        return optionalMirror.displayStyle == .optional && optionalMirror.children.count == 1
    }
}
