import Foundation
import XCTest

@testable import FluidAudio

final class AsrModelsLocalTests: XCTestCase {
    func testOfficialV2VocabularyWithExtraTokensReachesModelLoading() throws {
        let directory = try makeV2VocabularyDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // The published vocabulary has 1,031 entries; v2's blank ID is 1,024.
        // With no models installed, a valid vocabulary must reach the encoder check.
        XCTAssertThrowsError(try AsrModels.loadLocal(from: directory, version: .v2)) { error in
            guard case AsrModelsError.modelNotFound(let name, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(name, "Encoder.mlmodelc")
            XCTAssertEqual(location, directory.appendingPathComponent(name))
        }
    }

    func testV2VocabularyWithMissingRequiredTokenFailsDespiteExtraTokens() throws {
        let directory = try makeV2VocabularyDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("parakeet_vocab.json")
        var vocabulary = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
        XCTAssertEqual(vocabulary.count, 1031)
        vocabulary.removeValue(forKey: "0")
        try JSONEncoder().encode(vocabulary).write(to: url)

        XCTAssertThrowsError(try AsrModels.loadLocal(from: directory, version: .v2)) { error in
            guard case AsrModelsError.loadingFailed(let reason) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(reason, "Local vocabulary must contain every token before the blank ID")
        }
    }

    private func makeV2VocabularyDirectory() throws -> URL {
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/parakeet-v2-vocabulary", withExtension: "json"))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            try FileManager.default.copyItem(at: fixture, to: directory.appendingPathComponent("parakeet_vocab.json"))
            return directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    func testMissingLocalVocabularyFailsAtTheRequestedDirectory() {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try AsrModels.loadLocal(from: directory)) { error in
            guard case AsrModelsError.modelNotFound(let name, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(name, "parakeet_vocab.json")
            XCTAssertEqual(location, directory.appendingPathComponent(name))
        }
    }

    /// Opt in with a real, compiled Orukeet bundle and a mono 16 kHz recording.
    func testLocalOrukeetRepeatedTranscription() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let bundle = environment["FLUIDAUDIO_LOCAL_TEST_MODELS"],
            let audio = environment["FLUIDAUDIO_LOCAL_TEST_AUDIO"]
        else { throw XCTSkip("Set FLUIDAUDIO_LOCAL_TEST_MODELS and FLUIDAUDIO_LOCAL_TEST_AUDIO") }
        let models = try AsrModels.loadLocal(from: URL(fileURLWithPath: bundle))
        XCTAssertEqual(models.vocabulary.count, 8192)
        XCTAssertEqual(models.version, .v3)
        let manager = AsrManager(config: .default, models: models)
        let samples = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: audio))
        var firstState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let first = try await manager.transcribe(samples, decoderState: &firstState)
        var secondState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let second = try await manager.transcribe(samples, decoderState: &secondState)
        XCTAssertFalse(first.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertEqual(first.text, second.text)
        if let expected = environment["FLUIDAUDIO_LOCAL_TEST_TRANSCRIPT"] {
            XCTAssertEqual(first.text.trimmingCharacters(in: .whitespacesAndNewlines), expected)
        }
    }
}
