import Foundation
import XCTest

@testable import FluidAudio

final class AsrModelsLocalTests: XCTestCase {
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
