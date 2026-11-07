import CoreML
import XCTest

@testable import FluidAudio

/// The encoder input lengths `SenseVoiceManager` pads to come from the loaded model, so the
/// fp32 encoder (one fixed 1800-frame input) gets 1800 frames instead of a fp16 bucket, which
/// Core ML rejected: "MultiArray shape (1 x 256 x 560) does not match the shape (1 x 1800 x 560)".
final class SenseVoiceEncoderFrameCountsTests: XCTestCase {

    private func cachedModels(_ precision: SenseVoiceEncoderPrecision) throws -> SenseVoiceModels {
        let directory = MLModelConfigurationUtils.defaultModelsDirectory(for: .senseVoiceSmall)
        guard SenseVoiceModels.modelsExist(at: directory, precision: precision) else {
            throw XCTSkip("SenseVoice \(precision.rawValue) models not cached at \(directory.path)")
        }
        return try SenseVoiceModels.load(from: directory, precision: precision)
    }

    func testFp16EncoderUsesItsEnumeratedBuckets() throws {
        let models = try cachedModels(.fp16)
        XCTAssertEqual(SenseVoiceManager.acceptedFrameCounts(of: models.encoder), SenseVoiceConfig.buckets)
    }

    func testFp32EncoderUsesItsFixedLength() throws {
        let models = try cachedModels(.fp32)
        XCTAssertEqual(SenseVoiceManager.acceptedFrameCounts(of: models.encoder), [1800])
    }

    func testFp32TranscribesRealSpeech() async throws {
        let name = "03-diff-explanation-16.9s.wav"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        let manager = SenseVoiceManager(models: try cachedModels(.fp32))
        let samples = try AudioConverter().resampleAudioFile(url)
        let text = try await manager.transcribe(audio: samples)
        XCTAssertFalse(text.isEmpty)
    }
}
