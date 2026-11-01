import CoreML
import XCTest

@testable import FluidAudio

/// The CTC `MelSpectrogram` runs on the CPU (#1074). With the Neural Engine allowed, macOS 27.0.1 on M1 Pro and
/// M4 Max placed part of it there, and the encoder then emitted blank for every frame whatever the audio.
final class CtcModelsComputeUnitsTests: XCTestCase {

    func testMelSpectrogramRunsOnCpuAndEncoderKeepsTheNeuralEngine() {
        XCTAssertEqual(CtcModels.melSpectrogramConfiguration().computeUnits, .cpuOnly)
        XCTAssertEqual(CtcModels.defaultConfiguration().computeUnits, .cpuAndNeuralEngine)
    }

    /// Needs the CTC 110M models; runs when they are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download.
    func testLoadedModelsUseCpuMelAndDecodeRealSpeech() async throws {
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        let directory = CtcModels.defaultCacheDirectory(for: .ctc110m)
        try XCTSkipUnless(
            CtcModels.modelsExist(at: directory) || allowDownload,
            "CTC 110M models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        guard
            let url = Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav")
        else {
            throw XCTSkip("fixture missing from test bundle")
        }

        let loaded = try await CtcModels.downloadAndLoad()
        let direct = try await CtcModels.loadDirect(from: directory)
        for models in [loaded, direct] {
            XCTAssertEqual(models.melSpectrogram.configuration.computeUnits, .cpuOnly)
            XCTAssertEqual(models.encoder.configuration.computeUnits, .cpuAndNeuralEngine)
        }

        // On affected machines every frame's argmax was blank, so the CTC evidence ignored the audio.
        let samples = try AudioConverter().resampleAudioFile(url)
        let result = try await CtcKeywordSpotter(models: loaded).computeLogProbs(for: samples)
        XCTAssertFalse(result.logProbs.isEmpty)
        let blank = ContextBiasingConstants.defaultBlankId
        let speechFrames = result.logProbs.filter { frame in
            frame.indices.max { frame[$0] < frame[$1] } != blank
        }.count
        XCTAssertGreaterThan(speechFrames, 10, "CTC frames of a 21 s recording decode to (almost) all blank")
    }
}
