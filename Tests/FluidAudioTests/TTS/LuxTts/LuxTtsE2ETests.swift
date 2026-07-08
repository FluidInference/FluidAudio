import XCTest

@testable import FluidAudio

/// Model-dependent end-to-end synthesis test. Downloads/loads the LuxTTS
/// CoreML stages, so it only runs when explicitly requested:
/// `FLUIDAUDIO_RUN_LUXTTS_E2E=1 swift test --filter LuxTtsE2ETests`.
///
/// Gates mirror the Python fixture run (`dump_swift_fixtures.py`):
/// identical frame accounting (same tokenizer + duration math ⇒ exact),
/// 48 kHz output, and RMS within 1 dB of the Python CoreML pipeline
/// (waveforms differ — the noise RNG is not torch's).
final class LuxTtsE2ETests: XCTestCase {

    private var shouldRunHeavy: Bool {
        ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_LUXTTS_E2E"] == "1"
    }

    func testSynthesizeMatchesPythonFixtureStats() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_LUXTTS_E2E=1 to run end-to-end LuxTTS synth tests.")

        let fixtures = try LuxTtsFixtures.load()
        let text = fixtures.texts[fixtures.e2e.textIndex]

        // Reconstruct the prompt clip from the 24 kHz fixture waveform so the
        // test does not depend on files outside the repo. 16-bit quantization
        // perturbs the mel at ~1e-4 — irrelevant for the stat gates below.
        let promptSamples = try LuxTtsFixtures.loadFloats("prompt_24k_f32le.bin")
        let promptURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("luxtts_e2e_prompt_\(UUID().uuidString).wav")
        let promptWav = try AudioWAV.data(
            from: promptSamples,
            sampleRate: Double(LuxTtsConstants.melSampleRate),
            normalize: false)
        try promptWav.write(to: promptURL)
        defer { try? FileManager.default.removeItem(at: promptURL) }

        let manager = try await LuxTtsManager.downloadAndCreate()
        let isReady = await manager.isAvailable
        XCTAssertTrue(isReady, "Manager did not become available after initialize()")

        let result = try await manager.synthesize(
            phonemes: text.phonemeString,
            promptAudio: promptURL,
            promptPhonemes: fixtures.prompt.phonemeString,
            speed: Float(fixtures.e2e.speed),
            seed: UInt64(fixtures.e2e.seed))

        // Frame accounting must be identical to Python (pure integer math
        // over fixture-pinned token ids and mel frame counts).
        XCTAssertEqual(result.sampleRate, 48000)
        XCTAssertEqual(result.promptFrames, fixtures.prompt.melFrames)
        XCTAssertEqual(result.featuresLength, fixtures.e2e.featuresLen)
        XCTAssertEqual(result.generatedFrames, fixtures.e2e.genFrames)
        XCTAssertEqual(result.samples.count, fixtures.e2e.wavSamples)

        // Loudness within 1 dB of the Python CoreML pipeline, and non-silent.
        let sumSquares = result.samples.reduce(Double(0)) { $0 + Double($1) * Double($1) }
        let rms = (sumSquares / Double(result.samples.count)).squareRoot()
        XCTAssertGreaterThan(rms, 0.01, "output is (near-)silent")
        let dbDelta = 20.0 * log10(rms / fixtures.e2e.rms)
        XCTAssertLessThan(abs(dbDelta), 1.0, "RMS off by \(dbDelta) dB vs Python fixture")

        await manager.cleanup()
    }

    func testRawTextSynthesisRequiresInitialize() async throws {
        // No models needed: the raw-text overload phonemizes in-process
        // (bundled G2P, no download) and must still fail loudly with
        // `.notInitialized` before touching the pipeline.
        let manager = LuxTtsManager()
        do {
            _ = try await manager.synthesize(
                text: "hello",
                promptAudio: URL(fileURLWithPath: "/nonexistent.wav"),
                promptText: "hello")
            XCTFail("expected notInitialized")
        } catch let error as LuxTtsError {
            guard case .notInitialized = error else {
                XCTFail("expected notInitialized, got \(error)")
                return
            }
        }
    }
}
