// BISECT: temporarily compiled out to isolate a CI segfault (#901).
#if false
import XCTest

@testable import FluidAudio

/// Real-download check for #896: an English voice other than `af_heart` is
/// fetched from the repo-root `voices/<name>.json`, converted, and synthesizes.
/// Heavy (downloads the ANE bundle on a cold cache); gated like the TTS→ASR
/// roundtrip tests.
@available(macOS 14.0, iOS 17.0, *)
final class KokoroAneVoiceCatalogE2ETests: XCTestCase {

    func testNonDefaultEnglishVoiceSynthesizes() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1",
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run KokoroAne download tests.")

        let manager = KokoroAneManager(defaultVoice: "am_michael")
        try await manager.initialize()
        let detailed = try await manager.synthesizeDetailed(text: "Hello from FluidAudio.", voice: nil, speed: 1.0)
        XCTAssertGreaterThan(detailed.samples.count, detailed.sampleRate / 2, "expected at least 0.5 s of audio")
    }

    func testUnknownVoiceReportsCatalog() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1",
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run KokoroAne download tests.")

        let manager = KokoroAneManager(defaultVoice: "no_such_voice")
        do {
            try await manager.initialize()
            XCTFail("initialize() should fail for an unknown voice")
        } catch KokoroAneError.voiceNotFound(let voice, let variant, let available) {
            XCTAssertEqual(voice, "no_such_voice")
            XCTAssertEqual(variant, .english)
            XCTAssertTrue(available.contains("af_heart"))
        }
    }
}
#endif
