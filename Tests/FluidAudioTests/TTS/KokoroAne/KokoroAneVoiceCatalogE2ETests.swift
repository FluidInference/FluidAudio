import Foundation
import XCTest

@testable import FluidAudio

/// Real-download check for #896: an English voice other than `af_heart` is
/// fetched from the repo-root `voices/<name>.json`, converted, and synthesizes.
/// Heavy (downloads the ANE bundle on a cold cache); gated like the TTS→ASR
/// roundtrip tests.
///
/// Shape mirrors `KokoroAneAsrRoundtripTests` on purpose: a first revision with
/// an `@available` class attribute and an enum-pattern `catch` inside the async
/// test body segfaulted the CI test worker (Swift 6.1) even though both tests
/// skip on their first line.
final class KokoroAneVoiceCatalogE2ETests: XCTestCase {

    private var shouldRunHeavy: Bool {
        ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1"
    }

    func testNonDefaultEnglishVoiceSynthesizes() async throws {
        try XCTSkipUnless(shouldRunHeavy, "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run KokoroAne download tests.")

        let manager = KokoroAneManager(defaultVoice: "am_michael")
        try await manager.initialize()
        let detailed = try await manager.synthesizeDetailed(text: "Hello from FluidAudio.", voice: nil, speed: 1.0)
        XCTAssertGreaterThan(detailed.samples.count, detailed.sampleRate / 2, "expected at least 0.5 s of audio")
    }

    func testUnknownVoiceReportsCatalog() async throws {
        try XCTSkipUnless(shouldRunHeavy, "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run KokoroAne download tests.")

        let manager = KokoroAneManager(defaultVoice: "no_such_voice")
        var caught: Error?
        do {
            try await manager.initialize()
        } catch {
            caught = error
        }
        XCTAssertNotNil(caught, "initialize() should fail for an unknown voice")
        let unwrapped = caught as? KokoroAneError
        XCTAssertNotNil(unwrapped, "expected KokoroAneError, got \(String(describing: caught))")
        if case .voiceNotFound(let voice, let variant, let available)? = unwrapped {
            XCTAssertEqual(voice, "no_such_voice")
            XCTAssertEqual(variant, .english)
            XCTAssertTrue(available.contains("af_heart"))
        } else {
            XCTFail("expected voiceNotFound, got \(String(describing: unwrapped))")
        }
    }
}
