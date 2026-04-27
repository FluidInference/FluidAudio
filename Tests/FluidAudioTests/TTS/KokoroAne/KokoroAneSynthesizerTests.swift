import Foundation
import XCTest

@testable import FluidAudio

/// Heavy E2E tests gated by env var (require all 7 mlmodelc + voice + vocab
/// in cache). Skipped on CI by default.
final class KokoroAneSynthesizerTests: XCTestCase {

    private var shouldRunHeavy: Bool {
        ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1"
    }

    func testSynthesizeShortPhrase() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run end-to-end Kokoro-ANE synth tests."
        )

        let manager = KokoroAneManager()
        try await manager.initialize()
        let isReady = await manager.isAvailable()
        XCTAssertTrue(isReady, "Manager did not become available after initialize()")

        let result = try await manager.synthesizeDetailed(
            text: "Hello world", voice: nil, speed: 1.0)

        XCTAssertEqual(result.sampleRate, KokoroAneConstants.sampleRate)
        XCTAssertGreaterThan(result.samples.count, 0)
        // 24 kHz × ~0.5 s minimum for "Hello world" — generous lower bound.
        XCTAssertGreaterThan(result.samples.count, 24_000 / 2)
        XCTAssertGreaterThan(result.encoderTokens, 0)
        XCTAssertGreaterThan(result.acousticFrames, 0)
        XCTAssertLessThanOrEqual(
            result.acousticFrames, KokoroAneConstants.maxAcousticFrames)

        // Per-stage timings should all be > 0.
        XCTAssertGreaterThan(result.timings.totalMs, 0)
        XCTAssertGreaterThan(result.timings.albert, 0)
        XCTAssertGreaterThan(result.timings.tail, 0)

        // Audio should not be all-zeros.
        let peak = result.samples.lazy.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 0.001, "Synth produced silence")
    }

    func testSynthesizeProducesWavData() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run end-to-end Kokoro-ANE synth tests."
        )

        let manager = KokoroAneManager()
        try await manager.initialize()
        let wav = try await manager.synthesize(text: "Quick test")

        // RIFF header check.
        XCTAssertGreaterThan(wav.count, 44)
        let prefix = String(data: wav.prefix(4), encoding: .ascii)
        XCTAssertEqual(prefix, "RIFF")
        let waveTag = String(data: wav.subdata(in: 8..<12), encoding: .ascii)
        XCTAssertEqual(waveTag, "WAVE")
    }

    func testSynthesizeFromPhonemesBypassesG2P() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run end-to-end Kokoro-ANE synth tests."
        )

        let manager = KokoroAneManager()
        try await manager.initialize()
        // Direct IPA — skips G2P. Not all chars need to be in vocab; missing
        // ones are dropped silently.
        let wav = try await manager.synthesizeFromPhonemes("həloʊ wɹld")
        XCTAssertGreaterThan(wav.count, 44)
    }

    func testSynthesizeWithoutInitializeAttemptsLoadAndProceeds() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run end-to-end Kokoro-ANE synth tests."
        )

        // The manager calls store.loadIfNeeded() inside synthesize; an
        // uninitialized manager should still produce audio (slower first call).
        let manager = KokoroAneManager()
        let wav = try await manager.synthesize(text: "On demand load")
        XCTAssertGreaterThan(wav.count, 44)
    }
}
