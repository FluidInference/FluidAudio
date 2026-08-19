import Foundation
import XCTest

@testable import FluidAudio

/// Lightweight tests for the pure duration-rounding helper (no models needed).
final class KokoroAnePredictedDurationTests: XCTestCase {

    func testRoundsAndClampsToMinimumOne() throws {
        let out = try KokoroAneSynthesizer.predictedDurations(
            from: [0.2, 0.5, 1.4, 2.5, 7.9, -3.0])
        // .rounded() is half-away-from-zero: 2.5 → 3.
        XCTAssertEqual(out, [1, 1, 1, 3, 8, 1])
    }

    func testNaNThrowsInsteadOfTrapping() {
        // iOS 27 betas mis-execute the dynamic-shape PostAlbert stage and
        // return NaN durations (#738); Int32(NaN) would crash the host app.
        XCTAssertThrowsError(
            try KokoroAneSynthesizer.predictedDurations(from: [1.0, .nan, 2.0])
        ) { error in
            guard case KokoroAneError.nonFiniteModelOutput(let stage, let output) = error else {
                return XCTFail("Expected nonFiniteModelOutput, got \(error)")
            }
            XCTAssertEqual(stage, KokoroAneStage.postAlbert.rawValue)
            XCTAssertEqual(output, "duration")
        }
    }

    func testInfinityThrows() {
        XCTAssertThrowsError(
            try KokoroAneSynthesizer.predictedDurations(from: [.infinity]))
        XCTAssertThrowsError(
            try KokoroAneSynthesizer.predictedDurations(from: [-.infinity]))
    }

    func testHugeFiniteValueClampsWithoutTrapping() throws {
        // Garbage runtimes can also return huge finite values; Int32(1e30)
        // would trap. Clamped to the frame cap, the downstream T_a check
        // then throws acousticFramesExceedCap.
        let out = try KokoroAneSynthesizer.predictedDurations(from: [1e30, 3.0])
        XCTAssertEqual(out[0], Int32(KokoroAneConstants.maxAcousticFrames))
        XCTAssertEqual(out[1], 3)
    }

    func testEmptyInput() throws {
        XCTAssertEqual(try KokoroAneSynthesizer.predictedDurations(from: []), [])
    }

    func testSynthesisResultTimingMetadataDefaultsForSourceCompatibility() {
        let result = KokoroAneSynthesisResult(
            samples: [0],
            sampleRate: KokoroAneConstants.sampleRate,
            encoderTokens: 1,
            acousticFrames: 1,
            timings: KokoroAneStageTimings()
        )

        XCTAssertTrue(result.inputIds.isEmpty)
        XCTAssertTrue(result.predictedDurations.isEmpty)
    }
}

/// Heavy E2E tests gated by env var (require all 7 mlmodelc + voice + vocab
/// in cache). Skipped on CI by default.
final class KokoroAneSynthesizerTests: XCTestCase {

    private var shouldRunHeavy: Bool {
        ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1"
    }

    func testPublishedBundlesContainFlexibleShapeInformation() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to validate the real Kokoro-ANE models."
        )

        let repoDirectory = try await KokoroAneResourceDownloader.ensureModels()
        let incompatible = KokoroAneModelCompatibility.existingBundlesRequiringMigration(
            in: repoDirectory,
            modelNames: ModelNames.KokoroAne.requiredCoreMLModels)

        XCTAssertEqual(
            incompatible,
            [],
            "Published Kokoro ANE bundles must expose FlexibleShapeInformation to OS 27")
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

        // Duration metadata should describe exactly the token sequence and
        // acoustic-frame count that were used by the alignment stage.
        XCTAssertEqual(result.inputIds.count, result.encoderTokens)
        XCTAssertEqual(result.predictedDurations.count, result.inputIds.count)
        XCTAssertEqual(
            result.predictedDurations.reduce(0) { $0 + Int($1) },
            result.acousticFrames)
        XCTAssertTrue(result.predictedDurations.allSatisfy { $0 >= 1 })

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
