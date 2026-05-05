import Foundation
import XCTest

@testable import FluidAudio

/// Cross-backend parity smoke test: synthesize the same phrase through the
/// legacy 4-graph `StyleTTS2Manager` and the new 7-graph
/// `StyleTTS2AneManager`, then assert that both produce audio with
/// comparable duration and non-trivial energy.
///
/// **Gating:** This test requires the HF bundles (which are user-managed —
/// see Phase D of the plan) and a `ref_s.bin` voice blob. It self-skips
/// when the env vars are missing so CI stays green.
///
/// To run locally:
/// ```
/// FLUIDAUDIO_STYLETTS2_ANE_PARITY=1 \
/// FLUIDAUDIO_STYLETTS2_REF_S_PATH=/path/to/ref_s.bin \
///     swift test --filter StyleTTS2AneParityTests
/// ```
///
/// Once the HF bundles ship and we have a stable reference clip, this test
/// can be tightened to enforce a log-mel cosine ≥ 0.97 between the two
/// backends as the plan specifies. Until then, the parity assertions are
/// duration + RMS only.
final class StyleTTS2AneParityTests: XCTestCase {

    private static let phrase = "Hello world from the StyleTTS2 parity test."

    private struct Env {
        let refSURL: URL
    }

    private func loadEnvOrSkip() throws -> Env {
        let env = ProcessInfo.processInfo.environment
        guard env["FLUIDAUDIO_STYLETTS2_ANE_PARITY"] == "1" else {
            throw XCTSkip(
                "set FLUIDAUDIO_STYLETTS2_ANE_PARITY=1 to enable cross-backend "
                    + "parity test (requires HF bundle + ref_s.bin)")
        }
        guard let path = env["FLUIDAUDIO_STYLETTS2_REF_S_PATH"], !path.isEmpty else {
            throw XCTSkip("set FLUIDAUDIO_STYLETTS2_REF_S_PATH=/path/to/ref_s.bin")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("ref_s.bin not found at \(url.path)")
        }
        return Env(refSURL: url)
    }

    func testBothBackendsProduceComparableAudio() async throws {
        let env = try loadEnvOrSkip()

        // ---- Legacy 4-graph ----
        let legacy = StyleTTS2Manager()
        try await legacy.initialize()
        let legacyOut = try await legacy.synthesizeSamples(
            text: Self.phrase, voiceStyleURL: env.refSURL, randomSeed: 42)

        // ---- 7-graph ANE ----
        let ane = StyleTTS2AneManager()
        try await ane.initialize()
        let aneOut = try await ane.synthesizeSamples(
            text: Self.phrase, voiceStyleURL: env.refSURL, randomSeed: 42)

        // Same sample rate.
        XCTAssertEqual(legacyOut.sampleRate, aneOut.sampleRate)

        // Both produce non-empty audio.
        XCTAssertGreaterThan(legacyOut.samples.count, 0, "legacy backend produced 0 samples")
        XCTAssertGreaterThan(aneOut.samples.count, 0, "ANE backend produced 0 samples")

        // Duration parity (within 25%) — the same phrase + voice should not
        // produce wildly different lengths.
        let legacySec = Double(legacyOut.samples.count) / Double(legacyOut.sampleRate)
        let aneSec = Double(aneOut.samples.count) / Double(aneOut.sampleRate)
        let durRatio = max(legacySec, aneSec) / min(legacySec, aneSec)
        XCTAssertLessThanOrEqual(
            durRatio, 1.25,
            "duration mismatch too large: legacy=\(legacySec)s ane=\(aneSec)s")

        // RMS energy parity (within 6 dB) — catches dead-silent or massively
        // clipped output.
        let legacyRms = rms(legacyOut.samples)
        let aneRms = rms(aneOut.samples)
        XCTAssertGreaterThan(legacyRms, 1e-4, "legacy backend output is silent")
        XCTAssertGreaterThan(aneRms, 1e-4, "ANE backend output is silent")
        let dbDelta = abs(20 * log10(legacyRms / aneRms))
        XCTAssertLessThanOrEqual(
            dbDelta, 6.0,
            "RMS dB delta exceeds 6 dB: legacy=\(legacyRms) ane=\(aneRms)")
    }

    // MARK: - Helpers

    private func rms(_ x: [Float]) -> Float {
        guard !x.isEmpty else { return 0 }
        var acc: Double = 0
        for v in x { acc += Double(v) * Double(v) }
        return Float(sqrt(acc / Double(x.count)))
    }
}
