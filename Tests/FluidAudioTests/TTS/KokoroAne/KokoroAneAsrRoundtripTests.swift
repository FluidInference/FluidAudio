import Foundation
import XCTest

@testable import FluidAudio

/// Heavy E2E TTS→ASR roundtrip tests.
///
/// Gated by `FLUIDAUDIO_RUN_KOKOROANE_E2E=1` because it downloads ~hundreds of
/// MB of CoreML models (KokoroAne 7-stage chain + Parakeet ASR) and compiles
/// them on first run. Pass criteria: WER ≤ 0.10 per phrase (matches the
/// existing kokoro backend bar in TTSCommand).
final class KokoroAneAsrRoundtripTests: XCTestCase {

    private var shouldRunHeavy: Bool {
        ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKOROANE_E2E"] == "1"
    }

    /// Phrases mirror the suite in the Phase 3 plan with per-phrase WER bounds.
    /// Roundtrip WER compounds TTS + ASR error — OOV stress words and long
    /// sentences are looser. The ceilings are conservative; passing means the
    /// audio is intelligible enough for Parakeet to reproduce most words.
    private struct Phrase {
        let text: String
        let werCeiling: Double
        /// `true` for OOV/torture-test phrases where we only assert non-empty
        /// output rather than a WER bound.
        let isStressOnly: Bool
    }

    private let phrases: [Phrase] = [
        Phrase(text: "Hello world", werCeiling: 0.10, isStressOnly: false),
        Phrase(
            text: "The quick brown fox jumps over the lazy dog",
            werCeiling: 0.20, isStressOnly: false),
        Phrase(
            text: "Supercalifragilisticexpialidocious",
            werCeiling: 0.0, isStressOnly: true),
        Phrase(
            text:
                "Synthesis quality should remain stable across short and long inputs, "
                + "even when the input contains unusual punctuation, numbers like 2024, "
                + "and proper nouns like Cupertino.",
            werCeiling: 0.20, isStressOnly: false),
    ]

    func testRoundtripWERWithinThresholdForAllPhrases() async throws {
        try XCTSkipUnless(
            shouldRunHeavy,
            "Set FLUIDAUDIO_RUN_KOKOROANE_E2E=1 to run TTS→ASR roundtrip tests.")

        let tts = KokoroAneManager()
        try await tts.initialize()

        let asrModels = try await AsrModels.downloadAndLoad()
        let asr = AsrManager()
        try await asr.loadModels(asrModels)
        let decoderLayers = await asr.decoderLayerCount

        defer {
            Task { await asr.cleanup() }
        }

        for (i, phrase) in phrases.enumerated() {
            let detailed = try await tts.synthesizeDetailed(
                text: phrase.text, voice: nil, speed: 1.0)
            let wav = try AudioWAV.data(
                from: detailed.samples, sampleRate: Double(detailed.sampleRate))

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("kl-roundtrip-\(i)-\(UUID().uuidString).wav")
            try wav.write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }

            var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
            let transcription = try await asr.transcribe(url, decoderState: &decoderState)

            if phrase.isStressOnly {
                XCTAssertFalse(
                    transcription.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                    "Stress phrase \(i + 1) produced empty transcription:  ref: \(phrase.text)")
                continue
            }

            let m = WERCalculatorTestShim.calculateWER(
                hypothesis: transcription.text, reference: phrase.text)
            XCTAssertLessThanOrEqual(
                m, phrase.werCeiling,
                """
                Phrase \(i + 1) WER \(m) exceeds ceiling \(phrase.werCeiling).
                  ref: \(phrase.text)
                  hyp: \(transcription.text)
                """)
        }
    }
}

/// Library-internal WER shim — mirrors `WERCalculator` in the CLI layer so
/// these unit tests don't depend on the CLI module.
private enum WERCalculatorTestShim {

    static func calculateWER(hypothesis: String, reference: String) -> Double {
        let h = normalize(hypothesis)
            .components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let r = normalize(reference)
            .components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        guard !r.isEmpty else { return 0.0 }
        return Double(editDistance(h, r)) / Double(r.count)
    }

    private static func normalize(_ s: String) -> String {
        let lowered = s.lowercased()
        let scalars = lowered.unicodeScalars.filter {
            CharacterSet.letters.contains($0)
                || CharacterSet.decimalDigits.contains($0)
                || $0 == " "
        }
        return String(String.UnicodeScalarView(scalars))
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func editDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1]))
                }
            }
        }
        return dp[m][n]
    }
}
