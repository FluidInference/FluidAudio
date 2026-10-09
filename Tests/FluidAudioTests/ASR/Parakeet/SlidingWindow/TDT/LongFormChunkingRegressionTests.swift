import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Issue #954: exercise chunking with real speech and check content, not just WER.
final class LongFormChunkingRegressionTests: XCTestCase {
    private func bundledRecording() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav"))
    }

    func testDefaultChunkStartsUseRegularStrideOnRealSpeech() throws {
        let samples = try AudioConverter().resampleAudioFile(bundledRecording())
        XCTAssertGreaterThan(samples.count, ASRConstants.maxModelSamples)
        let processor = ChunkProcessor(audioSamples: samples)
        let enabled = ASRConfig.default.resolvedMelChunkContext(for: .v3)
        let layout = processor.chunkLayoutForTesting(melChunkContext: enabled, modelVersion: .v3)
        let starts = try processor.chunkStartsForTesting(melChunkContext: enabled, modelVersion: .v3)

        XCTAssertEqual(layout.melContextSamples, ASRConstants.samplesPerEncoderFrame)
        XCTAssertEqual(layout.warmupPrefixSamples, 0)
        XCTAssertGreaterThan(starts.count, 1)
        XCTAssertEqual(starts, starts.indices.map { $0 * layout.strideSamples })
    }

    func testDefaultMatchesExplicitMelContextOnRealSpeech() async throws {
        let directory = AsrModels.defaultCacheDirectory()
        try XCTSkipUnless(
            AsrModels.modelsExist(at: directory), "Requires cached Parakeet v3 models; no automatic download")
        let models = try AsrModels.loadLocal(from: directory)
        let url = try bundledRecording()
        let defaultResult = try await transcribe(url, config: .default, models: models)
        let explicitResult = try await transcribe(url, config: ASRConfig(melChunkContext: true), models: models)

        XCTAssertEqual(defaultResult.text, explicitResult.text)
        XCTAssertEqual(defaultResult.tokenTimings?.map(\.tokenId), explicitResult.tokenTimings?.map(\.tokenId))
        assertPhrases(
            in: defaultResult.text,
            required: ["can you do me a favor and validate your findings", "help them out"],
            forbidden: [])
    }

    /// Uses the full public talk selected by the caller, preserving its original
    /// chunk boundaries. The confidential interview is deliberately not a fixture.
    func testPublicTalkKnownAnswers() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let audio = environment["FLUIDAUDIO_ISSUE_954_AUDIO"],
            let directory = environment["FLUIDAUDIO_ISSUE_954_MODELS"]
        else {
            throw XCTSkip("Set FLUIDAUDIO_ISSUE_954_AUDIO and FLUIDAUDIO_ISSUE_954_MODELS to opt in")
        }
        let url = URL(fileURLWithPath: audio)
        let audioFile = try AVAudioFile(forReading: url)
        // Reject isolated clips: they cannot exercise the reported window starts.
        guard Double(audioFile.length) / audioFile.processingFormat.sampleRate > 500 else {
            return XCTFail("Select the complete public talk from issue #954 (approximately ten minutes)")
        }
        let models = try AsrModels.loadLocal(from: URL(fileURLWithPath: directory))
        let result = try await transcribe(url, config: .default, models: models)
        assertPhrases(
            in: result.text,
            required: ["501c3", "oversell", "thousands of students", "JupyterHub"],
            forbidden: ["thousands of s students"])
    }

    private func transcribe(_ url: URL, config: ASRConfig, models: AsrModels) async throws -> ASRResult {
        let manager = AsrManager(config: config, models: models)
        var state = TdtDecoderState.make(decoderLayers: models.version.decoderLayers)
        return try await manager.transcribe(url, decoderState: &state)
    }

    private func assertPhrases(
        in text: String, required: [String], forbidden: [String],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        // Only case and whitespace normalization; preserve numbers, negation,
        // punctuation and repeated words so content regressions remain visible.
        let normalized = text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for phrase in required {
            XCTAssertTrue(
                normalized.contains(phrase.lowercased()), "Missing required phrase: \(phrase)", file: file, line: line)
        }
        for phrase in forbidden {
            XCTAssertFalse(
                normalized.contains(phrase.lowercased()), "Unexpected phrase: \(phrase)", file: file, line: line)
        }
    }
}
