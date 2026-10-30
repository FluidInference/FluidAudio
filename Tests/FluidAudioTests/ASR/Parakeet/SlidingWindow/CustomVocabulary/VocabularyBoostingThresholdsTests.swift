import XCTest

@testable import FluidAudio

/// `VocabularyBoostingSession.Thresholds` on a real recording (#967): the default `detectedTerms` floor lists a
/// spoken term but not a junk one, and the `cbw` override reaches the rescorer.
///
/// Needs the Parakeet v3 and CTC 110M models; runs when both are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a
/// download.
@available(macOS 14.0, iOS 17.0, *)
final class VocabularyBoostingThresholdsTests: XCTestCase {

    func testThresholdsOnRealRecording() async throws {
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        let asrCached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory())
        let ctcCached = CtcModels.modelsExist(at: CtcModels.defaultCacheDirectory())
        try XCTSkipUnless(
            (asrCached && ctcCached) || allowDownload,
            "Parakeet v3 + CTC models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        guard
            let url = Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav")
        else {
            throw XCTSkip("fixture missing from test bundle")
        }

        let asr = AsrManager(config: .default)
        try await asr.loadModels(try await AsrModels.downloadAndLoad())
        let ctcModels = try await CtcModels.downloadAndLoad()
        let samples = try AudioConverter().resampleAudioFile(url)
        var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
        let result = try await asr.transcribe(samples, decoderState: &state)
        let timings = try XCTUnwrap(result.tokenTimings)
        XCTAssertTrue(result.text.lowercased().contains("help them out"), result.text)

        func rescore(
            _ terms: [String], _ thresholds: VocabularyBoostingSession.Thresholds = .default
        ) async throws
            -> VocabularyRescorer.RescoreOutput?
        {
            let vocabulary = CustomVocabularyContext(terms: terms.map { CustomVocabularyTerm(text: $0) })
            let session = try await VocabularyBoostingSession(
                vocabulary: vocabulary, ctcModels: ctcModels, thresholds: thresholds)
            return await session.rescore(text: result.text, tokenTimings: timings, audioSamples: samples)
        }

        // The spotter's own floor (-15 per token) listed every term on any speech, junk included.
        let detected = try await rescore(["Codex", "zzqqxvwj"])?.detectedTerms ?? []
        XCTAssertTrue(detected.contains("Codex"), "spoken term missing from detectedTerms: \(detected)")
        XCTAssertFalse(detected.contains("zzqqxvwj"), "junk term listed in detectedTerms: \(detected)")

        // A near-miss of the spoken "help": an overwhelming cbw forces the replacement, a prohibitive one blocks it.
        let forced = try await rescore(["Helpp"], .init(cbw: 100))
        XCTAssertEqual(forced?.wasModified, true)
        XCTAssertTrue(forced?.text.contains("Helpp them out") ?? false, forced?.text ?? "nil")
        let blocked = try await rescore(["Helpp"], .init(cbw: -100))
        XCTAssertFalse(blocked?.wasModified ?? false, blocked?.text ?? "nil")
    }

    /// The managers hand their thresholds to the session they build. Needs only the CTC 110M models.
    func testManagersPassThresholdsToTheirSession() async throws {
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            CtcModels.modelsExist(at: CtcModels.defaultCacheDirectory()) || allowDownload,
            "CTC models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        let ctcModels = try await CtcModels.downloadAndLoad()
        let vocabulary = CustomVocabularyContext(terms: [CustomVocabularyTerm(text: "Tuist")])
        let thresholds = VocabularyBoostingSession.Thresholds(
            cbw: 0, minSimilarity: 0.7, marginSeconds: 0.2, detectionMinScore: -5)

        let slidingWindow = SlidingWindowAsrManager()
        try await slidingWindow.configureVocabularyBoosting(
            vocabulary: vocabulary, ctcModels: ctcModels, thresholds: thresholds)
        let slidingWindowThresholds = await slidingWindow.vocabularyBoosting?.thresholds
        XCTAssertEqual(slidingWindowThresholds, thresholds)

        let unified = UnifiedAsrManager()
        try await unified.configureVocabularyBoosting(
            vocabulary: vocabulary, ctcModels: ctcModels, thresholds: thresholds)
        let unifiedThresholds = await unified.vocabularyBoosting?.thresholds
        XCTAssertEqual(unifiedThresholds, thresholds)

        let streamingUnified = StreamingUnifiedAsrManager()
        try await streamingUnified.configureVocabularyBoosting(
            vocabulary: vocabulary, ctcModels: ctcModels, thresholds: thresholds)
        let streamingUnifiedThresholds = await streamingUnified.vocabularyBoosting?.thresholds
        XCTAssertEqual(streamingUnifiedThresholds, thresholds)
    }
}
