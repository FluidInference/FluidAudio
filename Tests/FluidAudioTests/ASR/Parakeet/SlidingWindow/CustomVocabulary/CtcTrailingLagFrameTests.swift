import XCTest

@testable import FluidAudio

/// A clip that stops on its last word keeps that word's CTC evidence (#1151).
///
/// The CTC head emits a token about one frame after its sound, so the last token of a
/// clip cut right after a word lands in the first padding frame. Trimming the padding
/// at the audio's end scored the word as if it were cut off, and a vocabulary term
/// could then outscore it ("months." -> "Monte" at the end of a dictation).
///
/// Needs the CTC 110M models. Runs when they are already cached or when
/// `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download; otherwise skips.
final class CtcTrailingLagFrameTests: XCTestCase {

    /// `01-validation-request` cut right after "share" (8.32 s), the way push-to-talk
    /// dictation ends. Scored on the trimmed frames alone, "share" got -5.75; with the
    /// lag frame kept it gets -2.40.
    func testWordAtTheEndOfAClipKeepsItsLastTokenFrame() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            CtcModels.modelsExist(at: directory) || allowDownload,
            "CTC 110M not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        _ = try await CtcModels.download(to: directory)
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)

        let name = "01-validation-request-21.4s.wav"
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
        let clip = Array(try AudioConverter().resampleAudioFile(url).prefix(133_120))

        let result = try await spotter.computeLogProbs(for: clip)
        // The frame duration still spans the audio frames only, so the kept lag frame
        // shifts no timestamp.
        let audioFrames = result.logProbs.count - CtcKeywordSpotter.trailingLagFrames
        XCTAssertEqual(
            result.frameDuration * Double(audioFrames), Double(clip.count) / Double(ASRConstants.sampleRate),
            accuracy: 1e-9)

        let wordStart = 8.08 - 0.5
        let (score, _, _) = spotter.ctcWordSpotConstrained(
            logProbs: result.logProbs,
            keywordTokens: tokenizer.encode("share"),
            searchStartFrame: Int(wordStart / result.frameDuration),
            searchEndFrame: result.logProbs.count)
        XCTAssertGreaterThan(score, -4.0, "'share' at the end of the clip lost its last token frame")
    }
}
