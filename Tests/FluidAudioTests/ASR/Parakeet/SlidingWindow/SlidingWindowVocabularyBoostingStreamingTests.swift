import AVFoundation
import XCTest

@testable import FluidAudio

/// End-to-end check of the three #851 defects on a real recording, through the
/// public API exactly as an integrator uses it:
///
/// 1. terms built in code (`CustomVocabularyTerm(text:)`, no token IDs) must be
///    tokenized at configure time instead of silently ignored;
/// 2. a window that is never confirmed (this clip has two windows; the second is
///    the flush) must still be rescored;
/// 3. an unconfirmed trailing window must not erase the previous window's text —
///    with boosting on, `finish()` builds from that text and returned "".
///
/// Asserts the transcript is non-empty, keeps the recording's tail, and still
/// contains the vocabulary term's word. Needs the Parakeet v3 and CTC models;
/// runs when both are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download.
@available(macOS 14.0, iOS 17.0, *)
final class SlidingWindowVocabularyBoostingStreamingTests: XCTestCase {

    func testInMemoryTermsRescoreAndKeepTailOnRealRecording() async throws {
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

        let asrModels = try await AsrModels.downloadAndLoad()
        let ctcModels = try await CtcModels.downloadAndLoad()
        let manager = SlidingWindowAsrManager()
        try await manager.loadModels(asrModels)
        // Untokenized on purpose: this is the documented in-code path.
        let vocabulary = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: "Codex"), CustomVocabularyTerm(text: "follow-up"),
        ])
        try await manager.configureVocabularyBoosting(vocabulary: vocabulary, ctcModels: ctcModels)
        try await manager.startStreaming()

        let samples = try Self.loadSamples(url)
        var position = 0
        while position < samples.count {
            let end = min(position + 16_000, samples.count)
            guard let buffer = Self.makeChunk(samples[position..<end]) else {
                XCTFail("could not allocate chunk buffer")
                break
            }
            await manager.streamAudio(buffer)
            position = end
        }
        let text = try await manager.finish()
        let folded = text.lowercased()

        XCTAssertFalse(folded.isEmpty, "boosted streaming transcript must not be empty")
        XCTAssertTrue(folded.contains("help them out"), "tail lost: \(text)")
        XCTAssertTrue(folded.contains("codex"), "vocabulary word missing from: \(text)")
    }

    private static func loadSamples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw XCTSkip("could not allocate a buffer for \(url.lastPathComponent)") }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else {
            throw XCTSkip("fixture is not float PCM")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private nonisolated static func makeChunk(_ samples: ArraySlice<Float>) -> AVAudioPCMBuffer? {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
            let channel = buffer.floatChannelData?[0]
        else { return nil }
        for (offset, sample) in samples.enumerated() {
            channel[offset] = sample
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }
}
