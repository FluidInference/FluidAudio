import AVFoundation
import XCTest
import os

@testable import FluidAudio

/// End-to-end check of the three #851 defects on a real recording, through the
/// public API exactly as an integrator uses it. `minContextForConfirmation` is
/// set above the clip length so **no window ever confirms**, which is the state
/// every defect hid behind:
///
/// 1. terms built in code (`CustomVocabularyTerm(text:)`, no token IDs) must be
///    tokenized at configure time instead of silently ignored — the spotter can
///    only detect a term it has token IDs for;
/// 2. an unconfirmed window must still be rescored — the detection is asserted
///    on an update with `isConfirmed == false`;
/// 3. an unconfirmed window must not erase the previous window's volatile text —
///    asserted by the opening phrase surviving; with boosting on, `finish()`
///    builds from that text and returned "" before the fix.
///
/// The assertion is on spotter *activity* (`ctcDetectedTerms`, #899), not on a
/// particular misrecognition being corrected: decodes differ between machines
/// (a CI runner already emitted "follow-up" on this clip), so a text-only
/// assertion could pass on model output alone.
///
/// Needs the Parakeet v3 and CTC models; runs when both are cached or
/// `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download.
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
        // 60 s > 21.4 s clip: every window stays volatile for the whole stream.
        let config = SlidingWindowAsrConfig(minContextForConfirmation: 60)
        let manager = SlidingWindowAsrManager(config: config)
        try await manager.loadModels(asrModels)
        // Untokenized on purpose: this is the documented in-code path.
        let vocabulary = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: "Codex"), CustomVocabularyTerm(text: "follow-up"),
        ])
        try await manager.configureVocabularyBoosting(vocabulary: vocabulary, ctcModels: ctcModels)
        try await manager.startStreaming()

        let updates = OSAllocatedUnfairLock<[SlidingWindowTranscriptionUpdate]>(initialState: [])
        let consumer = Task {
            for await update in await manager.transcriptionUpdates {
                updates.withLock { $0.append(update) }
            }
        }
        defer { consumer.cancel() }

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
        let seen = updates.withLock { $0 }

        XCTAssertFalse(folded.isEmpty, "boosted streaming transcript must not be empty")
        // Fix 3: the first (volatile) window's text survives the second volatile window.
        XCTAssertTrue(folded.contains("before we go to them"), "first window lost: \(text)")
        // #855 fixture contract: the final window's tail survives.
        XCTAssertTrue(folded.contains("help them out"), "tail lost: \(text)")
        XCTAssertTrue(folded.contains("codex"), "vocabulary word missing from: \(text)")

        // Fixes 1 + 2: the spotter found "Codex" (so the in-code term had token
        // IDs) inside a window that was never confirmed (so it was rescored).
        XCTAssertFalse(seen.isEmpty, "no streaming updates observed")
        XCTAssertTrue(seen.allSatisfy { !$0.isConfirmed }, "no window may confirm in this configuration")
        let detectedInVolatile = seen.contains { update in
            !update.isConfirmed && (update.ctcDetectedTerms ?? []).contains { $0.lowercased() == "codex" }
        }
        XCTAssertTrue(
            detectedInVolatile,
            "spotter never reported 'Codex' on an unconfirmed window; detections: \(seen.map { $0.ctcDetectedTerms ?? [] })"
        )
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
