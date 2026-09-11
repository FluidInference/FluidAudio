import AVFoundation
import XCTest

@testable import FluidAudio

/// End-to-end `finish()` regression for the streaming final window (issue #855).
///
/// Three real recordings (16 kHz mono, cleared for public release by the speaker,
/// published at github.com/saurabhav88/FluidAudio releases `issue-855-actionable-repros`)
/// where the frame-0 re-decode of the final window with the *carried* decoder
/// state emitted nothing, silently dropping the last 5–24 words. Batch decode of
/// the same audio is complete. Each case asserts the streaming transcript still
/// carries the recording's final words.
///
/// Needs the Parakeet TDT v3 models. Runs when they are already cached or when
/// `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download; otherwise skips.
@available(macOS 14.0, iOS 17.0, *)
final class SlidingWindowFinalWindowRegressionTests: XCTestCase {

    private struct Fixture {
        let file: String
        /// Words that only the final window can produce, lower-cased.
        let tail: String
        /// Seam text that must be spelled as batch does (#897), lower-cased.
        var seam: String? = nil
        /// Seam artifact that must not survive (#897), lower-cased.
        var seamArtifact: String? = nil
    }

    private let fixtures: [Fixture] = [
        Fixture(file: "01-validation-request-21.4s.wav", tail: "help them out"),
        Fixture(file: "02-release-readiness-19.8s.wav", tail: "cutting a release"),
        // Window 1 ends on the fragment "an" of "analyzing"; the final window
        // re-decodes the word. Before #897: "code and an, and analyzing".
        Fixture(
            file: "03-diff-explanation-16.9s.wav", tail: "in that difference",
            seam: "code and analyzing", seamArtifact: "and an,"),
    ]

    private func loadModels() async throws -> AsrModels {
        let cacheDir = AsrModels.defaultCacheDirectory()
        let cached = AsrModels.modelsExist(at: cacheDir)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            cached || allowDownload,
            "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        return try await AsrModels.downloadAndLoad()
    }

    private func fixtureURL(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        return url
    }

    /// Decode the whole fixture to 16 kHz mono Float32 samples.
    private func loadSamples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        XCTAssertEqual(format.sampleRate, 16_000, "fixtures are 16 kHz")
        XCTAssertEqual(format.channelCount, 1, "fixtures are mono")
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else {
            throw XCTSkip("could not allocate a buffer for \(url.lastPathComponent)")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else {
            throw XCTSkip("fixture \(url.lastPathComponent) is not float PCM")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    /// A self-contained 1 s buffer, like a live microphone tap delivers. Built
    /// fresh per chunk (own format, own storage) so it can be sent to the actor.
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

    private func streamTranscript(
        _ url: URL, models: AsrModels, chunkSeconds: TimeInterval? = nil
    ) async throws
        -> String
    {
        let manager =
            chunkSeconds.map { SlidingWindowAsrManager(config: SlidingWindowAsrConfig(chunkSeconds: $0)) }
            ?? SlidingWindowAsrManager()
        try await manager.loadModels(models)
        try await manager.startStreaming()

        let samples = try loadSamples(url)
        let chunk = 16_000  // 1 s
        var position = 0
        while position < samples.count {
            let end = min(position + chunk, samples.count)
            guard let buffer = Self.makeChunk(samples[position..<end]) else {
                XCTFail("could not allocate chunk buffer")
                break
            }
            await manager.streamAudio(buffer)
            position = end
        }
        return try await manager.finish()
    }

    /// The shipped v3 vocabulary resolves `.` `?` `!` to 7883 / 7956 / 8020;
    /// the ids the constant carried before #905 (7952, 7948) are `й` and `ó`.
    func testPunctuationTokenIdsResolveAgainstShippedV3Vocabulary() async throws {
        let models = try await loadModels()
        XCTAssertEqual(ASRConstants.punctuationTokenIds(in: models.vocabulary), [7883, 7956, 8020])
        XCTAssertEqual(models.vocabulary[7952], "й")
        XCTAssertEqual(models.vocabulary[7948], "ó")
    }

    /// Interior windows are re-decoded on a fresh state like the final one
    /// (#897). Before: entering window 2 mid-way with the carried state blanked
    /// the rest of the window (clip 01 at 7 s lost `can you do me a favor …
    /// codex`, at 6 s `even share some audio`, at 8 s `can you do me a favor
    /// and validate your`), the re-walked overlap re-emitted `environment` as
    /// `air environment` (clip 02 at 7 s), and dedup's 2 s tolerance chopped
    /// `the net new code` after the seam (clip 03 at 9 s).
    func testInteriorWindowsKeepMidStreamSpeechOnRealRecordings() async throws {
        let models = try await loadModels()
        struct Case {
            let file: String
            let chunkSeconds: TimeInterval
            let expected: [String]
            var forbidden: [String] = []
        }
        let cases: [Case] = [
            Case(
                file: "01-validation-request-21.4s.wav", chunkSeconds: 7,
                expected: ["can you do me a favor and validate your findings", "help them out"]),
            Case(
                file: "01-validation-request-21.4s.wav", chunkSeconds: 6,
                expected: ["even share some audio recordings", "validate your findings"]),
            Case(
                file: "01-validation-request-21.4s.wav", chunkSeconds: 8,
                expected: ["can you do me a favor and validate your findings"]),
            Case(
                file: "02-release-readiness-19.8s.wav", chunkSeconds: 7,
                expected: ["release work we've done", "cutting a release"], forbidden: ["environment. air"]),
            Case(
                file: "03-diff-explanation-16.9s.wav", chunkSeconds: 9,
                expected: ["the old code and the net new code and analyzing"]),
            // 0.9 s of new audio behind the final seam: the 2.9 s flush window
            // emitted nothing from a fresh state until it was end-aligned.
            Case(
                file: "03-diff-explanation-16.9s.wav", chunkSeconds: 8,
                expected: ["problems in that difference"]),
        ]
        for c in cases {
            let url = try fixtureURL(c.file)
            let text = try await streamTranscript(url, models: models, chunkSeconds: c.chunkSeconds).lowercased()
            for phrase in c.expected {
                XCTAssertTrue(
                    text.contains(phrase), "\(c.file) @ \(c.chunkSeconds)s: expected '\(phrase)' in: \(text)")
            }
            for phrase in c.forbidden {
                XCTAssertFalse(
                    text.contains(phrase), "\(c.file) @ \(c.chunkSeconds)s: artifact '\(phrase)' in: \(text)")
            }
        }
    }

    func testFinalWindowKeepsTrailingWordsOnRealRecordings() async throws {
        let models = try await loadModels()
        for fixture in fixtures {
            let url = try fixtureURL(fixture.file)
            let text = try await streamTranscript(url, models: models).lowercased()
            XCTAssertTrue(
                text.contains(fixture.tail),
                "\(fixture.file): streaming transcript lost its tail; expected '\(fixture.tail)' in: \(text)")
            if let seam = fixture.seam {
                XCTAssertTrue(
                    text.contains(seam), "\(fixture.file): seam not reconciled; expected '\(seam)' in: \(text)")
            }
            if let artifact = fixture.seamArtifact {
                XCTAssertFalse(
                    text.contains(artifact), "\(fixture.file): seam artifact '\(artifact)' survived in: \(text)")
            }
        }
    }
}
