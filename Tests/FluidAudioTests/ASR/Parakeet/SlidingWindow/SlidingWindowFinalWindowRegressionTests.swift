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
    }

    private let fixtures: [Fixture] = [
        Fixture(file: "01-validation-request-21.4s.wav", tail: "help them out"),
        Fixture(file: "02-release-readiness-19.8s.wav", tail: "cutting a release"),
        Fixture(file: "03-diff-explanation-16.9s.wav", tail: "in that difference"),
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

    private func streamTranscript(_ url: URL, models: AsrModels) async throws -> String {
        let manager = SlidingWindowAsrManager()
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

    func testFinalWindowKeepsTrailingWordsOnRealRecordings() async throws {
        let models = try await loadModels()
        for fixture in fixtures {
            let url = try fixtureURL(fixture.file)
            let text = try await streamTranscript(url, models: models).lowercased()
            XCTAssertTrue(
                text.contains(fixture.tail),
                "\(fixture.file): streaming transcript lost its tail; expected '\(fixture.tail)' in: \(text)")
        }
    }
}
