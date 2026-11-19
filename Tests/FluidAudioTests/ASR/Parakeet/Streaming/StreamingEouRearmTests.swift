import AVFoundation
import XCTest
import os

@testable import FluidAudio

/// `rearm(keepTranscript:)` against the real Parakeet EOU 320 ms models, fed with real recordings from the repo:
/// the first fixture, a pause, then the third fixture ("diff means comparing what is net new ...").
///
/// The models are not bundled. Point `FLUIDAUDIO_EOU_MODELS_DIR` at a folder holding the `320ms` model directory, or
/// cache them in the default location; the tests skip when neither has them.
final class StreamingEouRearmTests: XCTestCase {

    private static let secondClipStart = "diff means comparing what is net new"
    private let sampleRate = 16000

    // MARK: - Fixtures

    private func modelDirectory() throws -> URL {
        let root: URL
        if let override = ProcessInfo.processInfo.environment["FLUIDAUDIO_EOU_MODELS_DIR"] {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            root = support.appendingPathComponent("FluidAudio/Models/parakeet-eou-streaming", isDirectory: true)
        }
        let directory = root.appendingPathComponent(StreamingChunkSize.ms320.modelSubdirectory, isDirectory: true)
        let encoder = directory.appendingPathComponent("streaming_encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoder.path) else {
            throw XCTSkip("Parakeet EOU 320ms models not found at \(directory.path)")
        }
        return directory
    }

    private func fixture(_ name: String) throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: "wav")
                ?? Bundle.module.url(forResource: name, withExtension: "wav")
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        return try AudioConverter().resampleAudioFile(url)
    }

    /// Clip, `pauseSeconds` of silence, clip, then trailing silence long enough for a second end to confirm.
    private func twoUtterances(pauseSeconds: Double) throws -> (audio: [Float], secondStartMs: Int) {
        let first = try fixture("01-validation-request-21.4s")
        let second = try fixture("03-diff-explanation-16.9s")
        let pause = [Float](repeating: 0, count: Int(pauseSeconds * Double(sampleRate)))
        let audio = first + pause + second + [Float](repeating: 0, count: 3 * sampleRate)
        return (audio, (first.count + pause.count) * 1000 / sampleRate)
    }

    private static func buffer(_ samples: ArraySlice<Float>, sampleRate: Int) throws -> AVAudioPCMBuffer {
        let format = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)
        )
        let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
        pcm.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            pcm.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }
        return pcm
    }

    private func loadedManager() async throws -> StreamingEouAsrManager {
        let manager = StreamingEouAsrManager(chunkSize: .ms320)
        try await manager.loadModels(from: try modelDirectory())
        return manager
    }

    /// Feeds `samples` in 100 ms steps. `afterStep` runs between steps, with nothing in flight.
    private func feed(
        _ samples: [Float], to manager: StreamingEouAsrManager,
        afterStep: () async throws -> Void = {}
    ) async throws {
        let step = sampleRate / 10
        var start = 0
        while start < samples.count {
            let end = min(start + step, samples.count)
            try await manager.appendAudio(try Self.buffer(samples[start..<end], sampleRate: sampleRate))
            try await manager.processBufferedAudio()
            try await afterStep()
            start = end
        }
    }

    private func confirmCounter(on manager: StreamingEouAsrManager) async -> OSAllocatedUnfairLock<Int> {
        let count = OSAllocatedUnfairLock(initialState: 0)
        await manager.setEouCallback { _ in count.withLock { $0 += 1 } }
        return count
    }

    // MARK: - Tests

    /// The words that follow a pause are the ones a raw `<EOU>` strands. Without a rearm the second clip's first
    /// words are gone from the transcript; rearming on the confirmed end replays them.
    func testRearmOnConfirmedEndRecoversTheWordsAfterThePause() async throws {
        let (audio, _) = try twoUtterances(pauseSeconds: 1.0)

        let control = try await loadedManager()
        _ = await confirmCounter(on: control)
        try await feed(audio, to: control)
        let controlTokenTimes = await control.getTokenTimestampsMs()
        let controlLastTokenMs = try XCTUnwrap(controlTokenTimes.last)
        let withoutRearm = try await control.finish()
        XCTAssertFalse(
            withoutRearm.contains(Self.secondClipStart),
            "the scenario no longer strands the words after the pause, pick a clip that does: \(withoutRearm)")

        let manager = try await loadedManager()
        let confirms = await confirmCounter(on: manager)
        var handled = 0
        try await feed(audio, to: manager) {
            let seen = confirms.withLock { $0 }
            guard seen > handled else { return }
            handled = seen
            try await manager.rearm(keepTranscript: true)
        }
        let tokenTimes = await manager.getTokenTimestampsMs()
        let lastTokenMs = try XCTUnwrap(tokenTimes.last)
        let withRearm = try await manager.finish()

        XCTAssertTrue(withRearm.contains(Self.secondClipStart), withRearm)
        XCTAssertEqual(
            Double(lastTokenMs), Double(controlLastTokenMs), accuracy: 640,
            "the replay put the conversation clock back where the audio is")
        XCTAssertTrue(withRearm.hasPrefix("before we go to them"), "the first utterance was kept: \(withRearm)")
    }

    /// The `eouDetected` latch is only cleared by `reset()`, so without a rearm the callback fires once per manager
    /// state. After a rearm the second utterance confirms too.
    func testRearmClearsTheLatchSoTheNextUtteranceConfirms() async throws {
        let (audio, secondStartMs) = try twoUtterances(pauseSeconds: 3.0)

        let control = try await loadedManager()
        _ = await confirmCounter(on: control)
        try await feed(audio, to: control)
        let controlEnds = await control.getEouTimestampsMs()
        XCTAssertEqual(controlEnds.count, 1, "without a rearm the end confirms once")
        XCTAssertLessThan(try XCTUnwrap(controlEnds.last), secondStartMs)

        let manager = try await loadedManager()
        let confirms = await confirmCounter(on: manager)
        var handled = 0
        try await feed(audio, to: manager) {
            let seen = confirms.withLock { $0 }
            guard seen > handled else { return }
            handled = seen
            try await manager.rearm(keepTranscript: true)
            let latched = await manager.eouDetected
            XCTAssertFalse(latched, "rearm clears the latch")
        }

        let ends = await manager.getEouTimestampsMs()
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(ends.last), secondStartMs, "the second utterance's end confirmed")
        XCTAssertEqual(ends.count, confirms.withLock { $0 })
    }

    /// `keepTranscript` decides whether the words decoded so far survive. The audio since the anchor here is the
    /// silence after the first clip, so a cleared transcript stays empty and a kept one is unchanged.
    func testKeepTranscriptDecidesWhetherEarlierWordsSurvive() async throws {
        let first = try fixture("01-validation-request-21.4s")
        let audio = first + [Float](repeating: 0, count: 3 * sampleRate)

        for keep in [true, false] {
            let manager = try await loadedManager()
            let confirms = await confirmCounter(on: manager)
            try await feed(audio, to: manager)
            XCTAssertEqual(confirms.withLock { $0 }, 1)
            let before = await manager.getPartialTranscript()
            XCTAssertFalse(before.isEmpty)

            try await manager.rearm(keepTranscript: keep)

            let after = await manager.getPartialTranscript()
            XCTAssertEqual(after, keep ? before : "", "keepTranscript: \(keep)")
        }
    }

    /// A rearm requested from the callback's own task lands while a chunk is in flight. It must wait for that chunk
    /// instead of replacing the state under it, and still take effect.
    /// The manager is an actor, so a second drain can start while the first is suspended in a decode: a `rearm` that
    /// drains while the feeder calls `processBufferedAudio` must not decode the same chunk twice.
    func testARearmAndAProcessCallRunningTogetherDecodeEachChunkOnce() async throws {
        let audio = try fixture("01-validation-request-21.4s")

        let control = try await loadedManager()
        try await control.appendAudio(try Self.buffer(audio[...], sampleRate: sampleRate))
        try await control.rearm()
        try await control.processBufferedAudio()
        let sequential = try await control.finish()

        let manager = try await loadedManager()
        try await manager.appendAudio(try Self.buffer(audio[...], sampleRate: sampleRate))
        async let rearmed: Void = manager.rearm()
        async let processed: Void = manager.processBufferedAudio()
        _ = try await (rearmed, processed)
        let together = try await manager.finish()

        XCTAssertFalse(sequential.isEmpty)
        XCTAssertEqual(together, sequential, "a chunk was decoded by two drains at once")
    }

    func testRearmRequestedFromTheCallbackTakesEffect() async throws {
        let (audio, secondStartMs) = try twoUtterances(pauseSeconds: 1.0)
        let manager = try await loadedManager()
        let confirms = OSAllocatedUnfairLock(initialState: 0)
        await manager.setEouCallback { _ in
            confirms.withLock { $0 += 1 }
            Task { try await manager.rearm(keepTranscript: true) }
        }

        try await feed(audio, to: manager)
        let ends = await manager.getEouTimestampsMs()
        let transcript = try await manager.finish()

        XCTAssertGreaterThanOrEqual(try XCTUnwrap(ends.last), secondStartMs, "the second utterance's end confirmed")
        XCTAssertTrue(transcript.contains(Self.secondClipStart), transcript)
    }
}
