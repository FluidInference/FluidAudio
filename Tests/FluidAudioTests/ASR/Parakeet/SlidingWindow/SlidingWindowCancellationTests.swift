@preconcurrency import AVFoundation
import XCTest

@testable import FluidAudio

@MainActor
final class SlidingWindowCancellationTests: XCTestCase {
    func testCancelJoinsRecognitionBeforeReleasingItsOwner() async throws {
        let cache = AsrModels.defaultCacheDirectory()
        try XCTSkipUnless(AsrModels.modelsExist(at: cache), "Install the Parakeet v3 models before running this test")
        let models = try await AsrModels.load(from: cache)
        let names = ["01-validation-request-21.4s", "02-release-readiness-19.8s", "03-diff-explanation-16.9s"]
        let samples = try names.flatMap { try recordedSpeech($0) }
        let owner = try await cancelBusyManager(models: models, samples: samples)
        XCTAssertNil(owner.manager, "cancel() must join the task retaining the manager before returning")
    }

    private func cancelBusyManager(models: AsrModels, samples: [Float]) async throws -> WeakOwner {
        let manager = SlidingWindowAsrManager()
        let owner = WeakOwner(manager)
        try await manager.loadModels(models)
        try await manager.startStreaming()

        let updates = await manager.transcriptionUpdates
        let firstWindow = expectation(description: "first decoded window")
        let consumer = Task {
            for await _ in updates {
                firstWindow.fulfill()
                break
            }
        }

        // A single queued buffer keeps several real windows in the same drain.
        // Cancelling after its first result must join that drain before returning.
        try await manager.streamAudio(makeBuffer(samples))
        await fulfillment(of: [firstWindow], timeout: 60)
        consumer.cancel()
        await consumer.value

        await manager.cancel()
        return owner
    }

    private func recordedSpeech(_ name: String) throws -> [Float] {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: "wav")
                ?? Bundle.module.url(forResource: name, withExtension: "wav"))
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private func makeBuffer(_ samples: [Float]) throws -> sending AVAudioPCMBuffer {
        let format = try XCTUnwrap(
            AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false))
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress { channel.update(from: base, count: samples.count) }
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }

    private final class WeakOwner {
        weak var manager: SlidingWindowAsrManager?

        init(_ manager: SlidingWindowAsrManager) {
            self.manager = manager
        }
    }
}
