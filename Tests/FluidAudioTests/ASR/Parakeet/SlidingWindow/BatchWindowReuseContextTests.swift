#if DEBUG
import AVFoundation
import XCTest

@testable import FluidAudio

/// Protects full-recording authority and recording isolation when prior batch windows are reused.
@MainActor
final class BatchWindowReuseContextTests: XCTestCase {
    private func assertTimingsEqual(
        _ actual: [TokenTiming]?, _ expected: [TokenTiming]?, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual?.map(\.token), expected?.map(\.token), file: file, line: line)
        XCTAssertEqual(actual?.map(\.tokenId), expected?.map(\.tokenId), file: file, line: line)
        XCTAssertEqual(actual?.map(\.startTime), expected?.map(\.startTime), file: file, line: line)
        XCTAssertEqual(actual?.map(\.endTime), expected?.map(\.endTime), file: file, line: line)
        XCTAssertEqual(actual?.map(\.confidence), expected?.map(\.confidence), file: file, line: line)
    }

    private func loadSpeech() throws -> [Float] {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_BATCH_WINDOW_REUSE"] == "1")
        #if SWIFT_PACKAGE
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/01-validation-request-21.4s", withExtension: "wav")
                ?? Bundle.module.url(forResource: "01-validation-request-21.4s", withExtension: "wav"))
        #else
        let url = URL(fileURLWithPath: try XCTUnwrap(ProcessInfo.processInfo.environment["NON_EOU_AUDIO"]))
        #endif
        let file = try AVAudioFile(forReading: url)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private func loadManager(_ version: AsrModelVersion) async throws -> AsrManager {
        AsrManager(models: try await AsrModels.downloadAndLoad(version: version))
    }

    func testAQuietWindowReusesItsSuccessfulEmptyDecode() async throws {
        let recording = try loadSpeech()
        // A quiet interval between words in the original recording, 5.15–5.95 s.
        let samples = Array(recording[82_400..<95_200])
        let version = AsrModelVersion.v3
        let manager = try await loadManager(version)
        let reuse = BatchWindowReuseContext()

        for _ in 0..<2 {
            let result = try await ChunkProcessor.transcribeChunk(
                samples: samples, contextSamples: 0, chunkStart: 82_400, isLastChunk: false,
                using: manager, decoderLayers: version.decoderLayers,
                maxModelSamples: ASRConstants.maxModelSamples, language: .english, reuse: reuse.cache
            )
            XCTAssertTrue(result.tokens.isEmpty, "The fixture must produce a successful empty decode")
            XCTAssertTrue(result.timestamps.isEmpty && result.confidences.isEmpty && result.durations.isEmpty)
        }
        let stats = await reuse.statistics()
        XCTAssertEqual(stats.misses, 1)
        XCTAssertEqual(stats.hits, 1, "Repeating a quiet window must not rerun inference")
        await manager.cleanup()
    }

    // An interior window may get a different flush decision when evaluated in
    // a longer recording. Its cached tokens must never cross that boundary.
    func testIdenticalAudioDoesNotReuseAcrossPendingTokenFlushPolicies() async throws {
        let samples = Array(try loadSpeech().prefix(ASRConstants.maxModelSamples))
        let version = AsrModelVersion.v3
        let manager = try await loadManager(version)
        func decode(flush: Bool, reuse: BatchWindowReuseContext?) async throws -> [Int] {
            try await ChunkProcessor.transcribeChunk(
                samples: samples, contextSamples: 0, chunkStart: 0, isLastChunk: false,
                flushPendingTokens: flush, using: manager, decoderLayers: version.decoderLayers,
                maxModelSamples: ASRConstants.maxModelSamples, language: .english, reuse: reuse?.cache
            ).tokens
        }

        let uncached = try await decode(flush: true, reuse: nil)
        XCTAssertFalse(uncached.isEmpty)
        let reuse = BatchWindowReuseContext()
        _ = try await decode(flush: false, reuse: reuse)
        let flushed = try await decode(flush: true, reuse: reuse)
        let stats = await reuse.statistics()
        XCTAssertEqual(stats.misses, 2, "A different flush decision requires a fresh decode")
        XCTAssertEqual(flushed, uncached)
        _ = try await decode(flush: true, reuse: reuse)
        let reused = await reuse.statistics()
        XCTAssertEqual(reused.hits, 1)
        await manager.cleanup()
    }

    /// The context is keyed by the loaded encoder: the same window decoded by another
    /// model set is never answered from the first one's result.
    func testAContextDoesNotReuseAcrossModels() async throws {
        let samples = Array(try loadSpeech().prefix(ASRConstants.maxModelSamples))
        let reuse = BatchWindowReuseContext()
        for version: AsrModelVersion in [.v3, .v2] {
            let manager = try await loadManager(version)
            _ = try await ChunkProcessor.transcribeChunk(
                samples: samples, contextSamples: 0, chunkStart: 0, isLastChunk: true,
                using: manager, decoderLayers: version.decoderLayers,
                maxModelSamples: ASRConstants.maxModelSamples, language: .english, reuse: reuse.cache
            )
            await manager.cleanup()
        }
        let stats = await reuse.statistics()
        XCTAssertEqual(stats.hits, 0)
        XCTAssertEqual(stats.misses, 2)
    }

    func testGrowingRealRecordingPreservesFinalTokensAndANewContextStartsEmpty() async throws {
        let samples = try loadSpeech()
        for version: AsrModelVersion in [.v2, .v3, .tdtCtc110m] {
            let manager = try await loadManager(version)
            func decode(_ audio: [Float], reuse: BatchWindowReuseContext?) async throws -> ASRResult {
                var state = TdtDecoderState.make(decoderLayers: version.decoderLayers)
                return try await manager.transcribe(audio, decoderState: &state, language: .english, reuse: reuse)
            }
            let baseline = try await decode(samples, reuse: nil)
            let recording = BatchWindowReuseContext()
            _ = try await decode(Array(samples.prefix(16 * 16_000)), reuse: recording)
            _ = try await decode(Array(samples.prefix(19 * 16_000)), reuse: recording)
            let cached = try await decode(samples, reuse: recording)
            XCTAssertEqual(cached.text, baseline.text)
            XCTAssertEqual(cached.confidence, baseline.confidence)
            XCTAssertEqual(cached.duration, baseline.duration)
            assertTimingsEqual(cached.tokenTimings, baseline.tokenTimings)
            let reused = await recording.statistics()
            XCTAssertGreaterThan(reused.hits, 0)

            // The next recording gets its own context and decodes like an uncached call.
            let nextContext = BatchWindowReuseContext()
            let empty = await nextContext.statistics()
            XCTAssertEqual(empty.entries, 0)
            let nextRecording = try await decode(Array(samples.dropFirst(16_000)), reuse: nextContext)
            let nextBaseline = try await decode(Array(samples.dropFirst(16_000)), reuse: nil)
            XCTAssertEqual(nextRecording.text, nextBaseline.text)
            assertTimingsEqual(nextRecording.tokenTimings, nextBaseline.tokenTimings)
            await manager.cleanup()
        }
    }
}
#endif
