@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Foundation
import XCTest
import os

@testable import FluidAudio

/// Contract tests for `setEncoderOutputCallback`: across a stream the callbacks'
/// global frames (origin + local range) tile `[0, n)` with no gap or overlap, cover
/// every decoded token's frame, and restart at 0 after `reset()`. Uses real FLEURS
/// speech and the cached models; skips when either is absent (e.g. on CI).
final class EncoderOutputCallbackTests: XCTestCase {

    private final class Recorder: Sendable {
        private let ranges = OSAllocatedUnfairLock<[Range<Int>]>(initialState: [])
        private let keys = OSAllocatedUnfairLock<Set<String>>(initialState: [])

        var callback: StreamingEncoderOutputCallback {
            { [ranges, keys] outputs, local, origin in
                ranges.withLock { $0.append((origin + local.lowerBound)..<(origin + local.upperBound)) }
                keys.withLock { $0.formUnion(outputs.keys) }
            }
        }

        func drain() -> [Range<Int>] {
            ranges.withLock { ranges in
                defer { ranges.removeAll() }
                return ranges
            }
        }

        var outputNames: Set<String> { keys.withLock { $0 } }
    }

    private static let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("FluidAudio", isDirectory: true)

    private func loadClip(_ index: Int) throws -> [Float] {
        let url = Self.appSupport.appendingPathComponent(String(format: "FLEURS/en_us/en_us_%04d.wav", index))
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "FLEURS clip not cached")
        return try AudioConverter().resampleAudioFile(url)
    }

    private func cachedModelDirectory(_ folder: String, requiredFile: String) throws -> URL {
        let dir = Self.appSupport.appendingPathComponent("Models/\(folder)", isDirectory: true)
        let required = dir.appendingPathComponent(requiredFile).path
        try XCTSkipUnless(FileManager.default.fileExists(atPath: required), "\(folder) models not cached")
        return dir
    }

    private func makeBuffer(_ samples: ArraySlice<Float>) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let dst = buffer.floatChannelData![0]
        for (i, sample) in samples.enumerated() { dst[i] = sample }
        return buffer
    }

    private func assertTilesStream(
        _ ranges: [Range<Int>], tokenFrames: [Int], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertFalse(ranges.isEmpty, "callback never fired", file: file, line: line)
        XCTAssertEqual(
            ranges.first?.lowerBound, 0, "first callback frame must be 0 for a fresh stream", file: file, line: line)
        for (previous, next) in zip(ranges, ranges.dropFirst()) {
            XCTAssertEqual(
                next.lowerBound, previous.upperBound, "gap/overlap between callbacks", file: file, line: line)
        }
        let span = (ranges.first?.lowerBound ?? 0)..<(ranges.last?.upperBound ?? 0)
        XCTAssertFalse(tokenFrames.isEmpty, "no tokens decoded", file: file, line: line)
        let outside = tokenFrames.filter { !span.contains($0) }
        XCTAssertTrue(outside.isEmpty, "token frames \(outside) outside callback span \(span)", file: file, line: line)
    }

    func testNemotronCallbackFramesTileStreamAndRestartAfterReset() async throws {
        let dir = try cachedModelDirectory(
            NemotronChunkSize.ms2240.repo.folderName, requiredFile: "encoder/\(NemotronEncoder.fileName)")
        let clips = try [0, 2].map(loadClip)
        let manager = StreamingNemotronAsrManager()
        try await manager.loadModels(from: dir)
        let recorder = Recorder()
        await manager.setEncoderOutputCallback(recorder.callback)
        let chunk = await manager.config.chunkSamples

        for samples in clips {  // stream A, reset(), stream B
            await manager.reset()
            for start in stride(from: 0, to: samples.count, by: chunk) {
                nonisolated(unsafe) let buffer = makeBuffer(samples[start..<min(start + chunk, samples.count)])
                try await manager.appendAudio(buffer)
                try await manager.processBufferedAudio()
            }
            let (_, timings) = try await manager.finishWithTokenTimings()
            let frames = timings.map { Int(($0.startTime / ASRConstants.secondsPerEncoderFrame).rounded()) }
            assertTilesStream(recorder.drain(), tokenFrames: frames)
        }
        XCTAssertTrue(recorder.outputNames.contains("encoded"))
        XCTAssertFalse(
            recorder.outputNames.contains { $0.hasPrefix("cache_") },
            "encoder cache outputs must not reach the callback: \(recorder.outputNames)")
    }

    func testUnifiedCallbackFramesTileStreamAndRestartAfterReset() async throws {
        let config = UnifiedConfig()
        let dir = try cachedModelDirectory(
            config.family.repo.folderName,
            requiredFile: config.family.streamingEncoderFile(precision: .int8, contextSuffix: config.contextSuffix))
        let clips = try [0, 2].map(loadClip)
        let manager = StreamingUnifiedAsrManager(config: config)
        try await manager.loadModels(from: dir)
        let recorder = Recorder()
        await manager.setEncoderOutputCallback(recorder.callback)
        let secondsPerFrame = Double(config.frameSamples) / Double(config.sampleRate)

        for samples in clips {  // stream A, reset(), stream B
            try await manager.reset()
            var timings: [TokenTiming] = []
            for start in stride(from: 0, to: samples.count, by: config.chunkSamples) {
                let end = min(start + config.chunkSamples, samples.count)
                nonisolated(unsafe) let buffer = makeBuffer(samples[start..<end])
                try await manager.appendAudio(buffer)
                try await manager.processBufferedAudio()
                timings += await manager.consumeTokenTimings()
            }
            _ = try await manager.finish()
            timings += await manager.consumeTokenTimings()
            let frames = timings.map { Int(($0.startTime / secondsPerFrame).rounded()) }
            assertTilesStream(recorder.drain(), tokenFrames: frames)
        }
    }
}
