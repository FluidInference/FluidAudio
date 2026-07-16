import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for end-aligning the final window (issue #747): a short final
/// chunk zero-padded to the model window is the degenerate input that decodes
/// to all-blank on quiet audio. `lastChunkWarmupSamples` instead fills the
/// window backwards with real audio, decoded as a suppressed warmup prefix so
/// emitted coverage is unchanged.
final class EndAlignedFinalWindowTests: XCTestCase {

    private let frameSamples = ASRConstants.samplesPerEncoderFrame  // 1280
    private let sampleRate = ASRConstants.sampleRate  // 16000

    /// Usable window size in the default mel-context layout.
    private var chunkSamples: Int {
        ChunkProcessor(audioSamples: [Float](repeating: 0, count: sampleRate))
            .chunkLayoutForTesting(melChunkContext: true, modelVersion: .v3)
            .chunkSamples
    }

    // MARK: - lastChunkWarmupSamples

    func testShortFinalChunkFillsTheWindowWithRealAudio() {
        // The #747 reproducer geometry: 17.28s file, second chunk starts at
        // one stride and leaves a ~4.4s tail that used to be ~70% zeros.
        let totalSamples = Int(17.28 * Double(sampleRate))
        let chunkStart = 206_080  // stride for the mel-context layout
        let remaining = totalSamples - chunkStart

        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: chunkStart,
            defaultWarmupSamples: 0,
            chunkSamples: chunkSamples,
            totalSamples: totalSamples
        )

        // Window is full up to frame rounding, prefix never crosses sample 0.
        XCTAssertEqual(warmup % frameSamples, 0)
        XCTAssertGreaterThanOrEqual(chunkStart - warmup, 0)
        XCTAssertGreaterThan(warmup, 0)
        XCTAssertGreaterThanOrEqual(warmup + remaining, chunkSamples - frameSamples)
        XCTAssertLessThanOrEqual(warmup + remaining, chunkSamples + frameSamples)
    }

    func testFullFinalWindowIsUnchanged() {
        // Remaining audio already fills the window: nothing to backfill.
        let chunkStart = chunkSamples
        let totalSamples = chunkStart + chunkSamples

        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: chunkStart,
            defaultWarmupSamples: 0,
            chunkSamples: chunkSamples,
            totalSamples: totalSamples
        )

        XCTAssertEqual(warmup, 0)
    }

    func testSingleChunkFileIsUnchanged() {
        // Nothing precedes chunk 0 — a sub-window file stays the whole-file
        // decode (padding there is unavoidable).
        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: 0,
            defaultWarmupSamples: 0,
            chunkSamples: chunkSamples,
            totalSamples: chunkSamples / 2
        )

        XCTAssertEqual(warmup, 0)
    }

    func testFillIsCappedByPrecedingAudio() {
        // File barely longer than one window: the backfill wants most of a
        // window but only `chunkStart` samples exist before the chunk.
        let chunkStart = 4 * frameSamples
        let totalSamples = chunkStart + frameSamples

        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: chunkStart,
            defaultWarmupSamples: 0,
            chunkSamples: chunkSamples,
            totalSamples: totalSamples
        )

        XCTAssertEqual(warmup, chunkStart)
    }

    func testKeepsLargerDefaultWarmup() {
        // A path-B style warmup larger than the backfill is never shrunk.
        let chunkStart = chunkSamples
        let totalSamples = chunkStart + chunkSamples - frameSamples
        let defaultWarmup = 2 * frameSamples

        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: chunkStart,
            defaultWarmupSamples: defaultWarmup,
            chunkSamples: chunkSamples,
            totalSamples: totalSamples
        )

        XCTAssertEqual(warmup, defaultWarmup)
    }

    func testUnalignedChunkStartStaysFrameAlignedAndInBounds() {
        let chunkStart = 5 * frameSamples + 137  // silence-aligned starts may drift
        let totalSamples = chunkStart + 2 * frameSamples

        let warmup = ChunkProcessor.lastChunkWarmupSamples(
            chunkStart: chunkStart,
            defaultWarmupSamples: 0,
            chunkSamples: chunkSamples,
            totalSamples: totalSamples
        )

        XCTAssertEqual(warmup % frameSamples, 0)
        XCTAssertGreaterThanOrEqual(chunkStart - warmup, 0)
    }

    // MARK: - Adaptive speech gate (still feeds seam-gap repair)

    func testAdaptiveThresholdScalesToQuietRecording() throws {
        // Quiet file at #747 reproducer levels: threshold drops below the
        // ceiling so the seam-gap pass can still see quiet speech.
        var samples = [Float](repeating: 0.0, count: sampleRate * 4)
        for i in 0..<samples.count {
            samples[i] = 0.004 * sin(Float(i) * 0.1)
        }
        let processor = ChunkProcessor(audioSamples: samples)
        let threshold = try processor.adaptiveSpeechRmsThresholdForTesting()
        XCTAssertLessThan(threshold, ChunkProcessor.speechRmsCeiling)
        XCTAssertGreaterThanOrEqual(threshold, ChunkProcessor.speechRmsFloor)
    }
}
