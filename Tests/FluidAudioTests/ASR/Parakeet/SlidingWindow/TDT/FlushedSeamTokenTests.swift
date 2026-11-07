import Foundation
import XCTest

@testable import FluidAudio

/// Interior windows now end their decode with the pending-token flush the
/// last window always had (issue #1110), so a window that ends inside the
/// pause after a sentence can emit the sentence-final mark the model was
/// holding for the next speech onset. The flush can also emit a stray
/// piece at a mid-speech seam. These pin the merge rule the change relies
/// on: a flushed token at the window's last frame survives only when the
/// right window has nothing before the left window ends — a pause — and
/// is dropped whenever the right window heard the same stretch. When the
/// right window's first word is clear of the left window's last word, the
/// drop never takes the right window's words with it.
final class FlushedSeamTokenTests: XCTestCase {

    private typealias Token = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    private func createMockAudioSamples(durationSeconds: Double, sampleRate: Int = 16000) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(sampleRate))
        return (0..<sampleCount).map { Float($0) / Float(sampleCount) }
    }

    private let vocabulary: [Int: String] = [
        1: "▁hello",
        2: "▁bright",
        3: "▁world",
        4: "▁and",
        5: "▁more",
        6: "▁here",
        7: "▁so",
        8: "▁this",
        9: "▁um",
        10: "me",
        90: ".",
    ]

    private var safeIds: Set<Int> {
        ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary)!
    }

    private var punctuationIds: Set<Int> {
        ChunkProcessor.punctuationTokenIds(vocabulary: vocabulary)!
    }

    private func decode(_ tokens: [Token]) -> String {
        var text = ""
        for entry in tokens {
            let piece = vocabulary[entry.token] ?? ""
            if isWordBoundary(piece) {
                let word = stripWordBoundaryPrefix(piece)
                text += text.isEmpty ? word : " " + word
            } else {
                text += piece
            }
        }
        return text
    }

    // MARK: - Which windows flush

    func testInteriorWindowsFlushOnTheV3FamilyWhenTheTailIsQuiet() {
        for version in [AsrModelVersion.v3, .ultra, .redux, .tdtJa] {
            XCTAssertTrue(
                ChunkProcessor.flushesPendingTokens(isLastChunk: false, modelVersion: version, quietTail: true),
                "\(version) interior window ending in a pause")
            XCTAssertFalse(
                ChunkProcessor.flushesPendingTokens(isLastChunk: false, modelVersion: version, quietTail: false),
                "\(version) interior window ending mid-speech")
        }
        XCTAssertFalse(ChunkProcessor.flushesPendingTokens(isLastChunk: false, modelVersion: .v2, quietTail: true))
        XCTAssertFalse(
            ChunkProcessor.flushesPendingTokens(isLastChunk: false, modelVersion: .tdtCtc110m, quietTail: true))
        XCTAssertFalse(ChunkProcessor.flushesPendingTokens(isLastChunk: false, modelVersion: nil, quietTail: true))
        // The last window flushes as it always did, whatever the decoder or the tail.
        XCTAssertTrue(ChunkProcessor.flushesPendingTokens(isLastChunk: true, modelVersion: .v2, quietTail: false))
        XCTAssertTrue(ChunkProcessor.flushesPendingTokens(isLastChunk: true, modelVersion: nil, quietTail: false))
    }

    func testQuietTailIsSixSubThresholdFramesAtTheVeryEnd() {
        let frame = ASRConstants.samplesPerEncoderFrame
        let quiet = [Float](repeating: 0.001, count: frame * 20)  // RMS 0.001
        let loud = [Float](repeating: 0.05, count: frame)
        func withSpeech(inFrame index: Int) -> [Float] {
            var samples = quiet
            samples.replaceSubrange((frame * index)..<(frame * (index + 1)), with: loud)
            return samples
        }

        XCTAssertTrue(ChunkProcessor.hasQuietTail(quiet, rmsThreshold: 0.008))
        XCTAssertFalse(
            ChunkProcessor.hasQuietTail(quiet, rmsThreshold: 0.0005),
            "room tone above the gate is not a pause")

        // Speech in the last frame: not a pause, however quiet the frames before it.
        XCTAssertFalse(ChunkProcessor.hasQuietTail(withSpeech(inFrame: 19), rmsThreshold: 0.008))
        // Speech five frames before the end: the quiet run is one frame short.
        XCTAssertFalse(ChunkProcessor.hasQuietTail(withSpeech(inFrame: 14), rmsThreshold: 0.008))
        // One frame earlier and the six-frame run is complete.
        XCTAssertTrue(ChunkProcessor.hasQuietTail(withSpeech(inFrame: 13), rmsThreshold: 0.008))

        XCTAssertFalse(
            ChunkProcessor.hasQuietTail([Float](repeating: 0, count: frame * 3), rmsThreshold: 0.008),
            "shorter than the run it has to measure")
    }

    // MARK: - Merge rule

    /// The window ended inside a pause: the right window's first token comes
    /// after the left window's last. The flushed period is the sentence's
    /// and is kept.
    func testFlushedPeriodBeforeAPauseIsKept() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 6, timestamp: 110, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),  // flushed at the last frame
        ]
        let right: [Token] = [
            (token: 7, timestamp: 130, confidence: 0.97, duration: 1),
            (token: 8, timestamp: 132, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 6, 90, 7, 8])
        XCTAssertEqual(decode(merged), "hello here. so this")
    }

    /// The window ended mid-speech: both windows heard the seam words. The
    /// flushed piece sits after the last matched pair, where the right
    /// window's tail takes over, so it never reaches the transcript.
    func testFlushedPieceAtAMidSpeechSeamIsDropped() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 2, timestamp: 110, confidence: 0.97, duration: 1),
            (token: 3, timestamp: 112, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),  // flushed at the last frame
        ]
        let right: [Token] = [
            (token: 2, timestamp: 110, confidence: 0.96, duration: 1),
            (token: 3, timestamp: 112, confidence: 0.96, duration: 1),
            (token: 4, timestamp: 114, confidence: 0.97, duration: 1),
            (token: 5, timestamp: 116, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 2, 3, 4, 5])
        XCTAssertEqual(decode(merged), "hello bright world and more")
    }

    /// No token matches across the seam but the right window heard
    /// something inside the left window's span: the midpoint split drops
    /// the flushed piece along with the rest of the left tail. "um" starts
    /// within the matcher's tolerance of "here", so it may be the same
    /// audio heard by both windows (their timestamps drift) and the windows
    /// are not joined: joining there doubles a word ("in Is", "frío El frío").
    func testFlushedPieceIsDroppedWhenTheRightWindowHeardTheOverlap() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 6, timestamp: 110, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),  // flushed at the last frame
        ]
        let right: [Token] = [
            (token: 9, timestamp: 118, confidence: 0.80, duration: 1),  // heard in the overlap, not by left
            (token: 7, timestamp: 130, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 6, 7])
        XCTAssertEqual(decode(merged), "hello here so")
    }

    /// The pause ended on the left window's last frame: the right window's
    /// first word starts on the frame the flushed mark was stamped with.
    /// The word must survive whole; a midpoint split drops its first piece
    /// and the splice guard its continuation ("Vakot Vihekhmi" -> "Vakot.").
    func testFlushedMarkOnTheRightWindowsFirstFrameKeepsTheWord() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 6, timestamp: 108, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),  // flushed at the last frame
        ]
        let right: [Token] = [
            (token: 7, timestamp: 124, confidence: 0.95, duration: 1),
            (token: 10, timestamp: 126, confidence: 0.95, duration: 1),
            (token: 8, timestamp: 130, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 6, 7, 10, 8])
        XCTAssertEqual(decode(merged), "hello here some this")
    }

    /// The right window heard the sentence close and emitted its own mark
    /// before the left window's flushed one: the right window's mark and
    /// timing win ("amor" + flushed "." | "." "End" lost both marks).
    func testFlushedMarkGivesWayToTheRightWindowsOwnMark() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 6, timestamp: 105, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),  // flushed at the last frame
        ]
        // The two marks sit 1.44 s apart, past the matcher's time tolerance.
        let right: [Token] = [
            (token: 90, timestamp: 106, confidence: 0.90, duration: 0),
            (token: 7, timestamp: 130, confidence: 0.97, duration: 1),
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 6, 90, 7])
        XCTAssertEqual(merged.map(\.timestamp), [100, 105, 106, 130])
        XCTAssertEqual(decode(merged), "hello here. so")
    }

    /// Two windows in a row ended in the same pause and each flushed the
    /// pending mark: it is kept once, with the left window's timing.
    func testMarkFlushedByBothWindowsIsKeptOnce() {
        let processor = ChunkProcessor(audioSamples: createMockAudioSamples(durationSeconds: 30.0))

        let left: [Token] = [
            (token: 1, timestamp: 100, confidence: 0.98, duration: 1),
            (token: 6, timestamp: 110, confidence: 0.97, duration: 1),
            (token: 90, timestamp: 124, confidence: 0.60, duration: 0),
        ]
        let right: [Token] = [
            (token: 90, timestamp: 140, confidence: 0.60, duration: 0)
        ]

        let merged = processor.mergeTokenWindowsForTesting(
            left: left, right: right, spliceSafeTokenIds: safeIds, punctuationTokenIds: punctuationIds
        )

        XCTAssertEqual(merged.map(\.token), [1, 6, 90])
        XCTAssertEqual(merged.map(\.timestamp), [100, 110, 124])
        XCTAssertEqual(decode(merged), "hello here.")
    }
}
