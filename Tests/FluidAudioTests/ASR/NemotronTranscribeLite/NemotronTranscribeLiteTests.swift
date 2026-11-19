import Foundation
import XCTest

@testable import FluidAudio

final class NemotronTranscribeLiteTests: XCTestCase {

    private typealias Manager = NemotronTranscribeLiteManager
    private typealias Speakers = NemotronTranscribeLiteSpeakers

    // MARK: - Config

    func testWindowSizesMatchTheModels() {
        let c = NemotronTranscribeLiteConfig.self
        XCTAssertEqual(c.windowMelFrames, 4800)  // lite_encoder_w48 mel input
        XCTAssertEqual(c.windowEncoderFrames, 600)  // encoder states / CTC head input
        XCTAssertEqual(c.windowMelFrames / c.stack, c.windowEncoderFrames)
        XCTAssertEqual(c.blankId, 23_862)
    }

    func testRequiredFilesAreTheSwiftArtifacts() {
        let names = ModelNames.NemotronTranscribeLite.self
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .nemotron35TranscribeLite, variant: nil),
            [names.encoderFile, names.ctcHeadFile, names.vocabularyFile, names.melWindowFile, names.melFilterbankFile])
        XCTAssertEqual(Repo.nemotron35TranscribeLite.folderName, "nemotron-3.5-transcribe-lite")
    }

    // MARK: - Windows

    func testWindowsUseCoresWithClippedContext() {
        let w = Manager.windows(length: 9000, core: 4000, context: 400)
        XCTAssertEqual(
            w,
            [
                .init(start: 0, end: 4000, contextStart: 0, contextEnd: 4400),
                .init(start: 4000, end: 8000, contextStart: 3600, contextEnd: 8400),
                .init(start: 8000, end: 9000, contextStart: 7600, contextEnd: 9000),
            ])
    }

    func testEncoderCoresTileTheRecording() {
        // Each window contributes ceil(end/8) - ceil(start/8) frames; together they must cover ceil(valid/8).
        for valid in [1, 7, 8, 3999, 4000, 4001, 297_226] {
            let windows = Manager.windows(length: valid, core: 4000, context: 400)
            let frames = windows.reduce(0) { $0 + Manager.ceilDiv($1.end, 8) - Manager.ceilDiv($1.start, 8) }
            XCTAssertEqual(frames, Manager.ceilDiv(valid, 8), "valid = \(valid)")
        }
    }

    // MARK: - Normalization

    func testPerFeatureNormalizationUsesValidFramesAndUnbiasedStd() {
        // 2 mel bins x 4 frames (frame-major), last frame padding.
        let frameMajor: [Float] = [1, 10, 2, 20, 3, 30, 99, 99]
        let out = Manager.normalizePerFeature(frameMajor: frameMajor, frames: 4, valid: 3, nMels: 2)
        // bin 0: mean 2, unbiased std 1 (+1e-5); bin 1: mean 20, std 10.
        XCTAssertEqual(out[0], -1 / 1.00001, accuracy: 1e-6)
        XCTAssertEqual(out[1], 0, accuracy: 1e-6)
        XCTAssertEqual(out[2], 1 / 1.00001, accuracy: 1e-6)
        XCTAssertEqual(out[3], 0)  // padding frame zeroed
        XCTAssertEqual(out[4], -10 / 10.00001, accuracy: 1e-6)
        XCTAssertEqual(out[7], 0)
    }

    // MARK: - Greedy CTC

    func testGreedyWordsCollapseSkipTagsAndStampFirstPiece() {
        let blank = Int32(NemotronTranscribeLiteConfig.blankId)
        let vocabulary = ["<unk>", "<spk:0>", "\u{2581}he", "llo", "\u{2581}world", "\u{2581}"]
        // he he <blank> llo <spk:0> world world ▁ llo
        let ids: [Int32] = [2, 2, blank, 3, 1, 4, 4, 5, 3]
        let words = Manager.greedyWords(tokenIds: ids, vocabulary: vocabulary)
        XCTAssertEqual(words.map(\.text), ["hello", "world", "llo"])
        XCTAssertEqual(words.map(\.start), [0.0, 5 * 0.04, 7 * 0.04])
    }

    // MARK: - Speakers

    func testDownsampleAveragesPartialLastGroup() {
        // 10 frames x 1 speaker -> groups of 8 and 2.
        let probs: [Float] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        XCTAssertEqual(Speakers.downsample(probs, frames: 10, speakers: 1, factor: 8), [0.5, 0.5])
    }

    func testAlignRepeatsLastFrameOrTruncates() {
        XCTAssertEqual(Speakers.align([1, 2, 3, 4], speakers: 2, to: 3), [1, 2, 3, 4, 3, 4])
        XCTAssertEqual(Speakers.align([1, 2, 3, 4], speakers: 2, to: 1), [1, 2])
    }

    func testShortSpeakersAreDropped() {
        // speaker 0 active 3 frames, speaker 1 active 1 frame; min 2 frames.
        let probs: [Float] = [0.9, 0.0, 0.9, 0.0, 0.9, 0.8]
        XCTAssertEqual(
            Speakers.dropShortSpeakers(probs, speakers: 2, threshold: 0.5, minFrames: 2),
            [0.9, 0.0, 0.9, 0.0, 0.9, 0.0])
    }

    func testAssignPicksMostLikelyActiveSpeakerNearbyElsePrevious() {
        // 5 frames x 2 speakers at 0.08 s: frame 0 has speaker 1 active, frames 1-3 are silent,
        // frame 4 has speaker 0 (0.6) and speaker 1 (0.9) active.
        let probs: [Float] = [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0.6, 0.9]
        func assign(_ starts: [Double], radius: Int) -> [Int] {
            Speakers.assign(
                starts: starts, probs: probs, speakers: 2, threshold: 0.5, searchFrames: radius, frameSeconds: 0.08)
        }
        XCTAssertEqual(assign([0.0], radius: 0), [1])
        // The most likely active speaker wins.
        XCTAssertEqual(assign([0.32], radius: 0), [1])
        // A word at frame 2 searching 2 frames finds frame 0 (earlier frames are checked first).
        XCTAssertEqual(assign([0.16], radius: 2), [1])
        // Searching 1 frame finds nothing: keep the previous word's speaker, or slot 0 for the first word.
        XCTAssertEqual(assign([0.0, 0.16], radius: 1), [1, 1])
        XCTAssertEqual(assign([0.16], radius: 1), [0])
    }

    func testTurnsGroupConsecutiveWordsBySpeaker() {
        let words = [
            NemotronTranscribeLiteWord(text: "a", start: 0, speaker: 0),
            NemotronTranscribeLiteWord(text: "b", start: 1, speaker: 0),
            NemotronTranscribeLiteWord(text: "c", start: 2, speaker: 2),
            NemotronTranscribeLiteWord(text: "d", start: 3, speaker: 0),
        ]
        let result = NemotronTranscribeLiteResult(
            words: words, durationSeconds: 4,
            timings: .init(encoderSeconds: 0, ctcSeconds: 0, diarizerSeconds: nil, totalSeconds: 1))
        XCTAssertEqual(result.turns.map(\.text), ["a b", "c", "d"])
        XCTAssertEqual(result.turns.map(\.speaker), [0, 2, 0])
        XCTAssertEqual(result.rtfx, 4)
    }
}
