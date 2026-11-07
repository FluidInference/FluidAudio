#if TTS
import Foundation
import XCTest

@testable import FluidAudio

final class KokoroAnePausesTests: XCTestCase {

    private let vocab = KokoroAneVocab(map: ["a": 43, "b": 44, ",": 3, ".": 4, "?": 6, "!": 5, " ": 16])

    // MARK: - Frames

    func testFramesRoundToTwentyFiveMilliseconds() {
        XCTAssertEqual(KokoroAnePauses.frames(for: .milliseconds(400)), 16)
        XCTAssertEqual(KokoroAnePauses.frames(for: .milliseconds(38)), 2)
        XCTAssertEqual(KokoroAnePauses.frames(for: .milliseconds(37)), 1)
        XCTAssertEqual(KokoroAnePauses.frames(for: .seconds(1)), 40)
    }

    func testFramesAreClamped() {
        XCTAssertEqual(KokoroAnePauses.frames(for: .zero), 1)
        XCTAssertEqual(KokoroAnePauses.frames(for: .milliseconds(-100)), 1)
        XCTAssertEqual(
            KokoroAnePauses.frames(for: .seconds(600)), Int32(KokoroAneConstants.maxAcousticFrames))
    }

    // MARK: - Spans

    func testSpanCoversMarkAndTrailingSpace() {
        // Tokens: BOS 0, a 1, `,` 2, ` ` 3, b 4, `.` 5.
        let spans = KokoroAnePauses.spans(phonemes: "a, b.", vocab: vocab, sources: [:], pauses: [",": 8])
        XCTAssertEqual(spans, [.init(tokens: 2..<4, frames: 8)])
    }

    func testSourceMarkWinsAndFallsBackToTheModelMark() {
        // `、` and `，` both reach the vocab as `,`.
        let sources: [Int: Character] = [1: "、", 4: "，"]
        let spans = KokoroAnePauses.spans(
            phonemes: "a, b, a", vocab: vocab, sources: sources, pauses: ["、": 12, ",": 6])
        XCTAssertEqual(spans, [.init(tokens: 2..<4, frames: 12), .init(tokens: 5..<7, frames: 6)])
        let dunOnly = KokoroAnePauses.spans(
            phonemes: "a, b, a", vocab: vocab, sources: sources, pauses: ["、": 12])
        XCTAssertEqual(dunOnly, [.init(tokens: 2..<4, frames: 12)])
    }

    func testConsecutiveMarksShareTheLongestPause() {
        let spans = KokoroAnePauses.spans(
            phonemes: "a?! b", vocab: vocab, sources: [:], pauses: ["?": 10, "!": 6])
        XCTAssertEqual(spans, [.init(tokens: 2..<5, frames: 10)])
    }

    func testScalarsOutsideTheVocabDoNotShiftTokens() {
        // ❓ is dropped by the encoder, so `,` is still token 2.
        let spans = KokoroAnePauses.spans(phonemes: "a❓,b", vocab: vocab, sources: [:], pauses: [",": 4])
        XCTAssertEqual(spans, [.init(tokens: 2..<3, frames: 4)])
    }

    func testNoPausesNoSpans() {
        XCTAssertEqual(KokoroAnePauses.spans(phonemes: "a, b.", vocab: vocab, sources: [:], pauses: [:]), [])
    }

    // MARK: - Durations

    func testLengthenedSpanReturnsItsInteriorForSilence() {
        var durations: [Int32] = [2, 3, 2, 1, 4, 3, 2]
        let silent = KokoroAnePauses.apply([.init(tokens: 2..<4, frames: 8)], to: &durations)
        XCTAssertEqual(durations, [2, 3, 7, 1, 4, 3, 2])
        // The span starts at frame 5 and ends at frame 13; one frame kept each side.
        XCTAssertEqual(silent, [6..<12])
    }

    func testShorterPauseTrimsWithoutSilence() {
        var durations: [Int32] = [2, 3, 4, 2, 3]
        let silent = KokoroAnePauses.apply([.init(tokens: 2..<4, frames: 3)], to: &durations)
        XCTAssertEqual(durations, [2, 3, 1, 2, 3])
        XCTAssertEqual(silent, [])
    }

    func testEveryTokenKeepsOneFrame() {
        var durations: [Int32] = [2, 3, 4, 2, 3]
        _ = KokoroAnePauses.apply([.init(tokens: 2..<4, frames: 1)], to: &durations)
        XCTAssertEqual(durations, [2, 3, 1, 1, 3])
    }

    func testOutOfRangeSpanIsIgnored() {
        var durations: [Int32] = [2, 3]
        XCTAssertEqual(KokoroAnePauses.apply([.init(tokens: 1..<4, frames: 9)], to: &durations), [])
        XCTAssertEqual(durations, [2, 3])
    }

    // MARK: - Silence

    func testSilenceZeroesF0AndSetsEdgeEnergy() throws {
        // Two values per frame; BOS and EOS frames carry the silent energy level.
        let durations: [Int32] = [1, 3, 1]
        var f0: [Float] = [0, 0, 200, 210, 220, 230, 240, 250, 0, 0]
        var n: [Float] = [-9, -9, 3, 4, 5, 4, 3, 2, -10, -10]
        try KokoroAnePauses.silence(f0: &f0, n: &n, frames: [2..<3], durations: durations)
        XCTAssertEqual(f0, [0, 0, 200, 210, 0, 0, 240, 250, 0, 0])
        XCTAssertEqual(n, [-9, -9, 3, 4, -9, -9, 3, 2, -10, -10])
    }

    func testSilenceRejectsCurvesOffTheFrameGrid() {
        var f0: [Float] = [0, 0, 0]
        var n: [Float] = [0, 0, 0]
        XCTAssertThrowsError(try KokoroAnePauses.silence(f0: &f0, n: &n, frames: [0..<1], durations: [1, 1]))
    }

    // MARK: - Chunking

    func testChunkRangesMatchChunks() {
        let text = "aa bb, cc dd. ee ff gg"
        let characters = Array(text)
        let ranges = PhonemeChunker.chunkRanges(characters, maxLength: 8)
        XCTAssertEqual(ranges.map { String(characters[$0]) }, PhonemeChunker.chunk(text, maxLength: 8))
        XCTAssertEqual(PhonemeChunker.chunkRanges(Array("  ab "), maxLength: 8), [2..<4])
    }

    // MARK: - Mandarin provenance

    func testMandarinG2PTracesRewrittenMarks() async throws {
        let dict = MandarinPinyinDict(
            phrases: ["你好": ["nǐ", "hǎo"], "世界": ["shì", "jiè"]], singles: [:])
        let traced = try await MandarinG2P(dict: dict).phonemizeTracingPunctuation("你好、世界，你好")
        XCTAssertEqual(traced.phonemes, "ㄋㄧ2ㄏㄠ3, ㄕ十4ㄐㄝ4, ㄋㄧ2ㄏㄠ3")
        XCTAssertEqual(traced.punctuationSources, [6: "、", 14: "，"])
    }

    // MARK: - End to end

    /// Runs the shipped Mandarin chain when it is cached.
    func testMandarinPauseIsRenderedAsSilence() async throws {
        let repoDirectory = try KokoroAneResourceDownloader.repositoryDirectory(variant: .mandarin)
        let cached = ModelNames.KokoroAne.requiredModelsZh.allSatisfy {
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent($0).path)
        }
        try XCTSkipUnless(
            cached || ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_TTS_E2E"] == "1",
            "Mandarin bundle not cached; set FLUIDAUDIO_RUN_TTS_E2E=1")
        let manager = KokoroAneManager(variant: .mandarin)
        let text = "苹果、香蕉，橙子。"
        let natural = try await manager.synthesizeDetailed(text: text)
        await manager.setPunctuationPauses(["、": .milliseconds(400)])
        let paused = try await manager.synthesizeDetailed(text: text)

        // The first `,` cluster (from 、) totals 16 frames; the 、's tokens are `,` then ` `.
        let comma = try XCTUnwrap(paused.inputIds.firstIndex(of: 3))
        XCTAssertEqual(paused.predictedDurations[comma] + paused.predictedDurations[comma + 1], 16)
        // The second comma (from ，) keeps the predicted pause.
        let secondComma = try XCTUnwrap(paused.inputIds[(comma + 1)...].firstIndex(of: 3))
        XCTAssertEqual(paused.predictedDurations[secondComma], natural.predictedDurations[secondComma])

        // Silence from the second frame of the 16; the vocoder starts the next
        // syllable's onset in the last two.
        let start = (Int(paused.predictedDurations[..<comma].reduce(0, +)) + 1) * 600
        let interior = paused.samples[start..<(start + 13 * 600)]
        XCTAssertLessThan(interior.map { abs($0) }.max() ?? 1, 0.001)
    }
}
#endif
