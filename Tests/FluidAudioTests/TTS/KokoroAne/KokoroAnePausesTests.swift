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
        let spans = KokoroAnePauses.spans(phonemes: "a, b.", vocab: vocab, sources: nil, pauses: [",": 8])
        XCTAssertEqual(spans, [.init(tokens: 2..<4, frames: 8, scalars: 1..<3)])
    }

    func testSourceMarkWinsAndFallsBackToTheModelMark() {
        // `、` and `，` both reach the vocab as `,`.
        let sources: [Int: Character] = [1: "、", 4: "，"]
        let spans = KokoroAnePauses.spans(
            phonemes: "a, b, a", vocab: vocab, sources: sources, pauses: ["、": 12, ",": 6])
        XCTAssertEqual(
            spans,
            [.init(tokens: 2..<4, frames: 12, scalars: 1..<3), .init(tokens: 5..<7, frames: 6, scalars: 4..<6)])
        let dunOnly = KokoroAnePauses.spans(
            phonemes: "a, b, a", vocab: vocab, sources: sources, pauses: ["、": 12])
        XCTAssertEqual(dunOnly, [.init(tokens: 2..<4, frames: 12, scalars: 1..<3)])
    }

    func testMarksTheFrontendAddedAreNotKeys() {
        // Mandarin `/` between words has no source; a typed `／` does.
        let vocab = KokoroAneVocab(map: ["a": 43, "b": 44, ",": 3, "/": 7, " ": 16])
        let generated = KokoroAnePauses.spans(
            phonemes: "a/b, a", vocab: vocab, sources: [3: ","], pauses: ["/": 9, ",": 4])
        XCTAssertEqual(generated, [.init(tokens: 4..<6, frames: 4, scalars: 3..<5)])
        let typed = KokoroAnePauses.spans(
            phonemes: "a/b", vocab: vocab, sources: [1: "／"], pauses: ["/": 9])
        XCTAssertEqual(typed, [.init(tokens: 2..<3, frames: 9, scalars: 1..<2)])
    }

    func testConsecutiveMarksShareTheLongestPause() {
        let spans = KokoroAnePauses.spans(
            phonemes: "a?! b", vocab: vocab, sources: nil, pauses: ["?": 10, "!": 6])
        XCTAssertEqual(spans, [.init(tokens: 2..<5, frames: 10, scalars: 1..<4)])
    }

    func testScalarsOutsideTheVocabDoNotShiftTokens() {
        // ❓ is dropped by the encoder, so `,` is still token 2.
        let spans = KokoroAnePauses.spans(phonemes: "a❓,b", vocab: vocab, sources: nil, pauses: [",": 4])
        XCTAssertEqual(spans, [.init(tokens: 2..<3, frames: 4, scalars: 2..<3)])
    }

    func testNoPausesNoSpans() {
        XCTAssertEqual(KokoroAnePauses.spans(phonemes: "a, b.", vocab: vocab, sources: nil, pauses: [:]), [])
    }

    // MARK: - Splitting

    func testSplitCutsAfterThePauseNearestTheMiddle() throws {
        let phonemes = "aa, bb, aa, bb"
        let spans = KokoroAnePauses.spans(phonemes: phonemes, vocab: vocab, sources: nil, pauses: [",": 4])
        let pieces = try XCTUnwrap(
            KokoroAnePauses.split(phonemes, atPauseNearestMiddle: spans, sources: [2: "、", 6: "、", 10: "，"]))
        XCTAssertEqual(pieces.map(\.phonemes), ["aa, bb, ", "aa, bb"])
        XCTAssertEqual(pieces.map(\.sources), [[2: "、", 6: "、"], [2: "，"]])
        // No pause inside the string: nothing to cut at.
        XCTAssertNil(KokoroAnePauses.split("aa bb", atPauseNearestMiddle: [], sources: nil))
    }

    func testChunksRebaseSourcesPastTheCap() throws {
        // 120 × "ㄋㄧ2, " is 600 scalars: two chunks, each comma still sourced.
        let phonemes = String(repeating: "ㄋㄧ2, ", count: 120)
        let sources = Dictionary(uniqueKeysWithValues: (0..<120).map { ($0 * 5 + 3, Character("、")) })
        let chunks = KokoroAneManager.chunks(of: phonemes, sources: sources)
        XCTAssertEqual(chunks.count, 2)
        var total = 0
        for chunk in chunks {
            let scalars = Array(chunk.phonemes.unicodeScalars)
            let chunkSources = try XCTUnwrap(chunk.sources)
            XCTAssertEqual(chunkSources.count, scalars.filter { $0 == "," }.count)
            for (offset, mark) in chunkSources {
                XCTAssertEqual(scalars[offset], ",")
                XCTAssertEqual(mark, "、")
            }
            total += chunkSources.count
        }
        XCTAssertEqual(total, 120)
        XCTAssertNil(KokoroAneManager.chunks(of: phonemes, sources: nil).first?.sources)
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
        // Input marks map to themselves; the `/` between words has no entry.
        let ascii = try await MandarinG2P(dict: dict).phonemizeTracingPunctuation("你好世界,你好")
        XCTAssertEqual(ascii.phonemes, "ㄋㄧ2ㄏㄠ3/ㄕ十4ㄐㄝ4,ㄋㄧ2ㄏㄠ3")
        XCTAssertEqual(ascii.punctuationSources, [13: ","])
    }

    // MARK: - End to end

    /// The shipped Mandarin chain, when it is cached.
    private func mandarinManager() throws -> KokoroAneManager {
        let repoDirectory = try KokoroAneResourceDownloader.repositoryDirectory(variant: .mandarin)
        let cached = ModelNames.KokoroAne.requiredModelsZh.allSatisfy {
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent($0).path)
        }
        try XCTSkipUnless(
            cached || ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_TTS_E2E"] == "1",
            "Mandarin bundle not cached; set FLUIDAUDIO_RUN_TTS_E2E=1")
        return KokoroAneManager(variant: .mandarin)
    }

    func testPausesPastTheFrameCapRenderInParts() async throws {
        // Five 10 s pauses (2000 frames) exceed one pass with the speech.
        let manager = try mandarinManager()
        await manager.setPunctuationPauses(["、": .seconds(10)])
        let result = try await manager.synthesizeDetailed(text: "苹果、香蕉、橙子、葡萄、西瓜、桃子。")
        XCTAssertGreaterThan(result.acousticFrames, KokoroAneConstants.maxAcousticFrames)
        XCTAssertEqual(result.samples.count, result.acousticFrames * 600)
        var pauseFrames: Int32 = 0
        for (index, id) in result.inputIds.enumerated() where id == 3 {
            pauseFrames += result.predictedDurations[index] + result.predictedDurations[index + 1]
        }
        XCTAssertEqual(pauseFrames, 5 * 400)
    }

    func testMandarinPauseIsRenderedAsSilence() async throws {
        let manager = try mandarinManager()
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
