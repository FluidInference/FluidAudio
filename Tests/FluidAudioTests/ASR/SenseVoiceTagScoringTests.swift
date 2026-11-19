import CoreML
import XCTest

@testable import FluidAudio

final class SenseVoiceTagScoringTests: XCTestCase {

    /// A tiny vocabulary with the real tag spellings at made-up ids.
    private let vocabulary: [Int: String] = [
        0: "<unk>", 1: "▁hello",
        2: "<|zh|>", 3: "<|en|>", 4: "<|nospeech|>",
        5: "<|HAPPY|>", 6: "<|SAD|>", 7: "<|NEUTRAL|>",
        8: "<|Speech|>", 9: "<|BGM|>", 10: "<|Laughter|>",
        11: "<|withitn|>", 12: "<|woitn|>", 13: "<|ASR|>", 14: "<|/Speech|>",
    ]

    private func row(_ logits: [Int: Float]) -> [Float] {
        var row = [Float](repeating: -20, count: vocabulary.count)
        for (id, value) in logits { row[id] = value }
        return row
    }

    func testCandidatesAreSortedIntoSlotsByName() {
        let candidates = SenseVoiceTagScoring.Candidates(vocabulary: vocabulary)
        XCTAssertEqual(candidates.language.map(\.tag), ["zh", "en", "nospeech"])
        XCTAssertEqual(candidates.emotion.map(\.tag), ["HAPPY", "SAD", "NEUTRAL"])
        XCTAssertEqual(candidates.event.map(\.tag), ["Speech", "BGM", "Laughter"])
    }

    func testEachSlotIsASoftmaxOverItsOwnTags() throws {
        let candidates = SenseVoiceTagScoring.Candidates(vocabulary: vocabulary)
        let rows = [
            row([2: 5, 3: 1]),
            // The text token outscores every emotion; it must not leak into the emotion slot.
            row([1: 9, 5: 2, 6: 2 + log(3), 7: 0]),
            row([8: 4, 10: 4]),
            row([12: 6]),
        ]
        let scores = SenseVoiceTagScoring.scores(rows: rows, candidates: candidates)

        XCTAssertEqual(scores.language.first?.tag, "zh")
        XCTAssertEqual(scores.emotion.map(\.tag), ["SAD", "HAPPY", "NEUTRAL"])
        let emotion = scores.emotion.map(\.probability)
        XCTAssertEqual(emotion.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertEqual(emotion[0] / emotion[1], 3, accuracy: 1e-3)
        XCTAssertEqual(scores.event[0].probability, 0.5, accuracy: 1e-3)
        XCTAssertEqual(scores.event[1].probability, 0.5, accuracy: 1e-3)
    }

    func testTooFewRowsGiveNoScores() {
        let candidates = SenseVoiceTagScoring.Candidates(vocabulary: vocabulary)
        XCTAssertEqual(SenseVoiceTagScoring.scores(rows: [row([:])], candidates: candidates), .empty)
    }

    func testNonFiniteLogitsAreSkipped() {
        let scores = SenseVoiceTagScoring.score(
            [Float.nan, 1, 1], [(id: 0, tag: "A"), (id: 1, tag: "B"), (id: 2, tag: "C")])
        XCTAssertEqual(scores.map(\.tag).sorted(), ["B", "C"])
        XCTAssertEqual(SenseVoiceTagScoring.score([Float.nan], [(id: 0, tag: "A")]), [])
    }

    /// fp16 logits with a padded row stride come back as unpadded fp32 rows.
    func testLeadingRowsWidenFloat16AndDropPadding() throws {
        let logits = try MLMultiArray(shape: [1, 3, 2], dataType: .float16)
        let values: [[Float]] = [[0.5, -1], [2, 0.25], [8, 9]]
        for t in 0..<3 {
            for v in 0..<2 { logits[[0, t as NSNumber, v as NSNumber]] = NSNumber(value: values[t][v]) }
        }
        XCTAssertEqual(LogitsArgmax.leadingRows(logits: logits, count: 2), [[0.5, -1], [2, 0.25]])
        XCTAssertEqual(LogitsArgmax.leadingRows(logits: logits, count: 9).count, 3)
    }
}
