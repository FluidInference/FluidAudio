import Foundation
import XCTest

@testable import FluidAudio

final class ChatterboxAlignmentAnalyzerTests: XCTestCase {

    private let vocab = 20
    private let eos = 19

    private func makeLogits(_ value: Float = 1.0) -> [Float] {
        [Float](repeating: value, count: vocab)
    }

    /// One attention row over a context of `ctx` positions with all mass at
    /// text position `focus` (absolute index textStart+focus).
    private func row(ctx: Int, textStart: Int, focus: Int) -> [Float] {
        var row = [Float](repeating: 0, count: ctx)
        row[textStart + focus] = 1.0
        return row
    }

    func testEosSuppressedWhileMidText() {
        let textStart = 4
        let span = 10
        var analyzer = ChatterboxAlignmentAnalyzer(
            textStart: textStart, textEnd: textStart + span, eosIndex: eos)
        var logits = makeLogits()
        // Alignment at text position 0 — far from the end.
        analyzer.step(
            logits: &logits,
            alignRows: [
                row(ctx: 16, textStart: textStart, focus: 0),
                row(ctx: 16, textStart: textStart, focus: 0),
            ],
            nextToken: nil)
        XCTAssertEqual(logits[eos], -32768)
        XCTAssertEqual(logits[0], 1.0, "non-EOS logits untouched")
    }

    func testTokenRepetitionForcesEos() {
        let textStart = 0
        let span = 10
        var analyzer = ChatterboxAlignmentAnalyzer(
            textStart: textStart, textEnd: span, eosIndex: eos)
        var logits = makeLogits()
        analyzer.step(
            logits: &logits,
            alignRows: [row(ctx: 12, textStart: 0, focus: 0)],
            nextToken: 5)
        logits = makeLogits()
        analyzer.step(
            logits: &logits,
            alignRows: [row(ctx: 13, textStart: 0, focus: 1)],
            nextToken: 7)
        logits = makeLogits()
        // Third step: last two sampled tokens identical → forced EOS.
        analyzer.step(
            logits: &logits,
            alignRows: [row(ctx: 14, textStart: 0, focus: 2)],
            nextToken: 7)
        XCTAssertEqual(logits[eos], 32768)
        XCTAssertEqual(logits[0], -32768)
    }
}
