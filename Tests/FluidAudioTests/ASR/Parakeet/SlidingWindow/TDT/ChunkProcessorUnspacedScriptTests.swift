import Foundation
import XCTest

@testable import FluidAudio

/// Japanese has no word-initial pieces, so the seam merger's word guard (issue #683) must stay off for `tdtJa`:
/// with it on, a seam inside "愛をつかいます。" kept "愛をつ。" and dropped "かいます".
final class ChunkProcessorUnspacedScriptTests: XCTestCase {

    private typealias Token = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    // 1 愛  2 を  3 つ  4 か  5 います  6 。  7 ▁Hello  8 ▁world
    private let vocabulary: [Int: String] = [
        1: "愛", 2: "を", 3: "つ", 4: "か", 5: "います", 6: "。", 7: "▁Hello", 8: "▁world",
    ]

    private func token(_ id: Int, _ frame: Int) -> Token {
        (token: id, timestamp: frame, confidence: 0.9, duration: 1)
    }

    private func text(_ tokens: [Token]) -> String {
        tokens.map { vocabulary[$0.token] ?? "?" }.joined()
    }

    private func processor() -> ChunkProcessor {
        ChunkProcessor(audioSamples: (0..<(16000 * 20)).map { Float($0) / Float(16000 * 20) })
    }

    private func merged(
        left: [Token], right: [Token], modelVersion: AsrModelVersion?
    ) -> String {
        let safe = ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary, modelVersion: modelVersion)
        return text(
            processor().mergeTokenWindowsForTesting(left: left, right: right, spliceSafeTokenIds: safe))
    }

    /// Left window ends after "愛をつ"; right window starts at "を" and carries the rest.
    func testJapaneseSeamKeepsEveryToken() {
        let left = [token(1, 100), token(2, 101), token(3, 102)]
        let right = [token(2, 101), token(3, 102), token(4, 103), token(5, 104), token(6, 105)]
        XCTAssertEqual(merged(left: left, right: right, modelVersion: .tdtJa), "愛をつかいます。")
    }

    /// The reproduction: the same streams through the guarded path lose "かいます", which is why the
    /// guard is switched off by model.
    func testTheWordGuardStillDropsTokensForAnUnspacedStream() {
        let left = [token(1, 100), token(2, 101), token(3, 102)]
        let right = [token(2, 101), token(3, 102), token(4, 103), token(5, 104), token(6, 105)]
        XCTAssertEqual(merged(left: left, right: right, modelVersion: .v3), "愛をつ。")
    }

    func testOtherModelsKeepTheirGuard() {
        XCTAssertEqual(
            ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary, modelVersion: .v3),
            ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary))
        XCTAssertEqual(
            ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary, modelVersion: nil),
            ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary))
        XCTAssertNil(ChunkProcessor.spliceSafeTokenIds(vocabulary: vocabulary, modelVersion: .tdtJa))
    }

    /// Overlapping windows that share no token fall to the midpoint cut. With the guard on it drops the right
    /// window up to the next punctuation mark ("愛を。"); `つ` is lost to the time cut either way.
    func testMidpointMergeKeepsTheRightWindow() {
        let left = [token(1, 100), token(2, 101)]
        let right = [token(3, 101), token(4, 102), token(5, 103), token(6, 104)]
        XCTAssertEqual(merged(left: left, right: right, modelVersion: .tdtJa), "愛をかいます。")
        XCTAssertEqual(merged(left: left, right: right, modelVersion: .v3), "愛を。")
    }

    /// The repair pass takes the same set: nil keeps in-gap tokens that do not start a "word".
    func testSeamGapRepairKeepsInGapJapaneseTokens() {
        let lead = token(1, 100)
        let tail = token(6, 200)
        let tokens: [Token] = [token(2, 120), token(3, 121), token(4, 122), token(5, 123)]
        let candidate = ChunkProcessor.spliceCandidate(
            windowTokens: tokens.map { $0.token },
            windowTimestamps: tokens.map { $0.timestamp },
            windowConfidences: tokens.map { _ in 0.9 },
            windowDurations: tokens.map { _ in 1 },
            gapStartFrame: 110,
            gapEndFrame: 190,
            leadNeighbor: lead,
            tailNeighbor: tail,
            spliceSafeTokenIds: ChunkProcessor.spliceSafeTokenIds(
                vocabulary: vocabulary, modelVersion: .tdtJa),
            vocabulary: vocabulary)
        XCTAssertEqual(text(candidate), "をつかいます")
    }
}
