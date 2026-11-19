import Foundation
import XCTest

@testable import FluidAudio

/// tdtJa emits a word-start piece only where a window's decode begins, so the streaming seam's
/// word rules read the whole previous window as its last word (#1118). On a 142 s Japanese file the
/// re-decode retired 9 of 12 windows and the transcript lost 68% of its characters. tdtJa re-decodes
/// the previous window's last second instead.
final class SlidingWindowUnspacedScriptTests: XCTestCase {

    // Window 1 "▁私たちは。" at frames 0–20, window 2 "▁日々食べる" at frames 100–118.
    private let previousPieces = ["▁", "私", "た", "ち", "は", "。", "▁", "日", "々", "食", "べ", "る"]
    private let previousTokens = Array(1...12)
    private let previousTimestamps = [0, 10, 12, 14, 16, 20, 100, 110, 112, 114, 116, 118]

    /// The reproduction: the last word starts at window 2's `▁`, and a next window that opens inside
    /// window 2's span with a different word retires all of window 2, which it cannot re-emit.
    func testWordRulesRetireAWholeUnspacedWindow() {
        XCTAssertEqual(AsrManager.redecodeWordStart(pieces: previousPieces, stableTokenFloor: 0), 6)

        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: previousTokens,
            previousTimestamps: previousTimestamps,
            trailingWordStart: 6,
            currentTokens: [20, 21, 22, 23],
            currentTimestamps: [115, 125, 127, 140],
            currentPieces: ["▁", "植", "物", "。"],
            previousPieces: previousPieces
        )
        XCTAssertEqual(seam.droppedPrevious, 6, "the whole of window 2 is retired")
    }

    /// The tail is the last second (12 frames) before the previous last token: "日々食べる" from 110.
    func testTheUnspacedTailIsTheLastSecond() {
        XCTAssertEqual(AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 0), 7)
        XCTAssertEqual(
            AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 9), 9,
            "never inside the published stable prefix")
        XCTAssertEqual(
            AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 0, minFrame: 113), 9,
            "a tail before the next window's warm-up stays")
        XCTAssertNil(AsrManager.unspacedTailStart(timestamps: [100, 110], stableTokenFloor: 0), "index 0")
        XCTAssertNil(AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 12))
        XCTAssertNil(AsrManager.unspacedTailStart(timestamps: [], stableTokenFloor: 0))
    }

    /// The tail never reaches into an older window than the previous one: with short windows the
    /// last second can start before the previous window's first token.
    func testTheTailStaysInsideThePreviousWindow() {
        XCTAssertEqual(
            AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 0, previousWindowStart: 9),
            9)
        XCTAssertEqual(
            AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 0, previousWindowStart: 6),
            7, "a floor ahead of the tail changes nothing")
        XCTAssertNil(
            AsrManager.unspacedTailStart(timestamps: previousTimestamps, stableTokenFloor: 0, previousWindowStart: 12))
    }

    /// The re-decode emits inside the tail and past it: its reading replaces the tail.
    func testACoveringRedecodeReplacesTheTail() {
        let seam = AsrManager.reconcileUnspacedSeam(
            previousTimestamps: previousTimestamps, tailStart: 7, currentTimestamps: [110, 112, 125, 140])
        XCTAssertEqual(seam.droppedPrevious, 5)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// The re-decode starts after the tail (it heard nothing there): the tail stays.
    func testARedecodeThatSkipsTheTailKeepsIt() {
        let seam = AsrManager.reconcileUnspacedSeam(
            previousTimestamps: previousTimestamps, tailStart: 7, currentTimestamps: [125, 140])
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// The re-decode ends inside the tail (the audio ran out): the tail stays and its copy goes,
    /// including る@120, two frames past the kept る, which dedup leaves as a single token:
    /// "食べる", never "食べるる".
    func testARedecodeThatStopsInTheTailKeepsItOnce() {
        let seam = AsrManager.reconcileUnspacedSeam(
            previousTimestamps: previousTimestamps, tailStart: 7, currentTimestamps: [110, 116, 120])
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 2)
        XCTAssertTrue(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: previousTokens, keptTimestamps: previousTimestamps, token: 12, timestamp: 120))
    }

    /// The re-decode re-emits 全, the last kept token ahead of the replaced tail, and dedup leaves
    /// a single token: "ほぼ全員が", never "ほぼ全全員が".
    func testAReplacedTailDropsTheCopyOfTheKeptTokenAheadOfIt() {
        // ほ@97 ぼ@99 全@104 | 員@106 が@108 理@110 解@112 し@114 ま@116 す@117 。@118
        let previous = [80, 97, 99, 104, 106, 108, 110, 112, 114, 116, 117, 118]
        XCTAssertEqual(AsrManager.unspacedTailStart(timestamps: previous, stableTokenFloor: 0), 4)
        let current = [103, 105, 107, 125, 130]
        let seam = AsrManager.reconcileUnspacedSeam(
            previousTimestamps: previous, tailStart: 4, currentTimestamps: current)
        XCTAssertEqual(seam.droppedPrevious, 8)
        XCTAssertEqual(seam.droppedCurrent, 0)
        XCTAssertTrue(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: [1, 2, 3, 4], keptTimestamps: Array(previous.prefix(4)), token: 4, timestamp: 103))
    }

    /// A copy can only be of the last kept token: the re-decode's い of "いわれ" equals the い of
    /// "多い" two tokens back, and stays ("多いといわれて", never "多いとわれて").
    func testAnEqualCharacterFurtherBackIsNotACopy() {
        // 多@969 い@971 と@972 | い@974 われ …
        XCTAssertFalse(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: [1, 2, 3], keptTimestamps: [969, 971, 972], token: 2, timestamp: 974))
        XCTAssertTrue(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: [1, 2, 3], keptTimestamps: [969, 971, 972], token: 3, timestamp: 973))
    }

    /// A token past the jitter margin, or a different token, is the re-decode's own.
    func testOnlyANearbyCopyOfTheLastKeptTokenGoes() {
        XCTAssertFalse(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: [1, 2, 3], keptTimestamps: [80, 100, 104], token: 3, timestamp: 110))
        XCTAssertFalse(
            AsrManager.isUnspacedJitterCopy(
                keptTokens: [1, 2, 3], keptTimestamps: [80, 100, 104], token: 9, timestamp: 104))
        XCTAssertFalse(AsrManager.isUnspacedJitterCopy(keptTokens: [], keptTimestamps: [], token: 3, timestamp: 104))
    }

    /// Only the tail can still be retired, so the tokens ahead of it are stable.
    func testJapaneseTokensAheadOfTheTailAreStable() {
        XCTAssertEqual(
            AsrManager.stableTokenCount(pieces: previousPieces, timestamps: previousTimestamps, modelVersion: .tdtJa),
            7)
        XCTAssertEqual(
            AsrManager.stableTokenCount(pieces: previousPieces, timestamps: previousTimestamps, modelVersion: .v3),
            6)
        XCTAssertEqual(AsrManager.stableTokenCount(pieces: previousPieces, timestamps: [], modelVersion: .tdtJa), 0)
    }

    /// A Japanese window opens after the cutoff, without its `▁`. That is not a word cut in half, and
    /// the #912 join must not treat it as one.
    func testAJapaneseWindowNeverOpensMidWord() {
        XCTAssertFalse(AsrManager.opensMidWord(pieces: ["植", "物", "。"], modelVersion: .tdtJa))
        XCTAssertTrue(AsrManager.opensMidWord(pieces: ["y", "▁went"], modelVersion: .v3))
        XCTAssertTrue(AsrManager.opensMidWord(pieces: ["<unk>", "y"], modelVersion: nil))
        XCTAssertFalse(AsrManager.opensMidWord(pieces: ["▁went"], modelVersion: .v3))
    }

    /// The retired tail ends the transcript text with no space before it.
    func testTheRetiredTailLeavesTheTextAsASuffix() {
        XCTAssertEqual(
            SlidingWindowAsrManager.removingTrailingWord("理解します。", from: "ほぼ全員が理解します。", unspaced: true),
            "ほぼ全員が")
        XCTAssertNil(SlidingWindowAsrManager.removingTrailingWord("理解します。", from: "ほぼ全員が理解します。"))
        XCTAssertNil(SlidingWindowAsrManager.removingTrailingWord("知っています", from: "ほぼ全員が", unspaced: true))
    }

    /// Window texts of an unspaced model join with no separator: a seam can fall mid-sentence
    /// ("ほぼ全員が" + "理解します。"). Spaced models keep the space.
    func testUnspacedWindowTextsJoinWithoutASpace() {
        XCTAssertEqual(SlidingWindowAsrManager.windowTextSeparator(unspaced: true), "")
        XCTAssertEqual(SlidingWindowAsrManager.windowTextSeparator(unspaced: false), " ")
        XCTAssertEqual(
            SlidingWindowAsrManager.appendingVolatile("ほぼ全員が", "理解します。", separator: ""), "ほぼ全員が理解します。")
        XCTAssertEqual(SlidingWindowAsrManager.appendingVolatile("first", "second"), "first second")
    }

    func testOnlyTheUnspacedModelDropsTheWordRules() {
        XCTAssertFalse(AsrManager.marksWordStarts(.tdtJa))
        XCTAssertTrue(AsrManager.marksWordStarts(.v3))
        XCTAssertTrue(AsrManager.marksWordStarts(.v2))
        XCTAssertTrue(AsrManager.marksWordStarts(nil))
    }
}
