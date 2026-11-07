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

    /// The re-decode ends inside the tail (the audio ran out): the tail stays and its copy goes.
    func testARedecodeThatStopsInTheTailKeepsItOnce() {
        let seam = AsrManager.reconcileUnspacedSeam(
            previousTimestamps: previousTimestamps, tailStart: 7, currentTimestamps: [110, 116, 120])
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 2)
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

    func testOnlyTheUnspacedModelDropsTheWordRules() {
        XCTAssertFalse(AsrManager.marksWordStarts(.tdtJa))
        XCTAssertTrue(AsrManager.marksWordStarts(.v3))
        XCTAssertTrue(AsrManager.marksWordStarts(.v2))
        XCTAssertTrue(AsrManager.marksWordStarts(nil))
    }
}
