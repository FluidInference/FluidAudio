import Foundation
import XCTest

@testable import FluidAudio

/// The re-decode at a window seam can re-emit the word *before* the previous last word with a
/// different spelling ("thirty" → "30"). Only same-spelled re-emissions were consumed, so the
/// differently spelled one fell through to the retire rule, which retires only the last word, and
/// both spellings stayed: "twenty-five to thirty 30 years" (#1113). Frames are the issue's token
/// times at 80 ms per frame.
final class SeamRespelledWordTests: XCTestCase {

    // "… to thirty years": " to"@151 " th"@153 "ir"@154 "ty"@156 " years"@158.
    private let previousPieces = [" lag", " to", " th", "ir", "ty", " years"]
    private let previousTokens = [10, 11, 12, 13, 14, 15]
    private let previousTimestamps = [140, 151, 153, 154, 156, 158]
    private let lastWordStart = 5

    // The re-decode: " "@153 "3"@155 "0"@159 " years"@163 "."@170.
    private let currentPieces = [" ", "3", "0", " years", "."]
    private let currentTokens = [20, 21, 22, 15, 23]
    private let currentTimestamps = [153, 155, 159, 163, 170]

    private func reconcile(stableTokenFloor: Int = 0) -> (droppedPrevious: Int, droppedCurrent: Int) {
        AsrManager.reconcileFinalWindowSeam(
            previousTokens: previousTokens,
            previousTimestamps: previousTimestamps,
            trailingWordStart: lastWordStart,
            currentTokens: currentTokens,
            currentTimestamps: currentTimestamps,
            currentPieces: currentPieces,
            previousPieces: previousPieces,
            stableTokenFloor: stableTokenFloor
        )
    }

    /// "30" starts on "thirty"'s frame: the re-decode's reading replaces "thirty years".
    func testARespelledEarlierWordReplacesTheKeptWordsFromThere() {
        let seam = reconcile()
        XCTAssertEqual(seam.droppedPrevious, 4, "thirty years")
        XCTAssertEqual(seam.droppedCurrent, 0, "30 years. stays")
    }

    /// A token subscriber already holds "thirty" as stable: it keeps its spelling, the re-reading
    /// goes, and the re-emitted "years" is the kept last word's copy.
    func testAStableEarlierWordKeepsItsSpelling() {
        let seam = reconcile(stableTokenFloor: lastWordStart)
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 4, "30 years goes, the period stays")
    }

    /// "say" starts nearer "safe" than the last word "for.": "safe for." is replaced
    /// ("the word safe say for you" in the issue's offset sweep).
    func testARespelledWordAheadOfAPunctuatedLastWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4],
            previousTimestamps: [286, 290, 294, 295],
            trailingWordStart: 2,
            currentTokens: [5, 3, 6],
            currentTimestamps: [291, 296, 305],
            currentPieces: [" say", " for", " you"],
            previousPieces: [" word", " safe", " for", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 3)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// "the"@288 starts on "knowledge"'s frame, but the exact re-emission "knowledge" after it
    /// anchors the alignment: "the" re-reads "a" (LibriSpeech 1284-134647, ref "with the knowledge").
    func testAnExactReemissionAnchorsTheRespelledWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4, 5, 6, 7, 8],
            previousTimestamps: [283, 286, 288, 289, 290, 291, 293, 295],
            trailingWordStart: 6,
            currentTokens: [9, 3, 4, 5, 6, 7, 8, 10],
            currentTimestamps: [288, 290, 292, 293, 295, 298, 300, 303],
            currentPieces: [" the", " know", "led", "ge", " of", " tr", "uth", ","],
            previousPieces: [" with", " a", " know", "led", "ge", " of", " tr", "uth"]
        )
        XCTAssertEqual(seam.droppedPrevious, 7, "a knowledge of truth")
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// "word"@289 is one frame from both "the"@288 and "world"@290: the tie goes to the later word,
    /// and "the" stays ("alongside the world word say" in the issue's offset sweep).
    func testATieGoesToTheLaterWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4, 5, 6],
            previousTimestamps: [280, 288, 290, 294, 296, 297],
            trailingWordStart: 3,
            currentTokens: [7, 8, 9, 10, 11],
            currentTimestamps: [289, 291, 295, 301, 304],
            currentPieces: [" wor", "d", " say", " for", " you"],
            previousPieces: [" alongside", " the", " world", " sa", "fe", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 4, "world safe.")
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// The re-decode splits "forever" into "for ever", so counting back from the "with" anchor lands
    /// on "parent", 9 frames away. The nearest word decides instead (LibriSpeech 1221-135766,
    /// ref "her parent for ever with").
    func testASplitWordFallsBackToTheNearestWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            previousTimestamps: [278, 282, 284, 286, 289, 291, 292, 294, 295],
            trailingWordStart: 8,
            currentTokens: [10, 11, 7, 8, 9, 12, 13],
            currentTimestamps: [291, 295, 296, 298, 300, 302, 304],
            currentPieces: [" for", " e", "ver", " with", " the", " r", "ace"],
            previousPieces: [" her", " p", "are", "nt", " f", "ore", "ver", " with", " the"]
        )
        XCTAssertEqual(seam.droppedPrevious, 5, "forever with the")
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// A word that starts nearest the last word is that word's re-reading, not an earlier
    /// word's: `and an` + `analyzing` still retires only `an` (#897).
    func testAWordNearestTheLastWordIsNotAnEarlierWordsRespelling() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [90, 100, 103],
            trailingWordStart: 2,
            currentTokens: [4, 5, 6, 7],
            currentTimestamps: [103, 105, 107, 115],
            currentPieces: [" anal", "y", "zing", " it"],
            previousPieces: [" code", " and", " an"]
        )
        XCTAssertEqual(seam.droppedPrevious, 1)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// A re-decode that ends without reaching past the previous window retires nothing; its
    /// re-reading goes instead of standing next to the kept spelling.
    func testAnEarlyEndingRedecodeKeepsTheKeptSpelling() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: previousTokens,
            previousTimestamps: previousTimestamps,
            trailingWordStart: lastWordStart,
            currentTokens: [20, 21, 22, 15],
            currentTimestamps: [153, 155, 159, 160],
            currentPieces: [" ", "3", "0", " years"],
            previousPieces: previousPieces
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 4)
    }

    func testAnUpdateCarriesTheManagersText() {
        let update = SlidingWindowTranscriptionUpdate(
            text: "30 years.", isConfirmed: true, confidence: 0.9, timestamp: Date(),
            confirmedTranscript: "by twenty-five to 30 years.", volatileTranscript: "")
        XCTAssertEqual(update.confirmedTranscript, "by twenty-five to 30 years.")
        XCTAssertEqual(update.volatileTranscript, "")
    }
}
