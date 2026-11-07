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

    /// "thirty" is the first accumulated word (speech starts at the end of window 1). Retiring from
    /// index 0 would drop the whole history, which the manager refuses, and both readings would stay:
    /// the kept spelling stays instead and its re-reading goes.
    func testARespelledFirstAccumulatedWordKeepsItsSpelling() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [12, 13, 14, 15],
            previousTimestamps: [153, 154, 156, 158],
            trailingWordStart: 3,
            currentTokens: currentTokens,
            currentTimestamps: currentTimestamps,
            currentPieces: currentPieces,
            previousPieces: [" th", "ir", "ty", " years"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 4, "thirty years.")
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

    /// Two kept words on one frame (duration-0 steps): the tie goes to the later word by position.
    func testATieOnOneFrameGoesToTheLaterPosition() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4],
            previousTimestamps: [190, 200, 200, 206],
            trailingWordStart: 3,
            currentTokens: [5, 6, 7],
            currentTimestamps: [201, 206, 215],
            currentPieces: [" oft", " cake", " is"],
            previousPieces: [" so", " of", " off", " ca"]
        )
        XCTAssertEqual(seam.droppedPrevious, 2, "off ca; of stays")
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

    /// The last word's own copy landing a couple of frames early, nearer the word before it, is
    /// still the last word's: "old and code" + "code"@101 keeps "and".
    func testTheLastWordsEarlyCopyIsNotAnEarlierWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [90, 100, 103],
            trailingWordStart: 2,
            currentTokens: [3, 4, 5],
            currentTimestamps: [101, 112, 120],
            currentPieces: [" code", " is", " new"],
            previousPieces: [" old", " and", " code"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 1, "the copy of code goes")
    }

    /// The same for a fragment's completion landing early: `and an` + `analyzing`@101 retires only
    /// `an` (#897).
    func testAnEarlyFragmentCompletionRetiresOnlyTheFragment() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [90, 100, 103],
            trailingWordStart: 2,
            currentTokens: [4, 5, 6, 7],
            currentTimestamps: [101, 105, 107, 115],
            currentPieces: [" anal", "y", "zing", " it"],
            previousPieces: [" code", " and", " an"]
        )
        XCTAssertEqual(seam.droppedPrevious, 1)
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

    /// "to"@155 re-emits a kept word; the next word "a" can't then re-read that word or an earlier
    /// one: "go to the a store now", as before, never "go a store now".
    func testARespelledTargetNeverPrecedesAMatchedKeptWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4, 5],
            previousTimestamps: [140, 156, 158, 160, 162],
            trailingWordStart: 3,
            currentTokens: [2, 6, 4, 7, 8],
            currentTimestamps: [155, 156, 160, 163, 175],
            currentPieces: [" to", " a", " st", "ore", " now"],
            previousPieces: [" go", " to", " the", " st", "o"]
        )
        XCTAssertEqual(seam.droppedPrevious, 2, "only the fragment sto")
        XCTAssertEqual(seam.droppedCurrent, 1, "the copy of to")
    }

    /// A word the re-decode inserts ahead of the last word starts later than a re-reading would:
    /// "the hedge fence" + "a fence." keeps "hedge" ("hedge a Offence" in the A/B).
    func testAnInsertedWordIsNotARespelling() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [100, 103, 107],
            trailingWordStart: 2,
            currentTokens: [4, 3, 5],
            currentTimestamps: [106, 108, 118],
            currentPieces: [" a", " fence", "."],
            previousPieces: [" the", " hedge", " fence"]
        )
        XCTAssertEqual(seam.droppedPrevious, 1, "only the last word")
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// The same with a token subscriber: the recovered "thirty" stays ("lag to thirty years.").
    func testAnInsertedWordStaysUnderAStableFloor() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [140, 151, 158],
            trailingWordStart: 2,
            currentTokens: [4, 3, 5],
            currentTimestamps: [154, 159, 170],
            currentPieces: [" thirty", " years", "."],
            previousPieces: [" lag", " to", " years"],
            stableTokenFloor: 2
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

    func testSpellingsAgree() {
        XCTAssertTrue(AsrManager.spellingsAgree("word", "world"))
        XCTAssertTrue(AsrManager.spellingsAgree("say", "safe"))
        XCTAssertTrue(AsrManager.spellingsAgree("for", "forever"), "a split")
        XCTAssertFalse(AsrManager.spellingsAgree("the", "a"))
        XCTAssertFalse(AsrManager.spellingsAgree("thirty", "to"))
        XCTAssertFalse(AsrManager.spellingsAgree("a", "and"), "a one-letter prefix is no evidence")
        XCTAssertFalse(AsrManager.spellingsAgree("offence", "a"))
        XCTAssertFalse(AsrManager.spellingsAgree("", "a"))
    }

    /// A retired span of several words leaves the text whether it holds the raw words or their
    /// vocabulary replacements (#1113 with boosting).
    func testARetiredSpanIsRenderedWithItsReplacements() {
        func replacement(_ original: String, _ term: String) -> VocabularyRescorer.RescoringResult {
            VocabularyRescorer.RescoringResult(
                originalWord: original, originalScore: -5, replacementWord: term, replacementScore: -1,
                shouldReplace: true, reason: "test")
        }
        XCTAssertEqual(
            SlidingWindowAsrManager.renderingReplacements("on cubernetis", [replacement("cubernetis", "Kubernetes")]),
            "on Kubernetes")
        XCTAssertEqual(
            SlidingWindowAsrManager.renderingReplacements("of truth,", [replacement("truth,", "Trooth")]),
            "of Trooth,", "the span's outer punctuation stays, as the rescorer writes it")
        XCTAssertEqual(
            SlidingWindowAsrManager.renderingReplacements(
                "word safe for.", [replacement("safe", "Sie"), replacement("unrelated", "Term")]),
            "word Sie for.")
        XCTAssertEqual(
            SlidingWindowAsrManager.removingTrailingWord("on Kubernetes", from: "runs on Kubernetes"), "runs")
        XCTAssertEqual(SlidingWindowAsrManager.removingTrailingWord("thirty years", from: "to thirty years"), "to")
    }

    func testTheUpdateInitStoresTheManagersText() {
        let update = SlidingWindowTranscriptionUpdate(
            text: "30 years.", isConfirmed: true, confidence: 0.9, timestamp: Date(),
            confirmedTranscript: "by twenty-five to 30 years.", volatileTranscript: "")
        XCTAssertEqual(update.confirmedTranscript, "by twenty-five to 30 years.")
        XCTAssertEqual(update.volatileTranscript, "")
    }
}
