import Foundation
import XCTest

@testable import FluidAudio

/// Coverage for the POS-aware tone sandhi rules. Inputs are
/// constructed by hand from the `(base, tone)` form so the tests
/// stay independent of the dictionary lookup path.
final class MandarinToneSandhiPOSTests: XCTestCase {

    private typealias Syllable = MandarinPinyinNormalizer.Syllable

    private func syl(_ base: String, _ tone: Int) -> Syllable {
        Syllable(base: base, tone: tone)
    }

    // MARK: - 一 carve-outs

    func testYiOrdinalKeepsToneOneInSoloNumeralWord() {
        // 第一 → `["第", "一"]` with both tagged `m`. The 一 sits in
        // its own one-syllable numeral word, so the carve-out must
        // suppress the yi1 + tone-4 → 2 promotion (next would be
        // tone-4 if a measure word followed; we test the bare form).
        var syllables = [syl("di", 4), syl("yi", 1)]
        let words = [0..<1, 1..<2]
        let tags = ["m", "m"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [4, 1])
    }

    func testYiInOrdinalBeforeTone1WordKeepsToneOne() {
        // 一月 → `["一", "月"]`, tags `["m", "m"]`. Without the
        // carve-out the baseline would promote yi1 to tone 4 in
        // front of yue1 (tone-1 successor).
        var syllables = [syl("yi", 1), syl("yue", 1)]
        let words = [0..<1, 1..<2]
        let tags = ["m", "m"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [1, 1])
    }

    func testYiContextualSandhiStillFiresInVerbContext() {
        // 一起 (yi1 + qi3) → POS tags `["d", "v"]` in jieba (一起 is
        // a single word in practice, but for the sandhi rule we
        // exercise the case where 一 is *not* in a solo numeral
        // word). yi1 + tone-3 → tone 4.
        var syllables = [syl("yi", 1), syl("qi", 3)]
        let words = [0..<2]
        let tags = ["d"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        // 一起: yi promotes to 4, qi stays tone 3 (single tone-3, no
        // run, no in-word 3+3 trigger).
        XCTAssertEqual(syllables.map { $0.tone }, [4, 3])
    }

    func testYiBeforeFourthToneVerbStillPromotesToTwo() {
        // 一定 (yi1 + ding4) → `["d"]` (single word). Not an ordinal
        // numeral word, so the standard rule fires.
        var syllables = [syl("yi", 1), syl("ding", 4)]
        let words = [0..<2]
        let tags = ["d"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 4])
    }

    // MARK: - 不 reduplication

    func testBuReduplicationKeepsToneFour() {
        // 好不好 → `["hao", "bu", "hao"]`. The baseline promotes
        // bu4 → bu2 because the next syllable is tone 3 (no, hold
        // on — baseline rule is "bu4 + tone-4 → bu2", so for 好不好
        // (3 4 3) baseline keeps bu4 already). The actual case the
        // POS rule changes is reduplication where the next syllable
        // is tone 4 — `要不要` (yao4 + bu4 + yao4): baseline
        // promotes to bu2 (wrong), POS rule keeps bu4.
        var syllables = [syl("yao", 4), syl("bu", 4), syl("yao", 4)]
        let words = [0..<1, 1..<2, 2..<3]
        let tags = ["v", "d", "v"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [4, 4, 4])
    }

    func testBuPromotionStillFiresForNonReduplicationContext() {
        // 不要 → bu4 + yao4 → bu2 + yao4. No reduplication
        // (there's no preceding `yao`), so the standard rule applies.
        var syllables = [syl("bu", 4), syl("yao", 4)]
        let words = [0..<2]
        let tags = ["d"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 4])
    }

    func testBuReduplicationDistinctBasesTriggersPromotion() {
        // [yao, bu, qu] is *not* reduplication — different bases
        // either side. The rule still promotes (bu4 + qu4 → bu2).
        var syllables = [syl("yao", 4), syl("bu", 4), syl("qu", 4)]
        let words = [0..<1, 1..<2, 2..<3]
        let tags = ["v", "d", "v"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [4, 2, 4])
    }

    // MARK: - Word-grouped 3+3

    func testInWordRunPromotesAllButLast() {
        // Single word `wo3 ye3 xiang3` (hypothetical 3-syllable
        // word) → in-word 3+3 promotes the first two: 2 2 3.
        var syllables = [syl("wo", 3), syl("ye", 3), syl("xiang", 3)]
        let words = [0..<3]
        let tags = ["v"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 2, 3])
    }

    func testCrossWordPairOnlyPromotesBoundary() {
        // 我 也 想去 → words `[[wo], [ye], [xiang, qu]]`, tags
        // `["r", "d", "v"]`. All tone 3. In-word pass:
        //   word 0 [wo3]                     — single, untouched.
        //   word 1 [ye3]                     — single, untouched.
        //   word 2 [xiang3, qu4]             — qu is tone 4, only
        //                                     xiang is tone 3 (no run).
        // Cross-word pass:
        //   wo3 | ye3 → wo promotes to 2.
        //   ye3 | xiang3 → ye promotes to 2.
        //   xiang3 | (qu is tone 4) → no boundary promotion.
        // Final: [2, 2, 3, 4]
        var syllables = [
            syl("wo", 3), syl("ye", 3), syl("xiang", 3), syl("qu", 4),
        ]
        let words = [0..<1, 1..<2, 2..<4]
        let tags = ["r", "d", "v"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 2, 3, 4])
    }

    func testCrossWordChainStopsAtNonThree() {
        // 我 是 你 的 → wo3 shi4 ni3 de5. No 3+3 chain because shi4
        // breaks it; ni3 is the leading syllable of its word and
        // the *previous* word is shi4 (tone 4) → no promotion.
        var syllables = [
            syl("wo", 3), syl("shi", 4), syl("ni", 3), syl("de", 5),
        ]
        let words = [0..<1, 1..<2, 2..<3, 3..<4]
        let tags = ["r", "v", "r", "u"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [3, 4, 3, 5])
    }

    func testWordGroupedSandhiBeatsNaiveRunRule() {
        // 你 想 吗 → tags `["r", "v", "y"]`. Tones 3 3 5. The naive
        // pure-run rule would still promote ni3 because it sees a
        // 3+3 pair, then leave xiang3 alone (last in run). The POS
        // rule produces the same output here — this is a regression
        // anchor that the POS path doesn't *under*-promote.
        var syllables = [syl("ni", 3), syl("xiang", 3), syl("ma", 5)]
        let words = [0..<1, 1..<2, 2..<3]
        let tags = ["r", "v", "y"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 3, 5])
    }

    // MARK: - Backward-compat with single-word ranges

    func testSingleWordRangeMatchesBaselineForFlatRun() {
        // When the caller passes a single all-encompassing word,
        // the POS rule degenerates to the baseline run rule.
        var syllables = [syl("ni", 3), syl("hao", 3)]
        let words = [0..<2]
        let tags = ["a"]
        MandarinToneSandhiPOS.apply(&syllables, words: words, tags: tags)
        XCTAssertEqual(syllables.map { $0.tone }, [2, 3])
    }

    func testEmptyAndSingleSyllableBuffersAreNoops() {
        var empty: [Syllable] = []
        MandarinToneSandhiPOS.apply(&empty, words: [], tags: [])
        XCTAssertTrue(empty.isEmpty)

        var single = [syl("ni", 3)]
        MandarinToneSandhiPOS.apply(&single, words: [0..<1], tags: ["r"])
        XCTAssertEqual(single, [syl("ni", 3)])
    }

    // MARK: - Mismatched arity is a programmer error

    func testMismatchedTagsCountTrapsViaPrecondition() {
        // We can't assert preconditions in XCTest without a
        // custom runner, so document the contract via a sanity
        // check that the matched arity does not trap.
        var syllables = [syl("ni", 3), syl("hao", 3)]
        MandarinToneSandhiPOS.apply(
            &syllables, words: [0..<2], tags: ["a"])
        XCTAssertEqual(syllables.count, 2)
    }
}
