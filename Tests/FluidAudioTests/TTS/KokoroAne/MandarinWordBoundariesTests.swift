#if TTS
import XCTest

@testable import FluidAudio

/// Expected separators follow misaki 0.9.4 `ZHG2P(version="1.1")` output.
final class MandarinWordBoundariesTests: XCTestCase {

    private func word(_ text: String, _ tones: Int...) -> MandarinWordBoundaries.Word {
        MandarinWordBoundaries.Word(text: Array(text), tones: tones)
    }

    private func separators(_ words: MandarinWordBoundaries.Word...) -> [Bool] {
        MandarinWordBoundaries.separators(for: words)
    }

    func testSingleWordHasNoSeparator() {
        XCTAssertEqual(separators(word("你好", 3, 3)), [false])
        XCTAssertEqual(MandarinWordBoundaries.separators(for: []), [])
    }

    func testAdjacentWordsAreSeparated() {
        // 我们/去/公园/吧
        XCTAssertEqual(
            separators(word("我们", 3, 5), word("去", 4), word("公园", 1, 2), word("吧", 5)),
            [false, true, true, true])
    }

    func testBuJoinsTheNextWord() {
        // 我/不去
        XCTAssertEqual(separators(word("我", 3), word("不", 4), word("去", 4)), [false, true, false])
        // A trailing 不 stays its own word.
        XCTAssertEqual(separators(word("去", 4), word("不", 4)), [false, true])
    }

    func testYiReduplicationAndStandaloneYi() {
        // 看一看
        XCTAssertEqual(separators(word("看", 4), word("一", 1), word("看", 4)), [false, false, false])
        // 一个/人
        XCTAssertEqual(separators(word("一", 1), word("个", 4), word("人", 2)), [false, false, true])
    }

    func testReduplicatedWordsJoin() {
        XCTAssertEqual(separators(word("看", 4), word("看", 4)), [false, false])
    }

    func testAllThirdToneWordsJoinUpToThreeCharacters() {
        // 我你好 (1 + 2 characters) joins; 你好/很好 (2 + 2) does not.
        XCTAssertEqual(separators(word("我", 3), word("你好", 3, 3)), [false, false])
        XCTAssertEqual(separators(word("你好", 3, 3), word("很好", 3, 3)), [false, true])
    }

    func testThirdToneSeamJoins() {
        // 我喜欢/苹果: 我 + 喜 meet on two third tones.
        XCTAssertEqual(separators(word("我", 3), word("喜欢", 3, 1), word("苹果", 2, 3)), [false, false, true])
    }

    func testMergedWordDoesNotChainInTheSamePass() {
        // 我也想/去: 我也 joins in the all-third pass, 想 in the seam pass.
        XCTAssertEqual(
            separators(word("我", 3), word("也", 3), word("想", 3), word("去", 4)),
            [false, false, false, true])
    }

    func testReduplicationIsNotMergedByThirdTones() {
        // 想想/我: an AA reduplication never takes the next word.
        XCTAssertEqual(separators(word("想想", 3, 3), word("我", 3)), [false, true])
    }

    func testErJoinsThePreviousWord() {
        XCTAssertEqual(separators(word("这", 4), word("儿", 2)), [false, false])
    }
}
#endif
