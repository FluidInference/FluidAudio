import Foundation
import XCTest

@testable import FluidAudio

/// Network-free unit tests for the jieba HMM tail. Exercises both the
/// pure Viterbi decoder and its integration with `MandarinG2P.segment`.
///
/// All HMM tables used here are synthesised in-process from a small set
/// of hand-picked log-probabilities — no `.bin` artefacts are read from
/// disk. The fixture deliberately makes the "join 3 chars into a word"
/// path overwhelmingly likely on a known set of characters so the
/// Viterbi output is deterministic and easy to assert against.
final class MandarinJiebaHmmTests: XCTestCase {

    // MARK: - Fixture builder

    /// Build a 4-state HMM that:
    ///   * Strongly prefers `B M E` for the chars in `groupChars`
    ///     (joins them into one word when contiguous).
    ///   * Emits `S` for any other character.
    /// The resulting table is small (≤ ~10 chars) but enough to exercise
    /// every branch of the Viterbi backtrace.
    private static func buildToyTables(
        groupChars: Set<Character>,
        singletonChars: Set<Character>
    ) -> MandarinJiebaHmmTables {
        let highProb = 0.0  // log(1)
        let lowProb = -100.0  // effectively never
        let mediumProb = -1.0

        // start[B]=high, start[S]=high, start[M]=start[E]=low
        let start: [Double] = [
            highProb,  // B
            lowProb,  // M
            lowProb,  // E
            highProb,  // S
        ]

        // trans[from][to] (rows: B/M/E/S):
        //   B → M (continue word) very likely; B → E (2-char word) possible
        //   M → M (long words) possible; M → E (close word) likely
        //   E → B (start next word) likely; E → S (start next single)
        //   S → B (start next word) likely; S → S (next single) likely
        // Disallowed transitions get lowProb so the
        // `allowedPredecessors` constraint isn't the only safeguard.
        let trans: [[Double]] = [
            // from B
            [lowProb, highProb, mediumProb, lowProb],
            // from M
            [lowProb, mediumProb, highProb, lowProb],
            // from E
            [highProb, lowProb, lowProb, mediumProb],
            // from S
            [highProb, lowProb, lowProb, highProb],
        ]

        // Emission table: chars in groupChars are happy in B/M/E,
        // chars in singletonChars are happy in S, everything else is
        // mildly unhappy everywhere (tested via the unknown-char path).
        var emit: [Character: [Double]] = [:]
        for ch in groupChars {
            emit[ch] = [highProb, highProb, highProb, lowProb]
        }
        for ch in singletonChars {
            emit[ch] = [lowProb, lowProb, lowProb, highProb]
        }
        return MandarinJiebaHmmTables(start: start, trans: trans, emit: emit)
    }

    // MARK: - Viterbi correctness

    func testEmptyInputProducesEmptyOutput() {
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特", "朗", "普"], singletonChars: ["他"]))
        XCTAssertEqual(hmm.segment(""), [])
    }

    func testSingleCharBypassesViterbi() {
        // Single char input must short-circuit (no backtrack possible).
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特"], singletonChars: []))
        XCTAssertEqual(hmm.segment("特"), ["特"])
    }

    func testGroupCharsCollapseIntoWord() {
        // 特朗普 should collapse to a single B-M-E run.
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特", "朗", "普"], singletonChars: []))
        XCTAssertEqual(hmm.segment("特朗普"), ["特朗普"])
    }

    func testSingletonCharsStaySeparate() {
        // 他 / 们 / 去 are tagged as `S` — each emerges as its own word.
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: [], singletonChars: ["他", "们", "去"]))
        XCTAssertEqual(hmm.segment("他们去"), ["他", "们", "去"])
    }

    func testMixedRunPreservesBoundaries() {
        // 他 + 特朗普 → ["他", "特朗普"]. The boundary between an `S`
        // and the start of a new word (`B`) is the meat of the
        // segmenter — if the trans/emit interaction is wrong this is
        // where it breaks.
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特", "朗", "普"], singletonChars: ["他"]))
        XCTAssertEqual(hmm.segment("他特朗普"), ["他", "特朗普"])
    }

    func testOutputAlwaysConcatenatesToInput() {
        // Invariant: the segments must concatenate back to the input
        // verbatim. Worth asserting on every test path because the
        // backtrace is easy to get off-by-one.
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["比", "特", "币"], singletonChars: ["他", "用"]))
        let input = "他用比特币"
        let words = hmm.segment(input)
        XCTAssertEqual(words.joined(), input)
    }

    func testUnknownCharsStillProduceSomething() {
        // Chars absent from `emit` use the unknownCharLogProb sentinel;
        // the decoder must still produce a (non-empty) segmentation.
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特"], singletonChars: ["他"]))
        let words = hmm.segment("乙丙丁")  // none in emit
        XCTAssertEqual(words.joined(), "乙丙丁")
        XCTAssertFalse(words.isEmpty)
    }

    // MARK: - Binary loader round-trip

    func testTablesRoundTripThroughBinaryEncoder() throws {
        // Build a fixture, encode to bytes, decode back, compare.
        let original = Self.buildToyTables(
            groupChars: ["特", "朗", "普", "比", "币"],
            singletonChars: ["他", "用"])
        let encoded = original.encoded()
        let decoded = try MandarinJiebaHmmTables(
            startData: encoded.start,
            transData: encoded.trans,
            emitData: encoded.emit)

        // start / trans should round-trip exactly (Float32 representable).
        for i in 0..<original.start.count {
            XCTAssertEqual(decoded.start[i], original.start[i], accuracy: 1e-6)
        }
        for i in 0..<original.trans.count {
            for j in 0..<original.trans[i].count {
                XCTAssertEqual(
                    decoded.trans[i][j], original.trans[i][j], accuracy: 1e-6)
            }
        }
        XCTAssertEqual(Set(decoded.emit.keys), Set(original.emit.keys))
        for key in original.emit.keys {
            for s in 0..<JiebaHmmState.allCases.count {
                XCTAssertEqual(
                    decoded.emit[key]![s], original.emit[key]![s], accuracy: 1e-6)
            }
        }
    }

    func testWrongStartSizeIsRejected() {
        let badStart = Data(count: 8)  // expected 16
        let goodTrans = Data(count: 64)
        let goodEmit = Data()
        XCTAssertThrowsError(
            try MandarinJiebaHmmTables(
                startData: badStart, transData: goodTrans, emitData: goodEmit)
        ) { error in
            guard case MandarinJiebaHmmTables.LoadError.wrongSize(let what, _, _) = error
            else { return XCTFail("Expected wrongSize, got \(error)") }
            XCTAssertEqual(what, "start")
        }
    }

    func testTruncatedEmitIsRejected() {
        let goodStart = Data(count: 16)
        let goodTrans = Data(count: 64)
        let badEmit = Data(count: 5)  // not a multiple of recordSize (20)
        XCTAssertThrowsError(
            try MandarinJiebaHmmTables(
                startData: goodStart, transData: goodTrans, emitData: badEmit)
        ) { error in
            guard case MandarinJiebaHmmTables.LoadError.truncated = error else {
                return XCTFail("Expected truncated, got \(error)")
            }
        }
    }

    // MARK: - Integration with MandarinG2P.segment

    func testMandarinG2PWithoutHmmKeepsPerCharFallback() async throws {
        // No HMM: 特朗普 (not in the toy phrase dict) should each char
        // get a per-singles lookup. Sanity check that wiring HMM as
        // optional preserves the baseline behaviour.
        let dict = Self.miniDict()
        let g2p = MandarinG2P(dict: dict)
        let phon = try await g2p.phonemize("特朗普")
        XCTAssertFalse(phon.isEmpty)
        // Each char emits at least an initial+final+tone fragment.
        XCTAssertGreaterThanOrEqual(phon.count, 3)
    }

    func testMandarinG2PWithHmmRetriesPhraseDict() async throws {
        // With HMM + a phrase entry for 特朗普, the per-char fallback
        // should be replaced by the phrase pinyin.
        var phrases: [String: [String]] = [:]
        phrases["特朗普"] = ["tè", "lǎng", "pǔ"]
        let singles: [UInt32: [String]] = [
            0x7279: ["tè"],  // 特
            0x6717: ["lǎng"],  // 朗
            0x666E: ["pǔ"],  // 普
        ]
        let dict = MandarinPinyinDict(phrases: phrases, singles: singles)
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特", "朗", "普"], singletonChars: []))
        let g2p = MandarinG2P(dict: dict, jiebaHmm: hmm)

        // The HMM-recovered word `特朗普` hits the phrase dict; the
        // result should equal what a pure-phrase lookup would produce.
        let g2pPhraseOnly = MandarinG2P(dict: dict)
        let withHmm = try await g2p.phonemize("特朗普")
        let withoutHmmButCached = try await g2pPhraseOnly.phonemize("特朗普")
        XCTAssertEqual(withHmm, withoutHmmButCached)
    }

    func testIntegrationFlushesHanziRunOnPunctuation() async throws {
        // 他特朗普,你好 — punctuation between hanzi-runs must split the
        // HMM input, otherwise the comma would end up inside the
        // segmenter's character buffer.
        var phrases: [String: [String]] = [:]
        phrases["特朗普"] = ["tè", "lǎng", "pǔ"]
        phrases["你好"] = ["nǐ", "hǎo"]
        let singles: [UInt32: [String]] = [
            0x4ED6: ["tā"],  // 他
            0x7279: ["tè"], 0x6717: ["lǎng"], 0x666E: ["pǔ"],
            0x4F60: ["nǐ"], 0x597D: ["hǎo"],
        ]
        let dict = MandarinPinyinDict(phrases: phrases, singles: singles)
        let hmm = MandarinJiebaHmm(
            tables: Self.buildToyTables(
                groupChars: ["特", "朗", "普"], singletonChars: ["他"]))
        let g2p = MandarinG2P(dict: dict, jiebaHmm: hmm)
        let phon = try await g2p.phonemize("他特朗普,你好")
        // The comma should appear in the bopomofo string (passthrough).
        XCTAssertTrue(phon.contains(","), "expected ',' in '\(phon)'")
    }

    // MARK: - Test fixtures

    private static func miniDict() -> MandarinPinyinDict {
        let singles: [UInt32: [String]] = [
            0x7279: ["tè"],  // 特
            0x6717: ["lǎng"],  // 朗
            0x666E: ["pǔ"],  // 普
            0x4F60: ["nǐ"],  // 你
            0x597D: ["hǎo"],  // 好
        ]
        return MandarinPinyinDict(phrases: [:], singles: singles)
    }
}
