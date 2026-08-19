import XCTest

@testable import FluidAudio

/// Matcher-level tests for the Nemotron decode-time hotword biasing engine,
/// against a toy SentencePiece-style table — no CoreML needed.
final class NemotronVocabularyBiasTests: XCTestCase {

    /// Toy piece table. `▁` marks word starts, case variants fold together,
    /// `<...>` pieces are specials the engine must ignore.
    private let pieces: [Int: String] = [
        0: "▁",
        1: "▁to", 2: "r", 3: "vane", 4: "▁tor", 5: "▁torvane",
        6: "▁The", 7: "▁the",
        8: "▁ran", 9: "▁t", 10: "▁cr", 11: "an",
        13: "东", 14: "京",
        15: "▁नम", 16: "स्", 17: "ते",
        18: "<unk>", 19: "<en-US>",
        21: "▁ab", 22: "▁c", 23: "v", 24: "▁vor",
    ]

    private func makeBias(
        _ terms: [CustomVocabularyTerm], defaultBoost: Float = NemotronVocabularyBias.defaultBoost
    ) -> NemotronVocabularyBias? {
        NemotronVocabularyBias(terms: terms, pieces: pieces, defaultBoost: defaultBoost)
    }

    private func candidateMap(_ bias: NemotronVocabularyBias) -> [Int: Float] {
        Dictionary(uniqueKeysWithValues: bias.candidates().map { ($0.tokenId, $0.boost) })
    }

    // MARK: - Term hygiene

    func testInitReturnsNilWithNoUsableTerms() {
        XCTAssertNil(makeBias([CustomVocabularyTerm(text: "ab")]))  // too short
        XCTAssertNil(makeBias([CustomVocabularyTerm(text: "torvane", weight: 0)]))
        XCTAssertNil(makeBias([CustomVocabularyTerm(text: "torvane", weight: -1)]))
        XCTAssertNil(makeBias([]))
    }

    func testIsUsableMatchesEngineGuards() {
        XCTAssertTrue(NemotronVocabularyBias.isUsable(CustomVocabularyTerm(text: "torvane")))
        XCTAssertFalse(NemotronVocabularyBias.isUsable(CustomVocabularyTerm(text: "ab")))
        XCTAssertFalse(NemotronVocabularyBias.isUsable(CustomVocabularyTerm(text: "torvane", weight: -1)))
        // Two CJK clusters are a full word, not a short keyword.
        XCTAssertTrue(NemotronVocabularyBias.isUsable(CustomVocabularyTerm(text: "东京")))
        // A term rescued by a usable alias is usable.
        XCTAssertTrue(
            NemotronVocabularyBias.isUsable(CustomVocabularyTerm(text: "ab", aliases: ["torvane"])))
    }

    // MARK: - Fresh-start candidates (word-start anchoring)

    func testFreshStartBoostsAnchoredPrefixPieces() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "Torvane")]))
        let map = candidateMap(bias)
        XCTAssertNotNil(map[1], "▁to is a 2-letter anchored prefix")
        XCTAssertNotNil(map[4], "▁tor is an anchored prefix")
        XCTAssertNotNil(map[5], "▁torvane is the whole term")
        XCTAssertNil(map[3], "vane does not start the term at a word boundary")
        XCTAssertNil(map[2], "bare r is not an anchored prefix")
    }

    func testFreshStartExcludesSingleLetterAndBareMarkerPieces() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        let map = candidateMap(bias)
        XCTAssertNil(map[9], "single-letter word-initial piece ▁t must not carry a standing boost")
        XCTAssertNil(map[0], "the bare ▁ piece narrows nothing and must never be boosted")
    }

    func testCaseVariantsOfAPieceAllFold() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "the")]))
        let map = candidateMap(bias)
        XCTAssertNotNil(map[6], "▁The folds to the term")
        XCTAssertNotNil(map[7], "▁the folds to the term")
    }

    // MARK: - Continuation matching

    func testContinuationAfterPartialMatch() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        bias.observe(4)  // ▁tor
        let map = candidateMap(bias)
        XCTAssertNotNil(map[3], "vane completes the term from the matched offset")
        XCTAssertNotNil(map[23], "a single-letter continuation (v) may narrow an open match")
        XCTAssertNotNil(map[5], "fresh start stays live alongside the continuation")
    }

    func testWordStartAnchoringBlocksMidWordMatch() throws {
        // The #702 over-fire mode: "ran" must not match into "CRAN".
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "ran")]))
        bias.observe(10)  // ▁cr
        let map = candidateMap(bias)
        XCTAssertNil(map[11], "an must not be boosted mid-word after ▁cr")
        XCTAssertNotNil(map[8], "the anchored fresh start ▁ran remains the only candidate")
        XCTAssertEqual(map.count, 1)
    }

    func testMultiOffsetOverlapSurvivesRepeatedLeadingWord() throws {
        // "ab ab c" hearing "ab ab ab c": the shorter overlap keeps the
        // term boostable.
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "ab ab c")]))
        bias.observe(21)  // ▁ab
        bias.observe(21)  // ▁ab
        var map = candidateMap(bias)
        XCTAssertNotNil(map[22], "▁c completes the full two-word match")
        XCTAssertNotNil(map[21], "▁ab is also live through the one-word overlap")
        bias.observe(21)  // a third ▁ab — full-match offset shifts to the overlap
        map = candidateMap(bias)
        XCTAssertNotNil(map[22], "▁c stays live through the repeated leading word")
    }

    // MARK: - Multilingual

    func testCJKTermIsUnanchoredAndAcceptsTwoClusters() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "东京")]))
        var map = candidateMap(bias)
        XCTAssertNotNil(map[13], "东 opens the unanchored CJK term")
        XCTAssertNil(map[14], "京 does not start the term")
        bias.observe(13)
        map = candidateMap(bias)
        XCTAssertNotNil(map[14], "京 completes the term after 东")
    }

    func testDevanagariConjunctRejoinsInScalarSpace() throws {
        // SentencePiece splits the नमस्ते conjunct across pieces (स् is a
        // half-form); matching must run on scalars to re-join them.
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "नमस्ते")]))
        var map = candidateMap(bias)
        XCTAssertNotNil(map[15], "▁नम opens the term")
        bias.observe(15)
        map = candidateMap(bias)
        XCTAssertNotNil(map[16], "स् continues the term mid-cluster")
        bias.observe(16)
        map = candidateMap(bias)
        XCTAssertNotNil(map[17], "ते completes the term across the split conjunct")
    }

    // MARK: - Aliases and weights

    func testAliasIsBoostedAsItself() throws {
        let bias = try XCTUnwrap(
            makeBias([CustomVocabularyTerm(text: "torvane", aliases: ["vortane"])]))
        let map = candidateMap(bias)
        XCTAssertNotNil(map[4], "the primary surface is live")
        XCTAssertNotNil(map[24], "▁vor opens the alias surface")
    }

    func testDefaultAndOverrideBoosts() throws {
        let plain = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        XCTAssertEqual(candidateMap(plain)[5], NemotronVocabularyBias.defaultBoost)
        let weighted = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane", weight: 5.5)]))
        XCTAssertEqual(candidateMap(weighted)[5], 5.5)
    }

    func testLegacyWeightsFallBackToDefault() throws {
        // The simple text-list loader assigns weight 10.0 on a CTC
        // rescoring scale; applied raw as a per-token logit bonus it would
        // over-bias, and pinning it to the aggressive 6.0 ceiling is still
        // hotter than the measured recall peak — so out-of-range weights
        // use the default instead.
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane", weight: 10.0)]))
        XCTAssertEqual(candidateMap(bias)[5], NemotronVocabularyBias.defaultBoost)
        XCTAssertEqual(
            NemotronVocabularyBias.effectiveBoost(of: CustomVocabularyTerm(text: "x", weight: 10)),
            NemotronVocabularyBias.defaultBoost)
        // An explicit in-range weight is still honored, up to the ceiling.
        XCTAssertEqual(
            NemotronVocabularyBias.effectiveBoost(
                of: CustomVocabularyTerm(text: "x", weight: NemotronVocabularyBias.maxBoost)),
            NemotronVocabularyBias.maxBoost)
    }

    func testStrongestBoostWinsPerToken() throws {
        let bias = try XCTUnwrap(
            makeBias([
                CustomVocabularyTerm(text: "torvane", weight: 2.0),
                CustomVocabularyTerm(text: "torment", weight: 5.0),
            ]))
        // ▁tor prefixes both terms; the stronger boost must win.
        XCTAssertEqual(candidateMap(bias)[4], 5.0)
    }

    // MARK: - Match-state lifecycle

    func testSpecialPiecesNeverDisturbMatchState() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        bias.observe(4)  // ▁tor
        bias.observe(19)  // <en-US> lang tag — no piece text
        bias.observe(18)  // <unk>
        XCTAssertNotNil(candidateMap(bias)[3], "vane stays live across special-token emissions")
    }

    func testUnmatchedEmissionDropsContinuation() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        bias.observe(4)  // ▁tor
        bias.observe(7)  // ▁the — breaks the match
        XCTAssertNil(candidateMap(bias)[3], "vane is no longer a continuation")
        XCTAssertNotNil(candidateMap(bias)[4], "fresh start is still live")
    }

    func testResetMatchStateClearsContinuationsKeepsVocabulary() throws {
        let bias = try XCTUnwrap(makeBias([CustomVocabularyTerm(text: "torvane")]))
        bias.observe(4)
        XCTAssertNotNil(candidateMap(bias)[3])
        bias.resetMatchState()
        let map = candidateMap(bias)
        XCTAssertNil(map[3], "continuation gone after reset")
        XCTAssertNotNil(map[5], "fresh-start candidates survive reset")
    }

    // MARK: - Biased selection

    func testPickBiasedFlipsOnlyWhenBoostedLogitWins() {
        let logits: [Float] = [1.0, 5.0, 3.0]
        let read: (Int) -> Float = { logits[$0] }
        // 3.0 + 4.5 beats 5.0 → flip.
        XCTAssertEqual(
            StreamingNemotronMultilingualAsrManager.pickBiased(
                plain: 1,
                candidates: [.init(tokenId: 2, boost: 4.5)], count: 3, logit: read),
            2)
        // 1.0 + 3.0 loses to 5.0 → no flip.
        XCTAssertEqual(
            StreamingNemotronMultilingualAsrManager.pickBiased(
                plain: 1,
                candidates: [.init(tokenId: 0, boost: 3.0)], count: 3, logit: read),
            1)
        // Out-of-range candidate ids are ignored, never read.
        XCTAssertEqual(
            StreamingNemotronMultilingualAsrManager.pickBiased(
                plain: 1,
                candidates: [.init(tokenId: 7, boost: 100)], count: 3, logit: read),
            1)
    }

    func testPickBiasedCanOvertakeBlank() {
        // Blank (id 2) is the plain argmax; a boosted term token must be
        // able to overtake it — that is how a term the decoder was about to
        // drop gets emitted.
        let logits: [Float] = [2.0, 1.0, 4.0]
        XCTAssertEqual(
            StreamingNemotronMultilingualAsrManager.pickBiased(
                plain: 2,
                candidates: [.init(tokenId: 0, boost: 4.5)], count: 3, logit: { logits[$0] }),
            0)
    }

    // MARK: - Scale

    func testLargeVocabularySmoke() throws {
        // 2000 terms sharing no prefix with the emitted tail: per-step cost
        // must stay bounded by the tail walk, not the term count. This is a
        // functional smoke (timing belongs to a benchmark rig), but a
        // regression to per-term scanning would show up as a timeout here.
        var terms = (0..<2000).map { CustomVocabularyTerm(text: "zqterm\($0)vx") }
        terms.append(CustomVocabularyTerm(text: "torvane"))
        let bias = try XCTUnwrap(makeBias(terms))
        for _ in 0..<200 {
            bias.observe(4)  // ▁tor
            XCTAssertNotNil(candidateMap(bias)[3])
            bias.observe(7)  // ▁the
            XCTAssertFalse(bias.candidates().isEmpty)
        }
    }
}
