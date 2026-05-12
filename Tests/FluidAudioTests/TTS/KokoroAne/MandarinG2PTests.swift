import Foundation
import XCTest

@testable import FluidAudio

/// Network-free unit tests for the Phase-2 Mandarin G2P pipeline.
///
/// The full HF MLX `pinyin_*.bin` dictionaries are ~3.6 MB and only
/// available after a network download, so each test below builds a
/// minimal in-memory `MandarinPinyinDict` that exercises the rule it
/// cares about. The orchestrator's segmenter, normalizer, sandhi, and
/// bopomofo encoder are all pure functions — exercising them in
/// isolation is enough to lock the contract.
final class MandarinG2PTests: XCTestCase {

    // MARK: - MandarinPinyinNormalizer

    func testNormalizerToneOne() {
        let s = MandarinPinyinNormalizer.normalize("qiū")
        XCTAssertEqual(s, .init(base: "qiu", tone: 1))
    }

    func testNormalizerToneTwo() {
        let s = MandarinPinyinNormalizer.normalize("ní")
        XCTAssertEqual(s, .init(base: "ni", tone: 2))
    }

    func testNormalizerToneThree() {
        let s = MandarinPinyinNormalizer.normalize("hǎo")
        XCTAssertEqual(s, .init(base: "hao", tone: 3))
    }

    func testNormalizerToneFour() {
        let s = MandarinPinyinNormalizer.normalize("shì")
        XCTAssertEqual(s, .init(base: "shi", tone: 4))
    }

    func testNormalizerNeutralTone() {
        // pypinyin emits neutral-tone syllables without diacritics.
        let s = MandarinPinyinNormalizer.normalize("de")
        XCTAssertEqual(s, .init(base: "de", tone: 5))
    }

    func testNormalizerUmlautCollapsesToV() {
        // ü-row → 'v' (matches pypinyin Style.TONE3 + ZH_MAP keys).
        XCTAssertEqual(
            MandarinPinyinNormalizer.normalize("lǜ"),
            .init(base: "lv", tone: 4))
        XCTAssertEqual(
            MandarinPinyinNormalizer.normalize("nǚ"),
            .init(base: "nv", tone: 3))
    }

    // MARK: - MandarinBopomofoMap

    func testEncodeBasicSyllables() {
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "ni", tone: 3), "ㄋㄧ3")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "hao", tone: 3), "ㄏㄠ3")
        // shi → sibilant fix → final "iii" maps to vocab token "十".
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "shi", tone: 4), "ㄕ十4")
        // ie → ㄝ alone (the ㄧ glide is implicit in the v1.1-zh vocab,
        // matching misaki/ZH_MAP).
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "jie", tone: 4), "ㄐㄝ4")
    }

    func testEncodeEmptyInitialNormalization() {
        // yi → i, wu → u, yu → v (then mapped via finalMap).
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "yi", tone: 1), "ㄧ1")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "wu", tone: 3), "ㄨ3")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "yu", tone: 2), "ㄩ2")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "yue", tone: 4), "月4")
    }

    func testEncodeSibilantIFix() {
        // zi/ci/si → ㄭ, zhi/chi/shi/ri → 十. Both forms appear verbatim
        // in the v1.1-zh vocab (kokoro was trained on these tokens).
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "zi", tone: 4), "ㄗㄭ4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "ci", tone: 4), "ㄘㄭ4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "si", tone: 1), "ㄙㄭ1")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "zhi", tone: 1), "ㄓ十1")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "chi", tone: 1), "ㄔ十1")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "shi", tone: 4), "ㄕ十4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "ri", tone: 4), "ㄖ十4")
    }

    func testEncodeJqxImplicitUmlaut() {
        // After j/q/x, surface "u" is actually ü (ㄩ). Without this rule
        // the model speaks `cu` for `qù`.
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "qu", tone: 4), "ㄑㄩ4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "ju", tone: 4), "ㄐㄩ4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "xu", tone: 1), "ㄒㄩ1")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "que", tone: 4), "ㄑ月4")
    }

    func testEncodePinyinContractionExpansion() {
        // Standard pinyin orthography contracts uei → ui, uen → un,
        // iou → iu after a consonant initial. The finalMap only carries
        // the full forms, so without expansion these syllables silently
        // drop. Hits common Hanzi like 贵/对/回/水/六/九/牛/顿/论.
        // ui → uei → finalMap["uei"] = "为"
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "gui", tone: 4), "ㄍ为4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "dui", tone: 4), "ㄉ为4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "hui", tone: 2), "ㄏ为2")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "shui", tone: 3), "ㄕ为3")
        // iu → iou → finalMap["iou"] = "又"
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "liu", tone: 4), "ㄌ又4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "jiu", tone: 3), "ㄐ又3")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "niu", tone: 2), "ㄋ又2")
        // un → uen → finalMap["uen"] = "文"
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "dun", tone: 4), "ㄉ文4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "lun", tone: 4), "ㄌ文4")
        XCTAssertEqual(MandarinBopomofoMap.encode(syllable: "chun", tone: 1), "ㄔ文1")
    }

    func testEncodeRejectsUnknownSyllable() {
        // No defined initial / final → drop.
        XCTAssertNil(MandarinBopomofoMap.encode(syllable: "xyzq", tone: 1))
        XCTAssertNil(MandarinBopomofoMap.encode(syllable: "", tone: 3))
    }

    // MARK: - MandarinToneSandhi

    func testSandhiThreeThree() {
        var syls: [MandarinPinyinNormalizer.Syllable] = [
            .init(base: "ni", tone: 3),
            .init(base: "hao", tone: 3),
        ]
        MandarinToneSandhi.apply(&syls)
        XCTAssertEqual(syls[0].tone, 2, "First 3 in 3+3 must promote to 2")
        XCTAssertEqual(syls[1].tone, 3, "Trailing 3 stays")
    }

    func testSandhiThreeThreeThree() {
        // Right-to-left scan locks the rightmost pair first → 3 3 3 → 2 2 3.
        var syls: [MandarinPinyinNormalizer.Syllable] = [
            .init(base: "wo", tone: 3),
            .init(base: "yi", tone: 3),
            .init(base: "qi", tone: 3),
        ]
        MandarinToneSandhi.apply(&syls)
        XCTAssertEqual(syls.map { $0.tone }, [2, 2, 3])
    }

    func testSandhiBu() {
        // 不 (bu4) before tone-4 → bu2.
        var syls: [MandarinPinyinNormalizer.Syllable] = [
            .init(base: "bu", tone: 4),
            .init(base: "yao", tone: 4),
        ]
        MandarinToneSandhi.apply(&syls)
        XCTAssertEqual(syls[0].tone, 2)
        XCTAssertEqual(syls[1].tone, 4)
    }

    func testSandhiBuStaysBeforeOtherTones() {
        var syls: [MandarinPinyinNormalizer.Syllable] = [
            .init(base: "bu", tone: 4),
            .init(base: "lai", tone: 2),
        ]
        MandarinToneSandhi.apply(&syls)
        XCTAssertEqual(syls[0].tone, 4, "bu4 before non-4 must stay 4")
    }

    func testSandhiYiBeforeTone4() {
        var syls: [MandarinPinyinNormalizer.Syllable] = [
            .init(base: "yi", tone: 1),
            .init(base: "ding", tone: 4),
        ]
        MandarinToneSandhi.apply(&syls)
        XCTAssertEqual(syls[0].tone, 2, "yi1 before tone-4 → tone-2")
    }

    func testSandhiYiBeforeTones123() {
        for nextTone in [1, 2, 3] {
            var syls: [MandarinPinyinNormalizer.Syllable] = [
                .init(base: "yi", tone: 1),
                .init(base: "x", tone: nextTone),
            ]
            MandarinToneSandhi.apply(&syls)
            XCTAssertEqual(
                syls[0].tone, 4,
                "yi1 before tone-\(nextTone) must promote to tone-4")
        }
    }

    // MARK: - MandarinG2P (orchestrator)

    /// Helper: build a tiny dict covering exactly the inputs used in
    /// the orchestrator-level tests.
    private static func smallDict() -> MandarinPinyinDict {
        let phrases: [String: [String]] = [
            "你好": ["nǐ", "hǎo"],
            "世界": ["shì", "jiè"],
        ]
        let singles: [UInt32: [String]] = [
            // 我 wǒ
            0x6211: ["wǒ"],
            // 一 yī
            0x4E00: ["yī"],
            // 不 bù
            0x4E0D: ["bù"],
            // 去 qù
            0x53BB: ["qù"],
            // 你 nǐ
            0x4F60: ["nǐ"],
            // 好 hǎo
            0x597D: ["hǎo"],
        ]
        return MandarinPinyinDict(phrases: phrases, singles: singles)
    }

    func testLooksLikeHanzi() {
        XCTAssertTrue(MandarinG2P.looksLikeHanzi("你好"))
        XCTAssertTrue(MandarinG2P.looksLikeHanzi("hello 世界"))
        XCTAssertFalse(MandarinG2P.looksLikeHanzi("ㄋㄧ3ㄏㄠ3"))
        XCTAssertFalse(MandarinG2P.looksLikeHanzi("hello"))
        XCTAssertFalse(MandarinG2P.looksLikeHanzi(""))
    }

    func testNormalizeTextFullwidthPunctuation() {
        XCTAssertEqual(
            MandarinG2P.normalizeText("你好，世界。"),
            "你好,世界.")
        XCTAssertEqual(MandarinG2P.normalizeText("！？；："), "!?;:")
    }

    func testPhonemizeRejectsEmpty() async throws {
        let g2p = MandarinG2P(dict: Self.smallDict())
        do {
            _ = try await g2p.phonemize("")
            XCTFail("Expected throw on empty input")
        } catch let e as KokoroAneError {
            guard case .inputProcessingFailed = e else {
                XCTFail("Expected inputProcessingFailed, got \(e)")
                return
            }
        }
        do {
            _ = try await g2p.phonemize("   ")
            XCTFail("Expected throw on whitespace-only input")
        } catch {
            // expected
        }
    }

    func testPhonemizePhraseLookupBeatsSingles() async throws {
        // "你好" should resolve via the phrases dict (FMM), not by
        // single-char fallback. Both produce the same syllables here so
        // we instead assert the phrase-level sandhi runs (3+3 → 2+3 →
        // bopomofo "ㄋㄧ2ㄏㄠ3").
        let g2p = MandarinG2P(dict: Self.smallDict())
        let out = try await g2p.phonemize("你好")
        XCTAssertEqual(out, "ㄋㄧ2ㄏㄠ3")
    }

    func testPhonemizeAppliesBuSandhi() async throws {
        let g2p = MandarinG2P(dict: Self.smallDict())
        let out = try await g2p.phonemize("我不去")
        // 我 wǒ → "uo" final tokenises as the vocab Hanzi token "我".
        // bu4-before-qu4 → bu2. qu4 → q + ü(ㄩ) + 4.
        XCTAssertEqual(out, "我3ㄅㄨ2ㄑㄩ4")
    }

    func testPhonemizeKeepsPunctuation() async throws {
        let g2p = MandarinG2P(dict: Self.smallDict())
        let out = try await g2p.phonemize("你好,世界。")
        // Sandhi never crosses punctuation, so 你好 → 2+3 and 世界 →
        // 4+4 (no sandhi between them). 世 shi4 → ㄕ + 十; 界 jie4 →
        // ㄐ + ㄝ (ㄧ glide is implicit in the v1.1-zh vocab).
        XCTAssertEqual(out, "ㄋㄧ2ㄏㄠ3,ㄕ十4ㄐㄝ4.")
    }

    func testPhonemizeSandhiResetsAtPunctuation() async throws {
        // Two adjacent third tones across a comma must NOT trigger
        // 3+3 sandhi.
        let g2p = MandarinG2P(dict: Self.smallDict())
        let out = try await g2p.phonemize("你好,你好")
        XCTAssertEqual(out, "ㄋㄧ2ㄏㄠ3,ㄋㄧ2ㄏㄠ3")
    }

    // MARK: - MandarinPinyinDict round-trip

    func testPinyinDictParsesSingles() throws {
        // Build a 1-entry singles payload by hand and verify the
        // parser walks the binary format exactly.
        var data = Data()
        // codepoint 0x4F60 (你) little-endian
        data.append(contentsOf: [0x60, 0x4F, 0x00, 0x00])
        // pinyin count = 1
        data.append(0x01)
        // pinyin entry: "ni" (2 bytes)
        data.append(0x02)
        data.append(contentsOf: Array("ni".utf8))
        let parsed = try MandarinPinyinDict.parseSingles(data)
        XCTAssertEqual(parsed[0x4F60], ["ni"])
    }

    func testPinyinDictParsesPhrases() throws {
        var data = Data()
        let phrase = "你好"
        let phraseBytes = Array(phrase.utf8)
        // u16_le phrase length
        data.append(UInt8(phraseBytes.count & 0xFF))
        data.append(UInt8((phraseBytes.count >> 8) & 0xFF))
        data.append(contentsOf: phraseBytes)
        // pinyin count = 2
        data.append(0x02)
        for s in ["ni", "hao"] {
            let b = Array(s.utf8)
            data.append(UInt8(b.count))
            data.append(contentsOf: b)
        }
        let parsed = try MandarinPinyinDict.parsePhrases(data)
        XCTAssertEqual(parsed[phrase], ["ni", "hao"])
    }
}
