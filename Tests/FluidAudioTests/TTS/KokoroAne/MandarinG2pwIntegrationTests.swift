import Foundation
import XCTest

@testable import FluidAudio

/// Network-free integration tests for the MandarinG2P + g2pW wiring.
///
/// The g2pW model itself is a 152 MB CoreML bundle that requires a
/// network download, so these tests exercise the *identification* of
/// polyphone targets and the dict-only fallback path. The actual
/// `MandarinG2pwModel.disambiguate(...)` call is covered by a
/// network-gated benchmark (`mandarin-cer-benchmark` CLI command).
final class MandarinG2pwIntegrationTests: XCTestCase {

    /// Build a minimal `MandarinPinyinDict` containing both
    /// monophonic chars (one reading) and one polyphonic char
    /// (`行` with 4 readings) so the segmenter has something to flag.
    private static func mixedDict() -> MandarinPinyinDict {
        var phrases: [String: [String]] = [:]
        phrases["你好"] = ["nǐ", "hǎo"]
        var singles: [UInt32: [String]] = [:]
        // 我 wǒ — single reading.
        singles[Character("我").unicodeScalars.first!.value] = ["wǒ"]
        // 去 qù — single reading.
        singles[Character("去").unicodeScalars.first!.value] = ["qù"]
        // 行 — polyphonic. The dict's first reading wins by default.
        singles[Character("行").unicodeScalars.first!.value] = [
            "xíng", "háng", "xìng",
        ]
        // 银 — single reading.
        singles[Character("银").unicodeScalars.first!.value] = ["yín"]
        return MandarinPinyinDict(phrases: phrases, singles: singles)
    }

    func testSegmenterFlagsPolyphonicTargets() throws {
        let g2p = MandarinG2P(dict: Self.mixedDict())
        let chars: [Character] = Array("我去银行")
        let result = g2p.segment(chars: chars)
        // 我 / 去 / 银 → single reading, no flag. 行 (idx 3) →
        // 3 readings → polyphone target.
        XCTAssertEqual(result.polyphoneTargets.count, 1)
        let target = try XCTUnwrap(result.polyphoneTargets.first)
        XCTAssertEqual(target.charPos, 3)
        // The flagged segment must still resolve via the dict in the
        // absence of g2pW.
        XCTAssertEqual(result.segments.count, 4)
    }

    func testPhraseSegmentsAreNotFlaggedAsTargets() throws {
        let g2p = MandarinG2P(dict: Self.mixedDict())
        let chars: [Character] = Array("你好")
        let result = g2p.segment(chars: chars)
        // FMM hits the phrase — neither char becomes a polyphone
        // candidate (the phrase already disambiguated context).
        XCTAssertEqual(result.polyphoneTargets.count, 0)
        XCTAssertEqual(result.segments.count, 1)
    }

    func testDictOnlyFallbackProducesBaselineOutput() async throws {
        // Without a g2pW model wired in, polyphonic chars fall back
        // to the dict's first-listed reading. This is the existing
        // pipeline behaviour and must not regress when the optional
        // g2pW field stays nil.
        let g2p = MandarinG2P(dict: Self.mixedDict())
        let out = try await g2p.phonemize("我去银行")
        // 我 wǒ (vocab Hanzi token "我"3) + 去 qù (ㄑㄩ4) +
        // 银 yín (ㄧㄣ2) + 行 xíng (ㄒ + 一ㄥ vocab token + 2).
        // Just assert the output is non-empty and contains the
        // dict-derived 行 → "ㄒ" prefix; the exact bopomofo of 行
        // depends on the v1.1-zh vocab special-token mapping.
        XCTAssertFalse(out.isEmpty)
        XCTAssertTrue(out.contains("ㄒ"))
    }

    private static func syllables(_ pinyin: String...) -> [MandarinPinyinNormalizer.Syllable] {
        pinyin.map(MandarinPinyinNormalizer.normalize)
    }

    private static func labels(_ indices: [Int], _ catalog: MandarinPolyphoneCatalog) -> [String] {
        indices.compactMap(catalog.bopomofo(forLabel:)).sorted()
    }

    /// Bins for `char` from tab-separated catalog rows, and a probability
    /// vector setter keyed by label.
    private static func bins(
        _ char: Character, _ labels: [String], readings: [String]
    ) throws -> (MandarinG2pwModel.MainlandBins?, ([String: Float]) -> [Float]) {
        let catalog = try MandarinPolyphoneCatalog(text: labels.map { "\(char)\t\($0)\n" }.joined())
        let bins = MandarinG2pwModel.MainlandBins(
            char: char, readings: readings.map(MandarinPinyinNormalizer.normalize), catalog: catalog)
        let probs: ([String: Float]) -> [Float] = { values in
            var probs = [Float](repeating: 0, count: catalog.labels.count)
            for (label, value) in values { probs[catalog.labels.firstIndex(of: label)!] = value }
            return probs
        }
        return (bins, probs)
    }

    func testTaiwanReadingCountsTowardTheDefault() throws {
        // 和 hàn (Taiwan, "and") is mainland hé: 0.6 + 0.15 beats huò's 0.25.
        let (bins, probs) = try Self.bins(
            "和", ["ㄏㄜ2", "ㄏㄢ4", "ㄏㄨㄛ4", "ㄏㄜ4"], readings: ["hé", "hè", "huó", "huò"])
        let mainland = try XCTUnwrap(bins)
        XCTAssertNil(mainland.pick(probs(["ㄏㄢ4": 0.6, "ㄏㄨㄛ4": 0.25, "ㄏㄜ2": 0.15])))
        XCTAssertEqual(mainland.pick(probs(["ㄏㄨㄛ4": 0.8, "ㄏㄢ4": 0.1, "ㄏㄜ2": 0.1])), "ㄏㄨㄛ4")
    }

    func testTaiwanToneOfTheDefaultStillLetsTheModelChoose() throws {
        // 削: the inventory has xiāo and Taiwan xuè, not the default xuē.
        let (bins, probs) = try Self.bins("削", ["ㄒㄧㄠ1", "ㄒㄩㄝ4"], readings: ["xuē", "xiāo", "qiào", "shào"])
        let mainland = try XCTUnwrap(bins)
        XCTAssertEqual(mainland.pick(probs(["ㄒㄧㄠ1": 1])), "ㄒㄧㄠ1")
        XCTAssertNil(mainland.pick(probs(["ㄒㄩㄝ4": 0.9, "ㄒㄧㄠ1": 0.1])))
    }

    func testInventoryWithoutTheDefaultKeepsTheDictReading() throws {
        // Traditional 吨 is another character (tún); simplified 吨 is "ton", dūn.
        let (bins, _) = try Self.bins("吨", ["ㄊㄨㄣ2", "ㄊㄨㄣ3"], readings: ["dūn", "tún", "tǔn"])
        XCTAssertNil(bins)
    }

    func testNeutralToneFormIsAReading() throws {
        let (bins, probs) = try Self.bins("夫", ["ㄈㄨ1", "ㄈㄨ2", "ㄈㄨ5"], readings: ["fū", "fú"])
        XCTAssertEqual(try XCTUnwrap(bins).pick(probs(["ㄈㄨ5": 0.8, "ㄈㄨ1": 0.2])), "ㄈㄨ5")
    }

    func testTaiwanDaiCountsTowardDei() throws {
        // 得 "must": Taiwan dǎi is mainland děi, not the default dé.
        let (bins, probs) = try Self.bins("得", ["ㄉㄜ2", "ㄉㄜ5", "ㄉㄟ3", "ㄉㄞ3"], readings: ["dé", "de", "děi"])
        let mainland = try XCTUnwrap(bins)
        XCTAssertEqual(mainland.pick(probs(["ㄉㄞ3": 0.5, "ㄉㄜ2": 0.3, "ㄉㄟ3": 0.2])), "ㄉㄟ3")
    }

    func testNothingToChooseWhenEveryLabelIsTheDefault() throws {
        // 期 qí is the Taiwan tone of the default qī.
        let (bins, _) = try Self.bins("期", ["ㄑㄧ1", "ㄑㄧ2"], readings: ["qī", "jī"])
        XCTAssertNil(bins)
    }

    func testNonFiniteScoresKeepTheDictReading() throws {
        let (bins, probs) = try Self.bins("和", ["ㄏㄜ2", "ㄏㄨㄛ4"], readings: ["hé", "huò"])
        XCTAssertNil(try XCTUnwrap(bins).pick(probs(["ㄏㄜ2": .nan, "ㄏㄨㄛ4": .nan])))
    }

    func testDictReadingsAreKeyedByTargetPosition() throws {
        let dict = MandarinPinyinDict(
            phrases: [:],
            singles: [
                Character("和").unicodeScalars.first!.value: ["hé", "hè", "hú"],
                Character("我").unicodeScalars.first!.value: ["wǒ"],
            ])
        let readings = MandarinG2P(dict: dict).dictReadings(at: [1], in: Array("我和你"))
        XCTAssertEqual(readings[1], Self.syllables("hé", "hè", "hú"))
        XCTAssertNil(readings[0])
    }

    /// Runs the shipped g2pW bundle when the Mandarin assets are cached.
    func testShippedG2pwReadsConjunctionHeAsMainlandHe() async throws {
        let repoDirectory = try KokoroAneResourceDownloader.repositoryDirectory(variant: .mandarin)
        let cached = FileManager.default.fileExists(
            atPath: repoDirectory.appendingPathComponent("g2pw/g2pw.mlmodelc").path)
        try XCTSkipUnless(
            cached || ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_TTS_E2E"] == "1",
            "Mandarin g2pW bundle not cached; set FLUIDAUDIO_RUN_TTS_E2E=1")
        let manager = KokoroAneManager(variant: .mandarin)
        for text in ["我和你是好朋友。", "老师和学生都来了。", "漫画和现实的世界是不可能相通的。"] {
            let phonemes = try await manager.phonemes(for: text)
            XCTAssertTrue(phonemes.contains("ㄏㄜ2"), "\(text) → \(phonemes)")
            XCTAssertFalse(phonemes.contains("ㄏㄢ4") || phonemes.contains("ㄏ我4"), "\(text) → \(phonemes)")
        }
        // 削 xiāo, 吨 dūn, 厂 chǎng, 咋 zǎ (the last three are other characters in the inventory).
        for (text, reading) in [
            ("妈妈在削苹果。", "ㄒ要1"), ("这辆卡车能装五吨货。", "ㄉ文1"), ("他在厂里上班。", "ㄔㄤ"), ("你咋不早说？", "ㄗㄚ"),
        ] {
            let phonemes = try await manager.phonemes(for: text)
            XCTAssertTrue(phonemes.contains(reading), "\(text) → \(phonemes)")
        }
    }

    func testMixedTextWithMultiplePolyphones() throws {
        // Two polyphonic chars in one sentence → two targets. The
        // segmenter records charPos in the *normalized* string so
        // upstream g2pW can pick by context.
        let g2p = MandarinG2P(dict: Self.mixedDict())
        let chars: [Character] = Array("行行去")
        let result = g2p.segment(chars: chars)
        XCTAssertEqual(result.polyphoneTargets.count, 2)
        XCTAssertEqual(
            result.polyphoneTargets.map { $0.charPos }, [0, 1])
    }
}
