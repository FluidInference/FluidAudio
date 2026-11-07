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

    func testMainlandLabelsFoldTaiwanReadingIntoDefault() throws {
        // g2pW lists 和 as hàn (Taiwan, "and") besides the mainland readings.
        let catalog = try MandarinPolyphoneCatalog(text: "和\tㄏㄜ2\n和\tㄏㄢ4\n和\tㄏㄨㄛ4\n和\tㄏㄜ4\n")
        let mainland = try XCTUnwrap(
            MandarinG2pwModel.mainlandLabels(
                for: "和", readings: Self.syllables("hé", "hè", "huó", "huò"), catalog: catalog))
        XCTAssertEqual(Self.labels(mainland.labels, catalog), ["ㄏㄜ2", "ㄏㄜ4", "ㄏㄨㄛ4"])
        let han = try XCTUnwrap(catalog.labels.firstIndex(of: "ㄏㄢ4"))
        XCTAssertEqual(mainland.folds[han].flatMap(catalog.bopomofo(forLabel:)), "ㄏㄜ2")
    }

    func testMainlandLabelsAllowNeutralToneForms() throws {
        let catalog = try MandarinPolyphoneCatalog(text: "夫\tㄈㄨ1\n夫\tㄈㄨ2\n夫\tㄈㄨ5\n")
        let mainland = try XCTUnwrap(
            MandarinG2pwModel.mainlandLabels(for: "夫", readings: Self.syllables("fū", "fú"), catalog: catalog))
        XCTAssertEqual(Self.labels(mainland.labels, catalog), ["ㄈㄨ1", "ㄈㄨ2", "ㄈㄨ5"])
        XCTAssertTrue(mainland.folds.isEmpty)
    }

    func testMainlandLabelsMapTaiwanDaiToDei() throws {
        // 得 "must": Taiwan dǎi is mainland děi, not the default dé.
        let catalog = try MandarinPolyphoneCatalog(text: "得\tㄉㄜ2\n得\tㄉㄜ5\n得\tㄉㄟ3\n得\tㄉㄞ3\n")
        let mainland = try XCTUnwrap(
            MandarinG2pwModel.mainlandLabels(for: "得", readings: Self.syllables("dé", "de", "děi"), catalog: catalog))
        let dai = try XCTUnwrap(catalog.labels.firstIndex(of: "ㄉㄞ3"))
        XCTAssertEqual(mainland.folds[dai].flatMap(catalog.bopomofo(forLabel:)), "ㄉㄟ3")
    }

    func testMainlandLabelsNeedTwoSharedReadings() throws {
        // 期 qí is Taiwan-only; one shared reading leaves g2pW nothing to pick.
        let catalog = try MandarinPolyphoneCatalog(text: "期\tㄑㄧ1\n期\tㄑㄧ2\n")
        XCTAssertNil(MandarinG2pwModel.mainlandLabels(for: "期", readings: Self.syllables("qī", "jī"), catalog: catalog))
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
