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
