import Foundation
import XCTest

@testable import FluidAudio

/// Runs the real frontend on the cached Japanese assets (trimmed unidic-lite
/// + Cutlet word list). Skips when they are not cached; set
/// `FLUIDAUDIO_RUN_TTS_E2E=1` to download them. Expected strings are Misaki's
/// `ja.JAG2P()` (Cutlet) outputs, which Kokoro's Japanese voices were trained on.
final class JapaneseG2PIntegrationTests: XCTestCase {
    private func loadG2P() async throws -> JapaneseG2P {
        let repoDirectory = try KokoroAneResourceDownloader.repositoryDirectory(variant: .japanese)
        let g2pDirectory = repoDirectory.appendingPathComponent(KokoroAneConstants.g2pSubdir)
        let cached = KokoroAneConstants.japaneseG2PFiles.allSatisfy {
            FileManager.default.fileExists(atPath: g2pDirectory.appendingPathComponent($0).path)
        }
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_TTS_E2E"] == "1"
        try XCTSkipUnless(cached || allowDownload, "Japanese G2P assets not cached; set FLUIDAUDIO_RUN_TTS_E2E=1")
        let directory = try await KokoroAneResourceDownloader.ensureJapaneseG2P(repoDirectory: repoDirectory)
        return try JapaneseG2P(directory: directory)
    }

    func testTokensMatchFugashi() async throws {
        let g2p = try await loadG2P()
        let tokens = await g2p.tokens("私は日本語を勉強しています。")
        XCTAssertEqual(tokens.map(\.surface), ["私", "は", "日本", "語", "を", "勉強", "し", "て", "い", "ます", "。"])
        XCTAssertEqual(tokens.map(\.pron), ["ワタクシ", "ワ", "ニッポン", "ゴ", "オ", "ベンキョー", "シ", "テ", "イ", "マス", ""])
        XCTAssertEqual(tokens.map(\.charType), [2, 6, 2, 2, 6, 2, 6, 6, 6, 6, 3])
        XCTAssertFalse(tokens.contains { $0.isUnknown })
        // Unknown words: category rules from char.bin / unk.dic.
        let unknown = await g2p.tokens("AIを使います。")
        XCTAssertEqual(unknown.map(\.surface), ["AI", "を", "使い", "ます", "。"])
        XCTAssertTrue(unknown[0].isUnknown)
        XCTAssertEqual(unknown[0].charType, 5, "ALPHA")
    }

    func testPhonemesMatchMisakiCutlet() async throws {
        let g2p = try await loadG2P()
        let cases: [(String, String)] = [
            ("今日は良い天気です。", "kʲoː βa joi teŋkʲi desɨ."),
            ("学校へ行った。", "ɡaʔkoː e iʔta."),
            ("東京駅で待っています。", "toːkʲoː ekʲi de maʔte imasɨ."),
            ("私は日本語を勉強しています。", "βatakɯɕi βa ɲiʔpoŋɡo o beŋkʲoː ɕite imasɨ."),
            ("コンピューターを買いました。", "kompʲɨːtaː o kai maɕita."),
            ("一万二千三百円です。", "iʨimaɴ ɲiseɴ sambʲakɯ eɴ desɨ."),
            ("彼女はパーティーに来ませんでした。", "kanoʥo βa paːtʲiː ɲi kʲi maseɴ deɕi ta."),
            ("ちょっと待ってください！", "ʨoʔto maʔte kɯdasai!"),
            ("先生、質問があります。", "seɴseː, ɕiʦɨmoɴ ɡa aɾʲimasɨ."),
            ("インターネットが遅いです。", "intaːneʔto ɡa osoi desɨ."),
            ("新幹線は速いですね。", "ɕiŋkaɴseɴ βa hajai desɨ ne."),
            ("何時に起きますか？", "naɲʥi ɲi okʲimasɨ ka?"),
            ("ありがとうございました。", "aɾʲiɡatoː ɡoʣaimaɕita."),
            ("AIを使います。", "AI o ʦɨkaimasɨ."),
            ("今日中に返します。", "koɲɲiʨi ʨɨː ɲi kaeɕimasɨ."),
            ("一日中勉強しました。", "iʨiɲiʨi ʨɨː beŋkʲoː ɕi maɕita."),
        ]
        for (text, expected) in cases {
            let actual = try await g2p.phonemize(text)
            XCTAssertEqual(actual, expected, text)
        }
    }

    /// Digits reach the frontend as kanji numerals from NeMo normalization,
    /// which MeCab reads correctly (`2人` → ふたり where Cutlet's digit path
    /// says ɲi çito); the phonemes differ from Misaki only in grouping.
    func testDigitsReadThroughNormalizedNumerals() async throws {
        let g2p = try await loadG2P()
        let people = try await g2p.phonemize("二人とも元気です。")
        XCTAssertEqual(people, "ɸɯtaɾʲi tomo ɡeŋkʲi desɨ.")
        let time = try await g2p.phonemize("午前十時から")
        XCTAssertEqual(time, "ɡoʣeɴ ʥɨːʥi kaɾa")
        // Inputs the pass-through predicate must not swallow: a lone digit and
        // half-width kana are read, not returned verbatim.
        let one = try await g2p.phonemize("1")
        XCTAssertEqual(one, "iʨi")
        let halfWidth = try await g2p.phonemize("ｶﾞｷﾞ")
        XCTAssertEqual(halfWidth, "ɡa ɡʲi")
        let fullWidthRange = try await g2p.phonemize("パン３～５個")
        XCTAssertEqual(fullWidthRange, "paɴ saɴ kaɾa ɡo ko")
    }
}
