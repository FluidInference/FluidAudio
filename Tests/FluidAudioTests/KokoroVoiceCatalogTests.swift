import XCTest

@testable import FluidAudio

@available(macOS 13.0, *)
final class KokoroVoiceCatalogTests: XCTestCase {

    private let expectedVoiceIds: Set<String> = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
        "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni",
        "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    func testAllVoicesPresent() {
        let catalogIds = Set(KokoroVoiceCatalog.allVoices.map { $0.id })
        XCTAssertEqual(
            catalogIds,
            expectedVoiceIds,
            "Voice catalog should expose all known Kokoro voices"
        )
        XCTAssertEqual(KokoroVoiceCatalog.allVoices.count, expectedVoiceIds.count)
    }

    func testCanonicalVoiceIdResolvesAliases() {
        XCTAssertEqual(KokoroVoiceCatalog.canonicalVoiceId(for: "afHeart"), "af_heart")
        XCTAssertEqual(KokoroVoiceCatalog.canonicalVoiceId(for: "AF_HEART"), "af_heart")
        XCTAssertEqual(KokoroVoiceCatalog.canonicalVoiceId(for: "af-heart"), "af_heart")
        XCTAssertEqual(KokoroVoiceCatalog.canonicalVoiceId(for: "hfOmega"), "hm_omega")
        XCTAssertEqual(KokoroVoiceCatalog.canonicalVoiceId(for: "HF_PSI"), "hm_psi")
    }

    func testVoiceLookupReturnsMetadata() {
        let option = KokoroVoiceCatalog.voice(for: "amPuck")
        XCTAssertEqual(option?.id, "am_puck")
        XCTAssertEqual(option?.languageCode, "en-US")
        XCTAssertEqual(option?.gender, .male)
    }

    func testLanguageGroupingIncludesAllVoices() {
        var seenIds: Set<String> = []
        for language in KokoroVoiceCatalog.supportedLanguageCodes {
            let voices = KokoroVoiceCatalog.voices(forLanguage: language)
            for voice in voices {
                XCTAssertEqual(voice.languageCode, language)
                seenIds.insert(voice.id)
            }
        }
        XCTAssertEqual(seenIds, expectedVoiceIds)
    }
}
