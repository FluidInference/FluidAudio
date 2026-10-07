import Foundation
import XCTest

@testable import FluidAudio

final class KokoroAneV3Tests: XCTestCase {
    func testCacheSeparatesLanguageAndPinnedRelease() throws {
        let directory = URL(fileURLWithPath: "/tmp/kokoro-v3-test")
        let english = try KokoroAneV3Assets.cacheDirectory(variant: .english, directory: directory)
        let japanese = try KokoroAneV3Assets.cacheDirectory(variant: .japanese, directory: directory)
        let mandarin = try KokoroAneV3Assets.cacheDirectory(variant: .mandarin, directory: directory)
        XCTAssertEqual(Set([english, japanese, mandarin]).count, 3)
        XCTAssertTrue(english.path.contains(KokoroAneV3Assets.revision))
        XCTAssertEqual(KokoroAneV3Assets.prefix(.english), "ANE-v3")
        XCTAssertEqual(KokoroAneV3Assets.prefix(.japanese), "ANE-v3/ja")
        XCTAssertEqual(KokoroAneV3Assets.prefix(.mandarin), "ANE-v3/zh")
        XCTAssertEqual(KokoroAneModelStore().version, .legacy)
        XCTAssertEqual(KokoroAneModelStore(version: .v3).version, .v3)
    }

    func testRejectsUnsafeManifestPathsAndWrongLanguageVoices() throws {
        for path in ["../weights", "/tmp/file", "fast/../../file", "fast//file", "a%2fb", "a?query", "a\\b"] {
            XCTAssertThrowsError(try KokoroAneV3Assets.validatePath(path))
        }
        XCTAssertNoThrow(try KokoroAneV3Assets.validatePath("fast/KokoroAlbert_32.mlpackage/Manifest.json"))
        XCTAssertNoThrow(try KokoroAneV3Assets.validateVoice("af_bella", variant: .english))
        XCTAssertNoThrow(try KokoroAneV3Assets.validateVoice("jm_kumo", variant: .japanese))
        XCTAssertNoThrow(try KokoroAneV3Assets.validateVoice("zm_010", variant: .mandarin))
        XCTAssertThrowsError(try KokoroAneV3Assets.validateVoice("zf_001", variant: .japanese))
        XCTAssertThrowsError(try KokoroAneV3Assets.validateVoice("af_heart/../../", variant: .english))
        XCTAssertThrowsError(try KokoroAneV3Assets.decodeEnglishVoice(Data("{}".utf8)))
    }

    func testRejectsUnsupportedRoutingBeforeDownloading() async throws {
        let store = KokoroAneModelStore(computeUnits: .cpuOnly, version: .v3)
        do {
            try await store.loadIfNeeded()
            XCTFail("Custom compute policy was silently ignored")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("routing"))
        }
        let loaded = await store.isLoaded
        XCTAssertFalse(loaded)
    }

    func testIntegrityRejectsChangedOrTruncatedContent() {
        // Standard SHA-256 test vector, independent of the implementation.
        let file = KokoroAneV3Assets.File(
            path: "manifest.json", bytes: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        XCTAssertTrue(KokoroAneV3Assets.matches(Data("abc".utf8), file: file))
        XCTAssertFalse(KokoroAneV3Assets.matches(Data("abd".utf8), file: file))
        XCTAssertFalse(KokoroAneV3Assets.matches(Data("ab".utf8), file: file))
    }

    func testUnsupportedLanguagesFailBeforeDownloading() async throws {
        for variant in [KokoroAneVariant.spanish, .french] {
            let store = KokoroAneModelStore(variant: variant, version: .v3)
            do {
                try await store.loadIfNeeded()
                XCTFail("Unsupported variant unexpectedly loaded")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("English, Japanese and Mandarin"))
            }
            let loaded = await store.isLoaded
            XCTAssertFalse(loaded)
        }
    }

    func testChunkTimingAccumulatesAllV3Stages() {
        var total = KokoroAneStageTimings()
        var first = KokoroAneStageTimings()
        first.albert = 1
        first.nativeSource = 2
        first.decoder = 3
        first.generator = 4
        var second = KokoroAneStageTimings()
        second.vocoder = 5
        second.tail = 6
        total.add(first)
        total.add(second)
        XCTAssertEqual(total.totalMs, 21)
        XCTAssertEqual(total.generator, 4)
        XCTAssertEqual(total.vocoder, 5)
    }

    func testRealSDKInputsAndFallbacks() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKORO_V3"] == "1",
            "Explicitly enable the bounded real-model integration check")
        let directory = ProcessInfo.processInfo.environment["KOKORO_V3_TEST_CACHE"].map { URL(fileURLWithPath: $0) }
        let english = KokoroAneManager(directory: directory, version: .v3)
        let hello = "həlˈoʊ wˈɜːld"
        let fox = "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."
        var parts: [KokoroAneSynthesisResult] = []
        // Uses the SDK's downloader, source-package compilation, vocab and voice indexing.
        for (phonemes, speed, fast) in [
            (hello, Float(1), true), (fox, 1, true), (fox, 0.7, true), (fox + " " + fox, 1, false),
        ] {
            let result = try await english.synthesizeFromPhonemesDetailed(phonemes, speed: speed)
            try check(result, fast: fast)
            parts.append(result)
        }
        let joined = KokoroAneSynthesisResult.concatenating(parts)
        XCTAssertFalse(joined.usedFastVocoder)
        XCTAssertEqual(joined.predictedDurations, parts.flatMap(\.predictedDurations))
        XCTAssertEqual(joined.timings.totalMs, parts.reduce(0) { $0 + $1.timings.totalMs }, accuracy: 0.0001)
        let alternate = try await english.synthesizeFromPhonemesDetailed(hello, voice: "af_bella")
        try check(alternate, fast: true)
        // Repair a corrupted asset without replacing valid locally compiled models.
        let root = try KokoroAneV3Assets.cacheDirectory(variant: .english, directory: directory)
        let vocabURL = root.appendingPathComponent("vocab.json")
        let vocabData = try Data(contentsOf: vocabURL)
        defer { try? vocabData.write(to: vocabURL, options: .atomic) }
        try Data("{}".utf8).write(to: vocabURL, options: .atomic)
        let voiceURL = try await KokoroAneV3Assets.ensureVoice("af_bella", variant: .english, root: root)
        let voiceData = try Data(contentsOf: voiceURL)
        defer { try? voiceData.write(to: voiceURL, options: .atomic) }
        try voiceData.prefix(128).write(to: voiceURL, options: .atomic)
        let repaired = try await KokoroAneV3Assets.ensureVoice("af_bella", variant: .english, root: root)
        XCTAssertEqual(try Data(contentsOf: repaired), voiceData)
        await english.cleanup()
        let reloaded = try await english.synthesizeFromPhonemesDetailed(hello)
        try check(reloaded, fast: true)
        XCTAssertEqual(try Data(contentsOf: vocabURL), vocabData)
        try await english.initialize()
        try check(try await english.synthesizeDetailed(text: "Hello world."), fast: true)
        await english.cleanup()

        let japanese = KokoroAneManager(variant: .japanese, directory: directory, version: .v3)
        let ja = "koɲɲiʨiβa, sekai."
        try check(try await japanese.synthesizeFromPhonemesDetailed(ja), fast: true)
        try check(try await japanese.synthesizeFromPhonemesDetailed(ja, voice: "jm_kumo"), fast: true)
        let japaneseText = try await japanese.synthesizeDetailed(text: "こんにちは、世界。")
        try check(japaneseText, fast: true)
        XCTAssertFalse(japaneseText.phonemes.isEmpty)
        XCTAssertEqual(japaneseText.normalizedText, "こんにちは、世界。")
        await japanese.cleanup()

        let mandarin = KokoroAneManager(variant: .mandarin, directory: directory, version: .v3)
        let zh = "ㄋㄧ2ㄏㄠ3/ㄕ十4ㄐㄝ4."
        try check(try await mandarin.synthesizeFromPhonemesDetailed(zh), fast: true)
        try check(try await mandarin.synthesizeFromPhonemesDetailed(zh, voice: "zm_010"), fast: true)
        // Exercise the published Mandarin Noise_v2 compiled fallback too.
        try check(
            try await mandarin.synthesizeFromPhonemesDetailed(
                Array(repeating: zh, count: 4).joined(separator: " "), speed: 0.7),
            fast: false)
        try check(try await mandarin.synthesizeDetailed(text: "你好世界。"), fast: true)
        await mandarin.cleanup()
    }

    private func check(_ result: KokoroAneSynthesisResult, fast: Bool) throws {
        XCTAssertEqual(result.inputIds.count, result.encoderTokens)
        XCTAssertEqual(result.predictedDurations.count, result.inputIds.count)
        XCTAssertEqual(result.predictedDurations.reduce(0) { $0 + Int($1) }, result.acousticFrames)
        XCTAssertFalse(result.phonemes.isEmpty)
        XCTAssertEqual(result.sampleRate, 24_000)
        XCTAssertEqual(result.samples.count, result.acousticFrames * 600)
        XCTAssertTrue(result.samples.allSatisfy(\.isFinite))
        XCTAssertTrue(result.samples.contains { abs($0) > 0.001 })
        XCTAssertEqual(result.usedFastVocoder, fast)
        XCTAssertGreaterThan(result.timings.albert, 0)
        if fast {
            XCTAssertGreaterThan(result.timings.nativeSource, 0)
            XCTAssertGreaterThan(result.timings.decoder, 0)
            XCTAssertGreaterThan(result.timings.generator, 0)
            XCTAssertEqual(result.timings.vocoder, 0)
        } else {
            XCTAssertEqual(result.timings.generator, 0)
            XCTAssertGreaterThan(result.timings.vocoder, 0)
            XCTAssertGreaterThan(result.timings.tail, 0)
        }
    }
}
