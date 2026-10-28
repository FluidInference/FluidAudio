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

    func testCompiledBundleMapsSourcePackageFiles() throws {
        // #1043: ensure() skips a source-package file once the pipeline has
        // compiled its sibling .mlmodelc. Pure path mapping; the gated
        // real-model test covers the filesystem behavior end to end.
        let root = URL(fileURLWithPath: "/models/en", isDirectory: true)
        func bundle(_ path: String) -> String? {
            KokoroAneV3Assets.compiledBundleURL(forSourceFile: path, root: root)?.path
        }
        XCTAssertEqual(
            bundle("fast/KokoroAlbert_32.mlpackage/Manifest.json"), "/models/en/fast/KokoroAlbert_32.mlmodelc")
        XCTAssertEqual(
            bundle("fast/KokoroAlbert_32.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            "/models/en/fast/KokoroAlbert_32.mlmodelc")
        XCTAssertEqual(
            bundle("decoder/KokoroDecoderPre_120.mlpackage/Manifest.json"),
            "/models/en/decoder/KokoroDecoderPre_120.mlmodelc")
        XCTAssertEqual(
            bundle("base/KokoroAlbert.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            "/models/en/base/KokoroAlbert.mlmodelc")
        // Loose assets and the published compiled bundles have no source
        // package, so they are always verified.
        for path in [
            "vocab.json", "af_heart.bin", "fast/native-source.json",
            "base/KokoroProsody_v2.mlmodelc/model.mil", "base/KokoroTail_v2.mlmodelc/weights/weight.bin",
        ] {
            XCTAssertNil(bundle(path), path)
            XCTAssertFalse(KokoroAneV3Assets.hasCompiledBundle(forSourceFile: path, root: root), path)
        }
        // No bundle on disk: the source is still checked.
        XCTAssertFalse(
            KokoroAneV3Assets.isCompiledBundle(at: root.appendingPathComponent("fast/KokoroAlbert_32.mlmodelc")))
        XCTAssertFalse(
            KokoroAneV3Assets.hasCompiledBundle(
                forSourceFile: "fast/KokoroAlbert_32.mlpackage/Manifest.json", root: root))
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

    /// Locally compiled stages under `fast/`, `decoder/` and `base/` that have
    /// (or had) a source package; the published compiled bundles are excluded.
    private func compiledStagesWithSource(under root: URL) throws -> [URL] {
        var bundles: [URL] = []
        for folder in ["fast", "decoder", "base"] {
            let directory = root.appendingPathComponent(folder)
            guard FileManager.default.fileExists(atPath: directory.path) else { continue }
            bundles += try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
                .filter {
                    $0.pathExtension == "mlmodelc" && KokoroAneV3Assets.isCompiledBundle(at: $0)
                        && !["KokoroProsody_v2", "KokoroTail_v2"].contains($0.deletingPathExtension().lastPathComponent)
                }
        }
        return bundles
    }

    private func sourcePackage(for compiled: URL) -> URL {
        compiled.deletingPathExtension().appendingPathExtension("mlpackage")
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
        // #1043: with every stage compiled, the .mlpackage sources are dead
        // weight. Deleting them must not trigger a re-download on the next
        // load, and synthesis must keep running from the compiled bundles.
        // Idempotent: on a cache that was already stripped, this only checks
        // that nothing comes back.
        await english.cleanup()
        let compiledStages = try compiledStagesWithSource(under: root)
        XCTAssertFalse(compiledStages.isEmpty)
        for bundle in compiledStages {
            try? FileManager.default.removeItem(at: sourcePackage(for: bundle))
        }
        let afterRemoval = try await english.synthesizeFromPhonemesDetailed(hello)
        try check(afterRemoval, fast: true)
        for bundle in compiledStages {
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: sourcePackage(for: bundle).path),
                "\(bundle.lastPathComponent) source was re-downloaded although the compiled bundle exists")
        }
        // Recovery with the source gone: a compiled bundle that no longer
        // loads is dropped, the load fails once, and the next load re-fetches
        // and recompiles that stage's source.
        await english.cleanup()
        let damaged = root.appendingPathComponent("fast/KokoroAlbert_32.mlmodelc")
        XCTAssertTrue(KokoroAneV3Assets.isCompiledBundle(at: damaged))
        try Data("x".utf8).write(to: damaged.appendingPathComponent("coremldata.bin"), options: .atomic)
        do {
            _ = try await english.synthesizeFromPhonemesDetailed(hello)
            XCTFail("Expected the damaged compiled bundle to fail to load")
        } catch {
            XCTAssertFalse(FileManager.default.fileExists(atPath: damaged.path), "damaged bundle must be dropped")
        }
        let recovered = try await english.synthesizeFromPhonemesDetailed(hello)
        try check(recovered, fast: true)
        XCTAssertTrue(KokoroAneV3Assets.isCompiledBundle(at: damaged))
        XCTAssertTrue(FileManager.default.fileExists(atPath: sourcePackage(for: damaged).path))
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
