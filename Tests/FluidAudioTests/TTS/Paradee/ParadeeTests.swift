import CoreML
import XCTest

@testable import FluidAudio

/// Paradee repo wiring and the host-side steps between the two CoreML graphs.
final class ParadeeTests: XCTestCase {

    // MARK: - Repo wiring

    func testRepoPathsAndSubdirectories() {
        XCTAssertEqual(Repo.paradeeInt8.remotePath, "FluidInference/paradee-8m-coreml")
        XCTAssertEqual(Repo.paradeeFp32.remotePath, "FluidInference/paradee-8m-coreml")
        XCTAssertEqual(Repo.paradeeInt8.subPath, "int8")
        XCTAssertEqual(Repo.paradeeFp32.subPath, "fp32")
        XCTAssertEqual(Repo.paradeeInt8.folderName, "paradee-8m-coreml/int8")
        XCTAssertEqual(Repo.paradeeFp32.folderName, "paradee-8m-coreml/fp32")
    }

    func testV12RepoPathsAndSubdirectories() {
        for (repo, variant) in [(Repo.paradeeV12Int8, "int8"), (.paradeeV12Fp32, "fp32")] {
            XCTAssertEqual(repo.remotePath, "FluidInference/paradee-8m-coreml")
            XCTAssertEqual(repo.subPath, "v1.2/\(variant)")
            XCTAssertEqual(repo.folderName, "paradee-8m-coreml/v1.2/\(variant)")
        }
    }

    func testVariantMapsToRepo() {
        XCTAssertEqual(ParadeeVariant.int8.repo, .paradeeInt8)
        XCTAssertEqual(ParadeeVariant.fp32.repo, .paradeeFp32)
        XCTAssertEqual(ParadeeVariant.int8.repo(for: .v10), .paradeeInt8)
        XCTAssertEqual(ParadeeVariant.fp32.repo(for: .v10), .paradeeFp32)
        XCTAssertEqual(ParadeeVariant.int8.repo(for: .v12), .paradeeV12Int8)
        XCTAssertEqual(ParadeeVariant.fp32.repo(for: .v12), .paradeeV12Fp32)
    }

    func testRequiredModels() {
        let required: Set<String> = ["ParadeeText.mlmodelc", "ParadeeAcoustic.mlmodelc", "vocab.json"]
        XCTAssertEqual(ModelNames.Paradee.requiredModels, required)
        for repo in [Repo.paradeeInt8, .paradeeFp32, .paradeeV12Int8, .paradeeV12Fp32] {
            XCTAssertEqual(ModelNames.getRequiredModelNames(for: repo, variant: nil), required)
        }
    }

    // MARK: - Voices

    func testVoicesPerVersion() {
        XCTAssertEqual(ParadeeVersion.v10.voices, [.afHeart])
        XCTAssertFalse(ParadeeVersion.v10.hasVoiceInput)
        // The v1.2 voice tables are ordered like config.json `voices`.
        XCTAssertEqual(ParadeeVersion.v12.voices.map(\.rawValue), ["af_heart", "bf_isabella", "am_puck", "bm_fable"])
        XCTAssertEqual(ParadeeVersion.v12.voices.map(\.modelIndex), [0, 1, 2, 3])
        XCTAssertTrue(ParadeeVersion.v12.hasVoiceInput)
        XCTAssertEqual(ParadeeVoice.allCases.filter(\.isBritish), [.bfIsabella, .bmFable])
    }

    func testV10RejectsOtherVoices() async {
        let manager = ParadeeManager(version: .v10)
        do {
            _ = try await manager.synthesize(phonemes: "həlˈO", voice: .bmFable)
            XCTFail("expected unsupportedVoice")
        } catch ParadeeError.unsupportedVoice(let voice, let version) {
            XCTAssertEqual(voice, "bm_fable")
            XCTAssertEqual(version, "1.0")
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    // MARK: - British frontend

    func testBritishLexiconFileName() {
        XCTAssertEqual(LexiconAssetCache.englishFileName(british: false), "us_lexicon_cache.json")
        XCTAssertEqual(LexiconAssetCache.englishFileName(british: true), "gb_lexicon_cache.json")
    }

    func testBritishFallbackRewritesAmericanPhonemes() {
        // Expected values from the Python rules the mapping was measured with.
        let cases: [(String, String)] = [
            ("kˈɑɹ", "kˈɑː"), ("ɡˈO", "ɡˈQ"), ("bˈæθ", "bˈaθ"), ("hˈɑt", "hˈɒt"), ("fˈIɹ", "fˈIə"),
            ("vˈɛɹi", "vˈɛɹi"), ("sˈɑɹi", "sˈɒɹi"), ("wˈɔTəɹ", "wˈɔtə"), ("bˈɜɹd", "bˈɜːd"), ("ˈWɹ", "ˈWə"),
            ("kwɪksˈɑTɪk", "kwɪksˈɒtɪk"), ("ɡlˈɔɹbɪŋtᵊnz", "ɡlˈɔːbɪŋtᵊnz"), ("ˌɛɹˈOd", "ˌɛɹˈQd"),
            ("bˈɑɹˌn", "bˈɑːˌn"), ("kˈɪʔnz", "kˈɪtnz"), ("ˈɪɹˌɪŋ", "ˈɪɹˌɪŋ"),
        ]
        for (american, british) in cases {
            XCTAssertEqual(MisakiBritishFallback.fromAmerican(american), british, american)
        }
    }

    func testGpuComputeUnitsRejected() {
        XCTAssertTrue(ParadeeModelStore.isSupported(.cpuOnly))
        XCTAssertTrue(ParadeeModelStore.isSupported(.cpuAndNeuralEngine))
        XCTAssertFalse(ParadeeModelStore.isSupported(.all))
        XCTAssertFalse(ParadeeModelStore.isSupported(.cpuAndGPU))
    }

    // MARK: - Durations

    func testFrameCountsRoundHalfToEvenLikeTorch() {
        // torch.round: 0.5 → 0, 1.5 → 2, 2.5 → 2; then max(1, ·).
        XCTAssertEqual(
            ParadeeSynthesizer.frameCounts(durations: [0.5, 1.5, 2.5, 3.49, 0.1], speed: 1),
            [1, 2, 2, 3, 1])
    }

    func testFrameCountsScaleWithSpeed() {
        XCTAssertEqual(ParadeeSynthesizer.frameCounts(durations: [4, 6], speed: 2), [2, 3])
        XCTAssertEqual(ParadeeSynthesizer.frameCounts(durations: [4, 6], speed: 0.5), [8, 12])
    }

    // MARK: - Expansion

    func testExpandRepeatsColumns() {
        // [2 channels, 3 tokens], counts [2, 1, 3] → [2, 6].
        let source: [Float] = [1, 2, 3, 10, 20, 30]
        XCTAssertEqual(
            ParadeeSynthesizer.expand(source, channels: 2, counts: [2, 1, 3]),
            [1, 1, 2, 3, 3, 3, 10, 10, 20, 30, 30, 30])
    }

    // MARK: - Text frontend helpers

    func testSentenceSplitMatchesUpstreamRegex() {
        XCTAssertEqual(
            ParadeeManager.sentences(in: "Hello there. How are you?  Fine!\nNew line…  End"),
            ["Hello there.", "How are you?", "Fine!", "New line…", "End"])
        // No split without whitespace after the punctuation ("3.5", "a.b").
        XCTAssertEqual(ParadeeManager.sentences(in: "Pi is 3.14 today."), ["Pi is 3.14 today."])
    }

    func testMisakiOutputFormRewritesFlapAndGlottalStop() {
        XCTAssertEqual(ParadeeManager.misakiOutputForm("kˈɪʔnz sˈæɾᵊlˌIt"), "kˈɪtnz sˈæTᵊlˌIt")
    }

    func testChunksRespectPhonemeLimit() {
        let word = "həlˈO "
        let long = String(repeating: word, count: 200)
        let chunks = ParadeeManager.chunks(long)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.unicodeScalars.count, ParadeeConstants.maxPhonemeLength)
        }
    }
}
