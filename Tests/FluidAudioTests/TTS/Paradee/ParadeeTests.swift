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

    func testVariantMapsToRepo() {
        XCTAssertEqual(ParadeeVariant.int8.repo, .paradeeInt8)
        XCTAssertEqual(ParadeeVariant.fp32.repo, .paradeeFp32)
    }

    func testRequiredModels() {
        let required: Set<String> = ["ParadeeText.mlmodelc", "ParadeeAcoustic.mlmodelc", "vocab.json"]
        XCTAssertEqual(ModelNames.Paradee.requiredModels, required)
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .paradeeInt8, variant: nil), required)
        XCTAssertEqual(ModelNames.getRequiredModelNames(for: .paradeeFp32, variant: nil), required)
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
